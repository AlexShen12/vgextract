"""Pinned local checkpoints and a batch-one, offline, node-local vLLM worker."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import threading
import urllib.error
import urllib.request

from .core import atomic_json, digest, read_json, StopRequested

LOCK = Path(__file__).with_name("models.local.json")


class ModelUnavailable(RuntimeError):
    """A failed GPU worker stops the stage instead of retrying every field."""


def checkpoint(cache, role):
    specification = read_json(LOCK)["models"][role]
    record_path = Path(cache) / "models" / role / "manifest.json"
    if not record_path.exists():
        raise ValueError(
            f"Missing pinned {role} checkpoint; run cache-models before the offline job"
        )
    record = read_json(record_path)
    if record.get("specification") != specification:
        raise ValueError(f"Stale {role} checkpoint pin; recache models")
    root = record_path.parent / "snapshot"
    if not record.get("files"):
        raise ValueError("Empty checkpoint manifest")
    for entry in record["files"]:
        path = root / entry["path"]
        if (
            not path.resolve().is_relative_to(root.resolve())
            or not path.is_file()
            or path.stat().st_size != entry["size"]
        ):
            raise ValueError(f"Incomplete checkpoint file: {path}")
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                hasher.update(block)
        if hasher.hexdigest() != entry.get("sha256"):
            raise ValueError(f"Checkpoint checksum mismatch: {path}")
    return root, {"specification": specification, "files_hash": digest(record["files"])}


def cache_models(cache, roles=("intake",)):
    from huggingface_hub import snapshot_download

    result = {}
    for role in roles:
        specification = read_json(LOCK)["models"][role]
        revision = specification["revision"]
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Checkpoint revisions must be immutable commit hashes")
        directory = Path(cache) / "models" / role
        root = directory / "snapshot"
        snapshot_download(
            specification["id"],
            revision=revision,
            local_dir=str(root),
            ignore_patterns=["*.md", "*.gguf", ".gitattributes"],
        )
        files = []
        for path in sorted(root.rglob("*")):
            if path.is_file() and ".cache" not in path.relative_to(root).parts:
                hasher = hashlib.sha256()
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        hasher.update(block)
                files.append(
                    {
                        "path": str(path.relative_to(root)),
                        "size": path.stat().st_size,
                        "sha256": hasher.hexdigest(),
                    }
                )
        if not any(f["path"].endswith((".safetensors", ".pt")) for f in files):
            raise ValueError(f"No model weights downloaded for {role}")
        atomic_json(
            directory / "manifest.json",
            {"specification": specification, "files": files},
        )
        result[role] = checkpoint(cache, role)[1]
    return result


def nvidia_uuid(value):
    """Torch may omit the prefix required by nvidia-smi's device selector."""
    value = str(value) if value else ""
    return value if not value or value.startswith(("GPU-", "MIG-")) else "GPU-" + value


def gpu_probe():
    import importlib.metadata
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ModelUnavailable("Expose exactly one allocated CUDA GPU")
    properties = torch.cuda.get_device_properties(0)
    if properties.total_memory < 45 * 10**9:
        raise ModelUnavailable("This experiment requires a 48 GB GPU allocation")
    return {
        "gpu": properties.name,
        "gpu_uuid": nvidia_uuid(getattr(properties, "uuid", "")),
        "total_memory_bytes": properties.total_memory,
        "compute_capability": list(torch.cuda.get_device_capability()),
        "cuda": torch.version.cuda,
        "python": sys.version.split()[0],
        "packages": {
            d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
        },
        "library_path": os.environ.get("LD_LIBRARY_PATH", ""),
        "python_base_prefix": sys.base_prefix,
        "source_installs": {
            d.metadata["Name"]: json.loads(d.read_text("direct_url.json"))
            for d in importlib.metadata.distributions()
            if d.read_text("direct_url.json")
        },
    }


def isolated_environment(cache=None):
    environment = {
        **os.environ,
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "VLLM_NO_USAGE_STATS": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "DO_NOT_TRACK": "1",
    }
    # Compiler/cache defaults under HOME are unsuitable on quota-limited nodes.
    cache_root = cache or os.environ.get("LEAGUE_ASSET_CACHE")
    if cache_root:
        root = Path(cache_root).resolve() / "runtime-cache"
        for name, relative in {
            "TRITON_CACHE_DIR": "triton",
            "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
            "VLLM_CACHE_ROOT": "vllm",
            "XDG_CACHE_HOME": "xdg",
            "NUMBA_CACHE_DIR": "numba",
            "CUDA_CACHE_PATH": "cuda",
            "TMPDIR": "tmp",
            "TMP": "tmp",
            "TEMP": "tmp",
            "TORCH_HOME": "torch",
            "XDG_CONFIG_HOME": "config",
            "XDG_DATA_HOME": "data",
            "XDG_STATE_HOME": "state",
            "VLLM_CONFIG_ROOT": "vllm-config",
            "MPLCONFIGDIR": "matplotlib",
            "FLASHINFER_WORKSPACE_BASE": "flashinfer",
            "HF_HOME": "huggingface",
            "HF_HUB_CACHE": "huggingface/hub",
            "HF_MODULES_CACHE": "huggingface/modules",
            "HF_ASSETS_CACHE": "huggingface/assets",
            "PIP_CACHE_DIR": "pip",
        }.items():
            path = root / relative
            path.mkdir(parents=True, exist_ok=True)
            environment[name] = str(path)
        work_root = os.environ.get("LEAGUE_WORK_ROOT")
        temporary = Path(work_root).resolve() / "tmp" if work_root else root / "tmp"
        temporary.mkdir(parents=True, exist_ok=True)
        for name in ("TMPDIR", "TMP", "TEMP", "VLLM_RPC_BASE_PATH"):
            environment[name] = str(temporary)
    # A venv still uses its base Python's native libraries. Load the Conda
    # runtime before Torch can bind an older system libstdc++, which breaks ICU.
    conda = Path(sys.base_prefix)
    runtime_lib = os.environ.get("LEAGUE_RUNTIME_LIBDIR")
    if (
        runtime_lib is None
        and sys.platform.startswith("linux")
        and (conda / "conda-meta").is_dir()
    ):
        runtime_lib = str(conda / "lib")
    if runtime_lib:
        if not Path(runtime_lib).is_dir():
            raise ModelUnavailable(
                "Missing configured native runtime directory: " + runtime_lib
            )
        paths = [runtime_lib] + [
            p
            for p in environment.get("LD_LIBRARY_PATH", "").split(":")
            if p and p != runtime_lib
        ]
        environment["LD_LIBRARY_PATH"] = ":".join(paths)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    return environment


def worker_signature(python, cache, role="intake"):
    root, pin = checkpoint(cache, role)
    result = subprocess.run(
        [python, "-B", "-m", "annotation.league.local_models", "--probe"],
        check=True,
        text=True,
        capture_output=True,
        timeout=90,
        env=isolated_environment(cache),
    )
    runtime = json.loads(result.stdout)
    if (role == "intake" or pin["specification"]["id"].endswith("-FP8")) and runtime["compute_capability"] < [8, 9]:
        raise ModelUnavailable(
            "Official FP8 intake requires an FP8-capable GPU (Ada or newer)"
        )
    return root, {
        "engine": "vllm",
        "checkpoint": pin,
        "runtime": runtime,
        "context": 16384,
        "batch_size": 1,
        "thinking": False,
        "memory_budget_fraction": 0.85,
        "video_input_policy": {"fps": 2, "frames": 16, "encoded_width": 512},
    }


class LocalVLRunner:
    def __init__(self, store, python, cache, role="intake", signature=None, *, image_limit=4, log_dir=None):
        if type(image_limit) is not int or not 1 <= image_limit <= 8:
            raise ValueError("Local VL image limit must be between one and eight")
        self.image_limit = image_limit
        self.store, self.python, self.role = store, python, role
        self.log_dir = Path(log_dir) if log_dir is not None else store.path("workers")
        self.cache = cache
        self.root, self.identity = (
            worker_signature(python, cache, role) if signature is None else signature
        )
        if image_limit != 4:
            self.identity = {**self.identity, "image_limit": image_limit}
        self.process = self.log = None
        self.failure = None
        self.monitor_stop = threading.Event()
        self.monitor = None
        self.peak_memory_bytes = 0
        self.tokens = {"prompt_tokens": 0, "completion_tokens": 0}
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]

    def _json(self, path, body=None, timeout=10):
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}" + path,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        # Never route local evidence through an inherited HTTP proxy.
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
            request, timeout=timeout
        ) as response:
            return json.load(response)

    def memory_sample(self):
        runtime = self.identity["runtime"]
        gpu_id = (
            runtime.get("gpu_uuid")
            or os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        )
        result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                gpu_id,
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode:
            raise ModelUnavailable(
                "Cannot measure allocated GPU memory: " + result.stderr[-400:]
            )
        self.peak_memory_bytes = max(
            self.peak_memory_bytes, int(result.stdout.strip()) * 1024**2
        )
        if self.peak_memory_bytes > runtime["total_memory_bytes"] * 0.9:
            raise ModelUnavailable("GPU memory exceeds the 90% validation limit")

    def _monitor_memory(self):
        while not self.monitor_stop.wait(0.25):
            try:
                self.memory_sample()
            except (
                OSError,
                ValueError,
                subprocess.SubprocessError,
                ModelUnavailable,
            ) as exc:
                self.failure = str(exc)
                return

    def start(self):
        if self.failure:
            raise ModelUnavailable(self.failure)
        if self.process is not None:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log = (self.log_dir/f"{self.role}-vllm.log").open("a")
        dtype = self.identity["checkpoint"]["specification"]["dtype"]
        command = [
            self.python,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            str(self.root),
            "--served-model-name",
            "league-local",
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--tensor-parallel-size",
            "1",
            "--max-model-len",
            "16384",
            "--max-num-seqs",
            "1",
            "--gpu-memory-utilization",
            "0.85",
            "--dtype",
            dtype,
            "--enforce-eager",
            "--limit-mm-per-prompt",
            json.dumps({"image": self.image_limit, "video": 1}),
            "--media-io-kwargs",
            '{"video":{"num_frames":16,"fps":2}}',
            "--default-chat-template-kwargs",
            '{"enable_thinking":false,"preserve_thinking":false}',
        ]
        try:
            self.process = subprocess.Popen(
                command,
                stdout=self.log,
                stderr=subprocess.STDOUT,
                env=isolated_environment(self.cache),
                start_new_session=True,
            )
            self.monitor_stop.clear()
            self.monitor = threading.Thread(target=self._monitor_memory, daemon=True)
            self.monitor.start()
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                self.store.check_stop()
                if self.failure:
                    raise ModelUnavailable(self.failure)
                if self.process.poll() is not None:
                    raise ModelUnavailable(
                        "vLLM worker exited during startup; inspect worker log"
                    )
                try:
                    self._json("/v1/models")
                    self.memory_sample()
                    return
                except (OSError, ValueError, urllib.error.URLError):
                    time.sleep(0.25)
            raise ModelUnavailable("vLLM startup exceeded ten minutes")
        except BaseException:
            self.close()
            raise

    def complete(self, messages, schema=None, *, max_tokens=512, seed=17, timeout=180):
        if not isinstance(timeout, (int, float)) or not 1 <= timeout <= 900:
            raise ValueError("Local VL request timeout must be 1..900 seconds")
        self.store.check_stop()
        self.start()
        started = time.monotonic()
        request = {
            "model": "league-local",
            "messages": messages,
            "temperature": 0,
            "presence_penalty": 0,
            "max_tokens": max_tokens,
            "seed": seed,
            "chat_template_kwargs": {
                "enable_thinking": False,
                "preserve_thinking": False,
            },
        }
        if schema is not None:
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "league_reading",
                    "strict": True,
                    "schema": schema,
                },
            }
        try:
            response = self._json("/v1/chat/completions", request, timeout=timeout)
            self.memory_sample()
            if self.failure:
                raise ModelUnavailable(self.failure)
            choice = response["choices"][0]
            usage = response.get("usage", {})
            for key in self.tokens:
                self.tokens[key] += usage.get(key, 0)
            return {
                "text": choice["message"]["content"],
                "finish_reason": choice["finish_reason"],
                "usage": usage,
                "wall_seconds": time.monotonic() - started,
                "model": self.identity,
                "peak_memory_bytes": self.peak_memory_bytes,
            }
        except StopRequested:
            raise
        except (OSError, ValueError, KeyError, IndexError, ModelUnavailable) as exc:
            self.failure = f"Local VL worker failed: {exc}"
            self.close()
            raise ModelUnavailable(self.failure) from exc

    def close(self):
        self.monitor_stop.set()
        if self.monitor is not None:
            self.monitor.join(timeout=12)
            self.monitor = None
        if self.process is not None:
            import signal

            process, self.process = self.process, None
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        if self.log is not None:
            self.log.close()
            self.log = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(gpu_probe()))
