"""Lazy GPU runtime; no model imports are required by preparation or review."""
from __future__ import annotations

import copy
import gc
import json
import math
import re
import time
from pathlib import Path

from .core import Config, Store, atomic_json, digest, now, read_json, runtime_info


class InputBudgetError(RuntimeError):
    pass


def parse_output(text: str | dict) -> dict:
    if isinstance(text, dict):
        return text
    # Never interpret JSON in a Thinking model's private reasoning as its answer.
    text = text.rsplit("</think>", 1)[-1].strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        # Permit prose before a single final JSON object, but not competing objects.
        decoder = json.JSONDecoder()
        candidates = []
        for match in re.finditer(r"\{", text):
            try:
                value, size = decoder.raw_decode(text[match.start():])
                if not text[match.start() + size:].strip() and isinstance(value, dict):
                    candidates.append(value)
            except json.JSONDecodeError:
                pass
        if len(candidates) != 1:
            raise ValueError("Model completion is not one final JSON object")
        value = candidates[0]
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    json.dumps(value, allow_nan=False)
    return value


def pin_models(store: Store, config: Config, download=False, processors_only=False) -> dict:
    from huggingface_hub import HfApi, snapshot_download
    manifest = read_json(store.path("manifest.json"))
    pins = manifest.setdefault("model_revisions", {})
    for role, model in config.models.items():
        identity = {"id": model["id"], "requested_revision": model["revision"]}
        old = pins.get(role, {})
        if all(old.get(k) == v for k, v in identity.items()):
            revision = old["revision"]
        elif re.fullmatch(r"[0-9a-f]{40}", model["revision"]):
            revision = model["revision"]
        else:
            revision = HfApi().model_info(model["id"], revision=model["revision"]).sha
        # With download=False, confirm that weights and processor already exist locally.
        patterns = ["*.json", "*.txt", "*.jinja", "*.model"] if processors_only else None
        snapshot = snapshot_download(model["id"], revision=revision, local_files_only=not download,
                                     allow_patterns=patterns)
        if not processors_only:
            index = Path(snapshot) / "model.safetensors.index.json"
            shards = set(read_json(index)["weight_map"].values()) if index.exists() else {"model.safetensors"}
            if not all((Path(snapshot)/shard).is_file() for shard in shards):
                raise RuntimeError("Incomplete model weight cache; run cache-models before the GPU job")
        pins[role] = {**identity, "revision": revision, "snapshot": snapshot}
        atomic_json(store.path("manifest.json"), manifest)
    atomic_json(store.path("environment.lock.json"), {"captured_at": now(), **runtime_info(), "models": pins})
    return pins


def preflight(store: Store, config: Config) -> dict:
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one CUDA GPU for the reference runner")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is not supported by the allocated GPU")
    # Catch an incompatible driver/runtime before loading either checkpoint.
    sample = torch.ones((32, 32), device="cuda:0", dtype=torch.bfloat16)
    _ = sample @ sample
    torch.cuda.synchronize()
    del sample, _
    pins = pin_models(store, config)
    properties = torch.cuda.get_device_properties(0)
    result = {"name": properties.name, "total_memory_bytes": properties.total_memory,
              "cuda": torch.version.cuda, "dtype": "bfloat16", "batch_size": 1,
              "models": pins, "environment": runtime_info(), "time": now()}
    packages = {k.casefold(): v for k, v in result["environment"]["packages"].items()}
    signature = {"packages": {k: packages.get(k) for k in (
        "torch", "torchvision", "transformers", "qwen-vl-utils", "pillow", "accelerate")},
        "cuda": torch.version.cuda, "dtype": "bfloat16", "attention": "sdpa", "device": properties.name}
    manifest = read_json(store.path("manifest.json"))
    manifest["inference_runtime"] = signature
    atomic_json(store.path("manifest.json"), manifest)
    atomic_json(store.path("preflight.json"), result)
    atomic_json(store.path("preflights/"+result["time"].replace(":", "-")+".json"), result)
    return result


def processor_preview(store: Store, config: Config) -> dict:
    """Download processor files only and audit the real source crops without GPU weights."""
    from transformers import AutoProcessor
    pins = pin_models(store, config, download=True, processors_only=True)
    audits = []
    for role in ("instruct", "thinking"):
        processor = AutoProcessor.from_pretrained(pins[role]["snapshot"], local_files_only=True)
        for route in (r for r in config.routes if r.model == role):
            frames = [f for f in store.rows("previews") if f["route"] == route.name]
            # Previews are separated source samples, represented as ordered images (not a continuous video).
            for frame in frames:
                _, audit = prepare_inputs(store, config, pins, processor,
                    [{"role": "user", "content": "Inspect this fixed gameplay layout crop."}], [frame], route,
                    store.path(f"processor-previews/{route.name}/{frame['id']}"))
                audits.append(audit)
            if route.name in {"selected_hud", "gameplay", "full_context"}:
                from .media import Media
                from .prompts import observer_prompt
                media = Media(store, config)
                lower, upper = media.manifest["interval"]
                begin = max(lower, min(780, upper-config.window_seconds))
                end = min(upper, begin+config.window_seconds)
                frames = media.frames(begin, end, config.fps, route)
                if role == "instruct":
                    frames = frames[:config.hud_frames_per_request]
                _, audit = prepare_inputs(store, config, pins, processor,
                    [{"role": "user", "content": observer_prompt(route, frames, config.fps)}], frames, route,
                    store.path(f"processor-previews/{route.name}/sequence"), fps=config.fps)
                audits.append(audit)
    validation = read_json(store.path("layout_validation.json"))
    validation.update(model_preprocessing_validated=True, processor_preview_count=len(audits),
                      preprocessing_key=store.fingerprint(config),
                      hud_legibility="Inspect actual resized previews; no semantic inference or human verification performed")
    atomic_json(store.path("layout_validation.json"), validation)
    return {"audits": len(audits), "models": pins, "model_weights_loaded": False}


def prepare_inputs(store, config, pins, processor, messages, frames, route, task_dir, *, fps=1.0, visual_limit=None):
    """CPU-capable preprocessing shared by previews and GPU inference."""
    from qwen_vl_utils import process_vision_info
    maximum = (config.instruct_output_tokens if route.model == "instruct"
               else config.thinking_output_tokens)
    task_dir.mkdir(parents=True, exist_ok=True)
    inputs = None
    for attempt in range(6):
        pixels = max(4096, int(route.max_pixels / 2 ** attempt))
        conversation = copy.deepcopy(messages)
        user = next(m for m in conversation if m["role"] == "user")
        if isinstance(user["content"], str):
            user["content"] = [{"type": "text", "text": user["content"]}]
        paths = [str(store.path(f["path"])) for f in frames]
        if route.model == "thinking" and len(paths) > 1:
            visual = [{"type": "video", "video": paths, "sample_fps": fps,
                       "min_pixels": max(4096, pixels//2), "max_pixels": pixels,
                       "total_pixels": pixels * len(paths)}]
        else:
            visual = [{"type": "image", "image": p, "min_pixels": max(4096, pixels//2),
                       "max_pixels": pixels} for p in paths]
        user["content"] = visual + user["content"]
        text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        images, videos, kwargs = process_vision_info(conversation, image_patch_size=16,
            return_video_kwargs=True, return_video_metadata=True)
        tensors, metadata = (list(v) for v in zip(*videos)) if videos else (None, None)
        if metadata:
            # The processor embeds timestamps from indices/fps. Keep them on the
            # source clock rather than resetting every supplied clip to zero.
            source = read_json(store.path("manifest.json"))["source"]
            for tensor, meta in zip(tensors, metadata):
                meta.update(fps=source["fps"], total_num_frames=math.ceil(source["duration"]*source["fps"]),
                            frames_indices=[round(frames[min(i, len(frames)-1)]["time"]*source["fps"])
                                            for i in range(len(tensor))])
        inputs = processor(text=[text], images=images, videos=tensors,
            video_metadata=metadata, do_resize=False, return_tensors="pt", **kwargs)
        input_tokens = int(inputs["input_ids"].shape[-1])
        visual_tokens = sum(int((grid.prod(-1) // 4).sum()) for name, grid in inputs.items()
                            if name in {"image_grid_thw", "video_grid_thw"})
        if input_tokens + maximum <= config.max_context:
            break
        del inputs
        inputs = None
    if inputs is None:
        raise InputBudgetError("Input cannot fit declared context without exceeding minimum pixel budget")
    if visual_limit is not None and visual_tokens > visual_limit:
        raise InputBudgetError(f"Review visual budget exhausted: need {visual_tokens}, remaining {visual_limit}")
    # Save the actual resized input pixels, before processor normalization/patch packing.
    processed = []
    for i, image in enumerate(images or []):
        path = task_dir / f"input-image-{i:04d}.png"
        image.save(path)
        processed.append({"path": str(path.relative_to(store.root)), "size": list(image.size),
                          "source_frame": frames[i % len(frames)]["id"]})
    if tensors:
        from PIL import Image
        for vi, tensor in enumerate(tensors):
            for fi, tensor_frame in enumerate(tensor):
                path = task_dir / f"input-video-{vi:02d}-{fi:04d}.png"
                array = tensor_frame.permute(1, 2, 0).cpu().numpy().astype("uint8")
                image = Image.fromarray(array)
                image.save(path)
                processed.append({"path": str(path.relative_to(store.root)), "size": list(image.size),
                                  "source_frame": frames[min(fi, len(frames)-1)]["id"],
                                  "temporal_padding": fi >= len(frames)})
    audit = {"frames": frames, "processed_inputs": processed, "pixel_budget_per_frame": pixels,
             "input_tokens": input_tokens, "visual_tokens": visual_tokens,
             "output_token_limit": maximum, "context_limit": config.max_context,
             "input_hash": digest({"messages": messages, "frames": frames}),
             "model": pins[route.model], "video_metadata": metadata,
                 "runtime": read_json(store.path("manifest.json")).get("inference_runtime"),
             "resized_once": True, "route": route.name, "time": now()}
    # Some processor metadata includes numpy integer scalars.
    audit = json.loads(json.dumps(audit, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v)))
    atomic_json(task_dir / "input-audit.json", audit)
    return inputs, audit


class QwenRunner:
    def __init__(self, store: Store, config: Config):
        self.store, self.config = store, config
        self.pins = preflight(store, config)["models"]
        self.role = None
        self.model = self.processor = None

    def load(self, role: str) -> None:
        if role == self.role:
            return
        self.close()
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        pin = self.pins[role]
        self.processor = AutoProcessor.from_pretrained(pin["snapshot"], local_files_only=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            pin["snapshot"], dtype=torch.bfloat16, device_map={"": 0},
            attn_implementation="sdpa", local_files_only=True).eval()
        self.role = role

    def close(self) -> None:
        self.model = self.processor = None
        self.role = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def infer(self, messages, frames, route, task_dir: Path, *, fps=1.0, visual_limit=None) -> dict:
        import torch
        self.load(route.model)
        self.store.check_stop()
        start = time.monotonic()
        inputs, audit = prepare_inputs(self.store, self.config, self.pins, self.processor,
            messages, frames, route, task_dir, fps=fps, visual_limit=visual_limit)
        input_tokens, visual_tokens = audit["input_tokens"], audit["visual_tokens"]
        maximum = audit["output_token_limit"]
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(self.config.seed)
        try:
            inputs = inputs.to("cuda:0")
            with torch.inference_mode():
                generated = self.model.generate(**inputs, max_new_tokens=maximum,
                    do_sample=route.model == "thinking", temperature=1.0, top_p=0.95, top_k=20)
            completion = generated[0, input_tokens:]
            raw = self.processor.decode(completion, skip_special_tokens=True)
            eos = self.model.generation_config.eos_token_id
            terminated = bool(len(completion)) and int(completion[-1]) in (eos if isinstance(eos, list) else [eos])
            metrics = {"wall_seconds": time.monotonic()-start, "input_tokens": input_tokens,
                       "visual_tokens": visual_tokens, "output_tokens": len(completion),
                       "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
                       "finish_reason": "stop" if terminated else "length"}
            atomic_json(task_dir / "completion.json", {"raw": raw, "metrics": metrics})
            if not terminated and len(completion) >= maximum:
                return {"status": "truncated", "output": None, "metrics": metrics}
            return {"status": "ok", "output": parse_output(raw), "metrics": metrics}
        except torch.cuda.OutOfMemoryError as exc:
            atomic_json(task_dir / "completion.json", {"raw": None, "status": "out_of_memory",
                "metrics": {"wall_seconds": time.monotonic()-start, "visual_tokens": visual_tokens,
                            "input_tokens": input_tokens, "output_tokens": 0,
                            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated()}})
            torch.cuda.empty_cache()
            raise InputBudgetError("GPU out of memory; lower configured context/pixel budgets and resume") from exc


class ScopedResolver:
    """Enforce solver static-reference policy outside the model prompt."""
    def __init__(self, resolver, category: str):
        self.resolver, self.category = resolver, category

    def system_context(self):
        return self.resolver.system_context() + ("\nThis task permits identity/icon references only."
                if self.category == "visual_only" else "\nPatch-specific mechanics references are permitted.")

    def handle_query(self, request):
        if self.category != "visual_only":
            return self.resolver.handle_query(request)
        if not isinstance(request.get("query_knowledge"), str) or set(request) - {
                "query_knowledge", "entity_type", "ocr_evidence"}:
            raise ValueError("Visual-only lookups require an identity string and optional OCR evidence")
        entity_type = request.get("entity_type", "champion")
        if entity_type not in {"champion", "item"}:
            return {"status": "forbidden", "error": "Only champion/item identity references are permitted"}
        try:
            identity = self.resolver._identify(request["query_knowledge"], entity_type)
        except (KeyError, ValueError, RuntimeError) as exc:
            return {"status": "unresolved", "error": str(exc)}
        from .knowledge import DD
        icon = (f"champion/{identity['internal_id']}.png" if entity_type == "champion"
                else f"item/{identity['numeric_id']}.png")
        return {"status": "ok", "scope": "identity_only", "identity": identity,
                "icon_url": f"{DD}/cdn/{self.resolver.patch.ddragon_version}/img/{icon}"}


def infer_with_knowledge(runner, resolver, messages, frames, route, directory,
                         max_queries=3, fps=1.0, visual_limit=None):
    from .knowledge import run_knowledge_loop
    calls = []

    def generate(conversation):
        remaining = None if visual_limit is None else visual_limit - sum(c["metrics"].get("visual_tokens", 0) for c in calls)
        result = runner.infer(conversation, frames, route, directory / f"call-{len(calls)}",
                              fps=fps, visual_limit=remaining)
        calls.append(result)
        if result["status"] != "ok":
            raise InputBudgetError(f"Model generation failed: {result['status']}")
        return result["output"]

    result = run_knowledge_loop(generate, messages, resolver, max_queries=max_queries)
    result.pop("messages", None)
    result["calls"] = [{k: v for k, v in call.items() if k != "output"} for call in calls]
    result["metrics"] = {k: sum(c["metrics"].get(k, 0) for c in calls)
                         for k in ("wall_seconds", "input_tokens", "visual_tokens", "output_tokens")}
    result["metrics"]["peak_gpu_memory_bytes"] = max(
        (c["metrics"].get("peak_gpu_memory_bytes", 0) for c in calls), default=0)
    atomic_json(directory / "knowledge-transcript.json", result)
    return result
