"""Isolated CPU Paddle recognizer and explicit online model-cache installer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
from urllib.request import urlopen

MODELS = {"english": "PP-OCRv5_server_rec", "korean": "korean_PP-OCRv5_mobile_rec"}
BASE = "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/"


def install(cache):
    cache.mkdir(parents=True, exist_ok=True)
    # Paddle 3.3 imports its dataset module even for OCR; that module hard-codes
    # ~/.cache/paddle. Preserve any existing cache, otherwise add a work-storage
    # compatibility link during online setup, never during an allocated job.
    legacy = Path.home() / ".cache/paddle"
    if not legacy.exists():
        if legacy.is_symlink():
            raise ValueError(
                "Existing Paddle cache link is broken; repair it before setup"
            )
        target = Path(
            os.environ.get("LEAGUE_PADDLE_CACHE", str(cache / "runtime/paddle"))
        ).resolve()
        target.mkdir(parents=True, exist_ok=True)
        legacy.parent.mkdir(parents=True, exist_ok=True)
        legacy.symlink_to(target, target_is_directory=True)
    for model in MODELS.values():
        directory = cache / model
        if not (directory / "inference.pdiparams").exists():
            with tempfile.TemporaryDirectory(dir=cache) as temporary:
                archive_path = Path(temporary) / "model.tar"
                with urlopen(
                    BASE + model + "_infer.tar", timeout=90
                ) as response, archive_path.open("wb") as output:
                    shutil.copyfileobj(response, output)
                with tarfile.open(archive_path) as archive:
                    archive.extractall(temporary, filter="data")
                candidates = list(Path(temporary).rglob("inference.pdiparams"))
                if len(candidates) != 1:
                    raise ValueError("Unexpected recognition model archive")
                if directory.exists():
                    raise ValueError(
                        f"Incomplete model directory already exists: {directory}; choose a new OCR cache"
                    )
                shutil.move(str(candidates[0].parent), directory)
    files = {
        str(p.relative_to(cache)): hashlib.sha256(p.read_bytes()).hexdigest()
        for model in MODELS.values()
        for p in (cache / model).rglob("*")
        if p.is_file()
    }
    temporary = cache / "models.json.tmp"
    temporary.write_text(
        json.dumps({"models": MODELS, "files": files}, sort_keys=True, indent=2) + "\n"
    )
    temporary.replace(cache / "models.json")
    print(json.dumps({"models": MODELS, "files": len(files)}))


def signature(cache):
    from importlib.metadata import version

    manifest = json.loads((cache / "models.json").read_text())
    if manifest.get("models") != MODELS:
        raise ValueError("OCR model manifest is incompatible")
    for name, expected in manifest["files"].items():
        path = (cache / name).resolve()
        if (
            not path.is_relative_to(cache.resolve())
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError(f"OCR model hash mismatch: {name}")
    return {
        "engine": "paddleocr",
        "packages": {
            name: version(name)
            for name in (
                "paddleocr",
                "paddlepaddle",
                "paddlex",
                "numpy",
                "pillow",
                "opencv-contrib-python",
            )
        },
        "model_manifest_sha256": hashlib.sha256(
            (cache / "models.json").read_bytes()
        ).hexdigest(),
        "worker_implementation": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "recognition_options": {
            "device": "cpu",
            "cpu_threads": 2,
            "enable_mkldnn": True,
        },
    }


def serve(cache):
    # Native libraries can print to fd 1. Reserve the original fd exclusively
    # for protocol responses and send every library diagnostic to stderr.
    protocol = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["PADDLE_PDX_CACHE_HOME"] = str(cache / "runtime/paddlex")
    os.environ["PADDLE_HOME"] = str(cache / "runtime/paddle")
    os.environ["XDG_CACHE_HOME"] = str(cache / "runtime/xdg")
    metadata = signature(cache)
    models = {}

    def model(language):
        if language not in models:
            from paddleocr import TextRecognition

            models[language] = TextRecognition(
                model_name=MODELS[language],
                model_dir=str(cache / MODELS[language]),
                device="cpu",
                cpu_threads=2,
                enable_mkldnn=True,
            )
        return models[language]

    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("op") == "ready":
                for language in MODELS:
                    model(language)
                result = metadata
            else:
                language = request.get("language", "english")
                if language not in MODELS:
                    raise ValueError("Unsupported recognition language")
                prediction = next(
                    iter(model(language).predict(input=request["path"], batch_size=1))
                )
                prediction = prediction.json
                if isinstance(prediction, str):
                    prediction = json.loads(prediction)
                value = prediction.get("res", prediction)
                result = {
                    "raw_text": value["rec_text"],
                    "score": float(value["rec_score"]),
                    "engine": "paddleocr:" + MODELS[language],
                }
            protocol.write(json.dumps(result, ensure_ascii=False) + "\n")
        except Exception as exc:
            protocol.write(json.dumps({"error": f"{type(exc).__name__}: {exc}"}) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.download:
        install(args.cache)
    else:
        serve(args.cache)
