"""Offline SigLIP vision embeddings for 3D skin render retrieval."""

from pathlib import Path
import re

import numpy as np
import PIL

from .core import digest, implementation_hash, read_json
from .media import file_hash


MODEL_ID = "google/siglip-base-patch16-224"
DEFAULT_SIGLIP_LOCK = Path("data/league_models/siglip-base-patch16-224/model-lock.json")
CHECKPOINT_FILES = {"config.json", "preprocessor_config.json", "model.safetensors"}


class SiglipEncoder:
    def __init__(self, lock_path, lock_hash):
        lock_path = Path(lock_path).resolve()
        lock = read_json(lock_path)
        if digest(lock) != lock_hash:
            raise ValueError("SigLIP model lock changed")
        if (lock.get("model_id") != MODEL_ID or not re.fullmatch(r"[0-9a-f]{40}", lock.get("revision", ""))
                or lock.get("files", {}).keys() != CHECKPOINT_FILES):
            raise ValueError(f"Expected a pinned {MODEL_ID} model lock with config, processor, and safetensors hashes")
        for name, expected in sorted(lock["files"].items()):
            path = lock_path.parent / name
            if not path.is_file() or file_hash(path) != expected:
                raise ValueError(f"SigLIP checkpoint file is missing or differs from the pin: {path}")

        try:
            import torch
            import transformers
            from transformers import SiglipImageProcessor, SiglipVisionModel
        except ImportError as exc:
            raise RuntimeError("SigLIP skin matching requires PyTorch and Transformers with SigLIP support") from exc

        self.torch = torch
        self.processor = SiglipImageProcessor.from_pretrained(lock_path.parent, local_files_only=True)
        model, loading = SiglipVisionModel.from_pretrained(
            lock_path.parent, local_files_only=True, use_safetensors=True, output_loading_info=True,
        )
        problems = {key: loading[key] for key in ("missing_keys", "mismatched_keys", "error_msgs") if loading.get(key)}
        if problems:
            raise ValueError(f"Incomplete SigLIP vision checkpoint: {problems}")
        self.model = model.eval().requires_grad_(False).to(device="cpu", dtype=torch.float32)
        self.signature = digest({"pin": lock, "implementation": implementation_hash(["body_siglip.py"]),
                                 "torch": str(torch.__version__), "transformers": transformers.__version__,
                                 "Pillow": PIL.__version__, "numpy": np.__version__, "device": "cpu", "dtype": "float32"})

    def describe_batch(self, images):
        pixels = self.processor(images=images, return_tensors="pt")["pixel_values"]
        with self.torch.inference_mode():
            return self.model(pixel_values=pixels).pooler_output.float().cpu().numpy()
