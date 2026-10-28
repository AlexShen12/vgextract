"""One fixed preprocessing and Tesseract call for the enabled numeric fields."""

from pathlib import Path
import subprocess
from .readers import parse_text, tesseract, measure_bar


def numeric_unknown(reason, raw=None, attempts=()):
    return {
        "acceptance": "unknown",
        "reason": reason,
        "value": None,
        "raw_text": raw,
        "attempts": list(attempts),
    }


def numeric_preprocess(image):
    """One fixed bright-glyph recipe for the reduced numeric HUD fields."""
    import numpy as np
    from PIL import Image, ImageOps

    image = image.convert("RGB").resize(
        (image.width * 4, image.height * 4), Image.Resampling.BICUBIC
    )
    bright = np.asarray(image).max(axis=2) >= 125
    return ImageOps.expand(
        Image.fromarray(np.where(bright, 0, 255).astype("uint8")), border=12, fill=255
    )


class NumericReader:
    """One Tesseract attempt, one preprocessing recipe, no Paddle fallback."""

    def __init__(self, threshold=0.85):
        self.threshold = threshold

    def read(self, image, field, directory):
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "attempt-0.png"
        numeric_preprocess(image).save(path)
        if field["reader"] == "bar":
            return measure_bar(image, field)
        whitelist = {
            "clock": "0123456789:",
            "timer": "0123456789:LIVE",
            "gold": "0123456789.kK",
            "kda": "0123456789/",
        }.get(field["parser"], "0123456789")
        result = {}
        try:
            result = tesseract(path, whitelist=whitelist)
            value = parse_text(result["raw_text"], field["parser"])
            attempt = {**result, "value": value, "path": str(path)}
            if result["score"] < self.threshold:
                return numeric_unknown(
                    "Tesseract confidence below frozen gate",
                    result["raw_text"],
                    [attempt],
                )
            return {
                "acceptance": "accepted",
                "reason": "numeric grammar and calibrated OCR gate",
                "value": value,
                "raw_text": result["raw_text"],
                "reader": result,
                "attempts": [attempt],
            }
        except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
            return numeric_unknown(
                str(exc),
                result.get("raw_text"),
                [{**result, "path": str(path), "error": str(exc)}],
            )
