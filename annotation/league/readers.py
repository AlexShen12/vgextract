"""Bounded field recognition. Importing this module never imports model runtimes."""

from __future__ import annotations

import csv
import io
import os
from pathlib import Path
import re
import subprocess
import unicodedata


def unknown(reason, raw=None, attempts=()):
    return {
        "acceptance": "unknown",
        "reason": reason,
        "value": None,
        "raw_text": raw,
        "attempts": list(attempts),
    }


def parse_text(raw: str, parser: str):
    text = unicodedata.normalize("NFKC", raw).strip()
    if not text:
        raise ValueError("no text recognized")
    if parser in {"integer", "level", "rank"}:
        if not re.fullmatch(r"\d{1,5}", text):
            raise ValueError("expected integer digits")
        value = int(text)
        # These are display-width bounds, not game-mechanics assumptions. Some
        # recording modes display levels above 18 and nonstandard ability ranks.
        maximum = {"integer": 99999, "level": 99, "rank": 9}[parser]
        if not (1 if parser == "level" else 0) <= value <= maximum:
            raise ValueError("number outside field range")
        return value
    if parser in {"clock", "timer"}:
        if parser == "timer" and text.upper() == "LIVE":
            return {"displayed_text": raw, "state": "LIVE"}
        match = re.fullmatch(r"(\d{1,3}):([0-5]\d)", text)
        if not match:
            raise ValueError("expected displayed m:ss timer")
        minutes, seconds = map(int, match.groups())
        return {"displayed_text": raw, "seconds": minutes * 60 + seconds}
    if parser == "gold":
        if not re.fullmatch(r"\d+(?:\.\d+)?[kK]?", text):
            raise ValueError("expected displayed gold number")
        return {
            "displayed_text": raw,
            "amount": float(text.rstrip("kK"))
            * (1000 if text[-1:].lower() == "k" else 1),
        }
    if parser == "health":
        match = re.fullmatch(r"(\d{1,6})\s*(?:/|\s+)\s*(\d{1,6})", text)
        if not match:
            raise ValueError("expected current/maximum health")
        current, maximum = map(int, match.groups())
        if not 0 <= current <= maximum or maximum == 0:
            raise ValueError("invalid health bounds")
        return {"displayed_text": raw, "current": current, "maximum": maximum}
    if parser == "kda":
        if not re.fullmatch(r"\d{1,3}/\d{1,3}/\d{1,3}", text):
            raise ValueError("expected K/D/A")
        return dict(zip(("kills", "deaths", "assists"), map(int, text.split("/"))))
    if parser == "cooldown":
        if not re.fullmatch(r"\d{1,3}(?:\.\d)?", text):
            raise ValueError("no explicit cooldown numeral")
        return {"displayed_text": raw, "seconds": float(text)}
    if parser == "objective":
        names = {
            "rift herald",
            "baron nashor",
            "cloud drake",
            "mountain drake",
            "infernal drake",
            "ocean drake",
            "hextech drake",
            "chemtech drake",
            "elder dragon",
            "atakhan",
        }
        if text.casefold() not in names:
            raise ValueError("unrecognized objective label")
        return text
    if parser == "name":
        if (
            len(text) > 40
            or not any(c.isalnum() for c in text)
            or any(ord(c) < 32 for c in text)
        ):
            raise ValueError("invalid visible name")
        return text
    if parser == "notice":
        if len(text) > 300 or any(ord(c) < 32 for c in text):
            raise ValueError("invalid visible notice")
        return text
    raise ValueError("unsupported parser")


def tesseract(path: Path, alternate=False, whitelist=None) -> dict:
    command = [
        os.environ.get("LEAGUE_TESSERACT", "tesseract"),
        str(path),
        "stdout",
        "--psm",
        "8" if alternate else "7",
        "-l",
        "eng",
    ]
    if os.environ.get("LEAGUE_TESSDATA"):
        command += ["--tessdata-dir", os.environ["LEAGUE_TESSDATA"]]
    if whitelist:
        command += ["-c", "tessedit_char_whitelist=" + whitelist]
    command += ["tsv"]
    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        timeout=15,
        check=True,
        env={**os.environ, "OMP_THREAD_LIMIT": "1"},
    )
    words = [
        row
        for row in csv.DictReader(io.StringIO(result.stdout), delimiter="\t")
        if row.get("text", "").strip() and float(row.get("conf", -1)) >= 0
    ]
    return {
        "raw_text": " ".join(w["text"] for w in words),
        "score": min((float(w["conf"]) / 100 for w in words), default=0),
        "engine": "tesseract",
    }


def measure_bar(image, field):
    import numpy as np

    array = np.asarray(image.convert("RGB"), dtype=float)
    r, g, b = array[..., 0], array[..., 1], array[..., 2]
    color = field.get("color")
    mask = (
        ((g > r * 1.3) & (g > b * 1.15) & (g > 60))
        if color == "green"
        else (
            (r > g * 1.3) & (b > g * 1.3) & (b > 60)
            if color == "purple"
            else (b > r * 1.3) & (b > 60)
        )
    )
    filled = mask.mean(axis=0) >= 0.4
    if not filled.any():
        return {
            "acceptance": "unknown",
            "reason": "bar color not established",
            "raw_text": None,
            "value": None,
        }
    return {
        "acceptance": "accepted",
        "reason": "measured colored bar",
        "raw_text": None,
        "value": float(filled.mean()),
        "reader": {"engine": "bar-color-v1"},
    }


class IconMatcher:
    def __init__(self, cache, manifest, thresholds):
        import numpy as np
        from PIL import Image

        self.thresholds = thresholds
        self.references = {}
        for entry in manifest["assets"]:
            with Image.open(Path(cache) / entry["path"]) as image:
                vectors = [self.vector(image, inset) for inset in (0.08, 0.18)]
            self.references.setdefault(entry["kind"], []).append((entry, vectors))
        self.matrices = {
            kind: np.asarray(
                [v for _, vectors in entries for v in vectors], dtype=np.float32
            )
            for kind, entries in self.references.items()
        }

    @staticmethod
    def vector(image, inset=0.08):
        import numpy as np
        from PIL import ImageFilter

        w, h = image.size
        image = (
            image.convert("RGB")
            .crop(
                (
                    int(w * inset),
                    int(h * inset),
                    int(w * (1 - inset)),
                    int(h * (1 - inset)),
                )
            )
            .resize((24, 24))
        )
        color = np.asarray(image, dtype=float) / 255
        # Centered color and edge texture reduce sensitivity to cooldown darkening.
        gray = (
            np.asarray(image.convert("L").filter(ImageFilter.FIND_EDGES), dtype=float)
            / 255
        )
        v = np.concatenate([(color - color.mean(axis=(0, 1))).ravel(), gray.ravel()])
        return v / max(float(np.linalg.norm(v)), 1e-9)

    def read(self, image, field):
        import numpy as np

        array = np.asarray(image.convert("RGB"), dtype=float)
        if field.get("empty_allowed") and array.mean() < 28 and array.std() < 18:
            return {
                "acceptance": "absent",
                "reason": "empty dark slot",
                "value": None,
                "raw_text": None,
            }
        queries = [self.vector(image, inset) for inset in (0.08, 0.18)]
        entries = self.references.get(field["kind"], [])
        scores = []
        if entries:
            similarities = (
                self.matrices[field["kind"]] @ np.asarray(queries, dtype=np.float32).T
            ).reshape(len(entries), 2, 2)
            similarities = similarities.max(axis=(1, 2))
            scores = [
                (float(score), entry)
                for score, (entry, _) in zip(similarities, entries)
            ]
        scores.sort(key=lambda v: v[0], reverse=True)
        if not scores:
            raise ValueError("Required icon reference category missing")
        score, entry = scores[0]
        margin = score - (scores[1][0] if len(scores) > 1 else 0)
        accepted = (
            score >= self.thresholds["icon"]
            and margin >= self.thresholds["icon_margin"]
        )
        return {
            "acceptance": "accepted" if accepted else "unknown",
            "raw_text": None,
            "reason": "matched visual reference" if accepted else "ambiguous icon",
            "value": entry["id"] if accepted else None,
            "reader": {
                "engine": "masked-template-v1",
                "score": score,
                "margin": margin,
            },
            "candidates": [{"id": e["id"], "score": s} for s, e in scores[:3]],
        }
