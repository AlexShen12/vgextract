"""Read the numeral in a champion's overhead level badge."""

from functools import lru_cache
import io
import os
import subprocess

import numpy as np
from PIL import Image, ImageOps


LEVEL_READER_VERSION = 2


@lru_cache(maxsize=4096)
def read_digit_mask(binary, size, executable):
    glyph = Image.frombytes("L", size, binary)
    enlarged = glyph.resize((glyph.width * 7, glyph.height * 7), Image.Resampling.LANCZOS)
    prepared = ImageOps.expand(enlarged, 18, fill=255)
    buffer = io.BytesIO()
    prepared.save(buffer, format="PNG")
    result = subprocess.run(
        [executable, "stdin", "stdout", "--psm", "10", "-c", "tessedit_char_whitelist=0123456789", "tsv"],
        input=buffer.getvalue(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=5,
    )
    words = [row.rsplit("\t", 2)[-2:] for row in result.stdout.decode().splitlines() if row.startswith("5\t")]
    if len(words) != 1:
        return None
    confidence, text = words[0]
    if text in {str(n) for n in range(1, 19)} and float(confidence) >= 55:
        return int(text)
    return None


def champion_level(image, bar_box, level_cell):
    x, y = bar_box[:2]
    # The colored fill can start a few pixels away from the health-bar frame.
    a, b, c, d = level_cell
    box = x + a - 3, y + b + 4, x + c, y + d - 1
    if box[0] < 0 or box[1] < 0 or box[2] > image.width or box[3] > image.height:
        return None
    rgb = np.asarray(image.crop(box).convert("RGB"))
    low, high = rgb.min(axis=2), rgb.max(axis=2)
    white = (low >= 145) & (high - low < 78)
    ys, xs = np.nonzero(white)
    if len(xs) < 7:
        return None
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    width, height = x1 - x0, y1 - y0
    if not (2 <= width <= 17 and 6 <= height <= 12 and .12 <= len(xs) / (width * height) <= .8):
        return None
    center_x = box[0] + round(float(xs.mean()))
    center_y = box[1] + round(float(ys.mean()))
    # Check the badge plate around the numeral, including pixels beyond the OCR crop.
    # Damage counters may look like digits but usually have no dark enclosing badge.
    badge = np.asarray(image.crop((center_x - 7, center_y - 7, center_x + 8, center_y + 8)).convert("RGB"))
    if (badge.max(axis=2) < 110).mean() < .8:
        return None
    binary = (~white[y0:y1, x0:x1]).astype("uint8") * 255
    return read_digit_mask(binary.tobytes(), (width, height), os.environ.get("LEAGUE_TESSERACT", "tesseract"))


def level_proposals(image, profile, proposals):
    cell = profile["body_binder"]["level_cell"]
    return [{**row, "level": level} for row in proposals
            if (level := champion_level(image, row["bar_box"], cell)) is not None]
