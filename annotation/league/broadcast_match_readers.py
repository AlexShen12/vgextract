"""Native Worlds spectator crops using the existing inventory and OCR readers."""

from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageOps

from .broadcast_ocr import grayscale
from .inventory_reader import ChampionPortraitMatcher, InventoryReader
from .killfeed_reader import Entry
from .numeric_reader import NumericReader, numeric_preprocess
from .readers import tesseract


def pil_box(box, y=0):
    return tuple(point + y if i % 2 else point for i, point in enumerate(box))


def white_mask(image, threshold=165):
    pixels = np.asarray(image.convert('RGB'), dtype=np.int16)
    return ((pixels.min(axis=-1) >= threshold) & (np.ptp(pixels, axis=-1) <= 60)).astype('uint8')


def gray_box(image, box):
    return np.asarray(image.crop(tuple(box)).convert('L'))


class SpectatorFeed:
    def __init__(self, profile, calibration, portrait_matcher):
        self.profile = profile
        self.matcher = portrait_matcher
        with Image.open(calibration / profile['sword_template']) as image:
            self.sword = white_mask(image)
        self.objectives = {}
        for name, filenames in profile['objective_templates'].items():
            self.objectives[name] = []
            for path in filenames:
                with Image.open(calibration / path) as image:
                    self.objectives[name].append(white_mask(image))

    def positions(self, image):
        import cv2

        x, top, right, bottom = self.profile['sword_search']
        probe = white_mask(image.crop((x, top, right, bottom)))
        correlation = cv2.matchTemplate(probe, self.sword, cv2.TM_CCOEFF_NORMED)
        scores = correlation.max(axis=1)
        candidates = []
        for i in np.argsort(-scores):
            if scores[i] < self.profile['sword_score']:
                break
            y = int(i) + top
            if all(abs(y - prior) > 21 for prior in candidates):
                candidates.append(y)
        return sorted(candidates)

    def objective(self, image):
        import cv2

        probe = np.pad(white_mask(image), 3)
        if probe.sum() < 30:
            return None
        scores = sorted(((max(float(cv2.matchTemplate(probe, template, cv2.TM_CCOEFF_NORMED).max())
                              for template in refs), name) for name, refs in self.objectives.items()),
                        reverse=True)
        if not scores or scores[0][0] < self.profile['objective_score']:
            return None
        if len(scores) > 1 and scores[0][0] - scores[1][0] < .08:
            return None
        return scores[0][1]

    def portrait(self, crop):
        read = self.matcher.read(crop, {})
        confidence = read['reader']['score']
        margin = read['reader']['margin']
        return {'champion': read['value'] if confidence >= self.profile['portrait_score']
                and margin >= self.profile['portrait_margin'] else None,
                'score': confidence, 'margin': margin, 'candidates': read['candidates']}

    @lru_cache(maxsize=4096)
    def portrait_bytes(self, size, pixels):
        return self.portrait(Image.frombytes('RGB', size, pixels))

    def read(self, image):
        entries = []
        for y in self.positions(image):
            edge = np.asarray(image.crop(pil_box(self.profile['side_box'], y)), dtype=float).mean(axis=(0, 1))
            r, g, b = edge
            minimum = self.profile['side_margin']
            side = 'red' if r - max(b, g) >= minimum else 'blue' if b - r >= minimum else None
            if side is None:
                continue
            roles, parts = {}, []
            for role in ('killer', 'victim'):
                crop = image.crop(pil_box(self.profile[role], y))
                parts.append(np.asarray(crop.resize((16, 16), Image.Resampling.BILINEAR), dtype=np.int16))
                roles[role] = self.portrait_bytes(crop.size, crop.tobytes())
            objective = self.objective(crop)
            if objective == 'rift_scuttler':
                continue
            kind = ('objective:' + objective if objective else 'kill' if roles['victim']['champion']
                    else 'unresolved')
            box = pil_box(self.profile['row_box'], y)
            entries.append(Entry(y, kind, side, box, roles, np.asarray(parts)))
        return entries


class ScoreboardReader:
    def __init__(self, profile, store, root, catalog):
        self.profile = profile
        self.reader = InventoryReader(store, root, catalog, profile['inventory_settings'])
        self.numeric = NumericReader(preprocess=grayscale, alternate=True)
        self.cs_numeric = NumericReader(preprocess=numeric_preprocess)

    def present(self, image):
        left, _, right, _ = self.profile['panel_box']
        ys = self.profile['separators']
        luminance = gray_box(image, (left + 30, ys[0] - 2, right - 30, ys[-1] + 3)).astype(float)
        ratios = []
        for y in ys:
            j = y - ys[0] + 2
            contrast = np.abs(np.diff(luminance[j - 2:j + 3], axis=0))
            ratios.append((contrast > 17).mean(axis=1).max())
        return sum(r >= self.profile['separator_ratio'] for r in ratios) >= len(ys) - 1

    @lru_cache(maxsize=4096)
    def text(self, field, size, pixels):
        crop = Image.frombytes('RGB', size, pixels)
        with TemporaryDirectory(prefix='broadcast-match-ocr-') as directory:
            root = Path(directory)
            if field == 'name':
                path = root / 'name.png'
                grayscale(crop).save(path)
                result = tesseract(path, alternate=True,
                                   whitelist='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 ')
                text = result['raw_text'].strip()
                value = text if result['score'] >= self.profile['name_confidence'] else None
                return {'value': value, 'raw_text': text, 'confidence': result['score'],
                        'status': 'accepted' if value else 'unreadable'}
            parser = {'cs': 'integer', 'level': 'level', 'kda': 'kda'}[field]
            reader = self.cs_numeric if field == 'cs' else self.numeric
            result = reader.read(crop, {'reader': 'number', 'parser': parser}, root)
            raw = result.get('raw_text')
            confidence = result['attempts'][0].get('score', 0) if result['attempts'] else 0
            value = result.get('value')
            return {'value': value, 'raw_text': raw, 'confidence': confidence,
                    'status': 'accepted' if result['acceptance'] == 'accepted' else 'unreadable'}

    @lru_cache(maxsize=8192)
    def icon(self, field, size, pixels):
        crop = Image.frombytes('RGB', size, pixels)
        kind = 'champion' if field == 'champion' else 'item'
        result = self.reader.match(crop, {'kind': kind, 'route': 'scoreboard', 'field': f'inventory.{field}'})
        status = {'accepted': 'accepted', 'absent': 'empty'}.get(result['acceptance'], 'unreadable')
        return {'value': result.get('value') if status == 'accepted' else None, 'status': status,
                'confidence': result.get('reader', {}).get('score', 0), 'candidates': result.get('candidates', []),
                'reason': result.get('reason')}

    def row(self, image, spec):
        side, row = spec
        fields = self.profile['columns'][side]
        reads = {}
        for field, base_box in fields.items():
            box = pil_box(base_box, round(row * self.profile['row_step']))
            crop = image.crop(box)
            if field in ('name', 'cs', 'kda', 'level'):
                read = self.text(field, crop.size, crop.tobytes())
            else:
                read = self.icon(field, crop.size, crop.tobytes())
            reads[field] = {**read, 'box': box}
        return side, row, reads


def center_notices(image, box):
    """Read tan notification glyphs without resizing the entire broadcast strip."""
    crop = image.crop(tuple(box))
    red, green, blue = np.asarray(crop, dtype=np.int16).transpose(2, 0, 1)
    gold = ((red >= 163) & (green >= 135) & (red - green >= 6) & (red - green <= 49)
            & (green - blue >= 12) & (green - blue <= 76))
    if np.count_nonzero(gold) < 120:
        return
    masks = (np.asarray(crop.convert('L')) > 175, gold)
    with TemporaryDirectory(prefix='broadcast-match-notice-') as directory:
        path = Path(directory) / 'notice.png'
        for mask in masks:
            glyphs = Image.fromarray(np.where(mask, 0, 255).astype('uint8'))
            enlarged = glyphs.resize((glyphs.width * 3, glyphs.height * 3), Image.Resampling.LANCZOS)
            ImageOps.expand(enlarged, border=12, fill=255).save(path)
            yield tesseract(path)['raw_text']


def visual_signature(crop):
    return np.asarray(crop.convert('RGB').resize((12, 12), Image.Resampling.BILINEAR), dtype=np.int16)
