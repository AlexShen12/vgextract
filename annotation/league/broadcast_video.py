"""Shot proposals and broadcast overlay evidence on a shared VOD timeline."""

from bisect import bisect_left
from collections import Counter
import math
import subprocess

from .darius_clip_pilot import DETECTOR_SETTINGS


WIDTH, HEIGHT, FPS, SAMPLE_FPS = 480, 270, 12, 2


def crop(frame, box):
    x0, y0, x1, y1 = box
    return frame[round(y0 * HEIGHT):round(y1 * HEIGHT), round(x0 * WIDTH):round(x1 * WIDTH)]


def read_template_frame(path):
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape != (HEIGHT, WIDTH):
        raise ValueError(f'Template frame must be {WIDTH}x{HEIGHT}: {path}')
    return image


def load_templates(profile, root):
    templates = []
    for rule in profile['templates']:
        patch = crop(read_template_frame(root / rule['image']), rule['box'])
        if min(patch.shape) < 3 or patch.std() < 5:
            raise ValueError(f"Template patch lacks visual detail: {rule['name']}")
        templates.append((rule, patch))
    return templates


def frame_evidence(frame, templates, screenshot):
    import cv2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scores, labels = {}, set()
    for rule, patch in templates:
        # Compare at the reference location with a small tolerance for transcode scaling.
        x0, y0, x1, y1 = rule['box']
        search = crop(gray, [max(0, x0 - .008), max(0, y0 - .012),
                             min(1, x1 + .008), min(1, y1 + .012)])
        score = float(cv2.matchTemplate(search, patch, cv2.TM_CCOEFF_NORMED).max())
        scores[rule['name']] = round(score, 4)
        if score >= rule['threshold']:
            labels.add(rule['label'])
    if 'replay' in labels:
        label = 'replay'
    elif 'live_gameplay' in labels:
        label = 'live_gameplay'
    elif 'draft' in labels:
        label = 'draft'
    elif 'non_gameplay' in labels:
        label = 'non_gameplay'
    else:
        label = 'no_gameplay_hud'
    visible_gameplay_hud = bool(labels & {'live_gameplay', 'gameplay_hud'})
    playfield = crop(gray, [.16, .16, .84, .75])
    reduced = cv2.resize(playfield, (17, 12), interpolation=cv2.INTER_AREA)
    signature = 0
    for bit in (reduced[:, 1:] > reduced[:, :-1]).ravel():
        signature = (signature << 1) | int(bit)
    screenshot_similarity = (round(float(cv2.matchTemplate(gray, screenshot,
                              cv2.TM_CCOEFF_NORMED)[0, 0]), 4) if screenshot is not None else None)
    return {'label': label, 'gameplay_hud_visible': visible_gameplay_hud,
            'replay_gameplay_visible': bool(labels & {'gameplay_hud', 'replay_gameplay'}),
            'screenshot_similarity': screenshot_similarity,
            'playfield_hash': f'{signature:048x}', 'template_scores': scores}


def scan_video(path, start, duration, templates, screenshot=None):
    import numpy as np
    from scenedetect import AdaptiveDetector, ContentDetector, FrameTimecode

    settings = {**DETECTOR_SETTINGS, 'min_scene_len': round(.5 * FPS)}
    detectors = {'adaptive': AdaptiveDetector(**settings),
                 'content': ContentDetector(min_scene_len=settings['min_scene_len'])}
    cuts = {key: [] for key in detectors}
    samples = []
    command = ['ffmpeg', '-v', 'error', '-nostdin', '-xerror', '-i', str(path), '-map', '0:v:0',
               '-an', '-t', str(duration), '-vf', f'fps={FPS},scale={WIDTH}:{HEIGHT}',
               '-pix_fmt', 'bgr24', '-f', 'rawvideo', 'pipe:1']
    n = 0
    with subprocess.Popen(command, stdout=subprocess.PIPE) as proc:
        try:
            size = WIDTH * HEIGHT * 3
            while raw := proc.stdout.read(size):
                if len(raw) != size:
                    raise RuntimeError(f'Incomplete decoded frame in {path}')
                frame = np.frombuffer(raw, dtype=np.uint8).reshape(HEIGHT, WIDTH, 3)
                timecode = FrameTimecode(n, float(FPS))
                for key, detector in detectors.items():
                    cuts[key].extend(round(start + cut.seconds, 6)
                                     for cut in detector.process_frame(timecode, frame))
                if n % (FPS // SAMPLE_FPS) == 0:
                    samples.append({'time': round(start + n / FPS, 6),
                                    **frame_evidence(frame, templates, screenshot)})
                n += 1
        except BaseException:
            proc.kill()
            raise
        if proc.wait() or not n or abs(n - duration * FPS) > FPS:
            raise RuntimeError(f'Incomplete video coverage for {path}: {n} frames at {FPS} fps')
    end = round(start + min(duration, n / FPS), 6)
    for key, detector in detectors.items():
        cuts[key].extend(round(start + cut.seconds, 6)
                         for cut in detector.post_process(FrameTimecode(n - 1, float(FPS))))
        cuts[key] = sorted({cut for cut in cuts[key] if start < cut < end})
    return {'start': start, 'end': end, 'frames': n, 'cuts': cuts,
            'detector_settings': settings, 'fps': FPS, 'sample_fps': SAMPLE_FPS, 'samples': samples}


def segment_video(scan, detector='content'):
    """Keep cut proposals and add sustained overlay changes inside a shot."""
    samples = scan['samples']
    times = [row['time'] for row in samples]
    cuts = scan['cuts'][detector]
    changes = set()
    for i in range(2, len(samples) - 1):
        left, right = samples[i - 1]['label'], samples[i]['label']
        if left == right or left != samples[i - 2]['label'] or right != samples[i + 1]['label']:
            continue
        interval = (times[i - 1], times[i])
        closest = min(cuts, key=lambda cut: abs(cut - times[i]), default=math.inf)
        if not interval[0] <= closest <= interval[1]:
            changes.add(times[i])

    edges = sorted({scan['start'], scan['end'], *cuts, *changes})
    rows = []
    for start, end in zip(edges, edges[1:]):
        support = samples[bisect_left(times, start):bisect_left(times, end)]
        votes = Counter(row['label'] for row in support)
        winner, count = votes.most_common(1)[0] if votes else ('unknown', 0)
        label = winner if count > len(support) / 2 else 'unknown'
        hud_samples = sum(row['gameplay_hud_visible'] for row in support)
        live_candidate = label == 'live_gameplay' and hud_samples > len(support) / 2
        rows.append({'start': start, 'end': end, 'label': label, 'votes': dict(votes),
                     'gameplay_hud_votes': hud_samples, 'live_gameplay_candidate': live_candidate,
                     'boundary': 'shot_cut' if start in cuts else
                     'overlay_change' if start in changes else 'clip_start'})
    return rows


def label_runs(shots):
    runs = []
    for shot in shots:
        if runs and runs[-1]['label'] == shot['label'] and runs[-1]['end'] == shot['start']:
            runs[-1]['end'] = shot['end']
        else:
            runs.append({key: shot[key] for key in ('start', 'end', 'label')})
    return runs


def replay_links(scan, runs):
    """Find same-speed gameplay repeats within the preceding three minutes of a clip."""
    samples = scan['samples']
    labels = [row['label'] for row in samples]
    signatures = [int(row['playfield_hash'], 16) for row in samples]
    times = [row['time'] for row in samples]
    links = []
    for run in runs:
        if run['label'] != 'replay':
            continue
        start, end = (bisect_left(times, run[key]) for key in ('start', 'end'))
        lower = bisect_left(times, run['start'] - 180)
        best = None
        for replay_index in range(start, end - 2):
            current = signatures[replay_index:replay_index + 3]
            if (current[0] ^ current[-1]).bit_count() < 12:
                continue
            for live_index in range(lower, start - 2):
                if not all(label == 'live_gameplay' for label in labels[live_index:live_index + 3]):
                    continue
                previous = signatures[live_index:live_index + 3]
                distances = [(a ^ b).bit_count() for a, b in zip(current, previous)]
                if max(distances) <= 28 and (best is None or sum(distances) < best[0]):
                    best = (sum(distances), live_index, replay_index, distances)
        if best:
            _, live_index, replay_index, distances = best
            links.append({'replay_run_start': run['start'], 'live_sample_start': times[live_index],
                          'replay_sample_start': times[replay_index],
                          'sample_spacing_seconds': 1 / SAMPLE_FPS, 'hamming_distances': distances,
                          'status': 'candidate'})
    return links
