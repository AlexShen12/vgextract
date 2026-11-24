"""Export source-paced Midbeast stills and their annotation provenance."""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import fcntl
from fractions import Fraction
import json
import math
from pathlib import Path
import shutil
import tempfile

import av
from PIL import Image, ImageDraw, ImageFont

from .core import atomic_json, read_json, read_rows
from .luna_evidence import cast_bar_onsets
from .media import file_hash

GAME_SIZE = (1280, 720)
CANVAS_SIZE = (1856, 944)
WINDOW = 15
JPEG_QUALITY = 80
PINNED_SOURCE = "f4f36e1a274617ecbe224eaf7fc3f8375f757acb7a517bcc0ee7780fcd83f240"
PASS1_JOB = "10064570"
TABLES = (
    "states", "events", "ability_slot_timeline", "cast_bar_readings", "playback_timeline",
    "selection_timeline", "commentary_timeline", "inventory_timeline", "inventory_history",
    "inventory_snapshots", "scoreboard_visibility", "evidence_frames", "hud_activations",
    "cast_timeline", "roster_timeline",
)
TAIL_SUMMARY = (
    "The gameplay footage has ended. A Midbeast end card with Twitch and Twitter handles "
    "fills the screen, then fades to black."
)
HUD_FIELDS = {
    "selected": ("health", "health_fraction", "resource", "resource_fraction",
                 "level", "cs", "kda", "displayed_gold"),
    "match": ("game_clock", "objective_timer", "notice"),
    "team:blue": ("kills",),
    "team:red": ("kills",),
}
INVENTORY_SLOTS = ("0", "1", "2", "3", "4", "5", "trinket")
ABILITY_SLOTS = ("Q", "W", "E", "R")
PASS1_REVISION = "96290645110fddfdd1ca"


@dataclass
class SourceFrames:
    time_base: Fraction
    pts: list[int]
    end: int

    @classmethod
    def probe(cls, video):
        with av.open(str(video)) as container:
            stream = container.streams.video[0]
            packets = sorted((p.pts, p.duration) for p in container.demux(stream) if p.pts is not None)
            pts = [start for start, _ in packets]
            end = stream.start_time + stream.duration
            if (len(pts) != 103858 or len(pts) != stream.frames or pts[0] != 0
                    or any(duration <= 0 or start + duration != next_start
                           for (start, duration), next_start in zip(packets, [*pts[1:], end]))):
                raise ValueError("The pinned source does not have a contiguous native frame timeline")
            return cls(stream.time_base, pts, end)

    def tick(self, seconds, *, sampled=False):
        position = seconds / float(self.time_base)
        if sampled:
            # Some Pass 1 sampled times are rounded to four decimal places.
            tolerance = .000051 / float(self.time_base)
            i = bisect_left(self.pts, position - tolerance)
            if i < len(self.pts) and abs(self.pts[i] - position) <= tolerance:
                return self.pts[i]
        return math.ceil(position - 1e-7)

    def ranges(self):
        for start in range(0, math.ceil(self.end * self.time_base), WINDOW):
            first = bisect_left(self.pts, self.tick(start))
            stop = bisect_left(self.pts, self.tick(start + WINDOW))
            if first < stop:
                yield Window(start, first, stop)


@dataclass(frozen=True)
class Window:
    start: int
    first: int
    stop: int

    @property
    def name(self):
        return f"{self.start:04d}-{self.start + WINDOW:04d}" if self.start < 1725 else "1725-end"


class Track:
    """Point samples and intervals sharing a single timeline key."""

    def __init__(self, entries):
        self.entries = sorted(entries, key=lambda row: row[0])
        self.starts = [start for start, _, _ in self.entries]

    def at(self, tick):
        i = bisect_right(self.starts, tick) - 1
        if i >= 0 and tick < self.entries[i][1]:
            return self.entries[i][2]
        return {}


class Active:
    """Materialize the active interval set at its boundaries."""

    def __init__(self, entries):
        changes = defaultdict(lambda: ([], []))
        for start, end, row in entries:
            if start < end:
                changes[start][1].append(row)
                changes[end][0].append(row["id"])
        self.starts = sorted(changes)
        self.values = []
        active = {}
        for tick in self.starts:
            remove, add = changes[tick]
            for key in remove:
                active.pop(key, None)
            active.update((row["id"], row) for row in add)
            self.values.append(tuple(active.values()))

    def at(self, tick):
        i = bisect_right(self.starts, tick) - 1
        return self.values[i] if i >= 0 else ()


@dataclass(frozen=True)
class Overlay:
    clock: str
    summary_label: str
    summary: str
    summary_window: str
    hud: tuple[str, ...]
    inventory: tuple[str, ...]
    actions: tuple[str, ...]
    commentary: tuple[str, ...]


def clock(seconds):
    whole = int(seconds)
    return f"{whole // 60:02d}:{whole % 60:02d}"


def display(row):
    value = row.get("value") if row.get("status") == "confirmed" else None
    if value is None:
        return "Unknown"
    if isinstance(value, dict):
        if row["field"] == "kda":
            return " / ".join(str(value[k]) for k in ("kills", "deaths", "assists"))
        return value.get("displayed_text") or str(value.get("current", value.get("amount", "Unknown")))
    return str(value)


def meter(numeric, fraction):
    if numeric != "Unknown":
        return numeric
    try:
        return f"Unknown ({float(fraction):.0%} bar)"
    except (ValueError, TypeError):
        return numeric


def read_inputs(bundle, video, annotations, catalog):
    manifest = read_json(bundle / "pass1_manifest.json")
    rev = manifest["final_revision"]
    if manifest["source_sha256"] != PINNED_SOURCE or file_hash(video) != PINNED_SOURCE:
        raise ValueError("The source video does not match the published Midbeast source")
    if rev != PASS1_REVISION:
        raise ValueError(f"Expected published Pass 1 revision {PASS1_REVISION}, found {rev}")
    revision = bundle / "revisions" / rev
    ledger = read_json(revision / "checksums.json")["files"]
    if not read_json(revision / "verification.json")["passed"]:
        raise ValueError("Pass 1 verification did not pass")
    files = sorted(revision.glob("*.jsonl"))
    if not set(TABLES) <= {path.stem for path in files}:
        raise ValueError("Required Pass 1 render tables are missing")
    for path in files:
        if file_hash(path) != ledger[str(path.relative_to(bundle))]:
            raise ValueError(f"Pass 1 checksum mismatch for {path.name}")
    catalog_hash = file_hash(catalog)
    item_names = {str(row["id"]): row["name"] for row in read_json(catalog)["items"]}
    luna = read_rows(annotations)
    if [(r["start"], r["end"]) for r in luna] != [(t, t + WINDOW) for t in range(0, 1725, WINDOW)]:
        raise ValueError("Expected the 115 original LUNA windows in source order")
    progress = read_json(annotations.with_name("progress.json"))
    if progress["completed"] != len(luna):
        raise ValueError("The selected LUNA run is incomplete")
    provenance = {
        "source_sha256": PINNED_SOURCE,
        "pass1_revision": rev,
        "pass1_assembly_job": PASS1_JOB,
        "pass1_checksums_sha256": file_hash(revision / "checksums.json"),
        "luna_run": annotations.parent.name,
        "luna_sha256": file_hash(annotations),
        "luna_input_pass1_revision": progress["local_pass1_revision"],
        "item_catalog_sha256": catalog_hash,
    }
    return revision, luna, item_names, provenance


class Timeline:
    def __init__(self, frames, revision, summaries, item_names):
        self.frames = frames
        self.item_names = item_names
        self.playback = self.intervals(read_rows(revision / "playback_timeline.jsonl"), sampled=True)
        self.selection = self.intervals(read_rows(revision / "selection_timeline.jsonl"), sampled=True)
        self.captions = self.intervals(read_rows(revision / "commentary_timeline.jsonl"), overlapping=True)
        self.summaries = self.intervals([
            *[{**r, "id": f"luna-{r['start']:04d}", "label": "LUNA DRAFT"} for r in summaries],
            {"start": 1725, "end": float(frames.end * frames.time_base),
             "id": "source-reviewed-outro", "label": "SOURCE-CHECKED SUMMARY", "annotation": TAIL_SUMMARY},
        ])
        self.states = self.samples(
            (r for r in read_rows(revision / "states.jsonl")
             if r["field"] in HUD_FIELDS.get(r["slot_id"], ())),
            lambda r: (r["slot_id"], r["field"]), 1.05,
        )
        self.slots = self.samples(read_rows(revision / "ability_slot_timeline.jsonl"),
                                  lambda r: r["slot_id"], .085)
        self.scoreboard = self.samples(read_rows(revision / "scoreboard_visibility.jsonl"),
                                       lambda r: "visibility", .27)["visibility"]
        self.inventory, self.last_seen = self.inventory_tracks(revision)
        readings = read_rows(revision / "cast_bar_readings.jsonl")
        self.actions, self.action_bounds, self.first_activation = self.event_tracks(
            read_rows(revision / "events.jsonl"), readings,
        )

    def intervals(self, rows, *, sampled=False, overlapping=False):
        indexed = [(self.frames.tick(r["start"], sampled=sampled),
                    self.frames.tick(r["end"], sampled=sampled), r) for r in rows]
        return (Active if overlapping else Track)(indexed)

    def samples(self, rows, key, lifetime):
        by_key = defaultdict(list)
        for row in rows:
            if row.get("review_status") == "rejected":
                continue
            start = self.frames.tick(row["time"], sampled=True)
            end = self.frames.tick(row["time"] + lifetime)
            by_key[key(row)].append((start, end, row))
        return {k: Track(entries) for k, entries in by_key.items()}

    def scope(self, tick):
        playback = self.playback.at(tick)
        selection = self.selection.at(tick)
        return playback.get("playback_segment_id"), selection.get("owner_epoch")

    def inventory_tracks(self, revision):
        snapshots = (r for r in read_rows(revision / "inventory_snapshots.jsonl")
                     if r["route"] == "selected_hud")
        rows = []
        for snapshot in snapshots:
            segment, epoch = self.scope(self.frames.tick(snapshot["time"], sampled=True))
            if epoch and segment == snapshot["playback_segment_id"]:
                rows.extend({**slot, "id": snapshot["id"], "time": snapshot["time"],
                             "evidence_ids": [slot["observation_id"], *snapshot.get("identity_evidence_ids", [])],
                             "champion": snapshot.get("champion"), "scope": (segment, epoch)}
                            for slot in snapshot["slots"])
        def key(r):
            return (r["scope"], r["slot"])
        current = self.samples(rows, key, 1.05)
        history = self.samples((r for r in rows if r["status"] in {"recognized", "empty"}),
                               key, float(self.frames.end * self.frames.time_base))
        return current, history

    def event_tracks(self, events, readings):
        onsets = cast_bar_onsets(readings)
        tracks, bounds, activation_starts = [], {}, []
        for row in events:
            if row.get("review_status") == "rejected":
                continue
            start = row["start"]
            kind = row["kind"]
            inclusive_end = True
            if kind == "kill_announcement":
                bracket = row.get("onset_bracket")
                start = bracket[-1] if bracket else start
                actor, target = row.get("actor") or "Unknown", row.get("target") or "Unknown"
                label = f"Kill announced · {actor} → {target}"
                if row.get("replay"):
                    label += " · replay"
            elif kind == "observed_channel":
                bracket = onsets.get(next(iter(row.get("evidence_ids", ())), None))
                start = bracket[-1] if bracket else start
                label = f"{row.get('observed_text') or 'Cast'} bar visible"
            elif kind == "hud_slot_activation" and row.get("onset_frame_ids"):
                start = row["confirmed_readings"][0]["time"] if row.get("confirmed_readings") else row["end"]
                bracket = (row["start"], start)
                label = f"{row['slot_id']} cooldown started"
                inclusive_end = False
            else:
                continue
            if bracket is None:
                label += " · ongoing, onset unknown"
            start_tick = self.frames.tick(start, sampled=True)
            end_tick = (self.frames.tick(row["end"], sampled=True) + 1 if inclusive_end
                        else start_tick + self.frames.tick(1.5))
            segment = row.get("playback_segment_id") or row.get("segment_id")
            epoch = row.get("owner_epoch")
            first = bisect_left(self.frames.pts, start_tick)
            stop = bisect_left(self.frames.pts, end_tick)
            # Stop at the first boundary so a later recurrence of the scope cannot revive a cue.
            for pt in self.frames.pts[first:stop]:
                actual_segment, actual_epoch = self.scope(pt)
                if actual_segment != segment or (epoch and actual_epoch != epoch):
                    end_tick = pt
                    break
            if first >= len(self.frames.pts) or self.frames.pts[first] >= end_tick:
                continue
            if kind == "hud_slot_activation":
                activation_starts.append((start_tick, row["id"]))
            bounds[row["id"]] = (start_tick, end_tick)
            tracks.append((start_tick, end_tick, {**row, "label": label}))
        if not activation_starts:
            raise ValueError("No confirmed HUD activation is available for preview")
        return Active(tracks), bounds, min(activation_starts)

    def at(self, tick):
        ids, evidence = set(), set()

        def cite(row):
            if row:
                ids.add(row["id"])
                for key in ("evidence_ids", "frame_ids", "selection_evidence_ids"):
                    evidence.update(row.get(key, ()))
                for key in ("frame_id", "first_frame_id", "last_frame_id", "onset_evidence_id"):
                    if row.get(key):
                        evidence.add(row[key])
            return row

        play, selected = cite(self.playback.at(tick)), cite(self.selection.at(tick))
        segment, epoch = play.get("playback_segment_id"), selected.get("owner_epoch")
        champion = selected.get("champion")
        summary = cite(self.summaries.at(tick))
        caption_rows = [cite(row) for row in self.captions.at(tick)]
        action_rows = [cite(row) for row in self.actions.at(tick)]

        def state(slot, field):
            track = self.states.get((slot, field))
            row = track.at(tick) if track else {}
            if not row or not segment or row["playback_segment_id"] != segment:
                return "Unknown"
            if slot == "selected":
                if not epoch or self.scope(self.frames.tick(row["time"], sampled=True))[1] != epoch:
                    return "Unknown"
                if row["entity"] not in ("unknown", "selected", champion):
                    return "Unknown"
            return display(cite(row))

        def cooldown(slot):
            row = self.slots[slot].at(tick)
            if not segment or not epoch or row.get("segment_id") != segment or row.get("owner_epoch") != epoch:
                return f"{slot} Unknown"
            cite(row)
            if row["state"] != "cooling_down" or row["seconds"] is None:
                return f"{slot} Unknown"
            return f"{slot} {row['raw'].get('raw_text') or format(row['seconds'], 'g')}s"

        def inventory(slot):
            if not segment or not epoch:
                return "Unknown"
            key = ((segment, epoch), slot)
            current = self.inventory.get(key)
            row = current.at(tick) if current else {}
            if row and row.get("champion") and row["champion"] != champion:
                return "Unknown"
            cite(row)
            carried = row.get("status") not in {"recognized", "empty"}
            if carried:
                last = self.last_seen.get(key)
                row = last.at(tick) if last else {}
                if row.get("champion") and row["champion"] != champion:
                    row = {}
                cite(row)
            if not row:
                return "Unknown"
            name = ("Empty" if row["status"] == "empty"
                    else self.item_names.get(str(row["item_id"]), f"Item {row['item_id']}"))
            return name + (" · last seen" if carried else "")

        def selected_value(field):
            return state("selected", field)

        board = self.scoreboard.at(tick)
        board_status = "Unknown"
        if segment and board.get("playback_segment_id") == segment:
            board_status = {"present": "Open", "absent": "Closed"}.get(cite(board)["status"], "Unknown")
        hud = (
            f"Selected  {champion or 'Unknown'}     Game clock  {state('match', 'game_clock')}",
            f"HP  {meter(selected_value('health'), selected_value('health_fraction'))}     Resource  "
            f"{meter(selected_value('resource'), selected_value('resource_fraction'))}",
            f"Level  {selected_value('level')}     CS  {selected_value('cs')}     Gold  {selected_value('displayed_gold')}",
            f"K/D/A  {selected_value('kda')}     Team kills  {state('team:blue', 'kills')}–{state('team:red', 'kills')}",
            "    ".join(cooldown(slot) for slot in ABILITY_SLOTS),
            f"Scoreboard  {board_status}     Objective timer  {state('match', 'objective_timer')}",
            f"Notice  {state('match', 'notice')}",
        )
        inventory_labels = tuple(
            f"{i + 1 if i < 6 else 'Trinket'}  {inventory(slot)}"
            for i, slot in enumerate(INVENTORY_SLOTS)
        )
        overlay = Overlay(
            clock(tick * self.frames.time_base), summary["label"], summary["annotation"],
            f"{clock(summary['start'])}–{clock(math.ceil(summary['end']))}", hud, inventory_labels,
            tuple(row["label"] for row in action_rows), tuple(row["text"] for row in caption_rows),
        )
        return overlay, sorted(ids), sorted(evidence)


class Renderer:
    BACKGROUND = "#111923"
    FOOTER = "#152130"
    TEXT = "#f0f4f8"
    SECONDARY = "#a8b8c9"
    ACCENT = "#8ad9c9"
    RULE = "#34485d"

    def __init__(self, fonts):
        self.fonts = fonts
        self.loaded = {}
        self.canvas = Image.new("RGB", CANVAS_SIZE, self.BACKGROUND)
        self.previous = None
        self.summary_key = None

    def font(self, size, bold=False):
        key = size, bold
        if key not in self.loaded:
            self.loaded[key] = ImageFont.truetype(str(self.fonts[int(bold)]), size)
        return self.loaded[key]

    def wrap(self, text, width, size, bold=False):
        font = self.font(size, bold)
        lines, line = [], ""
        for word in text.split():
            joined = f"{line} {word}" if line else word
            if font.getlength(joined) > width and line:
                lines.append(line)
                line = word
            else:
                line = joined
        if line:
            lines.append(line)
        return lines

    def line(self, draw, xy, value, width, size, *, color=None, bold=False, minimum=14):
        while size > minimum and self.font(size, bold).getlength(value) > width:
            size -= 1
        font = self.font(size, bold)
        if font.getlength(value) > width:
            raise ValueError(f"Line exceeds available space: {value!r}")
        draw.text(xy, value, font=font, fill=color or self.TEXT)

    def paragraph(self, draw, xy, value, width, height, maximum):
        for size in range(maximum, 15, -1):
            lines = self.wrap(value, width, size)
            leading = size + 5
            if len(lines) * leading <= height:
                break
        else:
            raise ValueError("Annotation paragraph does not fit its panel")
        for i, line in enumerate(lines):
            self.line(draw, (xy[0], xy[1] + i * leading), line, width, size, minimum=size)

    def annotate(self, overlay):
        if overlay == self.previous:
            return
        draw = ImageDraw.Draw(self.canvas)
        summary_key = (overlay.summary_label, overlay.summary, overlay.summary_window)
        if summary_key != self.summary_key:
            draw.rectangle((1280, 0, 1855, 339), fill=self.BACKGROUND)
            self.line(draw, (1306, 25), overlay.summary_label, 305, 16, color=self.ACCENT, bold=True)
            self.line(draw, (1668, 25), overlay.summary_window, 155, 16, color=self.SECONDARY)
            self.paragraph(draw, (1306, 66), overlay.summary, 524, 266, 21)
            self.summary_key = summary_key
        draw.rectangle((1280, 340, 1855, 719), fill=self.BACKGROUND)
        draw.line((1306, 342, 1830, 342), fill=self.RULE)
        self.line(draw, (1306, 357), "HUD READINGS", 300, 16, color=self.ACCENT, bold=True)
        self.line(draw, (1725, 357), f"Source {overlay.clock}", 105, 15, color=self.SECONDARY)
        for i, line in enumerate(overlay.hud):
            self.line(draw, (1306, 390 + i * 25), line, 524, 18)
        draw.line((1306, 569, 1830, 569), fill=self.RULE)
        self.line(draw, (1306, 581), "SELECTED INVENTORY", 300, 15, color=self.ACCENT, bold=True)
        for i, item in enumerate(overlay.inventory):
            self.line(draw, (1306 + 262 * (i // 4), 607 + 22 * (i % 4)), item, 255, 16)
        draw.rectangle((0, 720, 1855, 943), fill=self.FOOTER)
        draw.line((1013, 744, 1013, 918), fill=self.RULE)
        self.line(draw, (26, 744), "RECORDED ACTION CUES", 930, 16, color=self.ACCENT, bold=True)
        self.line(draw, (1040, 744), "COMMENTARY · AUTO CAPTIONS", 780, 16, color=self.ACCENT, bold=True)
        if len(overlay.actions) > 14:
            raise ValueError("Too many simultaneous cues for the footer")
        columns = 2 if len(overlay.actions) > 7 else 1
        for i, cue in enumerate(overlay.actions):
            self.line(draw, (26 + 485 * (i // 7), 778 + 21 * (i % 7)), cue, 960 // columns, 17)
        if overlay.commentary:
            self.paragraph(draw, (1040, 778), "  ".join(overlay.commentary), 785, 145, 23)
        self.previous = overlay

    def write(self, gameplay, overlay, path):
        self.annotate(overlay)
        self.canvas.paste(gameplay, (0, 0))
        self.canvas.save(path, "JPEG", quality=JPEG_QUALITY, subsampling=2)


def font_paths():
    families = (
        ("/System/Library/Fonts/Supplemental/Arial.ttf",
         "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    )
    for regular, bold in families:
        if Path(regular).is_file() and Path(bold).is_file():
            return Path(regular), Path(bold)
    raise FileNotFoundError("Arial or DejaVu Sans regular and bold fonts are required")


def frame_name(index):
    return f"frame-{index + 1:06d}.jpg"


def frame_duration(frames, i):
    return (frames.pts[i + 1] if i + 1 < len(frames.pts) else frames.end) - frames.pts[i]


def validate_window(path, window, frames):
    meta = read_json(path / "window.json")
    if meta["manifest_sha256"] != file_hash(path / "frames.jsonl"):
        raise ValueError(f"{path.name}: frame manifest checksum mismatch")
    if meta["frames"] != window.stop - window.first:
        raise ValueError(f"{path.name}: incorrect frame count in window receipt")
    expected = iter(range(window.first, window.stop))
    count = 0
    with (path / "frames.jsonl").open() as manifest:
        for line in manifest:
            row = json.loads(line)
            i = next(expected, None)
            if i is None or (row["frame_index"], row["image"], row["pts"], row["duration_pts"]) != (
                i, frame_name(i), frames.pts[i], frame_duration(frames, i),
            ):
                raise ValueError(f"{path.name}: timing or frame order mismatch at row {count + 1}")
            if (path / row["image"]).stat().st_size <= 0:
                raise ValueError(f"{path.name}: empty image at row {count + 1}")
            count += 1
    if next(expected, None) is not None or sum(1 for _ in path.glob("*.jpg")) != count:
        raise ValueError(f"{path.name}: missing or extra images")
    for i in (window.first, window.stop - 1):
        with Image.open(path / frame_name(i)) as still:
            if still.size != CANVAS_SIZE or still.format != "JPEG":
                raise ValueError(f"{path.name}: unexpected image format or dimensions")
            still.load()
    return meta


def render_window(video, output, frames, timeline, fonts, window):
    destination = output / window.name
    if destination.exists():
        return validate_window(destination, window, frames)
    renderer = Renderer(fonts)
    start_pts = frames.pts[window.first]
    with tempfile.TemporaryDirectory(prefix=f".{window.name}-", dir=output) as tmp:
        path = Path(tmp)
        image_bytes = 0
        i = window.first
        with av.open(str(video)) as container, (path / "frames.jsonl").open("w") as manifest:
            stream = container.streams.video[0]
            stream.thread_count = 2
            container.seek(start_pts, stream=stream)
            scale = av.video.reformatter.VideoReformatter()
            for frame in container.decode(stream):
                if frame.pts < start_pts:
                    continue
                if frame.pts != frames.pts[i] or frame.time_base != frames.time_base:
                    raise ValueError(f"Decoded PTS mismatch for frame {i} at {frame.pts}")
                if frame.duration != frame_duration(frames, i):
                    raise ValueError(f"Decoded duration mismatch for frame {i}")
                overlay, annotation_ids, evidence_ids = timeline.at(frame.pts)
                image = scale.reformat(frame, width=GAME_SIZE[0], height=GAME_SIZE[1], format="rgb24")
                image_path = path / frame_name(i)
                renderer.write(image.to_image(), overlay, image_path)
                image_bytes += image_path.stat().st_size
                manifest.write(json.dumps({
                    "frame_index": i, "image": image_path.name, "pts": frame.pts,
                    "duration_pts": frame.duration, "annotation_ids": annotation_ids,
                    "evidence_ids": evidence_ids,
                }, separators=(",", ":")) + "\n")
                i += 1
                if i == window.stop:
                    break
        if i != window.stop:
            raise ValueError(f"{window.name}: decoded only {i - window.first} frames")
        meta = {
            "window": window.name, "frames": i - window.first, "first_frame_index": window.first,
            "first_pts": start_pts, "end_pts": frames.pts[i] if i < len(frames.pts) else frames.end,
            "image_bytes": image_bytes, "manifest_sha256": file_hash(path / "frames.jsonl"),
        }
        atomic_json(path / "window.json", meta)
        validate_window(path, window, frames)
        path.rename(destination)
        return meta


def render_report(args):
    revision, summaries, items, inputs = read_inputs(args.bundle, args.video, args.annotations, args.item_catalog)
    frames = SourceFrames.probe(args.video)
    timeline = Timeline(frames, revision, summaries, items)
    fonts = font_paths()
    windows = list(frames.ranges())
    cue_at, cue_id = timeline.first_activation
    preview = next(window for window in windows if frames.pts[window.first] <= cue_at
                   <= frames.pts[window.stop - 1])
    recipe = {
        **inputs, "render_version": 1, "renderer_sha256": file_hash(Path(__file__)),
        "frame_count": len(frames.pts), "source_end_pts": frames.end,
        "time_base": [frames.time_base.numerator, frames.time_base.denominator],
        "first_preview_cue_id": cue_id, "first_preview_cue_pts": cue_at,
        "canvas": list(CANVAS_SIZE), "gameplay": list(GAME_SIZE), "jpeg_quality": JPEG_QUALITY,
        "font_sha256": [file_hash(font) for font in fonts], "tail_summary": TAIL_SUMMARY,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".render.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        recipe_path = args.output / "recipe.json"
        if recipe_path.exists() and read_json(recipe_path) != recipe:
            raise ValueError("Existing sequence uses different inputs or renderer; choose a new output directory")
        if not recipe_path.exists():
            atomic_json(recipe_path, recipe)
        if args.verify:
            receipts = [validate_window(args.output / w.name, w, frames) for w in windows]
        else:
            pilot = render_window(args.video, args.output, frames, timeline, fonts, preview)
            print(f"Preview {preview.name}: {pilot['frames']} frames, "
                  f"{pilot['image_bytes'] / 2**20:.1f} MiB, cue PTS {cue_at} ({cue_id})", flush=True)
            if args.preview:
                return
            receipts = [validate_window(args.output / w.name, w, frames) for w in windows
                        if (args.output / w.name).exists()]
            remaining = [w for w in windows if not (args.output / w.name).exists()]
            frames_left = sum(w.stop - w.first for w in remaining)
            forecast = int(frames_left * pilot["image_bytes"] / pilot["frames"] * 1.2)
            forecast += frames_left * (args.output / preview.name / "frames.jsonl").stat().st_size // pilot["frames"]
            free = shutil.disk_usage(args.output).free
            print(f"Disk preflight: {forecast / 2**30:.2f} GiB for remaining frames and manifests "
                  f"including image headroom, {free / 2**30:.2f} GiB free", flush=True)
            if remaining and free < forecast + 2 * 2**30:
                raise OSError("Insufficient free disk space for the complete image sequence")
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                pending = [pool.submit(render_window, args.video, args.output, frames, timeline, fonts, w)
                           for w in remaining]
                try:
                    for job in as_completed(pending):
                        receipt = job.result()
                        receipts.append(receipt)
                        print(f"{len(receipts)}/{len(windows)} windows ready · {receipt['window']}", flush=True)
                except BaseException:
                    for job in pending:
                        job.cancel()
                    raise
        receipts.sort(key=lambda r: r["first_frame_index"])
        if (len(receipts) != len(windows) or sum(r["frames"] for r in receipts) != len(frames.pts)
                or receipts[0]["first_pts"] != frames.pts[0] or receipts[-1]["end_pts"] != frames.end
                or any(left["end_pts"] != right["first_pts"] for left, right in zip(receipts, receipts[1:]))):
            raise ValueError("The output does not cover the complete source timeline")
        atomic_json(args.output / "index.json", {**recipe, "windows": receipts})
        print(f"Verified {len(frames.pts):,} images across {len(windows)} windows", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True, help="Collected published Longleaf render inputs")
    parser.add_argument("--annotations", type=Path, required=True, help="LUNA annotations.jsonl")
    parser.add_argument("--item-catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preview", action="store_true")
    modes.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    render_report(args)


if __name__ == "__main__":
    main()
