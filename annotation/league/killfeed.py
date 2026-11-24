"""Opt-in CPU killfeed intake and source-time announcement tracking."""
from contextlib import suppress
from functools import lru_cache
from itertools import groupby
from pathlib import Path
import math
import os
import shutil
import tempfile

from .core import atomic_json, digest, finite, implementation_hash, read_json, read_rows, uid, write_rows
from .intake_profile import check_box, layout_matches, load_profile
from .inventory_references import load_inventory_references, reference_root
from .killfeed_roster import OPPOSITE, fill_roster, load_scoreboard
from .media import file_hash
from .numeric_reader import NumericReader
from .playback import PlaybackTracker
from .scene_reference import sample_source

FPS = 4
KIND = "kill_announcement"
SCENE_BOX = (240, 75, 1660, 860)
PROFILE = Path(__file__).with_name("profiles") / "midbeast-killfeed-1080p-v1.json"
TIMELINE = "killfeed_timeline"
ROSTER = "killfeed_roster_timeline"
READINGS = "killfeed_readings"
ARTIFACTS = (TIMELINE, ROSTER, READINGS)


def read_clock_groups(frames, start, field, temp):
    def agrees(earlier, later):
        return (earlier and later and earlier[1] is not None and later[1] is not None
                and 0 < later[0] - earlier[0] <= 1.5 and 0 <= later[1] - earlier[1] <= 2)

    reader = NumericReader()
    previous = None
    prior = None
    for _, batch in groupby(frames, key=lambda sample: math.floor(sample[0] - start + 1e-6)):
        batch = list(batch)
        pts, frame = batch[0]
        value = reader.read(frame.crop(field["box"]), field, temp)
        current = (pts, value["value"]["seconds"]) if value["acceptance"] == "accepted" else (pts, None)
        if previous:
            samples, clock = previous
            # Adjacent reads guard against a lone OCR error being mistaken for a cut or rewind.
            verified = agrees(prior, clock) or agrees(clock, current)
            yield from ((t, image, clock if verified else None) for t, image in samples)
            prior = clock
        previous = (batch, current)
    if previous:
        samples, clock = previous
        verified = agrees(prior, clock)
        yield from ((t, image, clock if verified else None) for t, image in samples)


class FeedPlayback:
    def __init__(self, source):
        self.source = source
        self.tracker = PlaybackTracker(source, FPS)
        self.segment = None
        self.high_clock = -1

    def update(self, time, clock, image, valid):
        seconds = clock[1] if clock else None
        playback = self.tracker.update(time, seconds, image.crop(SCENE_BOX), valid)
        # Preserve visually continuous entries through temporary clock OCR gaps.
        # Those samples cannot establish an onset bracket or confirm a roster role.
        boundary = set(playback["discontinuity_reasons"]) - {"unverified_clock_continuity"}
        if self.segment is None or boundary:
            self.segment = uid("killfeed-playback", self.source, time, sorted(boundary))
        replay = bool(valid and clock and seconds < self.high_clock)
        if valid and clock:
            self.high_clock = max(self.high_clock, seconds)
        return {"playback_segment_id": self.segment, "replay": replay,
                "continuity_verified": bool(valid and clock and not playback["discontinuity_reasons"]
                                            and playback["playback_state"] == "playing" and not replay),
                "clock": {"seconds": seconds, "observed_at": clock[0]} if valid and clock else None}


def fit_tracks(previous, entries):
    """Match the ordered stack one-to-one while allowing insertions and expirations."""
    import numpy as np

    costs = {}
    for a, old in enumerate(previous):
        for b, new in enumerate(entries):
            if old.side != new.side:
                continue
            error = np.abs(old.signature - new.signature).mean(axis=(1, 2, 3)) / 255
            if max(error) <= .13 or (min(error) <= .055 and max(error) <= .24):
                costs[a, b] = float(error.mean()) + abs(old.top - new.top) / 6000

    @lru_cache(None)
    def best(a, b):
        if a >= len(previous) or b >= len(entries):
            return 0, ()
        options = [best(a + 1, b), best(a, b + 1)]
        if (a, b) in costs:
            score, pairs = best(a + 1, b + 1)
            options.append((score + 1 - costs[a, b], ((a, b), *pairs)))
        return max(options)

    return {b: a for a, b in best(0, 0)[1]}


def summarize_role(track, name):
    side = track[0]["killer_side"]
    if name == "victim":
        side = OPPOSITE.get(side)
    known = {row["roles"][name]["champion"] for row in track
             if row["classification"] == "candidate" and row["roles"][name]["champion"]}
    champion = next(iter(known)) if len(known) == 1 else None
    confirmation = None
    if champion and side:
        for first, second in zip(track, track[1:]):
            if (first["clock"] and not first["replay"] and second["continuity_verified"]
                    and all(row["classification"] == "candidate" and row["roles"][name]["champion"] == champion
                            for row in (first, second))):
                confirmation = {"observed_at": first["time"], "confirmed_at": second["time"],
                                "evidence_ids": [first["id"], second["id"]]}
                break
    score = max((row["roles"][name]["score"] for row in track if champion
                 and row["roles"][name]["champion"] == champion), default=0.)
    return {"side": side, "champion": champion, "observed_champion": champion,
            "status": "observed" if champion else "ambiguous" if known else "unknown",
            "confirmation": confirmation, "score": score}


def announcements(tracks, provenance):
    result = []
    for track, predecessor in tracks:
        # A shifting row can produce one incidental objective match. Require
        # repeated reads of both portraits to override an objective template.
        has_objective = any(row["classification"] == "objective" for row in track)
        champion_reads = sum(row["classification"] == "candidate" and
                             all(role["champion"] for role in row["roles"].values()) for row in track)
        if has_objective and champion_reads < 2:
            continue
        roles = {name: summarize_role(track, name) for name in ("killer", "victim")}
        if not any(role["champion"] for role in roles.values()):
            continue
        first, last = track[0], track[-1]
        onset = [predecessor["time"], first["time"]] if predecessor else None
        # A focus window may start after the row appeared. Its visible span still
        # belongs on the timeline, even when the announcement onset is unknown.
        result.append({"id": uid("kill-announcement", provenance["source_sha256"], first["id"]),
                       "kind": KIND, "actor": roles["killer"]["champion"], "target": roles["victim"]["champion"],
                       "object": None, "location": None, "start": first["time"], "end": last["time"],
                       "first_frame_id": first["frame_id"], "last_frame_id": last["frame_id"],
                       "first_box": first["box"],
                       "playback_segment_id": first["playback_segment_id"],
                       "game_clock": next((row["clock"] for row in track if row["clock"]), None),
                       "onset_bracket": onset, "onset_evidence_id": predecessor["id"] if predecessor else None,
                       "precision": "sample_bounded" if onset else "onset_unknown",
                       "replay": any(row["replay"] for row in track), "roles": roles, "sampling_fps": FPS,
                       "confidence": max(role["score"] for role in roles.values()),
                       "evidence_ids": [row["id"] for row in track], "routes": ["killfeed"], "task_ids": [],
                       "criteria": ["feed_announcement_visible"], "review_status": "unreviewed",
                       "evidence_modality": "visual", "experimental": True, **provenance})
    return sorted(result, key=lambda item: (item["start"], item["id"]))


class FeedTrackWriter:
    def __init__(self, directory, source, box, reader):
        self.directory, self.source, self.box = directory, source, box
        self.reader = reader
        self.rows, self.tracks = [], []
        self.previous = []
        self.earlier_tracks = []
        self.previous_sample = None
        self.earlier_sample = None

    def evidence(self, time, image, box, playback, **fields):
        frame = uid("source", self.source, time)
        identifier = uid("killfeed-crop", frame, box)
        path = self.directory / "crops" / f"{identifier}.png"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".writing")
            image.save(temporary, format="PNG")
            temporary.replace(path)
        row = {"id": identifier, "frame_id": frame, "time": time, "box": box,
               "path": str(path.relative_to(self.directory.parents[1])), **playback, **fields}
        self.rows.append(row)
        return row

    def update(self, time, image, entries, playback, valid):
        previous = self.previous_sample
        connected = bool(valid and previous and previous[2]["playback_segment_id"] == playback["playback_segment_id"])
        previous_entries = [entry for entry, _ in self.previous]
        matches = fit_tracks(previous_entries, entries) if connected else {}
        feed = image.crop(self.box) if valid else None
        current = []
        predecessors = {}
        visible_tracks = {id(track) for _, track in self.previous}
        missing = [(entry, track) for entry, track in self.earlier_tracks if id(track) not in visible_tracks]
        for index, entry in enumerate(entries):
            reading = self.evidence(time, image.crop(entry.box), entry.box, playback,
                                    classification=entry.kind, roles=entry.roles, killer_side=entry.side)
            if index in matches:
                track = self.previous[matches[index]][1]
            else:
                track = []
                absent = previous if connected and playback["continuity_verified"] else None
                if connected and (transition := self.reader.incoming(entry, feed, previous[1], previous_entries)):
                    box, crop, roles = transition
                    old_time, _, old_playback = previous
                    track.append(self.evidence(old_time, crop, box, old_playback,
                                               classification="incoming", roles=roles, killer_side=entry.side))
                    bridge = fit_tracks([old for old, _ in missing], [entry])
                    if 0 in bridge:
                        _, existing = missing.pop(bridge[0])
                        existing.extend([*track, reading])
                        current.append((entry, existing))
                        continue
                    earlier = self.earlier_sample
                    absent = (earlier if earlier and old_playback["continuity_verified"]
                              and earlier[2]["playback_segment_id"] == old_playback["playback_segment_id"] else None)
                predecessor = None
                if absent:
                    old_time, old_image, old_playback = absent
                    if old_time not in predecessors:
                        predecessors[old_time] = self.evidence(old_time, old_image, self.box, old_playback,
                                                              classification="preceding_feed")
                    predecessor = predecessors[old_time]
                self.tracks.append((track, predecessor))
            track.append(reading)
            current.append((entry, track))
        self.earlier_tracks = self.previous if connected else []
        self.previous = current
        self.earlier_sample = previous if connected else None
        self.previous_sample = (time, feed, playback) if valid else None


def active_snapshot(store):
    path = store.path("killfeed/active.json")
    if not path.exists():
        return None
    active = read_json(path)
    recipe = active["recipe"]
    if active["revision"] != digest(recipe)[:20]:
        raise ValueError("Invalid killfeed snapshot revision")
    manifest = read_json(store.path("manifest.json"))
    start, end = recipe["interval"]
    if (recipe["source_sha256"] != manifest["source"]["sha256"]
            or not manifest["interval"][0] <= start < end <= manifest["interval"][1]):
        raise ValueError("Active killfeed snapshot belongs to a different source or interval")
    return active


def snapshot_dir(store, active):
    return store.path(f'killfeed/{active["revision"]}')


def snapshot_rows(store, name, active=None):
    active = active or active_snapshot(store)
    if active is None:
        return []
    if name not in ARTIFACTS:
        raise ValueError("Unknown killfeed artifact")
    path = snapshot_dir(store, active) / (name + ".jsonl")
    if not path.is_file() or file_hash(path) != active["artifacts"][name]:
        raise ValueError("Incomplete or changed killfeed artifact. Rerun killfeed-intake.")
    return read_rows(path)


def preserve_scoreboard(store, directory, root, scoreboard):
    evidence = []
    for side, members in scoreboard.members.items():
        for champion, witness in members.items():
            path = (root / witness["crop"]).resolve()
            if (not path.is_relative_to(root.resolve()) or path.suffix != ".png"
                    or not path.is_file() or file_hash(path) != witness["crop_sha256"]):
                raise ValueError("Scoreboard portrait evidence changed. Regenerate the scoreboard skim.")
            copy = directory / "scoreboard" / f'{witness["crop_sha256"]}.png'
            copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, copy)
            evidence.append({"id": witness["id"], "time": witness["time"], "side": side,
                             "champion": champion, "classification": "scoreboard",
                             "path": str(copy.relative_to(store.root))})
    return evidence


def run(store, cache, scoreboard_root, start=None, end=None):
    from .killfeed_reader import KillfeedReader

    if not store.path("manifest.json").is_file():
        raise ValueError("Run prepare before killfeed-intake")
    manifest = read_json(store.path("manifest.json"))
    source = manifest["source"]
    limits = manifest["interval"]
    start = limits[0] if start is None else start
    end = limits[1] if end is None else end
    if not (finite(start) and finite(end) and limits[0] <= start < end <= limits[1]):
        raise ValueError("Killfeed interval lies outside the prepared run")
    profile = read_json(PROFILE)
    layout = load_profile(profile["layout_profile"])
    if (source["sha256"] != profile["source_sha256"] or source["sha256"] != layout["calibration_source_sha256"]
            or [source["width"], source["height"]] != layout["size"] or source["fps"] < FPS):
        raise ValueError("Killfeed intake requires the calibrated Midbeast 1080p source")
    video = Path(source["path"])
    if file_hash(video) != source["sha256"]:
        raise ValueError("Source video changed; prepare the run again")
    check_box(profile["feed_box"], layout["size"])
    clock_field, = (field for field in layout["fields"] if field["field"] == "game_clock")

    catalog_root = reference_root(cache, profile["reference_version"])
    catalog = load_inventory_references(catalog_root, profile["reference_version"])
    portraits = [asset for asset in catalog["assets"] if asset["kind"] == "champion"]
    scoreboard = load_scoreboard(scoreboard_root, source["sha256"], [start, end], {asset["id"] for asset in portraits})
    provenance = {"source_sha256": source["sha256"], "profile_hash": digest([profile, layout]),
                  "catalog_hash": digest(portraits), "scoreboard_hash": scoreboard.fingerprint}
    recipe = {**provenance, "interval": [start, end], "sampling_fps": FPS,
              "implementation": implementation_hash(["killfeed.py", "killfeed_reader.py", "killfeed_roster.py",
                                                     "playback.py", "scene_reference.py", "intake_profile.py",
                                                     "numeric_reader.py", "inventory_reader.py", "readers.py"])}
    active = {"schema_version": 1, "revision": digest(recipe)[:20], "recipe": recipe}
    directory = snapshot_dir(store, active)
    report_path = directory / "report.json"
    if report_path.is_file():
        saved = read_json(report_path)
        if saved["snapshot"]["recipe"] == recipe:
            with suppress(ValueError):
                for name in ARTIFACTS:
                    snapshot_rows(store, name, saved["snapshot"])
                atomic_json(store.path("killfeed/active.json"), saved["snapshot"])
                return saved

    if shutil.which(os.environ.get("LEAGUE_TESSERACT", "tesseract")) is None:
        raise ValueError("Killfeed intake needs Tesseract to verify playback continuity")
    reader = KillfeedReader(profile, catalog_root, portraits)
    writer = FeedTrackWriter(directory, source["sha256"], profile["feed_box"], reader)
    playback = FeedPlayback(source["sha256"])
    frames = sample_source(video, start, end, FPS, frame_threads=4)
    count = 0
    with tempfile.TemporaryDirectory(prefix="league-killfeed-clock-") as scratch:
        for time, image, clock in read_clock_groups(frames, start, clock_field, Path(scratch)):
            store.check_stop()
            valid, _ = layout_matches(image, layout)
            state = playback.update(time, clock, image, valid)
            writer.update(time, image, reader.read(image) if valid else [], state, valid)
            count += 1
    if not count:
        raise ValueError("No native source samples in the killfeed interval")
    events = announcements(writer.tracks, provenance)
    rows, conflicts = fill_roster(events, scoreboard, [start, end], provenance)
    scoreboard_evidence = preserve_scoreboard(store, directory, scoreboard_root, scoreboard)
    for conflict in conflicts:
        if ids := conflict.get("scoreboard_evidence_ids"):
            event = next(event for event in events if event["id"] == conflict["event_id"])
            event["roles"][conflict["role"]]["scoreboard_evidence_ids"] = ids
    readings = list({r["id"]: r for r in writer.rows + scoreboard_evidence}.values())
    for name, entries in zip(ARTIFACTS, (events, rows, readings)):
        write_rows(directory / (name + ".jsonl"), entries)
    active["artifacts"] = {name: file_hash(directory / (name + ".jsonl")) for name in ARTIFACTS}
    report = {"status": "completed_experimental", "snapshot": active, "path": str(directory), "frames": count,
              "announcements": len(events), "objective_rows": sum(r.get("classification") == "objective" for r in writer.rows),
              "provisional_roster_rows": sum(row["source"] == "killfeed" for row in rows),
              "scoreboard_disputed": scoreboard.disputed, "conflicts": conflicts}
    atomic_json(report_path, report)
    atomic_json(store.path("killfeed/active.json"), active)
    return report
