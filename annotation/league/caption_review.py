"""Pilot review, revisioned upstream corrections and independent source labels."""
import fcntl
from pathlib import Path
import re

from .caption_benchmark import ANNOTATION_CATEGORIES, assessments, validate_labels
from .caption_pilot import effective, pilot_lock, regenerate
from .caption_scene import accepted_events
from .action_v3 import load_profile
from .core import STATUSES, atomic_json, finite, now, read_json, read_rows


def pilot_root(store, pilot_id):
    if not re.fullmatch(r"[0-9a-f]{20}", pilot_id):
        raise ValueError("Invalid caption pilot ID")
    root = store.path("experiments/captions/"+pilot_id)
    if not (root/"input-audit.json").exists():
        raise ValueError("Caption pilot not found")
    return root


def correct_pilot(root, table, record_id, changes, status, reviewer):
    allowed = {
        "scene_proposals": {"kind", "start", "end", "actor", "actor_label", "actor_binding", "target", "target_label", "target_binding", "ability", "visual_evidence", "evidence_ids"},
        "normalized_observations": {"value", "visibility", "acceptance"},
        "body_readings": {"binding_visible", "actor_track", "actor_region"},
    }
    if table not in allowed or status not in STATUSES or not isinstance(changes, dict) or set(changes)-allowed[table] or not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("Invalid pilot correction fields/status/reviewer")
    with pilot_lock(root):
        rows = {r["id"]: r for r in effective(root, table)}
        if record_id not in rows:
            raise ValueError("Pilot evidence record not found")
        row = {**rows[record_id], **changes}
        if status != "rejected":
            if table == "scene_proposals":
                if row.get("ability") is not None and row["ability"] not in {"Q", "W", "E", "R"}:
                    raise ValueError("Pilot ability corrections require a grounded Q/W/E/R slot")
                audit = read_json(root/"input-audit.json")
                if not all(finite(row.get(k)) for k in ("start", "end")) or not audit["interval"][0] <= row["start"] <= row["end"] <= audit["interval"][1]:
                    raise ValueError("Correction outside source interval")
                accepted, omitted = accepted_events([row], audit["frames"], effective(root, "body_readings"))
                if not accepted:
                    raise ValueError("Correction lacks grounding: "+str(omitted))
                # Labels are renderer-controlled: review cannot introduce prose.
                if any(k in changes for k in ("actor_label", "target_label")):
                    raise ValueError("Correct entity/binding evidence, not generated labels")
            elif table == "normalized_observations":
                if (row.get("value") is not None and (type(row["value"]) is not int or row["value"] < 0)
                        or row.get("visibility") not in {"visible", "unknown", "occluded", "unreadable", "not_shown"}
                        or row.get("acceptance") not in {"accepted", "unknown", "rejected"}):
                    raise ValueError("Invalid counter evidence correction")
            elif row.get("binding_visible"):
                box = row.get("actor_region")
                if (not isinstance(row.get("actor_track"), str) or not row["actor_track"]
                        or not isinstance(box, list) or len(box) != 4 or not all(finite(v) for v in box)
                        or not 0 <= box[0] < box[2] <= 1920 or not 0 <= box[1] < box[3] <= 1080):
                    raise ValueError("Invalid corrected actor region/track")
        directory = root/"reviews"
        sequence = len(list(directory.glob("*.json")))+1
        correction = {"id": f"caption-review-{sequence:06d}", "table": table, "record_id": record_id,
            "changes": changes, "review_status": status, "reviewer": reviewer, "created_at": now()}
        atomic_json(directory/(f"{sequence:06d}.json"), correction)
        regenerate(root)
        return correction


def assess_caption(root, body):
    fields = ("supported", "actor_correct", "target_correct", "evidence_valid")
    if (not isinstance(body.get("reviewer"), str) or not body["reviewer"].strip()
            or body.get("independent_of_generation") is not True or any(type(body.get(k)) is not bool for k in fields)):
        raise ValueError("Independent identified reviewer and explicit assessment booleans are required")
    if body["supported"] and not all(body[k] for k in ("actor_correct", "target_correct", "evidence_valid")):
        raise ValueError("Supported claims require correct bindings and supporting evidence")
    with pilot_lock(root):
        captions = {c["id"]: c for c in read_rows(root/"captions.jsonl")}
        if body.get("caption_id") not in captions:
            raise ValueError("Caption not found")
        row = {k: body[k] for k in (*fields, "reviewer", "independent_of_generation", "caption_id")}
        row.update(generation_key=read_json(root/"caption_coverage.json")["key"], created_at=now())
        directory = root/"assessments"
        sequence = len(list(directory.glob("*.json")))+1
        atomic_json(directory/f"{sequence:06d}.json", row)
        return row


def install_caption_routes(app, store):
    from fastapi import HTTPException, Request
    from fastapi.responses import HTMLResponse

    def origin_check(request):
        origin = request.headers.get("origin")
        if origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
            raise HTTPException(403, "Review writes must originate from this local app")

    @app.get("/captions", response_class=HTMLResponse)
    def page():
        return Path(__file__).with_name("caption_review.html").read_text()

    @app.get("/caption-labels", response_class=HTMLResponse)
    def label_page():
        return Path(__file__).with_name("caption_labels.html").read_text()

    @app.get("/api/caption-pilots")
    def pilots():
        return [read_json(p) for p in sorted(store.path("experiments/captions").glob("*/report.json"))]

    @app.get("/api/caption-pilots/{pilot_id}")
    def pilot(pilot_id: str):
        try:
            root = pilot_root(store, pilot_id)
            audit = read_json(root/"input-audit.json")
            with pilot_lock(root):
                data = {name: read_rows(root/(name+".jsonl")) for name in ("captions", "caption_clips", "caption_omissions", "events", "states", "normalized_observations", "counter_readings", "resampling")}
                data.update(scene_proposals=effective(root, "scene_proposals"), body_readings=effective(root, "body_readings"),
                            report=read_json(root/"report.json"), frames=audit["frames"], assessments=assessments(root),
                            coverage=read_json(root/"caption_coverage.json"))
            return data
        except (OSError, ValueError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/caption-pilots/{pilot_id}/corrections")
    async def correct(pilot_id: str, request: Request):
        origin_check(request)
        try:
            body = await request.json()
            return correct_pilot(pilot_root(store, pilot_id), body["table"], body["record_id"], body.get("changes", {}), body["review_status"], body["reviewer"])
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/caption-pilots/{pilot_id}/assessments")
    async def assess(pilot_id: str, request: Request):
        origin_check(request)
        try:
            return assess_caption(pilot_root(store, pilot_id), await request.json())
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/caption-labels")
    def labels():
        path = store.path("benchmarks/captions/captions.json")
        if not path.exists():
            raise HTTPException(404, "Initialize experiment --suite captions first")
        return read_json(path)

    @app.post("/api/caption-labels/{clip_id}")
    async def save_labels(clip_id: str, request: Request):
        origin_check(request)
        try:
            body = await request.json()
            if (not isinstance(body.get("labeler"), str) or not body["labeler"].strip()
                    or body.get("independent_of_predictions") is not True or not isinstance(body.get("annotations"), dict)):
                raise ValueError("Independent source-only labeler provenance required")
            directory = store.path("benchmarks/captions")
            with (directory/".labels.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                path = directory/"captions.json"
                data = read_json(path)
                clip = next((c for c in data["clips"] if c["id"] == clip_id), None)
                if clip is None:
                    raise ValueError("Caption neighborhood not found")
                if (directory/"development-freeze.json").exists() and clip["split"] == "development":
                    raise ValueError("Development labels are frozen; create a new benchmark revision")
                allowed = {"start", "end", "repeat_group", "verified", "inspected", "explicit_negative", "negative_annotations", "case_tags",
                           "claims", "bindings", "phases", "playback", "repeats", "contaminated"}
                if set(body["annotations"])-allowed:
                    raise ValueError("Invalid caption label fields")
                before = dict(clip)
                clip.update(body["annotations"], labeler=body["labeler"], independent_of_predictions=True)
                validate_labels(store, data, load_profile())
                if clip["verified"] and (not clip["inspected"] or (not clip["claims"] and not clip["explicit_negative"]) or clip.get("contaminated")):
                    raise ValueError("Verification requires inspection and explicit negatives, without contamination")
                if clip["verified"] and any(not clip[k] and clip.get("negative_annotations", {}).get(k) is not True for k in ANNOTATION_CATEGORIES):
                    raise ValueError("Empty annotation arrays require explicit category negatives")
                sequence = len(list((directory/"revisions").glob("*.json")))+1
                atomic_json(directory/"revisions"/f"{sequence:06d}.json", {"before": before, "after": clip, "created_at": now()})
                atomic_json(path, data)
                return clip
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from exc
