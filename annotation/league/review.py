"""Optional CPU-only review UI with append-only corrections."""
from __future__ import annotations

import fcntl
import secrets
from pathlib import Path

from .core import (EVENTS, STATUSES, Config, Store, atomic_json, finite, now, read_json,
                   validate_observation, validate_question)

EDITABLE = {
    "observations": {"entity", "binding", "field", "value", "visibility", "confidence", "identity_evidence"},
    "events": {"kind", "actor", "target", "object", "location", "start", "end", "detail", "confidence", "evidence_ids", "criteria"},
    "questions": {"prompt", "answer", "answer_type", "interval", "ambiguity_flags", "evidence_ids", "evidence_frame_ids", "evidence_times", "evidence_intervals"},
}


def add_correction(store: Store, target: str, record_id: str, changes: dict, status: str, reviewer="local"):
    if target not in EDITABLE or status not in STATUSES or not isinstance(changes, dict):
        raise ValueError("Invalid review target/status/changes")
    if set(changes) - EDITABLE[target]:
        raise ValueError("Correction includes immutable or unsupported fields")
    records = {r["id"]: r for r in store.effective(store.rows(target), target)}
    if record_id not in records:
        raise ValueError("Record not found")
    record = {**records[record_id], **changes, "review_status": status}
    if changes.get("evidence_ids"):
        known = {r["id"] for name in ("observations", "states", "events") for r in store.rows(name)}
        known.update(frame_index(store))
        if not set(changes["evidence_ids"]) <= known:
            raise ValueError("Evidence IDs must refer to existing source evidence")
    if changes.get("evidence_frame_ids") and not set(changes["evidence_frame_ids"]) <= frame_index(store).keys():
        raise ValueError("Unknown evidence frame IDs")
    if target == "observations":
        validate_observation(record)
    elif target == "questions":
        validate_question(record)
    else:
        if record["kind"] not in EVENTS or not finite(record["start"]) or not finite(record["end"]) or not 0 <= record["start"] <= record["end"]:
            raise ValueError("Invalid event kind/timestamps")
    manifest = read_json(store.path("manifest.json"))
    interval = record.get("interval", [record.get("start", record.get("time", 0)), record.get("end", record.get("time", 0))])
    if not manifest["interval"][0] <= interval[0] <= interval[1] <= manifest["interval"][1]:
        raise ValueError("Review bounds outside run interval")
    lock_path = store.path("corrections/.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        revision = len(store.corrections()) + 1
        correction = {"id": f"correction_{revision:08d}_{secrets.token_hex(4)}", "revision": revision,
                      "target": target, "record_id": record_id, "changes": changes,
                      "review_status": status, "reviewer": str(reviewer)[:120], "created_at": now()}
        atomic_json(store.path(f"corrections/{revision:08d}.json"), correction)
        atomic_json(store.path("invalidated.json"), {"revision": revision,
                    "stages": ["fuse", "generate-qa", "evaluate", "report"] if target != "questions"
                    else ["evaluate", "report"], "reason": "Human correction; rerun downstream stages"})
    return correction


def frame_index(store):
    index = {f["id"]: f for f in store.rows("previews")}
    for path in store.path("tasks").glob("*.json"):
        for frame in read_json(path).get("frames", []):
            index[frame["id"]] = frame
    return index


def create_app(store: Store):
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, HTMLResponse
    app = FastAPI(title="League video evidence review")

    @app.get("/", response_class=HTMLResponse)
    def home():
        return Path(__file__).with_name("review.html").read_text()

    @app.get("/api/records")
    def records(target: str = "questions", offset: int = 0, limit: int = 200, q: str = ""):
        import json
        if target not in EDITABLE and target != "previews":
            raise HTTPException(400, "Invalid target")
        if offset < 0 or not 1 <= limit <= 500:
            raise HTTPException(400, "Invalid pagination")
        rows = store.effective(store.rows(target), target)
        if q:
            rows = [r for r in rows if q.casefold() in json.dumps(r, ensure_ascii=False).casefold()]
        invalidated = read_json(store.path("invalidated.json")) if store.path("invalidated.json").exists() else None
        if invalidated and store.path("config.json").exists() and store.path("stages/report.json").exists():
            config = Config.load(store.path("config.json"))
            if read_json(store.path("stages/report.json"))["key"] == store.stage_key("report", config, ["questions", "predictions"]):
                invalidated = None
        return {"records": rows[offset:offset+limit], "total": len(rows), "offset": offset,
                "editable": sorted(EDITABLE.get(target, set())), "corrections": store.corrections(),
                "invalidated": invalidated}

    @app.get("/api/evidence/{record_id}")
    def evidence(record_id: str):
        # Resolve state -> observation -> source frame, without exposing filesystem paths.
        rows = {r["id"]: r for target in ("questions", "events", "states", "observations", "previews")
                for r in store.effective(store.rows(target), target)}
        frames = frame_index(store)
        seen, selected = set(), {}
        def visit(key):
            if key in seen:
                return
            seen.add(key)
            if key in frames:
                selected[key] = frames[key]
            row = rows.get(key, {})
            for child in row.get("evidence_ids", []) + row.get("frame_ids", []) + row.get("evidence_frame_ids", []):
                visit(child)
            if row.get("frame_id"):
                visit(row["frame_id"])
        visit(record_id)
        processed = []
        paths = list(store.path("inference").rglob("input-audit.json")) + list(
            store.path("processor-previews").glob("*/*/input-audit.json"))
        for path in paths:
            audit = read_json(path)
            processed.extend(p for p in audit.get("processed_inputs", []) if p["source_frame"] in selected)
            if len(processed) >= 24:
                break
        return {"frames": sorted(selected.values(), key=lambda f: f["time"]), "processed": processed[:24]}

    # Read Request from the global namespace for FastAPI under postponed annotations.
    globals()["Request"] = Request
    @app.post("/api/corrections")
    async def correct(request: Request):
        origin = request.headers.get("origin")
        if origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
            raise HTTPException(403, "Review writes must originate from this local app")
        try:
            body = await request.json()
            return add_correction(store, body["target"], body["record_id"], body.get("changes", {}),
                                  body["review_status"], body.get("reviewer", "local"))
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/media/source")
    def source():
        video = Path(read_json(store.path("manifest.json"))["source"]["path"])
        return FileResponse(video, media_type="video/mp4", filename="source.mp4", content_disposition_type="inline")

    @app.get("/media/{relative:path}")
    def media(relative: str):
        candidate = store.path(relative).resolve()
        if (not candidate.is_relative_to(store.root) or candidate.suffix.lower() != ".png"
                or not candidate.is_file()):
            raise HTTPException(404)
        return FileResponse(candidate, media_type="image/png")

    return app
