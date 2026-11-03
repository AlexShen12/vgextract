"""Isolated action-experiment review routes on the existing League review app."""
from __future__ import annotations

import fcntl
from pathlib import Path
import re

from .action_state import CLASSES
from .core import STATUSES, atomic_json, finite, now, read_json, read_rows


def experiment_root(store, experiment_id):
    if not re.fullmatch(r"[0-9a-f]{20}", experiment_id):
        raise ValueError("Invalid action experiment ID")
    candidates = [store.path(f"experiments/actions-v{v}") / experiment_id for v in (2, 3)]
    matches = [p for p in candidates if (p/"input-audit.json").is_file()]
    if len(matches) != 1:
        raise ValueError("Action experiment not found")
    return matches[0]


def modes(root):
    return [mode for mode in ("core", "unassisted", "assisted") if (root/mode).is_dir()]


def effective_actions(root):
    indexed = {r["id"]: r for mode in modes(root) for r in read_rows(root / mode / "actions.jsonl")}
    for path in sorted((root / "reviews").glob("*.json")):
        correction = read_json(path)
        if correction["record_id"] in indexed:
            indexed[correction["record_id"]].update(correction["changes"], review_status=correction["review_status"])
    return list(indexed.values())


def correct_action(store, experiment_id, record_id, changes, status):
    root = experiment_root(store, experiment_id)
    allowed = {"champion", "name", "actor_description", "start", "end", "evidence_class", "terminal_status",
               "reward_gold", "count_delta", "target_kind", "slot_id", "actor_track", "review_notes"}
    if status not in STATUSES or not isinstance(changes, dict) or set(changes)-allowed:
        raise ValueError("Invalid experimental correction fields/status")
    indexed = {r["id"]: r for r in effective_actions(root)}
    if record_id not in indexed:
        raise ValueError("Action record not found")
    row = {**indexed[record_id], **changes}
    audit = read_json(root / "input-audit.json")
    if (not finite(row["start"]) or not finite(row["end"])
            or not audit["interval"][0] <= row["start"] <= row["end"] <= audit["interval"][1]
            or row["evidence_class"] not in CLASSES):
        raise ValueError("Invalid action correction bounds/class")
    for field in ("champion", "name", "actor_description"):
        if row.get(field) is not None and (not isinstance(row[field], str) or len(row[field]) > 500):
            raise ValueError("Invalid action correction text")
    if "terminal_status" in changes and (row.get("kind") != "recall" or changes["terminal_status"] not in {"completed","interrupted","ending_unknown"}):
        raise ValueError("Invalid recall terminal status")
    for field in ("reward_gold", "count_delta"):
        value = changes.get(field)
        if field in changes and value is not None and (type(value) is not int or value <= 0):
            raise ValueError("Reward/count corrections require positive integers or null")
    if changes.get("reward_gold") is not None and row.get("kind") not in {"minion_last_hit","jungle_last_hit","creep_last_hit_candidate"}:
        raise ValueError("A gold-counter change is not an exact kill reward")
    if "target_kind" in changes and changes["target_kind"] not in {"lane_minion","jungle","unknown"}:
        raise ValueError("Invalid creep target kind")
    if changes.get("slot_id") is not None:
        audit_slots = {s["id"] for o in audit["profile"]["owners"] for s in o["slots"]}
        if changes["slot_id"] not in audit_slots:
            raise ValueError("Invalid reviewed slot")
    for field in ("actor_track", "review_notes"):
        if field in changes and changes[field] is not None and (not isinstance(changes[field], str) or len(changes[field]) > 500):
            raise ValueError("Invalid review track/notes")
    directory = root / "reviews"
    directory.mkdir(exist_ok=True)
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        revision = len(list(directory.glob("*.json")))+1
        correction = {"revision": revision, "record_id": record_id, "changes": changes,
                      "review_status": status, "created_at": now(), "experimental": True}
        atomic_json(directory / f"{revision:08d}.json", correction)
    return correction


def install_action_routes(app, store):
    from fastapi import HTTPException, Request
    from fastapi.responses import HTMLResponse
    globals()["Request"] = Request

    @app.get("/actions", response_class=HTMLResponse)
    def page():
        return Path(__file__).with_name("action_review.html").read_text()

    @app.get("/action-labels", response_class=HTMLResponse)
    def labels_page():
        return Path(__file__).with_name("action_labels.html").read_text()

    @app.get("/api/action-labels")
    def labels():
        path = store.path("benchmarks/actions-v3/actions.json")
        if not path.exists():
            raise HTTPException(404, "Initialize experiment --suite actions labels first")
        return read_json(path)

    @app.post("/api/action-labels/{clip_id}")
    async def save_label(clip_id: str, request: Request):
        origin = request.headers.get("origin")
        if origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
            raise HTTPException(403, "Labels must be submitted from this local app")
        from .action_benchmark_v3 import validate
        from .action_v3 import load_profile as load_action_profile
        directory = store.path("benchmarks/actions-v3")
        try:
            body = await request.json()
            if (body.get("independent_of_predictions") is not True or not isinstance(body.get("labeler"), str)
                    or not body["labeler"].strip() or len(body["labeler"]) > 120):
                raise ValueError("State the independent labeler and confirm source-only annotation")
            with (directory/".labels.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                data = read_json(directory/"actions.json")
                clip = next(c for c in data["clips"] if c["id"] == clip_id)
                if clip["split"] == "development" and (directory/"development-freeze.json").exists():
                    raise ValueError("Development labels are frozen; start a new benchmark revision")
                changes = body["annotations"]
                allowed = {"identities", "bindings", "transitions", "actions", "case_tags", "repeat_group", "verified"}
                if not isinstance(changes, dict) or set(changes)-allowed or type(changes.get("verified", False)) is not bool:
                    raise ValueError("Invalid independent annotation fields")
                original = dict(clip)
                clip.update(changes, labeler=body["labeler"], independent_of_predictions=True)
                data.update(labeler=body["labeler"], independent_of_predictions=True)
                validate(store, data, load_action_profile(), draft=True)
                revisions = directory/"revisions"
                revision = len(list(revisions.glob("*.json")))+1
                atomic_json(revisions/f"{revision:08d}.json", {"original": original, "new": clip, "created_at": now()})
                atomic_json(directory/"actions.json", data)
            return {"clip_id": clip_id, "revision": revision, "verified": clip["verified"]}
        except (ValueError, KeyError, TypeError, OSError, StopIteration) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/actions")
    def experiments():
        rows = []
        paths = [p for version in (2, 3) for p in store.path(f"experiments/actions-v{version}").glob("*/report.json")]
        for path in sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True):
            report = read_json(path)
            rows.append({"id": path.parent.name, "status": report["status"], "interval": report["interval"],
                         "results": report.get("results", {}), "schema_version": report.get("schema_version", 2), "experimental": True})
        return {"experiments": rows}

    @app.get("/api/actions/{experiment_id}")
    def experiment(experiment_id: str):
        try:
            root = experiment_root(store, experiment_id)
            audit = read_json(root / "input-audit.json")
            frames = []
            for frame in audit["frames"]:
                native, processed = Path(frame["native_path"]).resolve(), Path(frame["input_path"]).resolve()
                if not native.is_relative_to(store.root) or not processed.is_relative_to(store.root):
                    raise ValueError("Action media outside run")
                frames.append({**frame, "native_path": str(native.relative_to(store.root)),
                               "input_path": str(processed.relative_to(store.root))})
            report = read_json(root / "report.json")
            if (root / "progress.json").exists() and report["status"] != "completed_experimental":
                report.update(read_json(root / "progress.json"))
            return {"report": report, "actions": effective_actions(root),
                    "frames": frames, "source_fps": audit["source"]["fps"],
                    "modes": modes(root),
                    **{name: [r for mode in modes(root) for r in read_rows(root / mode / (name+".jsonl"))]
                       for name in ("resampling", "transitions", "identities", "bindings", "hud_readings", "scene_proposals",
                                    "body_readings", "counter_readings", "recall_readings")}}
        except (ValueError, OSError, KeyError) as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/actions/{experiment_id}/abilities")
    def abilities(experiment_id: str):
        import os
        from .core import Config
        try:
            root = experiment_root(store, experiment_id)
            report = read_json(root/"report.json")
            version = report.get("icon_reference_version") or "16.13.1"
            if not re.fullmatch(r"\d+\.\d+\.\d+", version):
                raise ValueError("Invalid ability reference version")
            config = Config.load(store.path("config.json") if store.path("config.json").exists() else Path(__file__).with_name("midbeast.json"))
            cache = Path(os.environ.get("LEAGUE_ASSET_CACHE", config.asset_cache))
            path = cache/"abilities"/version/"catalog.json"
            if not path.exists():
                return {"status": "not_cached", "reference_version": version, "champions": {}}
            data = read_json(path)
            names = {r.get("hud_champion") for mode in modes(root) for r in read_rows(root/mode/"identities.jsonl")}
            return {"status": data.get("status", "downloading"), "reference_version": version, "source_patch": None,
                    "mechanics_enabled": False, "champions": {k: v for k, v in data["champions"].items() if k in names}}
        except (ValueError, OSError, KeyError) as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/actions/{experiment_id}/requests/{request_id}")
    def raw(experiment_id: str, request_id: str):
        try:
            root = experiment_root(store, experiment_id)
            if not re.fullmatch(r"action-request_[0-9a-f]{20}", request_id):
                raise ValueError("Invalid action request")
            directory = root / "requests" / request_id
            return {"input": read_json(directory / "input.json"), "response": read_json(directory / "response.json"),
                    "parsed": read_json(directory / "parsed.json")}
        except (ValueError, OSError) as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post("/api/actions/{experiment_id}/corrections")
    async def correction(experiment_id: str, request: Request):
        origin = request.headers.get("origin")
        if origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
            raise HTTPException(403, "Review writes must originate from this local app")
        try:
            body = await request.json()
            return correct_action(store, experiment_id, body["record_id"], body.get("changes", {}), body["review_status"])
        except (ValueError, KeyError, TypeError, OSError) as exc:
            raise HTTPException(422, str(exc)) from exc
