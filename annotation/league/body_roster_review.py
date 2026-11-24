"""Optional loopback review of timestamped scoreboard roster evidence."""

from pathlib import Path
from hashlib import sha256
import json

from .core import digest, now, read_json
from .inventory_references import load_inventory_references, reference_root
from .media import file_hash
from .scoreboard_skim import build_match_roster, checked_evidence, load_skim, save_roster_review


def create_app(output, cache):
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, HTMLResponse

    root = Path(output).resolve()
    manifest, inputs, samples, names = load_skim(root)
    if not manifest["recipe"]["single_match"]:
        raise ValueError("Cross-opening review requires a single-match skim")
    profile = read_json(root / "layout_profiles" / f"{manifest['recipe']['profile']}.json")
    if digest(profile) != manifest["recipe"]["profile"] or profile.get("calibration_source_sha256") != manifest["source"]["sha256"]:
        raise ValueError("Scoreboard layout changed or belongs to another source")
    version = profile["inventory"]["reference_version"]
    assets = load_inventory_references(reference_root(cache, version), version)["assets"]
    champions = sorted(asset["id"] for asset in assets if asset["kind"] == "champion")
    source = manifest["source"]
    video = Path(source["path"])
    video_verified = video.is_file() and file_hash(video) == source["sha256"]
    names_by_portrait = {row["portrait_id"]: row for row in names}
    rows, files = [], {}
    for sample in samples:
        for slot, portrait in sample["portraits"].items():
            identifier = portrait["id"]
            name = names_by_portrait.get(identifier)
            files[identifier] = {"portrait": (portrait["crop"], portrait["crop_sha256"]),
                                 "frame": (sample["frame"], sample["frame_sha256"])}
            name_hash = None
            if name:
                path = (root / name["crop"]).resolve()
                if path.is_relative_to(root) and path.is_file():
                    name_hash = file_hash(path)
                    files[identifier]["name"] = (name["crop"], name_hash)
            rows.append({"id": identifier, "slot": slot, "time": sample["time"], "opening": sample["opening"],
                         "candidate": portrait["value"], "name_id": name["id"] if name_hash else None,
                         "name_crop_sha256": name_hash})
    candidates = build_match_roster(samples, source["sha256"], manifest["interval"], single_match=True)
    roster_path = root / "match-roster.json"
    app = FastAPI(title="Scoreboard roster review")

    @app.get("/", response_class=HTMLResponse)
    def page():
        return Path(__file__).with_name("body_roster_review.html").read_text()

    @app.get("/api/roster")
    def roster():
        data = roster_path.read_bytes()
        current = json.loads(data)
        # Legacy skims may carry a pre-review confirmation flag.
        status = current["status"] if current.get("review") else candidates["status"]
        return {"interval": manifest["interval"], "status": status, "review": current.get("review"),
                "revision": sha256(data).hexdigest(), "video": video_verified,
                "champions": champions, "witnesses": rows, "conflicts": current.get("membership_conflicts"),
                "suggested": [{"portrait_id": row["best"]["id"], "champion": champion}
                              for team in candidates["members"].values() for champion, row in team.items()]}

    @app.get("/source")
    def source_video():
        if not video_verified:
            raise HTTPException(404, "The verified source video is unavailable")
        return FileResponse(video, media_type="video/mp4")

    @app.get("/evidence/{identifier}/{kind}")
    def evidence(identifier: str, kind: str):
        witness = files.get(identifier, {}).get(kind)
        if witness is None:
            raise HTTPException(404, "Scoreboard evidence unavailable")
        try:
            checked_evidence(root, *witness)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return FileResponse(root / witness[0], media_type="image/png")

    @app.post("/api/roster")
    def save(payload: dict, request: Request):
        origin = request.headers.get("origin")
        if origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
            raise HTTPException(403, "Review writes must originate from this local app")
        try:
            decisions = payload["portraits"]
            if any(row["champion"] is not None and row["champion"] not in champions for row in decisions):
                raise ValueError("Choose a champion from the pinned catalog")
            review = {key: payload[key] for key in ("interval", "continuity_verified", "reviewer", "portraits")}
            review.update(source_sha256=source["sha256"], samples_sha256=inputs["samples_sha256"], reviewed_at=now())
            revision = payload["revision"]
            if not isinstance(revision, str) or len(revision) != 64:
                raise ValueError("Reload the review before saving")
            return save_roster_review(root, review, expected_revision=revision)
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc

    return app
