"""Opt-in review of one campaign's current body identity evidence."""

from pathlib import Path

from .body_operational import effective_observations, track_revision
from .core import Store, digest, finite, read_json
from .media import file_hash


def install_body_routes(app, intake, campaign_root):
    from fastapi import HTTPException
    from fastapi.responses import FileResponse, HTMLResponse

    campaign = Store(Path(campaign_root))

    def current():
        pointer_path = campaign.path("identity-latest.json")
        if not pointer_path.exists():
            raise HTTPException(404, "Run body_campaign identify for this source first")
        pointer = read_json(pointer_path)
        path = Path(pointer["path"]).resolve()
        if not path.is_relative_to(campaign.path("identities").resolve()) or not path.is_file() or file_hash(path) != pointer["sha256"]:
            raise HTTPException(409, "Identity result changed")
        result = read_json(path)
        recipe = result["recipe"]
        source = read_json(intake.path("manifest.json"))["source"]["sha256"]
        if source != recipe["source_sha256"]:
            raise HTTPException(409, "Identity result belongs to another source")
        with campaign.review_snapshot():
            names = campaign.effective(campaign.rows("body_names"), "body_names")
            matches = campaign.effective(campaign.rows("body_matches"), "body_matches")
        if (digest(effective_observations(intake)) != recipe["intake_observations"]
                or digest(names) != recipe["names"] or digest(matches) != recipe["matches"]
                or track_revision(campaign.root) != recipe.get("tracks")):
            raise HTTPException(409, "Evidence changed. Rerun body_campaign identify to rebuild the timeline.")
        if recipe.get("slot_skim"):
            from .body_slot_context import skim_revision_matches

            if not skim_revision_matches(recipe["slot_skim"]):
                raise HTTPException(409, "Scoreboard evidence changed. Rerun body_campaign identify.")
        receipt = campaign.path(f"identity-{recipe['gallery_kind']}-held_out-score.json")
        if receipt.exists():
            score = read_json(receipt)
            if score.get("release_passed") is True and pointer["sha256"] in score["result_hashes"]:
                result["experimental"] = False
        return result

    @app.get("/body-identities", response_class=HTMLResponse)
    def page():
        return Path(__file__).with_name("body_review.html").read_text()

    @app.get("/api/body-identities")
    def timeline(time: float = 0, window: float = 1):
        if not finite(time) or not finite(window) or time < 0 or not 0 < window <= 12:
            raise HTTPException(422, "Enter a finite timestamp and a window up to twelve seconds")
        result = current()
        rows = []
        for row in result["predictions"]:
            if time <= row["time"] < time + window:
                rows.append({key: row.get(key) for key in ("id", "time", "vote_time", "track_id", "champion", "skin_id",
                             "player_name", "identity_source", "binding_id", "name_evidence_ids", "roster_id",
                             "roster_confirmed_at", "roster_status", "identity_available_at", "backfilled", "visual_attempt", "reason")})
        bindings = {row["binding_id"] for row in rows}
        roster_ids = {row["roster_id"] for row in rows}
        return {"predictions": rows, "counts": result["counts"], "experimental": result["experimental"],
                "bindings": [row for row in result["bindings"] if row["id"] in bindings],
                "rosters": [row for row in result["rosters"] if row["id"] in roster_ids]}

    @app.get("/body-identities/evidence/{identifier}/{kind}")
    def evidence(identifier: str, kind: str):
        result = current()
        row = next((row for row in result["predictions"] if row["id"] == identifier), {})
        fields = {"body": ("body_path", "body_sha256"), "mask": ("crop_mask_path", "crop_mask_sha256")}
        if kind not in fields or not row.get(fields[kind][0]):
            raise HTTPException(404, "Body crop unavailable")
        field, hash_field = fields[kind]
        path = Path(row[field]).resolve()
        if not path.is_relative_to(campaign.root) or path.suffix != ".png" or not path.is_file() or file_hash(path) != row[hash_field]:
            raise HTTPException(404, "Body crop unavailable")
        return FileResponse(path, media_type="image/png")
