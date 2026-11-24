"""Review the observed or last-seen inventory at a source-video timestamp."""

from bisect import bisect_right
from pathlib import Path

from .core import finite, read_json


def install_inventory_routes(app, store):
    from fastapi import HTTPException
    from fastapi.responses import HTMLResponse

    cached = {}

    def load(name, index):
        path = store.path(name + ".jsonl")
        try:
            stat = path.stat()
        except FileNotFoundError:
            return index([])
        version = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
        entry = cached.get(name)
        if entry is None or entry[0] != version:
            entry = (version, index(store.rows(name)))
            cached[name] = entry
        return entry[1]

    @app.get("/inventory", response_class=HTMLResponse)
    def page():
        return Path(__file__).with_name("inventory_review.html").read_text()

    @app.get("/api/inventory")
    def inventory(time: float = 0):
        if not finite(time) or time < 0:
            raise HTTPException(422, "Inventory time must be a finite source timestamp")
        times, samples = load("inventory_timeline", lambda rows: ([row["time"] for row in rows], rows))
        if not samples:
            return {"status": "empty", "snapshots": []}
        index = bisect_right(times, time + 1e-6) - 1
        if index < 0 or time >= samples[index]["end"]:
            return {"status": "unsampled", "snapshots": [], "first_time": times[0]}
        sample = samples[index]
        snapshots = load("inventory_snapshots", lambda rows: {row["id"]: row for row in rows})
        result, catalogs = [], {}
        for identifier in sample["snapshot_ids"] + sample["last_seen_ids"]:
            snapshot = snapshots.get(identifier)
            if snapshot is None:
                continue
            catalog_id = snapshot["catalog_id"]
            if catalog_id not in catalogs:
                root = f"inventory-catalogs/{catalog_id}"
                catalog = read_json(store.path(root + "/manifest.json"))
                catalogs[catalog_id] = (catalog, {item["id"]: item for item in catalog["items"]})
            catalog, items = catalogs[catalog_id]
            slots = []
            for slot in snapshot["slots"]:
                item = items.get(slot["item_id"])
                detail = {"name": item["name"], "description": item["description"],
                          "icon_path": f"inventory-catalogs/{catalog_id}/{item['path']}"} if item else None
                slots.append({**slot, "reference": detail})
            result.append({**snapshot, "slots": slots, "reference_version": catalog["reference_version"],
                           "champion_name": catalog.get("champions", {}).get(snapshot["champion"], snapshot["champion"]),
                           "last_seen": identifier in sample["last_seen_ids"]})
        return {"status": "sampled", "sample_time": sample["time"], "scoreboard": sample["scoreboard"], "snapshots": result}
