"""Cast-bar evidence and source-only dataset review on the existing local app."""
from pathlib import Path
import re

from .core import read_json


def experiment_root(store, namespace, experiment):
    if namespace not in {"actions-v3", "actions-v3-hud-grounded", "cast-bars"} or not re.fullmatch(r"[a-f0-9]{20}", experiment):
        raise ValueError("Invalid cast-bar experiment")
    root = store.path("experiments")/namespace/experiment
    if not (root/"cast_bar_readings.jsonl").exists():
        raise ValueError("Cast-bar experiment not found")
    return root


def install_cast_bar_routes(app, store):
    from fastapi import HTTPException, Request
    from fastapi.responses import HTMLResponse, FileResponse
    globals()["Request"] = Request
    from .cast_bars import correct_reading, rebuild
    from .recognition_dataset import coverage, save_label

    def same_origin(request):
        origin = request.headers.get("origin")
        if origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
            raise HTTPException(403, "Review writes must originate from this app")

    @app.get("/cast-bars", response_class=HTMLResponse)
    def page():
        return Path(__file__).with_name("cast_bar_review.html").read_text()

    @app.get("/api/cast-bars")
    def experiments():
        return [{"namespace": namespace, "id": p.parent.name} for namespace in ("actions-v3", "actions-v3-hud-grounded", "cast-bars")
                for p in sorted(store.path("experiments").glob(namespace+"/*/cast_bar_readings.jsonl"))]

    @app.get("/api/cast-bars/{namespace}/{experiment}")
    def records(namespace: str, experiment: str):
        try:
            root=experiment_root(store,namespace,experiment); rows,events=rebuild(root)
            catalogs={r["catalog_id"]:read_json(root/"cast-catalogs"/(r["catalog_id"]+".json")) for r in rows}
            for row in rows:
                catalog=catalogs[row["catalog_id"]]
                row["description"]=catalog["abilities"].get(row["ability_id"])
                row["text_path"]=str((root/row["text_path"]).relative_to(store.root))
            return {"readings":rows,"occurrences":events,"timeline":events}
        except (OSError,ValueError,KeyError) as exc:
            raise HTTPException(404,str(exc)) from exc

    @app.post("/api/cast-bars/{namespace}/{experiment}/corrections/{record_id}")
    async def correction(namespace: str, experiment: str, record_id: str, request: Request):
        same_origin(request)
        try:
            body=await request.json()
            return correct_reading(experiment_root(store,namespace,experiment),record_id,body["changes"],body["reviewer"])
        except (OSError,ValueError,KeyError) as exc:
            raise HTTPException(422,str(exc)) from exc

    @app.get("/api/cast-dataset")
    def dataset():
        root=store.path("benchmarks/cast-bars/v1")
        if not (root/"dataset.json").exists():
            raise HTTPException(404,"Initialize the cast-bar dataset first")
        data=read_json(root/"dataset.json")
        # Source labels only. Never load recognition output into this endpoint.
        return {**data,"coverage":coverage(root),"labels":{p.stem:read_json(p) for p in (root/"labels").glob("*.json")}}

    @app.post("/api/cast-dataset/{clip_id}")
    async def label(clip_id: str, request: Request):
        same_origin(request)
        try:
            return save_label(store.path("benchmarks/cast-bars/v1"),clip_id,await request.json())
        except (OSError,ValueError,KeyError,StopIteration) as exc:
            raise HTTPException(422,str(exc)) from exc

    @app.get("/api/cast-dataset/{clip_id}/source")
    def source(clip_id: str):
        data=read_json(store.path("benchmarks/cast-bars/v1/dataset.json"))
        clip=next((c for c in data["clips"] if c["id"]==clip_id),None)
        if not clip:
            raise HTTPException(404)
        source=next(s for s in data["sources"] if s["id"]==clip["source_id"])
        return FileResponse(source["path"],media_type="video/mp4",content_disposition_type="inline")
