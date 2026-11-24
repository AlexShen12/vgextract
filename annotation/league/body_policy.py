"""Calibrate body thresholds on development samples; bind subsequent runs to those inputs."""
from pathlib import Path

from .body_identity import BodyGallery, predict_samples
from .core import atomic_json, digest, now, read_json
from .recognition_metrics import identity_metrics


def freeze(root):
    root=Path(root)
    if (root/"policy.json").exists():raise ValueError("Body policy is frozen; create a new campaign revision")
    data=read_json(root/"embeddings.json");campaign=read_json(root/"campaign.json")
    labels=[l for l in campaign["labels"] if l["split"]=="development"]
    known={l["id"] for l in labels}
    pairs=[(s,v) for s,v in zip(data["samples"],data["query_vectors"]) if s["split"]=="development" and s["id"] in known]
    if not pairs or not labels:raise ValueError("Source development labels and embeddings are required")
    if {e["reference_kind"] for e in data["gallery"]}!={"gameplay","rendered"}:raise ValueError("Both reviewed reference galleries are required")
    gallery=BodyGallery(data["gallery"],data["gallery_vectors"],encoder_signature=data["encoder_signature"],excluded_groups=data["evaluation_match_groups"])
    best=None
    for score in (.55,.65,.75,.8,.85,.9,.95):
        for margin in (.03,.06,.1,.15,.2):
            policy={"score":score,"margin":margin,"development_freeze":"development-search"}
            predictions=list(predict_samples([p[0] for p in pairs],[p[1] for p in pairs],gallery,policy))
            metrics={}
            for team in ("friendly","enemy"):
                ids={l["id"] for l in labels if l["team"]==team}
                metrics[team]=identity_metrics([l for l in labels if l["id"] in ids],[p for p in predictions if p["id"] in ids])
            if not all(m["passed"] for m in metrics.values()):continue
            rank=(min(m["coverage"] for m in metrics.values()),score,margin)
            if best is None or rank>best[0]:best=(rank,policy,metrics)
    if best is None:raise ValueError("No development threshold achieves 98% precision and 70% coverage on both teams")
    result={**best[1],"development_metrics":best[2],"development_labels_hash":digest(labels),
            "encoder_signature":data["encoder_signature"],"gallery_hash":digest(data["gallery"]),
            "embeddings_hash":digest(data),"created_at":now()}
    result["development_freeze"]=digest(result)
    atomic_json(root/"policy.json",result);return result


def validate_freeze(root,policy):
    root=Path(root);data=read_json(root/"embeddings.json");campaign=read_json(root/"campaign.json")
    labels=[l for l in campaign["labels"] if l["split"]=="development"]
    expected={"development_labels_hash":digest(labels),"encoder_signature":data["encoder_signature"],
              "gallery_hash":digest(data["gallery"]),"embeddings_hash":digest(data)}
    if not policy.get("development_freeze") or any(policy.get(k)!=v for k,v in expected.items()):
        raise ValueError("Body development labels, gallery or embeddings changed after freezing")
    return data
