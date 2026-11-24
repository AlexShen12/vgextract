"""Development-only paired body crops using immutable completed SAM artifacts."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np

from .core import atomic_json, digest, implementation_hash, read_json, write_rows
from .media import file_hash, probe

REPRESENTATIONS=("box","oracle-mask","sam-verified","sam-automatic")


def iou(a,b):
    return float((a&b).sum()/max(1,(a|b).sum()))


def model_masks(prepared,clip,arm,frame_id):
    from PIL import Image
    selected=clip["arms"][arm]
    if selected["status"]!="prepared":return None,"unseeded"
    indices=[i for i,f in enumerate(selected["frames"]) if f["id"]==frame_id]
    if len(indices)!=1:return None,"outside_track_context"
    result_path=Path(prepared["root"])/arm/(clip["id"]+"-complete.json")
    if not result_path.exists():return None,"missing_result"
    result=read_json(result_path)
    frame=next((f for f in result["frames"] if f["frame_index"]==indices[0]),None)
    masks=[m for m in (frame or {}).get("masks",[]) if m["object_id"]==1]
    if len(masks)!=1:return None,"missing_mask"
    record=masks[0];path=Path(record["mask_path"])
    if file_hash(path)!=record["sha256"]:raise ValueError("Completed SAM mask was modified")
    with Image.open(path) as image:mask=np.asarray(image.convert("L"))>127
    return mask,"available"


def extract(spec,output):
    from PIL import Image,ImageDraw
    from .body_identity import healthbar_proposals
    prepared_path=Path(spec["sam_prepared"]);prepared=read_json(prepared_path)
    labels=read_json(Path(prepared["root"])/"labels.json")
    label_hash=digest({"labels":labels,"masks":{r["mask_path"]:file_hash(Path(prepared["labels_dir"])/r["mask_path"])
        for c in prepared["clips"] for r in c["references"] if r["visibility"]=="visible"}})
    if label_hash!=prepared["recipe"]["labels"]:raise ValueError("Independent source labels changed")
    output=Path(output);rows=[];localization=[]
    profile=prepared["recipe"]["profile"]
    for clip in prepared["clips"]:
        champion=clip["arms"]["verified"]["seed"].get("champion")
        if champion is None:raise ValueError("Diagnostic source identity must be independently supplied")
        for reference in clip["references"]:
            source=min(clip["frames"],key=lambda f:abs(f["time"]-reference["time"]))
            if abs(source["time"]-reference["time"])>.032:raise ValueError("Reference/source PTS mismatch")
            if file_hash(Path(source["native_path"]))!=source["native_sha256"]:raise ValueError("Source frame changed")
            with Image.open(source["native_path"]) as im:image=im.convert("RGB")
            models={arm:model_masks(prepared,clip,arm,source["id"]) for arm in ("verified","automatic")}
            truth=None
            if reference["visibility"]=="visible":
                path=Path(prepared["labels_dir"])/reference["mask_path"]
                with Image.open(path) as im:truth=np.asarray(im.convert("L"))>127
                if truth.shape!=(image.height,image.width) or not truth.any():raise ValueError("Invalid independent source mask")
            for arm,(mask,status) in models.items():
                if mask is not None and mask.shape!=(image.height,image.width):raise ValueError("SAM mask lost native coordinates")
                localization.append({"clip_id":clip["id"],"frame_id":source["id"],"time":source["time"],"arm":arm,
                    "reference_visibility":reference["visibility"],"availability":status,
                    "iou":iou(truth,mask) if truth is not None and mask is not None else 0 if truth is not None else None,
                    "false_visible":bool(mask is not None and mask.any()) if reference["visibility"]=="occluded" else None})
            if truth is None:continue
            y,x=np.where(truth);box=[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)]
            clean=image.copy();draw=ImageDraw.Draw(clean)
            for region in spec["masks"]+reference.get("excluded_overlay_rects",[]):draw.rectangle(region,fill=(127,127,127))
            for proposal in healthbar_proposals(image,profile):
                a,b,c,d=proposal["bar_box"];draw.rectangle((a-30,b-32,c+30,d+3),fill=(127,127,127))
            crop=clean.crop(box)
            for kind in REPRESENTATIONS:
                mask=truth if kind=="oracle-mask" else np.ones_like(truth) if kind=="box" else models[kind.removeprefix("sam-")][0]
                identifier=digest([clip["id"],source["id"],kind])[:20]
                record={"id":identifier,"clip_id":clip["id"],"frame_id":source["id"],"time":source["time"],
                        "champion_reference":champion,"representation":kind,"box":box,"source_sha256":source["native_sha256"],
                        "source_labeler":reference["labeler"],"human_verified":False}
                if mask is None:
                    rows.append({**record,"status":"missing_mask"});continue
                cropped=Image.fromarray(mask.astype("uint8")*255).crop(box)
                if not np.asarray(cropped).any():
                    rows.append({**record,"status":"empty_mask"});continue
                folder=output/"crops"/identifier;folder.mkdir(parents=True,exist_ok=True)
                crop.save(folder/"body.png");cropped.save(folder/"mask.png")
                rows.append({**record,"status":"available","image":str(folder/"body.png"),"mask":str(folder/"mask.png"),
                             "image_sha256":file_hash(folder/"body.png"),"mask_sha256":file_hash(folder/"mask.png")})
    data={"spec_hash":digest(spec),"sam_prepared_sha256":file_hash(prepared_path),"samples":rows,"localization":localization,
          "scope":"single-match selected-body diagnostic; oracle-defined shared crop geometry; no cross-champion accuracy claim"}
    atomic_json(output/"inputs.json",data);return data


def summarize_vectors(rows):
    result={}
    for representation in REPRESENTATIONS:
        group=[r for r in rows if r["representation"]==representation]
        available=[r for r in group if r.get("vector") is not None]
        scores=[]
        for row in available:
            candidates=[other for other in available if other["clip_id"]!=row["clip_id"]]
            if candidates:scores.append(max(float(np.dot(row["vector"],other["vector"])) for other in candidates))
        result[representation]={"eligible":len(group),"available":len(available),"missing":len(group)-len(available),
            "cross_clip_best_similarity_quantiles":np.quantile(scores,[0,.25,.5,.75,1]).tolist() if scores else None,
            "similarity_is_probability":False,"precision":None,"coverage_is_extraction_only":True}
    return result


def run(spec_path,output,*,device="cuda"):
    from PIL import Image
    from .body_identity import FrozenEncoder
    from .recognition_budget import reserve,finish
    spec=read_json(Path(spec_path));output=Path(output);output.mkdir(parents=True,exist_ok=True)
    reservation=reserve(output,"body-crop-diagnostic",900 if device=="cuda" else 0)
    started=time.monotonic();status="failed"
    try:
        inputs=extract(spec,output);encoder=FrozenEncoder(spec["encoder_lock"],device)
        rows=[]
        for row in inputs["samples"]:
            if row["status"]=="available":
                with Image.open(row["image"]) as image,Image.open(row["mask"]) as mask:vector=encoder.encode(image,mask).tolist()
                row={**row,"vector":vector,"encoder_signature":encoder.signature}
            rows.append(row);write_rows(output/"embeddings.jsonl",rows)
        render_path=Path(spec["render_report"])
        rendered=read_json(render_path) if render_path.exists() else {"renders":[]}
        references=[]
        for reference in rendered["renders"]:
            if file_hash(Path(reference["image"]))!=reference["image_sha256"]:raise ValueError("Static reference changed")
            if file_hash(Path(reference["mask"]))!=reference["mask_sha256"]:raise ValueError("Static reference mask changed")
            with Image.open(reference["image"]) as image,Image.open(reference["mask"]) as mask:vector=encoder.encode(image,mask)
            references.append({**reference,"vector":vector.tolist()})
        hypotheses=[]
        for row in rows:
            if not row.get("vector") or not references:continue
            by_champion=defaultdict(list)
            for reference in references:by_champion[reference["champion"]].append(float(np.dot(row["vector"],reference["vector"])))
            ranked=sorted(((max(values),name) for name,values in by_champion.items()),reverse=True)
            hypotheses.append({"sample_id":row["id"],"representation":row["representation"],"accepted_identity":None,
                "reference_status":"unreviewed_static_pose_candidates","candidates":[{"champion":name,"similarity":score} for score,name in ranked[:5]]})
        write_rows(output/"render_candidate_ranks.jsonl",hypotheses)
        report={"status":"completed_development_diagnostic","representations":summarize_vectors(rows),"localization":inputs["localization"],
            "encoder_signature":encoder.signature,"source_input_hash":digest(inputs),"render_candidates":len(references),
            "quality_gate_passed":False,"human_verified":False,"cross_match_evaluation":False,
            "scope":inputs["scope"],"seconds":time.monotonic()-started}
        atomic_json(output/"report.json",report);status="completed";return report
    except BaseException as exc:
        atomic_json(output/"failure.json",{"error":str(exc),"seconds":time.monotonic()-started});raise
    finally:finish(output,reservation["attempt"],time.monotonic()-started,status)


def source_geometry_probe(video, labels_path, encoder_lock, output):
    """Measure same-match feature sensitivity on independently drawn development masks."""
    from PIL import Image

    from .body_features import encode_features, patch_similarity, render_variant, silhouette_similarity
    from .body_identity import FrozenEncoder
    from .sam_tracking import validate_labels
    from .scene_reference import sample_source

    started = time.monotonic()
    source_hash = file_hash(video)
    labels = read_json(labels_path)
    clips = validate_labels(labels, {**probe(video), "sha256": source_hash}, labels_path.parent)
    crops = []
    for clip in clips:
        references = {round(row["time"], 6): row for row in clip["references"] if row["visibility"] == "visible"}
        for pts, frame in sample_source(video, clip["start"], clip["end"], 16):
            row = references.pop(round(pts, 6), None)
            if row is None:
                continue
            path = labels_path.parent / row["mask_path"]
            with Image.open(path) as original:
                mask = original.convert("L").point(lambda value: 255 if value > 127 else 0)
            box = mask.getbbox()
            if box is None:
                raise ValueError("Visible source mask is empty")
            crops.append((clip["id"], pts, file_hash(path), frame.crop(box), mask.crop(box)))
        if references:
            raise ValueError(f"Source mask PTS missing in {clip['id']}")
    cross_clip = [[j for j, other in enumerate(crops) if other[0] != row[0]] for row in crops]
    if not crops or not all(cross_clip):
        raise ValueError("The diagnostic needs source masks in two or more windows")

    encoder = FrozenEncoder(encoder_lock, "cpu")
    geometry = {"stretched": {"crop": "legacy"}, "padding_0": {"padding": 0},
                "padding_10": {"padding": .1, "patch_layer": -1, "weights": [.5, .25, .25]},
                "padding_20": {"padding": .2}}
    features = {name: [encode_features(encoder, image, mask, policy) for _, _, _, image, mask in crops]
                for name, policy in geometry.items()}

    def quantiles(values):
        return np.quantile(values, [0, .25, .5, .75, 1]).tolist()

    geometry_report = {}
    for name, rows in features.items():
        vectors = np.asarray([row["rgb"] for row in rows])
        cosines = vectors @ vectors.T
        scores = [max(cosines[i, j] for j in others) for i, others in enumerate(cross_clip)]
        top_three = [float(np.mean(sorted((cosines[i, j] for j in others), reverse=True)[:3]))
                     for i, others in enumerate(cross_clip) if len(others) >= 3]
        geometry_report[name] = {"best_rgb_cosine": quantiles(scores),
                                 "best_three_rgb_cosine": quantiles(top_three) if top_three else None,
                                 "input_transforms": [row["geometry"] for row in rows]}

    rows = features["padding_10"]
    anchors = [max(others, key=lambda j: float(np.dot(rows[i]["rgb"], rows[j]["rgb"])))
               for i, others in enumerate(cross_clip)]
    appearance = {"rgb": [], "grayscale": [], "silhouette": [], "fusion": []}
    for i, j in enumerate(anchors):
        rgb = float((np.dot(rows[i]["rgb"], rows[j]["rgb"]) + 1) / 2)
        gray = float((np.dot(rows[i]["gray"], rows[j]["gray"]) + 1) / 2)
        shape = silhouette_similarity(rows[i]["silhouette"], rows[j]["silhouette"])
        for key, value in (("rgb", rgb), ("grayscale", gray), ("silhouette", shape), ("fusion", .5 * rgb + .25 * gray + .25 * shape)):
            appearance[key].append(value)

    layers = {-1: rows, -4: [encode_features(encoder, image, mask, {"padding": .1, "patch_layer": -4})
                             for _, _, _, image, mask in crops]}
    patch_report = {}
    for layer, records in layers.items():
        matches = [patch_similarity(records[i]["patches"], records[j]["patches"])
                   for i, j in enumerate(anchors) if min(len(records[i]["patches"]), len(records[j]["patches"])) >= 4]
        patch_report[str(layer)] = {"eligible_pairs": len(matches), "global_fallback_pairs": len(anchors) - len(matches),
                                    "mutual_support": quantiles([match[1] for match in matches]) if matches else None}

    effects = {"downsample": {"body_height": 32}, "blur": {"blur": .6},
               "brightness": {"brightness": .85}, "compression": {"jpeg_quality": 70}}
    degradation = {}
    for name, variant in effects.items():
        changes = []
        for i, (_, _, _, image, mask) in enumerate(crops):
            image, mask = render_variant(image, mask, **variant)
            encoded = encode_features(encoder, image, mask, {"padding": .1})
            target = rows[anchors[i]]["rgb"]
            changes.append(float(np.dot(encoded["rgb"], target) - np.dot(rows[i]["rgb"], target)))
        degradation[name] = quantiles(changes)

    report = {"scope": "same-match development diagnostic with source masks", "release_passed": False,
              "source_sha256": source_hash, "labels_sha256": file_hash(labels_path),
              "encoder_lock_sha256": file_hash(encoder_lock), "encoder_signature": encoder.signature,
              "implementation": implementation_hash(["body_study.py", "body_features.py", "body_identity.py"]),
              "samples": [{"clip": clip, "time": pts, "mask_sha256": mask_hash} for clip, pts, mask_hash, _, _ in crops],
              "geometry": geometry_report, "patches": patch_report,
              "appearance_on_same_pairs": {name: quantiles(scores) for name, scores in appearance.items()},
              "query_degradation_rgb_cosine_change": degradation, "quantiles": [0, .25, .5, .75, 1],
              "precision": None, "coverage": None, "seconds": time.monotonic() - started}
    atomic_json(output / "source-geometry-report.json", report)
    return {"samples": len(crops), "seconds": report["seconds"], "path": str(output / "source-geometry-report.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--spec", type=Path)
    inputs.add_argument("--geometry-labels", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--encoder-lock", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device")
    args = parser.parse_args()
    if args.geometry_labels:
        if not args.video or not args.encoder_lock or args.device not in (None, "cpu"):
            parser.error("The CPU geometry probe requires --video and --encoder-lock")
        result = source_geometry_probe(args.video, args.geometry_labels, args.encoder_lock, args.output)
    else:
        result = run(args.spec, args.output, device=args.device or "cuda")
    print(json.dumps(result, indent=2))


if __name__=="__main__":main()
