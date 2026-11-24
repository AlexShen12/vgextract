"""Occurrence benchmark v2: exhaustive source labels, frozen arms and resumption."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import copy
import logging

from .action_v3 import load_profile
from .caption_benchmark import assessments, score_clip
from .caption_occurrences import EVENT_POLICY, RUBRIC, RUBRIC_VERSION
from .caption_pilot import COMPONENTS, caption_pilot
from .captions import PILOT_CRITERIA
from .core import atomic_json, digest, finite, implementation_hash, now, read_json, read_rows
from .intake_profile import load_profile as anchor_profile
from .local_models import LOCK

ACTION_KINDS = {"attack", "cast", "recall", "movement"}
CLAIM_KINDS = set(PILOT_CRITERIA) | {"displayed_state", "cs_increased"}
REQUIRED_CASES = {"q_phases", "recall_continuation", "rapid_attacks", "repeated_casts", "cast_miss",
                  "ambiguous_impact", "competing_targets", "occlusion", "counter_only"}
REQUIRED_CLAIMS = {"attack", "cast", "ability_slot", "targeted_attack", "ability_hit", "minion_last_hit"}


def reference_adequacy(clips):
    counts = Counter(r["kind"] for c in clips for r in c["claims"])
    conflicts = []
    for clip in clips:
        event_kinds = {r["id"]: r["kind"] for r in clip["events"]}
        for claim in clip["claims"]:
            event_id = claim.get("event_id")
            if claim["kind"] in ACTION_KINDS and event_kinds.get(event_id) != claim["kind"]:
                conflicts.append({"clip_id": clip["id"], "event_id": event_id, "claim_id": claim["id"],
                                  "event_kind": event_kinds.get(event_id), "claim_kind": claim["kind"]})
    return {"claim_counts": dict(sorted(counts.items())),
            "missing_cases": sorted(REQUIRED_CASES - {t for c in clips for t in c["case_tags"]}),
            "insufficient_claim_kinds": sorted(k for k in REQUIRED_CLAIMS if counts[k] < 5),
            "action_kind_conflicts": conflicts}


def prediction_fingerprint(root):
    """Bind assessments/receipts to actual artifacts, invariant to path projection."""
    root=Path(root)
    audit=read_json(root/'input-audit.json')
    source_path=audit.get('source',{}).get('path')
    def canonical(value):
        if isinstance(value,str):
            value=value.replace(str(root.resolve()),'<PILOT>')
            return value.replace(source_path,'<SOURCE>') if source_path else value
        if isinstance(value,list):return [canonical(v) for v in value]
        if isinstance(value,dict):return {k:canonical(v) for k,v in value.items()}
        return value
    tables=('captions','events','scene_occurrences','caption_clips','states','normalized_observations','body_readings')
    return digest(canonical({'audit':audit,**{t:read_rows(root/(t+'.jsonl')) for t in tables}}))


def exclusions(store, profile):
    from .qwen35_study import used_intervals
    intervals = used_intervals(store)
    source = read_json(store.path("manifest.json"))["source"]["sha256"]
    for path in store.path("benchmarks").rglob("*.json"):
        data = read_json(path)
        if isinstance(data, dict) and data.get("source_sha256") == source:
            intervals += [[c["start"], c["end"]] for c in data.get("clips", []) if "start" in c and "end" in c]
    intervals += [[t, t] for t in anchor_profile(profile["anchor_profile"]).get("calibration_frames", [])]
    return sorted(intervals)


def choose_clips(interval, excluded, development=()):
    lower, upper = interval
    # Known pilot mistakes are development fixtures, never held-out material.
    centers = [34, 105]+list(development)
    dev = []
    for center in centers:
        start = max(lower, min(upper-24, center-12))
        if not any(start < c[1]+8 and c[0]-8 < start+24 for c in dev):
            dev.append([start, start+24])
        if len(dev) == 10:
            break
    cursor = lower
    while len(dev) < 10 and cursor+24 <= upper:
        if not any(cursor < b+8 and a-8 < cursor+24 for a,b in dev):
            dev.append([cursor, cursor+24])
        cursor += 32
    blocked = sorted([[max(lower,a-8), min(upper,b+8)] for a,b in list(excluded)+dev])
    merged = []
    for a,b in blocked:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(b, merged[-1][1])
        else:
            merged.append([a,b])
    gaps, cursor = [], lower
    for a,b in merged+[[upper,upper]]:
        if a-cursor >= 12:
            gaps.append([cursor,a])
        cursor = max(cursor,b)
    held = []
    # Prefer wider context, falling back only where history exclusions demand it.
    for length in (24, 20, 16, 12):
        for a,b in gaps:
            t = a
            while t+length <= b:
                if not any(t < y+8 and x-8 < t+length for x,y in held):
                    held.append([t,t+length])
                    if len(held) == 10:
                        break
                t += 1
            if len(held) == 10:
                break
        if len(held) == 10:
            break
    if len(dev) != 10 or len(held) != 10:
        raise ValueError("Need ten development and ten fresh 12–24 second neighborhoods; do not reuse inspected footage")
    return [{"id": f"event-{split}-{i:02d}", "split": split, "start": a, "end": b, "repeat_group": None,
             "verified": False, "inspected": False, "labeler": None, "independent_of_predictions": False,
             "coverage_exhaustive": False, "explicit_negative": False, "case_tags": [],
             "events": [], "claims": [], "bindings": [], "phases": [], "playback": [], "repeats": [],
             "negative_annotations": {k: False for k in ("events", "claims", "bindings", "phases", "playback", "repeats")}}
            for split, rows in (("development", sorted(dev)), ("held_out", sorted(held))) for i,(a,b) in enumerate(rows)]


def initialize(store, directory, profile):
    target = Path(directory)/"events.json"
    if target.exists():
        raise ValueError("Refusing to overwrite occurrence references")
    manifest = read_json(store.path("manifest.json"))
    old = store.path("benchmarks/captions/captions.json")
    development = [((c["start"]+c["end"])/2) for c in read_json(old)["clips"] if c["split"] == "development"] if old.exists() else []
    excluded = exclusions(store, profile)
    data = {"schema_version": 2, "source_sha256": manifest["source"]["sha256"], "source": manifest["source"],
            "profile": profile, "excluded_intervals": excluded, "rubric_version": RUBRIC_VERSION,
            "clips": choose_clips(manifest["interval"], excluded, development)}
    atomic_json(target, data)
    return {"status": "source_label_templates_created", "labels": str(target), "quality_metrics": None}


def validate(store, data, *, splits=()):
    if data.get("schema_version") != 2 or data.get("source_sha256") != read_json(store.path("manifest.json"))["source"]["sha256"]:
        raise ValueError("Occurrence reference source/schema mismatch")
    clips = data.get("clips", [])
    if len(clips) != 20 or len({c["id"] for c in clips}) != 20 or any(sum(c["split"] == s for c in clips) != 10 for s in ("development", "held_out")):
        raise ValueError("Need ten development and ten held-out occurrence neighborhoods")
    lower, upper = read_json(store.path("manifest.json"))["interval"]
    groups = {}
    for c in clips:
        if not all(finite(c.get(k)) for k in ("start", "end")) or not lower <= c["start"] < c["end"] <= upper or not 12 <= c["end"]-c["start"] <= 24:
            raise ValueError("Occurrence neighborhoods must be 12–24 source seconds")
        if c.get("repeat_group") and groups.setdefault(c["repeat_group"],c["split"]) != c["split"]:
            raise ValueError("Repeated footage crosses splits")
        if c["split"] == "held_out" and any(c["start"] < b+8 and a-8 < c["end"] for a,b in data["excluded_intervals"]):
            raise ValueError("Held-out footage was previously inspected or calibrated")
        for category in ("events", "claims", "bindings", "phases", "playback", "repeats", "case_tags"):
            if not isinstance(c.get(category), list):
                raise ValueError("Annotation categories must be explicit arrays")
        if c["split"] in splits or c.get("verified") is True:
            independent = c.get("independent_of_predictions") is True or (c["split"] == "development" and c.get("independent_of_current_predictions") is True)
            if not independent or not all(c.get(k) is True for k in ("verified", "inspected", "coverage_exhaustive")) or not c.get("labeler") or c.get("contaminated"):
                raise ValueError("Independent exhaustive source inspection is required")
            if any(not c[k] and c.get("negative_annotations", {}).get(k) is not True for k in ("events", "claims", "bindings", "phases")):
                raise ValueError("Empty categories require explicit independent negatives")
        event_ids, claim_ids = set(), set()
        for e in c["events"]:
            if not isinstance(e.get("id"), str) or e["id"] in event_ids or e.get("kind") not in ACTION_KINDS | {"unknown"}:
                raise ValueError("Invalid reference occurrence")
            event_ids.add(e["id"])
            if not all(finite(e.get(k)) for k in ("start", "end")) or not c["start"] <= e["start"] <= e["end"] <= c["end"]:
                raise ValueError("Reference occurrence outside coverage")
            if type(e.get("boundary_resolvable")) is not bool or any(type(e.get(k)) is not bool for k in ("left_censored", "right_censored")):
                raise ValueError("Explicit boundary/censoring labels required")
            if e["boundary_resolvable"] and (not finite(e.get("onset")) or not e["start"] <= e["onset"] <= e["end"]):
                raise ValueError("Resolvable onset must lie in occurrence")
        for r in c["claims"]:
            if r.get("kind") not in CLAIM_KINDS or not isinstance(r.get("id"),str) or r["id"] in claim_ids:
                raise ValueError("Invalid component reference")
            claim_ids.add(r["id"])
            if r["kind"] in PILOT_CRITERIA and r.get("event_id") not in event_ids:
                raise ValueError("Scene reference must identify an occurrence")
            if not all(finite(r.get(k)) for k in ("start", "end")) or not c["start"] <= r["start"] <= r["end"] <= c["end"]:
                raise ValueError("Component outside coverage")
        for b in c["bindings"]:
            box = b.get("box")
            if (b.get("claim_id") not in claim_ids or b.get("role") not in {"actor","target"} or not finite(b.get("time"))
                    or not c["start"] <= b["time"] <= c["end"] or not isinstance(box,list) or len(box)!=4 or not all(finite(x) for x in box)
                    or not 0 <= box[0] < box[2] <= 1920 or not 0 <= box[1] < box[3] <= 1080):
                raise ValueError("Invalid source binding")
    ordered = sorted(clips,key=lambda c:c["start"])
    if any(a["end"] > b["start"] for a,b in zip(ordered,ordered[1:])):
        raise ValueError("Occurrence neighborhoods overlap")
    if any(max(a["start"],b["start"])-min(a["end"],b["end"]) < 8 for a in clips for b in clips if a["split"] != b["split"]):
        raise ValueError("Occurrence splits need an eight-second guard")
    return clips


def temporal_iou(a,b):
    overlap=max(0,min(a["end"],b["end"])-max(a["start"],b["start"]))
    union=max(a["end"],b["end"])-min(a["start"],b["start"])
    return overlap/union if union else float(a["start"]==b["start"])


def match_events(references, predictions):
    """Maximum-cardinality one-to-one matching, not prediction-order greediness."""
    edges={p["id"]:[r["id"] for r in references if p.get("actor_track","selected")==r.get("actor_track","selected")
        and p.get("kind")==r.get("kind") and temporal_iou(p,r)>=.5
        and p.get('onset') is not None and r.get('onset') is not None
        and abs(p["onset"]-r["onset"])<=.25] for p in predictions}
    assigned={}
    def augment(pid,seen):
        for rid in edges[pid]:
            if rid not in seen:
                seen.add(rid)
                if rid not in assigned or augment(assigned[rid],seen):
                    assigned[rid]=pid
                    return True
        return False
    for p in sorted(predictions,key=lambda p:p["id"]):
        augment(p["id"],set())
    matched=set(assigned.values())
    return assigned, sum(p["id"] not in matched and bool(edges[p["id"]]) for p in predictions)


def score(clip, root):
    root=Path(root)
    # The original component matcher remains available, extended locally for the
    # new pilot-only slot kind without changing old frozen benchmark semantics.
    metrics=score_clip(clip,root,kinds=CLAIM_KINDS)
    caps=read_rows(root/"captions.jsonl"); judged=assessments(root)
    occ=read_rows(root/"scene_occurrences.jsonl")
    emitted={}
    for c in caps:
        kind=c["claims"][0]["kind"]
        if kind not in ACTION_KINDS:continue
        eid=c.get("event_id") or c["id"]
        row=next((o for o in occ if o["id"]==eid),None)
        emitted.setdefault(eid,{"id":eid,"kind":kind,"start":row["start"] if row else c["start"],"end":row["end"] if row else c["end"],
            "onset": row.get("onset") if row else c["start"],"actor_track":"selected","supported":False})
        a=judged.get(c["id"],{})
        emitted[eid]["supported"] |= all(a.get(k) is True for k in ("supported","actor_correct","evidence_valid"))
    refs=[r for r in clip["events"] if r["boundary_resolvable"] and not r["left_censored"] and not r["right_censored"] and r["kind"] in ACTION_KINDS]
    predictions=list(emitted.values())
    supported=[p for p in predictions if p["supported"]]
    matched,_=match_events(refs,predictions)
    # Short fragments of a long Q/recall have low IoU. They are nevertheless
    # duplicates when multiple descriptions fall inside one labeled occurrence.
    related={}
    for p in predictions:
        candidates=[r for r in clip["events"] if r['kind'] in ACTION_KINDS and max(0,min(p["end"],r["end"])-max(p["start"],r["start"]))
                    >= .5*max(.001,p["end"]-p["start"])]
        if candidates:
            r=next((r for r in candidates if matched.get(r['id'])==p['id']),None) or max(candidates,key=lambda r:(temporal_iou(p,r),-abs((r.get("onset") or r["start"])-(p["onset"] if p['onset'] is not None else p['start']))))
            related.setdefault(r["id"],[]).append(p)
    duplicate_kinds=Counter()
    for rid,rows in related.items():
        representative=matched.get(rid) or min(rows,key=lambda p:(p['start'],p['id']))['id']
        duplicate_kinds.update(p['kind'] for p in rows if p['id']!=representative)
    duplicates=sum(duplicate_kinds.values())
    trusted,_=match_events(refs,supported)
    pi={p["id"]:p for p in predictions};ri={r["id"]:r for r in refs}
    metrics["event_metrics"]={"reference_count":len(refs),"emitted_count":len(predictions),"matched":len(matched),
        "supported_matched":len(trusted),"missed":len(refs)-len(trusted),"duplicates":duplicates,
        "censored_reference_count":len(clip["events"])-len(refs),
        'censored_prediction_count':sum(p['onset'] is None for p in predictions),
        "onset_errors":[abs(pi[p]["onset"]-ri[r]["onset"]) for r,p in matched.items()],
        "end_errors":[abs(pi[p]["end"]-ri[r]["end"]) for r,p in matched.items()],
        "noncounter_claims":sum(c["claims"][0]["kind"] in PILOT_CRITERIA for c in caps),
        "reviewed_noncounter":sum(c["id"] in judged and c["claims"][0]["kind"] in PILOT_CRITERIA for c in caps)}
    metrics["event_metrics"]["supported_noncounter"]=sum(judged.get(c["id"],{}).get("supported") is True and c["claims"][0]["kind"] in PILOT_CRITERIA for c in caps)
    targets=[c for c in caps if c["claims"][0].get("target_binding_id")]
    metrics["event_metrics"].update(target_claim_count=len(targets),target_assessed=sum(c["id"] in judged for c in targets),
        target_correct=sum(judged.get(c["id"],{}).get("target_correct") is True for c in targets))
    metrics['event_metrics']['by_kind']={k:{'reference_count':sum(r['kind']==k for r in refs),
        'emitted_count':sum(p['kind']==k for p in predictions),'matched':sum(ri[r]['kind']==k for r in matched),
        'supported_matched':sum(ri[r]['kind']==k for r in trusted),'duplicates':duplicate_kinds[k],
        'missed':sum(r['kind']==k for r in refs)-sum(ri[r]['kind']==k for r in trusted)} for k in sorted(ACTION_KINDS)}
    return metrics


def aggregate(results, clips):
    from . import caption_benchmark
    out=caption_benchmark.aggregate(results,clips,kinds=CLAIM_KINDS)
    totals=Counter()
    for r in results:totals.update({k:v for k,v in r["event_metrics"].items() if type(v) is int})
    recall=totals["supported_matched"]/totals["reference_count"] if totals["reference_count"] else None
    duplicate_rate=totals["duplicates"]/totals["emitted_count"] if totals["emitted_count"] else None
    adequacy=reference_adequacy(clips)
    sufficient=all(min(out["by_kind"][k].get("prediction_count",0),out["by_kind"][k].get("reference_count",0),out["by_kind"][k].get("reviewed",0))>=5 for k in REQUIRED_CLAIMS)
    out["event_metrics"]={**totals,"recall":recall,"duplicate_rate":duplicate_rate,
        "supported_fraction":totals["supported_noncounter"]/totals["reviewed_noncounter"] if totals["reviewed_noncounter"] else None,
        "target_binding_accuracy":totals["target_correct"]/totals["target_assessed"] if totals["target_assessed"] else None,
        "onset_errors":[v for r in results for v in r["event_metrics"]["onset_errors"]],
        "end_errors":[v for r in results for v in r["event_metrics"]["end_errors"]]}
    out['event_metrics']['by_kind']={}
    for kind in sorted(ACTION_KINDS):
        counts=Counter()
        for r in results:counts.update(r['event_metrics']['by_kind'][kind])
        out['event_metrics']['by_kind'][kind]={**counts,'recall':counts['supported_matched']/counts['reference_count'] if counts['reference_count'] else None,
            'precision':counts['matched']/counts['emitted_count'] if counts['emitted_count'] else None,
            'duplicate_rate':counts['duplicates']/counts['emitted_count'] if counts['emitted_count'] else None}
    out["reference_adequacy"]=adequacy
    out["missing_event_cases"]=adequacy["missing_cases"]
    out["event_stage_passed"]=bool(totals["noncounter_claims"]>=20 and totals["reviewed_noncounter"]==totals["noncounter_claims"]
        and sufficient and not adequacy["missing_cases"] and not adequacy["action_kind_conflicts"]
        and recall is not None and recall>=.8 and duplicate_rate is not None and duplicate_rate<=.05
        and totals["supported_noncounter"]/totals["noncounter_claims"]>=.95
        and out["reviewed"]==out["prediction_count"]
        and out["supported_fraction"] is not None and out["supported_fraction"]>=.95
        and out["actor_binding_accuracy"]>=.95 and out["evidence_valid_fraction"]==1
        and sum(r["reviewed_evidence_valid"] for r in out["by_kind"].values())==out["prediction_count"]
        and totals["target_assessed"]==totals["target_claim_count"] and totals["target_assessed"]>0
        and totals["target_correct"]/totals["target_assessed"]>=.95
        and all(out["by_kind"][k]["supported_fraction"]>=.95 and out["by_kind"][k]["precision"]>=.95 for k in REQUIRED_CLAIMS))
    out["broader_pilot_passed"]=out["passed"] and out["event_stage_passed"]
    out["passed"]=out["event_stage_passed"]
    return out


def experiment(store,config,*,labels=None,arm="event-v2",initialize_labels=False,draft=False,freeze=False,infer=False,
               prepare_only=False,preflight=False,collect=False,icon_references=None,action_profile=None,runner=None,runtime_recipe=None):
    directory=Path(labels) if labels else store.path("benchmarks/caption-events/v1")
    if collect:
        raise ValueError("Use caption_batch collect --report to preserve the execution report")
    profile=load_profile(action_profile)
    if initialize_labels:return initialize(store,directory,profile)
    data=read_json(directory/"events.json")
    clips=validate(store,data,splits=("development",) if freeze else ("development","held_out") if not draft and not prepare_only else ())
    from .action_icons import AbilityIdentity
    reference=Path(icon_references) if icon_references else Path(config.asset_cache)/"manifest.json"
    signature={"schema_version":2,"source":data["source_sha256"],"arm":arm,"model":read_json(LOCK)["models"]["intake"],
        "infer":bool(infer or freeze),
        "profile":profile,"anchors":anchor_profile(profile["anchor_profile"]),"masks":config.masks,"policy":EVENT_POLICY,"rubric":RUBRIC,"rubric_version":RUBRIC_VERSION,
        "icons":AbilityIdentity(reference).signature if reference.exists() else None,
        "implementation":implementation_hash(COMPONENTS+["caption_event_benchmark.py","caption_review.py"]),
        "development":[c for c in clips if c["split"]=="development"]}
    frozen=directory/("freeze-"+arm+".json")
    if freeze:
        receipts=[read_json(p) for p in store.path("experiments/caption-events").glob("*/assessed-report.json")]
        expected={c["id"] for c in clips if c["split"]=="development"}
        passed=[r for r in receipts if r.get("draft") is True and r.get("signature")==signature
                and set(r.get("clips",{}))==expected and (r.get("quality_metrics") or {}).get("event_stage_passed") is True]
        valid=[]
        development=[c for c in clips if c["split"]=="development"]
        for receipt in passed:
            if any(not (Path(r["root"])/"caption_coverage.json").exists() or
                   read_json(Path(r["root"])/"caption_coverage.json")["key"]!=r["generation_key"] or
                   r.get('prediction_fingerprint')!=prediction_fingerprint(Path(r['root'])) or
                   list((Path(r["root"])/"reviews").glob("*.json")) for r in receipt["clips"].values()):
                continue
            fresh=aggregate([score(c,receipt["clips"][c["id"]]["root"]) for c in development],development)
            if fresh["event_stage_passed"]:valid.append(receipt)
        if not valid:
            raise ValueError("Matching independently assessed development event gate must pass before freeze")
        runtime=valid[0].get("runtime_recipe")
        if not runtime:raise ValueError("Development runtime provenance is missing")
        if frozen.exists() and read_json(frozen)["key"]!=digest(signature):raise ValueError("Use a new frozen benchmark revision")
        atomic_json(frozen,{"key":digest(signature),"choices":signature,"runtime_recipe":runtime,"created_at":now()})
        return {"status":"development_frozen","freeze":str(frozen)}
    if not draft and not prepare_only and (not frozen.exists() or read_json(frozen)["key"]!=digest(signature)):
        raise ValueError("Freeze unchanged development choices before held-out inference")
    root=store.path("experiments/caption-events/"+digest([signature,draft,preflight,prepare_only])[:20]);root.mkdir(parents=True,exist_ok=True)
    path=root/"report.json"
    report=read_json(path) if path.exists() else {"schema_version":2,"arm":arm,"draft":draft,"signature":signature,
        "status":"running_experimental","report":str(path),"clips":{},"quality_metrics":None,"production_evidence_modified":False,
        "preflight":preflight}
    report.setdefault("implementation_root",str(Path(__file__).parent.resolve()))
    if runtime_recipe is not None:
        if report.get("runtime_recipe") not in (None,runtime_recipe):raise ValueError("Preflight/resume runtime mismatch")
        report["runtime_recipe"]=runtime_recipe
    snapshot=root/"reference-snapshot.json"
    if snapshot.exists() and digest(read_json(snapshot))!=digest(data):
        raise ValueError("Source references changed since prediction; use a new benchmark revision")
    atomic_json(snapshot,data)
    method=directory/"annotation-method.json"
    if method.exists() and not (root/"reference-method.json").exists():atomic_json(root/"reference-method.json",read_json(method))
    selected=[c for c in clips if draft and c["split"]=="development" or not draft and c["split"]=="held_out"]
    if prepare_only:selected=clips
    if preflight:
        selected=copy.deepcopy(selected[:1])
        for c in selected:
            center=(c["start"]+c["end"])/2
            c.update(start=center-2,end=center+2,verified=False,coverage_exhaustive=False)
    owned_runner=False
    try:
        from .action_pipeline import prepare_inputs
        from .caption_segments import POLICY
        for clip in selected:
            store.check_stop()
            if digest(read_json(directory/"events.json"))!=digest(data):raise ValueError("Reference labels changed during run")
            if implementation_hash(COMPONENTS+["caption_event_benchmark.py","caption_review.py"])!=signature["implementation"]:
                raise ValueError("Implementation changed during run; resume from the immutable job snapshot")
            previous=report["clips"].get(clip["id"])
            if prepare_only:
                audit=prepare_inputs(store,config,clip["start"],clip["end"],profile,POLICY,namespace="caption-events-source",max_duration=None)
                report["clips"][clip["id"]]={"split":clip["split"],"root":audit["root"],"status":"source_prepared"}
                atomic_json(path,report)
                continue
            else:
                logging.info("Caption occurrence %s %s: %.3f–%.3fs",arm,clip["id"],clip["start"],clip["end"])
                prior=Path(previous["root"]) if previous else None
                outputs=("captions","events","caption_clips","states","normalized_observations","caption_omissions")
                if arm=="event-v2":outputs+=("scene_occurrences",)
                complete=(prior is not None and (prior/"report.json").exists() and (prior/"caption_coverage.json").exists()
                          and read_json(prior/"report.json").get("status")=="completed_experimental"
                          and read_json(prior/"caption_coverage.json")["key"]==previous.get("generation_key")
                          and previous.get('prediction_fingerprint')==prediction_fingerprint(prior)
                          and all((prior/(table+".jsonl")).exists() for table in outputs))
                if complete:
                    result={"root":str(prior),"resource_metrics":previous["resources"]}
                else:
                    if infer:
                        from .local_models import LocalVLRunner
                        from .action_v3 import cache_runtime
                        import os
                        if runner is None:
                            runner=LocalVLRunner(store,config.vl_python or os.environ.get("LEAGUE_VL_PYTHON",""),config.asset_cache,image_limit=8)
                            owned_runner=True
                        runtime=cache_runtime(runner.identity)
                        expected_runtime=(read_json(frozen).get("runtime_recipe") if not draft else report.get("runtime_recipe"))
                        if expected_runtime is not None and runtime!=expected_runtime:
                            raise ValueError("Model/GPU-class/software runtime differs from frozen or resumed development recipe")
                        report["runtime_recipe"]=runtime
                    result=caption_pilot(store,config,clip["start"],clip["end"],infer=infer,runner=runner,arm=arm,
                                         icon_references=icon_references,action_profile=action_profile)
            pilot=Path(result["root"])
            if list((pilot/"reviews").glob("*.json")):raise ValueError("Corrected predictions cannot establish benchmark acceptance")
            report["clips"][clip["id"]]={"split":clip["split"],"root":str(pilot),"resources":result["resource_metrics"],
                "generation_key":read_json(pilot/"caption_coverage.json")["key"],
                'prediction_fingerprint':prediction_fingerprint(pilot),
                "metrics":score(clip,pilot) if clip["verified"] and clip["coverage_exhaustive"] else None}
            atomic_json(path,report)
        if not prepare_only:
            ready=[report["clips"][c["id"]]["metrics"] for c in selected]
            report["quality_metrics"]=aggregate(ready,selected) if ready and all(m is not None for m in ready) else None
        report["status"]="source_prepared" if prepare_only else "development_complete" if draft else "validation_passed" if report["quality_metrics"] and report["quality_metrics"]["passed"] else "validation_pending_or_failed"
    except BaseException as exc:
        report.update(status="interrupted_experimental" if store.stopping else "failed_experimental",error=str(exc))
        raise
    finally:
        if owned_runner and runner is not None:runner.close()
        atomic_json(path,report)
    return report
