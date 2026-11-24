"""Score separate physical and activation tracks without inventing complete references."""
from collections import Counter
from pathlib import Path
from statistics import mean

from .core import atomic_json, finite, read_json, read_rows

KINDS=("ability_activation","movement","basic_attack","ability_cast","item_use","recall","minion_last_hit","base_arrival")


def validate_reference(label,clip,sha,require_complete=False):
    if label.get("clip_id")!=clip["id"] or label.get("source_sha256")!=sha:
        raise ValueError("Source reference identity mismatch")
    review=label.get("review",{})
    if not review.get("reviewer") or review.get("prediction_outputs_seen") is not False or not review.get("inspected_sheets"):
        raise ValueError("Independent source-review provenance required")
    if label.get("review_status")!="source_reviewed_model_draft" or review.get("human_reviewed") is not False:
        raise ValueError("Keep source-agent draft status explicit")
    scope=label.get("scope",{})
    if set(scope)!={"hud_activations","physical_actions"} or any(x not in {"partial","complete_observable"} for x in scope.values()):
        raise ValueError("Declare physical and HUD reference completeness separately")
    if require_complete and scope["hud_activations"]!="complete_observable":
        raise ValueError("Complete observable activation references before held-out evaluation")
    seen=set()
    for event in label.get("events",[]):
        if event.get("id") in seen or not event.get("id") or event.get("kind") not in set(KINDS)|{"hud_activation"}:
            raise ValueError("Invalid or duplicate reference event")
        seen.add(event["id"])
        if not all(finite(event.get(k)) for k in ("start","end")) or not clip["start"]<=event["start"]<=event["end"]<=clip["end"]:
            raise ValueError("Reference event outside its source context")
        if event.get("actor") not in {"selected","other","unknown"} or event.get("slot") not in {None,"Q","W","E","R"}:
            raise ValueError("Reference actor or ability slot invalid")
        if not event.get("evidence"):
            raise ValueError("Reference cue evidence required")
        if any(type(event.get(key)) is not bool for key in ("left_censored","right_censored","boundary_resolvable")):
            raise ValueError("Reference boundaries require explicit uncertainty flags")
        if any(not finite(cue.get("time")) or not clip["start"]<=cue["time"]<=clip["end"] or not cue.get("cue")
               for cue in event["evidence"]):
            raise ValueError("Reference cues require in-source times and visible descriptions")
        if event["kind"]=="hud_activation" and (event.get("target") is not None or event.get("outcome") is not None):
            raise ValueError("HUD evidence cannot establish target or physical outcome")
        for key in ("target","outcome","ability_name"):
            if event.get(key) is not None and not event.get(key+"_evidence"):
                raise ValueError("Asserted reference fields need independent source evidence")
    for row in label.get("unknown_intervals",[]):
        if (row.get("scope") not in {"all","hud_activations","physical_actions"} or not row.get("reason")
            or not all(finite(row.get(k)) for k in ("start","end")) or not clip["start"]<=row["start"]<=row["end"]<=clip["end"]):
            raise ValueError("Invalid independently unknown reference interval")
    return True


def kind(row):
    return "ability_activation" if row["kind"]=="hud_activation" else row["kind"]


def fit(reference,prediction):
    if kind(reference)!=kind(prediction):
        return None
    if kind(reference) in {"ability_activation","minion_last_hit","base_arrival"} or reference["start"]==reference["end"]:
        error=abs(reference["end"]-prediction["end"])
        return 1-error/.25 if error<=.25+1e-9 else None
    union=max(reference["end"],prediction["end"])-min(reference["start"],prediction["start"])
    overlap=max(0,min(reference["end"],prediction["end"])-max(reference["start"],prediction["start"]))
    iou=overlap/union if union else 1
    return iou if iou>=.5 else None


def matches(references,predictions):
    edges={}
    for i,p in enumerate(predictions):
        candidates=[(value,j) for j,r in enumerate(references) if (value:=fit(r,p)) is not None]
        edges[i]=[j for _,j in sorted(candidates,reverse=True)]
    assigned={}
    def augment(i,visited):
        for j in edges[i]:
            if j in visited:continue
            visited.add(j)
            if j not in assigned or augment(assigned[j],visited):
                assigned[j]=i;return True
        return False
    for i in sorted(edges,key=lambda x:(len(edges[x]),x)):
        augment(i,set())
    return {p:r for r,p in assigned.items()},set(i for i in edges if edges[i])-set(assigned.values())


def ratio(a,b):
    return a/b if b else None


def role(row):
    if "actor" in row and row["actor"] in {"selected","other","unknown"}:
        return row["actor"]
    return "selected" if row.get("owner_id")=="selected" else "other" if (row.get("actor_track") or "").startswith("other-") else "unknown"


def score_clip(label,predictions,physical_capability=True):
    categories={};attribution={"actor":Counter(),"slot":Counter()}
    for category in KINDS:
        track="hud_activations" if category=="ability_activation" else "physical_actions"
        def eligible(row):
            return not any(x["scope"] in {"all",track} and max(row["start"],x["start"])<=min(row["end"],x["end"])
                           for x in label.get("unknown_intervals",[]))
        refs=[r for r in label["events"] if kind(r)==category and eligible(r)]
        preds=[p for p in predictions if kind(p)==category and eligible(p)]
        pairs,duplicates=matches(refs,preds)
        complete=label["scope"][track]=="complete_observable" and (physical_capability or track=="hud_activations")
        errors=[abs(refs[r]["end"]-preds[p]["end"]) for p,r in pairs.items()
                if refs[r].get("boundary_resolvable",False) and not refs[r].get("right_censored")]
        categories[category]={"reference_events":len(refs),"predictions":len(preds),"matched":len(pairs),
            "duplicates":len(duplicates),"complete_reference_scope":complete,"tested":bool(refs) and complete,
            "precision":ratio(len(pairs),len(preds)) if complete else None,
            "recall":ratio(len(pairs),len(refs)) if complete else None,
            "f1":ratio(2*len(pairs),len(refs)+len(preds)) if complete and refs else None,
            "listed_reference_recovery":ratio(len(pairs),len(refs)),
            "timing_mae_seconds":mean(errors) if errors else None,"timing_matches":len(errors),
            "predictions_in_unknown_scope":sum(kind(p)==category and not eligible(p) for p in predictions)}
        for p,r in pairs.items():
            for name,expected,actual in (("actor",role(refs[r]),role(preds[p])),("slot",refs[r].get("slot"),preds[p].get("slot_id"))):
                if expected in (None,"unknown"):continue
                attribution[name]["eligible_event_matches"]+=1
                if actual in (None,"unknown"):
                    attribution[name]["unresolved"]+=1
                else:
                    attribution[name]["resolved"]+=1
                    attribution[name]["correct" if actual==expected else "incorrect"]+=1
    return {"categories":categories,"attribution":{name:{**dict(count),"accuracy":ratio(count["correct"],count["resolved"]),
            "coverage":ratio(count["resolved"],count["eligible_event_matches"])} for name,count in attribution.items()},
            "reference_status":"independent_source_agent_draft","visible_support_audit":"pending",
            "target_outcome_assessment":"Not inferred from null reference fields; independent assertion review remains required."}


def score_campaign(root,references,split):
    from .scene_ablation import ARMS,assert_frozen,validate_run
    root,references=Path(root),Path(references);study=read_json(root/"study.json")
    if split=="held_out":assert_frozen(root,study)
    rows=[]
    for arm in ARMS:
        run=read_json(root/"results"/split/arm/"report.json")
        validate_run(run,study,arm,split)
        for result in run["clips"]:
            clip=next(c for c in study["clips"] if c["id"]==result["clip_id"])
            label=read_json(references/split/(clip["id"]+".json"))
            validate_reference(label,clip,study["source"]["sha256"])
            rows.append({"arm":arm,"clip_id":clip["id"],**score_clip(label,read_rows(Path(result["events"])),arm!="hud-only")})
    summaries={}
    for arm in ARMS:
        summaries[arm]={}
        for category in KINDS:
            eligible=[row["categories"][category] for row in rows if row["arm"]==arm and row["categories"][category]["complete_reference_scope"]]
            n=sum(row["reference_events"] for row in eligible);p=sum(row["predictions"] for row in eligible);m=sum(row["matched"] for row in eligible)
            summaries[arm][category]={"eligible_clips":len(eligible),"reference_events":n,"predictions":p,"matched":m,
                "precision":ratio(m,p),"recall":ratio(m,n),"f1":ratio(2*m,n+p) if n else None,
                "duplicates":sum(row["duplicates"] for row in eligible),"tested":bool(n)}
    report={"split":split,"evaluation_scope":study["evaluation_scope"],"clips":rows,"arm_summary":summaries,
        "conclusion_status":"draft_reference_comparison_pending_visible_support_audit",
        "denominator_policy":"Partial references yield listed-reference recovery only; full precision/recall require complete observable scope. Unknown intervals and missing categories remain explicit.",
        "timing_policy":"Activation error is measured against visible cooldown onset; physical cast interval timing is separate. Scene-inferred activation starts need not equal a cooldown start.",
        "generalization_claim":False}
    atomic_json(root/(split+"-scorecard.json"),report)
    return report
