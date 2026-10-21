"""Independent video-only input construction and reference-quality-stratified scoring."""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import replace

from .core import Store, atomic_json, digest, finite, read_json, uid, consistent_review, implementation_hash
from .inference import InputBudgetError, ScopedResolver, infer_with_knowledge
from .media import Media
from .prompts import solver_messages


def uniform_times(start, end, limit, fps=1.0):
    count = min(limit, max(2, math.ceil((end-start)*fps)))
    # End is exclusive; keep the last sample within the source duration.
    last = max(start, end-1/60)
    return [start + (last-start)*i/(count-1) for i in range(count)]


def evaluation_input(question, media, config):
    """Replaceable v1 protocol: uniform frames over the question interval, no oracle selection."""
    route = next(r for r in config.routes if r.name == "full_context")
    route = replace(route, max_pixels=config.solver_max_pixels, model="thinking")
    minimum = question.get("coverage_requirements", {}).get("minimum_fps") or 1.0
    frames = [media.frame(t, route) for t in uniform_times(*question["interval"], config.solver_max_frames, fps=minimum)]
    messages = solver_messages(question, frames)
    fps = ((len(frames)-1)/(frames[-1]["time"]-frames[0]["time"])) if frames[-1]["time"] > frames[0]["time"] else 1
    return route, frames, messages, fps


def evidence_audit(question, frames):
    sampled = [f["time"] for f in frames]
    distances = [min(abs(t-s) for s in sampled) for t in question["evidence_times"]] if sampled else []
    # Temporal presence is measurable here; legibility requires review of processed crops.
    return {"required_times": question["evidence_times"], "nearest_sample_distances": distances,
            "all_required_times_within_0_5s": bool(distances) and max(distances) <= .5,
            "legibility": "unverified", "note": "Temporal proximity does not establish readable evidence"}


def solver_request_key(store, config, question):
    """Reference edits invalidate scores, while identical solver inputs reuse original predictions."""
    manifest = read_json(store.path("manifest.json"))
    pin = manifest.get("model_revisions", {}).get("thinking", config.models["thinking"])
    context_route = next(r for r in config.routes if r.name == "full_context")
    return uid("solver", {"source": manifest["source"]["sha256"],
        "model": {k: v for k, v in pin.items() if k != "snapshot"},
        "runtime": {k: v for k, v in manifest.get("inference_runtime", {}).items() if k != "device"}, "protocol": "single_context_uniform_video_v1",
        "question": {k: question[k] for k in ("prompt", "answer_type", "interval", "knowledge_category")},
        "minimum_fps": question.get("coverage_requirements", {}).get("minimum_fps"),
        "max_frames": config.solver_max_frames, "max_pixels": config.solver_max_pixels,
        "context": config.max_context, "output": config.thinking_output_tokens,
        "lookups": config.knowledge_queries, "seed": config.seed, "masks": config.masks, "box": context_route.box,
        "implementation": implementation_hash(["core.py", "media.py", "inference.py", "prompts.py", "knowledge.py", "evaluation.py"])})


@consistent_review
def evaluate(store: Store, config, runner, resolver):
    media = Media(store, config)
    questions = store.effective(store.rows("questions"), "questions")
    key = store.stage_key("evaluate", config, ["questions"])
    predictions = []
    for question in questions:
        store.check_stop()
        if question["review_status"] == "rejected":
            continue
        task_id = solver_request_key(store, config, question)
        path = store.path(f"solver-tasks/{task_id}.json")
        if path.exists():
            cached = read_json(path)
            predictions.append({**cached, "question_id": question["id"], "review_status": question["review_status"],
                                "evaluation_key": key, "cache_reused": True,
                                "original_evaluation_key": cached["evaluation_key"]})
            continue
        route, frames, messages, fps = evaluation_input(question, media, config)
        directory = store.path("solver-inference/" + task_id)
        # Persist exactly the allowlisted textual payload, never the question record.
        atomic_json(directory / "solver-input.json", {"messages": messages, "frames": frames,
                    "protocol": "single_context_uniform_video_v1"})
        prediction = {"id": task_id, "question_id": question["id"], "evaluation_key": key,
                      "review_status": question["review_status"], "protocol": "single_context_uniform_video_v1", "cache_reused": False,
                      "runtime": media.manifest.get("inference_runtime", {}),
                      "device": media.manifest.get("inference_runtime", {}).get("device"),
                      "frames": frames, "input_coverage": evidence_audit(question, frames)}
        try:
            result = infer_with_knowledge(runner, ScopedResolver(resolver, question["knowledge_category"]),
                messages, frames, route, directory, max_queries=config.knowledge_queries, fps=fps)
            output = result.get("answer")
            if result["status"] != "answered" or not isinstance(output, dict) or type(output.get("abstain")) is not bool:
                raise ValueError("Solver must return answer/abstain/evidence JSON")
            if not output["abstain"] and "answer" not in output:
                raise ValueError("Missing typed answer")
            if not isinstance(output.get("evidence", []), list):
                raise ValueError("Invalid solver evidence")
            if not output["abstain"] and not output.get("evidence"):
                raise ValueError("A non-abstaining answer requires source evidence")
            for interval in output.get("evidence", []):
                if (not finite(interval.get("start")) or not finite(interval.get("end"))
                        or not question["interval"][0] <= interval["start"] <= interval["end"] <= question["interval"][1]):
                    raise ValueError("Solver evidence outside declared interval")
            prediction.update(status="ok", output=output, metrics=result["metrics"],
                              knowledge_queries=result["knowledge_queries"])
        except (ValueError, InputBudgetError) as exc:
            from .observers import _spent
            prediction.update(status="unresolved", error=str(exc), output={"answer": None, "abstain": True, "evidence": []},
                              metrics=_spent(directory))
        atomic_json(path, prediction)
        predictions.append(prediction)
    store.write("predictions", predictions)
    store.commit_stage("evaluate", key, questions=len(predictions))
    return {"questions": len(predictions), "unresolved": sum(p["status"] != "ok" for p in predictions)}


def exact_answer(expected, actual):
    # JSON numeric 10 and 10.0 represent the same displayed number; booleans do not.
    if finite(expected) and finite(actual):
        return expected == actual
    if type(expected) is not type(actual):
        return False
    if isinstance(expected, dict):
        return expected.keys() == actual.keys() and all(exact_answer(v, actual[k]) for k, v in expected.items())
    if isinstance(expected, list):
        return len(expected) == len(actual) and all(exact_answer(a, b) for a, b in zip(expected, actual))
    return expected == actual


def interval_iou(a, b):
    intersection = max(0, min(a[1], b[1])-max(a[0], b[0]))
    union = max(a[1], b[1])-min(a[0], b[0])
    return intersection/union if union else 1.0


def temporal_metrics(reference, predicted, tolerance=1.0):
    """Maximum-cardinality one-to-one matching, prioritizing highest overlap."""
    edges = {}
    for pi, p in enumerate(predicted):
        possibilities = []
        for ri, r in enumerate(reference):
            point = r[0] == r[1]
            distance = max(abs(p[0]-r[0]), abs(p[1]-r[0])) if point else abs(p[0]-r[0])
            score = max(0, 1-distance/max(tolerance, 1e-9)) if point else interval_iou(r, p)
            if (point and distance <= tolerance) or (not point and score >= .5):
                possibilities.append((score, ri))
        edges[pi] = [ri for _, ri in sorted(possibilities, reverse=True)]
    assigned = {}
    def assign(pi, visited):
        for ri in edges[pi]:
            if ri in visited:
                continue
            visited.add(ri)
            if ri not in assigned or assign(assigned[ri], visited):
                assigned[ri] = pi
                return True
        return False
    for pi in edges:
        assign(pi, set())
    matched = len(assigned)
    durations = [interval_iou(reference[ri], predicted[pi]) for ri, pi in assigned.items()
                 if reference[ri][0] != reference[ri][1]]
    duration_count = sum(r[0] != r[1] for r in reference)
    errors = [abs(reference[ri][0]-predicted[pi][0]) for ri, pi in assigned.items()]
    return {"precision": matched/len(predicted) if predicted else 0.0,
            "recall": matched/len(reference) if reference else 0.0,
            "matched": matched, "predicted": len(predicted), "reference": len(reference),
            "mean_iou": sum(durations)/duration_count if duration_count else None,
            "mean_iou_matched": sum(durations)/len(durations) if durations else None,
            "mean_timestamp_error": sum(errors)/len(errors) if errors else None}


@consistent_review
def report(store: Store, config, gpu_hour_rate=None):
    questions = store.effective(store.rows("questions"), "questions")
    predictions = {p["question_id"]: p for p in store.rows("predictions")}
    strata = {}
    scores = []
    for status in ("human-verified", "unreviewed"):
        eligible = [q for q in questions if q["review_status"] == status]
        correct = abstentions = evaluated = unsupported = failures = 0
        categories = {}
        temporal = []
        for q in eligible:
            prediction = predictions.get(q["id"])
            if not prediction:
                continue
            evaluated += 1
            failures += prediction["status"] != "ok"
            output = prediction["output"]
            abstain = output.get("abstain", True)
            abstentions += abstain
            hit = not abstain and exact_answer(q["answer"], output.get("answer"))
            correct += hit
            unsupported += not abstain and not hit
            reference = q.get("evidence_intervals", [[t, t] for t in q["evidence_times"]])
            predicted = [[e["start"], e["end"]] for e in output.get("evidence", [])]
            temporal_score = temporal_metrics(reference, predicted)
            reference_boundaries = [[t, t] for t in sorted({v for interval in reference for v in interval})]
            predicted_boundaries = [[t, t] for t in sorted({v for interval in predicted for v in interval})]
            temporal_score["boundary_metrics"] = temporal_metrics(reference_boundaries, predicted_boundaries)
            temporal.append(temporal_score)
            group = categories.setdefault(q["capability"], {"evaluated": 0, "correct": 0})
            group["evaluated"] += 1
            group["correct"] += hit
            scores.append({"id": uid("score", q["id"]), "question_id": q["id"], "reference_status": status,
                           "exact_match": hit, "abstain": abstain, "temporal": temporal_score,
                           "evidence_temporally_sampled": prediction["input_coverage"]["all_required_times_within_0_5s"]})
        matched = sum(t["matched"] for t in temporal)
        predicted_count = sum(t["predicted"] for t in temporal)
        reference_count = sum(t["reference"] for t in temporal)
        ious = [t["mean_iou"] for t in temporal if t["mean_iou"] is not None]
        errors = [t["mean_timestamp_error"] for t in temporal if t["mean_timestamp_error"] is not None]
        strata[status] = {"eligible": len(eligible), "evaluated": evaluated, "exact_match": correct/evaluated if evaluated else None,
                          "abstention_rate": abstentions/evaluated if evaluated else None,
                          "generation_failure_rate": failures/evaluated if evaluated else None,
                          "reference_disagreement_rate": unsupported/evaluated if evaluated else None,
                          "coverage": evaluated/len(eligible) if eligible else None,
                          "by_capability": categories,
                          "temporal_summary": {"precision": matched/predicted_count if predicted_count else None,
                              "recall": matched/reference_count if reference_count else None,
                              "mean_iou": sum(ious)/len(ious) if ious else None,
                              "mean_timestamp_error": sum(errors)/len(errors) if errors else None},
                          "temporal_evidence": temporal}
    tasks = [read_json(p) for p in store.path("tasks").glob("*.json")
             if read_json(p).get("run_key") == store.fingerprint(config)]
    metrics = [t.get("metrics", {}) for t in tasks] + [p.get("metrics", {}) for p in predictions.values()]
    wall = sum(m.get("wall_seconds", 0) for m in metrics)
    manifest = read_json(store.path("manifest.json"))
    hours = (manifest["interval"][1]-manifest["interval"][0])/3600
    stage_times = {p.stem: read_json(p) for p in store.path("timings").glob("*.json")}
    pipeline_wall = stage_times.get("run", {}).get("wall_seconds")
    if pipeline_wall is None:
        pipeline_wall = sum(stage_times[s]["wall_seconds"] for s in ("prepare", "annotate", "fuse", "generate-qa", "evaluate") if s in stage_times)
    effective_events = store.effective(store.rows("events"), "events")
    reviewed = [e for e in effective_events if e.get("review_status") in {"human-verified", "rejected"}]
    known_ids = {name: {r["id"] for r in store.rows(name)} for name in ("observations", "events", "questions")}
    audit_roots = [store.path("inference/"+t["id"]) for t in tasks] + [
        store.path("solver-inference/"+p["id"]) for p in predictions.values()]
    audits = [read_json(path) for root in audit_roots for path in root.rglob("input-audit.json")]
    result = {"reference_strata": strata, "correction_revision": len(store.corrections()),
              "interpretation": {"human-verified": "Accuracy against human-verified reference answers",
                                 "unreviewed": "Agreement with provisional automatic references; not benchmark accuracy"},
              "efficiency": {"measured_inference_seconds": wall,
                 "pipeline_wall_seconds": pipeline_wall or None,
                 "pipeline_seconds_per_game_hour": pipeline_wall/hours if pipeline_wall else None,
                 "inference_seconds_per_game_hour": wall/hours, "game_hours": hours,
                 "peak_gpu_memory_bytes": max((m.get("peak_gpu_memory_bytes", 0) for m in metrics), default=0),
                 "annotation_devices": sorted({t.get("device") for t in tasks if t.get("device")}),
                 "solver_devices": sorted({p.get("device") for p in predictions.values() if p.get("device")}),
                 "visual_tokens": sum(m.get("visual_tokens", 0) for m in metrics),
                 "processed_frame_presentations": sum(len(a.get("processed_inputs", [])) for a in audits),
                 "unique_source_timestamps": len({f["time"] for a in audits for f in a["frames"]}),
                 "processed_resolutions": sorted({tuple(p["size"]) for a in audits for p in a.get("processed_inputs", [])}),
                 "output_tokens": sum(m.get("output_tokens", 0) for m in metrics),
                 "solver_cache_hits": sum(p.get("cache_reused", False) for p in predictions.values()),
                 "new_solver_inference_seconds": sum(p.get("metrics", {}).get("wall_seconds", 0)
                     for p in predictions.values() if not p.get("cache_reused", False)),
                 "review_visual_tokens": sum(t.get("metrics", {}).get("visual_tokens", 0) for t in tasks if t["pass"] > 0),
                 "gpu_cost": (pipeline_wall or wall)/3600*gpu_hour_rate if gpu_hour_rate is not None else None,
                 "cost_basis": "pipeline_elapsed" if pipeline_wall else "inference_time_only",
                 "gpu_hour_rate": gpu_hour_rate,
                 "note": "Inference timings exclude decode/model loading; stage wall times are reported separately"},
              "questions_by_capability": dict(Counter(q["capability"] for q in questions if q["review_status"] != "rejected")),
              "hallucination_measurement": "Reference disagreements are a proxy; unsupported-claim labels require reviewed evidence",
              "event_candidate_review": {"reviewed": len(reviewed),
                  "verified": sum(e["review_status"] == "human-verified" for e in reviewed),
                  "rejected": sum(e["review_status"] == "rejected" for e in reviewed),
                  "exhaustive_event_recall": None,
                  "limitation": "Approving candidates alone does not establish missed-event recall over source video"},
              "orphaned_corrections": [c["id"] for c in store.corrections()
                  if c["record_id"] not in known_ids.get(c["target"], set())],
              "evaluation_inputs": [{"question_id": q["id"], "temporal_span_seconds": q["interval"][1]-q["interval"][0],
                                      "sampled_frame_count": len(predictions[q["id"]].get("frames", [])),
                                      "coverage": predictions[q["id"]]["input_coverage"]}
                                     for q in questions if q["id"] in predictions],
              "stages": {p.stem: read_json(p) for p in store.path("stages").glob("*.json")},
              "stage_wall_times": stage_times}
    store.write("scores", scores)
    atomic_json(store.path("report.json"), result)
    store.commit_stage("report", store.stage_key("report", config, ["questions", "predictions"]))
    return result
