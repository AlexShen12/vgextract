"""Deterministic answers first; wording never supplies or changes a fact."""
from __future__ import annotations

from collections import Counter, defaultdict
from .core import CAPABILITIES, Store, atomic_json, digest, finite, uid, validate_question, consistent_review


def stamp(time):
    return f"{int(time)//60:02d}:{time%60:05.2f}"


def horizon(span):
    return "local" if span <= 5 else "phase" if span <= 90 else "match_trajectory"


def _question(capability, prompt, answer, answer_type, evidence, category="visual_only"):
    times = [r.get("time", r.get("start")) for r in evidence]
    ends = [r.get("time", r.get("end")) for r in evidence]
    start, end = max(0, min(times)-2), max(ends)+2
    frame_ids = sorted({i for r in evidence for i in r.get("frame_ids", r.get("evidence_ids", []))})
    row = {"id": uid("question", capability, prompt, answer, [r["id"] for r in evidence]),
           "prompt": prompt, "answer": answer, "answer_type": answer_type,
           "interval": [start, end], "evidence_ids": [r["id"] for r in evidence],
           "evidence_frame_ids": frame_ids, "evidence_times": sorted(set(times + ends)),
           "evidence_intervals": [[r.get("time", r.get("start")), r.get("time", r.get("end"))] for r in evidence],
           "capability": capability, "horizon": horizon(max(ends)-min(times)),
           "knowledge_category": category, "review_status": "unreviewed", "ambiguity_flags": [],
           "coverage_requirements": {"separated_evidence": len(evidence) > 1,
                                     "continuous_visibility_required": False,
                                     "semantic_scope": "Only explicitly observed states/events",
                                     "minimum_fps": None},
           "reference_precision": "displayed", "reference_source": "automatic_evidence_query"}
    return validate_question(row)


def construct(states, events, *, max_questions=60, issues=(), duration=None, resolver=None):
    candidates = defaultdict(list)
    confirmed = [s for s in states if s["status"] == "confirmed" and s["entity"] != "unknown"
                 and s.get("review_status") != "rejected"]
    usable_events = [e for e in events if (e["confidence"] >= .75 or e.get("review_status") == "human-verified") and e.get("actor")
                     and e["actor"] != "unknown" and e.get("review_status") != "rejected"]

    def add(question):
        if duration is not None:
            question["interval"][1] = min(duration, question["interval"][1])
        bad = [i for i in issues if i.get("kind") in {"editorial_overlay", "layout_mismatch"}
               and question["interval"][0] <= i.get("time", -1) <= question["interval"][1]]
        if not bad:
            candidates[question["capability"]].append(question)

    for event in usable_events:
        # Avoid exact action/reaction timing at discovery FPS.
        if event["kind"] in {"attack", "cast"} and event.get("sampling_fps", 0) < 16:
            continue
        prompt = (f"Which champion is the actor of the visibly recorded {event['kind'].replace('_', ' ')} "
                  f"near source-video time {stamp(event['start'])}?")
        q = _question("episodic_retrieval", prompt, event["actor"], "entity", [event])
        q["coverage_requirements"]["minimum_fps"] = 16 if event["kind"] in {"attack", "cast"} else None
        add(q)

    grouped = defaultdict(list)
    for state in confirmed:
        grouped[state["entity"], state["field"]].append(state)
    for (entity, field), sequence in grouped.items():
        sequence.sort(key=lambda s: s["time"])
        # Use separated observations; never turn sparse samples into continuous longevity.
        for first in sequence:
            following = next((s for s in reversed(sequence) if s["time"] - first["time"] >= 30), None)
            if not following:
                continue
            first_time, last_time = stamp(first["time"]), stamp(following["time"])
            if field in {"structure_status", "vision_status", "inventory", "level", "alive"}:
                add(_question("persistent_state",
                    f"What was the displayed {field.replace('_',' ')} of {entity} at {first_time} and {last_time}, respectively? "
                    "Report the two observations without assuming visibility between them.",
                    [first["value"], following["value"]], "sequence", [first, following]))
            if field in {"kills", "deaths", "assists", "cs"} and finite(first["value"]) and finite(following["value"]):
                difference = following["value"] - first["value"]
                if difference >= 0 and int(difference) == difference:
                    add(_question("cumulative_counting",
                        f"By how much did {entity}'s displayed {field} counter increase between source-video times "
                        f"{first_time} and {last_time}?", difference, "number", [first, following]))
            break
        if field == "position":
            points = []
            for state in sequence:
                location = state["value"]
                if isinstance(location, dict):
                    location = location.get("region")
                if isinstance(location, str) and location and (not points or
                        state["time"] - points[-1]["time"] >= 30):
                    points.append({**state, "location": location})
            if len(points) >= 3:
                points = [points[0], points[len(points)//2], points[-1]]
                add(_question("trajectory_memory",
                    f"At source-video times {', '.join(stamp(p['time']) for p in points)}, "
                    f"which observed map regions did {entity} occupy, in that order?",
                    [p["location"] for p in points], "sequence", points))

    actors = defaultdict(list)
    for event in usable_events:
        actors[event["actor"]].append(event)
    for actor, sequence in actors.items():
        sequence.sort(key=lambda e: e["start"])
        for first in sequence:
            second = next((e for e in reversed(sequence)
                           if e["kind"] != first["kind"] and e["start"] - first["end"] >= 30), None)
            if second:
                add(_question("event_linking",
                    f"Which champion is the common actor of the {first['kind'].replace('_',' ')} near "
                    f"{stamp(first['start'])} and the {second['kind'].replace('_',' ')} near {stamp(second['start'])}?",
                    actor, "entity", [first, second]))
                add(_question("episodic_retrieval",
                    f"For {actor}, which two visibly recorded event types occur near source-video times "
                    f"{stamp(first['start'])} and {stamp(second['start'])}, respectively?",
                    [first["kind"], second["kind"]], "sequence", [first, second]))
                break
        repeating = defaultdict(list)
        for event in sequence:
            if event.get("location") and event["kind"] in {"recall", "channel", "reset", "encounter", "skirmish"}:
                repeating[event["kind"], event["location"]].append(event)
        for (kind, location), repeats in repeating.items():
            separated = []
            for event in repeats:
                if not separated or event["start"] - separated[-1]["end"] >= 30:
                    separated.append(event)
            if len(separated) >= 3:
                points = separated[:3]
                add(_question("recurring_behavior",
                    f"For {actor}'s visibly recorded {kind.replace('_',' ')} events near "
                    f"{', '.join(stamp(p['start']) for p in points)}, which location recurs in all three?",
                    location, "string", points))

    if resolver:
        # Mechanics questions add a static fact to an already grounded visible inventory.
        # No knowledge request is allowed to establish that the purchase occurred.
        cached = {}
        for state in confirmed:
            if state["field"] != "inventory" or not isinstance(state["value"], list):
                continue
            for slot, item in enumerate(state["value"]):
                if not isinstance(item, (str, int)) or not str(item).isdigit():
                    continue
                key = str(item)
                if key not in cached:
                    cached[key] = resolver.handle_query({"entity_type": "item", "query_knowledge": key})
                response = cached[key]
                reference = response.get("reference", {})
                stat = reference.get("stats", {}).get("values", {}).get("ability_haste:flat")
                if stat:
                    q = _question("episodic_retrieval",
                        f"At source-video time {stamp(state['time'])}, which item occupies visible inventory slot {slot+1} "
                        f"of {state['entity']} (slots read left to right), and how much flat Ability Haste does it grant on this patch? "
                        "Answer with item (display name) and ability_haste (number).",
                        {"item": reference["identity"]["name"], "ability_haste": stat["value"]},
                        "object", [state], "mechanics_augmented")
                    q["static_evidence"] = {"item": key, "stat": stat, "provenance": reference["provenance"]}
                    add(q)

    # Round robin across capabilities. Deduplicate identical evidence/answer formulations.
    # The same visual question cannot have competing reference answers (e.g. simultaneous deaths).
    answers_by_input = defaultdict(set)
    def input_signature(q):
        return digest([q["prompt"], q["interval"], q["answer_type"], q["knowledge_category"]])
    for group in candidates.values():
        for q in group:
            answers_by_input[input_signature(q)].add(digest(q["answer"]))
    for capability in CAPABILITIES:
        candidates[capability] = [q for q in candidates[capability] if len(answers_by_input[input_signature(q)]) == 1]
    # Interleave horizons within a capability, so local candidates cannot crowd out memory items.
    for capability in CAPABILITIES:
        buckets = {h: [q for q in candidates[capability] if q["horizon"] == h]
                   for h in ("match_trajectory", "phase", "local")}
        candidates[capability] = [bucket[i] for i in range(max(map(len, buckets.values()), default=0))
                                 for bucket in buckets.values() if i < len(bucket)]
    selected, seen = [], set()
    while len(selected) < max_questions:
        progress = False
        for capability in CAPABILITIES:
            while candidates[capability]:
                question = candidates[capability].pop(0)
                signature = input_signature(question)
                if signature not in seen:
                    seen.add(signature)
                    selected.append(question)
                    progress = True
                    break
            if len(selected) >= max_questions:
                break
        if not progress:
            break
    return selected


@consistent_review
def generate_qa(store: Store, config, resolver=None):
    from .core import read_json
    manifest = read_json(store.path("manifest.json"))
    questions = construct(store.rows("states"), store.effective(store.rows("events"), "events"),
                          max_questions=config.max_questions, issues=store.rows("issues"),
                          duration=manifest["interval"][1], resolver=resolver)
    for q in questions:
        q["interval"][0] = max(q["interval"][0], manifest["interval"][0])
    store.write("questions", questions)
    counts = Counter(q["capability"] for q in questions)
    summary = {"generated": len(questions), "target": config.max_questions,
               "counts": {c: counts[c] for c in CAPABILITIES},
               "shortfalls": {c: max(0, config.max_questions // len(CAPABILITIES)-counts[c]) for c in CAPABILITIES},
               "note": "Coverage shortfalls are retained; unsupported questions are not manufactured"}
    atomic_json(store.path("question_coverage.json"), summary)
    store.commit_stage("generate-qa", store.stage_key("generate-qa", config, ["states", "events", "issues"]), **summary)
    return summary
