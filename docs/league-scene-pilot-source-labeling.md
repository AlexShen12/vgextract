# Blinded source labels for the five-window input pilot

Inspect only the assigned source-inputs.json, its source image sheets/native frames, and original source video when necessary. Do not read recognition outputs, scores, model requests, automatic HUD readings, body-binding predictions, or other reviewers' labels. Reviewers may have prior development source-labeling exposure; disclose that separately. Held-out answers must not be sent to the recognition developer before recipe freeze.

Review all supplied 16 FPS source sheets in source-time order. Check enlarged native HUD pixels and native source frames around uncertain transitions. Label what is observable at the declared review cadence; unknown intervals remain explicit. No human verification claim is permitted. A quiet declaration requires complete native inspection, so leave quiet untested if only sampled sheets were reviewed.

Write one JSON file per assigned clip with this structure:

```json
{
  "clip_id": "holdout-00",
  "source_sha256": "from the source index",
  "review_status": "source_reviewed_model_draft",
  "review": {
    "reviewer": "gpt-6-luna:task-id",
    "prediction_outputs_seen": false,
    "human_reviewed": false,
    "method": "Actual source review method and cadence",
    "inspected_sheets": ["actual viewed paths"],
    "native_boundary_frames": ["actual viewed paths"]
  },
  "scope": {"hud_activations": "complete_observable", "physical_actions": "partial"},
  "events": [],
  "unknown_intervals": [],
  "notes": "Specific review limits"
}
```

Use scope `complete_observable` only after actually inventorying all observable events of that track in the entire supplied clip; otherwise use `partial`. Sampling and source-agent labels do not establish human ground truth. A partial physical-action inventory is acceptable, but cannot establish full event precision/recall. The primary shared track is observed Q/W/E/R cooldown activations, reviewed from the source HUD; target and physical outcomes are separate.

Each event has `{id,kind,start,end,actor,slot,ability_name,target,outcome,evidence,left_censored,right_censored,boundary_resolvable}`. Use actual absolute source PTS from the input index and uncertainty brackets when needed. Actors: selected, other, unknown. Slots: Q/W/E/R or null. Names, targets and outcomes remain null unless independently visible with separate `ability_name_evidence`, `target_evidence`, or `outcome_evidence`. Evidence is a list of `{time,cue}`. Event IDs are unique within the clip.

Kinds: `hud_activation`, `movement`, `basic_attack`, `ability_cast`, `item_use`, `recall`, `minion_last_hit`, `base_arrival`. For HUD activations, `start/end` bracket the last no-countdown frame and first persistent centered cooldown frame; do not use the corner mana cost, a continuing countdown tick, or the initial already-cooling state as a new onset. W priming and later consumption are one linked use for this cooldown-onset track, not two cooldown starts. Put priming in notes when relevant. A cooldown return/reset alone is not another use. Record unresolved recasts/charges as unknown rather than guessing.

Physical casts/attacks need their own visible animation/cast-bar cue and event record. Do not turn a cooldown alone into a physical occurrence. Continuing preparation/release/recovery belong to one occurrence; a second attack needs a new onset. A visible miss needs exposed effect/target geometry; missing damage alone is insufficient. Recall uses `terminal_status`: completed, interrupted, or ending_unknown. Completion needs observed channel ending and compatible base arrival; later fountain confirmation is not the boundary. Source camera cuts, ownership changes, scoreboard coverage and unclear actor spans belong in unknown intervals.

Unknown intervals are `{start,end,scope,reason}`, with scope `hud_activations`, `physical_actions`, or `all`. Do not silently discard difficult model cases; unknowns must be independently justified by the source. Where only one slot is unreadable, document the affected slot in the reason rather than claiming all observations were absent.

Held-out output goes to the assigned local sealed-label directory. Return only filenames, hashes and completion/validation status to the parent; do not reveal labels, event counts, case categories or scene descriptions before recipe freeze. Development labels can be reported normally. Keep revisions; do not overwrite source evidence or fabricate viewed-frame paths.
