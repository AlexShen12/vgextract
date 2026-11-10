# Source-only development review, stages 2 and 3

Inspect only assigned source review inputs, original source frames/video, and these instructions. Do not open detector outputs, other benchmarks, experiments outside the assigned source-input folders, old caption drafts, caches, or model results. Candidate selection used historical development neighborhoods; you are not prediction-blind to candidate selection, but must not see recognition predictions yourself. No recognition inference is requested.

Use the source-inputs.json in each assigned clip folder. Open every overview sheet, then inspect original native-resolution source images for decisive cues. Use `python -B -m annotation.league.scene_reference detail` with the supplied manifest/clip ID and a fresh output directory inside your assigned clip to extract dense/native source intervals when needed. The --fps option can be omitted for native cadence. Record precisely what you inspected. Overview sampling cannot prove absence of rapid actions. Do not call a clip quiet without complete native inspection of the stated quiet scope; otherwise quiet is unresolved or not_present.

Write one JSON file per clip in the assigned labels folder. Do not modify other agents' labels or historical files. Use this contract:

```json
{
  "clip_id": "negative-00",
  "source_sha256": "from the source input manifest",
  "review_status": "source_reviewed_model_draft",
  "exhaustive": false,
  "review": {
    "reviewer": "gpt-6-luna:<your-task-name>",
    "prediction_outputs_seen": false,
    "human_reviewed": false,
    "full_native_review": false,
    "method": "Actual overview and native-boundary inspection performed",
    "inspected_sheets": ["absolute paths actually viewed"],
    "inspected_native_frames": ["absolute paths actually viewed"]
  },
  "events": [],
  "cases": [],
  "unknown_intervals": [],
  "notes": "Concrete source observations and limits"
}
```

An event is:
`{id,kind,start,end,actor,slot,ability_name,target,outcome,evidence,phases,left_censored,right_censored,uncertainty}`.
Kinds: movement, basic_attack, ability_cast, item_use, recall, minion_last_hit, base_arrival, hud_activation. Actors: selected, other, unknown. Nullable slot: Q/W/E/R. Times are absolute source seconds within the supplied clip. `evidence` is a non-empty list of `{time,cue}`. `phases` is a list of `{phase,start,end}` inside the event. Record onset uncertainty as a separate `onset_bracket:[lower,upper]` when possible. Unknown names/targets/outcomes stay null; each non-null name/target/outcome requires its own `ability_name_evidence`, `target_evidence`, or `outcome_evidence` list. A HUD-only activation cannot establish target/outcome. A recall needs `terminal_status`: completed, interrupted, or ending_unknown, with evidence. No cause inferred from interruption.

Cases use `{tag,status,start,end,scope,evidence,reason,forbidden_claim}`. `status`: supported, not_present, unresolved. Supported cases need in-clip source times, evidence `[{time,cue}]`, and a narrowly defined scope and expected decision. Unsupported cases can omit times/evidence but must state why. Add exactly one case row for each of the six current-stage tags below; use separate additional cases only for different supported scopes. `quiet` supported requires `exhaustive:true`, `full_native_review:true`, and complete native review of all measured actions in that explicit scope. The default draft is non-exhaustive.

Stage 2 tags: initial_cooldown, cost_countdown_ambiguity, camera_motion, competing_caster, interrupted_recall, quiet.
- Existing cooldown and ordinary ticks are not fresh uses. A small upper mana cost is not a centered countdown.
- Camera movement can coexist with player movement. State exactly what is and is not justified; do not falsely say the player is stationary.
- Another champion's visible cast is not the selected player's cast. Actor linkage needs source evidence.
- Interrupted recall is a positive attempt and a negative example for completion/base arrival. A clipped ending or camera cut is ending_unknown, not interrupted.
- Absence of an accepted event record is not proof of quiet footage.

Stage 3 tags (only label when assigned stage 3): isolated_ability, rapid_attacks, empowered_attack, observable_miss, obscured_outcome, recall_lifecycle.
- Inspect preparation, release and recovery when visible; continuing phases are one physical occurrence.
- New attacks need new onsets, not lingering damage or continuing packets.
- Empowered attack requires visible evidence linking empowerment to a physical attack. Keep the slot activation separate and link event IDs when justified.
- A miss requires observable effect/target geometry with enough visibility to exclude contact. Missing impact evidence alone leaves outcome unknown.
- Record scoreboard/brush/off-camera/cut gaps as unknown_intervals with `{start,end,reason}`. Do not interpolate hidden actions.

Never mark human_verified. Source model reviews remain drafts. Report missing patterns honestly rather than matching the requested category counts with weak evidence. Labeling does not measure recognition accuracy.
