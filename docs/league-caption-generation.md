# League gameplay caption generation

## Selected-player pilot

The production renderer below remains deterministic. The separate caption pilot
adds selected-player scene proposals, adaptive clip manifests, replay linkage and
independent validation infrastructure. It is experimental: runnable output and
schema-valid evidence do not establish recognition accuracy or production release.

```bash
# CPU binding/counters plus compatible cached scene requests; never loads a GPU:
python -B -m annotation.league --offline caption-pilot --start 30 --end 38
# Explicit pinned local Qwen inference and dense/native inspection:
python -B -m annotation.league --offline caption-pilot --start 34 --end 38 --infer \
  --icon-references /path/to/ability-icons-16.13.1/manifest.json
# Source-only reference templates, separate from the existing action benchmark:
python -B -m annotation.league --offline experiment --suite captions --initialize
python -B -m annotation.league --offline review --port 8767
```

Global `--run-dir` and `--config` precede the command. `--action-profile` reuses the
v3 action profile; this pilot requires its single `selected` owner. Missing icon
references leave kit identity unknown and do not invent ability names. A grounded
visible body can still be described as “the selected champion.” Other actors use
anonymous local target tracks and source-coordinate regions; the model cannot
supply player/champion names or enemy/allied relationships. A directed attack
additionally requires `targeting_visible`; a generic attack implies no target or
impact. Failed impact/target chains may retain an independently supported cast or
attack precursor, restricted to that precursor's own frames.

All artifacts live under `experiments/captions/<key>/`: `input-audit.json`, native
and masked inputs, identities, body readings, counters, normalized observations,
raw scene proposals, accepted events, atomic captions, `caption_clips.jsonl`,
omissions, resampling decisions, request inputs/responses and resource reports.
Production observations/events/questions and existing action experiments are not
modified. Each experimental row remains marked experimental; pilot-specific
fusion/rendering must be explicitly requested. These captions cannot supply QA
references or silently authorize production scene events.

The source is decoded once at 16 FPS across the requested interval, with one
playback/selection timeline. Overlapping eight-image inference packets reuse frame
IDs and context. Baseline inspection is 4 FPS; scene proposals and CS-credit
brackets trigger 16 FPS review, then unresolved dense proposals trigger native
inspection. Sparse proposals remain diagnostic. Continuous overlapping effect
fragments can be assembled across packets; independent rapid casts cannot be
combined merely because their text/target agrees. Actor bindings are checked at
cues; targets require regions at the relevant cues. Dense evidence tolerates only
the recorded source-frame PTS quantization, not arbitrary gaps.

Clip schema 1 records source hash/time basis, interval, selected owner/epoch and
playback segment, ordered caption IDs, `phases.setup/action/result` evidence,
setup/result status, censoring flags, continuation IDs and replay links. Setup
defaults to two seconds and result context to three; interactions separated by at
most two seconds can share a clip. Clips stop at unsupported continuity or
selection boundaries and at 30 seconds, using linked continuations when needed.
Phase frames describe visible context, not a successful action or inferred result.
Physical actions anchor setup/action/result phases when available; an early HUD
reading does not erase later action setup. Summaries concatenate supported atomic
wording. Point readings stay points.

Playback pause detection requires two elapsed seconds of unchanged clock and
low masked-scene motion. Ordinary one-second clock ticks at 16 FPS are not pauses.
Cuts/rewinds/unknown continuity remain barriers. Replay occurrences keep their own
source timestamps; three corresponding moving-scene anchors spanning at least a
second can link compatible occurrences, including earlier pilot neighborhoods.
Static screens and equal game clocks alone cannot establish a replay. Matching
thresholds are development defaults, not a validated general discontinuity model.

`/captions` provides source playback bounded to each clip, per-claim evidence
selection, actor/target overlays, omissions and continuation/replay navigation.
Corrections are revisioned in that pilot's `reviews/` and regenerate derived
artifacts. Correct upstream proposals, body bindings or counter evidence; generated
wording stays unreviewed. An exclusive pilot lock prevents concurrent generation
and review writes. Independent wording/binding/evidence assessments are separate
artifacts tied to the exact generation key and expire after upstream changes.

### Independent acceptance

`/caption-labels` reads only the source and reference manifest, with paused-frame
region capture and revisioned annotation saves. Its twenty neighborhoods have ten
development and ten held-out splits, default to 24 seconds, and may be adjusted to
8–60 seconds for actual source coverage. Preserve eight-second cross-split guards,
held-out calibration exclusions and repeat-group isolation. Empty claim arrays
are not negative labels until explicitly inspected and marked negative. Every
empty category (`claims`, `bindings`, `phases`, `playback`, `repeats`) also requires
an explicit `negative_annotations: {category: true}` attestation before scoring;
an uninspected or unknown category must remain false.

Reference claims use `{id, kind, start, end, values}`; exact CS comparisons need
`values: {before, after, delta}`. Use `ability` only when visibly grounded. Bindings
use `{claim_id, role, time, box}` in native source coordinates. Phases use
`{phase, start, end}`. Playback uses `{kind, start, end}`; repeats additionally
record `earlier_start`. Keep source-only notes and uncertainty, named labeler
provenance, inspection/verification status and truthful independence attestations.
Reference labels must be completed without detector output. Assess generated
wording separately after those references are fixed.

```bash
# Diagnostic development predictions only; no quality metrics from draft labels:
python -B -m annotation.league --offline experiment --suite captions --draft --infer
# After independent development verification, freeze the unchanged choices:
python -B -m annotation.league --offline experiment --suite captions --freeze --infer
# After held-out verification, generate/score and collect independent assessments:
python -B -m annotation.league --offline experiment --suite captions --infer
```

Benchmark reports persist under `experiments/captions/benchmarks/`; interrupted
jobs retain completed neighborhoods and cached requests. Freeze keys include code,
policy, masks, model pin, icon references, profile and development labels. Editing
labels during evaluation aborts scoring. Corrected detector outputs cannot be used
for benchmark acceptance. `experiment` retains the repository's exit-code-2
convention for experimental results; inspect report status and metrics.

Report per-kind precision/recall, duplicates, timing, source-region binding matches,
independent support/binding assessments, evidence integrity, phase retention,
playback cases, replay correctness and resources with denominators. No output or
unreviewed output can pass. Require at least 20 held-out emitted claims overall,
at least five references/predictions/assessments per core claim kind, at least 95%
support and binding correctness, valid evidence, and coverage of the required case
tags. Per-kind precision and support must also reach 95%. Missing/insufficient
cases remain untested. These minima are pilot criteria, not statistical evidence
for a general production release. Freeze only development choices before scoring
held-out data; do not tune against held-out failures.

For Slurm use `LEAGUE_STAGE=caption-pilot`, `LEAGUE_START`, `LEAGUE_END`,
`LEAGUE_ACTION_INFER=1`, and the existing profile/icon variables. Caption benchmark
jobs use `LEAGUE_STAGE=experiment LEAGUE_SUITE=captions`; `all` continues to exclude
this opt-in pilot. GPU execution needs the existing pinned local worker environment.

Execution measurements and current acceptance limitations are recorded in
[the pilot validation report](league-caption-pilot-validation-2026-11-02.md).

`python -m annotation.league --run-dir <run> generate-captions` refreshes fusion
and renders accepted visual evidence without another model request. `run` includes
this stage. QA continues to use its own evidence queries; captions do not supply
reference answers or become an independent verifier.

## Output contract

`captions.jsonl` contains one independently checkable claim per caption:

- `schema_version: 1` versions the caption contract independently of intake.
- `source_sha256`, `time_basis: source_video_seconds`, `start`, `end` identify the source and interval.
- `text` contains the caption wording.
- `claims` contains a stable ID, kind, text, interval, typed `values`, actor and
  optional target binding IDs, `criteria`, `evidence_refs`, `support: accepted_evidence`
  and `review_status: unreviewed`.
- `actor_bindings` contains actor/target roles, entity (null for unresolved HUD
  owners), binding/slot identifier, scope and evidence references.
- `evidence_refs` contains `{table, id}` references, with tables `states`,
  `normalized_observations`, `events` and `frames`. Normalized observation records
  retain native crop paths, reader artifacts and derivation links.
- `evidence_modality: visual` and `generator: deterministic-evidence-v1` declare
  what was used. These are gameplay descriptions, not speech transcriptions.

Point readings have `start == end`. They do not imply continuous visibility.
Unchanged known-entity readings in a verified segment may be omitted, without
extending the earlier point interval. A counter comparison spans its two readings;
it does not assign an exact time to individual actions inside the interval.

`caption_coverage.json` reports emitted claim counts, omission reasons and zero
incremental VLM tokens. `caption_omissions.jsonl` identifies omitted input records.
An empty output is valid when no eligible evidence exists. These counts measure
generation coverage, not accuracy or salient-event recall against independent labels.

The stage key includes fused states/events, normalized observations, issues,
caption implementation and upstream correction revision. Question-only corrections
do not invalidate captions. Missing caption artifacts are rebuilt on resume.
Captions are viewable in the review UI; correct their upstream evidence to change
them. Human verification of an input is not silently copied to generated wording.

## Evidence distinctions

| Claim kind | Required evidence | Wording boundary |
|---|---|---|
| `displayed_state` | Confirmed visual state and matching accepted observations; resolved entity or explicit HUD slot | Reports the display at that sample. Names do not establish champion identity. Notices are quoted display text. |
| `cs_increased` | Two integer CS readings for the same resolved entity/slot, same non-null playback segment, verified continuity at both endpoints, no intervening unknown/conflict/issue | Reports before/after/delta; does not establish lane-minion kills or individual last hits. |
| `minion_last_hit` | Explicit actor and minion target bindings; `minion_death_visible` and `actor_action_linked_to_death`, each linked to actual event frames | Describes one observed last hit. CS/gold income alone cannot supply it. No gold reward is inferred. |
| `ability_hit` | Named ability; explicit actor/target bindings; `ability_cast_visible`, `target_impact_visible`, `cast_linked_to_impact`, each linked to event frames | Describes a connected impact. A cast animation/cooldown alone cannot supply it. No damage amount or kill is inferred. |
| `cast` / `attack` | Bound actor plus `cast_visible` / `attack_visible` linked to event frames | Reports the action; does not assert that it connected. |
| `recall` / `movement` | Bound actor plus `recall_channel_visible` / `movement_visible` linked to event frames | Reports observed channeling/movement; does not assert recall completion, intent or destination. |

Fast-action claims require at least 16 FPS and two distinct evidence timestamps,
following the existing conservative exact-action restriction. This is a necessary
sampling gate, not validated proof of event recognition. Actor/action/outcome
linkage must still be independently audited. Impact/death evidence cannot precede
all of its linked action evidence. The cited cue chain must also be dense at the
declared FPS (2 ms timestamp tolerance); an FPS label cannot repair sparse evidence.

Experimental, audio-only, rejected, conflicted, unknown and low-acceptance evidence
cannot be promoted into caption claims. Layout/identity issues and editorial
overlays suppress affected intervals. Unsupported fields/event kinds are recorded
as omissions. The current production intake does not validate scene events;
experimental action artifacts are not imported automatically.

Scene producers can supply the following richer event to `normalize_output`.
Every frame ID must belong to the supplied task and lie within the event bounds;
the actor/target names must agree with their bindings. Fusion preserves the
grounding chain rather than merging competing bindings.

```json
{
  "kind": "ability_hit",
  "actor": "Aatrox",
  "target": "Jax",
  "ability": "Q",
  "start_frame": "f1",
  "end_frame": "f3",
  "confidence": 0.9,
  "actor_binding": {"entity": "Aatrox", "binding": "player:1", "frame_ids": ["f1"]},
  "target_binding": {"entity": "Jax", "binding": "target:2", "frame_ids": ["f3"]},
  "criteria": ["ability_cast_visible", "target_impact_visible", "cast_linked_to_impact"],
  "visual_evidence": [
    {"criterion": "ability_cast_visible", "frame_ids": ["f1"]},
    {"criterion": "target_impact_visible", "frame_ids": ["f3"]},
    {"criterion": "cast_linked_to_impact", "frame_ids": ["f1", "f2", "f3"]}
  ]
}
```

## Starting Qwen budget at 11 generated tokens/second

The repository pins `Qwen/Qwen3.8-27B-FP8`. Separate **generated output tokens**
from **visual input tokens**, prompt tokens, and the context-window limit. A
generated-token rate is a decode measurement; it does not measure image encoding
or prompt prefill. Qwen supports explicitly disabling thinking, which the local
runner already does. [Official model card](https://huggingface.co/Qwen/Qwen3.8-27B-FP8).

These are initial tuning budgets, not measured quality optima:

| Task | Suggested generated-output budget | Decode time at 11 tokens/sec if filled |
|---|---|---|
| One HUD field crop | 128-token cap; 256 if long IDs/text truncate | 11.6–23.3 sec |
| Up to four HUD crops | 512 tokens per request | 46.5 sec |
| Scene packet, 4–8 nearby frames | Start at 1,024 tokens per packet; aim for 128–256 output tokens per frame averaged over the packet | 93.1 sec per packet |
| Difficult scene packet | Escalate to 2,048 only after recorded truncation or unresolved event chains | 186.2 sec per packet |
| Caption rendering from accepted facts | 0 | No decoding |

Caps do not force a model to generate that many tokens. Lowering an unused cap
will not speed up a normally stopping completion. Rich schemas and long evidence
IDs consume output tokens; size the cap using actual serialized responses. Reject
truncated JSON and record `finish_reason`, output-token counts, latency and
acceptance/omission rates when comparing budgets.

The current HUD reader retains its existing 512-token cap and the separate action
experiment retains 4,096; this change does not silently alter extraction settings
or invalidate a benchmark's budget choices. At the reported decode rate, a filled
4,096-token response takes about 6.2 minutes. A stream averaging 256 generated
tokens/frame at 1 FPS would require about 23.3 GPU-hours per video-hour, before
prefill, image encoding, retries or model startup. For a single serial stream to
keep up with 1 FPS, it can average at most 11 output tokens/frame before overhead.

Use the existing numeric readers for HUD values and spend Qwen on uncertain
crops and short temporal packets. Retain dense/native evidence around candidate
actions; spending more text tokens cannot recover skipped impact/death frames.
As an initial scene-only input experiment, compare roughly 512–1,024 visual
tokens/frame, using the actual processor's token accounting and independently
labeled events to choose resolution. Keep HUD text in native crops; no universal
visual-token budget guarantees readability of a full 1080p HUD. These suggested
input limits are not applied by the caption renderer.
