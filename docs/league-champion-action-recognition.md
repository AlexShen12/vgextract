# Player actions in the League annotation pipeline

The current action workflow is version 3. It runs inside `annotation.league` and
keeps experimental results separate from production observations, events and
questions. Historical version-2 and version-3 runs remain available in review.
The original prototype and v2 execution APIs have been retired; their saved
artifacts are preserved.

## Commands

```bash
# Prepare source-PTS images only; no recognition or GPU loading:
python -B -m annotation.league action-window --start 30 --end 38
# CPU player binding, counter/slot observations and recall:
python -B -m annotation.league action-window --start 30 --end 38 --detect
# CPU stages plus pinned local Qwen scene interpretation:
python -B -m annotation.league action-window --start 30 --end 38 --infer
# Optional explicit icon catalog, especially on Longleaf:
python -B -m annotation.league action-window --start 30 --end 38 --detect \
  --icon-references /path/to/ability-icons-16.13.1/manifest.json
# Official descriptions, independently of the footage's unknown patch:
python -B -m annotation.league cache-abilities --reference-version 16.13.1 --champions all
# Independent labels: initialize once; never overwrite a labeling manifest:
python -B -m annotation.league experiment --suite actions --initialize
# Development CPU predictions without quality scores:
python -B -m annotation.league experiment --suite actions --draft
# Add --infer explicitly for GPU scene interpretation.
# After independent development labels are verified:
python -B -m annotation.league experiment --suite actions --freeze
# After held-out labels are verified and development choices remain frozen:
python -B -m annotation.league experiment --suite actions
python -B -m annotation.league review
```

Global `--run-dir`, `--config` and `--offline` precede the command. Individual
windows are at most 30 source-video seconds; in-game clock values are continuity
evidence, not event timestamps. `--action-profile` selects geometry/palette;
the v3 default is `midbeast-actions-1080p-v2.json`. `--assistance` is rejected in
new action runs. Other experiment suites and their `all` behavior are unchanged.

The v3 entry points remain in `action_v3.py` and `action_benchmark_v3.py`.
`action_pipeline.py` now contains only shared frame/HUD preparation and validation;
`action_benchmark.py` contains shared independent-label and scoring helpers. New
label manifests are initialized directly as v3, without an assistance template.
Forking old benchmark clip locations clears labels, verification and labeler
attestations, and leaves the old manifest untouched. Implementation fingerprints
change with refactors: existing artifacts remain reviewable, but release/freeze
checks must be repeated for the changed code. Compatible v3 inference requests
retain their existing cache identities.

The Slurm entry point accepts `LEAGUE_STAGE=action-window`, `LEAGUE_START`,
`LEAGUE_END`, `LEAGUE_ACTION_DETECT=1` or `LEAGUE_ACTION_INFER=1`, and optional
`LEAGUE_ACTION_PROFILE`/`LEAGUE_ICON_REFERENCES`. For actions experiments, inference
also requires `LEAGUE_ACTION_INFER=1`; core evaluation is CPU based by default.

## Identity and body binding

Q/W/E/R icon matching is the only automatic player-kit identity source. Compare
like slots against the versioned reference catalog; require three agreeing slots
and no strong contradictory slot. Development cosine/margin defaults are .45/.06,
with a recheck every .25 seconds. Missing references, obscured icons or ambiguity
produce unknown identity. Reference version is not footage patch; mechanics remain
disabled. Qwen is never asked to name the player and cannot override identity.

The CPU binder examines overhead bars at 16 FPS. A candidate needs the configured
player palette, horizontal HP-strip geometry, a dark frame and adjacent champion
level cell. Geometry excludes HUD panels, portraits, minimap and facecam; minion
bars without level cells are rejected. Same-frame HUD health fraction is compared
when readable. The initial development profile supports green, not every client
palette or layout.

Require one compatible candidate across three consecutive samples. Store actual
bar bounds, approximate actor region, source frames and track ID. The region is
localization below the bar, not segmentation. Cuts, disappearance, competing bars,
selection changes, invalid layout and supported kit changes interrupt continuity.
Unknown icon identity alone does not change an otherwise supported visible body
track. Unknown champions retain local player tracks. Native packets reconcile
tracks only through nearby overlapping detected bars.

## Farming and recall

CS and displayed gold are sampled at 16 FPS with pixel-content OCR caching and
stable neighboring values. Unknown readings, source gaps and ownership/kit changes
cannot establish counter deltas. Positive CS change is a `creep_credit` aggregate;
CS points are not assumed to count lane-minion deaths. Gold changes, including
passive income and purchases, are separate `gold_change` records.

A uniquely compatible scene proposal can corroborate `minion_last_hit` when it
links the bound player action to a dying lane minion. Jungle and unknown creep
candidates remain separate. Exact reward gold requires a readable uniquely
associated popup; the displayed-gold delta is never substituted for reward.

Recall needs explicit label and channel-bar evidence across neighboring frames.
One interval retains its observed start, channeling evidence and terminal status:
`completed`, `interrupted`, or `ending_unknown`. Completion additionally requires
observed channel ending and compatible visible base arrival. Early termination
plus compatible subsequent activity supports interruption. A cut, disappearance,
health increase or vanishing bar alone establishes neither outcome. Clipped
attempts retain censoring flags. Brief observed tooltip occlusions can remain in
one interval with explicit uncertain frame references; they do not establish a
hidden completion. Longer gaps terminate supported continuity.

Evidence classes remain `corroborated`, `hud-supported`, `scene-only`, `unresolved`.
They describe evidence, not calibrated confidence. Activation, impact and successful
outcome remain separate facts.

## Scene inference and official references

`--infer` uses only the pinned local Qwen3.8-27B-FP8 worker. It receives source-time
images, deterministic track IDs and outlined approximate player regions. New
schemas contain no identity predictions. Other visible actors remain anonymous.
Only cited visual animation/effect evidence can establish scene actions.

Baseline gameplay inspection is 4 FPS; HUD is 16 FPS. Ambiguous slot transitions,
rapid sequences and unmatched creep credit trigger 16 FPS inspection, then native
frames around unresolved brackets. Existing one-second association margins,
overlapping packets, eight-image limit and 16K context remain. GPU/context errors
preserve CPU results and completed requests and stop the stage without lowering
image evidence quality. Cached inference includes PTS/image hashes, body context,
profile, sampling, prompt/schema and model/runtime identity.

`cache-abilities` builds `assets/abilities/<version>/catalog.json` from Riot Data
Dragon's official roster and champion files. It preserves passive/Q/W/E/R names,
raw/clean descriptions, available tooltips, unresolved substitutions, icons, URL
provenance and hashes. Raw JSON and verified icons support offline reuse. Failures
remain partial and can be retried by champion ID. The catalog does not enable
mechanics or set the footage patch. Descriptions are currently review context,
not a source of expected cooldowns or invented visual outcomes.

## Artifacts, review and independent labels

New artifacts live in `experiments/actions-v3/<key>/core/`: identities, body
readings/bindings, raw counters, recall evidence, HUD slots/transitions, scene
proposals, actions and resampling history. `core_actions.jsonl` preserves CPU-only
output for comparison with scene-corroborated results. Reports include elapsed
preparation/detection time, requests/tokens, cache reuse and peak reported memory.
Completed requests resume from the v3 shared cache. Nothing is promoted to production.
Physical GPU UUID is retained as provenance; equivalent GPU classes with the same
model/software can reuse completed requests. Cached and newly generated resource
usage are reported separately.

The existing review app provides `/actions` with synchronized source playback,
bar/actor overlays, raw evidence, lifecycle status and official descriptions.
Corrections remain revisioned within the experiment. Historical v2 runs remain
available, including their archived identity diagnostics.

`/action-labels` is a separate source-only reference editor with no detector
predictions. Capture a paused native frame and drag HP-strip/actor reference
rectangles; add positive or unbound samples, then edit temporal action labels.
Saving requires an identified independent labeler and explicit independence
attestation. Revisions are separate from experiment corrections and production.

The v3 benchmark forks twenty unverified eight-second clips (10 development,
10 held-out). The development grid includes binding/farming, cast-candidate and
recall neighborhoods; actual case tags remain for independent annotation. Preserve
cross-split guards, calibration exclusions and repeat grouping. Supplementary
recall development windows may be longer. Freeze verified development choices
before held-out scoring; annotation edits during evaluation abort the stage.
No quality metric is produced from unverified labels or draft predictions.

Metrics compare core-only and combined recognition, binding accuracy/coverage,
creep/minion precision/recall and known count error, recall timing/status, slot
transitions, unresolved cases, duplicates and resources. Missing cases remain
untested. The separate independent-agent instructions are in
`docs/league-actions-labeling-handoff.md`.
Unresolved slot candidates are scored separately from established activations;
rapid ordering and known-slot attribution retain explicit scoring denominators.
