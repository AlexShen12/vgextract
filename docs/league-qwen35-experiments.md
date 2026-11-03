# Four Qwen3.5 champion-observation experiments

These runnable pilots operate on the pinned Midbeast recording. Their observations,
identity memory, predictions, corrections, and scorecards are isolated from production
HUD intake, gameplay captions, QA, and the historical action/caption pilots. No
recognition acceptance is implied by a successful command or inference request.

## Models and runtime

| Role | Official checkpoint | Immutable commit | Use |
|---|---|---|---|
| `qwen35_2b` | `Qwen/Qwen3.5-2B` | `15852e8c16360a2fea060d615a32b45270f8a8fc` | HP |
| `qwen35_4b` | `Qwen/Qwen3.5-4B` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | HP |
| `qwen35_9b` | `Qwen/Qwen3.5-9B` | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | HP |
| `qwen35_27b` | `Qwen/Qwen3.5-27B-FP8` | `97f5941bf617e31c5e237364a8602ce3f03a551a` | HP reference and scene/identity/ability experiments |

The small models use BF16; 27B uses the official FP8 checkpoint. Model manifests
verify file SHA256. The runner uses deterministic decoding, thinking off, a 16K
context, up to eight images, and one sequential model on one actual L40S. FP8 requires
an Ada-or-newer GPU. Worker memory monitoring stops work above 90% allocation use.
Preflight exercises an image request, structured JSON, normal completion, and GPU
identity/memory before experimental requests. It establishes runtime compatibility,
not recognition accuracy.

Set `LEAGUE_QWEN35_PYTHON` to an existing compatible VL interpreter. The established
Longleaf `league-midbeast-validation/env-vl` has the required architecture support;
these experiments do not reinstall or change that environment. It currently contains
vLLM 0.32.0, Transformers 5.18.0 and Torch 2.13.0. The existing production model pin
is preserved. Official model cards: [2B](https://huggingface.co/Qwen/Qwen3.5-2B),
[4B](https://huggingface.co/Qwen/Qwen3.5-4B), [9B](https://huggingface.co/Qwen/Qwen3.5-9B),
[27B FP8](https://huggingface.co/Qwen/Qwen3.5-27B-FP8).

## Commands

Global options precede commands. The default run remains
`data/league_runs/midbeast-OLLZBe5-fg4-v2`. New suites are explicit: the existing
`experiment --suite all` and `cache-models --suite all` retain their previous meanings.

```bash
# Online, before offline GPU jobs. Downloads official weights into the asset cache:
python -B -m annotation.league cache-models --suite qwen35
# Cache portraits plus Q/W/E/R icons without assigning a footage patch:
python -B -m annotation.league cache-qwen35-references --reference-version 16.13.1

# Initialize twenty unverified neighborhoods, with ten clips per split:
python -B -m annotation.league experiment --suite qwen35 --initialize
# Prepare source-only images for BOTH splits, including held-out labeling evidence:
python -B -m annotation.league experiment --suite qwen35 --prepare-only
python -B -m annotation.league review --port 8765
# Open /qwen35-labels for source-only annotation, /qwen35 for predictions.

# Actual GPU preflight (uses LEAGUE_QWEN35_PYTHON):
python -B -m annotation.league --offline experiment --suite qwen35 --draft --infer --preflight-only
# Development predictions; no held-out inference or quality scores:
python -B -m annotation.league --offline experiment --suite hp-vlm --draft --infer
python -B -m annotation.league --offline experiment --suite scene-identity --draft --infer
python -B -m annotation.league --offline experiment --suite ability-alignment --draft --infer
python -B -m annotation.league --offline experiment --suite cooldown-identity --draft --infer
# Or explicitly run all four:
python -B -m annotation.league --offline experiment --suite qwen35 --draft --infer

# After independently verifying development labels/case coverage:
python -B -m annotation.league experiment --suite qwen35 --freeze
# After independently verifying held-out labels too:
python -B -m annotation.league --offline experiment --suite qwen35 --infer
```

Without `--infer`, requests may reuse compatible recorded responses; missing requests
remain explicit invalid/missing-cache entries, rather than silently loading a GPU.
Experimental commands return exit code **2** by the repository's existing convention;
errors return 1 and interruption returns 130. The Longleaf helper translates successful
experimental exit code 2 to batch exit code 0.

Existing Longleaf setup:

```bash
# Run from the established remote source directory; no environment recreation:
bash annotation/league/qwen35_longleaf.sh cache
bash annotation/league/qwen35_longleaf.sh initialize
bash annotation/league/qwen35_longleaf.sh prepare
bash annotation/league/submit_qwen35_longleaf.sh preflight
LEAGUE_SUITE=hp-vlm bash annotation/league/submit_qwen35_longleaf.sh draft
```

Overrides: `LEAGUE_QWEN35_DRIVER`, `LEAGUE_QWEN35_PYTHON`, `LEAGUE_QWEN35_RUN`,
`LEAGUE_QWEN35_ASSETS`, `LEAGUE_LABELS`, `LEAGUE_ICON_REFERENCES`, and the existing
Slurm resource variables. All installer/model/log/compiler/temp writes use
`/work/users/a/l/alshen` through `longleaf_paths.sh`. GPU jobs are sequential;
cache weights on a login node before submitting offline jobs.

## Experiment behavior

### Numeric HP: `hp-vlm`

Targets twelve unique HP crops per neighborhood (240 overall); identical contents
are deduplicated within neighborhoods and across the evaluated split boundary.
A shortage is reported, rather than counted as 240 independent samples. The native
HP crop is enlarged 4x with nearest-neighbor sampling and an 8px black border.
Each model gets the same crop, exact-text/current/maximum schema and 128 output-token
cap. Invalid or unreadable requests receive one independent retry. First-attempt and
one-retry results remain separate. A difference greater than .06 between numeric
HP ratio and independently measured green HUD bar rejects the numeric reading in
separate bar-checked results; bar pixels never supply missing HP digits.

Reports include exact text and pair accuracy, accepted precision, readable coverage,
false acceptance of unreadable crops, request latency/throughput, generated tokens,
and memory. Candidate qualification uses 99% precision and 90% coverage on
`retry_policy_bar`; eligible candidates are ranked by valid-request latency.
This point estimate is a small-pilot result, not production certification.

### Boxes and identity memory: `scene-identity`

At 4 FPS, compare direct Qwen localization against CPU bar proposals followed by
Qwen verification. CPU proposals support green/red/blue palettes, dark bar frames,
and level-cell geometry. Proposed body regions are approximate; they are not
segmentation masks. Both arms retain native body, bar, and nameplate boxes. Scene
image coordinates are explicitly converted to native pixels. Native actor crops
are independently passed to Qwen for exact overhead text and actor verification.

A uniquely verified green-bar champion is the main player; actual green pixels
must also be present in the cited bar crop. Multiple green candidates are ambiguous.
Only unique mutual spatial matches connect adjacent visible tracks. Disappearance,
name contradiction, ambiguity or a playback boundary terminates continuity.

The open scoreboard is read using calibrated name/portrait crops. Versioned portrait
matching proposes champion identity and Qwen confirms it. The name-to-champion
registry requires a unique exact Unicode-normalized player-name join, an agreeing
independent portrait match, and repeated overhead observations. Case, punctuation,
and tags are preserved. Champion-name display mode is not joined as a player name.
Competing supported roster identities block that name. Appearance guesses remain
hypotheses and cannot overwrite the registry.

Pass one collects source observations and registry evidence across permitted clips
of one split. Pass two resolves compatible earlier observations and later appearances.
An unreadable name can inherit identity through an uninterrupted track only. Registries
reset between splits/arms, and spatial tracks reset between neighborhoods/cuts.
Descriptions are one row per scene sample: movement, basic attack, ability action,
recall, idle or unknown, using short temporal packets. Unknown visibility is explicit.

### Ability association: `ability-alignment`

First generate scene-only actions from playing-area images without activation hints.
Read the selected Q/W/E/R HUD at 16 FPS. Fresh ready/recast-to-cooldown, supported
charge consumption, and visible recast transitions create activation brackets.
Cooldown ticks, unknown gaps, and newly revealed cooldowns do not establish casts.

Associate activations with same-owner, same-track scene actions within +/-1 second;
both activation endpoints need compatible visible ownership. Matching must be unique
in both directions. Multiple eligible actions or slots remain unresolved. A separate
assisted pass receives scene images, same-frame HUD strips, and the fresh activation
brackets. Fresh activations also trigger 16 FPS scene inspection in merged +/-1-second
neighborhoods. These dense observations remain a separate layer.

Score scene-only recognition, HUD-only activation detection, combined attribution,
and assisted descriptions separately. Slot attribution establishes an activation,
not a hit, damage amount, kill or intention. Rapid abilities with inseparable motion
can remain ambiguous; per-frame descriptions are not frame-exact cast boundaries.

### Cooldown-resistant identity: `cooldown-identity`

Compare same-slot baseline icon matching and raw Qwen, overlay-aware reference
matching, raw+mask+normalized Qwen inputs, and temporal clean-icon memory. Q/W/E/R
kit identity needs at least three agreeing slots with no contradictory accepted slot.
Masks heuristically exclude bright neutral central glyphs and very dark pixels.
Normalization applies a capped brightness gain to remaining observed pixels; excluded
pixels are magenta. The masks can cover genuine artwork and are themselves experimental.
They do not reconstruct hidden content or remove an overlay with guaranteed accuracy.

Temporal memory needs agreement between a clean-looking Qwen observation and an
independent masked reference match. Memory resets on owner/playback boundaries and
supported kit changes. Clean reference images retain their actual source timestamps.
Cold raw/masked/normalized recognition and memory-assisted recognition are scored
separately by natural overlay class. One explicitly synthetic darkening/radial/digit
stress view per neighborhood supplements natural evidence and is scored separately
only against independently labeled clean originals.

## Labels, artifacts, review and acceptance

Labels live in `benchmarks/qwen35/study.json`; choose `--labels DIRECTORY` for a new
benchmark revision. Initial neighborhoods are evenly distributed, source-only
candidates outside historical inference windows. Inspect their coverage before
inference; they do not automatically guarantee scoreboards, recasts, or every other
required condition. Adjust candidate intervals or create a new study if cases are
missing. Missing cases block frozen scoring and remain visible in draft reports.

The source-only editor displays original frames, exact source timestamps/IDs, HP
crops and HUD strips. It does not request prediction endpoints. Reference arrays:

| Category | Required observation fields beyond `frame_id`, `time` |
|---|---|
| `hp` | `status`, `text`, `current`, `maximum`; nonvisible fields null |
| `detections` | `reference_track`, native `body_box`, `hp_box`, `name_box`, `name`, `champion`, `main_player` |
| `scoreboard` | `row_id`, exact `name`, `champion` |
| `actions` | `action` using the six scene categories |
| `activations` | `slot_id`, source `start`, `end` |
| `icons` | `champion`, `overlay`: clean/digits/shading/unavailable/mixed |

Do not round source timestamps before constructing IDs. Label every sampled frame
needed for the evaluated category and explicitly mark inspected negative categories.
Empty uninspected arrays are not negatives. Reference tracks identify the same player
across labeled reappearances. Use `case_tags` to attest the conditions listed in
`qwen35_study.REQUIRED_CASES`. Each clip and the study need independent source-only
labeler provenance; no model confidence or generated annotation can establish it.
Source-only inspection may use native video frames for rapid-action labels.

Freeze stores code/model/profile/reference/correction signatures and development
label hashes. Held-out inference is disabled in draft mode. Frozen evaluation requires
verified references for the selected suites, sufficient case coverage, and unchanged
development choices. Registry inputs never include labels or answer keys. Both splits
are separate inference/memory contexts even though they come from one match; results
do not establish performance on unseen matches or layouts.

Artifacts live under `experiments/qwen35/<recipe hash>/`: individual request inputs,
raw/parsed responses, runtime/preflight records, progress, per-suite predictions,
registries, overlays, icon masks, and JSON/Markdown scorecards. A shared raw-response
cache allows unrelated derivation fixes to reuse exact compatible requests. Truncated
or invalid responses are retained as failures; GPU failures preserve completed work.
They never count as accepted HP readings or valid scene observations.

The prediction viewer offers experiment/suite/layer selection and linked source,
actor crops, scoreboard inputs, same-frame ability HUD, masks and normalized views.
Observation corrections are append-only under the study directory; rerunning rebuilds
tracks, memory, actions, associations and scores with a new correction signature.
Reference-label edits are separate from prediction corrections.

Software tests exercise coordinates, green-bar ambiguity, identity collisions and late
resolution, track loss, fresh activations and competing slots, HP digit validation,
overlay provenance, cache reuse/invalidation, source-only review, and partial GPU failure.
Frozen real-footage scores still require independently completed labels. All recognition
reports remain experimental; no automatic production promotion occurs.

## Validation recorded November 3, 2026

- The complete local suite passed **277 tests**, including 23 Qwen3.5 study tests.
- Source-only preparation completed twenty neighborhoods (ten per split), 640 frames,
  and 240 candidate HP crops before cross-split deduplication. References are unverified.
- Browser inspection confirmed source video playback and native-frame, HP-crop and
  HUD-strip loading in the source-only label viewer.
- Longleaf job **8019878** completed with exit code `0:0` in **17 minutes 13 seconds**
  on one actual **NVIDIA L40S**, eight CPUs and 96 GB host memory. All four models
  completed the image request with normal stopping and valid strict JSON. Their
  checkpoint files were SHA256-verified before offline execution.
- Maximum sampled GPU memory was 39,266,025,472 bytes out of 47,667,740,672 bytes
  (about 82.4%). These measurements include allocator reservations and warm-up;
  they are not weight-only memory footprints or model-speed comparisons.

The preflight evidence is under the Longleaf run's
`experiments/qwen35/ac18d3144aa69d1f9f4758fc2b08f44573499e293c1bf4444dfb357eb8d3d306/`.
Individual `preflight/*.json`, request status/response files, and `report.json` /
`report.md` retain the runtime evidence. This qualifies image/JSON execution;
HP, identity, tracking, action and ability-recognition accuracy remain unmeasured
until independently verified labels are available.
