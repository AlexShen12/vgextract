# League annotation and memory evaluation

For the unified experimental Midbeast Pass 1 workflow, see the
[Pass 1 runbook](../../docs/league-pass1-runbook.md). It reuses existing readers,
adds shared playback/selection scopes, inventory identity links and histories,
and conservative scoreboard reopen groups. Its manifest distinguishes completed
full-recording producers from pilot coverage and pending quality validation.
`python -m annotation.league.pass1 --help` lists the commands; `/pass1` in the
existing review app opens the assembled bundle. SAM windows remain out of scope.

The [four-pass pipeline design](../../docs/league-annotation-pipeline.md) defines
the target flow from HUD, commentary and killfeed intake through SAM masks,
clip captions, and descriptions. Pass 1 prefers retained YouTube auto-captions
for commentary, with ASR as a fallback. Use `pass1 import-captions` to import a
timed JSON3 track, then finalize to publish the shared timeline. The current
bundle uses the imported YouTube track and retains the completed ASR comparison. This README covers the current commands and release scope.

The [reference refresh and cast/body recognition workflow](../../docs/league-recognition-refresh.md)
documents the 16.22 catalogs, opt-in cast-bar reader and source-labeling page,
gated profile promotion, and isolated body-recognition experiments. New Longleaf
scripts prepare separate jobs and do not submit them.

The default is a Midbeast intake: native field crops → numeric Tesseract,
bounded Qwen transcription, direct bar measurement and inventory recognition → schema-2 observations →
fusion → evidence-backed gameplay captions and supported visual questions → optional independent solver evaluation.
**Release acceptance is pending. Full-video intake remains gated.**

The opt-in `caption-pilot --start <source seconds> --end <source seconds> [--infer]`
adds selected-player scene proposals and adaptive caption clips in isolated
`experiments/captions/` artifacts. Review them at `/captions`; independent source
labels live at `/caption-labels`. `experiment --suite captions` initializes,
freezes and scores a separate caption benchmark. CPU-only runs reuse compatible
scene caches and report missing scene evidence. See
[caption workflow and acceptance](../../docs/league-caption-generation.md).

### Darius cut and clip experiment

Install `annotation/league/requirements-darius-clips.txt`, then run
`python -m annotation.league.darius_clip_pilot` from the repository root. The
command scans the pinned upload and writes an experimental comparison under
`data/league_runs/midbeast-pass1-final-v1/experiments/darius-clips/r1/`. Open
`review.html` to play the source ranges. To reproduce the source-reviewed
comparison, pass the experiment's `source-review.json` via `--review`.

`report.json` records the comparison coverage, source-review method, and phase
retention before and after adding visual context. Clip candidates are limited to
the dense Pass 1 sample windows, less the caption, occurrence, and scene
holdout guards. `detector-scan.json` can be reused for the same source and
detector configuration.

## Source and enabled fields

`midbeast-source.json` pins the latest eligible upload found in the latest 20
Midbeast main-channel uploads inspected on 2026-10-27:
[OLLZBe5-fg4](https://www.youtube.com/watch?v=OLLZBe5-fg4),
“#1 DARIUS is TAKING OVER KOREA... *CRAZY 1v1s*”, 28:53, nonlive, 1080p60.
The downloaded MP4 hash is
`f4f36e1a274617ecbe224eaf7fc3f8375f757acb7a517bcc0ee7780fcd83f240`.
English speech eligibility still requires the audio pilot audit; source metadata
reports English and inspected frames show gameplay plus commentator facecam.

Fresh runs use `profiles/midbeast-1080p-inventory-v1.json`. The profile retains the
21 original fields and adds the inventory crops.

| Reader | Fields |
|---|---|
| Tesseract, one fixed bright-glyph preprocessing, one call | Clock, team kills, selected K/D/A, CS, gold, level, displayed objective timer |
| Qwen3.8-27B-FP8 | Selected numeric HP, ten displayed scoreboard player-name slots, central notices including objective text |
| Direct pixels | Selected HP bar proportion, independent of numeric HP |
| CPU reference icon matching | Selected portrait, six item slots and trinket, plus the same fields for all ten rows of a visible scoreboard |

Inventory ownership is established from the same-frame portrait. Unresolved rows
retain their readable items. For a dimmed scoreboard portrait with red respawn
digits, the reader compares grayscale references while excluding digit pixels.
The Midbeast selected portrait also has a
source-calibrated skin example corroborated by three ability icons. Spell
matching, ability ranks/cooldowns, minimap
tracking, gameplay-event reasoning and audio-derived game state are deferred in
production. Unknown patch is allowed for visual processing;
mechanics are off and require an explicitly established patch.

Frames are sampled at 1 FPS using actual PyAV source PTS. Calibrated panel anchors
suppress absent panels; unknown layouts preserve diagnostic images and unknown
records. Clock rewinds, gaps, clock uncertainty, layout transitions and detected
cuts break playback segments. Revisited clocks mark potential repeated footage;
cumulative questions cannot span segments or verified replayed intervals.
This conservative detector is not a validated universal edit detector.

## Normal commands

```bash
python -m annotation.league prepare
python -m annotation.league processor-preview
python -m annotation.league preflight  # requires the allocated GPU and cached Qwen
python -m annotation.league experiment --suite all --initialize
python -m annotation.league review
```

The default local run is `data/league_runs/midbeast-OLLZBe5-fg4-v2`.
Existing Domisum runs, labels and profiles are preserved for inspection, not reused
as Midbeast benchmark truth. Version-1 runs remain readable in review. Extraction
refuses version-1 runs or runs prepared with the earlier OCR engine; prepare a new
run directory instead of mixing engines.

| Command | Purpose |
|---|---|
| `prepare` | Pin source hash/geometry and save layout previews. |
| `processor-preview` | Show enabled native crops and preprocessing without GPU inference. |
| `annotate` | Run the reduced readers, bounded validation and extraction gates. |
| `generate-captions` | Fuse current evidence and write timestamped gameplay captions with structured claims and bindings; no GPU call. |
| `run` | Prepare, annotate, fuse, generate captions and supported questions, optionally evaluate, report. |
| `review` | Inspect native/processed evidence, uncertainty, corrections and calibration. |
| `intake-benchmark --labels …` | Score exact text, parsed numbers, bars and presence separately. |
| `experiment --suite crops\|events\|audio\|all` | Separate experimental artifacts; no production evidence changes. |
| `cache-models --suite intake\|events\|audio\|all` | Download immutable pinned checkpoints for later offline use. |
| `cache-inventory-references --reference-version 16.13.1` | Cache Data Dragon item icons, descriptions, and champion portraits for the bundled inventory profile. |

Qwen intake is a transcription reader, separate from the existing optional QA
solver. It uses no mechanics retrieval or target-video examples. Requests contain
at most four labeled lossless field crops, deterministic decoding, thinking off,
a 512-token cap and a strict JSON schema. Missing/duplicate/invented field IDs,
malformed JSON and non-normal finish reasons are rejected. Failed fields get one
individual retry. GPU worker/OOM failures stop the stage instead of restarting
for every crop. Grammar acceptance is not independent semantic verification.
Names use individual requests: Longleaf development probes exposed false absence
in grouped name requests. Required name/HP text in a present panel cannot be
accepted as absent; it gets one retry and otherwise remains unknown.

Native crops, fixed processed crops, prompts, raw responses, model/runtime pins,
tokens and latency are retained. Identical compatible crop contents may reuse
recognition at each sampled timestamp; no continuous visibility is inferred.
Numeric preprocessing and VL reading have separate cache keys; unrelated mechanics
or solver changes do not repeat intake. Model manifests verify file SHA256 offline.

## Killfeed timeline pilot

The CPU pilot supports the calibrated 1080p Midbeast recording `OLLZBe5-fg4`.
Pass a completed single-match scoreboard skim for the same recording and use
`--start` and `--end` to limit extraction to a source-video interval.

```sh
python -m annotation.league --run-dir data/league_runs/midbeast-OLLZBe5-fg4-v2 \
  killfeed-intake --scoreboard-skim data/league_recognition_v1/scoreboard-skim-20261111-v1
python -m annotation.league --run-dir data/league_runs/midbeast-OLLZBe5-fg4-v2 review
```

**Killfeed announcements** and **Killfeed roster pilot** in review open the
source video and saved evidence crops. For an annotated run, `fuse` also adds
the announcements to **Events**. The roster view shows provisional team
membership until the first accepted scoreboard witness. Opposing or complete
scoreboard evidence can veto a feed identity. The pilot records team membership
only.

The active revision is under `killfeed/active.json`. Its directory contains
`killfeed_timeline.jsonl`, `killfeed_roster_timeline.jsonl`, and the raw readings
and crops. Rerun intake after a scoreboard review to use its new revision.
Score a full-match run against the saved source-only labels with
`python -m annotation.league.killfeed_benchmark --run-dir RUN --scoreboard-skim SKIM`.

## Minimap timeline

```sh
python -m annotation.league minimap-intake
python -m annotation.league review
```

The opt-in CPU reader uses the prepared 1080p Midbeast source `OLLZBe5-fg4`.
It samples the native minimap every five source-video seconds. Set `--interval`
to change the cadence or use `--start` and `--end` to focus on a window.
Select **Minimap timeline** in review to seek to each sample and see its crop.
Each execution replaces `minimap_timeline.jsonl` with the requested interval.

Each sample records approximate visible minion-wave centers by team and lane
and the sampled status of the 18 lane towers. Centers are pixel coordinates
relative to the minimap crop. A tower icon can be standing, absent at a clear
site, or unknown. Unknown statuses include the reason the reader abstained.
Wave readings are presence candidates, and tower statuses describe the sampled
frame. Review includes the native crop for checking both.

## Inventory timeline

Cache the inventory references before annotation. Use `--offline` before the
subcommand when the pinned metadata and icons already exist in the local caches.
Use a fresh `--run-dir` for the inventory profile if the default run directory
already has a saved config. The reference version identifies catalog artwork and
descriptions independently of the footage's gameplay patch.

Selected HUD items use the top unique reference above a score floor. Trinket
crops account for the selected HUD's frame and the scoreboard's vision score.
Ordinary scoreboard items also require separation between the top two matches.
On a close miss, the reader tries one-pixel shifts in the four cardinal directions.
Among passing shifts, it prefers the item with the most agreeing views, then the
highest similarity score. Review links to the winning native crop.

Use `midbeast-1080p-inventory-v3` for development checks of scoreboard portraits.
It pins Data Dragon 16.22.1 and CommunityDragon 16.22 and scores the center, left,
right, up, and down crops for a visible portrait. Passing views vote for their
top champion. Equal vote counts use the strongest similarity. The inventory
review shows each top match, score, margin, and gate outcome. Hidden portraits
remain unknown, and the respawn view uses its own score and margin gates.
The v3 color margin is 0.05. Independent labels and the five-minute pilot are
required for a release comparison.

`annotate` saves individual item and portrait crops. `fuse` writes
`inventory_snapshots.jsonl` and `inventory_timeline.jsonl`. The review app's
`/inventory` page follows the source video and links each slot to its crop. A
confirmed scoreboard inventory can be shown as **last seen** after the panel
closes within the same playback segment. Apply changes through HUD observation
corrections and rerun `fuse` to refresh the inventory timeline.

For release validation, extend the source-only crop labels with `inventory.*`
field IDs from the new profile. Use canonical Data Dragon IDs as strings and
mark visibly empty slots or closed panels `absent`. Mark obscured or visually
indistinguishable identities `unreadable`. Record an `inventory_labeling` object
with `labeler` and `independent_of_predictions: true`, and include
`inventory_scoreboard_open`, `inventory_scoreboard_closed`, and
`inventory_hidden_portrait` in each split. `annotate --sample-times LABELS.json`
extracts the samples. `intake-benchmark --labels LABELS.json --freeze` checks
development results and freezes the profile. Re-extract the samples and score
with `intake-benchmark --labels LABELS.json`. The report scores reference matching
by surface and requires 98% precision and 70% readable
coverage for both items and portrait attribution. Full intake also requires the
existing five-minute runtime pilot.

## Gameplay captions

```bash
python -m annotation.league --run-dir data/league_runs/midbeast-OLLZBe5-fg4-v2 generate-captions
```

Writes `captions.jsonl`, `caption_coverage.json`, `caption_omissions.jsonl`, and a
resumable `stages/generate-captions.json` marker. Each caption contains source-video
seconds, the source SHA256, text, typed claims, actor/target bindings and typed
references to states, normalized observations, events and source frames. The
review dropdown exposes captions and their source evidence. Correct upstream
readings/events and rerun `generate-captions` to regenerate wording and claims.

The renderer consumes accepted fusion evidence and uses fixed wording, with zero
additional VLM output tokens. Unknown champion identities remain HUD slot labels.
CS comparisons require resolved identity and verified playback continuity; they
never establish individual last hits. A last hit additionally requires a bound
minion and linked action/death frames. An ability hit requires a named ability,
bound target and linked cast/impact frames. Experimental actions remain excluded;
the production scene reader is still deferred, so current intake normally emits
displayed-state captions. Generated captions remain unreviewed and do not certify
their own factual accuracy. See [caption contracts and token budget](../../docs/league-caption-generation.md).

## Longleaf deployment

```bash
bash annotation/league/setup_longleaf.sh
LEAGUE_STAGE=preflight bash annotation/league/submit_longleaf.sh
```

Setup creates a Python 3.12 driver with FFmpeg/Tesseract and a separate `-vl`
vLLM environment. The selected vLLM build uses CUDA 13 dependencies; L40S runtime
probes passed on Longleaf, while model inference still requires validation. It does not
install Paddle or download icon assets. `LEAGUE_WITH_EVALUATION=1` installs/caches
the optional existing solver. `LEAGUE_WITH_EXPERIMENTS=1` installs the separate
`-audio` worker and caches event/audio checkpoints. `LEAGUE_WITH_TRACKING=1`
installs the separate gated SAM worker; checkpoint access must already be granted.
Each environment saves its resolved Linux dependency closure.

Important overrides: `LEAGUE_ENV`, `LEAGUE_RUN_DIR`, `LEAGUE_VIDEO`,
`LEAGUE_ASSET_CACHE`, `LEAGUE_VL_PYTHON`, `LEAGUE_AUDIO_PYTHON`,
`LEAGUE_TRACKING_PYTHON`, `LEAGUE_CONFIG`, `LEAGUE_HF_HOME`.
The remote run default ends in `midbeast-OLLZBe5-fg4-v2`.
All Longleaf write defaults are beneath `/work/users/a/l/alshen`: model downloads,
Conda packages, environments, run artifacts, temporary files, compiler caches and
Slurm logs. `longleaf_paths.sh` configures these without changing HOME or existing
home files. The submission helper creates `league_logs/` before submission;
direct `sbatch` logs go to the existing work root. Conda home registration and
Python bytecode writes are disabled.

Allocated jobs are offline. The node-local server binds loopback and evidence is
sent only to it. Preflight requires exactly one exposed GPU, records actual name,
UUID, capacity, CUDA and package versions, and checks FP8 capability. The `l40-gpu` partition contains both L40 and L40S nodes. Submission now
explicitly requests `gpu:nvidia_l40s:1`, confirmed available on2026-10-28.
Preflight still verifies the actual device; never report L40 measurements as L40S. Concurrency/batch is one, context 16K, vLLM memory budget .85.
GPU memory is sampled during loading/inference; exceeding .90 of capacity fails
validation. Hardware/runtime probes are recorded in VALIDATION.md; accepted-model
quality and complete-video throughput have not yet been established.

## Labels and release gates

`experiment --suite all --initialize` saves neutral native evidence and empty,
unverified manifests under the run's `benchmarks/`. Existing files are not
overwritten. Selection is an initial sampling grid, not proof that all difficult
cases are included. Inspect and adjust samples to cover HP changes, English/Korean
text where present, absent panels, overlays, pauses and rewinds. Supply independent
labels and playback-segment IDs: 20 development and 20 held-out frames from separate
segments, at least eight seconds apart across splits. Never copy predictions into
reference labels. Templates intentionally cannot pass a benchmark.

```bash
python -m annotation.league annotate --sample-times /path/to/crops.json
python -m annotation.league intake-benchmark --labels /path/to/crops.json --freeze
# Freeze returns exit 2 and writes a revised profile; re-extract under that profile.
python -m annotation.league annotate --sample-times /path/to/crops.json
python -m annotation.league intake-benchmark --labels /path/to/crops.json
python -m annotation.league annotate --pilot
# Only after quality and pilot gates pass:
python -m annotation.league run
```

Enabled-family targets: numbers ≥99% exact/90% readable coverage; text ≥95%/70%;
bars ≥95% within five percentage points/90%; presence ≥99%/95%. Parsed-number
correctness is also reported and does not replace exact transcription. Deferred
icon families do not block this reduced benchmark. Freeze uses development only;
draft labels cannot authorize release. Processing completeness replaces blanket
coverage across all 212 legacy regions.

The cold pilot uses a fresh dependent-reader cache and five dispersed one-minute
windows. Resumes preserve original per-frame costs and memory measurements. It
requires an actual L40S, all 300 sampled seconds, memory below 90% of capacity, no OOM and
a full-video intake projection ≤4 hours. Evaluation and experiments are measured
separately. Passing code tests cannot substitute for these gates.

## Experiments

All artifacts live under `experiments/`; failures save partial reports. They never
overwrite accepted observations or automatically feed question references.

For player actions, `action-window --start 30 --end 38` preserves prepare-only
behavior. Add `--detect` for CPU color-based player binding, CS/gold observations
and recall; add `--infer` for anonymous Qwen scene interpretation. Ability icons
are the sole player identity source. Version-3 action results and independent
labels remain isolated from production. Cache official descriptions with
`cache-abilities --reference-version 16.13.1 --champions all`; the reference
version never establishes the footage patch. The existing review app offers
`/actions` and a prediction-free `/action-labels` editor. See
[the action workflow](../../docs/league-champion-action-recognition.md).

New action runs use v3 only. `action_pipeline.py` holds shared frame/HUD helpers;
`action_benchmark.py` holds shared label/scoring helpers. The original and v2
execution APIs and the retired Paddle worker/dependency files have been removed.
Historical v2/v3 artifacts and corrections remain available in review. Existing
labels and caches are preserved; source-code fingerprints require fresh validation
after a code change, while compatible v3 model responses can still be reused.

- `crops`: isolated transcription diagnostics against the 40-frame labels.
- `events`: Qwen and BF16 InternVL3.5-14B-HF sequentially on the same twenty
  eight-second clips at 2 FPS, ten development/ten held-out. Native PNG evidence,
  model inputs and raw responses are retained. Reports include event precision,
  recall, actor attribution, timing error, abstentions, latency and memory by split.
  SAM 3.1 is a separate five-clip experiment with one to three manual point prompts,
  mask-IoU references and screen coordinates, not event/minimap/world tracking.
- `audio`: five minutes split across source-aligned intervals; packaged Silero VAD,
  Qwen3-ASR-1.7B, then its forced aligner, unloading between GPU models. Reports
  include transcripts, word timestamps, WER, name recall, alignment error and memory.
- `all`: also compares visual-only and audio-assisted event proposals. Commentary
  is untrusted audio evidence and cannot establish hidden HUD digits or actors.

Use `--labels DIRECTORY`; `--draft` allows diagnostic predictions with unverified
references but never acceptance. Tracking requires actual point prompts even in
draft mode. Complete independent event, transcript/alignment and mask labels before
reporting quality. Marengo retrieval capabilities are documented only: no public
offline L40S deployment weights have been established and no hosted inference or
external video uploads are implemented.

Model/processor hashes are in `models.local.json`; runtime libraries have separate
requirement files. Official implementation references:
[Qwen](https://huggingface.co/Qwen/Qwen3.8-27B-FP8),
[vLLM](https://recipes.vllm.ai/Qwen/Qwen3.8-27B),
[InternVL](https://huggingface.co/OpenGVLab/InternVL3_5-14B-HF),
[SAM](https://github.com/facebookresearch/sam3),
[ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf),
[Marengo](https://docs.twelvelabs.io/docs/concepts/models/marengo).

See `VALIDATION.md` for measured status and
`docs/league-intake-implementation-handoff.md` for continuation instructions.

## Qwen3.5 observation experiments

Four isolated suites (`hp-vlm`, `scene-identity`, `ability-alignment`,
`cooldown-identity`) and the explicit `qwen35` aggregate compare smaller HP readers,
champion boxes/name memory, HUD-to-scene ability association, and cooldown-resistant
identity. Existing `all` behavior is preserved. See the
[commands, labels, Longleaf runtime and evidence rules](../../docs/league-qwen35-experiments.md).
