# League annotation and memory evaluation

The default is a reduced Midbeast intake: native field crops → numeric Tesseract,
bounded Qwen transcription and direct bar measurement → schema-2 observations →
fusion → supported visual questions → optional independent solver evaluation.
**Release acceptance is pending. Full-video intake remains gated.**

## Source and enabled fields

`midbeast-source.json` pins the latest eligible upload found in the latest 20
Midbeast main-channel uploads inspected on 2026-10-27:
[OLLZBe5-fg4](https://www.youtube.com/watch?v=OLLZBe5-fg4),
“#1 DARIUS is TAKING OVER KOREA... *CRAZY 1v1s*”, 28:53, nonlive, 1080p60.
The downloaded MP4 hash is
`f4f36e1a274617ecbe224eaf7fc3f8375f757acb7a517bcc0ee7780fcd83f240`.
English speech eligibility still requires the audio pilot audit; source metadata
reports English and inspected frames show gameplay plus commentator facecam.

`midbeast.json` and `profiles/midbeast-1080p-v1.json` enable 21 calibrated fields:

| Reader | Fields |
|---|---|
| Tesseract, one fixed bright-glyph preprocessing, one call | Clock, team kills, selected K/D/A, CS, gold, level, displayed objective timer |
| Qwen3.8-27B-FP8 | Selected numeric HP, ten displayed scoreboard player-name slots, central notices including objective text |
| Direct pixels | Selected HP bar proportion, independent of numeric HP |

No champion identity is inferred from a player name or portrait. Unknown slots
retain their readable facts. Portrait/item/spell matching, ability ranks/cooldowns,
minimap tracking, gameplay-event reasoning and audio-derived game state are visibly
unsupported in production. Unknown patch is allowed for visual processing;
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
| `run` | Prepare, annotate, fuse, generate supported questions, optionally evaluate, report. |
| `review` | Inspect native/processed evidence, uncertainty, corrections and calibration. |
| `intake-benchmark --labels …` | Score exact text, parsed numbers, bars and presence separately. |
| `experiment --suite crops\|events\|audio\|all` | Separate experimental artifacts; no production evidence changes. |
| `cache-models --suite intake\|events\|audio\|all` | Download immutable pinned checkpoints for later offline use. |

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
