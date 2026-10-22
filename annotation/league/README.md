# League visual annotation and memory evaluation

An end-to-end, pixel-only pipeline: fixed observers → evidence timeline → deterministic
question/answer construction → independent Qwen video evaluation. No match API,
telemetry, engine logs, audio, coaching, or inferred intent is used as ground truth.

**Review is optional.** Every batch stage works unattended. Automatic answers are
`unreviewed`; only a human correction/approval creates `human-verified` references.
Reports separate benchmark accuracy from agreement with provisional references.

## Local preparation and review

Run from the repository root with Python 3.10+ and FFmpeg/ffprobe on PATH.
Preparation, fusion, question templates, reference lookup, and reporting use the
standard library. Install the review interface separately:

```bash
python3 -m venv .venv-league
.venv-league/bin/python -m pip install -r annotation/league/requirements-review.txt
python3 -m annotation.league prepare
.venv-league/bin/python -m annotation.league review
```

Open <http://127.0.0.1:8765>. Before inference, choose **Layout/crop previews** to
inspect the real source video and fixed regions. Later, review observations,
events, and questions; edit their JSON fields and save/verify/reject. Click an
evidence crop or “Seek to evidence” to synchronize the video. The UI loads no model.

The default run is `data/league_runs/aatrox-26.13`; source data and results are
ignored by Git. Use `--run-dir` before the command to keep runs separate. The
default input is the existing `xICbe-OOPvE.mp4`, patch 26.13 / DD 16.13.1 / CD 16.13.
`--config` selects a new fixed layout or patch. A source/interval change requires a
new run directory. Configuration/model/code changes invalidate inference task keys.

```bash
python3 -m annotation.league --run-dir data/league_runs/aatrox-reset prepare \
  --video data/youtube/lol_domisum_aatrox/xICbe-OOPvE.mp4 --start 760 --end 810
```

## GPU setup and Longleaf

Use a separate GPU environment with Python 3.12 for the locked setup. The initial stack
is pinned in `requirements-gpu.txt`. Its actual CUDA environment is captured at
runtime; CUDA inference and memory fit must be validated on the allocated L40.
`inference-environment.lock.txt` freezes the complete set tested for CPU image/video
preprocessing; it requires Python 3.12. The CUDA-specific wheel dependencies are
resolved by PyTorch on Linux and recorded by preflight.

For a one-time interactive setup on Longleaf, transfer the video to the default
repository path (or set `LEAGUE_VIDEO`) and choose persistent storage for the
model downloads. The setup script creates/updates the environment, installs and
checks every pinned Python package, installs FFmpeg/ffprobe in the Conda
environment, prepares the run and caches patch references and both model
checkpoints. Keeping FFmpeg in the environment makes it available after Slurm
purges modules. Setup runs on a login node and does not require a GPU allocation:

```bash
cd /users/a/l/alshen/vgextract/vgextract
bash annotation/league/setup_longleaf.sh
```

The environment and model-cache defaults match `slurm_annotate.sl`. The model
cache defaults to `/users/a/l/alshen/hf-cache`. Set
`LEAGUE_ENV` or `LEAGUE_HF_HOME` before setup only if you need different paths.
The dedicated `LEAGUE_HF_HOME` setting is used by both setup and Slurm, so a stale
`HF_HOME` value cannot redirect the offline job to a different cache. The lock file
pins the tested preprocessing and inference stack;
the setup script checks exact installed versions and imports the Qwen inference
APIs. CUDA availability and BF16 support are checked separately by the allocated-
GPU `preflight` stage.

If setup reports that `av==19.0.1` requires Python >=3.12, the installation is
using an older Python. Rerun the updated setup script: it requests Python 3.12,
pip and FFmpeg for existing environments as well as new ones. Before installing
the lock it prints and checks the exact environment interpreter, Python version,
prefix, local pip and writable site-packages. A missing local pip or an unwritable
prefix can otherwise cause pip to fall back to an external or user installation.
Setup explicitly disables user installs; setup and Slurm both ignore inherited
`PYTHONHOME`/`PYTHONPATH` and user-site packages. Packages left in `~/.local` by an
earlier attempt do not need to be deleted. If the prefix is not writable, choose
a directory you own with `LEAGUE_ENV` and use that setting for both setup and
Slurm submission. Conda initialization and activation tolerate unset shell
variables while the rest of each script retains strict shell checks.

`cache-references` caches the complete static champion roster/details and item
dictionary, including available same-patch CDragon fallbacks. Unresolvable formula
graphs remain explicit gaps. To cache a small pilot instead, use
`cache-references --champions Aatrox Jax`; `--base-only` caches just roster/items.
Missing offline knowledge yields an unresolved lookup rather than remembered kits.
`cache-models` downloads both full checkpoints; do this outside a GPU allocation.
The Aatrox configuration pins the two model commit SHAs, not moving model branches.

Setup activates Conda inside its own process. For later manual commands, activate
the environment in your terminal or use its full Python path:

```bash
HF_HOME=/users/a/l/alshen/hf-cache /work/users/a/l/alshen/conda_envs/videogame-league/bin/python -m annotation.league --debug --run-dir data/league_runs/aatrox-26.13 cache-models
```

Activate Conda for media stages such as `processor-preview` so the environment's
FFmpeg/ffprobe executables are also on `PATH`.

`cache-models` reports the actual snapshot paths. If a downloaded weight-index
JSON is malformed, it fetches that small file again once, retaining the downloaded
weight shards. Remaining JSON errors identify their source file; `--debug` also
includes the traceback. The offline GPU job does not attempt repairs or downloads.

Inspect actual processor-resized inputs without loading weights:

```bash
python -m annotation.league processor-preview
```

This downloads only processor/tokenizer files, produces image and video-sequence
audits, and records the resolved environment. `annotate` also checks preprocessing
before starting inference. Resizing happens once using Qwen's official utilities,
with source PTS supplied explicitly alongside the frames.

On an allocated GPU:

```bash
python -m annotation.league --offline preflight
python -m annotation.league --offline run
```

For Longleaf, submit from the repository root. The batch file defaults to the
environment, model cache, source video and run directory used by setup, so the
full run takes one command:

```bash
sbatch annotation/league/slurm_annotate.sl
```

To run the GPU preflight first, submit it with a stage override, wait for it to
finish successfully, then submit the full run:

```bash
sbatch --export=ALL,LEAGUE_STAGE=preflight annotation/league/slurm_annotate.sl
# Wait for the preflight job to finish successfully, then:
sbatch annotation/league/slurm_annotate.sl
```

The launcher defaults to `l40-gpu`, `gpu_access`, one GPU, eight CPUs, 64 GB host
memory, and 12 hours. Override `LEAGUE_PARTITION`, `LEAGUE_QOS`, `LEAGUE_GRES`,
`LEAGUE_CPUS`, `LEAGUE_MEM`, or `LEAGUE_TIME`, or pass sbatch flags to the launcher.
These settings follow [UNC's GPU job examples](https://help.rc.unc.edu/longleaf-slurm-examples/).
Longleaf's partition may allocate an L40 or L40S; the exact GPU is recorded.

`LEAGUE_STAGE` can select a single stage; `LEAGUE_VIDEO`, `LEAGUE_CONFIG`,
`LEAGUE_START` and `LEAGUE_END` select the source/configuration/span. The script
uses offline model/reference caches, sends no email, and forwards shutdown signals.
Submit the batch file from the repository root; Slurm writes `%x-%j.out` and
`%x-%j.err` there. The `submit_longleaf.sh` wrapper remains available for creating
`logs/` and overriding the default resource requests. Resubmit to resume. Completed
HUD microbatches and observation tasks are retained.
Use `LEAGUE_RETRY_FAILED=1` to retry unresolved/partial tasks after inspecting them.

## Stages and evidence contracts

```bash
python -m annotation.league annotate
python -m annotation.league fuse
python -m annotation.league generate-qa
python -m annotation.league evaluate
python -m annotation.league report
```

`run` performs preparation through reporting without opening review. `generate-qa`
and `evaluate` refresh stale upstream derived stages. After corrections, rerun
`evaluate` and `report`; reporting refuses stale predictions. Neither generation nor
evaluation requires verified references.

- **Observers:** Instruct reads selected HUD, both sidebars, scoreboard/inventory,
  minimap, global HUD, objective panel and notices. Thinking reads gameplay and a
  fixed full frame. One checkpoint resides on GPU at a time, in BF16, batch size one.
- **Discovery:** 1 FPS, eight-second windows, two-second overlap. HUD requests use
  compact per-entity values and two-frame microbatches to fit the 2,048-token output
  cap; Thinking receives the temporal window and has an 8,192-token cap. Total
  request context is capped at 32,768 tokens. Caps and microbatch size are configurable.
- **Pilot:** lane, reset, late fight, and a quiet-control location are processed
  before the full pass. A route that consistently fails or reports incompatible
  geometry stops the batch and leaves inspectable artifacts. This technical check
  does not establish semantic accuracy or require human review.
- **Automatic review:** deduplicated, prioritized requests use the same boxes at
  4 then 16 FPS with temporal padding and 2×/4× pixel budgets. Up to two rounds;
  review visual tokens are capped at 50% of discovery, reserved proportionally for
  each model role. Every lookup generation counts. Budget exhaustion preserves
  partial evidence and creates an unresolved review item. Seeded quiet audits use
  the same limits. Optional human review is separate from automatic review.
- **Fusion:** source timestamps and game clocks stay separate. Field values carry
  visible/unreadable/not-shown/occluded/unknown status, confidence, identity/slot
  binding, source pixels and model provenance. Conflicts stay unresolved. Last
  confirmation does not fill an unobserved interval; disappearing wards do not
  become destroyed wards. Duplicate notices and overlapping routes are reconciled.
- **Composite events:** labels require declared visual criteria. An objective LIVE
  timer is not a contest. Episodes link observed events without claiming causality.
- **Knowledge:** static-only allowlisted DD/CD hosts, exact patch/identity matching,
  hash-verified offline cache, units and unresolved variables retained. At most
  three lookups per inference task. Korean player text is preserved; player names
  never become champion names by fuzzy matching.

Main artifacts (JSON/JSONL schema version 1):

| Artifact | Contents |
|---|---|
| `manifest.json`, `config.json`, `environment.lock.json`, `preflight.json` | Source SHA256, interval, patch, fixed layout, exact model revisions, runtime/hardware |
| `previews.jsonl`, `processor-previews/`, `inference/` | Native crops, actual resized model pixels, frame timestamps, token counts, completions, static-lookup transcripts |
| `tasks/`, `coverage.jsonl`, `pilot_report.json` | Atomic resume records, sampled coverage, technical pilot findings |
| `observations.jsonl`, `states.jsonl`, `conflicts.jsonl` | Raw field readings and conservative fused state |
| `event_candidates.jsonl`, `events.jsonl`, `episodes.jsonl` | Atomic/composite candidates, deduplicated events, objective episodes |
| `review_requests.jsonl`, `human_review_queue.jsonl`, `corrections/` | Automatic rereads, unresolved items, append-only human revisions |
| `questions.jsonl`, `question_coverage.json` | Typed answers, prompts, capability/horizon/category, evidence, ambiguity and coverage requirements |
| `solver-inference/`, `predictions.jsonl`, `scores.jsonl`, `report.json` | Independent solver inputs, predictions, stratified accuracy/temporal/efficiency reports |

Raw predictions are retained. Corrections are separate immutable records;
downstream artifact hashes include their revisions. A single batch writer owns a run;
the review service can append corrections while annotation is running.
Each derived stage freezes its correction revision, so concurrent review cannot
silently change a reference halfway through an evaluation. Question-limit and
solver-budget changes retain annotation tasks. An unchanged solver input reuses its
original prediction after answer/status corrections, while scores are regenerated.

## Questions and evaluation limits

Deterministic queries establish answers before language generation. The initial
release uses templates directly. It targets at most 60 questions, balanced across
episodic retrieval, persistent state, cumulative counters, event linking, trajectory
memory and recurring observed behavior. Missing evidence yields a coverage shortfall.
Counts use explicit displayed counter changes rather than assuming every cast was
sampled. Repeated behavior names specific observed events; sparse state pairs do
not assert continuous visibility. Precise casts below adequate sampling density are
excluded. Editorial-overlay/layout issues disqualify affected question intervals.

`visual_only` questions permit identity/icon reference responses only;
`mechanics_augmented` questions permit patch mechanics. Inventory questions can
combine visibly resolved item IDs with verified static item stats. A static lookup
never establishes that an item or cast appeared in the match.

The Thinking solver receives one uniformly sampled full-frame sequence over the
question interval, capped at 128 frames, plus the question and allowed static
references. Construction logs, answers, gold timestamps and prior summaries are
excluded by an allowlisted input builder. The declared interval is visible to the
solver; gold evidence does not select its frames. This protocol can later be
replaced by a separately labeled streaming/external-memory implementation.
Local questions can retain up to native 1080p detail; preprocessing reduces pixel
budgets when necessary to fit the declared context. Fast-action questions request
their declared minimum FPS within the same frame cap. Actual processed resolution
and retained evidence are audited for every request.

Reports include exact typed-answer agreement, abstention and coverage, one-to-one
temporal matching, interval IoU/timestamp error, and input-sampling audits. Temporal
proximity is measured automatically; actual HUD legibility remains an explicit
review dimension. Wrong answers are reported as reference disagreements, not
automatically labeled hallucinations. A partial human review cannot establish
exhaustive missed-event recall. Cost is unset unless `report --gpu-hour-rate` is
provided. Inference timing and full pipeline timing are identified separately.

## Validation

```bash
python3 -m unittest discover -s tests -p 'test_league*.py' -v
# To include the review API integration test:
python -m pip install -r annotation/league/review-environment.lock.txt
python -m unittest discover -s tests -p 'test_league*.py' -v
```

Fixtures inject a test runner to exercise interruption/resume, budgets, fusion,
question families, correction propagation and solver isolation. The production CLI
has no synthetic inference backend. Real FFmpeg tests verify decoding and PTS.
Local Aatrox processor previews exercise the real pinned Qwen image/video processors.
GPU generation, full-game annotations, model accuracy and L40 memory/throughput
measurements require a real GPU allocation; processor success is not evidence of
those results. See `VALIDATION.md` for the checks actually performed.
