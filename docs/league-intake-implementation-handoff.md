# Current handoff — League player actions v3, 2026-10-30

## Scoreboard portrait work — 2026-11-11

The scoreboard portrait candidate scores the center and four one-pixel shifts.
The focused comparison recovered three visible portraits, raising the inspected
probe from 17 to 20 accepted rows out of 30. See the
[scoreboard and body handoff](league-body-recognition-handoff.md) for the remaining
occlusions, reproducible evidence, and the next implementation phase.

## Source cleanup — 2026-11-02

The local package now executes only v3 actions. Shared frame/HUD helpers remain in
`action_pipeline.py`; label/scoring helpers remain in `action_benchmark.py`. The
original action prototype, v2 runners/assistance handling and unused Paddle OCR
worker/dependency files have been removed. Current CLI and v3 entry points remain;
review still supports historical v2/v3 results and corrections. Saved artifacts,
profiles, labels and caches have not been deleted or migrated. Older implementation
sections below describe history, including files that no longer exist.

Production annotation now separates release checks, sampling/resume, per-frame
reading and reporting. Crop/audio/event/tracking experiments use separate helpers
with the same sequential worker lifetime and partial-report behavior. Refactored
code changes implementation fingerprints and requires normal release/freeze
revalidation; compatible v3 response caches remain reusable. Fresh benchmark
labels are v3 directly; forks from older clips clear labels and attestations.

Cleanup validation: **204 repository tests pass with no skips** using the local
CPU intake/review dependencies. CLI help and whitespace checks pass; v3 scene/HUD
request schemas match their pre-cleanup versions. Python source decreased from
10,500 to 9,127 lines. These CPU/mocked checks do not establish model accuracy or
full-video release acceptance.

## Current implementation and directories

Player binding, CS/gold observations, recall tracking, official ability descriptions
and independent review/scoring now run inside the existing League annotation
pipeline. New runs use `action_v3.py`, `action_body.py`, `action_events.py`,
`action_benchmark_v3.py` and `profiles/midbeast-actions-1080p-v2.json`.
Artifacts are isolated under `experiments/actions-v3`; no production observations,
events or questions have been promoted. Q/W/E/R reference matching is the sole
automated kit identity source. New Qwen prompts/schemas have no identity opinion,
and receive deterministic local player track IDs and approximate regions.

Continue in these existing directories; the user explicitly rejected creating
another checkout:

- Local source: `/Users/alex/Desktop/Projects/videogame`.
- Longleaf source, updated in place:
  `/work/users/a/l/alshen/league-midbeast-validation/source-20261030-actions-r7`.
- Existing Longleaf run:
  `/work/users/a/l/alshen/league-midbeast-validation/run`.
- Existing Longleaf assets:
  `/work/users/a/l/alshen/league-midbeast-validation/assets`.

Authenticated SSH uses `unc-cluster` and
`/private/tmp/league-longleaf-01a16144/control`. All remote writes remain beneath
`/work/users/a/l/alshen`. No hosted inference, uploaded source video, Git commit
or PR was introduced. Preserve the existing working-tree edits, including the
preexisting ingestion bytecode changes.

The version 16.13.1 official Data Dragon catalog completed for all 173 champions,
with passive/Q/W/E/R names, descriptions, available tooltips, icons, raw responses
and hashes. It is cached locally at `data/league_assets/abilities/16.13.1` and
remotely at `assets/abilities/16.13.1`. The footage patch remains unknown and
mechanics stay disabled. Runtime model/GPU-class/software identity enters inference
caches; physical GPU UUID remains provenance so equivalent GPUs can resume.

## Review and independent annotation

Start the existing review server with `python -B -m annotation.league review`.
`http://127.0.0.1:8765/actions` shows synchronized source playback, HP/actor
overlays, raw counter/recall evidence, resampling, responses and official
descriptions. Experimental corrections stay isolated.

The user requested both a personal review option and a separate handoff for
another agent to label without detector outputs. Use
`http://127.0.0.1:8765/action-labels` for source-only reference annotation and
`docs/league-actions-labeling-handoff.md` for that separate task. No reference
labels or corrections were saved during UI verification. All twenty labels are
still unverified, development choices are unfrozen, and held-out scoring has not
run. Accuracy metrics remain null; missing-case coverage must be independently
established. Do not give the labeling agent this implementation-result handoff
or prediction artifacts.

## Development measurements and current GPU pilot

These are detector outputs on inspected development footage, not accuracy scores:

- Source 30–38 seconds, local experiment `a321b36727b3c06c41e5`: 125/128 frames
  bound, one aggregate CS increase of 1, displayed gold changes recorded separately.
  Three Q cooldown-visibility changes remain unresolved, not confirmed activations.
  CPU detection took 38.00 seconds; preparation plus detection 64.52 seconds;
  process peak host memory 492,683,264 bytes. No GPU requests.
- Source 747–759 seconds, experiment `94e617d8815a39baac41`: one observed recall
  interval 747.630–755.438, `ending_unknown`, including three explicitly uncertain
  tooltip-occluded frames. CPU detection 65.47 seconds, total 103.67 seconds,
  process peak host memory 390,676,480 bytes. No completion inferred.
- Source 448–456 seconds, experiment `574994bb46b279e96b85`: 113/128 frames bound,
  two unresolved W visibility changes. CPU detection 45.83 seconds, total 72.82
  seconds. Actual rapid activation coverage is not verified.

The CPU body/farming and recall snapshots were imported into the existing
Longleaf run as read-only review aliases `e849882bbca11606c4bb` and
`43a1328dc61025c4a100`; all 128/192 model-input image hashes matched. Alias imports
preserve both hosts' canonical output when decoder/image-library differences
produce different image bytes. Never overwrite an active canonical experiment
with another host's images merely because source/recipe keys match.

Longleaf job **7355970** is the resumed two-clip scene/cast development pilot,
running on an L40S with the pinned local model. Its script is
`league-midbeast-validation/artifacts/actions-v3-cast-pilot.py`; aggregate status
is `run/experiments/actions-v3/cast-pilot.json`, and logs are
`logs/actions-v3-7355970.{out,err}`. It reuses completed requests from cleanly
checkpointed job 7353514. Dense overlapping contexts are coalesced, individual
native brackets preserved, and dense HUD requests target changed-slot brackets.
Do not change active inference modules on disk mid-run. Await actual completion
or retain the failure/partial report; do not infer success from allocation time.
Both development clips are candidates: neither isolated-cast nor rapid-cast
coverage is established until source annotation confirms it.

Validation before the 2026-11-02 cleanup: **218 repository tests passed**, including independent scoring,
unresolved-transition accounting, false green/minion/portrait rejection, selection
and continuity changes, counter fluctuation, recall lifecycle/censoring, invalid
citations, equivalent-GPU resume, interrupted inference and production isolation.
The generic workflow and review are implemented. Full pilot accuracy/completion
still requires independent labels, frozen development choices and resource results
from completed scene inspection. See `docs/league-champion-action-recognition.md`
for CLI commands and evidence semantics. Source synchronization proofs live in
`data/league_midbeast_dev/longleaf-20261030-actions/v3-source-sync.json` and the
existing remote `artifacts/v3-source-sync.json`.

The CPU-only development draft completed all ten development clips in 618.55
seconds total (including preparation), with peak host process memory 500,924,416
bytes and zero GPU calls. Held-out clips were not processed. Its aggregate report
is `experiments/actions-v3/benchmarks/d25a77a0ef502e74c5cf.json`, and a compact
resource/output summary is in
`data/league_midbeast_dev/longleaf-20261030-actions/v3-core-development-summary.json`.
These outputs and the running GPU pilot are being synchronized by the one-shot
`sync-v3-evidence.py` helper in that same development-artifact directory. It saves
progress/host aliases to `v3-evidence-sync.json` and waits for job 7355970 before
pulling its completed evidence. Its local log is
`/private/tmp/league-v3-evidence-sync.out`. A stable interrupted GPU checkpoint is
already available locally as `c6910fc8ca8320e612c5`. Local review can inspect its
raw scene proposals without treating them as verified labels.

## Earlier v2 implementation history (superseded by v3 above)

**Local implementation is complete enough for GPU integration testing; release and
rollout are not complete.** Preserve all existing edits. No commit or PR created.
Experimental L40S measurements are recorded below; production acceptance remains open.

## Champion action implementation and icon identity — 2026-10-30

The new universal action workflow is integrated into `annotation/league`, using
`action_pipeline.py`, `action_state.py`, `action_icons.py`, `action_benchmark.py`
and the existing review app's `/actions` view. See
`docs/league-champion-action-recognition.md` for contracts and commands. The old
prepare prototype remains intact. No production observations/events/questions
have been promoted from the action experiment.

The user restored Longleaf authentication and authorized continuing there. Use
`unc-cluster` through `/private/tmp/league-longleaf-01a16144/control` while it is
authenticated. All remote writes remain under `/work/users/a/l/alshen`. This
supersedes the older connection status below.

Latest user steering makes Q/W/E/R ability icons the primary HUD identity signal;
skin recognition is not required. The generic matcher uses the full versioned
spell catalog, compares only like slots, requires three agreeing slots with no
strong contradictory slot, and rechecks every 0.25 seconds. Default experimental
cosine/margin thresholds are .45/.06. A separate unassisted icon-only Qwen request
is preserved for comparison. Reference-assisted identity is explicitly labeled
even inside an unassisted scene experiment. Reference asset version 16.13.1 does
not establish the recording's patch; mechanics remain disabled.

Actual development findings at source seconds 30–38:

- The first frame's Q is obscured by cooldown. W/E/R independently match Darius.
- Across 32 reference checks: 31 Darius, one unknown. These are output counts,
  not benchmark accuracy. The imported snapshot has 207 source frames including
  native resampling; repeated rows are not independent decisions.
- Unassisted Qwen portrait-plus-icons returned Riven; icons alone returned
  Vladimir. A verified five-champion blue-roster subset with portrait references
  returned Darius. These are single-selection diagnostics, not a full roster eval.
- Scene requests repeatedly hallucinated Vladimir, invalid body coordinates and
  attacks without animation evidence. Revalidation retains identity errors for
  scoring while rejecting unsupported binding/actions; it does not turn wrong
  identities into hidden abstentions.
- Snapshot: 18 completed requests, 76,627 input / 20,799 output tokens,
  2,029.95 cumulative request seconds, peak reported GPU memory 39,691,747,328
  bytes on NVIDIA L40S. Queue/overlap can contribute to request wall time.
- Local snapshot and raw model inputs are under
  `data/league_runs/midbeast-OLLZBe5-fg4-v2/experiments/actions-v2/913d33a46f064d1433d7`.
  Identity findings are also in
  `data/league_midbeast_dev/longleaf-20261030-actions/identity-findings.json`.
  Its `partial_snapshot` status is intentional; there are no accepted actions in
  that snapshot. Review source playback, icon alternatives and raw responses at
  `http://127.0.0.1:8765/actions` while the local server is running.

GPU job 7300739 processed adaptive scene inspection from
`source-20261030-actions-r3`; it does not include the latest icon integration.
After preserving its evidence, SIGTERM was sent to its driver PID 1119400.
Checkpointed shutdown is confirmed: `Shutdown requested; committed tasks can be
resumed`. The final driver report retains 23 completed scene/HUD requests
(124,965 input / 28,871 output tokens, 2,723.78 cumulative request seconds),
excluding the three separate identity probes. The old driver left its aggregate
resource field null; those totals were reconstructed from its retained request
records without altering the original report. Final metadata and original report
are in `data/league_midbeast_dev/longleaf-20261030-actions/checkpointed-*`.
The run reached native inspection but is incomplete. Its requests are resumable;
its one-hour allocation
does not imply that the exhaustive native stage is complete. Latest component
diagnostics used `source-20261030-actions-r5` against the existing warm server.
The final tested code was synced separately to `source-20261030-actions-r7`;
the older run's checkout was preserved. The remote reference catalog is at
`league-midbeast-validation/assets/ability-icons-16.13.1/manifest.json`.

The separate 20-clip benchmark exists with unverified labels. Development choices
have not been frozen and held-out scoring has not run. Independent annotations,
missing-case coverage, measured transition/binding accuracy and completed resource
evaluation remain required before calling the full pilot complete. No full-video
or production acceptance is claimed. All 194 repository tests pass, including
new temporal, cache/interruption, review isolation and icon-consensus contracts.

Latest user steering proposes matching HUD identity to the uniquely colored
overhead player health bar. At source 30 seconds, the player bar is green and
opponent bar red. The four minimap-adjacent portraits are teammates and all have
green HUD bars, so their color does not individually bind allies. Implement and
measure the player-bar/neighboring-body tracking cue next; it is documented as
observed, not already implemented. This can avoid requiring skin recognition for
the HUD owner's body while preserving independent binding evidence.

## Champion action follow-up — 2026-10-29 (historical)

User requested champion behavior, ability casts and item use within a chosen
timeframe, and selected the current Midbeast Darius video. Added an isolated
`action-window` command; see `docs/league-champion-action-recognition.md` for
the current perception pipeline and new contract. Normal production intake is
still HUD transcription at1FPS, with no released cast/item-use recognition.

Prepared source seconds30–38 at4FPS with explicit native HUD crop
`700 930 1225 1080`:32source-PTS frames, five overlapping packets with at most
eight images each. Actual input PNGs combine a960pixel-wide masked scene and
same-frame native HUD. Untouched native PNGs, exact prompts/schema, input hashes,
an unverified independent-label template and report are local under:
`data/league_runs/midbeast-OLLZBe5-fg4-v2/experiments/actions/4ad3411d06cffff4275c/`.

`--infer` uses the existing pinned local Qwen runner with an opt-in8image limit.
Default4image intake identity is unchanged. Experimental proposals require actual
frame citations; cooldown/charge/inventory changes cannot alone establish a cast
or item use. Actor/ability/item names may remain unknown. No continuous identity
tracking, semantic deduplication, validated cut handling or complete use counts
are claimed. Accepted observations/events/questions are untouched.

**No action inference ran and no action accuracy was measured.** Local173tests
pass, including nine new action contract/isolation tests; real32frame preparation
and visual inspection of the combined input passed. No commit/PR created.

The previous SSH control socket `/private/tmp/league-longleaf-01a16144/control`
is now absent (local directory inspection,2026-10-29); no SSH login, reconnect,
remote write or GPU job was attempted. This supersedes the connection status
below. Following the standing disconnected-state instruction, save this handoff
and stop remote work without asking the user to reconnect. Next remote action,
when an authenticated session already exists, is to transfer the new code to a
fresh source directory under `/work/users/a/l/alshen`, run the explicit eight-image
action inference on L40S, and independently review raw candidates against native
video before developing/scoring labels. The action inputs' GPU resource budget
has not been tested. All earlier release/benchmark gates remain pending.

## Live remote validation — 2026-10-28

The user restored the SSH master `/private/tmp/league-longleaf-01a16144/control`;
it is working. **If authentication expires again, save this handoff and stop.**
Do not ask the user to reconnect. Historical disconnected-state instructions are
superseded by this live state.

**All future Longleaf writes must be beneath `/work/users/a/l/alshen`.**
`longleaf_paths.sh` redirects installer/model/Conda caches, compiler files,
temporary files and Slurm logs. Workers independently redirect these paths.
HOME is unchanged; existing home files were neither removed nor moved.
The installed Conda supports `CONDA_REGISTER_ENVS=false`, preventing home registry
writes. Python bytecode writes and worker telemetry are disabled. Unix sockets and
temporary files use short `/work/users/a/l/alshen/tmp` to respect Linux's107-byte
Unix socket path limit. Other worker caches use assets/runtime-cache.

Remote root: `/work/users/a/l/alshen/league-midbeast-validation`.
Newest frozen source: `source-20261028-r6`. Driver/VL/audio venvs and immutable
checkpoint manifests are under root; source MP4 hash matched the local pin.
The original League Conda environment is healthy and unchanged: Python3.12.15,
Torch2.8.0, Transformers4.57.1, imports and pipcheck pass; no vLLM/qwen3_5 support.
Registered environments also include base,r,avp,learnerpr. OCR/new workers are venvs.

- Hardware probe7064617: g181005, NVIDIA L40S46,068MiB, driver595.58.03.
  CUDA/runtime probe7067005: Torch2.13.0+cu130, Transformers5.18.0, vLLM0.32.0;
  BF16/FP8 GEMM and vLLM import passed. Jobs explicitly request nvidia_l40s.
- Qwen7071029 completed three development frames30,90,750 in530.93stage seconds;
  peak39,289,094,144GPU bytes. All3 HP parses matched independently inspected
  native crops. Runtime chose Marlin weight-only FP8. **Names failed**, including
  clearly visible Singed reported not_shown. The21/21 accepted-or-absent log line
  does not establish semantic quality. Original report preserved at
  artifacts/qwen-smoke-7071029-report.json and in local ignored evidence.
- Name diagnostic7072632 completed on g181010: semantic/neutral single requests
  both read Singed; grouped outputs varied, including false absence with neutral
  IDs. No target examples were used. Report artifacts/name-binding-7072632/report.json.
  Root logs/name-binding-7072632.{out,err}; outputs are experimental only.
- Source-r6/profile revision2 now use individual name requests. Name/HP text
  required in a detected panel cannot be accepted absent: one retry, then unknown.
  **Normal annotate smoke7073515 completed**, one cached frame at750seconds.
  Eight name crops returned text, two remained unknown/unreadable; HP matched.
  No candidate is independently benchmark-verified. Stage228.42seconds includes
  cold model startup and warm decode; peak39,274,414,080GPU bytes. This is not the
  cold five-minute intake timing benchmark. Script artifacts/name-smoke-r6.sl;
  logs/name-smoke-7073515.{out,err}; host allocation reduced to64GB with scontrol.
  Report/task preserved in artifacts/name-smoke-7073515-{report,task}.json.
- ASR/aligner7071860 **completed_experimental** on L40S:300seconds input across
  five intervals,72speech chunks,624in-span word timings,2rejected timings,
  78.14worker seconds,4,753,195,008peak GPU bytes; resource_passed true.
  artifacts/audio-pilot-7071860.json and experiment-cache/audio report retain all
  transcriptions and rejected raw candidates. Independent WER/name/alignment
  scoring remains absent. Audio never feeds accepted visual facts/questions.
- The audio pilot is separate from the required five-minute **intake** runtime
  pilot. No fresh intake pilot, held-out benchmark acceptance or full-video run
  has passed. Forty-frame manifests remain unverified; no manufactured labels,
  playback discontinuities or benchmark precision claims.
- Local164 repository tests pass. Linux126 League tests passed in
  artifacts/league-tests-r9.out against source-r6.
  FFmpeg is on PATH, review API tests included; existing edits/pyc preserved.
- InternVL/SAM experiments have not run; weights are not cached. No hosted
  inference or external video uploads. No commit or PR created.

Resolved integration failures: native Conda ICU/SQLite CXXABI mismatch (prepend
base Conda lib, leaving packages untouched); home .triton quota (explicit work
caches); long IPC path (short work tmp/RPC directory); NFS anonymous temporary-file
cleanup (close named probe before unlink); fractional source PTS versus requested
sample_time; Torch GPU UUID prefix. ASR out-of-span alignment words are rejected,
not clamped or allowed to invalidate completed transcription. Original failed
jobs7067349,7068523,7068769,7070562,7070609 remain inspectable. Old Domisum CPU
pilot6151847 failed130:0 after42:49/280frames with SIGTERM, no release acceptance.

No validation jobs remain active at this handoff. SSH remains connected.
Next: independently verify benchmark labels and freeze development choices before
held-out scoring; inspect unverified name candidates against native pixels. Measure
fresh five-minute intake cold processing separately from cache reuse. Full video
requires quality, memory and projected<=4hour gates. Keep event/audio experimental.
If SSH expires first, update this file and stop under the standing instruction.

## Approved current scope

The latest approved plan supersedes the earlier OCR-only default. Use only
Qwen3.8-27B-FP8, bounded transcription for HP/names/notices; Tesseract only for a
small numeric field list; direct bar measurements. No Paddle worker or icon assets
in normal setup/intake. Icons/ranks/cooldowns/minimap remain deferred. Mechanics
are off and patch may be unknown. The existing optional QA solver stays separate.
Experimental event/audio/tracking results never become accepted game facts or
visual-only question references. No hosted inference or external video uploads.

## Actual local source and evidence

- Repo: `/Users/alex/Desktop/Projects/videogame`.
- Source: `data/youtube/lol_midbeast/OLLZBe5-fg4.mp4`, 1080p60, 28:53.
- Title: #1 DARIUS is TAKING OVER KOREA... *CRAZY 1v1s* (2026-10-27).
- SHA256: `f4f36e1a274617ecbe224eaf7fc3f8375f757acb7a517bcc0ee7780fcd83f240`.
- Pin/catalog of twenty uploads: `annotation/league/midbeast-source.json`.
- Main-channel ID: `UC6mWuX4R9Yn5y0fWCwzqiSg`.
- New run: `data/league_runs/midbeast-OLLZBe5-fg4-v2` (ignored data).
- Native twelve-frame calibration/smoke evidence: `data/league_midbeast_dev`.
- New profile: `annotation/league/profiles/midbeast-1080p-v1.json`, 21 fields.
- Both global HUD anchors matched all twelve inspected frames and all three real
  CLI previews. Intermittent panel anchors and semantic accuracy need benchmarks.
- Twelve clocks track source +33 seconds. No actual pause/rewind was demonstrated.
  Locate genuine difficult cases and playback discontinuities before certifying
  benchmark coverage; do not invent labels or segment boundaries to pass gates.
- English metadata plus gameplay/commentator facecam observed; actual English
  commentary still needs listening/transcription validation.
- Forty neutral native frames and empty unverified label manifests were saved in
  the run's `benchmarks/`. Portable templates are under
  `annotation/league/benchmarks/midbeast-OLLZBe5-fg4/`.
- Earlier windows are development and later windows held-out, excluding the twelve
  inspected calibration frames from held-out eight-second neighborhoods.
  Split playback IDs are intentionally null until independently checked.
- All crop/event/audio/tracking labels remain unverified. No Domisum labels or
  thresholds were reused. Full-video intake remains blocked.

## Implemented locally

- `hybrid_intake.py`: production orchestration, enabled fields, panel gating,
  content/settings caches, playback metadata, resume, completeness and release gates.
- `numeric_reader.py`: one fixed bright-glyph preprocessing and Tesseract attempt;
  direct bars. Tightened level/gold/KDA/timer geometry after native development
  smoke checks. A red zero at second30 remained rejected; no coverage claims.
- `vl_reader.py`: four labeled crops/request, thinking off, deterministic512token
  strict JSON; explicit unreadable/not-shown, local parsing, bounded singles retry,
  conflict rejection. Native/processed PNGs, prompts and raw responses retained.
- `local_models.py`: immutable model pins, downloaded file SHA256 validation,
  offline loopback-only vLLM process, batch/concurrency1/context16K/.85memory,
  actual GPU/runtime provenance, continuous memory sampling, sticky worker failures.
- `playback.py`: clock/cut/layout/gap segment breaks, pauses and replay flags.
  Missing clock continuity is conservative. This is not a validated edit detector.
- Schema2 extensions, fusion, questions and review preserve slots, playback,
  provenance and corrections. Audio/experimental rows are explicitly excluded from
  visual references; cumulative questions cannot cross playback segments.
- `experiments.py` / `experiment_worker.py`: separate crops/events/audio/all;
  twenty8second2FPS clips, Qwen vs InternVL BF16, five prompted SAM clips and
  five-minute Silero→ASR→forced-aligner pipeline, sequential GPU model lifecycles.
  Per-component failures preserve completed results; artifacts and dependent caches
  are isolated from production observations. Native clip PNGs are retained.
- Event reports: precision/recall, actor attribution, onset error, abstentions,
  latency/memory by split. Tracking: native screen-mask IoU, never semantic events.
  Audio: source-aligned spans/word times, WER/name recall/alignment error, memory.
  Visual-only and audio-assisted event proposals stay separate. No experiment ran
  on an actual GPU in this turn; adapters still require integration validation.
- Setup/Slurm/default CLI switch together to Midbeast. Driver, vLLM, optional audio
  and tracking workers are isolated; installed Linux closures are saved. No normal
  Paddle installation or icon downloads. Solver opt-in only. Login-node setup
  does previews/cache downloads; allocation preflight actually loads the VL worker.

Pins are in `models.local.json`: intake017b9c7..., InternVL226b96d...,
ASRbcd2b5b..., alignerc07281d..., SAMdaa6319...; SAM codef1647e8....
The selected vLLM0.32.0 requires CUDA13 dependencies; actual Longleaf driver/kernel
compatibility is UNTESTED. SAM checkpoint is gated; do not accept license terms
on the user's behalf. The l40-gpu partition may supply L40 rather than L40S.
Record the actual hardware and never relabel L40 measurements.

## Validation and next execution

Local test interpreter: `/private/tmp/league-midbeast-test/bin/python` (proper
Python3.12 venv; use `-B`). Do not use the old league-intake-test interpreter:
its venv metadata was missing and it resolved to the system Python.

All **157 repository tests passed with no skips**, including real PyAV decode/cache tests, restricted VL
binding/truncation/conflict/retry/OOM, independent numeric/VL invalidation,
sequential experiment isolation, setup, legacy review, corrections and resume.
Python AST, shell and review-JavaScript syntax plus Git whitespace checks passed.
Detailed current validation: `annotation/league/VALIDATION.md`.

Next work after separately restoring authorized SSH access:

1. Preserve/collect old CPU pilot6151847 results without overwriting its checkout.
2. Transfer the exact pinned MP4 and updated code to a separate Midbeast checkout.
   Preserve old runs/user edits. Do not reuse a previous extraction directory.
3. Install workers/cache pinned checkpoints online using setup, then perform GPU
   preflight inside an allocation, offline. Verify actual L40S name/UUID/capacity,
   CUDA driver support, FP8 kernel loading and local-only serving. Missing assets,
   model/API mismatch or OOM must remain failures, not fallback model selection.
4. Independently complete/adjust forty-frame labels and difficult-case coverage,
   preserving held-out separation from all inspected calibration/development data.
   Resolve the unproven pause/rewind/segment cases honestly; current templates
   cannot certify the protocol. Freeze development rules only, re-extract and
   pass held-out numeric/text/bar/presence gates. Deferred icons do not block.
5. Run a fresh five-minute cold pilot (`annotate --pilot`). Require300sampled
   seconds, an actual L40S, completeness, no OOM, <90%GPUcapacity and ≤4hour projection.
   Do not extrapolate cache reuse as cold processing. No such measurements exist.
6. Verify20event clips/5tracking prompts+masks/5minute transcripts and word labels.
   Run `experiment --suite all` separately; record experimental results and resource
   measurements. Commentary cannot establish unseen values or identity.
7. Only after intake gates pass, process the complete source, then validate review,
   corrections, fusion, eligible visual questions, optional solver and reports.

Commands and paths: `annotation/league/README.md`. Current test log:
`/private/tmp/league-midbeast-tests-final.out`.

---

# Historical OCR-only handoff (superseded production scope; retain evidence)

# League intake implementation handoff

Status: replacement intake is implemented in the local working tree; release
acceptance and full-video rollout are **not complete**. Preserve local changes.
No commits or PRs have been created. This file is updated as validation proceeds.

## Stop state — 2026-10-23

The authenticated SSH master has expired. The local control check returned
`Control socket connect(.../control): No such file or directory` (exit 255).
No new login was attempted. Per the user's instruction, remote work stopped and
this handoff was updated before ending the turn. Do not describe job 6151847 as
currently running or completed: its final state has not been collected.

Last observed pilot progress: 6151847 RUNNING, elapsed 23:45, last committed source
time 801 seconds (third one-minute window). A resource snapshot reported MaxRSS
2,918,260K. Final pilot report/projection remain unknown.

Latest user request asks for parsing status and whether to retain OCR, use a
larger Qwen VL model such as 72B, and add ASR. The response recommends a hybrid,
but no Qwen intake reader or ASR implementation has been added or authorized as a
new production default. Existing approved production intake still excludes Qwen.

Discussion notes for a future continuation:

- Keep field-specific OCR for clock, gold, K/D/A, CS, level and visible timers;
  use direct image measurements for bars and visual identity matching for icons.
- HP remains low coverage (8/20 readable development fields); all four remaining
  accepted development numeric disagreements are whitespace around the slash.
  Preserve raw OCR while defining parsed-value/exact-text scoring separately.
- Portrait acceptance is zero at current thresholds; this needs further reader
  work and representative labels, not just a lower confidence cutoff. Missing
  Korean/objective/item/spell/passive/absence cases still prevent release.
- Compare a larger **vision-language** Qwen model on rejected native crops and
  short event clips, with bounded field-specific outputs and the same labels;
  do not reinstate broad generative observation prompts or accept self-confidence
  as verification. No such head-to-head benchmark has run.
- ASR is unimplemented. Local ffprobe confirms an AAC audio track, 44.1 kHz stereo,
  language unspecified; speech/commentary presence has not been established.
  First audit representative audio, then test speech detection/transcription and
  timestamps. Announcer speech or commentary can propose event windows, but must
  retain audio provenance and cannot supply unseen HUD digits or actor identities.
  Non-speech effects require audio-event recognition rather than ASR.
- Keep visual-only and audio-assisted evaluation separate. The earlier approved
  pipeline used no audio; adding audio is a scope extension, not existing behavior.

## User instructions

Implement the approved replacement plan behind the existing League commands and
Longleaf setup/Slurm scripts. Do not retain a selectable broad Qwen intake engine.
Do not process the full video unless the quality and five-minute runtime gates pass.
If Longleaf needs another authenticated login, **write/update this handoff and end
the chat**. Do not ask the user to log in again during this continuation.

## Local workspace

- Repository: `/Users/alex/Desktop/Projects/videogame`.
- Test environment: `/private/tmp/league-intake-test/bin/python` (use `-B`).
- Original user edits concerned generation flags in `inference.py` and logging in
  `observers.py`; the former remains, and the replacement has per-frame reader logs.
- Local source video: `data/youtube/lol_domisum_aatrox/xICbe-OOPvE.mp4`.
- Source SHA256: `9352784b7d6e9faafc1b3ab57aad0f44849daeb0d64df3096a6cdb73b8994825`.
- Source: 1920×1080, 60 FPS, 1735.62195 seconds.
- Version-1 runs are preserved. The local version-2 run is
  `data/league_runs/aatrox-26.13-v2`; preparation and field previews completed.
- Ignored diagnosis snapshot: `data/league_diagnosis/2026-10-21-longleaf`.
- Ignored development images/results: `data/league_intake_dev`.
- Native benchmark images: `data/league_intake_dev/benchmark-frames`.
- Cached visual assets copied from Longleaf: `data/league_assets`.

## Implemented

`intake.py`, `readers.py`, `ocr_worker.py`, `intake_profile.py`, `visual_assets.py`,
`normalization.py` and `intake_benchmark.py` implement calibrated crop readers,
isolated CPU Paddle English/Korean recognition, Tesseract numerics, bounded
fallback, explicit acceptance, patch image matching, stable slots, native evidence,
PTS decoding, resumable tasks, independent caches and strict release checks.

The existing CLI, setup and Slurm defaults use schema 2 and a new run directory.
`annotate` does not initialize Qwen or retrieve mechanics. Solver initialization is
lazy and evaluation-only. Fusion keeps unknown slots separate; source corrections
rebuild derived K/D/A, inventory and identity bindings. Mechanics are optional and
downstream. Minimap, scene events and notices are explicitly unsupported.

The review UI includes native/processed inputs, icon masks and profile calibration.
Legacy runs remain readable and cannot resume extraction with the new engine.
Read `annotation/league/README.md` for commands, contracts and gates.

The 40-frame manifest `annotation/league/benchmarks/aatrox-26.13.json` contains
**draft** visual transcriptions, not human-verified release labels. Every frame is
still `verified: false`. Required reader groups and cases are incomplete. Do not
mark labels verified, freeze thresholds, or claim benchmark acceptance merely to
unblock full-video processing. `--draft` scores cannot authorize a release.

## Longleaf connection and deployment

An authenticated multiplexed connection was restored by the user, but is now
expired as recorded above. Former socket:
`/private/tmp/league-longleaf-01a16144/control`; host alias `unc-cluster`.
Before relying on it, `ssh -S SOCKET -O check unc-cluster` is a local-only check.
Use `-o BatchMode=yes -o ConnectTimeout=10` for remote commands. Never collect
credentials in tools/messages. If authentication is needed again, follow the user
instruction above and end the turn with this handoff.

Isolated validation checkout (original remote repository/run untouched):
`/work/users/a/l/alshen/league-intake-validation/source`.

Validation run: `/work/users/a/l/alshen/league-intake-validation/run`.
Visual/model cache: `/work/users/a/l/alshen/league-intake-validation/assets`.
OCR interpreter: `/work/users/a/l/alshen/conda_envs/videogame-league-ocr/bin/python`.
Driver interpreter: `/work/users/a/l/alshen/conda_envs/videogame-league/bin/python`.
Original remote video:
`/users/a/l/alshen/vgextract/vgextract/data/youtube/lol_domisum_aatrox/xICbe-OOPvE.mp4`.

The OCR environment was installed and its 72-package Linux closure is checked in.
Both official Paddle models and 1372 patch image references were downloaded and
verified. Paddle initially failed because home was over quota; its existing absent
home cache was linked without deleting data:
`~/.cache/paddle -> /work/users/a/l/alshen/.league-paddle`.
PaddleX/runtime caches are redirected to work storage. The CPU worker successfully
recognized a real clock crop. The actual cluster Tesseract is 4.1.1 and omits its
data directory from `--list-langs`; the compatibility fix now finds and explicitly
uses `/usr/share/tesseract/tessdata/eng.traineddata`, SHA256
`7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2`.

## Jobs and results so far

- 6146904: initial benchmark failed because its frame manifest was not uploaded.
- 6147166: queued GPU job canceled in favor of CPU OCR validation.
- 6147397: slow baseline, stopped gracefully; committed evidence retained.
- 6148879: corrected concurrent readers completed all 40 frames. Coverage failed.
  About 8–10 seconds per sampled frame, substantially faster than the minute/frame
  baseline. Results copied locally to `data/league_intake_dev/longleaf-r2`.
- 6150705: preflight failed on Tesseract 4's directory-less language listing;
  compatibility fix implemented and directly verified on Longleaf.
- 6150993: completed all 40 frames. Cold reader work: 290.47 seconds; stage:
  322.88 seconds. Operational and draft quality gates failed. Results copied to
  `data/league_intake_dev/longleaf-final`. Development numeric agreement was
  91.7% at 93.9% coverage; one English name and health bars agreed with draft
  labels, selected portrait coverage was zero at the unfrozen .92 threshold.
- 6151792: completed the cached 40-frame refresh with an independent HP/bar
  consistency check. Results are local in `data/league_intake_dev/longleaf-6151792`.
  After independent native-pixel correction of three development draft labels,
  numeric agreement is 158/162 (97.53%) at 90% coverage; diagnostic held-out is
  161/166 (96.99%) at 92.74% coverage. Selected portrait coverage is zero. Presence
  coverage is 84.17%/86.19%. English-name and bar subsets agree 20/20 each per split,
  but do not certify complete families. Draft labels and missing cases independently
  prevent acceptance. Original OCR text and candidate numbers remain available
  when the HP/bar check rejects a reading.
  All four remaining accepted development numeric disagreements are spacing around
  `/`, not digit differences; exact-text scoring remains strict. HP readable-field
  coverage is only 8/20. The gross digit errors were rejected. A local audit is in
  `longleaf-6151792/development-accepted-disagreements.json` (bar float differences
  in that file are within benchmark tolerance and are not bar failures).
- **6151847**: fresh five-minute runtime pilot last observed running in
  `/work/users/a/l/alshen/league-intake-validation/pilot-run`, eight CPUs/64 GB,
  `general`, no GPU. Logs: `league_v2_runtime_pilot-6151847.{out,err}` in the
  validation checkout. Final state uncollected because SSH expired. It cannot
  authorize release while quality is failing.

The deployment includes the reader consistency check, runtime/profile/asset
provenance snapshots, normalization and review fixes. Subsequent local changes:
evaluation preserves StopRequested; benchmark freezing calibrates both icon score
and margin from development labels and rejects ties. Sync those small changes after
the active pilot completes. They do not change its extraction implementation, but
the updated benchmark implementation changes the release-policy hash; rescore draft
results after syncing. Do not overwrite active extraction source files.
No full recording has been processed by the replacement engine.

All **134 repository tests passed**, with no skips. Python/shell/review-JavaScript
syntax, new-module formatting and Git whitespace checks passed. A development-only
HP preprocessing experiment did not establish sufficient reliability; its variants
were not shipped. `annotation/league/VALIDATION.md` now records actual remote results
and replaces the obsolete version-1 validation status.

## Next work

1. Collect job 6151847 output and `intake_report.json`; expect a coverage failure,
   not release acceptance. Preserve results.
2. Record the actual pilot hardware/allocation, stage elapsed time and full-video
   projection. Hardware: Intel Xeon Gold 6140 @2.30 GHz, node c0310, eight allocated
   CPUs/64 GB, general partition; this is not an L40-node runtime validation.
3. After the job, sync the local evaluation/benchmark updates and rerun
   `intake-benchmark --labels annotation/league/benchmarks/aatrox-26.13.json --draft`
   with the isolated asset cache and benchmark run. It must exit 2. Refresh the local
   result copy because the benchmark policy hash changes. Do not freeze draft labels.
4. Update `annotation/league/VALIDATION.md` and this handoff with final pilot results.
5. Do not claim production readiness: complete verified labels, missing cases,
   development-only threshold freezing, held-out gates and runtime gate first.
   Do not run the full video while any family fails.

Full test command:

```bash
/private/tmp/league-intake-test/bin/python -B -m unittest discover -s tests -q
```
