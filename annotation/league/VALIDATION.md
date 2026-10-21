# Validation record

Local validation completed 2026-10-19 (America/New_York), Python 3.12.7 on macOS arm64.
This records implementation checks, not a completed GPU annotation benchmark.

## Passed

- **87 repository tests**, including all existing ingestion tests, static knowledge
  tests, and the new pipeline/API tests. Command:
  `python -m unittest discover -s tests -q` in the temporary test environment with
  review dependencies installed. No skipped tests in that run.
- End-to-end **test-fixture** annotation → fusion → question generation → solver
  evaluation → reporting, including interruption/resume at a HUD microbatch,
  automatic review-budget exhaustion, all six question capabilities, Korean player
  text, mirrored sidebar bindings, unknown/occluded states, overlapping notices,
  unsupported composite labels, optional human verification, append-only corrections,
  frozen review revisions, stale-report refusal and solver-input leakage checks.
- Corrected reference answers/statuses reuse original solver predictions when their
  video/question inputs are unchanged. Question-count and solver-budget changes
  retain annotation caches. These are exercised with an injected fixture runner;
  the production CLI has no fixture inference backend.
- Real FFmpeg decoding tests establish source PTS rather than requested seek times.
  Incorrect source/layout resolution is rejected. Python compilation, shell syntax
  checks and review JavaScript syntax checks passed.
- The real Aatrox video was hashed and probed: **1920×1080, 60 FPS, 1735.62195 seconds**.
  SHA256: `9352784b7d6e9faafc1b3ab57aad0f44849daeb0d64df3096a6cdb73b8994825`.
- **33 native crop previews:** eleven routes at source seconds 30, 780 and 1620.
- **36 real Qwen processor audits**, using the pinned image/video processors:
  the 33 previews plus selected-HUD, gameplay and full-frame temporal inputs. The
  final preprocessing key matches the checked implementation/configuration.
  No model weights were downloaded or loaded during these checks.
- Actual sequence preprocessing (without the full static-roster prompt):

  | Route | Source frames | Input tokens | Visual tokens | Resized frame |
  |---|---:|---:|---:|---|
  | Selected HUD | 2 | 1,342 | 270 | 480×288 |
  | Gameplay | 8 | 2,963 | 1,536 | 1024×384 |
  | Full-frame context | 8 | 2,130 | 720 | 576×320 |

  Video metadata uses the source clock: for example indices 46,800–47,220 at
  60 FPS represent source seconds 780–787. These are annotation preview settings;
  the QA solver has its own bounded, adaptive pixel budget.
- Real static references for **all 173 patch-roster champions** and the item
  dictionary/fallback were cached. A timed-out Nunu request succeeded on retry.
  Raw spell calculation graphs and unsupported placeholders remain explicit
  knowledge gaps; complete caching does not imply complete formula resolution.
- The local review UI was tested through its HTTP API and in the in-app browser.
  Actual video playback/seeking at 780 seconds, native/processed crops, read-only
  layout previews, paging, editable records and review-state behavior were checked.
  A screenshot is saved with the local run artifacts as `review-preview.jpg`.

## Pinned model revisions

- Instruct: `Qwen/Qwen3-VL-8B-Instruct` at
  `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`.
- Thinking: `Qwen/Qwen3-VL-8B-Thinking` at
  `92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b`.

The exact CPU preprocessing package closure is in
`inference-environment.lock.txt`; the tested review environment is in
`review-environment.lock.txt`. GPU preflight records actual CUDA packages,
BF16 kernel availability, device and model-cache completeness for each allocation.

## GPU acceptance still pending

Longleaf login, transfers and GPU job execution are user-managed. No further
remote access attempts or job submissions are part of this implementation work.

The local CUDA preflight correctly exits before weight loading because the Mac
does not expose a CUDA GPU. A read-only SSH probe of the configured Longleaf host
(`unc-cluster`) was attempted with batch authentication, strict host-key checking
and a connection timeout. It failed with **port 22 connection timed out**, before
any remote command or job submission. No cluster files or jobs were changed.

Consequently, none of the following has been established:

- Real Qwen semantic annotation/generation quality on the Aatrox video.
- An unattended full-game GPU run or a human-verified question benchmark.
- L40 peak memory, throughput, generation truncation rate or processing cost.
- Accuracy thresholds for the pilot, temporal labels or champion/ability identity.

From an accessible Longleaf session, follow the README to create the environment,
prepare the source/run, cache weights/references and submit
`bash annotation/league/submit_longleaf.sh`. The automatic pilot precedes the full
pass and writes `pilot_report.json`; completed work survives allocation expiry.
Human review remains optional during all of those stages.
