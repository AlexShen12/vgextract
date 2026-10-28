# Midbeast hybrid intake validation — 2026-10-27

**Not released.** Local implementation and contract tests pass; GPU readers,
independent benchmark acceptance and full-video rollout remain unvalidated.
The historical Domisum OCR scores do not validate this new engine or source.

## Remote integration — 2026-10-28 (in progress)

The authenticated connection is restored. Hardware probe7064617 completed on
`g181005`: NVIDIA L40S,46,068MiB, driver595.58.03, CUDA13.2 in nvidia-smi.
Runtime probe7067005 on g181007 passed BF16 and FP8 GEMM and vLLM import:
PyTorch2.13.0+cu130, Transformers5.18.0, vLLM0.32.0. Model inference still pending.

Separate remote root `/work/users/a/l/alshen/league-midbeast-validation`, source
`source-20261028`, isolated env-driver/env-vl/env-audio and pinned assets cache.
The exact transferred source hash matched. Native prepare/layout previews passed.
The pinned Qwen checkpoint is30.89GB across79 verified files; ASR/aligner cached.
The original League Conda environment was inspected and left unchanged: Python3.12.15,
PyTorch2.8.0, Transformers4.57.1, all tested imports and pipcheck passed; it lacks
vLLM and qwen3_5 architecture support. Registered environments also include base,r,
avp and learnerpr; the OCR and new validation workers are separate venvs.

First Qwen smoke7067349 failed before model loading: Torch bound system libstdc++
which lacks CXXABI_1.3.15 required by the base Conda Python ICU/SQLite. Explicitly
selecting Conda's native libraries passed Torch→SQLite→vLLM API-server imports.
The worker launcher and deployment scripts now select that runtime without changing
old environment packages. Corrected three-frame smoke7068523 and audio7068769
then failed on home `.triton` quota before completed inference. All installer,
compiler, HF, temporary and Slurm write defaults now use `/work/users/a/l/alshen`,
per the user's instruction. HOME stays unchanged. Conda registration is disabled.
Retry7070562 passed the quota failure but hit Linux's107-byte Unix socket limit.
TMPDIR and VLLM_RPC_BASE_PATH now use short `/work/users/a/l/alshen/tmp`.
Qwen7071029 completed3frames in530.93stage seconds, peak39,289,094,144bytes.
All3 HP parses matched independently inspected development crops. Names failed:
visible text was reported absent. The21/21 accepted-or-absent log line is not
a semantic quality pass. Diagnostic7072632 compares neutral/semantic IDs and
single/grouped crops. Profile revision2 and local reader code now retry not_shown
for text required in a present panel, retaining unknown rather than false absence.
The comparison completed: semantic and neutral single-crop requests both read
Singed; grouped outputs varied, including false absence with neutral IDs.
Names now use individual requests. Source-r6 deploys this behavior and profile
revision2. Normal annotate smoke7073515 completed on an L40S (64GB host memory):
eight name crops returned text, two remained unknown/unreadable; HP matched.
Stage228.42seconds includes cold model startup and warm frame decode; peak
39,274,414,080GPU bytes. This is not a fresh five-minute intake timing pilot.
No name-quality acceptance is claimed; no validation jobs remain active.
Audio7070609 transcribed72 speech chunks without truncation but failed on an
out-of-span alignment word. Source-r5 rejects invalid word timings, preserves raw
candidates and transcriptions, and records partial alignment. Retry7071860 completed:
72speech chunks,624in-span word timings,2rejected timings,78.14worker seconds,
4,753,195,008peak GPU bytes; resource check passed. Independent scoring is pending.

Other fixes from actual deployment: retain source PTS separately from requested
sample_time on59.94FPS video; benchmark lookup uses sample_time. Normalize Torch's
bare GPU UUID for nvidia-smi. Explicitly request gpu:nvidia_l40s:1. Disable worker
telemetry. Local164 repository tests pass; Linux126 League tests pass with no skips
when FFmpeg is on PATH. Earlier fixture failures were inherited Lmod BASH_ENV and
an asset-cache override; tests now isolate those settings. Review APIs also passed.
The work-filesystem run exposed NFS anonymous-file cleanup in the installation
writability probe; it now closes a named temporary file before unlinking it.

Old CPU pilot6151847 was collected: failed130:0 after42:49,280committed frames,
SIGTERM recorded, operational/quality false. It did not complete the300second pilot.
Raw evidence is local under `data/league_midbeast_dev/longleaf-20261028`.

## Measured locally

- Selected newest eligible of 20 main-channel catalog entries and downloaded
  Midbeast `OLLZBe5-fg4`, 1920×1080 at 60 FPS, approximately 1733 seconds.
  The pin and catalog are in `midbeast-source.json`; source SHA256 is
  `f4f36e1a274617ecbe224eaf7fc3f8375f757acb7a517bcc0ee7780fcd83f240`.
- Inspected twelve dispersed native frames, the contact sheet and enlarged HUD.
  Both static HUD anchors matched all twelve inspected frames. Calibrated the
  21 enabled fields and intermittent scoreboard/objective-panel anchors.
- Ran real `prepare` and `processor-preview` locally without loading a GPU model,
  Paddle, icon assets or mechanics. All three preview layouts matched. Native
  crops and fixed preprocessing evidence are saved under the new run.
- Ran the actual Tesseract/bar path on twelve native development frames. This
  exposed crop borders and low-confidence rejection. Tightened K/D/A, gold,
  level and timer boundaries and chose one fixed bright-glyph recipe. Repeated
  the smoke check at source seconds 30 and 750: clock, blue kills, K/D/A, CS,
  gold and level were accepted; red kills at second 30 remained unknown and
  the absent timer was not accepted. At second 750 accepted HP-bar proportion
  was .9203; separately inspected displayed HP was 1877/2043 (~.9187).
  These are development smoke checks, not precision/coverage measurements.
- Created neutral sampling/label initialization for forty frames, twenty
  eight-second event clips, five sixty-second audio intervals and five tracking
  clips. Labels remain unverified, without model-derived truth or fabricated
  manual prompts. Required case coverage and split-by-playback-segment still
  need independent inspection and sample adjustment. Twelve inspected clocks
  followed source time +33 seconds; actual pauses/rewinds and separate playback
  segments have not been demonstrated.
- Immutable Hugging Face checkpoint revisions were resolved for Qwen3.8-27B-FP8,
  InternVL3.5-14B-HF, Qwen3-ASR-1.7B, its forced aligner and gated SAM3.1.
  Full model weights were not downloaded on this Mac.

## Implementation verification

The isolated test interpreter is
`/private/tmp/league-midbeast-test/bin/python` (Python 3.12).
All **160 repository tests passed**, with no skips, including strict field binding,
truncation/example/empty-output rejection, one group-to-single retry, OOM/worker
failure, no Paddle fallback, unknown-slot preservation, playback discontinuities,
audio/experimental evidence isolation, checkpoint SHA256 corruption, legacy review,
corrections, cached decoding, resume, setup isolation and optional solver behavior.
Test results are implementation evidence, not model accuracy measurements.

```bash
/private/tmp/league-midbeast-test/bin/python -B -m unittest discover -s tests
```

## Pending before release

1. Independently verify and adjust all forty Midbeast labels, 20 development and
   20 held-out from distinct playback segments. Include required difficult cases.
   Freeze only development rules; keep exact-text and parsed-number metrics separate.
2. Install isolated Linux workers, cache pinned model/processor files and resolved
   dependency closures, then run offline allocation preflight. The chosen vLLM
   build depends on CUDA 13 packages: actual Longleaf driver compatibility is
   unknown. Gated SAM access must already be granted.
3. Verify actual GPU name/UUID/capacity. The existing `l40-gpu` partition may supply
   L40, not L40S; never present L40 measurements as L40S. No Qwen, InternVL, SAM,
   ASR or aligner GPU execution has been performed for this change.
4. Pass enabled numeric/text/bar/presence held-out gates and a fresh five-minute
   cold intake pilot: all samples processed, no OOM, measured memory below 90%
   of actual capacity and projected full-video intake at most four hours.
5. Run event comparisons, tracking and ASR separately against independent labels.
   English commentary has not yet been verified through transcription/listening.
   Experimental predictions cannot authorize production events or actor identities.
6. Only after intake gates pass, process the complete recording and verify review,
   correction propagation, fusion, supported questions and stage reports.

Longleaf remains disconnected. No SSH login was attempted in this implementation
turn. Follow the standing instruction to save a handoff and stop when remote work
requires renewed authentication.

## Historical Domisum validation

The earlier OCR implementation, unverified draft scores, preserved runs and
uncollected CPU pilot job 6151847 are recorded in
`docs/league-intake-implementation-handoff.md`. That pilot's final status remains
unknown. Existing evidence was preserved; none of its labels or thresholds were
used to certify Midbeast.
