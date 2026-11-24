# Midbeast Pass 1 bundle

`annotation.league.pass1` assembles the HUD, scoreboard, roster, inventory,
ability-slot, cast/channel, killfeed and ASR commentary producers in one run. SAM windows,
scene-action inference, captions and descriptions are outside this command.
The default output is `data/league_runs/midbeast-pass1-final-v1/`.

The run is **experimental**. Operational completion means all requested samples
are accounted for, including explicit unreadable or unsupported observations.
It does not mean that the reader outputs are verified. Verification will use
agent annotation in a separate future pass; manual review is not required for
this deliverable. Ordinary
production intake still requires its existing frozen quality and runtime gates.

## Tournament broadcasts

Use `python -m annotation.league.pass1 broadcast` for the profile-driven
broadcast path. This produces a searchable full-VOD transcript and scopes
screening, scoreboard OCR, and commentary evidence to the supplied excerpts.
The broadcast review also opens at `/pass1`.
[Worlds Game 1 commands and results](research/league-worlds-broadcast-pilot-2026-11-22.md#game-1-broadcast-pass-1)

Run `python -m annotation.league.pass1 broadcast-match` on a verified
broadcast bundle to add reviewed draft and roster evidence, tracked killfeed
and objective events, and timestamped player scoreboard reads. The match
tables and evidence crops appear in the same `/pass1` review.
[Worlds match readers and benchmark](research/league-worlds-broadcast-pilot-2026-11-22.md#worlds-draft-roster-and-gameplay-events)

## Prepare and reuse

From the repository root, using the League Python environment:

```sh
python -m annotation.league.pass1 prepare \
  --import-run data/league_runs/midbeast-pass1-full \
  --scoreboard-skim data/league_recognition_v1/scoreboard-skim-20261111-v1
```

`prepare` also accepts `--run-dir`, `--video`, `--asset-cache`, additional
`--import-run` arguments, and `--import-producer NAME=PATH`. A different recording
is rejected because this profile is calibrated for the pinned 1080p Midbeast
source. Imported raw artifacts retain their IDs, provenance and original files.
Generated adapters resolve evidence through bundle-relative paths.

Inventory extraction is skipped only when an imported full-interval producer
has the expected sample count, matching inventory geometry and reader settings.
Partial imports do not suppress missing work. Names and row positions alone do
not establish champion identity.

## Five-minute pilot and budget checkpoint

The five windows are `[0,60)`, `[418,478)`, `[836,896)`, `[1254,1314)` and
`[1672,1732)` in source-video seconds. CPU slow fields run at 1 FPS; ability
slots and cast bars at 16 FPS; scoreboard presence at 4 FPS. Refinement retries
unreadable clock crops at 4 FPS and rebuilds shared playback scopes without
repeating the slot or cast recognizers.

```sh
python -m annotation.league.pass1 pilot --component cpu
python -m annotation.league.pass1 pilot --component dense
python -m annotation.league.pass1 pilot --component vl \
  --model-cache /work/users/a/l/alshen/league-midbeast-validation/assets
```

The VL command runs in the GPU allocation, with `LEAGUE_VL_PYTHON` pointing to
the existing isolated vLLM environment. `--model-cache` is separate from bundled
image-reference assets; model weights are not duplicated into the review bundle.
`--component context` repeats only clock refinement after a clock correction.

Use the dedicated Longleaf files from the staged, verified checkout:

```sh
sbatch annotation/league/slurm_pass1_cpu.sl pilot
sbatch annotation/league/slurm_pass1_vl.sl pilot
```

CPU and GPU jobs write separate producer directories. The VL pilot checkpoints
each source sample. Experimental resumption can reuse a completed sample from a
different GPU only when the source scope, profile, reader implementation, model
files, CUDA/software contract and inference settings agree. GPU serial numbers
and small differences in available VRAM do not invalidate identical inputs.
`experimental_reuse.json` links every reused task to its original recipe and
runtime. Recognition results are never restamped to claim they ran again.

After collecting a completed remote producer, import it explicitly:

```sh
python -m annotation.league.pass1 prepare \
  --import-producer vl=/path/to/collected/run/producers/vl
python -m annotation.league.pass1 finalize --draft
python -m annotation.league.pass1 verify
```

Inspect `pass1_report.json` for measured times and projections with 30% headroom.
Allocated time for failed and resumed jobs is retained separately from useful
reader work. Pilot footage is development evidence, not held-out evaluation.
**Stop here for the full-video budget decision.**

After that decision, `run` requires `--budget-approved`. Use the measured report
to estimate full-run compute and give Longleaf jobs substantial wall-time headroom
above that estimate:

```sh
python -m annotation.league.pass1 run --component cpu --budget-approved
python -m annotation.league.pass1 run --component dense --budget-approved
python -m annotation.league.pass1 run --component vl --budget-approved --model-cache MODEL_CACHE
python -m annotation.league.pass1 finalize
python -m annotation.league.pass1 verify --require-complete
```

Missing full-run roster and killfeed producers are generated with the existing
CPU implementations. A completed compatible import is reused. No full-recording
roster, killfeed or HUD scan is launched by the pilot command.
Run the CPU and VL components in their separate allocations to avoid reserving
a GPU while CPU-only stages execute. The assembly job can depend on both jobs.
The standalone Longleaf context rebuild and assembly jobs default to a 12-hour
wall-time limit. Keep those defaults for full recordings. The Midbeast assembly
retry took 54 minutes, including 17 minutes for the final integrity check.

## Evidence and derived outputs

`pass1_manifest.json` records producer revisions, checksums, inputs, sampling
windows, implementations and job receipts. Each producer has its own lock.
`source/` holds the pinned video, `assets/` holds image references, `profiles/`
holds calibrated layouts and model locks, and `code/` contains `annotation` and
`ingestion` with `code_checksums.json`. Immutable media may be hardlinked on the
same filesystem. The bundle does not depend on external symlinks.

The root retains compatible `observations.jsonl`, `states.jsonl`, `events.jsonl`,
`inventory_snapshots.jsonl` and `inventory_timeline.jsonl`. Additional outputs:

| Output | Meaning |
| --- | --- |
| `playback_timeline`, `selection_timeline` | Shared source-time continuity and selected-owner scopes |
| `roster_timeline`, `roster_evidence` | Team membership and its portrait witnesses |
| `inventory_identity_links` | Exact-name/portrait evidence for delayed identity attribution |
| `inventory_history` | Per-champion, per-slot observations and bracketed changes |
| `scoreboard_visibility`, `scoreboard_openings`, `scoreboard_opening_groups` | Native samples, boundaries and conservative reopen merges |
| `ability_icon_timeline`, `ability_slot_timeline`, `hud_activations` | Icon evidence, displayed countdowns and HUD-supported transitions |
| `cast_bar_readings`, `cast_timeline` | Raw named bars and reconstructed channel occurrences |
| `evidence_frames`, `issues` | Source crop references and unresolved observations |

All table names above have a `.jsonl` extension. Identity backfill requires two
concordant accepted name/portrait witnesses in one verified segment, rejects
conflicts, and records when the identity evidence became available. It is an
experimental evidence binding, with verification deferred to agent annotation. A row number or a
complete team pool cannot fill a hidden portrait by itself.

Reopen merging requires a gap of at most two seconds, verified continuity,
fully resolved inventories for all ten champions, and matching items, known
empty slots and trinkets. Changed or unreadable slots prevent merging. Original
openings, samples and closures remain available. The gap is a development
policy; source-review status is reported separately.

A cooldown transition is a `hud_slot_activation`; a readable bar produces an
`observed_channel`; the killfeed produces a `kill_announcement`. None of these
alone establishes a physical cast, hit, target, exact death frame or channel
completion. Uncertain channel endings stay unknown.

## Review, corrections and portable verification

```sh
python -m annotation.league --run-dir data/league_runs/midbeast-pass1-final-v1 review
```

Open `/pass1` for the unified table browser, source playback and native crops.
HUD corrections use the existing review page; inventory retains its existing
viewer. Channel corrections are available on the `cast_bar_readings` table.
Corrections are append-only and invalidate dependent fusion outputs. Rebuild
with `finalize --draft` while the full run is incomplete, or `finalize` once all
required producers cover the recording. Neither command repeats inference.

Finalization verifies a fixed producer/correction snapshot before publishing a
revision. Tables under `revisions/` are immutable review snapshots; the overview
reads the manifest's active revision. Checksums seal generated tables, cited
evidence, reference assets and code. `verify` detects stale corrections,
modified evidence, missing files and external symlinks.

After moving the complete directory, the bundled snapshot can serve review:

```sh
PYTHONPATH=/absolute/path/to/bundle/code python -m annotation.league \
  --run-dir /absolute/path/to/bundle review
```

The report lists incomplete producers and missing field outputs explicitly.
A final revision is refused while required full-recording producers are missing.
Agent annotation is the planned verification step and has not started. The
ordinary production release gates remain separate from output assembly; no
manual review gate is imposed by this experimental workflow.

## Commentary transcripts in Pass 1

The pipeline prefers YouTube automatic captions for commentary in
Pass 1. Retain the original-language timed JSON3 track with yt-dlp. For the
current Midbeast video, the downloaded track is
`data/league_asr_handoff/20261119-v1/autocaptions/OLLZBe5-fg4.en-orig.json3`.

```sh
.venv/bin/yt-dlp --skip-download --write-auto-subs --sub-langs en-orig \
  --sub-format json3 -o 'CAPTION_DIRECTORY/%(id)s.%(ext)s' \
  'https://www.youtube.com/watch?v=OLLZBe5-fg4'
```

The caption adapter imports this track into `commentary_timeline.jsonl`
using source-video seconds, available word offsets or segment timing, and
caption provenance. Preserve the raw downloaded track, language, video ID,
caption checksum and source-video checksum. Merge rolling display events into
the transcript and retain the original event references. Uncertain League names
remain reviewable in their original wording.

Import the saved track and rebuild the timeline with the current CLI.

```sh
python -m annotation.league.pass1 import-captions \
  --captions data/league_asr_handoff/20261119-v1/autocaptions/OLLZBe5-fg4.en-orig.json3 \
  --video-id OLLZBe5-fg4 --language en-orig
python -m annotation.league.pass1 finalize --draft
python -m annotation.league.pass1 verify
```

A registered YouTube track is selected ahead of the ASR producer for the
commentary timeline. The import stores the original track and normalized cues
under `producers/youtube_captions/`. Caption rows retain original event indices,
language, video ID and checksums. Word ends are unknown. Cue ends use bounded caption display intervals, refined
by saved VAD speech ends when a compatible full-recording ASR coverage file is
available. This refinement runs on the CPU and reuses the coverage file. It
merges adjacent chunks and pauses up to 150 ms, then trims each cue using the
speech region containing its last word onset. It keeps the existing boundary
when coverage is ambiguous or only segment timing is available. Each row records
`end_timing` with its estimated status, method and evidence. The producer stores
a checksummed copy of the speech coverage used for the estimate. ASR supplies the fallback when a caption track has
not been imported.

### ASR fallback and comparison workflow

Use the isolated audio environment and pinned model cache on Longleaf. Submit
the ten-second pilot from the staged checkout.

```sh
sbatch annotation/league/slurm_pass1_asr_pilot.sl
```

The equivalent CLI is `python -m annotation.league.pass1 pilot --component asr
--model-cache MODEL_CACHE`, with `LEAGUE_AUDIO_PYTHON` set. The pilot interval is
source seconds 30–40. Each recipe has its own directory under
`producers/asr/pilot/`, containing the clip, raw result, word alignment,
commentary, coverage and report. Compatible completed speech chunks are reused
after interruption. Model weights stay in the existing model cache.

Prepare an independent source-listening reference before comparing the prediction.
The reference JSON contains `source_sha256`, `interval` equal to `[30,40]`,
`method` equal to `source_listening`, `reviewer`, `transcript`, `speaker`
equal to `Midbeast`, `names` containing clearly audible League names and terms,
`no_invented_or_skipped_phrases` equal to `true`, and `words` containing at
least three independently checked objects with `text` and source-time `start`.

```sh
python -m annotation.league.pass1 verify-asr --reference reference.json
```

The gate requires WER at most 15%, the listed League terms, the source phrase and
speaker checks, complete word alignment, and each checked onset within 350 ms.
It saves the reference and verification alongside the pilot and seals their
checksums. Inspect the pilot report's measured runtime and GPU memory. Submit the
full job with a time limit derived from
`projected_full_wall_seconds_with_headroom`, allowing scheduler startup overhead.

```sh
sbatch --time=HH:MM:SS annotation/league/slurm_pass1_asr_full.sl
```

The full job checks the saved pilot verification before inference and covers the
entire source audio interval. Collect `producers/asr` and import it into the
current bundle using `prepare --import-producer asr=/collected/producers/asr`.
Finalize with `--draft` while other producers have incomplete coverage, then run
`verify`. Finalization assembles `commentary_timeline.jsonl` and the chronological
`timeline` view in `/pass1`. Each row seeks the existing source player.

Older bundles load with an empty commentary table. ASR becomes a required
full-coverage producer after it has been registered in a bundle. Its report
accounts for transcribed speech, unresolved speech and VAD-negative spans.
Unreviewed speaker attribution is displayed explicitly.

### YouTube caption comparison results

The pilot used YouTube automatic captions as the comparison reference while
audio inspection was unavailable. The subsequent full-recording comparison led
to preferring these captions in the Pass 1 plan. Fetch the original
English JSON3 track using yt-dlp with `--skip-download --write-auto-subs
--sub-langs en-orig --sub-format json3`. Keep the downloaded track and its checksum.

The `verify-asr` command also accepts a reference with `method` equal to
`youtube_auto_captions`, `video_id`, `source_sha256`, `interval`, `reviewer`,
`caption_path`, `caption_sha256`, `meaning_matches`, and `onset_words`.
Use three distinct words spread across the pilot for onset comparison.
The command extracts tokens by their source onset in [30,40), saves the reference,
and reports WER, onset differences and the recorded meaning review. Caption
verification retains unverified speaker identity and pending vocabulary review.

The current comparison passed with 6 edits in 46 reference tokens (13.04% WER).
The onsets of “probably”, “likely” and “video” differed by 0, 82 and 62 ms.
The saved reference records differences including “Drew” versus “True” and words
clipped by the pilot boundaries. The completed full run used one L40S GPU with a
30-minute allocation and a 15-minute backfill minimum.


The full L40S run completed in 443 seconds under job 9841093. It produced 475
utterances and 3,694 aligned words across the 1,732.765896-second recording.
The collected source, audio checksums, coverage partition, timestamp bounds,
ordering and timeline playback checks passed.

Full-recording word disagreement with YouTube captions is 375 edits across
3,940 reference tokens (9.52%). For 3,609 matching tokens, median onset difference
is 78 ms and the 95th percentile is 150 ms. Agreement within 350 ms is 99.61%.
Beginning, middle and ending excerpts preserved the meaning. This comparison
measures agreement between two automated transcripts. Vocabulary and speaker
review remain open.

The ASR output records two partial alignments and 147 zero-duration word
timestamps. Its raw output and flags remain available as experiment evidence.
The current assembled transcript uses the imported YouTube track. The ASR
comparison remains available under its original producer.

The [verification report](../data/league_asr_handoff/20261120-results/verification.json)
contains the samples, common wording differences, alignment exceptions and
bundle verification result.
