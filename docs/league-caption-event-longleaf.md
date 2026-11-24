# Selected-player occurrence pilot: Longleaf handoff

The new `event-v2` arm is an experimental event protocol using the existing pinned
Qwen3.8-27B-FP8 checkpoint. The `baseline` arm preserves packet-based proposals.
The implementation is runnable; recognition acceptance requires new GPU results
and independent assessments. No accuracy result is implied by software tests.

## What changed

`caption-pilot` defaults to `--arm event-v2`; `--arm baseline` selects the previous
protocol. Production `generate-captions` remains unchanged. Event artifacts stay
under `experiments/captions/<key>/`, with arm/policy/model/code in the key.

`scene_occurrences.jsonl` schema 2 contains physical occurrence IDs, observed
intervals, onset anchors/brackets, phase-frame evidence, censoring, supported
components, unresolved interpretations and source-proposal lineage. One onset
plus windup/release/recovery is one occurrence. Another action needs another
positive onset anchor; continuing packets need shared visible ledger anchors.
Coarse and sparse context proposals never directly supply emitted action evidence.

Candidate discovery remains 4 FPS. Each candidate gets up to two seconds before
and three seconds after within one continuous source/owner context. At most eight
sparse frames provide context; overlapping eight-frame 16 FPS packets establish
actual cues. Linked impact/death fragments must share visible chain evidence,
target identity and sampling rate. Missing target regions or source gaps prevent
joining. Unresolved cues receive one native-frame pass in brackets no longer than
one second; density itself does not prove attribution.

Atomic captions keep their existing fields and add `event_id`. Action type,
`ability_slot`, targeting and outcomes are separate facts. A generic ability hit
may have `ability:null`, with a complete cast-to-target impact chain. Unknown slots
never become Q/W/E/R guesses. Counter-derived `creep_credits.jsonl` records remain
aggregate comparisons and cannot identify individual dying minions. New target
boxes use displayed gameplay pixels with an explicit source transform; HUD and
masked areas are rejected. Legacy source-coordinate proposals keep their old path.

Review at `/captions` groups related claims and shows occurrence phases,
uncertainty and independent assessments. `/caption-labels?events=1` edits only the
new source reference revision. Corrections to proposals/body evidence regenerate
derived outputs; assessments tied to older generations expire.

## Source references and acceptance

New references are under `benchmarks/caption-events/v1/events.json`. There are ten
development and ten held-out neighborhoods, 12–24 seconds each; the initial held-out
selection excludes 344 recorded inspection/calibration intervals with eight-second
guards. The 20 neighborhoods have independent source inspection. The
[validation report](league-caption-event-validation-2026-11-05.md) records the
reference counts, remaining coverage gaps and CPU measurements.

Each reference occurrence has `{id, kind, start, end, onset, boundary_resolvable,
left_censored, right_censored}`. Claims link through `event_id`; slot identity belongs
only to `ability_slot` values. Counter values retain exact before/after/delta.
Bindings identify the claim, role, source time and native-source box. Exhaustive
coverage concerns visible action occurrences and documented unknown intervals;
hidden gameplay actions are never invented. Previously exposed development
examples disclose historical exposure and use the narrower
`independent_of_current_predictions` attestation. Held-out labels require strict
prediction blindness. Empty event/claim/binding/phase categories need explicit
negatives. Unknown playback/repeat categories keep broader validation untested.

Freeze the source labels, prompts/rubric, views, sampling, model/runtime and
implementation before held-out inference. One-to-one matching uses temporal IoU
at least 0.5, onset error at most 0.25 seconds and actor/type compatibility.
Additional fragments contained in one reference occurrence count as duplicates,
including censored recall excerpts. A merged pair leaves a missed event. Multiple
component captions sharing one occurrence ID count once.

Event-stage acceptance requires at least 20 independently assessed non-counter
claims, at least five references/predictions/assessments per required core kind,
95% supported non-counter claims, 95% per-kind support/precision, at least 80%
independently supported event recall, at most 5% duplicate occurrences, correct
bindings and valid evidence. Target accuracy uses actual target assertions as
its denominator. Missing hard-negative/positive cases remain untested. A separate
broader-pilot result retains the original playback/repeat requirements. Independent
assessments supply support; model self-consistency and resolvable links do not.

## Stage and preflight

The verified Longleaf snapshot is ready at
`/work/users/a/l/alshen/league-midbeast-validation/artifacts/caption-events-20261105-media-r2/source`.
The following staging instructions also apply to a later revision.

Stage `annotation/__init__.py`, `annotation/league/`, `ingestion/__init__.py`,
`ingestion/common/__init__.py` and `ingestion/common/replay_state.py` in a new
Longleaf checkout. Keep older source snapshots. Stage the new label directory as `caption-reference-labels/v1/` within that
checkout, preserving its provenance and revisions. Each job snapshots those
references alongside its code. Existing source video, icons and environments are
reused at `/work/users/a/l/alshen/league-midbeast-validation/`.

From the staged repository root, submit each experiment separately:

```bash
cd /work/users/a/l/alshen/league-midbeast-validation/artifacts/caption-events-20261105-media-r2/source
sbatch annotation/league/slurm_caption_events_baseline.sl preflight
sbatch annotation/league/slurm_caption_events_development.sl preflight
```

Each preflight uses four seconds centered within the first development neighborhood.
The files request one L40S, eight CPUs, 96 GB host RAM and two hours. Inspect
requests, truncations, memory, runtime and driver/worker logs before full batches.
Preflight must finish with requests and no malformed/truncated responses before
the corresponding development batch is accepted. Full development also requires
all ten development neighborhoods to have independently verified exhaustive
source labels; unverified templates cannot start the expensive batch. Preflight is not an accuracy test.

Each job runs from its own archived/extracted immutable code and reference snapshot. Baseline
and event-v2 write to separate run roots, so neither conflicts with production
or the separate Qwen3.5 suites. A single model worker is reused across neighborhoods. GPU class and software
runtime must match the saved development recipe; physical GPU UUID stays provenance.
Jobs log progress under the existing work root;
model requests and per-neighborhood reports checkpoint throughout execution.

## Run development and return later

```bash
sbatch --time=24:00:00 annotation/league/slurm_caption_events_baseline.sl development
sbatch --time=24:00:00 annotation/league/slurm_caption_events_development.sl development
squeue -u alshen
```

Use preflight measurements to adjust the requested wall time. A historical L40S
run took about 30 minutes for 3.5 source seconds with heavy native inspection.
At that pace, 240 development seconds would take about 34 hours per arm. Start
with a 24-hour allocation and budget about 48 GPU-hours per arm provisionally.
Native escalation can make the total longer. Resubmit the same
file/mode after a time limit: completed neighborhoods are skipped, complete
requests are cached, partial failures survive, and missing derived artifacts are
rebuilt. Jobs survive logout; this chat need not remain open. Source/prompt/code or
reference changes produce a new version or an explicit stale-run error rather
than silently mixing results. A Slurm exit of zero means execution completed,
not that recognition passed.

The isolated roots are:

```text
/work/users/a/l/alshen/league-midbeast-validation/caption-events/v1/baseline/
/work/users/a/l/alshen/league-midbeast-validation/caption-events/v1/event-v2/
```

Find the full-development `experiments/caption-events/<key>/report.json` (the report
has `draft:true`, `preflight:false`), then create a portable bundle:

```bash
/work/users/a/l/alshen/league-midbeast-validation/env-driver/bin/python -B \
  -m annotation.league.caption_batch bundle --report /absolute/path/to/report.json \
  --output /work/users/a/l/alshen/caption-development-results.tar.gz
```

Return the bundle and the Slurm `.out`/`.err` logs. The default compact bundle contains execution
reports, source-only reference snapshot, occurrence/claim metadata, evidence
references, worker logs, hashes and implementation source. Review uses the exact
local source MP4. Add `--include-images` to include the original cited native PNGs;
this can make the bundle substantially larger. It exports no video. Original full model
inputs and unbundled frames remain on Longleaf. Import verifies included files; it
does not claim to have verified PNGs omitted from a metadata-only bundle.

Import to a **new** local review run (existing evidence is never overwritten):

```bash
python -B -m annotation.league.caption_batch import \
  --archive /absolute/path/to/caption-development-results.tar.gz \
  --run-dir /absolute/path/to/new-caption-review-run \
  --source /absolute/path/to/OLLZBe5-fg4.mp4
python -B -m annotation.league --run-dir /absolute/path/to/new-caption-review-run review --port 8767
```

Import verifies archive members and the exact MP4 hash, projects paths, preserves
claim facts/generation keys, and records original remote provenance. Independent
review can then assess generated claims without changing the source references.

## Collect, freeze, then evaluate held-out

Collection never loads a model or reruns perception:

```bash
python -B -m annotation.league.caption_batch collect --report /absolute/path/to/report.json
```

It writes `assessed-report.json` next to the immutable execution report. All
neighborhoods must be present, references must match the executed snapshot,
generation keys must match and corrected outputs cannot establish acceptance.
Copy local assessment files back to their corresponding isolated Longleaf pilot
roots before freezing there. Keep imported source-only reference facts unchanged.

After the independently assessed development event gate passes for the unchanged
candidate configuration:

```bash
/work/users/a/l/alshen/league-midbeast-validation/env-driver/bin/python -B \
  -m annotation.league.caption_batch freeze --arm event-v2 \
  --source-run /work/users/a/l/alshen/league-midbeast-validation/run \
  --work-root /work/users/a/l/alshen/league-midbeast-validation/caption-events/v1 \
  --labels "$PWD/caption-reference-labels/v1" \
  --icon-references /work/users/a/l/alshen/league-midbeast-validation/assets/ability-icons-16.13.1/manifest.json
sbatch annotation/league/slurm_caption_events_held_out.sl
```

Use the same environment and asset variables as `caption_longleaf.sh` when running
freeze/collection commands directly. Freeze requires the matching complete,
independently assessed development receipt; it cannot be bypassed by a successful
preflight or counter-only output. Held-out inference requires that exact freeze
and verified held-out labels. If development fails, keep iterating there and do
not submit the held-out job.
