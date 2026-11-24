# Selected-player occurrence pilot validation

November 5, 2026

The event-v2 arm and review are runnable. The recognition gate awaits GPU
predictions and independent assessment. Precision, action recall, duplicate rate,
binding accuracy, boundary errors, and repeat-link accuracy have no measured
acceptance result for event-v2.

## Local verification

The full test suite passed 312 tests, including occurrence reconciliation,
phase censoring, competing targets, separate slot claims, unknown-slot hits,
minion attribution, correction invalidation, resumption and production isolation.
The tests use synthetic or cached fixtures for inference behavior.

On the pinned Midbeast source, the CPU pilot processed source seconds 30–38. It
decoded 128 frames and accepted 125 visible body bindings. Two clips contain
one CS display reading and one CS increase. The pilot reported all 17 missing
scene requests explicitly. It ran in 59.3 seconds with 374 MiB peak driver RSS,
zero GPU requests and zero generated tokens. It emitted zero scene occurrences.
In review, the first clip played at 1920×1080 and stopped at 33.012 seconds.
The [playable pilot](http://127.0.0.1:8767/captions?pilot=021fee6d41eabd0a4666)
uses the exact source MP4 and the local review server.

## Source references

Independent source-only labelers covered ten development
neighborhoods (240 seconds) and ten held-out neighborhoods (224 seconds).
The held-out cohort was selected outside 344 recorded calibration and prior
inspection intervals, with eight-second guards. The labelers recorded
unresolvable and censored boundaries explicitly. Held-out labeling was
prediction-blind. Development provenance records historical pilot exposure
where applicable and independence from the current recognizer.

The revised references contain 357 event rows, 361 component claims and 353
bindings. The event rows include 110 explicit unknown intervals. A separate
source-only adjudicator confirmed that ten W-empowered swings are ten basic
attacks with separate W-slot facts. One onset and one impact remained
unresolved. The original labels and exact adjudication overlay are archived.

The November 5 revision had four directed-attack references in each split, below the five-case
minimum. Development lacks a confirmed cast-miss example. Held-out lacks a
recall-channel continuation. Pause, rewind, ownership change and repeat linkage
also lack independently verified positive cases in this revision. These cases
remain untested. Hidden actions in occluded intervals remain unknown.

### November 6 source coverage revision

Independent source review added one directed attack to each split, bringing each
to five. Development now includes the Q miss beside the dragon pit. The visible
release area excludes both the dragon and the opposing champion. Held-out now
includes a recall with a visible onset, uninterrupted channel through scoreboard
and shop overlays, and arrival at the fountain. The fresh 1090.5–1102.5-second
neighborhood replaces the 1399.013–1423.013-second neighborhood and clears the
eight-second guard against recorded and current exclusions. The original labels
and native evidence for the additions are in
`benchmarks/caption-events/v1/revisions/source-coverage-2026-11-06/`.

The strict reference validator passes for all 20 neighborhoods. Both splits
now meet the occurrence benchmark's source case and claim-count requirements.
Prediction metrics are still pending. Screening the pinned source at one-second
intervals found the same game-time offset in all 1,705 readable clock samples
and no selected HUD portrait change during that span. The visual repeat screen
found no candidate with three matching seconds of distinct gameplay motion.
Pauses, rewinds, ownership changes, and repeat links still lack verified positive
examples for the broader pilot.

## Longleaf evaluation

The initial staged snapshot `517851de7f38` passed verification of 128 file hashes,
the 20 reference neighborhoods against the Longleaf source manifest, Slurm script
syntax checks and 78 caption tests in the Longleaf driver environment. Preflight
jobs `8255269` and `8255270` started and then failed because the batch environment
could not find `ffprobe`. The replacement snapshot `caption-events-20261105-media-r2`
adds the installed media environment to `PATH` and checks FFmpeg and Tesseract
before model startup. The GPU preflights need to be resubmitted.

The [batch handoff](league-caption-event-longleaf.md) covers the separate baseline
and revised preflights, development submission and resumption, collection, and
the gated held-out job. Development can run as a diagnostic with these source
references. A passing independent development assessment and sufficient source
coverage are required before freezing the held-out configuration.
