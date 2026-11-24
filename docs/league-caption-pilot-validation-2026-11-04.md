# Selected-player caption pilot follow-up — November 4, 2026

The complete 95% pilot acceptance gate remains unmet. This follow-up completes the
Longleaf development experiment that was running when the earlier session stopped.
The original frozen benchmark and its source-only references are unchanged.

## Completed development run

Job **7935286**, pilot `29c025e1959615c09863`, inspected source seconds
103.7–107.2 around a positive development ability-impact example. The experiment
used a native 640×640 player crop, source-coordinate target-box remapping, separate
HUD context, explicit target-localizer instructions and compact response limits.
The experimental source is archived at
`data/league_caption_pilot_validation/longleaf-r5-source/`; it has not been adopted
into the main caption generator.

Independent source review supported **4/16 claims (25%)**: the two counter facts
and two excerpts of one recall channel. The second recall excerpt is redundant.
Actor bindings were correct in 16/16 rows; no target was asserted. All 125 citation
links resolve, but cited evidence supports only 4/16 typed claims. The reviewer
inspected 69 distinct native images against independent source decoding. The
other twelve claims confuse Q windup/release/recovery or recall animation with
new attacks, casts or movement. Higher sampling density and schema-valid model
responses did not establish action identity or occurrence boundaries.

The crop prototype **is not adopted**. The original sixteen claims and their
generation-matched assessments are preserved in the experimental root, alongside
`development-review.json` with per-claim rationale and duplicate/competing-event
notes. No predictions were corrected to improve this result. It is development
evidence and cannot establish held-out precision. A new scene approach must first
resolve these development failures before another held-out GPU evaluation.

The run completed 29 requests in 1,781.54 pipeline seconds, using 15,151 generated
and 151,817 prompt tokens. Peak reported GPU allocation was 39,607,861,248 bytes on
one L40S. Reported host RSS was 248,098,816 bytes for the driver; it excludes the
separate model worker. No requests were truncated or rejected as malformed.

It emitted 16 atomic claims: one displayed CS reading, one CS increase, seven
attacks, four casts, two recall channels and one movement. **No ability hit, directed
targeted attack or individual last hit survived grounding.** Parseable responses
and dense native sampling did not resolve those outcomes.

All evidence was copied into the local experimental pilot root. Its 314 native and
model-input image hashes and the pinned MP4 source hash were verified before
independent review. `local-projection.json` records path projection without claim
changes; the original remote JSON/JSONL metadata is retained in
`data/league_caption_pilot_validation/29c025e1959615c09863-remote-metadata.tar`.

## Verification

`python -B -m pytest -q -p no:cacheprovider tests`: **287 passed**. The affected
caption/pilot subset is **53 passed**, including the original 25 caption tests.
The repository has acquired additional Qwen3.5 experiments since the earlier
254-test run; those jobs and artifacts are separate from this caption experiment.

Body-region corrections now invalidate scene evidence inspected with the old
binding. Scene proposals retain signatures of the body evidence shown to the
reader; extraction and rendering share the signature check. Legacy proposals are
compared with their original CPU body rows. Unaffected evidence remains usable,
while changed cues require reinspection with the corrected body context. A
regression test verifies invalidation for both formats and successful fresh
inspection; completed model requests for unaffected context remain reusable.
The review UI displays independent support counts and per-claim failures, and its
header wraps within the narrow review pane. Source playback was checked at
1920×1080 and stopped at 37.999 seconds for the clip ending at 38 seconds.

The local review player is restored at
[the corrected development clip](http://127.0.0.1:8767/captions?pilot=33d5dd113378e5b7037d).
That earlier three-claim example has separate 3/3 independent support, following
recorded corrections. It is ineligible for benchmark acceptance. The frozen
held-out counter-only result remains 36/36 supported; it does not test scene
recognition. Full historical measurements and unsupported cases are in
[the November 2 report](league-caption-pilot-validation-2026-11-02.md).
