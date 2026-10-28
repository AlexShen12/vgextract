# Unverified Midbeast benchmark templates

These are neutral manifests for the pinned source, not completed labels.
Every sample is `verified: false`; crop labels, transcripts, events, tracking
prompts, reference masks and playback-segment IDs must be independently supplied.
No predictions or Domisum labels are included.

The forty native PNG frames are generated under the run's `benchmarks/frames/`
by `experiment --suite all --initialize`. The default split puts earlier sampled
windows in development and later windows in held-out, excluding the twelve
already inspected calibration frames from held-out windows. Adjust sample choices
to locate real difficult cases and real playback discontinuities; do not invent
segments to pass a gate. Twelve inspected clocks all followed source time +33s,
so pauses/rewinds and separate playback segments are not yet demonstrated in this
video. This can prevent release even if reader accuracy is good.

The `labels` list on each crop frame should contain enabled field IDs with
`availability` (`visible`, `absent`, `unreadable`), independently read `raw_text`
and parsed `value` for visible fields. Preserve spaces and punctuation in exact
text. Name slots are player names; they do not automatically resolve champions.
For example, schema descriptions are in `intake_benchmark.py` and field IDs in
`profiles/midbeast-1080p-v1.json`. Do not use sample numbers as target-video truth.

Event `start`/`end` labels and ASR word timestamps use source seconds. Tracking
reference masks use native frame dimensions and frame indices 0–15. Inspect
independent labels before setting `verified: true`. Draft runs remain experimental
and cannot authorize full-video processing.
