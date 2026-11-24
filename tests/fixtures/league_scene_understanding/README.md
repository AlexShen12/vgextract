# Source-backed League HUD regression

`apprehend-hud.json` identifies 16 source samples at 16 FPS over source seconds 93.3–94.3 of the local Midbeast recording. The source SHA-256 and exact PTS are preserved. Each PNG is the unmodified native HUD rectangle `[789,978,983,1032]` from its source frame. The E/W slot boxes from the source profile are translated into that crop's coordinates; the fixture supplies only the HUD reader fields it uses. The PNG hashes are pinned in the JSON.

A parent assistant inspected the enlarged QWER strips and the original gameplay/cast-bar images. The cast bar reads Apprehend before E's centered 26-second cooldown appears. E's upper 70 and W's upper 40 are costs. The new cooldown is bracketed by source PTS 93.56013333333334 and 93.62686666666667. Targets and outcomes remain unresolved. These are model-reviewed development references, with no human verification or independent performance claim.

The tests run the actual HUD OCR, persistent-cooldown detector, HUD-primary action builder and saved-action review reader. They verify cost/countdown separation, onset retention through scene abstention, no duplicate activation when a clip starts in the countdown, and ownership-safe association. Scene proposals in the ownership check are injected interface inputs; that check does not evaluate actor recognition.

Reproduce source samples from the repository root using the local benchmark manifest:

```sh
.venv/bin/python -B -m annotation.league.scene_reference detail \
  --manifest data/league_runs/midbeast-OLLZBe5-fg4-v2/benchmarks/scene-understanding/v1/negative-candidates.json \
  --clip negative-01 --start 93.3 --end 94.3 --fps 16 \
  --output /private/tmp/scene-reference-hud-reproduction
```

The fixture PNGs are cropped directly from the decoded MP4, without a JPEG intermediate. Source-review JPEG exports are for visual review; re-extract the exact PTS from the MP4 to reproduce lossless fixture pixels. Fixture pixel edits require an explicit source re-review and updated hashes. OCR checks require Tesseract and are skipped when it is unavailable; reference-contract tests do not depend on Tesseract.
