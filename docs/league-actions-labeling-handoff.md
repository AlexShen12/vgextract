# Independent League action labeling handoff

This is a separate annotation task for an agent working without the recognition
chat or detector predictions. Do not implement recognition changes or run the
recognition model. Establish reference labels by inspecting source footage.

## Locations

- Local project: `/Users/alex/Desktop/Projects/videogame`.
- Source: `data/youtube/lol_midbeast/OLLZBe5-fg4.mp4` under that project.
- Source SHA256: `f4f36e1a274617ecbe224eaf7fc3f8375f757acb7a517bcc0ee7780fcd83f240`.
- Label manifest: `data/league_runs/midbeast-OLLZBe5-fg4-v2/benchmarks/actions-v3/actions.json`.
- Source-only review: `http://127.0.0.1:8765/action-labels` while the League review
  server is running. Start with `python -B -m annotation.league review` if needed.
- Existing Longleaf source directory:
  `/work/users/a/l/alshen/league-midbeast-validation/source-20261030-actions-r7`.
  Its sibling `run/benchmarks/actions-v3` contains the remote copy of the labels.
  Label locally, then synchronize that manifest and its revisions into the
  existing remote run. Do not create another project checkout.

Do not open action prediction review, `experiments/`, inference caches, pilot
findings, model responses, or implementation-result handoffs. Do not copy OCR or
recognition outputs into reference labels. If prior exposure prevents independent
annotation of a neighborhood, mark it contaminated and select an eligible
replacement before detector evaluation. Use source frames/video directly; record
the actual annotation method and unresolved cases.

## Required work

1. Read the manifest and verify source identity. It contains twenty eight-second
   clips, ten development and ten held-out; all begin unverified. Inspect every
   clip before calling empty action arrays quiet negatives.
2. Adjust development windows for real coverage: visible/obscured player bars,
   competing green regions, minion last hits, quiet farming negatives, jungle
   credit, passive gold/purchases, recall and rapid slot transitions. Missing cases
   remain untested. Preserve the ten/ten split, eight-second cross-split guard,
   calibration exclusions and repeat-group isolation enforced by the validator.
3. Use absolute source seconds and uncertainty bounds. Replay rapid sequences at
   native frame resolution. Label unknowns rather than inferring hidden actions.
4. Save independent annotations, case tags and a specific labeler identifier such
   as `independent-agent:<task-id>`. Set each clip `verified:true` only after a
   complete source inspection. Set `independent_of_predictions:true` truthfully.
   Record `labeler` and `independent_of_predictions:true` on each verified clip as
   well as on the manifest. The source-only editor does this automatically.
   The review editor records revisions separately from production corrections.
5. Validate the full manifest with the code below. Do not freeze development
   choices, run detector inference or score the held-out split in this task.

## Label contract

Each clip keeps its `id`, `start`, `end`, `split` and optional `repeat_group`.
Update these arrays and `case_tags`:

- `identities`: `{time, owner_id, hud_champion, scene_champion}`. Use owner
  `selected`; champion names are nullable. Ground HUD identity in visible icon or
  explicit independent in-game identity evidence, not the video title or skin.
- `bindings`: `{time, owner_id, binding_visible, champion, bar_box, actor_region}`.
  Sample visible and negative intervals at 4 FPS, with native frames around
  disappearance, cuts, swaps and ambiguity. Bounds are native 1920×1080 pixel
  rectangles `[left, top, right, bottom]`. Positive references need both boxes;
  `bar_box` covers the full-width HP strip including its unfilled tail, excluding
  the level cell and separate resource bar. The source-only editor can capture a
  paused frame and collect both rectangles by dragging on the reference canvas.
  negatives use `binding_visible:false` and nullable boxes. The actor region is
  approximate localization below the overhead bar, not segmentation.
- `transitions`: `{start, end, owner_id, slot_id}`. Record newly established
  ready-to-cooldown activations, visible consumed charges/recasts. Do not label
  countdown ticks, initial already-cooling slots or cooldown completion as uses.
- `actions`: `{start, end, kind, owner_id, champion, slot_id, count_delta,
  terminal_status}` with fields relevant to each kind and unknowns null.
  `creep_credit` records observed CS delta as an aggregate, not a number of
  lane-minion deaths. `minion_last_hit` needs a visible player action linked to a
  dying lane minion. Keep jungle targets separate. A gold counter increase alone
  never establishes a last hit or exact reward.
- `recall` actions use `terminal_status`: `completed`, `interrupted`, or
  `ending_unknown`, plus `left_censored`/`right_censored` booleans. Completion needs
  an observed channel ending and compatible base arrival; a cut/disappearance is
  insufficient. Do not infer interruption causes. Supplementary development
  sequences up to 24 seconds may document a full lifecycle, but do not alter the
  benchmark's eight-second clip length.

Capture source-time evidence notes and `sequence_id` for independently ordered
rapid actions. Do not fabricate a precise timestamp when only a bracket is visible.

## Validation and deliverable

Run from the local project using its driver Python:

```python
from pathlib import Path
from annotation.league.core import Store, read_json
from annotation.league.action_v3 import load_profile
from annotation.league.action_benchmark_v3 import validate

store = Store(Path('data/league_runs/midbeast-OLLZBe5-fg4-v2'))
data = read_json(store.path('benchmarks/actions-v3/actions.json'))
validate(store, data, load_profile(), draft=False)
```

Deliver the verified manifest/revisions, annotation method, coverage and unresolved
or contaminated cases. Report any unavailable case without manufacturing labels.
Preserve model, production and existing experimental artifacts. This task does
not authorize production promotion or full-video processing.
