# League SAM 3 champion-body fine-tuning pilot

This experiment fine-tunes the SAM 3 image detector/segmenter, with the visual and
text backbones frozen. The SAM 3.1 multiplex tracker stays fixed. A predicted body
mask supplies interior foreground points directly to a new tracker object; the
old box/text detector is not a prerequisite. Champion and skin recognition remain
separate. Production checkpoints and previous experiment receipts are unchanged.

## Inputs and labels

Run from the repository root, using Python with Pillow, NumPy, SciPy, PyAV,
pycocotools and the YouTube downloader dependencies. The candidate source list is
`annotation/league/sam_finetune_sources.json`. Six additional videos are collected,
three Midbeast and three DomisumReplay. The latter cover Singed, Orianna and Xayah.
Titles are discovery hints, not labels or proof of distinct underlying matches.

```bash
.venv/bin/python -m annotation.league.sam_finetune collect
.venv/bin/python -m annotation.league.sam_finetune prepare \
  --selection data/league_recognition_v1/sam3-finetune-v1/selection.json
```

`selection.json` records each video's reviewed match identity, evidence, four
five-second intervals (`walking`, `casting`, `minions`, `interaction`), a native
pixel crop box per interval, and four negative timestamp/crop pairs. Coordinates
are `[left, top, right, bottom]`, with exclusive right/bottom edges. Actual decoded
source PTS, hashes, source geometry, and crop offsets are saved in `dataset.json`.
Each interval retains 16 FPS source frames and seven selected annotation frames.
Four/one/one matches belong exclusively to train/validation/test. No adjacent
frame or alternative POV of the same match may cross those boundaries.

The expected initial dataset is 192 images (168 positive candidates and 24
negative candidates). A candidate remains excluded until reviewed. Copy
`reviews.template.json` to `reviews.json`, retaining its dataset and image hashes.
For each image, record `accepted`, `uncertain`, or `rejected`, and a reason for
exclusion. Every accepted crop requires `all_champions_labeled: true`. Every visible
champion has a clip-local `track_id` plus either crop-coordinate polygons or a
binary PNG `mask_path` and `mask_sha256`. Use `target` for the tracked champion;
its first accepted appearance also needs a crop-coordinate `bar_box`.

Trace visible body, costume and equipment, excluding HUD, shadows, detached
projectiles and spell effects. A negative crop has no champion instances. Label
other visible champions too, including visible portions at crop boundaries. Do
not invent hidden body geometry. Reject unresolved contours rather than treating
an automatic proposal as truth. Label provenance is explicitly agent-reviewed.

```bash
.venv/bin/python -m annotation.league.sam_finetune export
```

Export writes COCO RLE masks, boxes and areas under `coco/{train,validation,test}`
and freezes `labels.lock.json`. Every split must retain positives and negatives.
Unreviewed samples, missing exhaustive review, contradictory negatives, hash
changes and match leakage stop export. Frozen reviews cannot be edited in place;
start a new experiment root for changed labels.

## Longleaf setup and execution

Stage this repository snapshot under
`/work/users/a/l/alshen/league-sam3-finetune-v1/source`, and the prepared dataset
under its sibling `data` directory. This pilot has its own environment, logs and
weights. `setup_sam_finetune_longleaf.sh` reuses the existing CUDA library
environment without installing into it. The current upstream source pin is
`f1647e8d0c1d280991f4d1b7d974c2704537fe66`.

```bash
bash annotation/league/setup_sam_finetune_longleaf.sh
```

Cache SAM 3 with existing authenticated Hugging Face access before offline GPU
jobs. Resolve and pass an immutable model revision. The pilot initialization is
`facebook/sam3@3c879f39826c281e95690f02c7821c4de09afae7`; its weight SHA-256 is
`9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`.

```bash
export LEAGUE_HF_HOME=/work/users/a/l/alshen/league-midbeast-validation/hf-home
source annotation/league/longleaf_paths.sh
/work/users/a/l/alshen/league-sam3-finetune-v1/env/bin/python -m annotation.league.sam_finetune \
  cache-model --root /work/users/a/l/alshen/league-sam3-finetune-v1/data \
  --revision 3c879f39826c281e95690f02c7821c4de09afae7
sbatch annotation/league/slurm_sam_finetune_propose.sl
```

The proposal job writes unreviewed masks under `proposals/`. Inspect and correct
these locally, then transfer the reviews and exported COCO artifacts. Submit each
training job separately, after its prerequisite completes:

```bash
sbatch annotation/league/slurm_sam_finetune_smoke.sl
sbatch annotation/league/slurm_sam_finetune_train.sl
sbatch annotation/league/slurm_sam_finetune_evaluate.sl
```

When GPU scheduling delays annotation, `sam_finetune_cpu_proposals cache` caches
the official Transformers representation at the same checkpoint revision and
`slurm_sam_finetune_cpu_propose.sl` generates proposals in `proposals-cpu/` using
16 CPU cores, 64 GB RAM and a 45-minute bound. Its own initialization hashes,
library versions and recipe are recorded; these masks are annotation drafts only.
Training and held-out inference retain the Meta implementation. Both proposal
paths resume only compatible, hash-verified receipts. Render review images with:

```bash
.venv/bin/python -m annotation.league.sam_finetune review-panels --proposal-dir proposals-cpu
```

The smoke checks positive and negative loading, finite focal/Dice and detection
losses, trainable parameter changes, frozen parameter equality and checkpoint
reload equality. Full training refuses a missing or mismatched smoke receipt.
The jobs request one L40S, four CPUs and 64 GB host RAM; their wall-clock limits
are 20 minutes, three hours and two hours. Proposal generation has its own
20-minute job. Account and partition availability must be checked before launch.

The loop uses upstream SAM datasets, transforms, collator, matcher and losses,
with an explicit single-GPU optimizer loop. No empty-target filter is applied.
Configuration is in `sam_finetune_config.json`: AdamW, learning rate `1e-5`, batch
one, accumulation four, BF16 and at most 20 epochs. Validate every epoch on the
validation match; select by mean instance IoU, counting misses as zero. The held-out
test match never selects weights. Head checkpoints require the exact pinned base
checkpoint and frozen label manifest; they are not standalone SAM checkpoints.

## Reading results

`evaluation/scorecard.json` compares baseline and fine-tuned segmentation on the
same held-out crops. It records IoU, boundary F-score with two-pixel tolerance,
recall at IoU 0.5, missing instances, contamination and negative false positives.
Tracks compare baseline-mask points, fine-tuned-mask points and reviewed-mask
points with identical sampling and fixed SAM 3.1 weights. Health-bar association
abstains on ambiguous candidates. No corrective prompts are supplied during this
comparison.

Tracking records include expected and returned frame counts, errors, per-frame
mask hashes, sparse-reference drift and identity switches, seeding-plus-tracking
time and peak GPU memory. A successful Slurm exit is insufficient evidence of
complete propagation. Overlays and a Markdown report accompany the scorecard.
Agent labels and one held-out match make this exploratory evidence, not a claim
of generalization across League champions or skins. Existing Midbeast failures
remain available under `sam-skin-study-darius-masking-v1/inspection` as diagnostic
evidence, separate from the new held-out match.

```bash
.venv/bin/python -B -m unittest tests.test_league_sam_finetune -v
```
