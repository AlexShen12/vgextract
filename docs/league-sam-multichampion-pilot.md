# Simultaneous champion masks for DINOv2

This development pilot samples a full-frame gameplay window, reuses the stable allied/enemy health-bar proposals, and chooses a common frame where both bindings have enough evidence. It runs three SAM conditions on that same native-frame sequence:

1. One box-prompt session per champion.
2. Both champion boxes in one SAM session.
3. Both boxes and the generic `video game character` text prompt in one SAM session.

Every output target has a native binary mask per returned frame, a second mask with HUD and health-bar pixels removed, a matching body image crop, source time, team, SAM object ID, and hashes. The crops are shaped for the existing `FrozenEncoder` DINOv2 body matcher. DINOv2 results are appearance scores; they are not probabilities.

## Prepare the pilot

Use the source Midbeast recording and its calibrated action profile. The default interval covers the simultaneous friendly/enemy proposal near 40 seconds:

```bash
python -m annotation.league.sam_multichampion prepare \
  --video data/youtube/lol_midbeast/OLLZBe5-fg4.mp4 \
  --profile annotation/league/profiles/midbeast-actions-1080p-v2.json \
  --start 39.9 --end 41.5 --fps 16 \
  --output data/league_longleaf_results/2026-11-12/sam-multichampion-40s
```

Inspect `seed-preview.jpg` and `experiment.json` before staging. The preview shows both seed boxes on the same full frame. The generated champion labels are intentionally blank; the health-bar detector only supplies anonymous team bindings. If more than one pair ties on stable evidence, preparation records the alternatives and the SAM command requires `--allow-ambiguous` after the targets have been reviewed.

The `inputs/` directory contains the full-frame SAM sequence. `frames/` keeps lossless source frames for the DINOv2 crops. Preparation also snapshots the required Python package under `source/`, copies the batch file, and writes SHA-256 manifests. Transfer the entire prepared directory to Longleaf, then submit from inside it:

```bash
cd /work/users/a/l/alshen/league-midbeast-validation/artifacts/sam-multichampion-20261112-v1
sbatch slurm_sam_multichampion.sl
```

The job writes to `results/<job-id>/`. Inspect `report.json`, per-target previews, and the binary masks before using them for identity evaluation.

## Match the saved masks with DINOv2

The static gallery builder reads patch-pinned CommunityDragon SKN meshes and diffuse textures, including older SKN versions used by alternate skins. It uses the skin BIN metadata to resolve textures per submesh and hide alternate geometry. Each skin gets 16 orthographic views with alpha masks. Static poses approximate the shape and colors in gameplay. The renderer leaves animation and in-game shader effects to the rigged render path.

```bash
gallery=data/league_recognition_v1/body-render-gallery-16.22-enemy-base
python -m annotation.league.body_static_skn --offline --output "$gallery" \
  --mesh-review "$gallery/mesh-review.json" \
  --champion Ahri --champion JarvanIV --champion Leona --champion Singed --champion Xayah
python -m annotation.league.body_render --input "$gallery/render-report.json" \
  --audit "$gallery/render-review.json" --output "$gallery/gallery.json"
```

The pilot uses the five opposing scoreboard champions from job 9001798. Use `--all-skins CHAMPION` for every skin ID in the pinned catalog or repeat `--skin CHAMPION SKIN_ID` for a subset. Omit `--offline` to fetch missing assets. The per-skin mesh review ties detached equipment and game shader exclusions to the original mesh hashes. Review each skin’s renders and masks before adding it to the gallery.

The body crops are already separated by SAM arm and health-bar target. After copying `results/<job-id>/` back, run the appearance matcher in the body-model environment with an audited gallery and a policy frozen on development data:

```bash
python -m annotation.league.sam_multichampion match-dino \
  --experiment data/league_longleaf_results/2026-11-12/sam-multichampion-40s/experiment.json \
  --results data/league_longleaf_results/2026-11-12/sam-multichampion-40s/results/JOB_ID \
  --encoder-lock data/league_assets/body-encoder/dinov2-vits14/encoder-lock.json \
  --gallery path/to/audited-body-gallery.json \
  --policy path/to/frozen-body-policy.json
```

The gallery entries use the existing body-gallery fields (`champion`, `reference_kind`, image and mask paths/hashes, reviewer and identity evidence). Entries from this pilot's source video are excluded. `dino-match.json` reports per-frame candidates and a pooled embedding per target and arm; the frozen policy controls whether a champion name is accepted. It also records the encoder signature and comparison candidates so box-only and joint prompting can be compared on identical frames.

This is a short two-champion pilot. It does not establish crowded team-fight performance or identity accuracy without an independent reference gallery and human review.

## Compare skin encoders on saved SAM crops

Use `compare-skins` to rank the same audited 3D render gallery with DINOv2 ViT-S/14 and Google SigLIP ViT-B/16. Each encoder sees the same foreground-only crops on a gray canvas. For each frame, the highest cosine across a skin's views gives that skin's score. The track ranking averages those scores across the visible frames. Give `--champion` once per candidate to restrict a shared gallery to a reviewed roster or a known champion.

Run in a CPU PyTorch environment with Transformers and Safetensors installed. The local pilot used Transformers 4.57.1. Stage Google’s `config.json`, `preprocessor_config.json`, and `model.safetensors` next to `model-lock.json`. The lock records `model_id`, the full `revision`, and a `files` object mapping those three filenames to their SHA-256 hashes. The command verifies those files before loading the vision tower offline. [Pinned SigLIP checkpoint](https://huggingface.co/google/siglip-base-patch16-224/tree/7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed)

```bash
gallery=data/league_recognition_v1/body-render-gallery-16.22-enemy-skins
OMP_NUM_THREADS=4 python -m annotation.league.sam_multichampion compare-skins \
  --results data/league_longleaf_results/2026-11-12/sam-multichampion-40s/results/9001798 \
  --arm joint-boxes-text --target enemy-1 --gallery "$gallery/gallery.json" \
  --encoder-lock data/league_assets/body-encoder/dinov2-vits14/encoder-lock.json \
  --siglip-lock data/league_models/siglip-base-patch16-224/model-lock.json \
  --output "$gallery/sam-9001798-encoder-comparison.json"
```

The report records input hashes, encoder signatures, the ranking for each encoder, and per-frame scores keyed by the shared frame list. Compare ranks against independently verified gameplay labels to assess exact-skin accuracy.

Add `--top-foreground-frames 5` to also rank skins using the five frames with the highest foreground pixel counts. The count comes from the cleaned SAM mask at source resolution. The report includes the fraction of the tight mask bounding box filled by foreground pixels, which can help flag masks with stray pixels or separated pieces. The subset scores follow the order in `top_foreground_frame_indices`.

## Run the scoreboard-backed appearance pass

Prepare anonymous sightings from a bounded interval covered by an existing scoreboard. The health-bar proposal requires Tesseract to read a level from 1 to 18 in the overhead badge before the bar can seed SAM. Use `LEAGUE_TESSERACT` to override the executable. Supply the player champion to exclude the HUD-bound player track before choosing seeds. The preparer saves one seed preview for each admitted sighting and records each unresolved friendly track. When source review shows the player offscreen, `--player-offscreen` takes a JSON object with the source SHA-256 and a `frames` list of reviewed source frame IDs.

```bash
run=data/league_recognition_v1/midbeast-sam-siglip-35-215
python -m annotation.league.sam_multichampion prepare-appearances --root "$run" \
  --video data/youtube/lol_midbeast/OLLZBe5-fg4.mp4 \
  --scoreboard-skim data/league_recognition_v1/body-pilot-20261112/scoreboard \
  --profile annotation/league/profiles/midbeast-actions-1080p-v2.json \
  --player Darius --start 35 --end 215
```

Stage `appearances.json`, the referenced `sam-inputs/` files, and the batch file in a Longleaf run directory. Keep the lossless `frames/` files with the local run for crop extraction. Snapshot the repository's `annotation/` directory under `source/annotation/` and generate `source/SOURCE_SHA256SUMS` from inside `source/`. SAM uses the seed box and the generic text prompt `video game character` in the same request. Submit array index 0 as a smoke test, then indices `1-6%2` for the full run. The smoke tests up to three of the longest appearances per represented team and needs successful masks for each represented team. The full shards reuse checked smoke outputs. Resubmit a failed index to rerun only its incomplete or damaged appearances.

If SAM loses its tracking state in one direction, the worker attempts the other direction and retries the same seed at 8 FPS and then 4 FPS. If coverage stays incomplete, it saves the best partial result for inspection and excludes it from skin-lock voting.

```bash
sbatch --array=0 slurm_sam_appearances.sl
sbatch --array=1-6%2 slurm_sam_appearances.sl
```

Copy `masks/` back into the local run. Combine audited render galleries for both teams, keeping the player outside the gallery for this pilot. Run the CPU matcher with the staged SigLIP checkpoint.

```bash
OMP_NUM_THREADS=4 python -m annotation.league.sam_multichampion match-appearances \
  --root "$run" --gallery "$run/gallery.json"
```

For a dependent Longleaf CPU job, also stage the lossless frames covering the admitted appearances, a gallery with paths valid on Longleaf, the pinned skin catalog manifest under `asset-cache/`, the scoreboard evidence under `scoreboard/`, and the SigLIP model files under `siglip/`. Submit `slurm_siglip_appearances.sl` with `--dependency=afterok:SAM_ARRAY_JOB_ID`. The matcher verifies the relocated scoreboard and each selected native frame against the prepared input hashes.

`appearance-matches-siglip.json` lists outcomes, selected masks, full and lock-restricted rankings, render gaps, and the evidence for each vote. `mask-previews/` shows the selected crops. A completed pass writes `skin-locks-siglip.json` for a compatible `match-frame` run. These locks are model decisions based on separate appearances and the audited gallery.

## Compare Qwen mask verifiers

Use `python -m annotation.league.mask_verifier prepare --run RUN --root PILOT` to freeze the 40 crops from the two smoke appearances. Review the anonymous sheets and write `reviews.json` before viewing model results. The review records the manifest SHA-256, reviewer, `predictions_seen` set to `false`, and one verdict and reason per crop ID.

Stage the manifest, images, and checksum-verified source snapshot on Longleaf. The scripts `slurm_mask_qwen_9b.sl` and `slurm_mask_qwen_27b_fp8.sl` run one request per crop and checkpoint results separately. Run `python -m annotation.league.mask_verifier evaluate --run RUN --root PILOT` after collecting `predictions/`, or submit `slurm_mask_qwen_score.sl` with both GPU jobs as `afterok` dependencies. For the cluster score job, stage the frozen reviews, the relocated gallery and selected original crop and mask files under `replay/`, and the scorer snapshot under `score-source/`.

The scorecard reports false acceptances, retained clean masks, abstentions, latency, and memory. The replay compares the original five-mask ranking with the ranking using all accepted masks in the sample. Scores and any selected verifier are experimental.
