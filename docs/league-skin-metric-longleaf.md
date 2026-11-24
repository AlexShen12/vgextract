# League skin and chroma metric learning

The v1 catalog is pinned to Data Dragon 16.22.1 and CommunityDragon 16.22. It contains 9,263 identities across 173 champions, including 173 bases, 1,964 named skins, and 7,126 chromas. The dataset uses static body renders. Exact skin accuracy on gameplay requires a separately verified set of gameplay crops.

## Run on Longleaf

The campaign root is /work/users/a/l/alshen/league-skin-metric/16.22-v1. The tools use the existing env-tracking Python environment, which inherits PyTorch and Torchvision from env-vl. Staged source snapshots include SOURCE_SHA256SUMS and are checked at job start. Run the inventory builder against the local catalog before staging the inventory and source.

~~~bash
python -B -m annotation.league.skin_metric_assets --root CAMPAIGN inventory \
  --catalog-cache data/league_assets/skin-references/16.22.1_16.22
~~~

Create the campaign's logs and jobs directories before submitting. From the staged source directory, submit slurm_skin_metric_pilot_assets.sl and slurm_skin_metric_weights.sl. Submit slurm_skin_metric_pilot_render.sl and slurm_skin_metric_all_assets.sl with an afterok dependency on the pilot asset job. Store the parsable Slurm IDs under jobs/pilot-assets.id, jobs/weights.id, jobs/pilot-render.id, and jobs/all-assets.id.

Review the PNG sheets in pilot/ for Aatrox, Ahri, and Lux. They pair a training view and a validation view per identity. Compare parent and sibling textures and reject renders with incorrect colors, missing body geometry, or detached pieces. Record each rejected ID with a repeated --reject argument. The approval records the checked shard hashes and requires a valid base and chroma for all three pilot champions.

~~~bash
root=/work/users/a/l/alshen/league-skin-metric/16.22-v1
cd "$(cat "$root/jobs/source-path")"
source annotation/league/longleaf_paths.sh
export LD_LIBRARY_PATH="$LEAGUE_WORK_ROOT/conda_envs/videogame-league/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python_bin="$LEAGUE_WORK_ROOT/league-midbeast-validation/env-tracking/bin/python"
"$python_bin" -B -m annotation.league.skin_metric_dataset --root "$root" approve-pilot \
  --reviewer REVIEWER
bash annotation/league/submit_skin_metric_full_longleaf.sh
~~~

The submission script records the two-epoch GPU smoke run, render array, manifest merge, both full training jobs, and their comparison. Slurm starts dependent stages after successful upstream completion. The render worker verifies the reviewed pilot and checksums of existing views before reuse. Reissue render work for failed array indexes only. Run the same training script against the same snapshot and dataset to recover from last.pt. The full runs use eight epochs, 16 classes per batch group, and four views per class in each of four updates. Both objectives combine ArcFace with the same batch-hard triplet term. The hard sampling arm uses the same class/view exposure and replaces random batch composition after epoch one.

## Artifacts and verification

- inventory.json and downloads/ account for every identity and asset request. Shared blobs in assets/objects are keyed by SHA-256. Cached 404 and 410 results describe gaps in the pinned export. Transient failures are retried and remain repairable.
- renders/ and shards/ contain 224 × 224 RGBA body renders, asset binding provenance, and per-image hashes. The 16 training views use headings from 0° through 315° in 45° steps and elevations of 45° and 60°. Validation uses 50° elevation and headings 22.5°, 112.5°, 202.5°, and 292.5°. Test uses 55° elevation and headings 67.5°, 157.5°, 247.5°, and 337.5°.
- dataset.json reports coverage by base, skin, and chroma, and excludes unresolved and indistinguishable identities from training. Identical output across a split boundary removes the affected identity. A complete 24-view signature shared by several identities removes the whole indistinguishable group.
- runs/{random,hard}/ contains resumable last.pt checkpoints, best.pt selected by macro validation top-1, inference.pt, gallery.pt, and metrics.json. The latter includes exact top-1/top-5, per-kind metrics, and within-parent confusions. The trainer reloads the exported encoder and gallery and checks that they reproduce the rankings before publishing metrics. comparison.json reports test deltas for the two arms.

To inspect predictions for a verified gameplay crop separately, pass it as a rectangular RGB crop or as an RGBA crop with the foreground in the alpha channel. Keep independently verified labels and the resulting predictions outside the synthetic scores.

~~~bash
"$python_bin" -B -m annotation.league.skin_metric_train --root "$root" predict \
  --run "$root/runs/hard" --crop /absolute/path/body.png
~~~

Use sacct against the IDs under jobs/ to check terminal status, elapsed time, peak memory, and exit codes. The campaign writes live data under /work. Longleaf's /users space can hold an inactive copy after the experiment.

## Follow-up experiments

The follow-up source snapshot is recorded in jobs/followup-source-path. Submit both standalone scripts from that snapshot. It contains the gameplay track manifest and the two masked frames from the Aatrox smoke test.

~~~bash
sbatch annotation/league/slurm_skin_metric_hard_continue.sl
sbatch annotation/league/slurm_skin_metric_rerank.sl
~~~

- jobs/hard-continue.id records the continuation from epoch 8 through epoch 16. The run restores the encoder, ArcFace head, optimizer and history in runs/hard/last.pt. It selects by validation top-1 across the original best and the new epochs. Results and restart checkpoints go to runs/hard-epochs-9-16/.
- jobs/rerank.id records the frozen epoch 8 evaluation. It tests the mean of the closest 1, 2, 4, or 8 reference views among the top 20 identities. Validation picks the rule and the best rule that uses more than one view. The job reports their synthetic test scores alongside the one-view baseline in runs/hard-rerank/metrics.json. The gameplay section compares per-frame and pooled scores with the full gallery and with the champion supplied. The Aatrox track has no verified skin label.
