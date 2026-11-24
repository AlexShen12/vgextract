#!/bin/bash
#SBATCH --job-name=league_scene_holdout_refs
#SBATCH --output=/work/users/a/l/alshen/%x-%j.out
#SBATCH --error=/work/users/a/l/alshen/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8g
#SBATCH --time=00:15:00
#SBATCH --partition=general
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the pinned input-comparison checkout}"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
driver=/work/users/a/l/alshen/league-midbeast-validation/env-driver/bin/python
scene_pilot=/work/users/a/l/alshen/league-midbeast-validation/scene-understanding/v2-pilot
srun --ntasks=1 --cpus-per-task=2 "$driver" -B -m annotation.league.scene_split \
  --prepare-reference reference_sets/league-scene-understanding/v2-pilot/reservation.json \
  --source /work/users/a/l/alshen/league-midbeast-validation/source-20261028/data/youtube/lol_midbeast/OLLZBe5-fg4.mp4 \
  --output "$scene_pilot/references/inputs"
