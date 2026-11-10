#!/bin/bash
#SBATCH --job-name=league_scene_dev_refs
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
mkdir -p "$scene_pilot/reference-metadata"
cp -n reference_sets/league-scene-understanding/v2-pilot/development-candidates.json "$scene_pilot/reference-metadata/development-candidates.json"
srun --ntasks=1 --cpus-per-task=2 "$driver" -B -m annotation.league.scene_reference prepare \
  --manifest "$scene_pilot/reference-metadata/development-candidates.json" \
  --output "$scene_pilot/references/development-inputs" --fps 16
