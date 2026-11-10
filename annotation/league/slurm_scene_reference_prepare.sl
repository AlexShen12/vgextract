#!/bin/bash
#SBATCH --job-name=league_scene_refs
#SBATCH --output=/work/users/a/l/alshen/league_scene_refs-%j.out
#SBATCH --error=/work/users/a/l/alshen/league_scene_refs-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8g
#SBATCH --time=00:15:00
#SBATCH --partition=general

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the isolated scene-reference Git checkout}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=tests:.

driver=/work/users/a/l/alshen/league-midbeast-validation/env-driver/bin/python
scene_run=/work/users/a/l/alshen/league-midbeast-validation/scene-understanding/v1
scene_source=/work/users/a/l/alshen/league-midbeast-validation/source-20261028/data/youtube/lol_midbeast/OLLZBe5-fg4.mp4
mkdir -p "$scene_run"
git diff --quiet
git diff --cached --quiet
git rev-parse HEAD > "$scene_run/code-commit.txt"

srun --ntasks=1 --cpus-per-task=2 "$driver" -B -m unittest \
  tests.test_league_scene_reference tests.test_league_scene_reference_campaign \
  tests.test_league_scene_source_regressions tests.test_league_adaptive_actions \
  tests.test_league_player_actions -q 2>&1 | tee "$scene_run/test-results.log"

srun --ntasks=1 --cpus-per-task=2 "$driver" -B -m annotation.league.scene_reference_campaign \
  --references reference_sets/league-scene-understanding/v1 \
  --run-dir "$scene_run" --source "$scene_source"
