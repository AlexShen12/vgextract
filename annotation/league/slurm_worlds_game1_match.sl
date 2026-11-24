#!/bin/bash
#SBATCH --job-name=league_worlds_game1_match
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8g
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the staged match workspace}"
work_root=/work/users/a/l/alshen
broadcast="$work_root/league-worlds-broadcast"
run="$broadcast/results-10248305"
driver="$work_root/league-midbeast-validation/env-driver/bin/python"
media="$work_root/conda_envs/videogame-league"

export PATH="$media/bin:$PATH"
export LD_LIBRARY_PATH="$media/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$broadcast/deps"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LEAGUE_TESSERACT=/usr/bin/tesseract OMP_THREAD_LIMIT=1 OPENBLAS_NUM_THREADS=1

srun "$driver" -Bu -m annotation.league.pass1 broadcast-match \
  --run-dir "$run" --match-profile annotation/league/profiles/worlds-2026-spectator-match.json \
  --calibration inputs/calibration --asset-cache "$PWD" --review inputs/game1-match-review.json \
  --workers "${SLURM_CPUS_PER_TASK:-4}"
srun "$driver" -Bu -m annotation.league.pass1 verify --run-dir "$run"
