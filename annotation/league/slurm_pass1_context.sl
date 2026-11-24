#!/bin/bash
# Rebuild shared playback and selection from the completed dense scan and clock cache.
#SBATCH --job-name=league_pass1_context
#SBATCH --output=/work/users/a/l/alshen/%x-%j.out
#SBATCH --error=/work/users/a/l/alshen/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8g
#SBATCH --time=12:00:00
#SBATCH --partition=general
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the verified Pass 1 checkout}"
base=/work/users/a/l/alshen/league-midbeast-validation
export LEAGUE_WORK_ROOT=/work/users/a/l/alshen
source annotation/league/longleaf_paths.sh
export PATH="/work/users/a/l/alshen/conda_envs/videogame-league/bin:$PATH"
export LEAGUE_TESSERACT=/usr/bin/tesseract
export OMP_THREAD_LIMIT=1 OPENBLAS_NUM_THREADS=1
run="${LEAGUE_PASS1_RUN:-$PWD/run}"
srun "$base/env-driver/bin/python" -Bu -m annotation.league.pass1 run \
    --run-dir "$run" --component context --budget-approved
