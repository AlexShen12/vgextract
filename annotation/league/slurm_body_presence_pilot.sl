#!/bin/bash
# Submit from the frozen source snapshot. Pass the scan inputs as arguments.
#SBATCH --job-name=league_body_presence_pilot
#SBATCH --output=/work/users/a/l/alshen/%x-%j.out
#SBATCH --error=/work/users/a/l/alshen/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=00:45:00
#SBATCH --partition=general

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the frozen source snapshot}"
sha256sum --quiet -c SOURCE_SHA256SUMS
base=/work/users/a/l/alshen/league-midbeast-validation
export LEAGUE_WORK_ROOT=/work/users/a/l/alshen
export PATH="/work/users/a/l/alshen/conda_envs/videogame-league/bin:${PATH}"
export LD_LIBRARY_PATH="/work/users/a/l/alshen/conda_envs/videogame-league/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" OPENBLAS_NUM_THREADS=1
export XFORMERS_DISABLED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset PYTHONHOME PYTHONPATH

exec srun "$base/env-body/bin/python" -u -B -m annotation.league.body_presence scan "$@"
