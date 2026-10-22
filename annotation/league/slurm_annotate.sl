#!/bin/bash
# Submit from the repository root. Override resource directives with sbatch flags.
# Longleaf GPU examples: https://help.rc.unc.edu/longleaf-slurm-examples/
#SBATCH --job-name=league_annotation
# Keep Slurm logs beside the repository so direct `sbatch` submission does not
# depend on a pre-created logs/ directory.
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=12:00:00
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --signal=B:TERM@120

set -euo pipefail
module purge
module add anaconda/2024.02

LEAGUE_ENV="${LEAGUE_ENV:-/work/users/a/l/alshen/conda_envs/videogame-league}"
CONDA_BASE="$(conda info --base)"
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${LEAGUE_ENV}"
set -u
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:-/work/users/a/l/alshen/hf-cache}"
LEAGUE_RUN_DIR="${LEAGUE_RUN_DIR:-data/league_runs/aatrox-26.13}"
LEAGUE_VIDEO="${LEAGUE_VIDEO:-data/youtube/lol_domisum_aatrox/xICbe-OOPvE.mp4}"
LEAGUE_STAGE="${LEAGUE_STAGE:-run}"

args=(--run-dir "${LEAGUE_RUN_DIR}" --offline)
if [[ -n "${LEAGUE_CONFIG:-}" ]]; then args+=(--config "${LEAGUE_CONFIG}"); fi
args+=("${LEAGUE_STAGE}")
if [[ "${LEAGUE_STAGE}" == run || "${LEAGUE_STAGE}" == prepare ]]; then
    args+=(--video "${LEAGUE_VIDEO}" --start "${LEAGUE_START:-0}")
    if [[ -n "${LEAGUE_END:-}" ]]; then args+=(--end "${LEAGUE_END}"); fi
fi
if [[ "${LEAGUE_RETRY_FAILED:-0}" == 1 ]]; then
    if [[ "${LEAGUE_STAGE}" == run || "${LEAGUE_STAGE}" == annotate ]]; then args+=(--retry-failed); fi
fi

# Signal the Python process directly so it can finish and commit its current task.
child=""
stop_child() { if [[ -n "${child}" ]]; then kill -TERM "${child}" 2>/dev/null || true; fi; }
trap stop_child TERM INT
python -m annotation.league "${args[@]}" &
child=$!
status=0
wait "${child}" || status=$?
if kill -0 "${child}" 2>/dev/null; then wait "${child}" || status=$?; fi
exit "${status}"
