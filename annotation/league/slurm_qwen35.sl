#!/bin/bash
# Submit from the repository root. Arguments: [suite] [draft|preflight|evaluate].
#SBATCH --job-name=league_qwen35
#SBATCH --output=/work/users/a/l/alshen/%x-%j.out
#SBATCH --error=/work/users/a/l/alshen/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --time=04:00:00
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --signal=B:TERM@120

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
unset PYTHONHOME PYTHONPATH

export LEAGUE_SUITE="${1:-${LEAGUE_SUITE:-qwen35}}"
mode="${2:-${LEAGUE_QWEN35_MODE:-draft}}"
case "${LEAGUE_SUITE}" in
    hp-vlm|scene-identity|ability-alignment|cooldown-identity|qwen35) ;;
    *) echo "Expected hp-vlm, scene-identity, ability-alignment, cooldown-identity or qwen35" >&2; exit 1;;
esac
case "${mode}" in
    draft|preflight|evaluate) ;;
    *) echo "Expected draft, preflight or evaluate" >&2; exit 1;;
esac
exec bash annotation/league/qwen35_longleaf.sh "${mode}"
