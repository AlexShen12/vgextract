#!/bin/bash
# Annotation fallback only; the fine-tuning and comparison jobs retain the L40S.
#SBATCH --job-name=league_sam3_ft_cpu_propose
#SBATCH --output=/work/users/a/l/alshen/league-sam3-finetune-v1/logs/%x-%j.out
#SBATCH --error=/work/users/a/l/alshen/league-sam3-finetune-v1/logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=00:45:00
#SBATCH --partition=general
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source annotation/league/longleaf_paths.sh
export LD_LIBRARY_PATH="$LEAGUE_WORK_ROOT/conda_envs/videogame-league/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS=1
exec srun "$LEAGUE_WORK_ROOT/league-sam3-finetune-v1/env/bin/python" -B -u \
    -m annotation.league.sam_finetune_cpu_proposals run \
    --root "$LEAGUE_WORK_ROOT/league-sam3-finetune-v1/data" "$@"
