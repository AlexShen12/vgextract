#!/bin/bash
#SBATCH --job-name=league_mask_qwen_27b_fp8
#SBATCH --output=/work/users/a/l/alshen/%x-%j.out
#SBATCH --error=/work/users/a/l/alshen/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=00:15:00
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --signal=TERM@60

set -euo pipefail
root="${SLURM_SUBMIT_DIR:?Submit from the staged mask verifier directory}"
cd "$root/source"
sha256sum --quiet -c SOURCE_SHA256SUMS
source annotation/league/longleaf_paths.sh
base=/work/users/a/l/alshen/league-midbeast-validation
export LD_LIBRARY_PATH="$LEAGUE_WORK_ROOT/conda_envs/videogame-league/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" OPENBLAS_NUM_THREADS=1
exec srun "$base/env-driver/bin/python" -B -u -m annotation.league.mask_verifier infer \
    --root "$root" --role intake --cache "$base/assets" --vl-python "$base/env-vl/bin/python"
