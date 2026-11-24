#!/bin/bash
#SBATCH --job-name=league_sam_appearances
#SBATCH --output=/work/users/a/l/alshen/%x-%A-%a.out
#SBATCH --error=/work/users/a/l/alshen/%x-%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=00:30:00
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:nvidia_l40s:1

set -euo pipefail
root="${SLURM_SUBMIT_DIR:?Submit from the staged appearance run directory}"
base=/work/users/a/l/alshen/league-midbeast-validation
cd "$root/source"
sha256sum --quiet -c SOURCE_SHA256SUMS
cd "$root"
export PYTHONPATH="$root/source:$base/artifacts/sam-tracking-20261110-r3/source"
export LD_LIBRARY_PATH="/work/users/a/l/alshen/conda_envs/videogame-league/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
task="${SLURM_ARRAY_TASK_ID:?Use array index 0 for smoke, 1 through 6 for the full run}"
args=(--smoke)
if (( task > 0 )); then
    args=(--require-smoke --shard "$((task - 1))" --shards 6)
fi
srun "$base/env-tracking/bin/python" -B -u -m annotation.league.sam_multichampion mask-appearances \
    --root "$root" --cache "$base/assets" "${args[@]}"
