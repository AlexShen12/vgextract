#!/bin/bash
#SBATCH --job-name=league_sam_npc_boxes
#SBATCH --output=/work/users/a/l/alshen/%x-%j.out
#SBATCH --error=/work/users/a/l/alshen/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=00:12:00
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:nvidia_l40s:1

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the staged NPC trial directory}"
sha256sum -c INPUT_SHA256SUMS >/dev/null
base=/work/users/a/l/alshen/league-midbeast-validation
export PYTHONPATH="$base/artifacts/sam-tracking-20261110-r3/source"
export LD_LIBRARY_PATH="/work/users/a/l/alshen/conda_envs/videogame-league/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
srun "$base/env-tracking/bin/python" -u -B annotation/league/sam_npc_boxes.py \
    --manifest manifest.json --inputs inputs --output "results/${SLURM_JOB_ID}" --cache "$base/assets"
