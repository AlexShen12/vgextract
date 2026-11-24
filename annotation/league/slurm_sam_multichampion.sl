#!/bin/bash
#SBATCH --job-name=league_sam_multichampion
#SBATCH --output=/work/users/a/l/alshen/%x-%j.out
#SBATCH --error=/work/users/a/l/alshen/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=00:08:00
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:nvidia_l40s:1

set -euo pipefail
experiment_root="${SLURM_SUBMIT_DIR:?Submit from the prepared experiment directory}"
base=/work/users/a/l/alshen/league-midbeast-validation
[[ -f "$experiment_root/experiment.json" ]]
[[ -d "$experiment_root/inputs" ]]
[[ -f "$experiment_root/source/annotation/league/sam_multichampion.py" ]]
cd "$experiment_root"
sha256sum -c INPUT_SHA256SUMS --quiet
cd "$experiment_root/source"
sha256sum -c SOURCE_SHA256SUMS --quiet
cd "$experiment_root"
mkdir -p "$experiment_root/results"
export PYTHONPATH="$experiment_root/source:$base/artifacts/sam-tracking-20261110-r3/source"
export LD_LIBRARY_PATH="/work/users/a/l/alshen/conda_envs/videogame-league/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
srun "$base/env-tracking/bin/python" -u -B -m annotation.league.sam_multichampion run \
    --experiment "$experiment_root/experiment.json" \
    --output "$experiment_root/results/${SLURM_JOB_ID}" \
    --cache "$base/assets"
