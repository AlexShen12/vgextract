#!/bin/bash
#SBATCH --job-name=league_skin_rerank
#SBATCH --output=/work/users/a/l/alshen/league-skin-metric/16.22-v1/logs/%x-%j.out
#SBATCH --error=/work/users/a/l/alshen/league-skin-metric/16.22-v1/logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1

set -euo pipefail
root=/work/users/a/l/alshen/league-skin-metric/16.22-v1
exec bash "${SLURM_SUBMIT_DIR:?}/annotation/league/skin_metric_longleaf.sh" rerank \
    --source "$root/runs/hard" --output "$root/runs/hard-rerank" \
    --track "${SLURM_SUBMIT_DIR:?}/gameplay-track"
