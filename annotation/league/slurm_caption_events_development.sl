#!/bin/bash
# Submit from the staged repository root. Development wrappers accept preflight|development.
#SBATCH --job-name=league_caption_events_development
#SBATCH --output=/work/users/a/l/alshen/%x-%j.out
#SBATCH --error=/work/users/a/l/alshen/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --time=02:00:00
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --signal=B:TERM@120

set -euo pipefail
exec bash "${SLURM_SUBMIT_DIR:?Submit from the staged repository root}/annotation/league/caption_longleaf.sh" event-v2 "${1:-preflight}"
