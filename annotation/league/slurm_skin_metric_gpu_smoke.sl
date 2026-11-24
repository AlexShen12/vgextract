#!/bin/bash
#SBATCH --job-name=league_skin_gpu_smoke
#SBATCH --output=/work/users/a/l/alshen/league-skin-metric/16.22-v1/logs/%x-%j.out
#SBATCH --error=/work/users/a/l/alshen/league-skin-metric/16.22-v1/logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --signal=TERM@120

set -euo pipefail
exec bash "${SLURM_SUBMIT_DIR:?}/annotation/league/skin_metric_longleaf.sh" train-pilot
