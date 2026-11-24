#!/bin/bash
#SBATCH --job-name=league_body_preflight
#SBATCH --output=/work/users/a/l/alshen/%x-%j.out
#SBATCH --error=/work/users/a/l/alshen/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=00:30:00
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:nvidia_l40s:1
set -euo pipefail
exec bash "${SLURM_SUBMIT_DIR:?}/annotation/league/body_longleaf.sh" preflight
