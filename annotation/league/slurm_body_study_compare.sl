#!/bin/bash
#SBATCH --job-name=league_body_mask_comparison
#SBATCH --output=/work/users/a/l/alshen/%x-%j.out
#SBATCH --error=/work/users/a/l/alshen/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=00:15:00
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:nvidia_l40s:1
set -euo pipefail
exec bash "${SLURM_SUBMIT_DIR:?}/annotation/league/body_study_longleaf.sh" compare
