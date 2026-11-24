#!/bin/bash
# Draft masks for visual correction before the training jobs can run.
#SBATCH --job-name=league_sam3_ft_propose
#SBATCH --output=/work/users/a/l/alshen/league-sam3-finetune-v1/logs/%x-%j.out
#SBATCH --error=/work/users/a/l/alshen/league-sam3-finetune-v1/logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=00:20:00
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:nvidia_l40s:1
set -euo pipefail
exec bash "${SLURM_SUBMIT_DIR:?}/annotation/league/sam_finetune_longleaf.sh" propose "$@"
