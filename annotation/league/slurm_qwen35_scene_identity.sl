#!/bin/bash
# Submit from the repository root. Argument: [draft|preflight|evaluate].
#SBATCH --job-name=league_qwen35_scene_identity
#SBATCH --output=/work/users/a/l/alshen/%x-%j.out
#SBATCH --error=/work/users/a/l/alshen/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --time=12:00:00
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --signal=B:TERM@120
#SBATCH --mail-user=alshen@unc.edu
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT

set -euo pipefail
exec bash "${SLURM_SUBMIT_DIR:?Submit from the repository root}/annotation/league/slurm_qwen35.sl" scene-identity "$@"
