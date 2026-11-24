#!/bin/bash
#SBATCH --job-name=league_skin_sam21_points
#SBATCH --output=/work/users/a/l/alshen/%x-%j.out
#SBATCH --error=/work/users/a/l/alshen/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=01:00:00
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --signal=TERM@120

source_path="${SLURM_SUBMIT_DIR:?Submit from the prepared SAM skin study directory}/source"
source "$source_path/annotation/league/sam_skin_study_longleaf.sh"
use_pinned_sam sam2 2b90b9f5ceec907a1c18123530e92e794ad901a4
exec srun "$study_base/env-sam21-skin-study/bin/python" -B -u -m annotation.league.sam_skin_study \
    track --arm sam21_points --sam21-weights "$study_base/assets/sam2.1/sam2.1_hiera_large.pt" \
    --root "$study_root" --cache "$study_base/assets" "$@"
