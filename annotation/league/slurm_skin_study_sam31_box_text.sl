#!/bin/bash
#SBATCH --job-name=league_skin_sam31_box_text
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
use_pinned_sam sam3 f1647e8d0c1d280991f4d1b7d974c2704537fe66
exec srun "$study_base/env-tracking/bin/python" -B -u -m annotation.league.sam_skin_study \
    track --arm sam31_box_text \
    --root "$study_root" --cache "$study_base/assets" "$@"
