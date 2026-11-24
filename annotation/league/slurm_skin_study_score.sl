#!/bin/bash
#SBATCH --job-name=league_skin_score
#SBATCH --output=/work/users/a/l/alshen/%x-%j.out
#SBATCH --error=/work/users/a/l/alshen/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24g
#SBATCH --time=02:00:00
#SBATCH --partition=general

source_path="${SLURM_SUBMIT_DIR:?Submit from the prepared SAM skin study directory}/source"
source "$source_path/annotation/league/sam_skin_study_longleaf.sh"
exec srun "$study_base/env-vl/bin/python" -B -u -m annotation.league.sam_skin_study \
    score --siglip-lock "$study_root/siglip/model-lock.json" --require-complete \
    --root "$study_root" --cache "$study_base/assets" "$@"
