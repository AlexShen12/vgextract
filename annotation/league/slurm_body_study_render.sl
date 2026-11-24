#!/bin/bash
#SBATCH --job-name=league_body_static_candidates
#SBATCH --output=/work/users/a/l/alshen/%x-%j.out
#SBATCH --error=/work/users/a/l/alshen/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=00:30:00
#SBATCH --partition=general
set -euo pipefail
exec bash "${SLURM_SUBMIT_DIR:?}/annotation/league/body_study_longleaf.sh" render
