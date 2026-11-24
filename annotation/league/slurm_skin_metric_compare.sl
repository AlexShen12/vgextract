#!/bin/bash
#SBATCH --job-name=league_skin_compare
#SBATCH --output=/work/users/a/l/alshen/league-skin-metric/16.22-v1/logs/%x-%j.out
#SBATCH --error=/work/users/a/l/alshen/league-skin-metric/16.22-v1/logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --partition=general

set -euo pipefail
exec bash "${SLURM_SUBMIT_DIR:?}/annotation/league/skin_metric_longleaf.sh" compare
