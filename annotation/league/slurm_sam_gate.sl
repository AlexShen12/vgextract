#!/bin/bash
#SBATCH --job-name=league_sam_gate
#SBATCH --output=/work/users/a/l/alshen/%x-%j.out
#SBATCH --error=/work/users/a/l/alshen/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2g
#SBATCH --time=00:05:00
#SBATCH --partition=general
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the frozen SAM source snapshot}"
sha256sum -c GATE_SHA256SUMS >/dev/null
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
prepared=$(cat prepared-path.txt)
/work/users/a/l/alshen/league-midbeast-validation/env-driver/bin/python -B \
    -m annotation.league.sam_tracking_gate --prepared "$prepared"
