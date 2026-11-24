#!/bin/bash
# Submit from the checksum-verified checkout. Default: five-minute pilot.
#SBATCH --job-name=league_pass1_cpu
#SBATCH --output=/work/users/a/l/alshen/%x-%j.out
#SBATCH --error=/work/users/a/l/alshen/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16g
#SBATCH --time=01:00:00
#SBATCH --partition=general
#SBATCH --signal=B:TERM@120
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the staged checkout}"
base=/work/users/a/l/alshen/league-midbeast-validation
export LEAGUE_WORK_ROOT=/work/users/a/l/alshen
source annotation/league/longleaf_paths.sh
export PATH="/work/users/a/l/alshen/conda_envs/videogame-league/bin:$PATH"
export LEAGUE_TESSERACT=/usr/bin/tesseract
export OMP_THREAD_LIMIT=1 OPENBLAS_NUM_THREADS=1
run="${LEAGUE_PASS1_RUN:-$PWD/run}"
mode="${1:-pilot}"
case "$mode" in pilot) extra=();; run) extra=(--budget-approved);; *) exit 2;; esac
child=""
checkpoint() { if [[ -n "$child" ]]; then kill -TERM "$child" 2>/dev/null || true; fi; }
trap checkpoint TERM INT
launch() {
    srun "$@" &
    child=$!
    local status=0
    wait "$child" || status=$?
    if kill -0 "$child" 2>/dev/null; then wait "$child" || status=$?; fi
    return "$status"
}
launch "$base/env-driver/bin/python" -Bu -m annotation.league.pass1 "$mode" --run-dir "$run" --component cpu "${extra[@]}"
launch "$base/env-driver/bin/python" -Bu -m annotation.league.pass1 "$mode" --run-dir "$run" --component dense "${extra[@]}"
