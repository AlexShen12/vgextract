#!/bin/bash
# One L40S, using the existing measured 27B FP8 worker; pilot by default.
#SBATCH --job-name=league_pass1_vl
#SBATCH --output=/work/users/a/l/alshen/%x-%j.out
#SBATCH --error=/work/users/a/l/alshen/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=01:15:00
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --signal=B:TERM@120
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the staged checkout}"
base=/work/users/a/l/alshen/league-midbeast-validation
export LEAGUE_WORK_ROOT=/work/users/a/l/alshen
source annotation/league/longleaf_paths.sh
export PATH="/work/users/a/l/alshen/conda_envs/videogame-league/bin:$PATH"
export LD_LIBRARY_PATH="/work/users/a/l/alshen/conda_envs/videogame-league/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LEAGUE_VL_PYTHON="$base/env-vl/bin/python"
export LEAGUE_TESSERACT=/usr/bin/tesseract
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
run="${LEAGUE_PASS1_RUN:-$PWD/run}"
mode="${1:-pilot}"
case "$mode" in pilot) extra=();; run) extra=(--budget-approved);; *) exit 2;; esac
child=""
checkpoint() { if [[ -n "$child" ]]; then kill -TERM "$child" 2>/dev/null || true; fi; }
trap checkpoint TERM INT
srun "$base/env-driver/bin/python" -Bu -m annotation.league.pass1 "$mode" --run-dir "$run" --component vl --model-cache "$base/assets" "${extra[@]}" &
child=$!
status=0
wait "$child" || status=$?
if kill -0 "$child" 2>/dev/null; then wait "$child" || status=$?; fi
exit "$status"
