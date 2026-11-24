#!/bin/bash
# Ten source seconds, including model load and source-alignment measurements.
#SBATCH --job-name=league_pass1_asr_pilot
#SBATCH --output=/work/users/a/l/alshen/%x-%j.out
#SBATCH --error=/work/users/a/l/alshen/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=00:05:00
#SBATCH --partition=rtx6k_gpu
#SBATCH --qos=gpu_rtx6k
#SBATCH --gres=gpu:1
#SBATCH --signal=B:TERM@120
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the staged checkout}"
base=/work/users/a/l/alshen/league-midbeast-validation
export PATH="/work/users/a/l/alshen/conda_envs/videogame-league/bin:$PATH"
export LD_LIBRARY_PATH="/work/users/a/l/alshen/conda_envs/videogame-league/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LEAGUE_AUDIO_PYTHON="$base/env-audio/bin/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DO_NOT_TRACK=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
run="${LEAGUE_PASS1_RUN:-$PWD/run}"
child=""
checkpoint() { if [[ -n "$child" ]]; then kill -TERM "$child" 2>/dev/null || true; fi; }
trap checkpoint TERM INT
srun "$base/env-driver/bin/python" -Bu -m annotation.league.pass1 pilot \
    --run-dir "$run" --component asr --model-cache "$base/assets" &
child=$!
status=0
wait "$child" || status=$?
if kill -0 "$child" 2>/dev/null; then wait "$child" || status=$?; fi
exit "$status"
