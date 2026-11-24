#!/bin/bash
# All modes use the existing HUD profile and one pinned SAM environment.
set -euo pipefail
mode="${1:?Expected preflight or run}"
arm="${2:?Expected verified or automatic}"
cd "${SLURM_SUBMIT_DIR:?Submit from the staged SAM source snapshot}"
if [[ -f SOURCE_SHA256SUMS ]]; then sha256sum -c SOURCE_SHA256SUMS >/dev/null; fi
export LEAGUE_WORK_ROOT="${LEAGUE_WORK_ROOT:-/work/users/a/l/alshen}"
base="${LEAGUE_WORK_ROOT}/league-midbeast-validation"
export LEAGUE_ASSET_CACHE="${base}/assets" LEAGUE_HF_HOME="${base}/hf-home"
source annotation/league/longleaf_paths.sh
unset PYTHONHOME PYTHONPATH
export PATH="${LEAGUE_WORK_ROOT}/conda_envs/videogame-league/bin:${PATH}"
export LD_LIBRARY_PATH="${LEAGUE_WORK_ROOT}/conda_envs/videogame-league/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
for program in ffprobe ffmpeg tesseract; do command -v "$program"; done
prepared=$(cat prepared-path.txt)
[[ -f "$prepared" ]]
hostname
pwd
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"${base}/env-tracking/bin/python" -u -B -m annotation.league.sam_tracking "$mode" \
    --run-dir "${base}/sam-tracking/v1/run" --prepared "$prepared" --arm "$arm" &
child=$!
checkpoint() { kill -TERM "$child" 2>/dev/null || true; }
trap checkpoint TERM INT
status=0
wait "$child" || status=$?
if kill -0 "$child" 2>/dev/null; then wait "$child" || status=$?; fi
exit "$status"
