#!/bin/bash
# Shared environment only. Every experiment has its own explicit .sl entry point.
set -euo pipefail
arm="${1:?Expected baseline or event-v2}"
mode="${2:?Expected preflight, development or held-out}"
case "$arm:$mode" in baseline:preflight|baseline:development|event-v2:preflight|event-v2:development|event-v2:held-out) ;; *) echo 'Invalid caption experiment arm/mode' >&2; exit 1;; esac
cd "${SLURM_SUBMIT_DIR:?Submit from the staged repository root}"
unset PYTHONHOME PYTHONPATH
export LEAGUE_WORK_ROOT="${LEAGUE_WORK_ROOT:-/work/users/a/l/alshen}"
caption_base="${LEAGUE_WORK_ROOT}/league-midbeast-validation"
export LEAGUE_ASSET_CACHE="${LEAGUE_ASSET_CACHE:-${caption_base}/assets}"
export LEAGUE_HF_HOME="${LEAGUE_HF_HOME:-${caption_base}/hf-home}"
source annotation/league/longleaf_paths.sh
export LEAGUE_VL_PYTHON="${caption_base}/env-vl/bin/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
caption_media_env="${LEAGUE_WORK_ROOT}/conda_envs/videogame-league"
export PATH="${caption_media_env}/bin:${PATH}"
export LD_LIBRARY_PATH="${caption_media_env}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
for program in ffprobe ffmpeg tesseract; do
    command -v "$program" || { echo "Required caption tool unavailable: $program" >&2; exit 2; }
done
caption_driver="${caption_base}/env-driver/bin/python"
caption_source_run="${caption_base}/run"
caption_work="${caption_base}/caption-events/v1"
caption_labels="${SLURM_SUBMIT_DIR}/caption-reference-labels/v1"
caption_icons="${caption_base}/assets/ability-icons-16.13.1/manifest.json"
# Source-only labels are staged explicitly; do not silently initialize a different split.
[[ -f "${caption_labels}/events.json" ]] || { echo "Stage ${caption_labels}/events.json before submitting" >&2; exit 1; }
child=""
checkpoint() {
    echo "$(date -u +%FT%TZ) checkpoint requested job=${SLURM_JOB_ID:-none} arm=${arm} mode=${mode}" >&2
    if [[ -n "$child" ]]; then kill -TERM "$child" 2>/dev/null || true; fi
}
trap checkpoint TERM INT
# Pin the complete code snapshot in a per-job archive before launching inference.
caption_archive="${caption_work}/source-jobs"
mkdir -p "$caption_archive"
caption_job="${SLURM_JOB_ID:?Launch this wrapper with its experiment-specific sbatch file}"
caption_snapshot="${caption_archive}/${arm}-${mode}-${caption_job}-source"
mkdir "$caption_snapshot"
tar --exclude='__pycache__' --exclude='*.pyc' -czf "${caption_archive}/${arm}-${mode}-${caption_job}.tar.gz" \
    annotation/__init__.py annotation/league ingestion/__init__.py \
    ingestion/common/__init__.py ingestion/common/replay_state.py
tar -xzf "${caption_archive}/${arm}-${mode}-${caption_job}.tar.gz" -C "$caption_snapshot"
cd "$caption_snapshot"
mkdir -p caption-reference-labels
cp -R "$caption_labels" caption-reference-labels/v1
caption_labels="${caption_snapshot}/caption-reference-labels/v1"
hostname
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv
echo "$(date -u +%FT%TZ) caption arm=${arm} mode=${mode} work=${caption_work} labels=${caption_labels}"
"$caption_driver" -u -B -m annotation.league.caption_batch "$mode" --arm "$arm" \
  --source-run "$caption_source_run" --work-root "$caption_work" --labels "$caption_labels" \
  --icon-references "$caption_icons" 2>&1 &
child=$!
status=0
wait "$child" || status=$?
if kill -0 "$child" 2>/dev/null; then wait "$child" || status=$?; fi
echo "$(date -u +%FT%TZ) caption exit=${status} job=${SLURM_JOB_ID:-none} arm=${arm} mode=${mode}"
exit "$status"
