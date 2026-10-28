#!/bin/bash
# Online setup on a login node. GPU preflight runs only inside an allocation.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
source "${SCRIPT_DIR}/longleaf_paths.sh"
LEAGUE_ENV="${LEAGUE_ENV:-${LEAGUE_WORK_ROOT}/conda_envs/videogame-league}"
LEAGUE_RUN_DIR="${LEAGUE_RUN_DIR:-${LEAGUE_WORK_ROOT}/league_runs/midbeast-OLLZBe5-fg4-v2}"
LEAGUE_VIDEO="${LEAGUE_VIDEO:-${REPO_ROOT}/data/youtube/lol_midbeast/OLLZBe5-fg4.mp4}"
export LEAGUE_WITH_EVALUATION="${LEAGUE_WITH_EVALUATION:-0}"
export LEAGUE_VL_PYTHON="${LEAGUE_VL_PYTHON:-${LEAGUE_ENV}-vl/bin/python}"
export LEAGUE_AUDIO_PYTHON="${LEAGUE_AUDIO_PYTHON:-${LEAGUE_ENV}-audio/bin/python}"
export LEAGUE_TRACKING_PYTHON="${LEAGUE_TRACKING_PYTHON:-${LEAGUE_ENV}-tracking/bin/python}"
mkdir -p "${HF_HOME}"
[[ -f "${LEAGUE_VIDEO}" ]] || { echo "Transfer the pinned Midbeast video and set LEAGUE_VIDEO: ${LEAGUE_VIDEO}" >&2; exit 2; }
command -v module >/dev/null 2>&1 || { echo "Run in a Longleaf login shell." >&2; exit 2; }
module add anaconda/2024.02
unset PYTHONHOME PYTHONPATH HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export PYTHONNOUSERSITE=1
CONDA_BASE="$(conda info --base)"
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
set -u
if [[ -x "${LEAGUE_ENV}/bin/python" ]]; then action=install; else action=create; fi
conda "${action}" -p "${LEAGUE_ENV}" -c conda-forge python=3.12 pip ffmpeg tesseract -y
set +u
conda activate "${LEAGUE_ENV}"
set -u
unset PYTHONHOME PYTHONPATH
LEAGUE_PYTHON="${LEAGUE_ENV}/bin/python"
# Keep Conda ICU/SQLite and PyTorch on a compatible C++ runtime.
export LD_LIBRARY_PATH="${LEAGUE_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
"${LEAGUE_PYTHON}" annotation/league/check_environment.py "${LEAGUE_ENV}" --writable
LEAGUE_LOCK=annotation/league/requirements-intake.txt
if [[ "${LEAGUE_WITH_EVALUATION}" == 1 ]]; then LEAGUE_LOCK=annotation/league/inference-environment.lock.txt; fi
"${LEAGUE_PYTHON}" -m pip install --no-user -r "${LEAGUE_LOCK}"
"${LEAGUE_PYTHON}" -m pip check

install_worker() {
    local interpreter="$1" requirement="$2" prefix
    prefix="$(dirname "$(dirname "${interpreter}")")"
    if [[ ! -x "${interpreter}" ]]; then "${LEAGUE_PYTHON}" -m venv "${prefix}"; fi
    "${interpreter}" annotation/league/check_environment.py "${prefix}" --writable
    "${interpreter}" -m pip install --no-user -r "${requirement}"
    "${interpreter}" -m pip check
    # Preserve the resolved Linux dependency closure alongside each environment.
    "${interpreter}" -m pip freeze --all > "${prefix}/requirements.resolved.txt"
}
install_worker "${LEAGUE_VL_PYTHON}" annotation/league/requirements-vl.txt
cache_args=(--suite intake)
if [[ "${LEAGUE_WITH_EXPERIMENTS:-0}" == 1 ]]; then
    install_worker "${LEAGUE_AUDIO_PYTHON}" annotation/league/requirements-audio.txt
    cache_args=(--suite all)
fi
if [[ "${LEAGUE_WITH_TRACKING:-0}" == 1 ]]; then
    install_worker "${LEAGUE_TRACKING_PYTHON}" annotation/league/requirements-tracking.txt
    cache_args+=(--tracking)
fi
if [[ "${LEAGUE_WITH_EVALUATION}" == 1 ]]; then cache_args+=(--solver); fi
for executable in ffmpeg ffprobe tesseract; do
    command -v "${executable}" >/dev/null 2>&1 || { echo "Missing ${executable}" >&2; exit 2; }
done
evaluation_args=(--no-evaluation)
if [[ "${LEAGUE_WITH_EVALUATION}" == 1 ]]; then evaluation_args=(--evaluation); fi
"${LEAGUE_PYTHON}" -m annotation.league --run-dir "${LEAGUE_RUN_DIR}" "${evaluation_args[@]}" prepare --video "${LEAGUE_VIDEO}"
"${LEAGUE_PYTHON}" -m annotation.league --run-dir "${LEAGUE_RUN_DIR}" cache-models "${cache_args[@]}"
"${LEAGUE_PYTHON}" -m annotation.league --run-dir "${LEAGUE_RUN_DIR}" processor-preview
cat <<EOF
Setup completed. Numeric OCR and bounded Qwen transcription use separate environments.
No Paddle worker or icon assets are required. Models are pinned in models.local.json.
The vLLM CUDA build and actual GPU must still pass allocation preflight:
  LEAGUE_STAGE=preflight bash annotation/league/submit_longleaf.sh
Create independent label templates:
  ${LEAGUE_PYTHON} -m annotation.league --run-dir ${LEAGUE_RUN_DIR} experiment --suite all --initialize
After manual labels: extract benchmark frames, freeze development rules, score held-out,
then submit LEAGUE_STAGE=annotate,LEAGUE_PILOT=1. Full intake stays blocked until both pass.
Event/audio/tracking experiments are separate and never establish production game facts.
Use LEAGUE_WITH_EXPERIMENTS=1 or LEAGUE_WITH_TRACKING=1 to install optional workers.
EOF
