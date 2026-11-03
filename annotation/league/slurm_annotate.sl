#!/bin/bash
# Submit from the repository root. Override resource directives with sbatch flags.
# Longleaf GPU examples: https://help.rc.unc.edu/longleaf-slurm-examples/
#SBATCH --job-name=league_annotation
# Direct sbatch uses the existing work root; the submission helper uses league_logs/.
#SBATCH --output=/work/users/a/l/alshen/%x-%j.out
#SBATCH --error=/work/users/a/l/alshen/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=12:00:00
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --signal=B:TERM@120

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
source annotation/league/longleaf_paths.sh
module purge
module add anaconda/2024.02
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1

LEAGUE_ENV="${LEAGUE_ENV:-${LEAGUE_WORK_ROOT}/conda_envs/videogame-league}"
CONDA_BASE="$(conda info --base)"
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${LEAGUE_ENV}"
set -u
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
LEAGUE_PYTHON="${LEAGUE_ENV}/bin/python"
# Keep Conda ICU/SQLite and PyTorch on a compatible C++ runtime.
export LD_LIBRARY_PATH="${LEAGUE_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
"${LEAGUE_PYTHON}" annotation/league/check_environment.py "${LEAGUE_ENV}"

for executable in ffmpeg ffprobe; do
    command -v "${executable}" >/dev/null 2>&1 || {
        echo "Required executable not found in the active environment: ${executable}" >&2
        echo "Run annotation/league/setup_longleaf.sh to install the media tools." >&2
        exit 2
    }
done

export LEAGUE_VL_PYTHON="${LEAGUE_VL_PYTHON:-${LEAGUE_ENV}-vl/bin/python}"
export LEAGUE_AUDIO_PYTHON="${LEAGUE_AUDIO_PYTHON:-${LEAGUE_ENV}-audio/bin/python}"
export LEAGUE_TRACKING_PYTHON="${LEAGUE_TRACKING_PYTHON:-${LEAGUE_ENV}-tracking/bin/python}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
LEAGUE_RUN_DIR="${LEAGUE_RUN_DIR:-${LEAGUE_WORK_ROOT}/league_runs/midbeast-OLLZBe5-fg4-v2}"
LEAGUE_VIDEO="${LEAGUE_VIDEO:-data/youtube/lol_midbeast/OLLZBe5-fg4.mp4}"
LEAGUE_STAGE="${LEAGUE_STAGE:-run}"

args=(--run-dir "${LEAGUE_RUN_DIR}" --offline)
if [[ -n "${LEAGUE_CONFIG:-}" ]]; then args+=(--config "${LEAGUE_CONFIG}"); fi
args+=("${LEAGUE_STAGE}")
if [[ "${LEAGUE_STAGE}" == run || "${LEAGUE_STAGE}" == prepare ]]; then
    args+=(--video "${LEAGUE_VIDEO}" --start "${LEAGUE_START:-0}")
    if [[ -n "${LEAGUE_END:-}" ]]; then args+=(--end "${LEAGUE_END}"); fi
fi
if [[ "${LEAGUE_RETRY_FAILED:-0}" == 1 ]]; then
    if [[ "${LEAGUE_STAGE}" == run || "${LEAGUE_STAGE}" == annotate ]]; then args+=(--retry-failed); fi
fi

if [[ "${LEAGUE_STAGE}" == annotate ]]; then
    if [[ -n "${LEAGUE_SAMPLE_TIMES:-}" ]]; then args+=(--sample-times "${LEAGUE_SAMPLE_TIMES}"); fi
    if [[ "${LEAGUE_PILOT:-0}" == 1 ]]; then args+=(--pilot); fi
fi

if [[ "${LEAGUE_STAGE}" == experiment ]]; then
    args+=(--suite "${LEAGUE_SUITE:-all}")
    if [[ -n "${LEAGUE_LABELS:-}" ]]; then args+=(--labels "${LEAGUE_LABELS}"); fi
    if [[ "${LEAGUE_DRAFT:-0}" == 1 ]]; then args+=(--draft); fi
    if [[ "${LEAGUE_FREEZE:-0}" == 1 ]]; then args+=(--freeze); fi
    if [[ "${LEAGUE_INITIALIZE:-0}" == 1 ]]; then args+=(--initialize); fi
    if [[ ( "${LEAGUE_SUITE:-all}" == actions || "${LEAGUE_SUITE:-all}" == captions ) && "${LEAGUE_ACTION_INFER:-0}" == 1 ]]; then args+=(--infer); fi
fi

if [[ "${LEAGUE_STAGE}" == action-window ]]; then
    args+=(--start "${LEAGUE_START:-30}" --end "${LEAGUE_END:-38}" --fps "${LEAGUE_ACTION_FPS:-4}")
    if [[ "${LEAGUE_ACTION_INFER:-0}" == 1 ]]; then args+=(--infer); fi
    if [[ "${LEAGUE_ACTION_DETECT:-0}" == 1 ]]; then args+=(--detect); fi
fi

if [[ "${LEAGUE_STAGE}" == caption-pilot ]]; then
    args+=(--start "${LEAGUE_START:-30}" --end "${LEAGUE_END:-60}")
    if [[ "${LEAGUE_ACTION_INFER:-0}" == 1 ]]; then args+=(--infer); fi
fi

if [[ "${LEAGUE_STAGE}" == action-window || "${LEAGUE_STAGE}" == caption-pilot || "${LEAGUE_STAGE}" == experiment ]]; then
    if [[ -n "${LEAGUE_ACTION_PROFILE:-}" ]]; then args+=(--action-profile "${LEAGUE_ACTION_PROFILE}"); fi
    if [[ -n "${LEAGUE_ICON_REFERENCES:-}" ]]; then args+=(--icon-references "${LEAGUE_ICON_REFERENCES}"); fi
    if [[ -n "${LEAGUE_ASSISTANCE:-}" && "${LEAGUE_STAGE}" != caption-pilot ]]; then args+=(--assistance "${LEAGUE_ASSISTANCE}"); fi
fi

# Request the confirmed L40S type explicitly. Preflight still records the actual
# name, UUID, capacity and runtime; never label L40 timings as L40S.
# Signal the Python process directly so it can finish and commit its current task.
child=""
stop_child() { if [[ -n "${child}" ]]; then kill -TERM "${child}" 2>/dev/null || true; fi; }
trap stop_child TERM INT
"${LEAGUE_PYTHON}" -m annotation.league "${args[@]}" &
child=$!
status=0
wait "${child}" || status=$?
if kill -0 "${child}" 2>/dev/null; then wait "${child}" || status=$?; fi
exit "${status}"
