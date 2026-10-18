#!/bin/bash

# Submit from the videogame project directory on Longleaf:
#   sbatch ingestion/youtube/slurm_download_league.sl
# Run a small trial:
#   sbatch --export=ALL,MAX_DOWNLOADS=1 ingestion/youtube/slurm_download_league.sl
# Catalog only, without downloading media:
#   sbatch --export=ALL,MODE=catalog ingestion/youtube/slurm_download_league.sl
# Use catalogs already saved on the cluster:
#   sbatch --export=ALL,MODE=use-catalog ingestion/youtube/slurm_download_league.sl
#
# Re-submit after each 12-hour allocation to resume. Completed files are skipped
# and yt-dlp continues partial downloads. YouTube verification/rate-limit
# responses stop the downloader; this job does not attempt to bypass them.
#SBATCH --job-name=league_youtube
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

#SBATCH --time=12:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4g
#SBATCH --partition=general

# Signal the batch shell shortly before wall-time expiry.
#SBATCH --signal=B:TERM@120
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=alshen@unc.edu

set -uo pipefail

module purge
module add anaconda/2024.02

# Under sbatch, use the project directory from which the job was submitted.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${REPO_ROOT}"

export PYTHON_BIN="${PYTHON_BIN:-python3}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/data/youtube}"
export MODE="${MODE:-download}"
export MAX_HEIGHT="${MAX_HEIGHT:-1080}"
export PYTHONUNBUFFERED=1

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "ERROR: Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi
if ! "${PYTHON_BIN}" -c 'import yt_dlp' >/dev/null 2>&1; then
    echo "ERROR: yt-dlp is not installed for ${PYTHON_BIN}. Install ingestion/youtube/requirements.txt in this Python environment." >&2
    exit 1
fi
if [[ "${MODE}" != "catalog" ]]; then
    if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
        echo "ERROR: ffmpeg and ffprobe must be available on PATH for media downloads." >&2
        exit 1
    fi
    if ! command -v node >/dev/null 2>&1 && ! command -v deno >/dev/null 2>&1; then
        echo "ERROR: yt-dlp YouTube extraction requires Node.js or Deno on PATH." >&2
        exit 1
    fi
fi
if [[ ! "${MAX_HEIGHT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: MAX_HEIGHT must be a positive integer." >&2
    exit 2
fi
if [[ -n "${MAX_DOWNLOADS:-}" && ! "${MAX_DOWNLOADS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: MAX_DOWNLOADS must be a positive integer when set." >&2
    exit 2
fi

mkdir -p "${REPO_ROOT}/logs" "${OUTPUT_ROOT}"

echo "============================================"
echo " Job ${SLURM_JOB_ID:-local} on $(hostname)"
echo " Started     : $(date)"
echo " REPO_ROOT   : ${REPO_ROOT}"
echo " OUTPUT_ROOT : ${OUTPUT_ROOT}"
echo " MODE        : ${MODE}"
echo " MAX_HEIGHT  : ${MAX_HEIGHT}"
echo " PYTHON_BIN  : ${PYTHON_BIN}"
if [[ -n "${MAX_DOWNLOADS:-}" ]]; then
    echo " MAX_DOWNLOADS: ${MAX_DOWNLOADS}"
fi
echo "============================================"
"${PYTHON_BIN}" --version
ffmpeg -version 2>/dev/null | head -1 || true
echo ""

args=(--output-root "${OUTPUT_ROOT}" --max-height "${MAX_HEIGHT}")
case "${MODE}" in
    download)
        ;;
    catalog)
        args+=(--catalog-only)
        ;;
    use-catalog)
        args+=(--use-catalog)
        ;;
    *)
        echo "ERROR: MODE must be download, catalog, or use-catalog." >&2
        exit 2
        ;;
esac
if [[ -n "${MAX_DOWNLOADS:-}" ]]; then
    args+=(--max-downloads "${MAX_DOWNLOADS}")
fi

shutdown_notice() {
    echo "Shutdown requested at $(date); completed videos and partial downloads can be resumed by re-submitting."
}
trap shutdown_notice TERM INT

echo "[$(date)] Starting League yt-dlp job"
status=0
srun --ntasks=1 --cpus-per-task=2 \
    "${PYTHON_BIN}" -m ingestion.youtube.download_league "${args[@]}" || status=$?

if [[ "${status}" -ne 0 ]]; then
    echo "[$(date)] League downloader exited with status ${status}; catalogs, completed videos, and partial downloads are preserved for retry." >&2
else
    echo "[$(date)] League downloader completed successfully."
fi

echo "============================================"
echo " Job stopped : $(date)"
echo " Output      : ${OUTPUT_ROOT}"
echo "============================================"
exit "${status}"
