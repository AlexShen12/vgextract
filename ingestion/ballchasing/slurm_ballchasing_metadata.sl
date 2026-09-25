#!/bin/bash

# Submit from the videogame project directory with:
#   sbatch ingestion/ballchasing/slurm_ballchasing_metadata.sl
# Override the request budget, for example:
#   sbatch --export=ALL,MAX_REQUESTS=100 ingestion/ballchasing/slurm_ballchasing_metadata.sl
#
# The collector itself persists state before each API request and reserves a
# global 10-second request slot. Cancellation and wall-time expiry are safe:
# submit the same command again to resume.
#SBATCH --job-name=ballchasing_metadata
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

#SBATCH --time=12:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4g
#SBATCH --partition=general

# Notify the batch shell two minutes before wall-time expiry.
#SBATCH --signal=B:TERM@120
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=alshen@unc.edu

set -uo pipefail

module purge
module add anaconda/2024.02

# Under sbatch, the script can be copied to /var/spool/slurmd/.../slurm_script.
# SLURM_SUBMIT_DIR therefore remains the reliable project directory.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

cd "${REPO_ROOT}"

export PYTHON_BIN="${PYTHON_BIN:-python3}"
PYTHON_DIR="$(dirname "$(command -v "${PYTHON_BIN}")")"
export PATH="${PYTHON_DIR}:${PATH}"
export PYTHONUNBUFFERED=1

# Override these via sbatch --export or the submitter environment. The API
# token is loaded from the repository .env file when not already exported.
export CATALOG_PATH="${CATALOG_PATH:-${REPO_ROOT}/ingestion/ballchasing/professional_groups.json}"
export METADATA_STATE_PATH="${METADATA_STATE_PATH:-${REPO_ROOT}/data/ballchasing/metadata_state.json}"
export PACING_STATE_PATH="${PACING_STATE_PATH:-${REPO_ROOT}/data/ballchasing/api_pacing.json}"
export DEBUG_DIR="${DEBUG_DIR:-${REPO_ROOT}/data/ballchasing/api_debug}"
# 4,000 requests fit below the 12-hour wall time at the 10-second cadence.
export MAX_REQUESTS="${MAX_REQUESTS:-4000}"
ENV_FILE="${ENV_FILE:-${REPO_ROOT}/.env}"

if [[ -z "${BALLCHASING_API_TOKEN:-}" && -f "${ENV_FILE}" ]]; then
    set -a
    # The repository's .env is operator-controlled; export its variables to srun.
    source "${ENV_FILE}"
    set +a
fi

mkdir -p "${REPO_ROOT}/logs" "$(dirname "${METADATA_STATE_PATH}")" "${DEBUG_DIR}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "ERROR: Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi
if [[ -z "${BALLCHASING_API_TOKEN:-}" ]]; then
    echo "ERROR: BALLCHASING_API_TOKEN must be exported before submitting this job." >&2
    exit 2
fi
if ! [[ "${MAX_REQUESTS}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: MAX_REQUESTS must be a non-negative integer." >&2
    exit 2
fi

echo "============================================"
echo " Job ${SLURM_JOB_ID:-local} on $(hostname)"
echo " Started: $(date)"
echo " Node: ${SLURMD_NODENAME:-unknown}"
echo "============================================"
echo " REPO_ROOT          : ${REPO_ROOT}"
echo " CATALOG_PATH       : ${CATALOG_PATH}"
echo " METADATA_STATE_PATH: ${METADATA_STATE_PATH}"
echo " PACING_STATE_PATH  : ${PACING_STATE_PATH}"
echo " DEBUG_DIR          : ${DEBUG_DIR}"
echo " ENV_FILE           : ${ENV_FILE}"
echo " MAX_REQUESTS       : ${MAX_REQUESTS}"
echo " PYTHON_BIN         : ${PYTHON_BIN}"
echo "============================================"
"${PYTHON_BIN}" --version
echo ""

SHUTDOWN_REQUESTED=0
shutdown() {
    SHUTDOWN_REQUESTED=1
    echo "Shutdown requested at $(date); the next submission resumes durable state."
}
trap shutdown TERM INT

echo "[$(date)] Starting Ballchasing metadata collector"
status=0
srun --ntasks=1 --cpus-per-task=1 \
    "${PYTHON_BIN}" -m ingestion.ballchasing.collect_metadata \
    --catalog "${CATALOG_PATH}" \
    --state "${METADATA_STATE_PATH}" \
    --pacing-state "${PACING_STATE_PATH}" \
    --debug-dir "${DEBUG_DIR}" \
    --max-requests "${MAX_REQUESTS}" || status=$?

if [[ "${status}" -ne 0 ]]; then
    echo "[$(date)] Collector exited with status ${status}; state is resumable." >&2
else
    echo "[$(date)] Collector completed its request budget."
fi

echo "============================================"
echo " Job stopped: $(date)"
echo " State: ${METADATA_STATE_PATH}"
echo "============================================"

exit "${status}"
