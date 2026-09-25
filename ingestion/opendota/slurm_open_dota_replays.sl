#!/bin/bash

# ── Job metadata ─────────────────────────────────────────────────────────────
# Submit from the videogame project directory with:
#   sbatch ingestion/opendota/slurm_open_dota_replays.sl
# Override the wall time at submission if needed, for example:
#   sbatch --time=12:00:00 ingestion/opendota/slurm_open_dota_replays.sl
#
# The default is intentionally aligned with the reference Longleaf job.
# The job uses CPU resources only; no GPU is needed for API calls or bzip2.
#SBATCH --job-name=opendota_replays
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

#SBATCH --time=12:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4g
#SBATCH --partition=general

# Send TERM to the batch shell two minutes before wall-time expiration so the
# current Python process can finish its current durable state transition.
#SBATCH --signal=B:TERM@120
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=alshen@unc.edu

# ────────────────────────────────────────────────────────────────────────────
set -uo pipefail

# ── Modules (Longleaf uses "module add") ─────────────────────────────────────
module purge
module add anaconda/2024.02

# ── Project root ──────────────────────────────────────────────────────────────
# Under sbatch, the script may be copied to /var/spool/slurmd/.../slurm_script.
# SLURM_SUBMIT_DIR is therefore the reliable project directory.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="${SCRIPT_DIR}"
fi

cd "${REPO_ROOT}"

# Prefer the module-provided interpreter.  Override with PYTHON_BIN if the
# project is deployed inside a different Python environment on Longleaf.
export PYTHON_BIN="${PYTHON_BIN:-python3}"
PYTHON_DIR="$(dirname "$(command -v "${PYTHON_BIN}")")"
export PATH="${PYTHON_DIR}:${PATH}"
export PYTHONUNBUFFERED=1

# ── Persistent paths ──────────────────────────────────────────────────────────
# These paths are absolute so the job remains correct regardless of sbatch's
# submission directory behavior.  Override them with sbatch --export or by
# exporting the variables before submission.
export MANIFEST_PATH="${MANIFEST_PATH:-${REPO_ROOT}/data/pro_replays.json}"
export DOWNLOAD_STATE_PATH="${DOWNLOAD_STATE_PATH:-${REPO_ROOT}/data/replay_downloads.json}"
export DEBUG_DIR="${DEBUG_DIR:-${REPO_ROOT}/data/api_debug}"
export REPLAY_OUTPUT_DIR="${REPLAY_OUTPUT_DIR:-${REPO_ROOT}/data/replays}"
export CYCLE_SLEEP_SECONDS="${CYCLE_SLEEP_SECONDS:-60}"

# SLURM opens --output/--error before the script runs, so this directory must
# exist in the submitted project checkout.  It is tracked via logs/.gitkeep.
mkdir -p "${REPO_ROOT}/logs" "${REPO_ROOT}/data" "${DEBUG_DIR}" "${REPLAY_OUTPUT_DIR}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "ERROR: Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi
if ! command -v curl >/dev/null 2>&1; then
    echo "ERROR: curl is required but was not found on PATH" >&2
    exit 1
fi
if ! command -v bzip2 >/dev/null 2>&1; then
    echo "ERROR: bzip2 is required but was not found on PATH" >&2
    exit 1
fi

# ── Diagnostics ───────────────────────────────────────────────────────────────
echo "============================================"
echo " Job ${SLURM_JOB_ID:-local} on $(hostname)"
echo " Started: $(date)"
echo " Node: ${SLURMD_NODENAME:-unknown}"
echo "============================================"
echo " REPO_ROOT          : ${REPO_ROOT}"
echo " MANIFEST_PATH      : ${MANIFEST_PATH}"
echo " DOWNLOAD_STATE_PATH: ${DOWNLOAD_STATE_PATH}"
echo " DEBUG_DIR          : ${DEBUG_DIR}"
echo " REPLAY_OUTPUT_DIR  : ${REPLAY_OUTPUT_DIR}"
echo " CYCLE_SLEEP_SECONDS: ${CYCLE_SLEEP_SECONDS}"
echo " PYTHON_BIN         : ${PYTHON_BIN}"
echo "============================================"
"${PYTHON_BIN}" --version
curl --version | head -1
bzip2 --version | head -1
echo ""

# ── Graceful wall-time/cancellation handling ─────────────────────────────────
SHUTDOWN_REQUESTED=0
shutdown() {
    SHUTDOWN_REQUESTED=1
    echo "Shutdown requested at $(date); durable scripts will stop after the current step."
}
trap shutdown TERM INT

run_collector() {
    echo "[$(date)] Starting OpenDota collector"
    local status=0
    srun --ntasks=1 --cpus-per-task=1 \
        "${PYTHON_BIN}" -m ingestion.opendota.collect_pro_replays \
        --manifest "${MANIFEST_PATH}" \
        --debug-dir "${DEBUG_DIR}" || status=$?
    if [[ "${status}" -ne 0 ]]; then
        echo "[$(date)] Collector exited with status ${status}; state is resumable." >&2
    fi
    return "${status}"
}

run_downloader() {
    echo "[$(date)] Starting replay downloader"
    local status=0
    srun --ntasks=1 --cpus-per-task=1 \
        "${PYTHON_BIN}" -m ingestion.opendota.download_replays \
        --manifest "${MANIFEST_PATH}" \
        --state "${DOWNLOAD_STATE_PATH}" \
        --output-dir "${REPLAY_OUTPUT_DIR}" || status=$?
    if [[ "${status}" -ne 0 ]]; then
        echo "[$(date)] Downloader exited with status ${status}; partial files/state are resumable." >&2
    fi
    return "${status}"
}

# ── Run until the requested Slurm wall time ends ──────────────────────────────
# The collector fetches the current proMatches snapshot once per cycle, then
# resolves only newly discovered/unresolved IDs.  The downloader independently
# consumes every resolved replay URL.  Both scripts persist after each state
# transition, so cancellation or wall-time expiry is safe.
cycle=0
while [[ "${SHUTDOWN_REQUESTED}" -eq 0 ]]; do
    cycle=$((cycle + 1))
    echo "============================================"
    echo " Cycle ${cycle} started: $(date)"
    echo "============================================"

    run_collector || true
    [[ "${SHUTDOWN_REQUESTED}" -eq 1 ]] && break

    run_downloader || true
    [[ "${SHUTDOWN_REQUESTED}" -eq 1 ]] && break

    echo "[$(date)] Cycle ${cycle} complete; sleeping ${CYCLE_SLEEP_SECONDS}s before refresh."
    sleep "${CYCLE_SLEEP_SECONDS}" || true
done

echo "============================================"
echo " Job stopped: $(date)"
echo " Manifest: ${MANIFEST_PATH}"
echo " Replays : ${REPLAY_OUTPUT_DIR}"
echo "============================================"
