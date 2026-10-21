#!/usr/bin/env bash

# Run a local macOS trial from the already saved Aatrox catalog.
# This allows at most one new video download in the whole run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
CHANNELS_FILE="${CHANNELS_FILE:-${SCRIPT_DIR}/local_trial_channels.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/data/youtube}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "ERROR: Python environment not found at ${PYTHON_BIN}." >&2
    echo "Create it with: python3 -m venv .venv" >&2
    echo "Then install requirements: .venv/bin/python -m pip install -r ingestion/youtube/requirements.txt" >&2
    exit 1
fi

if ! "${PYTHON_BIN}" -c 'import yt_dlp' >/dev/null 2>&1; then
    echo "ERROR: yt-dlp is missing from ${PYTHON_BIN}." >&2
    echo "Install with: .venv/bin/python -m pip install -r ingestion/youtube/requirements.txt" >&2
    exit 1
fi

for executable in ffmpeg ffprobe; do
    if ! command -v "${executable}" >/dev/null 2>&1; then
        echo "ERROR: ${executable} is required. On macOS, install it with: brew install ffmpeg node" >&2
        exit 1
    fi
done
if ! command -v node >/dev/null 2>&1 && ! command -v deno >/dev/null 2>&1; then
    echo "ERROR: yt-dlp YouTube extraction requires Node.js or Deno. On macOS: brew install node" >&2
    exit 1
fi

if [[ ! -f "${OUTPUT_ROOT}/lol_domisum_aatrox/catalog.json" ]]; then
    echo "ERROR: saved Aatrox catalog not found under ${OUTPUT_ROOT}." >&2
    echo "This trial intentionally uses a saved catalog to avoid rescanning all League channels." >&2
    exit 1
fi

echo "Using saved catalog; maximum new downloads this run: 1"
echo "Output root: ${OUTPUT_ROOT}"
exec "${PYTHON_BIN}" -m ingestion.youtube.download_channels \
    --channels "${CHANNELS_FILE}" \
    --output-root "${OUTPUT_ROOT}" \
    --use-catalog \
    --max-downloads 1
