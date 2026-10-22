#!/bin/bash
# One-time interactive setup on a Longleaf login node. Run from any directory.
# Uses the same dedicated model-cache setting as slurm_annotate.sl.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

LEAGUE_ENV="${LEAGUE_ENV:-/work/users/a/l/alshen/conda_envs/videogame-league}"
LEAGUE_RUN_DIR="${LEAGUE_RUN_DIR:-${REPO_ROOT}/data/league_runs/aatrox-26.13}"
LEAGUE_VIDEO="${LEAGUE_VIDEO:-${REPO_ROOT}/data/youtube/lol_domisum_aatrox/xICbe-OOPvE.mp4}"
LEAGUE_HF_HOME="${LEAGUE_HF_HOME:-/users/a/l/alshen/hf-cache}"
export HF_HOME="${LEAGUE_HF_HOME}"
mkdir -p "${HF_HOME}"

if [[ ! -f "${LEAGUE_VIDEO}" ]]; then
    echo "Video not found: ${LEAGUE_VIDEO}" >&2
    echo "Transfer it with Globus or set LEAGUE_VIDEO to its path." >&2
    exit 2
fi

if ! command -v module >/dev/null 2>&1; then
    echo "The module command is unavailable; run this script in a Longleaf login shell." >&2
    exit 2
fi
module add anaconda/2024.02
command -v conda >/dev/null 2>&1 || { echo "conda not found after loading Anaconda." >&2; exit 2; }
# Ignore inherited Python paths and user packages, including a failed earlier
# installation into ~/.local. Apply this to Conda's Python as well.
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
CONDA_BASE="$(conda info --base)"
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
set -u

# Keep media tools inside the environment so they remain available after the
# Slurm script purges modules and activates this Conda prefix.
if [[ ! -x "${LEAGUE_ENV}/bin/python" ]]; then
    conda create -p "${LEAGUE_ENV}" -c conda-forge python=3.12 pip ffmpeg -y
else
    # An existing prefix may have an older Python or no local pip. Enforce the
    # bootstrap requirements on reruns, rather than only installing FFmpeg.
    conda install -p "${LEAGUE_ENV}" -c conda-forge python=3.12 pip ffmpeg -y
fi
set +u
conda activate "${LEAGUE_ENV}"
set -u
unset PYTHONHOME PYTHONPATH
export HF_HOME PYTHONNOUSERSITE=1
LEAGUE_PYTHON="${LEAGUE_ENV}/bin/python"

"${LEAGUE_PYTHON}" annotation/league/check_environment.py "${LEAGUE_ENV}" --writable
# Explicitly refuse a user install even if pip configuration requests one.
"${LEAGUE_PYTHON}" -m pip install --no-user -r annotation/league/inference-environment.lock.txt
"${LEAGUE_PYTHON}" -m pip check

# Verify every exact pin in the lock file and import the inference stack.
"${LEAGUE_PYTHON}" - <<'PY'
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import os
import re
import sys

print(f"Python interpreter: {sys.executable}")
print(f"Hugging Face cache: {os.environ['HF_HOME']}")
lock = Path("annotation/league/inference-environment.lock.txt")
requirements = [line.strip() for line in lock.read_text().splitlines()
                if line.strip() and not line.lstrip().startswith("#")]
errors = []
for requirement in requirements:
    match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^;\s]+)", requirement)
    if not match:
        errors.append(f"Unsupported lock entry: {requirement}")
        continue
    package, expected = match.groups()
    try:
        actual = version(package)
    except PackageNotFoundError:
        errors.append(f"Missing package: {package}=={expected}")
    else:
        print(f"{package}=={actual}")
        if actual != expected:
            errors.append(f"Version mismatch: {package} expected {expected}, found {actual}")

if not errors:
    try:
        import torch
        import torchvision
        import transformers
        import accelerate
        import av
        from PIL import Image
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    except Exception as exc:
        errors.append(f"Inference import failed: {type(exc).__name__}: {exc}")

if errors:
    print("\n".join(errors), file=sys.stderr)
    raise SystemExit(1)
print("All locked Python packages and Qwen inference imports are available.")
PY

for executable in ffmpeg ffprobe; do
    command -v "${executable}" >/dev/null 2>&1 || {
        echo "Required executable not found on PATH: ${executable}" >&2
        echo "Load the Longleaf module that provides it, then rerun setup." >&2
        exit 2
    }
    command -v "${executable}"
done

"${LEAGUE_PYTHON}" -m annotation.league --run-dir "${LEAGUE_RUN_DIR}" prepare --video "${LEAGUE_VIDEO}"
"${LEAGUE_PYTHON}" -m annotation.league --run-dir "${LEAGUE_RUN_DIR}" cache-references
"${LEAGUE_PYTHON}" -m annotation.league --run-dir "${LEAGUE_RUN_DIR}" cache-models
"${LEAGUE_PYTHON}" -m annotation.league --run-dir "${LEAGUE_RUN_DIR}" processor-preview

cat <<EOF

Initial setup is complete. From the repository root, submit the GPU preflight and run:
  sbatch --export=ALL,LEAGUE_STAGE=preflight annotation/league/slurm_annotate.sl
  # Wait for preflight to finish successfully, then:
  sbatch annotation/league/slurm_annotate.sl

For manual stages, activate ${LEAGUE_ENV} in your terminal so Python and FFmpeg are on PATH.
Setup's Conda activation does not persist in the calling terminal.
EOF
