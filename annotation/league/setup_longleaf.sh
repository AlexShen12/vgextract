#!/bin/bash
# One-time interactive setup on a Longleaf login node. Run from any directory.
# Uses the same model-cache default as slurm_annotate.sl; can be overridden.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

LEAGUE_ENV="${LEAGUE_ENV:-/work/users/a/l/alshen/conda_envs/videogame-league}"
LEAGUE_RUN_DIR="${LEAGUE_RUN_DIR:-${REPO_ROOT}/data/league_runs/aatrox-26.13}"
LEAGUE_VIDEO="${LEAGUE_VIDEO:-${REPO_ROOT}/data/youtube/lol_domisum_aatrox/xICbe-OOPvE.mp4}"
HF_HOME="${HF_HOME:-/work/users/a/l/alshen/hf-cache}"
export HF_HOME
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
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if [[ ! -x "${LEAGUE_ENV}/bin/python" ]]; then
    conda create -p "${LEAGUE_ENV}" python=3.12 -y
fi
conda activate "${LEAGUE_ENV}"
export HF_HOME

python -m pip install -r annotation/league/inference-environment.lock.txt
python -m pip check

# Verify every exact pin in the lock file and import the inference stack.
python - <<'PY'
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import re
import sys

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

python -m annotation.league --run-dir "${LEAGUE_RUN_DIR}" prepare --video "${LEAGUE_VIDEO}"
python -m annotation.league --run-dir "${LEAGUE_RUN_DIR}" cache-references
python -m annotation.league --run-dir "${LEAGUE_RUN_DIR}" cache-models
python -m annotation.league --run-dir "${LEAGUE_RUN_DIR}" processor-preview

cat <<EOF

Initial setup is complete. From the repository root, submit the GPU preflight and run:
  sbatch --export=ALL,LEAGUE_STAGE=preflight annotation/league/slurm_annotate.sl
  # Wait for preflight to finish successfully, then:
  sbatch annotation/league/slurm_annotate.sl
EOF
