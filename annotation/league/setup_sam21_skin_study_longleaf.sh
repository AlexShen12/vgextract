#!/bin/bash
# Run from a prepared study's source directory with the existing tracking environment available.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source annotation/league/longleaf_paths.sh
base="${LEAGUE_WORK_ROOT}/league-midbeast-validation"
export LD_LIBRARY_PATH="$LEAGUE_WORK_ROOT/conda_envs/videogame-league/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
tracking_python="$base/env-tracking/bin/python"
revision=2b90b9f5ceec907a1c18123530e92e794ad901a4
checkout="$base/sam2-$revision"
if [[ ! -d "$checkout/.git" ]]; then
    git init "$checkout"
    git -C "$checkout" remote add origin https://github.com/facebookresearch/sam2.git
fi
if [[ "$(git -C "$checkout" rev-parse --verify HEAD 2>/dev/null || true)" != "$revision" ]]; then
    git -C "$checkout" fetch --depth=1 origin "$revision"
    git -C "$checkout" checkout --detach FETCH_HEAD
fi
[[ "$(git -C "$checkout" rev-parse HEAD)" == "$revision" ]]
git -C "$checkout" diff --quiet HEAD --

env_dir="$base/env-sam21-skin-study"
if [[ ! -x "$env_dir/bin/python" ]]; then
    "$tracking_python" -m venv "$env_dir"
fi
"$env_dir/bin/python" - "$base" <<'PY'
from pathlib import Path
import sys
import sysconfig

base = Path(sys.argv[1])
suffix = Path('lib') / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages'
paths = [base / environment / suffix for environment in ('env-tracking', 'env-vl')]
if not all(path.is_dir() for path in paths):
    raise RuntimeError('The qualified tracking and VL environments are required')
Path(sysconfig.get_path('purelib'), 'league_gpu_libraries.pth').write_text(''.join(f'{path}\n' for path in paths))
PY
"$env_dir/bin/python" -m pip install --no-deps 'hydra-core==1.3.2' 'omegaconf==2.3.0' 'antlr4-python3-runtime==4.9.3'
PYTHONPATH="$checkout" "$env_dir/bin/python" -B -c 'from sam2.build_sam import build_sam2_video_predictor'

weights="$base/assets/sam2.1/sam2.1_hiera_large.pt"
sha=2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318
mkdir -p "$(dirname "$weights")"
if ! printf '%s  %s\n' "$sha" "$weights" | sha256sum --quiet -c - 2>/dev/null; then
    temporary="$(mktemp "${weights}.XXXXXX")"
    trap 'rm -f "$temporary"' EXIT
    curl --fail --location --retry 3 --output "$temporary" \
        https://huggingface.co/facebook/sam2.1-hiera-large/resolve/665f8e2ad61cf5f53d65644ff27c8ee525124610/sam2.1_hiera_large.pt
    printf '%s  %s\n' "$sha" "$temporary" | sha256sum --quiet -c -
    mv "$temporary" "$weights"
fi
