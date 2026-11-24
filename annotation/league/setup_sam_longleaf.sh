#!/bin/bash
# Install SAM locally while reusing the existing immutable GPU library environment.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export LEAGUE_WORK_ROOT="${LEAGUE_WORK_ROOT:-/work/users/a/l/alshen}"
base="${LEAGUE_WORK_ROOT}/league-midbeast-validation"
export LEAGUE_ASSET_CACHE="${base}/assets" LEAGUE_HF_HOME="${base}/hf-home"
source annotation/league/longleaf_paths.sh
export LD_LIBRARY_PATH="${LEAGUE_WORK_ROOT}/conda_envs/videogame-league/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
sam_env="${base}/env-tracking"
sam_code="${base}/sam3-f1647e8d0c1d280991f4d1b7d974c2704537fe66"
if [[ ! -d "$sam_code" ]]; then
    git init "$sam_code"
    git -C "$sam_code" remote add origin https://github.com/facebookresearch/sam3.git
    git -C "$sam_code" fetch --depth=1 origin f1647e8d0c1d280991f4d1b7d974c2704537fe66
    git -C "$sam_code" checkout --detach FETCH_HEAD
fi
[[ "$(git -C "$sam_code" rev-parse HEAD)" == f1647e8d0c1d280991f4d1b7d974c2704537fe66 ]]
if [[ ! -x "$sam_env/bin/python" ]]; then
    "${base}/env-vl/bin/python" -m venv "$sam_env"
fi
"$sam_env/bin/python" - "$base" <<'PY'
from pathlib import Path
import sys, sysconfig
base=Path(sys.argv[1])
gpu=base/'env-vl/lib/python3.12/site-packages'
if not gpu.is_dir(): raise RuntimeError('Expected existing Python 3.12 GPU environment')
Path(sysconfig.get_path('purelib'),'league_gpu_libraries.pth').write_text(str(gpu)+'\n')
PY
"$sam_env/bin/python" -m pip install 'setuptools<81' 'numpy>=1.26,<2' 'timm>=1.0.17' 'ftfy==6.1.1' iopath pycocotools decord scipy matplotlib
"$sam_env/bin/python" -m pip install --no-deps --no-build-isolation "$sam_code"
"$sam_env/bin/python" -B -m pip freeze > "${sam_env}/installed.txt"
"$sam_env/bin/python" -B - <<'PY'
from sam3.model_builder import build_sam3_multiplex_video_predictor
from annotation.league.local_models import cache_models
import os
cache_models(os.environ['LEAGUE_ASSET_CACHE'],roles=('tracking',))
print('Pinned SAM import and checkpoint cache are ready')
PY
