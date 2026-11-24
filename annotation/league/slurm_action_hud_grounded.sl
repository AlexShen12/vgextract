#!/bin/bash
# From the staged source directory, run the three clip HUD grounding pilot.
#SBATCH --job-name=league_action_hud_grounded
#SBATCH --output=/work/users/a/l/alshen/%x-%j.out
#SBATCH --error=/work/users/a/l/alshen/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=00:45:00
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --signal=B:TERM@120

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the staged source directory}"
run_dir="$(dirname "$PWD")/run"
unset PYTHONHOME PYTHONPATH
export LEAGUE_WORK_ROOT="${LEAGUE_WORK_ROOT:-/work/users/a/l/alshen}"
base="${LEAGUE_WORK_ROOT}/league-midbeast-validation"
media_env="${LEAGUE_WORK_ROOT}/conda_envs/videogame-league"
export LEAGUE_ASSET_CACHE="${base}/assets" LEAGUE_HF_HOME="${base}/hf-home"
source annotation/league/longleaf_paths.sh
export LEAGUE_VL_PYTHON="${base}/env-vl/bin/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PATH="${media_env}/bin:${PATH}"
export LD_LIBRARY_PATH="${media_env}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
mkdir -p "$run_dir"
cp -n "${base}/run/manifest.json" "$run_dir/manifest.json"
cp -n "${base}/run/config.json" "$run_dir/config.json"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv
child=""
checkpoint() { if [[ -n "$child" ]]; then kill -TERM "$child" 2>/dev/null || true; fi; }
trap checkpoint TERM INT
"${base}/env-driver/bin/python" -B -u - "$run_dir" <<'PY' &
import logging
import os
import sys
from pathlib import Path

from annotation.league.action_v3 import action_window
from annotation.league.core import Config, Store, atomic_json
from annotation.league.local_models import LocalVLRunner

logging.basicConfig(level=logging.INFO)
store = Store(Path(sys.argv[1]))
store.install_signals()
config = Config.load(store.path('config.json'))
config.asset_cache = os.environ['LEAGUE_ASSET_CACHE']
icons = Path(config.asset_cache)/'ability-icons-16.13.1/manifest.json'
runner = LocalVLRunner(store, os.environ['LEAGUE_VL_PYTHON'], config.asset_cache, image_limit=8)
reports = {}
try:
    for clip, start, end in (('action-clip-11', 946.421, 954.421), ('action-clip-18', 1548.689, 1556.689),
                             ('action-clip-19', 1658.728, 1666.728)):
        result = action_window(store, config, start, end, infer=True, runner=runner,
                               icon_references=icons, hud_grounded=True)
        reports[clip] = result['report']
        atomic_json(store.path('experiments/actions-v3-hud-grounded/pilot.json'), reports)
        print(clip, result['results'], flush=True)
finally:
    runner.close()
PY
child=$!
status=0
wait "$child" || status=$?
if kill -0 "$child" 2>/dev/null; then wait "$child" || status=$?; fi
exit "$status"
