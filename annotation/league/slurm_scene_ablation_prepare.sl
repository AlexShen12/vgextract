#!/bin/bash
#SBATCH --job-name=scene_ablation_prepare
#SBATCH --output=/work/users/a/l/alshen/%x-%j.out
#SBATCH --error=/work/users/a/l/alshen/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8g
#SBATCH --time=01:00:00
#SBATCH --partition=general

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the pinned input-comparison checkout}"
unset PYTHONHOME PYTHONPATH
export LEAGUE_WORK_ROOT=/work/users/a/l/alshen
base="${LEAGUE_WORK_ROOT}/league-midbeast-validation"
media_env="${LEAGUE_WORK_ROOT}/conda_envs/videogame-league"
export LEAGUE_ASSET_CACHE="${base}/assets" LEAGUE_HF_HOME="${base}/hf-home"
source annotation/league/longleaf_paths.sh
export LEAGUE_VL_PYTHON="${base}/env-vl/bin/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PATH="${media_env}/bin:${PATH}"
export LD_LIBRARY_PATH="${media_env}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
scene_comparison="${base}/scene-understanding/v2-pilot/comparison-r1"
srun --ntasks=1 --cpus-per-task=2 "${base}/env-driver/bin/python" -B -m unittest \
  tests.test_league_scene_ablation tests.test_league_scene_split tests.test_league_scene_reference tests.test_league_scene_reference_campaign tests.test_league_scene_source_regressions tests.test_league_scene_reference_audit
srun --ntasks=1 --cpus-per-task=2 "${base}/env-driver/bin/python" -B -u -m annotation.league.scene_ablation prepare \
  --run-dir "$scene_comparison" --metadata reference_sets/league-scene-understanding/v2-pilot \
  --source "${base}/source-20261028/data/youtube/lol_midbeast/OLLZBe5-fg4.mp4"
