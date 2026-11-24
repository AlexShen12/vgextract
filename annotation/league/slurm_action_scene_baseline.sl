#!/bin/bash
# From the staged source directory, submit with preflight, development or held-out.
# Held-out first freezes the completed development recipe if needed.
#SBATCH --job-name=league_action_scene_baseline
#SBATCH --output=/work/users/a/l/alshen/%x-%j.out
#SBATCH --error=/work/users/a/l/alshen/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=04:00:00
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --signal=B:TERM@120

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
stage_dir="$PWD"
run_dir="$(dirname "$stage_dir")/run"
labels="${run_dir}/benchmarks/actions-v3/scene-baseline-v1"
case "${1:?Expected preflight, development, freeze or held-out}" in
    preflight) labels="${stage_dir}/scene-labels-preflight"; flags=(--preflight-only) ;;
    development) flags=(--draft) ;;
    freeze) flags=(--freeze) ;;
    held-out) flags=() ;;
    *) echo 'Invalid action scene baseline mode' >&2; exit 2 ;;
esac

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
args=(--run-dir "$run_dir" --offline experiment --suite actions --infer --scene-baseline
      --labels "$labels" --icon-references "${LEAGUE_ASSET_CACHE}/ability-icons-16.13.1/manifest.json" "${flags[@]}")
if [[ "$1" == freeze ]]; then
    exec "${base}/env-driver/bin/python" -B -m annotation.league "${args[@]}"
fi
if [[ "$1" == held-out && ! -f "${labels}/scene-baseline-freeze.json" ]]; then
    "${base}/env-driver/bin/python" -B -m annotation.league "${args[@]}" --freeze
fi
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv
child=""
checkpoint() { if [[ -n "$child" ]]; then kill -TERM "$child" 2>/dev/null || true; fi; }
trap checkpoint TERM INT
"${base}/env-driver/bin/python" -B -u -m annotation.league "${args[@]}" &
child=$!
status=0
wait "$child" || status=$?
if kill -0 "$child" 2>/dev/null; then wait "$child" || status=$?; fi
exit "$status"
