#!/bin/bash
# Submit from the staged checkout containing inventory-reference-labels/crops.json.
#SBATCH --job-name=league_inventory_draft
#SBATCH --output=/work/users/a/l/alshen/%x-%j.out
#SBATCH --error=/work/users/a/l/alshen/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=02:00:00
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --signal=B:TERM@120

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the staged checkout}"
unset PYTHONHOME PYTHONPATH
export LEAGUE_WORK_ROOT="${LEAGUE_WORK_ROOT:-/work/users/a/l/alshen}"
base="${LEAGUE_WORK_ROOT}/league-midbeast-validation"
media_env="${LEAGUE_WORK_ROOT}/conda_envs/videogame-league"
export LEAGUE_ASSET_CACHE="${LEAGUE_ASSET_CACHE:-${base}/assets}"
export LEAGUE_HF_HOME="${LEAGUE_HF_HOME:-${base}/hf-home}"
source annotation/league/longleaf_paths.sh
export LEAGUE_VL_PYTHON="${LEAGUE_VL_PYTHON:-${base}/env-vl/bin/python}"
export PATH="${media_env}/bin:${PATH}"
export LD_LIBRARY_PATH="${media_env}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

driver="${base}/env-driver/bin/python"
video="${LEAGUE_VIDEO:-${base}/source-20261028/data/youtube/lol_midbeast/OLLZBe5-fg4.mp4}"
run="${LEAGUE_RUN_DIR:-${SLURM_SUBMIT_DIR}/run}"
labels="${SLURM_SUBMIT_DIR}/inventory-reference-labels/crops.json"
args=(--run-dir "$run" --offline)
[[ -r "$labels" && -r "$video" && -x "$driver" && -x "$LEAGUE_VL_PYTHON" ]]
for program in ffmpeg ffprobe tesseract; do command -v "$program"; done
gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader)"
[[ "$gpu" == *L40S* ]]
echo "GPU $gpu, run $run"

child=""
stop_child() { if [[ -n "$child" ]]; then kill -TERM "$child" 2>/dev/null || true; fi; }
trap stop_child TERM INT

run_stage() {
    local status=0
    "$driver" -u -B -m annotation.league "${args[@]}" "$@" &
    child=$!
    wait "$child" || status=$?
    if kill -0 "$child" 2>/dev/null; then wait "$child" || status=$?; fi
    child=""
    return "$status"
}

run_stage prepare --video "$video"
run_stage annotate --sample-times "$labels"

# Diagnostic scores intentionally return 2 because they cannot pass release.
score_status=0
run_stage intake-benchmark --labels "$labels" --draft || score_status=$?
if [[ "$score_status" != 2 ]]; then
    echo "Unexpected draft benchmark exit status $score_status" >&2
    exit 1
fi
"$driver" -B -c 'import json, sys; report = json.load(open(sys.argv[1])); assert report["draft_labels"] and not report["passed"]' "$run/intake_benchmark.json"
echo "Draft report saved at $run/intake_benchmark.json"
