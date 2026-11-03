#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source annotation/league/longleaf_paths.sh
mode="${1:-preflight}"
case "${mode}" in preflight|draft|evaluate) ;; *) echo "Submit GPU preflight, draft or evaluate; use qwen35_longleaf.sh directly for CPU/cache stages" >&2; exit 1;; esac
suite="${LEAGUE_SUITE:-qwen35}"
case "${suite}" in
    hp-vlm|scene-identity|ability-alignment|cooldown-identity)
        batch=("annotation/league/slurm_qwen35_${suite//-/_}.sl" "${mode}");;
    qwen35) batch=(annotation/league/slurm_qwen35.sl qwen35 "${mode}");;
    *) echo "Expected hp-vlm, scene-identity, ability-alignment, cooldown-identity or qwen35" >&2; exit 1;;
esac
sbatch --parsable --job-name="league_qwen35_${mode}" --nodes=1 --ntasks=1 \
    --partition="${LEAGUE_PARTITION:-l40-gpu}" --qos="${LEAGUE_QOS:-gpu_access}" \
    --gres="${LEAGUE_GRES:-gpu:nvidia_l40s:1}" --cpus-per-task="${LEAGUE_CPUS:-8}" \
    --mem="${LEAGUE_MEM:-96g}" --time="${LEAGUE_TIME:-04:00:00}" \
    --output="${LEAGUE_LOG_DIR}/%x-%j.out" --error="${LEAGUE_LOG_DIR}/%x-%j.err" \
    "${batch[@]}"
