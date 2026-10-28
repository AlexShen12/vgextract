#!/bin/bash
# Resource settings are overridable without editing the batch script.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source annotation/league/longleaf_paths.sh
sbatch --partition="${LEAGUE_PARTITION:-l40-gpu}" --qos="${LEAGUE_QOS:-gpu_access}" \
    --gres="${LEAGUE_GRES:-gpu:nvidia_l40s:1}" --cpus-per-task="${LEAGUE_CPUS:-8}" \
    --mem="${LEAGUE_MEM:-64g}" --time="${LEAGUE_TIME:-12:00:00}" \
    --output="${LEAGUE_LOG_DIR}/%x-%j.out" --error="${LEAGUE_LOG_DIR}/%x-%j.err" \
    "$@" annotation/league/slurm_annotate.sl
