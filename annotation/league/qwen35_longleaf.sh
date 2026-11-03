#!/bin/bash
# Reuse the established driver/VL installations. All writes stay under the work root.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source annotation/league/longleaf_paths.sh
league_validation="${LEAGUE_VALIDATION_ROOT:-${LEAGUE_WORK_ROOT}/league-midbeast-validation}"
league_driver="${LEAGUE_QWEN35_DRIVER:-${league_validation}/env-driver/bin/python}"
export LEAGUE_QWEN35_PYTHON="${LEAGUE_QWEN35_PYTHON:-${league_validation}/env-vl/bin/python}"
export LEAGUE_ASSET_CACHE="${LEAGUE_QWEN35_ASSETS:-${league_validation}/assets}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LD_LIBRARY_PATH="${LEAGUE_WORK_ROOT}/conda_envs/videogame-league/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
league_run="${LEAGUE_QWEN35_RUN:-${league_validation}/run}"
mode="${1:-preflight}"
if [[ "${mode}" == cache ]]; then
    unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
    "${league_driver}" -B -m annotation.league --run-dir "${league_run}" cache-models --suite qwen35
    "${league_driver}" -B -m annotation.league --run-dir "${league_run}" cache-qwen35-references
    exit 0
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
args=(--run-dir "${league_run}" --offline experiment --suite "${LEAGUE_SUITE:-qwen35}")
if [[ -n "${LEAGUE_LABELS:-}" ]]; then args+=(--labels "${LEAGUE_LABELS}"); fi
if [[ -n "${LEAGUE_ICON_REFERENCES:-}" ]]; then args+=(--icon-references "${LEAGUE_ICON_REFERENCES}"); fi
case "${mode}" in
    initialize) args+=(--initialize);;
    prepare) args+=(--prepare-only);;
    preflight) args+=(--draft --infer --preflight-only);;
    draft) args+=(--draft --infer);;
    freeze) args+=(--freeze);;
    evaluate) args+=(--infer);;
    *) echo "Expected cache, initialize, prepare, preflight, draft, freeze or evaluate" >&2; exit 1;;
esac
# Experimental CLI commands conventionally return 2, even on successful completion.
status=0
"${league_driver}" -B -m annotation.league "${args[@]}" || status=$?
if [[ "${status}" == 2 ]]; then exit 0; fi
exit "${status}"
