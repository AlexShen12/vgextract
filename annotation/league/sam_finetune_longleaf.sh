#!/bin/bash
# Shared setup for independently named pilot jobs. Submit from the staged repo root.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the staged repository root}"
source annotation/league/longleaf_paths.sh
base="$LEAGUE_WORK_ROOT/league-midbeast-validation"
pilot="$LEAGUE_WORK_ROOT/league-sam3-finetune-v1"
sam="$base/sam3-f1647e8d0c1d280991f4d1b7d974c2704537fe66"
export LD_LIBRARY_PATH="$LEAGUE_WORK_ROOT/conda_envs/videogame-league/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$PWD:$sam" PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" OPENBLAS_NUM_THREADS=1
stage="${1:?smoke, train or evaluate}"
shift
extra=()
if [[ "$stage" == evaluate ]]; then
    extra+=(--tracking-cache "$base/assets")
fi
exec srun "$pilot/env/bin/python" -B -u -m annotation.league.sam_finetune "$stage" \
    --root "$pilot/data" --sam-source "$sam" "${extra[@]}" "$@"
