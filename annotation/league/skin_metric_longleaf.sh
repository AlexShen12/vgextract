#!/usr/bin/env bash
# Shared environment for the standalone Slurm jobs.
set -euo pipefail
root=/work/users/a/l/alshen/league-skin-metric/16.22-v1
cd "${SLURM_SUBMIT_DIR:?Submit from the staged source directory}"
sha256sum --quiet -c SOURCE_SHA256SUMS
source annotation/league/longleaf_paths.sh
unset PYTHONHOME PYTHONPATH
export LD_LIBRARY_PATH="${LEAGUE_WORK_ROOT}/conda_envs/videogame-league/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python_bin="${LEAGUE_SKIN_PYTHON:-/work/users/a/l/alshen/league-midbeast-validation/env-tracking/bin/python}"
stage="${1:?Stage required}"
shift
case "$stage" in
    fetch)
        srun "$python_bin" -B -u -m annotation.league.skin_metric_assets --root "$root" fetch "$@"
        ;;
    render-pilot)
        srun "$python_bin" -B -u -m annotation.league.skin_metric_dataset --root "$root" render --pilot
        srun "$python_bin" -B -u -m annotation.league.skin_metric_dataset --root "$root" sheets
        srun "$python_bin" -B -u -m annotation.league.skin_metric_dataset --root "$root" merge --pilot-only
        ;;
    render-shard)
        srun "$python_bin" -B -u -m annotation.league.skin_metric_dataset --root "$root" render \
            --index "${SLURM_ARRAY_TASK_ID:?Array index required}"
        ;;
    merge)
        srun "$python_bin" -B -u -m annotation.league.skin_metric_dataset --root "$root" merge
        ;;
    weights)
        srun "$python_bin" -B -u -m annotation.league.skin_metric_train --root "$root" init
        ;;
    train)
        srun "$python_bin" -B -u -m annotation.league.skin_metric_train --root "$root" train --device cuda "$@"
        ;;
    rerank)
        srun "$python_bin" -B -u -m annotation.league.skin_metric_rerank --root "$root" "$@"
        ;;
    train-pilot)
        srun "$python_bin" -B -u -m annotation.league.skin_metric_train --root "$root" train \
            --dataset "$root/pilot/dataset.json" --strategy hard --device cuda --epochs 2 --workers 3
        ;;
    compare)
        srun "$python_bin" -B -u -m annotation.league.skin_metric_train --root "$root" compare
        ;;
    *) printf 'Unknown skin metric stage %s\n' "$stage" >&2; exit 2 ;;
esac
