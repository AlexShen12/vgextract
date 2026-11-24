#!/bin/bash
# Called only by the separate experiment scripts. Never submits jobs.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the frozen body campaign source}"
sha256sum -c SOURCE_SHA256SUMS >/dev/null
base=/work/users/a/l/alshen/league-midbeast-validation
campaign=$(cat body-campaign-path.txt)
[[ "$campaign" == /work/users/a/l/alshen/* ]]
[[ -f "$campaign/prepared.json" ]]
export LEAGUE_WORK_ROOT=/work/users/a/l/alshen
export PATH="/work/users/a/l/alshen/conda_envs/videogame-league/bin:${PATH}"
export LD_LIBRARY_PATH="/work/users/a/l/alshen/conda_envs/videogame-league/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" OPENBLAS_NUM_THREADS=1
unset PYTHONHOME PYTHONPATH
srun "$base/env-body/bin/python" -u -B -m annotation.league.body_stage "$@" \
  --root "$campaign" --cache "$base/assets"
