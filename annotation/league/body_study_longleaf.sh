#!/bin/bash
set -euo pipefail
stage="${1:?stage required}"
root=/work/users/a/l/alshen/league-midbeast-validation/artifacts/body-comparison-dev-20261110-r1
export LD_LIBRARY_PATH="/work/users/a/l/alshen/conda_envs/videogame-league/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PATH="/work/users/a/l/alshen/conda_envs/videogame-league/bin:$PATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export XFORMERS_DISABLED=1
export HF_HUB_OFFLINE=1
export LEAGUE_TESSERACT=/usr/bin/tesseract4.1.1
export PYTHONDONTWRITEBYTECODE=1
cd "$root/source"
sha256sum --quiet -c SOURCE_SHA256SUMS
case "$stage" in
  preflight|casts)
    srun "$root/env/bin/python" -B -m annotation.league.body_study_driver "$stage" --spec "$root/study.json" --output "$root/run/$stage" ;;
  render)
    srun "$root/tools/blender-4.5.4-linux-x64/blender" --background --factory-startup --threads "${SLURM_CPUS_PER_TASK:-4}" \
      --python annotation/league/body_static.py -- worker --input "$root/assets/static/manifest.json" --output "$root/run/static" ;;
  compare)
    srun "$root/env/bin/python" -B -m annotation.league.body_study --spec "$root/study.json" --output "$root/run/comparison" ;;
  *) exit 2 ;;
esac
