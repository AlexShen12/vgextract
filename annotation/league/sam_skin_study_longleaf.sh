#!/bin/bash
# Sourced by the standalone jobs after Slurm sets the study directory.
set -euo pipefail
study_root="${SLURM_SUBMIT_DIR:?Submit from the prepared SAM skin study directory}"
cd "$study_root/source"
sha256sum --quiet -c SOURCE_SHA256SUMS
source annotation/league/longleaf_paths.sh
study_base="${LEAGUE_WORK_ROOT}/league-midbeast-validation"
export PYTHONPATH="$study_root/source"
export LD_LIBRARY_PATH="$LEAGUE_WORK_ROOT/conda_envs/videogame-league/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

use_pinned_sam() {
    local name="$1" revision="$2"
    local checkout="$study_base/$name-$revision"
    [[ "$(git -C "$checkout" rev-parse HEAD)" == "$revision" ]]
    git -C "$checkout" diff --quiet HEAD --
    export PYTHONPATH="$PYTHONPATH:$checkout"
}
