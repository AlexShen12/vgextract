#!/bin/bash
#SBATCH --job-name=league_mask_qwen_score
#SBATCH --output=/work/users/a/l/alshen/%x-%j.out
#SBATCH --error=/work/users/a/l/alshen/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=00:20:00
#SBATCH --partition=general

set -euo pipefail
root="${SLURM_SUBMIT_DIR:?Submit from the staged mask verifier directory}"
cd "$root/score-source"
sha256sum --quiet -c SOURCE_SHA256SUMS
source annotation/league/longleaf_paths.sh
base=/work/users/a/l/alshen/league-midbeast-validation
export LD_LIBRARY_PATH="$LEAGUE_WORK_ROOT/conda_envs/videogame-league/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
exec srun "$base/env-vl/bin/python" -B -u -m annotation.league.mask_verifier evaluate \
    --root "$root" --run "$root/replay" --cache "$root/replay-cache" \
    --siglip-lock "$base/artifacts/midbeast-sam-siglip-35-215/siglip/model-lock.json" --require-complete
