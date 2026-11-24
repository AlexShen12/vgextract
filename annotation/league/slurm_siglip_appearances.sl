#!/bin/bash
#SBATCH --job-name=league_siglip_appearances
#SBATCH --output=/work/users/a/l/alshen/%x-%j.out
#SBATCH --error=/work/users/a/l/alshen/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=00:20:00
#SBATCH --partition=general

set -euo pipefail
root="${SLURM_SUBMIT_DIR:?Submit from the staged appearance run directory}"
base=/work/users/a/l/alshen/league-midbeast-validation
cd "$root/source"
sha256sum --quiet -c SOURCE_SHA256SUMS
cd "$root"
export PYTHONPATH="$root/source"
export LD_LIBRARY_PATH="/work/users/a/l/alshen/conda_envs/videogame-league/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
srun "$base/env-vl/bin/python" -B -u -m annotation.league.sam_multichampion match-appearances \
    --root "$root" --gallery "$root/gallery.json" --scoreboard-skim "$root/scoreboard" \
    --siglip-lock "$root/siglip/model-lock.json" --cache "$root/asset-cache" --require-complete
