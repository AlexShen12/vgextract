#!/bin/bash
#SBATCH --job-name=league_worlds_game1_broadcast
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8g
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the staged broadcast directory}"
work_root=/work/users/a/l/alshen
driver="$work_root/league-midbeast-validation/env-driver/bin/python"
media="$work_root/conda_envs/videogame-league"
inputs="$PWD/inputs"
output="$PWD/results-${SLURM_JOB_ID:?}"
export PATH="$media/bin:$PATH"
export LD_LIBRARY_PATH="$media/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$PWD/deps"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LEAGUE_TESSERACT=/usr/bin/tesseract OMP_THREAD_LIMIT=1 OPENBLAS_NUM_THREADS=1

clips=()
while IFS=$'\t' read -r filename start end; do
    [[ -r "$inputs/$filename" ]]
    clips+=(--clip "$inputs/$filename" "$start" "$end")
done < "$inputs/clips.tsv"
[[ ${#clips[@]} -gt 0 ]]
"$driver" -B -c 'import av, cv2, scenedetect'
ffmpeg -version | sed -n '1p'
tesseract --version | sed -n '1p'
srun "$driver" -Bu -m annotation.league.pass1 broadcast \
  --run-dir "$output" --profile annotation/league/profiles/worlds-2026-finals-game1.json \
  --assets "$inputs/overlay-profile" --captions "$inputs/fFlOTEiQ9x0.en-orig.json3" \
  --review "$inputs/manual-review.json" --workers "${SLURM_CPUS_PER_TASK:-4}" "${clips[@]}"
srun "$driver" -Bu -m annotation.league.pass1 verify --run-dir "$output"
