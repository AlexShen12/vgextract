#!/bin/bash
#SBATCH --job-name=league_midbeast_pass1
#SBATCH --output=/work/users/a/l/alshen/%x-%j.out
#SBATCH --error=/work/users/a/l/alshen/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16g
#SBATCH --time=01:20:00
#SBATCH --partition=general

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the staged source directory}"
stage="$PWD"
base=/work/users/a/l/alshen/league-midbeast-validation
media=/work/users/a/l/alshen/conda_envs/videogame-league
driver="$base/env-driver/bin/python"
video="$base/source-20261028/data/youtube/lol_midbeast/OLLZBe5-fg4.mp4"
output="$stage/results-${SLURM_JOB_ID:?}"
run="$output/run"
scoreboard="$output/scoreboard"
export LEAGUE_ASSET_CACHE="$stage/assets"
source annotation/league/longleaf_paths.sh
export PATH="$media/bin:$PATH"
export LD_LIBRARY_PATH="$media/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_THREAD_LIMIT=1 OPENBLAS_NUM_THREADS=1
export LEAGUE_TESSERACT=/usr/bin/tesseract
mkdir -p "$output"

"$driver" -B - "$output" "$video" <<'PY'
from pathlib import Path
import sys

from annotation.league.core import Config, atomic_json
from annotation.league.intake_profile import load_profile

output, video = map(Path, sys.argv[1:])
profile = load_profile("midbeast-1080p-inventory-v3")
profile["fields"] = [field for field in profile["fields"]
                     if field.get("backend") == "inventory" or field["field"] == "game_clock"]
config = Config.load(Path("annotation/league/midbeast.json"))
config.source_video = str(video)
config.layout_profile = str(output / "inventory-cpu-profile.json")
atomic_json(Path(config.layout_profile), profile)
atomic_json(output / "inventory-cpu-config.json", config.to_dict())
PY

cli=("$driver" -Bu -m annotation.league --offline --debug --run-dir "$run")
srun --exact -c 2 --mem=4g "${cli[@]}" --config "$output/inventory-cpu-config.json" prepare --video "$video" > "$output/prepare.log" 2>&1
"$driver" -B - "$run" "$output" <<'PY'
import math
from pathlib import Path
import sys

from annotation.league.core import atomic_json, read_json

run, output = map(Path, sys.argv[1:])
manifest = read_json(run / "manifest.json")
assert manifest["source"]["sha256"] == "f4f36e1a274617ecbe224eaf7fc3f8375f757acb7a517bcc0ee7780fcd83f240"
start, end = manifest["interval"]
fps = read_json(run / "config.json")["fps"]
atomic_json(output / "inventory-sample-times.json",
            [start + i / fps for i in range(math.ceil((end - start) * fps))])
print(f"Source interval {start:.3f} to {end:.3f} seconds")
PY

# Cast bars, inventory and scoreboard scans each write their own artifacts.
srun --exclusive --exact -c 3 --mem=4g "$driver" -Bu -m annotation.league.cast_bar_intake \
    --run-dir "$run" --video "$video" \
    --profile annotation/league/profiles/midbeast-cast-bars-1080p-v1.json \
    --catalog "$LEAGUE_ASSET_CACHE/abilities/16.22.1/catalog.json" \
    --start 0 --end 1732.765896 > "$output/casts.log" 2>&1 &
casts=$!
srun --exclusive --exact -c 2 --mem=6g "${cli[@]}" annotate \
    --sample-times "$output/inventory-sample-times.json" > "$output/inventory.log" 2>&1 &
inventory=$!
srun --exclusive --exact -c 3 --mem=4g "$driver" -Bu -m annotation.league.scoreboard_skim \
    --video "$video" --profile midbeast-1080p-inventory-v3 --cache "$LEAGUE_ASSET_CACHE" \
    --output "$scoreboard" --single-match --skip-names > "$output/scoreboard.log" 2>&1 &
roster=$!

status=0
wait "$roster" || status=1
if [[ "$status" == 0 ]]; then
    # The standalone reader writes only killfeed artifacts and can run alongside inventory.
    srun --exclusive --exact -c 3 --mem=4g "$driver" -Bu - "$run" "$scoreboard" > "$output/killfeed.log" 2>&1 <<'PY' &
import json
import os
from pathlib import Path
import sys

from annotation.league.core import Store
from annotation.league.killfeed import run

store = Store(Path(sys.argv[1]))
store.install_signals()
print(json.dumps(run(store, os.environ["LEAGUE_ASSET_CACHE"], Path(sys.argv[2])), indent=2))
PY
    feed=$!
    wait "$feed" || status=1
fi
wait "$inventory" || status=1
wait "$casts" || status=1
[[ "$status" == 0 ]]

srun --exact -c 2 --mem=6g "$driver" -Bu - "$run" <<'PY'
import json
from pathlib import Path
import sys

from annotation.league.core import Config, Store, read_json
from annotation.league.timeline import fuse

store = Store(Path(sys.argv[1]))
report = read_json(store.path("intake_report.json"))
assert report["operational_passed"] and report["frames"] == report["expected_frames"]
print(json.dumps(fuse(store, Config.load(store.path("config.json"))), indent=2))
PY
srun --exact -c 2 --mem=4g "$driver" -Bu -m annotation.league.killfeed_benchmark \
    --run-dir "$run" --scoreboard-skim "$scoreboard" --cache "$LEAGUE_ASSET_CACHE" > "$output/killfeed-benchmark.log" 2>&1
echo "Full Midbeast results at $output"
