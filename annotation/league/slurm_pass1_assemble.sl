#!/bin/bash
# Submit after all required producers; does no extraction or model inference.
#SBATCH --job-name=league_pass1_assemble
#SBATCH --output=/work/users/a/l/alshen/%x-%j.out
#SBATCH --error=/work/users/a/l/alshen/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8g
#SBATCH --time=12:00:00
#SBATCH --partition=general
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the staged checkout}"
base=/work/users/a/l/alshen/league-midbeast-validation
run="${LEAGUE_PASS1_RUN:-$PWD/run}"
if [[ $# == 2 ]]; then
    "$base/env-driver/bin/python" -B - "$run" "$1" "$2" "$SLURM_JOB_ID" <<'PY'
from pathlib import Path
import subprocess
import sys

from annotation.league.core import atomic_json, read_json
from annotation.league.pass1 import lock

root = Path(sys.argv[1])
cpu, vl, assembly = sys.argv[2:]
if not all(job.isdigit() for job in (cpu, vl, assembly)):
    raise ValueError('Expected numeric Slurm job IDs')
try:
    accounting = subprocess.check_output(
        ['sacct', '-n', '-X', '-P', '-j', f'{cpu},{vl}',
         '-o', 'JobIDRaw,State,ElapsedRaw,ExitCode'], text=True, timeout=20)
except (OSError, subprocess.SubprocessError) as exc:
    print(f'Longleaf accounting unavailable: {exc}', file=sys.stderr)
    accounting = ''
with lock(root, 'manifest'):
    path = root / 'pass1_manifest.json'
    manifest = read_json(path)
    jobs = manifest.setdefault('longleaf_jobs', {})
    for line in accounting.splitlines():
        job, state, elapsed, exit_code = line.split('|')
        if job in (cpu, vl):
            jobs.setdefault(job, {}).update(status=state, elapsed_seconds=int(elapsed), exit_code=exit_code)
    jobs.setdefault(assembly, {}).update(component='assembly', scope='full_video', status_at_assembly='RUNNING')
    atomic_json(path, manifest)
PY
fi
srun "$base/env-driver/bin/python" -Bu -m annotation.league.pass1 finalize --run-dir "$run"
srun "$base/env-driver/bin/python" -Bu -m annotation.league.pass1 verify --run-dir "$run" --require-complete
