#!/bin/bash
# Create only the pilot environment; reuse immutable CUDA libraries and upstream source.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export LEAGUE_HF_HOME="${LEAGUE_HF_HOME:-/work/users/a/l/alshen/league-midbeast-validation/hf-home}"
source annotation/league/longleaf_paths.sh
base="$LEAGUE_WORK_ROOT/league-midbeast-validation"
pilot="$LEAGUE_WORK_ROOT/league-sam3-finetune-v1"
env="$pilot/env"
sam="$base/sam3-f1647e8d0c1d280991f4d1b7d974c2704537fe66"
[[ "$(git -C "$sam" rev-parse HEAD)" == f1647e8d0c1d280991f4d1b7d974c2704537fe66 ]]
git -C "$sam" diff --quiet HEAD --
mkdir -p "$pilot/logs"
if [[ ! -x "$env/bin/python" ]]; then
    "$base/env-tracking/bin/python" -m venv "$env"
fi
"$env/bin/python" - "$base" <<'PY'
from pathlib import Path
import sys, sysconfig
base = Path(sys.argv[1])
folders = [base / f'{name}/lib/python3.12/site-packages' for name in ('env-tracking', 'env-vl')]
assert all(p.is_dir() for p in folders)
Path(sysconfig.get_path('purelib'), 'league_gpu_libraries.pth').write_text(''.join(str(p)+'\n' for p in folders))
PY
"$env/bin/python" -m pip install -r annotation/league/requirements-sam-finetune.txt
"$env/bin/python" -m pip freeze > "$pilot/installed.txt"
echo "Isolated training environment: $env"
