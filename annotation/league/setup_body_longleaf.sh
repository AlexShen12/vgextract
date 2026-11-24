#!/bin/bash
# A new environment only. Existing SAM/Qwen environments and pins remain unchanged.
set -euo pipefail
base=/work/users/a/l/alshen/league-midbeast-validation
target="$base/env-body"
export LD_LIBRARY_PATH="/work/users/a/l/alshen/conda_envs/videogame-league/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
if [[ ! -x "$target/bin/python" ]]; then "$base/env-vl/bin/python" -m venv "$target"; fi
"$target/bin/python" - "$base" <<'PY'
from pathlib import Path
import sys,sysconfig
base=Path(sys.argv[1])
paths=[base/'env-tracking/lib/python3.12/site-packages',base/'env-vl/lib/python3.12/site-packages']
if not all(p.is_dir() for p in paths):raise RuntimeError('Existing read-only GPU dependencies are missing')
Path(sysconfig.get_path('purelib'),'league_body_dependencies.pth').write_text('\n'.join(map(str,paths))+'\n')
PY
"$target/bin/python" -B -m pip freeze > "$target/installed.txt"
"$target/bin/python" -B -c 'import torch, numpy, sam3; print(torch.__version__, numpy.__version__)'
