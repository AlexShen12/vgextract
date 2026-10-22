from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from annotation.league.check_environment import check_environment


ROOT = Path(__file__).resolve().parents[1]


class EnvironmentCheckTests(unittest.TestCase):
    def test_old_python_is_rejected_before_pip(self):
        with patch("annotation.league.check_environment.sys.version_info", (3, 11)), \
             patch("annotation.league.check_environment.find_spec") as spec:
            with self.assertRaisesRegex(RuntimeError, "requires Python 3.12"):
                check_environment(Path(sys.prefix))
            spec.assert_not_called()

    def test_wrong_prefix_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Expected Conda prefix"):
            check_environment(Path(sys.prefix)/"other-prefix")

    def test_missing_or_external_pip_is_rejected(self):
        for spec in (None, SimpleNamespace(origin="/external/pip/__init__.py")):
            with self.subTest(spec=spec), \
                 patch("annotation.league.check_environment.sys.version_info", (3, 12)), \
                 patch("annotation.league.check_environment.find_spec", return_value=spec):
                with self.assertRaisesRegex(RuntimeError, "pip must be installed inside"):
                    check_environment(Path(sys.prefix))

    def test_writability_is_checked_only_for_setup(self):
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary)
            spec = SimpleNamespace(origin=str(prefix/"pip/__init__.py"))
            with patch("annotation.league.check_environment.sys.prefix", str(prefix)), \
                 patch("annotation.league.check_environment.sys.version_info", (3, 12)), \
                 patch("annotation.league.check_environment.find_spec", return_value=spec), \
                 patch("annotation.league.check_environment.sysconfig.get_path", return_value=temporary), \
                 patch("annotation.league.check_environment.tempfile.TemporaryFile",
                       side_effect=PermissionError("read-only")) as probe, \
                 contextlib.redirect_stdout(io.StringIO()):
                check_environment(prefix)
                probe.assert_not_called()
                with self.assertRaisesRegex(RuntimeError, "site-packages is not writable"):
                    check_environment(prefix, writable=True)

    def test_valid_writable_prefix_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary)
            spec = SimpleNamespace(origin=str(prefix/"pip/__init__.py"))
            with patch("annotation.league.check_environment.sys.prefix", str(prefix)), \
                 patch("annotation.league.check_environment.sys.version_info", (3, 12)), \
                 patch("annotation.league.check_environment.find_spec", return_value=spec), \
                 patch("annotation.league.check_environment.sysconfig.get_path", return_value=temporary), \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                check_environment(prefix, writable=True)
            self.assertIn("Python interpreter:", output.getvalue())
            self.assertEqual(list(prefix.iterdir()), [])


class LongleafShellTests(unittest.TestCase):
    """Run the real shell entry points with local Conda/module/Python stand-ins."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="league setup ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bin = self.root/"bin"
        self.bin.mkdir()
        self.prefix = self.root/"environment"
        self.log = self.root/"calls.jsonl"
        self.stub = self.root/"python-stub"
        self.executable(self.stub, f"#!{sys.executable}\n" + '''
import json, os, sys
assert os.environ.get("PYTHONNOUSERSITE") == "1"
assert "PYTHONPATH" not in os.environ
assert "PYTHONHOME" not in os.environ
with open(os.environ["CALL_LOG"], "a") as log:
    log.write(json.dumps(["python", *sys.argv[1:]]) + "\\n")
if "annotation/league/check_environment.py" in sys.argv:
    sys.exit(int(os.environ.get("GUARD_STATUS", "0")))
''')
        self.executable(self.bin/"conda", f"#!{sys.executable}\n" + '''
import json, os, pathlib, shutil, sys
if sys.argv[1:] == ["info", "--base"]:
    print(os.environ["FAKE_CONDA_BASE"])
else:
    with open(os.environ["CALL_LOG"], "a") as log:
        log.write(json.dumps(["conda", *sys.argv[1:]]) + "\\n")
    prefix = pathlib.Path(sys.argv[sys.argv.index("-p") + 1])
    (prefix / "bin").mkdir(parents=True, exist_ok=True)
    shutil.copy2(os.environ["PYTHON_STUB"], prefix / "bin/python")
''')
        hook = self.root/"conda-base/etc/profile.d/conda.sh"
        hook.parent.mkdir(parents=True)
        # Real Conda hooks may reference unset variables. An unrelated `python`
        # function exercises the use of explicit interpreter paths.
        hook.write_text('''
unused="${UNSET_CONDA_HOOK_VARIABLE}"
python() { return 99; }
conda() {
    if [[ "$1" == activate ]]; then
        unused="${UNSET_CONDA_ACTIVATE_VARIABLE}"
        export PATH="${LEAGUE_ENV}/bin:${PATH}"
        export PYTHONPATH=/unrelated/python/packages
    else
        command conda "$@"
    fi
}
''')
        for name in ("module", "ffmpeg", "ffprobe"):
            self.executable(self.bin/name, "#!/bin/bash\nexit 0\n")
        video = self.root/"video.mp4"
        video.touch()
        self.env = {**os.environ, "PATH": f"{self.bin}:/usr/bin:/bin",
                    "LEAGUE_ENV": str(self.prefix), "LEAGUE_VIDEO": str(video),
                    "LEAGUE_HF_HOME": str(self.root/"cache"),
                    "LEAGUE_RUN_DIR": str(self.root/"run"),
                    "FAKE_CONDA_BASE": str(self.root/"conda-base"),
                    "PYTHON_STUB": str(self.stub), "CALL_LOG": str(self.log),
                    "PYTHONPATH": "/inherited/python/packages", "PYTHONHOME": "/wrong/python",
                    "SLURM_SUBMIT_DIR": str(ROOT), "PIP_USER": "true"}

    @staticmethod
    def executable(path, source):
        path.write_text(source)
        path.chmod(0o755)

    def make_existing_prefix(self):
        (self.prefix/"bin").mkdir(parents=True)
        self.executable(self.prefix/"bin/python", self.stub.read_text())

    def run_script(self, name):
        return subprocess.run(["/bin/bash", str(ROOT/"annotation/league"/name)],
                              cwd=self.root, env=self.env, text=True, capture_output=True, timeout=10)

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def test_create_and_repair_install_python_pip_and_media_before_inference(self):
        for existing in (False, True):
            with self.subTest(existing=existing):
                if self.log.exists():
                    self.log.unlink()
                result = self.run_script("setup_longleaf.sh")
                self.assertEqual(result.returncode, 0, result.stderr)
                calls = self.calls()
                self.assertEqual(calls[0][:2], ["conda", "install" if existing else "create"])
                for requirement in ("python=3.12", "pip", "ffmpeg"):
                    self.assertIn(requirement, calls[0])
                self.assertEqual(calls[1], ["python", "annotation/league/check_environment.py",
                                            str(self.prefix), "--writable"])
                self.assertEqual(calls[2], ["python", "-m", "pip", "install", "--no-user", "-r",
                                            "annotation/league/inference-environment.lock.txt"])
                self.assertEqual(calls[-1][-1], "processor-preview")

    def test_setup_stops_before_pip_when_environment_check_fails(self):
        self.env["GUARD_STATUS"] = "2"
        result = self.run_script("setup_longleaf.sh")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(len(self.calls()), 2)

    def test_batch_checks_the_same_prefix_before_running(self):
        self.make_existing_prefix()
        result = self.run_script("slurm_annotate.sl")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls(), [
            ["python", "annotation/league/check_environment.py", str(self.prefix)],
            ["python", "-m", "annotation.league", "--run-dir", self.env["LEAGUE_RUN_DIR"],
             "--offline", "run", "--video", self.env["LEAGUE_VIDEO"], "--start", "0"]])

    def test_batch_stops_before_inference_when_environment_check_fails(self):
        self.make_existing_prefix()
        self.env["GUARD_STATUS"] = "2"
        result = self.run_script("slurm_annotate.sl")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(len(self.calls()), 1)


if __name__ == "__main__":
    unittest.main()
