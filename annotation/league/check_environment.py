"""Fail before installation or inference if the Longleaf prefix is misconfigured."""
from __future__ import annotations

import argparse
from importlib.util import find_spec
from pathlib import Path
import sys
import sysconfig
import tempfile


def check_environment(prefix: Path, *, writable: bool = False) -> None:
    prefix = prefix.resolve()
    if Path(sys.prefix).resolve() != prefix:
        raise RuntimeError(f"Expected Conda prefix {prefix}, but Python is using {sys.prefix}")
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            f"The inference lock requires Python 3.12; found {sys.version.split()[0]} "
            f"at {sys.executable}"
        )
    pip = find_spec("pip")
    if pip is None or not pip.origin or not Path(pip.origin).resolve().is_relative_to(prefix):
        raise RuntimeError(f"pip must be installed inside {prefix}; refusing to use an external pip")
    if writable:
        packages = Path(sysconfig.get_path("purelib"))
        if not packages.resolve().is_relative_to(prefix):
            raise RuntimeError(f"site-packages is outside the requested environment: {packages}")
        try:
            with tempfile.TemporaryFile(dir=packages):
                pass
        except OSError as exc:
            raise RuntimeError(f"Environment site-packages is not writable: {packages}: {exc}") from exc
    print(f"Python interpreter: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Conda prefix: {prefix}")
    print(f"Environment pip: {pip.origin}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", type=Path)
    parser.add_argument("--writable", action="store_true")
    args = parser.parse_args()
    try:
        check_environment(args.prefix, writable=args.writable)
    except RuntimeError as exc:
        parser.exit(2, f"Environment check failed: {exc}\n"
                    "Rerun bash annotation/league/setup_longleaf.sh. "
                    "If the prefix is not writable, set LEAGUE_ENV to a directory you own "
                    "for both setup and job submission.\n")


if __name__ == "__main__":
    main()
