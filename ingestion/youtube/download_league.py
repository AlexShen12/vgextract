"""Catalog or download the first 100 videos from verified DomisumReplay channels."""

from __future__ import annotations

import sys
from pathlib import Path

from ingestion.youtube.download_channels import main as download_channels_main

DEFAULT_CHANNELS = Path(__file__).with_name("league_channels.json")


def main(argv: list[str] | None = None) -> int:
    """Run the shared downloader with League's channel list and 100-video window."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    defaults = ["--channels", str(DEFAULT_CHANNELS)]
    if "--use-catalog" not in arguments:
        defaults += ["--limit", "100"]
    return download_channels_main([*defaults, *arguments])


if __name__ == "__main__":
    raise SystemExit(main())
