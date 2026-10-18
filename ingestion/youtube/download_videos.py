"""Download a bounded, resumable slice of DOTA gameplay using yt-dlp."""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

from ingestion.common.replay_state import exclusive_lock

CHANNEL_URL = "https://www.youtube.com/@ProGameplayDota2/videos"
DEFAULT_OUTPUT = Path("data/youtube/pro_gameplay_dota2")


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=CHANNEL_URL, help="channel, playlist, or video URL")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=positive_int, default=1, help="inspect the first N channel entries (default: 1)")
    parser.add_argument("--max-height", type=positive_int, default=1080, help="maximum video height (default: 1080)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true", help="print channel metadata as JSON lines without downloading")
    mode.add_argument("--simulate", action="store_true", help="resolve formats without writing files or archive entries")
    return parser


def build_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable, "-m", "yt_dlp", "--ignore-config",
        "--playlist-end", str(args.limit),
        "--socket-timeout", "30", "--retries", "3", "--fragment-retries", "3",
    ]
    if args.list:
        command += ["--flat-playlist", "--dump-json", "--skip-download"]
    else:
        # Node must be explicitly enabled in yt-dlp; Deno is enabled by default.
        if shutil.which("node"):
            command += ["--js-runtimes", "node"]
        command += [
            "--format", f"bv*[height<={args.max_height}]+ba/b[height<={args.max_height}]",
            "--format-sort", "res,fps,vcodec:h264,acodec:aac",
            "--merge-output-format", "mp4", "--remux-video", "mp4",
        ]
        if args.simulate:
            command += ["--simulate", "--print", "%(id)s | %(title)s | %(duration_string)s | %(format)s"]
        else:
            command += [
                "--paths", str(args.output_dir), "--output", "%(id)s.%(ext)s",
                "--write-info-json", "--download-archive", str(args.output_dir / "downloaded.txt"),
                "--continue", "--sleep-requests", "1", "--sleep-interval", "5",
            ]
    return command + ["--", args.source]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if importlib.util.find_spec("yt_dlp") is None:
        parser.error("install dependencies: python -m pip install -r ingestion/youtube/requirements.txt")
    if not args.list:
        if not (shutil.which("node") or shutil.which("deno")):
            parser.error("YouTube extraction requires Node.js or Deno on PATH")
        if not args.simulate and not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
            parser.error("video downloading requires ffmpeg and ffprobe on PATH")
    command = build_command(args)
    try:
        if args.list or args.simulate:
            return subprocess.run(command, check=False).returncode
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with exclusive_lock(args.output_dir / ".download.lock"):
            return subprocess.run(command, check=False).returncode
    except KeyboardInterrupt:
        return 130
    except (OSError, RuntimeError) as error:
        print(f"Download failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
