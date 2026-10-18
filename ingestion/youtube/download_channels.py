"""Catalog configured YouTube channels, then download all missing videos."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import shutil
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ingestion.common.replay_state import atomic_write_bytes, atomic_write_json, exclusive_lock, utc_now
from ingestion.youtube.download_videos import build_command, positive_int

DEFAULT_CHANNELS = Path(__file__).with_name("channels.json")
VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}\Z")
BOT_CHECK_PATTERNS = (
    re.compile(r"sign in to confirm.{0,30}not a bot", re.IGNORECASE),
    re.compile(r"captcha|verify (?:that )?you(?:'re| are) human", re.IGNORECASE),
    re.compile(r"http(?: error)? 429|too many requests", re.IGNORECASE),
)


class BotCheckDetected(RuntimeError):
    """YouTube requested verification or rate-limited the current run."""


def bot_check_message(output: str) -> str | None:
    for pattern in BOT_CHECK_PATTERNS:
        match = pattern.search(output)
        if match:
            return match.group(0)
    return None


def load_channels(path: Path) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if payload.get("schema_version", 1) != 1:
            raise ValueError("unsupported channel list schema_version")
        payload = payload.get("channels")
    if not isinstance(payload, list) or not payload:
        raise ValueError("channel JSON must contain a nonempty list or a 'channels' list")
    channels = []
    seen_ids: set[str] = set()
    seen_urls: set[str] = set()
    for entry in payload:
        if isinstance(entry, str):
            entry = {"url": entry}
        if not isinstance(entry, dict):
            raise ValueError("each channel must be a URL string or an object")
        enabled = entry.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError("channel enabled must be true or false")
        if not enabled:
            continue
        url = entry.get("url")
        if not isinstance(url, str):
            raise ValueError("each enabled channel requires a URL")
        parsed = urlsplit(url)
        segments = parsed.path.strip("/").split("/")
        if (parsed.scheme != "https" or parsed.netloc not in ("www.youtube.com", "youtube.com")
                or not (segments[0].startswith("@") and len(segments[0]) > 1
                        or segments[0] in ("channel", "c", "user") and len(segments) >= 2)):
            raise ValueError(f"expected a public YouTube channel URL: {url}")
        url = url.rstrip("/")
        default_id = re.sub(r"[^A-Za-z0-9_-]", "_", segments[0][1:] if segments[0].startswith("@") else segments[1]).lower()
        channel_id = entry.get("id", default_id)
        if not isinstance(channel_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", channel_id):
            raise ValueError("channel id must be a safe directory name containing letters, numbers, '-' or '_'")
        if channel_id in seen_ids or url in seen_urls:
            raise ValueError(f"duplicate channel id or URL: {channel_id}")
        seen_ids.add(channel_id)
        seen_urls.add(url)
        channels.append({"id": channel_id, "name": str(entry.get("name", channel_id)), "url": url})
    return channels


def discover(channel: dict[str, str], limit: int | None) -> dict[str, Any]:
    command = [
        sys.executable, "-m", "yt_dlp", "--ignore-config", "--flat-playlist",
        "--dump-single-json", "--skip-download", "--abort-on-error",
        "--socket-timeout", "30", "--retries", "3", "--sleep-requests", "1",
    ]
    if limit is not None:
        command += ["--playlist-end", str(limit)]
    # Keep JSON on stdout separate from diagnostics and stop cleanly on challenges.
    result = subprocess.run(command + ["--", channel["url"]], check=False,
                            capture_output=True, text=True)
    diagnostic = result.stderr or ""
    if diagnostic:
        print(diagnostic, end="" if diagnostic.endswith("\n") else "\n", file=sys.stderr)
    blocked = bot_check_message(diagnostic)
    if blocked:
        raise BotCheckDetected(f"YouTube verification/rate-limit response: {blocked}")
    if result.returncode:
        raise RuntimeError(f"channel discovery exited with status {result.returncode}: {diagnostic[-500:]}")
    payload = json.loads(result.stdout)
    videos: dict[str, dict[str, Any]] = {}

    def visit(item: Any) -> None:
        if not isinstance(item, dict):
            raise ValueError("channel discovery contains an invalid entry")
        if "entries" in item:
            if not isinstance(item["entries"], list):
                raise ValueError("channel discovery contains an invalid playlist")
            for child in item["entries"]:
                visit(child)
            return
        video_id = item.get("id")
        if not isinstance(video_id, str) or not VIDEO_ID.fullmatch(video_id):
            raise ValueError("channel discovery contains an invalid video ID")
        videos[video_id] = {
            "id": video_id, "title": item.get("title"),
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "duration": item.get("duration"), "live_status": item.get("live_status"),
        }

    if not isinstance(payload, dict) or "entries" not in payload:
        raise ValueError("channel discovery did not return a playlist")
    visit(payload)
    return {"schema_version": 1, "channel": channel, "scanned_at": utc_now(),
            "entry_limit": limit, "video_count": len(videos), "videos": list(videos.values())}


def load_catalog(path: Path, channel: dict[str, str]) -> dict[str, Any]:
    catalog = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(catalog, dict) or catalog.get("schema_version") != 1
            or not isinstance(catalog.get("channel"), dict)
            or any(catalog["channel"].get(key) != channel[key] for key in ("id", "url"))
            or not isinstance(catalog.get("videos"), list)):
        raise ValueError(f"invalid catalog or changed channel URL: {path}; refresh it first")
    seen: set[str] = set()
    for video in catalog["videos"]:
        if (not isinstance(video, dict) or not isinstance(video.get("id"), str)
                or not VIDEO_ID.fullmatch(video["id"]) or video["id"] in seen):
            raise ValueError(f"invalid or duplicate catalog video ID: {path}")
        seen.add(video["id"])
        video["url"] = f"https://www.youtube.com/watch?v={video['id']}"
        video.setdefault("title", video["id"])
    catalog["video_count"] = len(catalog["videos"])
    return catalog


def read_archive(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def write_archive(path: Path, entries: set[str]) -> None:
    # The enclosing output lock serializes this with yt-dlp's archive writes.
    atomic_write_bytes(path, ("\n".join(sorted(entries)) + "\n").encode("utf-8"))


def valid_existing_video(path: Path, video: dict[str, Any]) -> bool:
    """Validate files without a completion record, e.g. after an interrupted merge."""
    expected = video.get("duration")
    metadata_path = path.with_suffix(".info.json")
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(metadata, dict):
                expected = metadata.get("duration", expected)
        except (ValueError, OSError):
            pass
    if not isinstance(expected, (int, float)) or not math.isfinite(expected) or expected <= 0:
        return False
    result = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type",
        "-of", "json", str(path),
    ], check=False, capture_output=True, text=True)
    if result.returncode:
        return False
    try:
        info = json.loads(result.stdout)
        duration = float(info["format"]["duration"])
        return (math.isfinite(duration) and abs(duration - expected) <= 2
                and any(stream.get("codec_type") == "video" for stream in info["streams"]))
    except (KeyError, ValueError, TypeError):
        return False


def download_catalog(catalog: dict[str, Any], output: Path, max_height: int,
                     remaining: int | None) -> dict[str, Any]:
    archive_path = output / "downloaded.txt"
    archive = read_archive(archive_path)
    result: dict[str, Any] = {"skipped": 0, "downloaded": 0, "attempted": 0, "failed": [], "deferred": 0,
                              "bot_check": None}
    for video in catalog["videos"]:
        video_id = video["id"]
        key = f"youtube {video_id}"
        path = output / f"{video_id}.mp4"
        exists = path.is_file() and path.stat().st_size > 0
        if exists and (key in archive or valid_existing_video(path, video)):
            if key not in archive:
                archive.add(key)
                write_archive(archive_path, archive)
            result["skipped"] += 1
            continue
        if key in archive:
            archive.remove(key)
            write_archive(archive_path, archive)
        if remaining is not None and result["attempted"] >= remaining:
            result["deferred"] += 1
            continue
        if video.get("live_status") in ("is_live", "is_upcoming"):
            result["deferred"] += 1
            continue
        if path.is_file():
            # Do not let yt-dlp adopt an incomplete final file as already downloaded.
            path.replace(path.with_suffix(".mp4.incomplete"))
        print(f"[{catalog['channel']['id']}] Downloading {video_id}: {video['title']}", flush=True)
        args = argparse.Namespace(source=video["url"], output_dir=output, limit=1,
                                  max_height=max_height, list=False, simulate=False)
        result["attempted"] += 1
        command = build_command(args)
        # Flat listings can omit live status; guard again after full extraction
        # so an active broadcast cannot occupy the batch indefinitely.
        command = command[:-2] + ["--match-filters", "!is_live"] + command[-2:]
        process = subprocess.run(command, check=False, capture_output=True, text=True)
        process_output = (process.stdout or "") + (process.stderr or "")
        if process_output:
            print(process_output, end="" if process_output.endswith("\n") else "\n")
        blocked = bot_check_message(process_output)
        status = process.returncode
        archive = read_archive(archive_path)
        if status == 0 and key in archive and path.is_file() and path.stat().st_size > 0:
            result["downloaded"] += 1
        else:
            result["failed"].append(video_id)
            print(f"[{catalog['channel']['id']}] Failed {video_id}; it will be retried on a later run.", file=sys.stderr)
        if blocked:
            result["bot_check"] = blocked
            print(f"[{catalog['channel']['id']}] Stopping after YouTube verification/rate-limit response: {blocked}",
                  file=sys.stderr)
            break
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channels", type=Path, default=DEFAULT_CHANNELS, help="JSON list of channels")
    parser.add_argument("--output-root", type=Path, default=Path("data/youtube"))
    parser.add_argument("--catalog-only", action="store_true", help="refresh channel catalogs without downloading")
    parser.add_argument("--use-catalog", action="store_true", help="download from saved catalogs without rescanning channels")
    parser.add_argument("--limit", type=positive_int, help="optional discovery entry limit per playlist; default: all")
    parser.add_argument("--max-downloads", type=positive_int, help="optional limit on video attempts across this run; default: all")
    parser.add_argument("--max-height", type=positive_int, default=1080)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.use_catalog and (args.catalog_only or args.limit is not None):
        parser.error("--use-catalog cannot be combined with --catalog-only or --limit")
    try:
        channels = load_channels(args.channels)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if importlib.util.find_spec("yt_dlp") is None:
        parser.error("install dependencies: python -m pip install -r ingestion/youtube/requirements.txt")
    if not args.catalog_only:
        if not (shutil.which("node") or shutil.which("deno")):
            parser.error("YouTube extraction requires Node.js or Deno on PATH")
        if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
            parser.error("video downloading requires ffmpeg and ffprobe on PATH")
    try:
        # Hold all channel locks through discovery and downloading, including the
        # original single-source downloader's lock, to protect shared archives.
        with ExitStack() as locks:
            for channel in sorted(channels, key=lambda item: item["id"]):
                locks.enter_context(exclusive_lock(args.output_root / channel["id"] / ".download.lock"))
            catalogs = []
            failures: dict[str, str] = {}
            bot_check_channel: str | None = None
            for channel in channels:
                print(f"[{channel['id']}] {'Loading saved catalog' if args.use_catalog else 'Cataloging ' + channel['url']} ...", flush=True)
                try:
                    catalog_path = args.output_root / channel["id"] / "catalog.json"
                    if args.use_catalog:
                        catalog = load_catalog(catalog_path, channel)
                    else:
                        catalog = discover(channel, args.limit)
                        atomic_write_json(catalog_path, catalog)
                    catalogs.append(catalog)
                    print(f"[{channel['id']}] Found {catalog['video_count']} videos.", flush=True)
                except (OSError, ValueError, RuntimeError) as error:
                    failures[channel["id"]] = str(error)
                    print(f"[{channel['id']}] Discovery failed: {error}", file=sys.stderr)
                    if isinstance(error, BotCheckDetected):
                        bot_check_channel = channel["id"]
                        break
            outcomes: dict[str, dict[str, Any]] = {}
            remaining = args.max_downloads
            if not args.catalog_only and bot_check_channel is None:
                for catalog in catalogs:
                    channel_id = catalog["channel"]["id"]
                    outcome = download_catalog(catalog, args.output_root / channel_id, args.max_height, remaining)
                    outcomes[channel_id] = outcome
                    if remaining is not None:
                        remaining -= outcome["attempted"]
                    print(f"[{channel_id}] {outcome['downloaded']} downloaded, {outcome['skipped']} already saved, "
                          f"{len(outcome['failed'])} failed, {outcome['deferred']} deferred.", flush=True)
                    if outcome.get("bot_check"):
                        bot_check_channel = channel_id
                        break
            # Per-channel summaries avoid conflicting runs using disjoint channel lists.
            for channel in channels:
                channel_id = channel["id"]
                atomic_write_json(args.output_root / channel_id / "last_run.json", {
                    "finished_at": utc_now(), "catalog_only": args.catalog_only,
                    "discovery_error": failures.get(channel_id), **outcomes.get(channel_id, {}),
                    "bot_check_channel": bot_check_channel,
                })
            return int(bool(bot_check_channel or failures or any(outcome["failed"] for outcome in outcomes.values())))
    except KeyboardInterrupt:
        return 130
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Channel download failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
