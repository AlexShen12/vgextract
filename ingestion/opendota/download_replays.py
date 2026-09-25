#!/usr/bin/env python3
"""Download and decompress replay URLs resolved by collect_pro_replays.py."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from ingestion.common.replay_state import SCHEMA_VERSION, atomic_write_json, exclusive_lock, load_json, utc_now


def empty_download_state() -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "downloads": {}}


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported or missing collector manifest schema_version")
    if not isinstance(manifest.get("matches"), dict):
        raise ValueError("collector manifest matches must be an object")


def _validate_download_state(state: dict[str, Any]) -> None:
    if state.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported or missing download-state schema_version")
    if not isinstance(state.get("downloads"), dict):
        raise ValueError("download-state downloads must be an object")


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _error_text(result: subprocess.CompletedProcess[Any]) -> str:
    text = (_text(result.stderr) or _text(result.stdout) or f"exit status {result.returncode}").strip()
    return text[-2_000:]


class Downloader:
    def __init__(
        self,
        manifest_path: Path,
        state_path: Path,
        output_dir: Path,
        *,
        runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.manifest_path = manifest_path
        self.state_path = state_path
        self.output_dir = output_dir
        self.runner = runner
        self.log = log or (lambda message: print(message, file=sys.stderr, flush=True))
        self.state: dict[str, Any] = {}

    def _save(self) -> None:
        atomic_write_json(self.state_path, self.state)

    def _move_stale(self, path: Path) -> None:
        if not path.exists():
            return
        stale = path.with_name(path.name + ".stale")
        suffix = 1
        while stale.exists():
            stale = path.with_name(f"{path.name}.stale.{suffix}")
            suffix += 1
        path.replace(stale)
        self.log(f"[Replay state] moved stale file {path} to {stale}")

    def _record(self, match_id: int, replay_url: str) -> dict[str, Any]:
        key = str(match_id)
        downloads = self.state["downloads"]
        record = downloads.get(key)
        archive = self.output_dir / f"{match_id}.dem.bz2"
        replay = self.output_dir / f"{match_id}.dem"
        if record is None:
            record = {
                "match_id": match_id,
                "replay_url": replay_url,
                "archive_path": str(archive),
                "replay_path": str(replay),
                "status": "pending",
                "download_attempts": 0,
                "decompression_attempts": 0,
                "last_error": None,
                "updated_at": utc_now(),
            }
            downloads[key] = record
        else:
            # A replay URL may be regenerated with a different salt; use the latest one.
            previous_url = record.get("replay_url")
            previous_archive_url = record.get("archive_replay_url")
            if (previous_url and previous_url != replay_url) or (
                previous_archive_url and previous_archive_url != replay_url
            ):
                self.log(
                    f"[Replay state] match_id={match_id} replay URL changed; "
                    "invalidating old archive/output"
                )
                self._move_stale(archive)
                self._move_stale(archive.with_name(archive.name + ".part"))
                self._move_stale(replay)
                self._move_stale(replay.with_name(replay.name + ".part"))
                record["status"] = "pending"
                record["archive_replay_url"] = None
            record["replay_url"] = replay_url
            record["archive_path"] = str(archive)
            record["replay_path"] = str(replay)
        return record

    def _set_status(self, record: dict[str, Any], status: str, error: str | None = None) -> None:
        record["status"] = status
        record["last_error"] = error
        record["updated_at"] = utc_now()
        self._save()

    def _download_archive(self, record: dict[str, Any], archive: Path) -> bool:
        partial = archive.with_name(archive.name + ".part")
        record["download_attempts"] += 1
        self._set_status(record, "downloading")
        self.log(
            f"[Replay download] match_id={record['match_id']} GET {record['replay_url']} "
            f"archive={archive} partial={partial} attempt={record['download_attempts']}"
        )
        result = self.runner(
            [
                "curl",
                "--fail",
                "--location",
                "--continue-at",
                "-",
                "--output",
                str(partial),
                record["replay_url"],
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.log(
            f"[Replay download] match_id={record['match_id']} curl_exit={result.returncode} "
            f"curl_stderr={_text(result.stderr).strip()!r}"
        )
        if result.returncode != 0:
            self._set_status(record, "error", f"curl failed: {_error_text(result)}")
            return False
        if not partial.exists():
            self._set_status(record, "error", "curl returned success but did not create the partial archive")
            return False
        partial.replace(archive)
        record["archive_replay_url"] = record["replay_url"]
        self.log(
            f"[Replay download] match_id={record['match_id']} archive_complete "
            f"bytes={archive.stat().st_size} path={archive}"
        )
        self._set_status(record, "downloaded")
        return True

    def _decompress_archive(self, record: dict[str, Any], archive: Path, replay: Path) -> bool:
        partial = replay.with_name(replay.name + ".part")
        record["decompression_attempts"] += 1
        self._set_status(record, "decompressing")
        self.log(
            f"[Replay decompress] match_id={record['match_id']} archive={archive} "
            f"output={replay} attempt={record['decompression_attempts']}"
        )
        with partial.open("wb") as output:
            result = self.runner(
                ["bzip2", "-dc", str(archive)],
                stdout=output,
                stderr=subprocess.PIPE,
                text=False,
                check=False,
            )
        self.log(
            f"[Replay decompress] match_id={record['match_id']} bzip2_exit={result.returncode} "
            f"bzip2_stderr={_text(result.stderr).strip()!r}"
        )
        if result.returncode != 0:
            self._set_status(record, "error", f"bzip2 failed: {_error_text(result)}")
            return False
        partial.replace(replay)
        self.log(
            f"[Replay decompress] match_id={record['match_id']} complete "
            f"bytes={replay.stat().st_size} path={replay}"
        )
        self._set_status(record, "complete")
        return True

    def _process_one(self, match_id: int, replay_url: str) -> bool:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        record = self._record(match_id, replay_url)
        archive = self.output_dir / f"{match_id}.dem.bz2"
        replay = self.output_dir / f"{match_id}.dem"
        self.log(
            f"[Replay candidate] match_id={match_id} replay_url={replay_url} "
            f"archive={archive} output={replay} state={record['status']}"
        )
        if replay.exists() and replay.stat().st_size > 0:
            self.log(
                f"[Replay candidate] match_id={match_id} already complete "
                f"bytes={replay.stat().st_size}"
            )
            self._set_status(record, "complete")
            return True
        self._save()
        if not archive.exists() and not self._download_archive(record, archive):
            return False
        return self._decompress_archive(record, archive, replay)

    def run(self) -> dict[str, int]:
        manifest = load_json(self.manifest_path, {})
        _validate_manifest(manifest)
        self.state = load_json(self.state_path, empty_download_state())
        _validate_download_state(self.state)

        candidates: list[tuple[int, str]] = []
        for entry in manifest["matches"].values():
            lookup = entry.get("lookup", {})
            replay_url = lookup.get("replay_url")
            if lookup.get("status") == "resolved" and isinstance(replay_url, str) and replay_url:
                candidates.append((entry["match_id"], replay_url))
        candidates.sort(key=lambda item: item[0])
        self.log(f"[Replay downloader] resolved candidates={len(candidates)} manifest={self.manifest_path}")

        completed = failed = 0
        for match_id, replay_url in candidates:
            if self._process_one(match_id, replay_url):
                completed += 1
            else:
                failed += 1
        return {"candidates": len(candidates), "completed": completed, "failed": failed}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/pro_replays.json"),
        help="collector manifest (default: data/pro_replays.json)",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=Path("data/replay_downloads.json"),
        help="durable downloader state (default: data/replay_downloads.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/replays"),
        help="directory for .dem.bz2 archives and decompressed .dem files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        with exclusive_lock(args.state.with_suffix(args.state.suffix + ".lock")):
            result = Downloader(args.manifest, args.state, args.output_dir).run()
    except KeyboardInterrupt:
        print("Interrupted; partial downloads and state can be resumed by rerunning this command.", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Downloader failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(
        f"Processed {result['candidates']} replay URLs: "
        f"{result['completed']} complete, {result['failed']} failed (safe to retry)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
