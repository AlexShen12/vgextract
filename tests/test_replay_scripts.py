from __future__ import annotations

import bz2
import subprocess
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from collect_pro_replays import API_ROOT, PRO_MATCHES_URL, Collector, empty_manifest
from download_replays import Downloader
from replay_state import SCHEMA_VERSION, atomic_write_json, load_json, parse_utc


class FakeClock:
    def __init__(self) -> None:
        self.value = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.sleeps: list[float] = []

    def now(self) -> datetime:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.value += timedelta(seconds=seconds)


class CollectorTests(unittest.TestCase):
    def test_snapshot_ids_are_the_only_new_records_and_are_paced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "pro_replays.json"
            clock = FakeClock()
            calls: list[str] = []

            def fetch(url: str):
                calls.append(url)
                if url == PRO_MATCHES_URL:
                    return [{"match_id": 200}, {"match_id": 100}, {"match_id": 200}]
                if url == f"{API_ROOT}/matches/200":
                    return {"match_id": 200, "replay_url": "https://example.test/200.dem.bz2"}
                if url == f"{API_ROOT}/matches/100":
                    return {"match_id": 100, "replay_url": None}
                self.fail(f"unexpected request {url}")

            result = Collector(
                manifest_path, fetch=fetch, sleep=clock.sleep, now=clock.now, interval_seconds=60
            ).run()

            self.assertEqual(result, {"snapshot_count": 2, "processed_to_terminal": 2})
            self.assertEqual(calls, [PRO_MATCHES_URL, f"{API_ROOT}/matches/200", f"{API_ROOT}/matches/100"])
            self.assertEqual(clock.sleeps, [60.0])
            manifest = load_json(manifest_path, {})
            self.assertEqual(manifest["source"]["last_snapshot_count"], 2)
            self.assertEqual(manifest["matches"]["200"]["lookup"]["status"], "resolved")
            self.assertEqual(manifest["matches"]["100"]["lookup"]["status"], "unavailable")

    def test_error_retries_the_same_match_after_a_minute(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "pro_replays.json"
            clock = FakeClock()
            detail_attempts = 0

            def fetch(url: str):
                nonlocal detail_attempts
                if url == PRO_MATCHES_URL:
                    return [{"match_id": 300}]
                detail_attempts += 1
                if detail_attempts == 1:
                    raise OSError("temporary network failure")
                return {"match_id": 300, "replay_url": "https://example.test/300.dem.bz2"}

            Collector(
                manifest_path, fetch=fetch, sleep=clock.sleep, now=clock.now, interval_seconds=60
            ).run()

            manifest = load_json(manifest_path, {})
            lookup = manifest["matches"]["300"]["lookup"]
            self.assertEqual(detail_attempts, 2)
            self.assertEqual(clock.sleeps, [60.0])
            self.assertEqual(lookup["attempts"], 2)
            self.assertEqual(lookup["status"], "resolved")

    def test_interruption_leaves_pacing_state_for_a_safe_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "pro_replays.json"
            clock = FakeClock()

            def interrupted_fetch(url: str):
                if url == PRO_MATCHES_URL:
                    return [{"match_id": 400}]
                raise KeyboardInterrupt()

            with self.assertRaises(KeyboardInterrupt):
                Collector(
                    manifest_path,
                    fetch=interrupted_fetch,
                    sleep=clock.sleep,
                    now=clock.now,
                    interval_seconds=60,
                ).run()

            manifest = load_json(manifest_path, {})
            self.assertEqual(manifest["matches"]["400"]["lookup"]["status"], "pending")
            deadline = parse_utc(manifest["request_pacing"]["next_detail_attempt_at"])
            self.assertEqual((deadline - clock.now()).total_seconds(), 60)


class DownloaderTests(unittest.TestCase):
    def _manifest(self, path: Path, match_id: int, replay_url: str) -> None:
        manifest = empty_manifest()
        manifest["matches"][str(match_id)] = {
            "match_id": match_id,
            "lookup": {"status": "resolved", "replay_url": replay_url},
        }
        atomic_write_json(path, manifest)

    def test_downloads_decompresses_and_skips_completed_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "pro_replays.json"
            state_path = root / "replay_downloads.json"
            output_dir = root / "replays"
            self._manifest(manifest_path, 500, "https://example.test/500.dem.bz2")
            compressed = bz2.compress(b"fake replay bytes")
            commands: list[list[str]] = []

            def runner(command, **kwargs):
                commands.append(command)
                if command[0] == "curl":
                    target = Path(command[command.index("--output") + 1])
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(compressed)
                    return subprocess.CompletedProcess(command, 0, "", "")
                return subprocess.run(command, **kwargs)

            downloader = Downloader(manifest_path, state_path, output_dir, runner=runner)
            result = downloader.run()
            self.assertEqual(result, {"candidates": 1, "completed": 1, "failed": 0})
            self.assertEqual((output_dir / "500.dem").read_bytes(), b"fake replay bytes")
            self.assertTrue((output_dir / "500.dem.bz2").exists())
            self.assertFalse((output_dir / "500.dem.bz2.part").exists())
            self.assertFalse((output_dir / "500.dem.part").exists())
            self.assertEqual(load_json(state_path, {})["downloads"]["500"]["status"], "complete")

            def no_network(*args, **kwargs):
                self.fail("a completed replay must not be downloaded again")

            second_result = Downloader(manifest_path, state_path, output_dir, runner=no_network).run()
            self.assertEqual(second_result, {"candidates": 1, "completed": 1, "failed": 0})

    def test_failed_download_keeps_partial_file_for_a_later_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "pro_replays.json"
            state_path = root / "replay_downloads.json"
            output_dir = root / "replays"
            self._manifest(manifest_path, 600, "https://example.test/600.dem.bz2")

            def failed_curl(command, **kwargs):
                target = Path(command[command.index("--output") + 1])
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"partial archive")
                return subprocess.CompletedProcess(command, 22, "", "HTTP 500")

            result = Downloader(manifest_path, state_path, output_dir, runner=failed_curl).run()
            self.assertEqual(result, {"candidates": 1, "completed": 0, "failed": 1})
            self.assertTrue((output_dir / "600.dem.bz2.part").exists())
            state = load_json(state_path, {})
            self.assertEqual(state["downloads"]["600"]["status"], "error")
            self.assertIn("curl failed", state["downloads"]["600"]["last_error"])


if __name__ == "__main__":
    unittest.main()
