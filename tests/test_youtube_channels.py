from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ingestion.common.replay_state import exclusive_lock
from ingestion.youtube.download_channels import (
    discover, download_catalog, load_catalog, load_channels, main, read_archive,
    valid_existing_video, write_archive,
)

A, B, C, D = "abcdefghijA", "abcdefghijB", "abcdefghijC", "abcdefghijD"
CHANNEL = {"id": "dota", "name": "Dota", "url": "https://www.youtube.com/@Dota"}


def catalog(ids: list[str], channel: dict | None = None) -> dict:
    return {"schema_version": 1, "channel": channel or CHANNEL, "video_count": len(ids),
            "videos": [{"id": video_id, "title": f"Title {video_id}", "duration": 60,
                        "url": f"https://www.youtube.com/watch?v={video_id}"} for video_id in ids]}


class ChannelDownloaderTests(unittest.TestCase):
    def setUp(self) -> None:
        dependency = patch("ingestion.youtube.download_channels.importlib.util.find_spec", return_value=object())
        executables = patch("shutil.which", side_effect=lambda name: f"/bin/{name}")
        dependency.start()
        executables.start()
        self.addCleanup(dependency.stop)
        self.addCleanup(executables.stop)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.config = self.root / "channels.json"
        self.config.write_text(json.dumps({"channels": [CHANNEL]}))
        self.output = self.root / "output"
        self.args = ["--channels", str(self.config), "--output-root", str(self.output)]

    def test_channel_list_accepts_urls_and_disabled_objects(self) -> None:
        self.config.write_text(json.dumps([CHANNEL["url"], {"enabled": False}]))
        self.assertEqual(load_channels(self.config)[0]["url"], CHANNEL["url"])

    def test_unsafe_or_duplicate_storage_ids_are_rejected(self) -> None:
        for channels in ([{**CHANNEL, "id": "../escape"}], [CHANNEL, CHANNEL]):
            with self.subTest(channels=channels):
                self.config.write_text(json.dumps(channels))
                with self.assertRaises(ValueError):
                    load_channels(self.config)

    def test_full_discovery_flattens_tabs_and_deduplicates_ids(self) -> None:
        payload = {"entries": [{"entries": [{"id": A, "title": "Original"}]},
                               {"entries": [{"id": A, "title": "Renamed"}, {"id": B, "title": "Second"}]}]}
        with patch("ingestion.youtube.download_channels.subprocess.run", return_value=subprocess.CompletedProcess([], 0, json.dumps(payload))) as run:
            result = discover(CHANNEL, None)
        self.assertEqual([video["id"] for video in result["videos"]], [A, B])
        self.assertEqual(result["videos"][0]["title"], "Renamed")
        self.assertIsNone(result["entry_limit"])
        self.assertNotIn("--playlist-end", run.call_args.args[0])

    def test_reruns_skip_renamed_videos_retry_failures_and_restore_deleted_files(self) -> None:
        output = self.output / "dota"
        calls = []
        fail_once = {C}

        def download(command, **kwargs):
            video_id = command[-1].split("v=")[1]
            calls.append(video_id)
            if video_id in fail_once:
                fail_once.remove(video_id)
                (output / f"{video_id}.f299.mp4.part").write_bytes(b"partial")
                return subprocess.CompletedProcess(command, 1)
            (output / f"{video_id}.mp4").write_bytes(b"complete fixture")
            archive = read_archive(output / "downloaded.txt")
            archive.add(f"youtube {video_id}")
            write_archive(output / "downloaded.txt", archive)
            return subprocess.CompletedProcess(command, 0)

        first = catalog([A, B, C])
        second = catalog([A, B, C, D])
        second["videos"][0]["title"] = "Title changed since first run"
        with patch("ingestion.youtube.download_channels.discover", side_effect=[first, second]) as scan, patch("ingestion.youtube.download_channels.subprocess.run", side_effect=download):
            self.assertEqual(main(self.args), 1)
            self.assertNotIn(f"youtube {C}", read_archive(output / "downloaded.txt"))
            (output / f"{B}.mp4").unlink()
            self.assertEqual(main(self.args), 0)
            self.assertEqual(scan.call_args.args[1], None)
            self.assertEqual(main(self.args + ["--use-catalog"]), 0)
        self.assertEqual(calls, [A, B, C, B, C, D])
        outcome = json.loads((output / "last_run.json").read_text())
        self.assertEqual(outcome["skipped"], 4)
        self.assertEqual(outcome["attempted"], 0)

    def test_all_channels_are_cataloged_before_any_download(self) -> None:
        other = {"id": "other", "name": "Other", "url": "https://www.youtube.com/@Other"}
        self.config.write_text(json.dumps([CHANNEL, other]))

        def download(saved, *args):
            for channel in (CHANNEL, other):
                self.assertTrue((self.output / channel["id"] / "catalog.json").exists())
            return {"downloaded": 0, "skipped": 0, "attempted": 0, "failed": [], "deferred": 0}

        with patch("ingestion.youtube.download_channels.discover", side_effect=[catalog([A]), catalog([B], other)]), patch("ingestion.youtube.download_channels.download_catalog", side_effect=download):
            self.assertEqual(main(self.args), 0)

    def test_catalog_only_does_not_start_downloads_and_preserves_archive(self) -> None:
        output = self.output / "dota"
        output.mkdir(parents=True)
        write_archive(output / "downloaded.txt", {f"youtube {A}"})
        with patch("ingestion.youtube.download_channels.discover", return_value=catalog([A, B])), patch("ingestion.youtube.download_channels.download_catalog") as download:
            self.assertEqual(main(self.args + ["--catalog-only"]), 0)
            download.assert_not_called()
        self.assertEqual(read_archive(output / "downloaded.txt"), {f"youtube {A}"})
        self.assertEqual(json.loads((output / "catalog.json").read_text())["video_count"], 2)

    def test_discovery_failure_preserves_previous_catalog(self) -> None:
        output = self.output / "dota"
        output.mkdir(parents=True)
        previous = json.dumps(catalog([A]))
        (output / "catalog.json").write_text(previous)
        with patch("ingestion.youtube.download_channels.discover", side_effect=RuntimeError("network failure")), patch("ingestion.youtube.download_channels.download_catalog") as download:
            self.assertEqual(main(self.args), 1)
            download.assert_not_called()
        self.assertEqual((output / "catalog.json").read_text(), previous)
        self.assertEqual(json.loads((output / "last_run.json").read_text())["discovery_error"], "network failure")

    def test_completed_unarchived_file_is_adopted_without_downloading(self) -> None:
        (self.root / f"{A}.mp4").write_bytes(b"complete fixture")
        with patch("ingestion.youtube.download_channels.valid_existing_video", return_value=True), patch("ingestion.youtube.download_channels.subprocess.run") as download:
            result = download_catalog(catalog([A]), self.root, 1080, None)
            download.assert_not_called()
        self.assertEqual(result["skipped"], 1)
        self.assertIn(f"youtube {A}", read_archive(self.root / "downloaded.txt"))

    def test_zero_length_completed_file_is_not_skipped(self) -> None:
        (self.root / f"{A}.mp4").touch()
        write_archive(self.root / "downloaded.txt", {f"youtube {A}"})
        with patch("ingestion.youtube.download_channels.subprocess.run", return_value=subprocess.CompletedProcess([], 1)) as download:
            result = download_catalog(catalog([A]), self.root, 1080, None)
            download.assert_called_once()
        self.assertEqual(result["failed"], [A])
        self.assertFalse((self.root / f"{A}.mp4").exists())
        self.assertTrue((self.root / f"{A}.mp4.incomplete").exists())
        self.assertNotIn(f"youtube {A}", read_archive(self.root / "downloaded.txt"))

    def test_orphan_validation_rejects_truncated_and_audio_only_files(self) -> None:
        for duration, streams, valid in ((59.8, [{"codec_type": "video"}], True),
                                         (10, [{"codec_type": "video"}], False),
                                         (60, [{"codec_type": "audio"}], False)):
            with self.subTest(duration=duration, streams=streams):
                probe = {"format": {"duration": str(duration)}, "streams": streams}
                with patch("ingestion.youtube.download_channels.subprocess.run", return_value=subprocess.CompletedProcess([], 0, json.dumps(probe))):
                    self.assertEqual(valid_existing_video(self.root / f"{A}.mp4", catalog([A])["videos"][0]), valid)

    def test_attempt_limit_does_not_mark_deferred_videos_completed(self) -> None:
        with patch("ingestion.youtube.download_channels.subprocess.run", return_value=subprocess.CompletedProcess([], 1)) as download:
            result = download_catalog(catalog([A, B]), self.root, 1080, 1)
            download.assert_called_once()
        self.assertEqual(result["deferred"], 1)
        self.assertEqual(read_archive(self.root / "downloaded.txt"), set())

    def test_success_exit_without_a_completed_file_is_still_a_failure(self) -> None:
        with patch("ingestion.youtube.download_channels.subprocess.run", return_value=subprocess.CompletedProcess([], 0)):
            result = download_catalog(catalog([A]), self.root, 1080, None)
        self.assertEqual(result["downloaded"], 0)
        self.assertEqual(result["failed"], [A])
        self.assertEqual(read_archive(self.root / "downloaded.txt"), set())

    def test_active_livestream_is_deferred_without_downloading(self) -> None:
        saved = catalog([A])
        saved["videos"][0]["live_status"] = "is_live"
        with patch("ingestion.youtube.download_channels.subprocess.run") as download:
            result = download_catalog(saved, self.root, 1080, None)
            download.assert_not_called()
        self.assertEqual(result["deferred"], 1)
        self.assertEqual(result["attempted"], 0)

    def test_download_limit_is_shared_across_channels(self) -> None:
        other = {"id": "other", "name": "Other", "url": "https://www.youtube.com/@Other"}
        self.config.write_text(json.dumps([CHANNEL, other]))
        outcome = {"downloaded": 0, "skipped": 0, "attempted": 1, "failed": [A], "deferred": 0}
        with patch("ingestion.youtube.download_channels.discover", side_effect=[catalog([A]), catalog([B], other)]), patch("ingestion.youtube.download_channels.download_catalog", return_value=outcome) as download:
            self.assertEqual(main(self.args + ["--max-downloads", "1"]), 1)
        self.assertEqual([call.args[3] for call in download.call_args_list], [1, 0])

    def test_saved_catalog_rejects_path_escape_or_changed_channel(self) -> None:
        path = self.root / "catalog.json"
        for payload in (catalog(["../escape"]), catalog([A], {**CHANNEL, "url": "https://www.youtube.com/@Changed"})):
            with self.subTest(payload=payload):
                path.write_text(json.dumps(payload))
                with self.assertRaises(ValueError):
                    load_catalog(path, CHANNEL)

    def test_competing_writer_is_rejected_before_discovery(self) -> None:
        with exclusive_lock(self.output / "dota" / ".download.lock"), patch("ingestion.youtube.download_channels.discover") as scan:
            self.assertEqual(main(self.args), 1)
            scan.assert_not_called()


if __name__ == "__main__":
    unittest.main()
