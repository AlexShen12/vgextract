from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ingestion.common.replay_state import exclusive_lock
from ingestion.youtube.download_videos import main


class YouTubeDownloaderTests(unittest.TestCase):
    def setUp(self) -> None:
        dependency = patch("ingestion.youtube.download_videos.importlib.util.find_spec", return_value=object())
        executables = patch("ingestion.youtube.download_videos.shutil.which", side_effect=lambda name: f"/bin/{name}")
        dependency.start()
        executables.start()
        self.addCleanup(dependency.stop)
        self.addCleanup(executables.stop)

    def test_preview_modes_do_not_create_output_or_write_download_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "videos"
            for mode in ("--list", "--simulate"):
                with self.subTest(mode=mode), patch("ingestion.youtube.download_videos.subprocess.run") as run:
                    run.return_value.returncode = 0
                    self.assertEqual(main([mode, "--output-dir", str(output)]), 0)
                    command = run.call_args.args[0]
                    self.assertNotIn("--download-archive", command)
                    self.assertNotIn("--force-write-archive", command)
                    self.assertFalse(output.exists())

    def test_failure_is_propagated_and_lock_is_released_for_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "videos"
            with patch("ingestion.youtube.download_videos.subprocess.run") as run:
                run.return_value = subprocess.CompletedProcess([], 1)
                self.assertEqual(main(["--output-dir", str(output)]), 1)
                run.return_value = subprocess.CompletedProcess([], 0)
                self.assertEqual(main(["--output-dir", str(output)]), 0)
            self.assertFalse((output / "downloaded.txt").exists())

    def test_competing_downloader_does_not_start(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with exclusive_lock(output / ".download.lock"), patch("ingestion.youtube.download_videos.subprocess.run") as run:
                self.assertEqual(main(["--output-dir", str(output)]), 1)
                run.assert_not_called()

    def test_source_is_passed_as_a_literal_argument(self) -> None:
        source = "https://www.youtube.com/watch?v=example&list=abc"
        with patch("ingestion.youtube.download_videos.subprocess.run") as run:
            run.return_value.returncode = 0
            self.assertEqual(main(["--list", "--source", source]), 0)
            self.assertEqual(run.call_args.args[0][-2:], ["--", source])
            self.assertNotIn("shell", run.call_args.kwargs)

    def test_unbounded_or_empty_window_is_rejected(self) -> None:
        for value in ("0", "-1"):
            with self.subTest(value=value), self.assertRaises(SystemExit) as error:
                main(["--limit", value])
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
