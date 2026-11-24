"""Portable source-reference transfers preserve provenance and remote revisions."""
from pathlib import Path
import tempfile
import unittest

from annotation.league.core import atomic_json, read_json
from annotation.league.scene_reference import file_hash
from annotation.league.scene_reference_campaign import inside, relocate, stage


class SceneReferenceCampaignTests(unittest.TestCase):
    def fixture(self, root):
        source = root/"source.mp4"
        source.write_bytes(b"fixture-source-identity")
        references = root/"references"
        references.mkdir()
        value = {"schema_version": 1, "stage": 2, "split": "development", "guard_seconds": 8,
                 "source": {"path": "/old/source.mp4", "sha256": file_hash(source), "width": 1920, "height": 1080},
                 "clips": [{"id": "negative-00", "start": 10, "end": 20, "split": "development"}]}
        path = references/"negative-candidates.json"
        atomic_json(path, value)
        atomic_json(references/"transfer-manifest.json", {"source_sha256": file_hash(source),
            "original_run_root": "/old/run", "original_relative_run_root": "data/old-run",
            "review_input_requests": [], "files": [{"path": path.name, "sha256": file_hash(path)}]})
        return references, source

    def test_paths_relocate_without_rewriting_historical_originals(self):
        value = {"path": "/old/run/clip/frame.jpg", "original_path": "/old/run/clip/frame.jpg"}
        moved = relocate(value, {"/old/run": "/new/run"})
        self.assertEqual(moved["path"], "/new/run/clip/frame.jpg")
        self.assertEqual(moved["original_path"], value["original_path"])
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "escapes"):
                inside(Path(temporary), "../../outside")

    def test_identical_remote_source_can_stage_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            references, source = self.fixture(root)
            for _ in range(2):
                benchmark, _ = stage(references, root/"remote-run", source)
            staged = read_json(benchmark/"negative-candidates.json")
            self.assertEqual(staged["source"]["path"], str(source.resolve()))
            self.assertFalse(read_json(root/"remote-run/reference-transfer.json")["method"].startswith("human"))

    def test_changed_source_and_tampered_transfer_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            references, source = self.fixture(root)
            source.write_bytes(b"different-source")
            with self.assertRaisesRegex(ValueError, "differs"):
                stage(references, root/"remote-run", source)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            references, source = self.fixture(root)
            (references/"negative-candidates.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "modified"):
                stage(references, root/"remote-run", source)

    def test_remote_annotation_changes_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            references, source = self.fixture(root)
            benchmark, _ = stage(references, root/"remote-run", source)
            path = benchmark/"negative-candidates.json"
            value = read_json(path)
            value["review_note"] = "new remote source adjudication"
            atomic_json(path, value)
            with self.assertRaisesRegex(ValueError, "Remote reference changed"):
                stage(references, root/"remote-run", source)
            self.assertEqual(read_json(path)["review_note"], value["review_note"])


if __name__ == "__main__":
    unittest.main()
