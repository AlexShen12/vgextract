"""Dependent comparison jobs require independent mask overlap and resource headroom."""
from pathlib import Path
import tempfile
import unittest

from annotation.league.core import atomic_json, read_json
from annotation.league.media import file_hash
from annotation.league.sam_tracking_gate import qualify


class SAMGateTests(unittest.TestCase):
    def setUp(self):
        from PIL import Image
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.mask = self.root / "mask.png"
        image = Image.new("L", (20, 20))
        image.putpixel((10, 10), 255)
        image.save(self.mask)
        selected = {"status": "prepared", "seed": {"time": 0}, "frames": list(range(32)),
                    "prompts": [{"frame_index": 0}]}
        self.prepared = self.root / "prepared.json"
        atomic_json(self.prepared, {"root": str(self.root), "labels_dir": str(self.root), "clips": [{
            "id": "clip", "references": [{"time": 0, "visibility": "visible", "mask_path": "mask.png"}],
            "arms": {"verified": selected, "automatic": selected}}]})
        atomic_json(self.root / "preflight/clip-complete.json", {"frames": [{"frame_index": i, "masks": [
            {"object_id": 1, "mask_path": str(self.mask), "sha256": file_hash(self.mask)}]} for i in range(32)]})
        self.report = {"status": "completed", "seed_check": {"contains_seed": True}, "runtime": {"gpu": "NVIDIA L40S"},
                       "resources": {"torch_peak_memory_bytes": 1024, "peak_host_memory_bytes": 1024, "wall_seconds_this_run": 1}}
        atomic_json(self.root / "preflight/report.json", self.report)

    def test_matching_mask_and_measured_headroom_qualify(self):
        self.assertEqual(qualify(self.prepared)["status"], "passed")

    def test_excessive_memory_blocks_and_preserves_failure_report(self):
        self.report["resources"]["torch_peak_memory_bytes"] = 40 * 1024**3
        atomic_json(self.root / "preflight/report.json", self.report)
        with self.assertRaisesRegex(ValueError, "headroom"):
            qualify(self.prepared)
        self.assertEqual(read_json(self.root / "gate.json")["status"], "failed")

    def test_corrupted_mask_cannot_qualify(self):
        self.mask.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            qualify(self.prepared)


if __name__ == "__main__":
    unittest.main()
