import copy
from pathlib import Path
import unittest

from annotation.league.scene_reference_audit import validate_coverage
from tests.test_league_scene_ablation import label


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.clip={"id":"negative-00","start":0,"end":4}
        self.path=Path("/source/references/negative-00/source-inputs.json")
        self.index={"id":"negative-00","source_sha256":"a"*64,"start":0,"end":4,"sampling_fps":16,
                    "frames":[{}]*64,"sheets":[{}]*16}
        self.label=label();self.label["review"].update(source_index=str(self.path),inspected_interval=[0,4],
            sampling_fps=16,overview_sheets_reviewed=16,native_hud_samples_reviewed=64,native_source_samples_reviewed=64)

    def test_complete_matching_source_inventory(self):
        self.assertTrue(validate_coverage(self.label,self.clip,self.index,self.path,"a"*64))

    def test_shorter_old_export_cannot_claim_full_context(self):
        old=copy.deepcopy(self.index);old["end"]=2
        with self.assertRaisesRegex(ValueError,"complete reserved context"):
            validate_coverage(self.label,self.clip,old,self.path,"a"*64)

    def test_partial_viewing_and_wrong_source_index_fail(self):
        for update in ({"native_hud_samples_reviewed":32},{"overview_sheets_reviewed":8},
                       {"source_index":"/old/source-inputs.json"},{"inspected_interval":[0,2]}):
            changed=copy.deepcopy(self.label);changed["review"].update(update)
            with self.assertRaises(ValueError):validate_coverage(changed,self.clip,self.index,self.path,"a"*64)


if __name__=="__main__":unittest.main()
