"""Holdout reservations cannot reuse inspected context or calibration frames."""
import copy
import unittest

from annotation.league.scene_split import merged_intervals, reserve, validate_reservation


def snapshot():
    return {"source":{"sha256":"a"*64,"duration":300},"records":[
        {"start":0,"end":20},{"start":60,"end":80},{"start":120,"end":140},
        {"start":180,"end":200},{"start":240,"end":260}],"guard_seconds":8}


class SceneSplitTests(unittest.TestCase):
    def test_full_context_and_calibration_points_receive_eight_second_guards(self):
        self.assertEqual(merged_intervals([[10,20],[30,30]],0,100),[[2,38]])
        with self.assertRaisesRegex(ValueError,"eight-second"):
            merged_intervals([[10,20]],0,100,guard=2)

    def test_selection_is_metadata_only_deterministic_and_separated(self):
        data=snapshot();value=reserve(data)
        self.assertEqual(value,reserve(data))
        self.assertEqual(len(value["clips"]),5)
        for clip in value["clips"]:
            self.assertEqual(clip["reference_status"],"unlabeled")
            self.assertLessEqual(clip["end"]-clip["start"],24)
            self.assertFalse(any(clip["start"]<r["end"]+8 and r["start"]-8<clip["end"] for r in data["records"]))
        self.assertFalse(value["cross_match_generalization"])

    def test_insufficient_fresh_coverage_does_not_relax_guards(self):
        data=snapshot();data["records"]=[{"start":0,"end":300}]
        with self.assertRaisesRegex(ValueError,"do not relax"):
            reserve(data)

    def test_changed_split_or_exposure_requires_new_revision(self):
        data=snapshot();value=reserve(data)
        self.assertTrue(validate_reservation(value,data))
        changed=copy.deepcopy(value);changed["clips"][0]["start"]-=1
        with self.assertRaisesRegex(ValueError,"changed"):
            validate_reservation(changed,data)
        data["records"].append({"start":35,"end":40})
        with self.assertRaisesRegex(ValueError,"changed"):
            validate_reservation(value,data)


if __name__=="__main__":
    unittest.main()
