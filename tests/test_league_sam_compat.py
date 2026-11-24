"""Regression coverage for the actual pinned multiplex API failure."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from annotation.league.sam_compat import adapt_session_init, load_verified_state


class SAMCompatibilityTests(unittest.TestCase):
    def test_disabled_unsupported_offload_is_omitted_without_dropping_other_options(self):
        class Model:
            def init_state(self, resource_path, async_loading_frames=False):
                return resource_path, async_loading_frames
        model = Model()
        self.assertEqual(adapt_session_init(model)["session_adapter"], "omit_disabled_offload_state_to_cpu")
        self.assertEqual(model.init_state(resource_path="frames", offload_state_to_cpu=False,
                                        async_loading_frames=True), ("frames", True))
        with self.assertRaises(ValueError):
            model.init_state("frames", offload_state_to_cpu=True)
        with self.assertRaises(TypeError):
            model.init_state("frames", unrelated_option=False)

    def test_supported_interface_is_not_modified(self):
        model = SimpleNamespace(init_state=lambda offload_state_to_cpu=False: offload_state_to_cpu)
        original = model.init_state
        self.assertEqual(adapt_session_init(model)["session_adapter"], "unneeded")
        self.assertIs(model.init_state, original)

    def model(self):
        tensor = SimpleNamespace(shape=(2, 3))
        return SimpleNamespace(state_dict=lambda: {"tracker.weight": tensor, "detector.attn.freqs_cis": tensor},
                               named_parameters=lambda: [("tracker.weight", tensor)], load_state_dict=Mock()), tensor

    def test_complete_weights_and_only_derived_buffer_omission_are_allowed(self):
        model, tensor = self.model()
        result = load_verified_state(model, {"model": {"sam2_predictor.weight": tensor}})
        self.assertTrue(result["all_parameters_loaded"])
        model.load_state_dict.assert_called_once_with({"tracker.weight": tensor}, strict=False)

    def test_missing_parameters_extra_weights_and_bad_shapes_fail_before_loading(self):
        for state in ({}, {"tracker.weight": SimpleNamespace(shape=(5,))},
                      {"tracker.weight": SimpleNamespace(shape=(2, 3)), "unknown.weight": SimpleNamespace(shape=(1,))}):
            model, _ = self.model()
            with self.assertRaises(ValueError):
                load_verified_state(model, state)
            model.load_state_dict.assert_not_called()


if __name__ == "__main__":
    unittest.main()
