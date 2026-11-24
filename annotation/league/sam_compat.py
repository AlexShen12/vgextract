"""Audited compatibility for the pinned SAM 3.1 multiplex predictor."""
from functools import wraps
import inspect


def load_multiplex_predictor(weights):
    import torch
    from sam3.model_builder import build_sam3_multiplex_video_predictor

    predictor = build_sam3_multiplex_video_predictor(
        checkpoint_path=str(weights), max_num_objects=16, use_fa3=False, use_rope_real=False,
        compile=False, warm_up=False,
    )
    state = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
    receipt = load_verified_state(predictor.model, state)
    receipt.update(adapt_session_init(predictor.model))
    return predictor, receipt


def adapt_session_init(model):
    """Drop only the unsupported, disabled state-offload option in the pinned API."""
    original = model.init_state
    if "offload_state_to_cpu" in inspect.signature(original).parameters:
        return {"session_adapter": "unneeded"}

    @wraps(original)
    def init_state(*args, offload_state_to_cpu=False, **kwargs):
        if offload_state_to_cpu is not False:
            raise ValueError("Pinned multiplex model does not support state offloading")
        return original(*args, **kwargs)

    model.init_state = init_state
    return {"session_adapter": "omit_disabled_offload_state_to_cpu"}


def load_verified_state(model, state):
    """Require every parameter; only derived rotary-position buffers may differ."""
    if "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    remapped = {}
    for key, value in state.items():
        if key.startswith("sam3_model."):
            key = "detector." + key[len("sam3_model."):]
        elif key.startswith("sam2_predictor."):
            key = "tracker." + key[len("sam2_predictor."):]
        if key in remapped:
            raise ValueError("Colliding SAM checkpoint keys")
        remapped[key] = value
    expected = model.state_dict()
    parameters = dict(model.named_parameters())
    missing = sorted(set(expected) - set(remapped))
    unexpected = sorted(set(remapped) - set(expected))
    derived = lambda key: key.endswith((".attn.freqs_cis", ".attn.freqs_cis_real", ".attn.freqs_cis_imag"))
    if any(k in parameters or not derived(k) for k in missing) or any(not derived(k) for k in unexpected):
        raise ValueError(f"SAM checkpoint coverage failed: missing={missing[:12]}, unexpected={unexpected[:12]}")
    mismatched = [k for k in set(expected) & set(remapped) if expected[k].shape != remapped[k].shape]
    if mismatched:
        raise ValueError(f"SAM checkpoint shape mismatch: {sorted(mismatched)[:12]}")
    # Apply the audited state explicitly, rather than relying on upstream non-strict loading.
    model.load_state_dict(remapped, strict=False)
    return {"parameter_tensors": len(parameters), "loaded_parameter_tensors": len(parameters),
            "missing_derived_buffers": missing, "unused_derived_buffers": unexpected,
            "all_parameters_loaded": True}
