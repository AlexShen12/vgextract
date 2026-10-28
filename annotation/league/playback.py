"""Source time is authoritative; game clocks never imply a continuous replay."""

from __future__ import annotations

from .core import uid


class PlaybackTracker:
    def __init__(self, source, fps=1, state=None):
        self.source, self.fps = source, fps
        self.state = state or {
            "previous": None,
            "segment": None,
            "seen_clocks": {},
            "paused_samples": 0,
        }

    def update(self, timestamp, clock, image=None, layout_valid=True):
        import numpy as np

        previous = self.state["previous"]
        feature = (
            list(np.asarray(image.convert("L").resize((32, 18)), dtype=float).ravel())
            if image is not None
            else None
        )
        reasons = []
        delta = timestamp - previous["time"] if previous else 0
        if previous:
            if delta > 2.5 / self.fps:
                reasons.append("unsampled_gap")
            if layout_valid != previous["layout_valid"]:
                reasons.append("layout_transition")
            if clock is None or previous["clock"] is None:
                reasons.append("unverified_clock_continuity")
            if clock is not None and previous["clock"] is not None:
                advance = clock - previous["clock"]
                if advance < 0:
                    reasons.append("game_clock_rewind")
                elif advance > max(4, delta * 3):
                    reasons.append("game_clock_discontinuity")
            if feature and previous.get("feature"):
                change = (
                    float(np.abs(np.asarray(feature) - previous["feature"]).mean())
                    / 255
                )
                if change > 0.3:
                    reasons.append("visual_cut")
        if not previous or reasons:
            self.state["segment"] = uid("playback", self.source, timestamp, reasons)
            self.state["paused_samples"] = 0
            self.state["segment_replayed"] = False
        elif clock is not None and clock == previous["clock"]:
            self.state["paused_samples"] += 1
        else:
            self.state["paused_samples"] = 0
        key = str(clock)
        first_segment = (
            self.state["seen_clocks"].get(key) if clock is not None else None
        )
        repeated = bool(first_segment and first_segment != self.state["segment"])
        self.state["segment_replayed"] = (
            self.state.get("segment_replayed", False) or repeated
        )
        if clock is not None:
            self.state["seen_clocks"].setdefault(key, self.state["segment"])
        self.state["previous"] = {
            "time": timestamp,
            "clock": clock,
            "layout_valid": layout_valid,
            "feature": feature,
        }
        return {
            "playback_segment_id": self.state["segment"],
            "playback_state": (
                "unknown"
                if clock is None or not layout_valid
                else "paused" if self.state["paused_samples"] >= 2 else "playing"
            ),
            "continuity_verified": bool(
                previous
                and not reasons
                and layout_valid
                and clock is not None
                and not self.state["segment_replayed"]
            ),
            "revisited_game_clock": repeated,
            "discontinuity_reasons": reasons,
        }
