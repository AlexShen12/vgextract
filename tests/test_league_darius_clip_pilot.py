from annotation.league.darius_clip_pilot import (
    assess_clips, clip_contexts, collect_seeds, compare_cuts, group_clips, split_intervals,
)


def selection(transition=None, epoch=0):
    rows = []
    for n in range(16 * 8):
        time = n / 16
        rows.append({"id": f"selected-{n}", "start": time, "end": time + 1/16,
                     "champion": "Darius", "owner_epoch": epoch if time >= 4 else 0,
                     "playback_segment_id": "b" if transition and time >= 4 else "a",
                     "playback_state": "playing", "discontinuity_reasons": [transition] if transition and time == 4 else []})
    return rows


def seeds():
    return [{"id": "attack", "start": 3.5, "end": 3.625, "evidence_ids": ["frame"]}]


def refs():
    return [{"id": "fight", "seed_ids": ["attack"],
             "phases": {"setup": [1, 2], "action": [3.5, 3.625], "outcome": [6.75, 7]}}]


def test_cut_comparison_matches_once_and_marks_uncovered_timestamps():
    playback = [{"id": "cut", "start": 2, "end": 2.07, "visual_motion": .4,
                 "discontinuity_reasons": ["visual_cut"]},
                {"id": "adjacent", "start": 2.08, "end": 2.14, "visual_motion": .06},
                {"id": "layout", "start": 5, "end": 5.1, "visual_motion": .1,
                 "discontinuity_reasons": ["layout_transition"]}]
    clocks = [{"time": 1.5, "seconds": 10, "evidence_ids": ["cl1"]},
              {"time": 2.5, "seconds": 11, "evidence_ids": ["cl2"]}]
    result = compare_cuts([2.02, 2.075, 8], playback, clocks, [(0, 6)], [(1, 5)])
    assert [row["comparison"] for row in result] == ["matched", "detector_only", "context_only", "outside_tracker_coverage"]
    assert result[0]["clock_readings"] == clocks
    assert result[2]["nearby_context_changes"][0]["reasons"] == ["layout_transition"]
    assert result[3]["clock_readings"] == [] and not result[3]["review_eligible"]


def test_reviewed_edit_and_rewind_bound_clips_at_source_time():
    for classification in ("edit", "rewind"):
        cut = {"classification": classification, "time": 4.015, "tracker_time": None}
        groups = clip_contexts(selection(), [(0, 8)], [cut], 60)
        [clip] = group_clips(seeds(), groups, refs())
        assert clip["end"] == 4.015
        assert assess_clips([clip], refs())[0]["phases"] == {
            "setup": "retained", "action": "retained", "outcome": "missing",
        }


def test_reviewed_camera_switch_preserves_verified_context_only():
    cut = {"classification": "camera_switch", "time": 3.99, "tracker_time": 4}
    [baseline] = group_clips(seeds(), clip_contexts(selection("visual_cut"), [(0, 8)], [], 60), refs())
    [reviewed] = group_clips(seeds(), clip_contexts(selection("visual_cut"), [(0, 8)], [cut], 60), refs())
    assert baseline["end"] == 4 and reviewed["end"] == 7
    changed = clip_contexts(selection("visual_cut", epoch=1), [(0, 8)], [cut], 60)
    assert group_clips(seeds(), changed, refs())[0]["end"] == 4
    unknown = selection("visual_cut")
    unknown[64]["playback_state"] = "unknown"
    assert group_clips(seeds(), clip_contexts(unknown, [(0, 8)], [cut], 60), refs())[0]["end"] == 4


def test_guarded_clips_and_commentary_provenance():
    windows = split_intervals([(0, 8)], [(4, 6)])
    commentary = [{"id": "line", "start": 3.75, "end": 4.25}]
    cues = collect_seeds([], commentary, {"line"}, windows)
    [clip] = group_clips(cues, clip_contexts(selection(), windows, [], 60), [])
    assert windows == [(0, 4), (6, 8)]
    assert cues[0]["kind"] == "commentary_lead"
    assert clip["end"] == 4 and clip["evidence_ids"] == ["line"]


def test_nested_refinements_keep_one_clip_and_all_seed_evidence():
    second = {"id": "second", "start": 6, "end": 6, "evidence_ids": ["second-frame"]}
    reference = {"id": "one-exchange", "seed_ids": ["attack", "second"],
                 "phases": {"setup": [1, 2], "action": [3.5, 6], "outcome": [6.75, 7]}}
    [clip] = group_clips(seeds() + [second], clip_contexts(selection(), [(0, 8)], [], 60), [reference])
    assert clip["seed_ids"] == ["attack", "second"]
    assert clip["evidence_ids"] == ["frame", "second-frame"]
    assert len(clip["initial_ranges"]) == 2
