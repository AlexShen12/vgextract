import json

import numpy as np

from annotation.league.minimap import LAYOUT_ANCHORS, MINION_EXAMPLES, REFERENCE_TIME, TOWERS, Reader


def test_minimap_accepts_one_covered_anchor_and_serializes_tower_readings():
    reference = np.random.default_rng(1).random((290, 290, 3), dtype=np.float32)
    times = {REFERENCE_TIME}
    for *_, additional, absent in TOWERS:
        times.update((*additional, *absent))
    times.update(time for examples in MINION_EXAMPLES.values() for time, _, _ in examples)
    reader = Reader(dict.fromkeys(times, reference))

    image = reference.copy()
    for index, (y, x, _) in enumerate(LAYOUT_ANCHORS):
        image[y, x] = 0
        row = json.loads(json.dumps(reader.read(image)))
        assert row["layout_visible"] is (index == 0)
        if index:
            assert row["visible_minion_waves"] is None
            assert {tower["status_reason"] for tower in row["towers"]} == {"minimap_unreadable"}
