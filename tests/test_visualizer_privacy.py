from __future__ import annotations

import numpy as np

from output.visualizer import VisualizationConfig, Visualizer
from utils.schemas import Detection


def test_visualizer_person_blur_changes_person_region() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    frame[20:100, 40:120] = (40, 180, 220)

    cfg = VisualizationConfig(
        enabled=False,
        privacy_mode="person_blur",
        privacy_blur_kernel=31,
        privacy_expand_ratio=0.0,
    )
    vis = Visualizer(stream_id="t0", cfg=cfg)
    det = Detection(bbox=(40.0, 20.0, 120.0, 100.0), confidence=0.9)

    keep = vis.render(
        frame=frame,
        detections=[det],
        tracks=[],
        risk_events={},
        fps=15.0,
        bed_zones=None,
    )
    out = vis.get_last_output_frame()

    assert keep is True
    assert out is not None
    assert out.shape == frame.shape
    # At least some pixels in person ROI should be modified by blur pipeline.
    assert np.any(out[20:100, 40:120] != frame[20:100, 40:120])
