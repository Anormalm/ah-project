import numpy as np
import pytest

from features.depth_gate import filter_by_depth
from ingestion.video_loader import FramePacket
from utils.schemas import Detection, PoseResult


def test_depth_gate_filters_detection_and_corresponding_pose():
    depth = np.full((40, 40), 2000, dtype=np.uint16)
    depth[:, 20:] = 9000
    packet = FramePacket(np.zeros((40, 40, 3)), 1.0, depth, 0.001)
    detections = [Detection(bbox=box, confidence=0.9) for box in [(0, 0, 20, 40), (20, 0, 40, 40)]]
    poses = [PoseResult(bbox=d.bbox, keypoints=[]) for d in detections]
    kept, kept_poses, measurements = filter_by_depth(packet, detections, poses)
    assert kept == detections[:1]
    assert kept_poses == poses[:1]
    assert measurements[0]["distance_m"] == pytest.approx(2.0)


def test_depth_holes_are_not_valid_measurements():
    packet = FramePacket(np.zeros((20, 20, 3)), 1.0, np.zeros((20, 20)), 0.001)
    kept, _, info = filter_by_depth(packet, [Detection(bbox=(0, 0, 20, 20), confidence=1)], [])
    assert not kept
    assert info[0]["distance_m"] is None


@pytest.mark.parametrize("depth,scale", [(None, 0.001), (np.ones((5, 5)), 0.001), (np.ones((20, 20)), 0)])
def test_depth_enforcement_rejects_missing_or_misaligned_input(depth, scale):
    with pytest.raises(RuntimeError):
        filter_by_depth(FramePacket(np.zeros((20, 20, 3)), 1.0, depth, scale), [], [])
