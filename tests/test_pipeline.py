from __future__ import annotations

import time

import numpy as np

from detection.yolo_detector import MockDetectionEngine, YOLOPersonDetector
from pose.pose_estimator import MockPoseEngine, PoseEstimator
from risk.risk_scoring import RiskScorer
from tracking.tracker import ByteTrackLikeTracker
from utils.schemas import Detection, RuleDecision


def test_detection_output_format() -> None:
    detector = YOLOPersonDetector(backend=MockDetectionEngine(), conf_threshold=0.1)
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    detections = detector.detect(frame)

    assert len(detections) >= 1
    det = detections[0]
    assert isinstance(det.bbox, tuple)
    assert len(det.bbox) == 4
    assert 0.0 <= det.confidence <= 1.0
    assert det.class_name == "person"


def test_pose_output_format() -> None:
    estimator = PoseEstimator(backend=MockPoseEngine())
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    bboxes = [(100.0, 80.0, 220.0, 300.0)]

    poses = estimator.predict(frame, bboxes)
    assert len(poses) == 1
    pose = poses[0]
    assert pose.bbox == bboxes[0]
    assert len(pose.keypoints) == 17
    for x, y, c in pose.keypoints:
        assert isinstance(x, float)
        assert isinstance(y, float)
        assert 0.0 <= c <= 1.0


def test_tracking_consistency() -> None:
    tracker = ByteTrackLikeTracker(iou_threshold=0.2, max_misses=5)
    pose_estimator = PoseEstimator(backend=MockPoseEngine())
    frame = np.zeros((360, 640, 3), dtype=np.uint8)

    det1 = [Detection(bbox=(100.0, 80.0, 220.0, 300.0), confidence=0.9)]
    poses1 = pose_estimator.predict(frame, [det1[0].bbox])
    tracks1 = tracker.update(det1, poses1, timestamp=time.time())

    det2 = [Detection(bbox=(106.0, 84.0, 226.0, 304.0), confidence=0.9)]
    poses2 = pose_estimator.predict(frame, [det2[0].bbox])
    tracks2 = tracker.update(det2, poses2, timestamp=time.time() + 0.1)

    assert len(tracks1) == 1
    assert len(tracks2) == 1
    assert tracks1[0].track_id == tracks2[0].track_id


def test_risk_scoring_logic() -> None:
    scorer = RiskScorer(ml_weight=0.4)
    rule_decision = RuleDecision(
        track_id=7,
        timestamp=time.time(),
        rule_score=0.8,
        rule_level="HIGH",
        reasons=["lean_instability"],
    )

    event = scorer.score(rule_decision, ml_probability=0.35)
    assert event.track_id == 7
    assert event.risk_level in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert 0.0 <= event.confidence <= 1.0
    assert "lean_instability" in event.reasons

