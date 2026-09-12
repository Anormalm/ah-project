from __future__ import annotations

import time

import numpy as np
import pytest

from detection.yolo_detector import MockDetectionEngine, YOLOPersonDetector
from pipelines.main_pipeline import _history_buffer_length, _risk_config_with_model_threshold
from pose.pose_estimator import MockPoseEngine, PoseEstimator
from risk.risk_scoring import RiskScorer
from temporal.temporal_model import TemporalRiskModel
from tracking.tracker import ByteTrackLikeTracker
from utils.schemas import Detection, FeatureVector, RuleDecision


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


def test_tracking_uses_center_proximity_when_pose_box_shape_changes() -> None:
    tracker = ByteTrackLikeTracker(
        iou_threshold=0.3,
        max_misses=5,
        center_distance_threshold=0.45,
    )
    pose_estimator = PoseEstimator(backend=MockPoseEngine())
    frame = np.zeros((400, 400, 3), dtype=np.uint8)

    upright = Detection(bbox=(100.0, 50.0, 200.0, 350.0), confidence=0.9)
    lying = Detection(bbox=(20.0, 170.0, 280.0, 230.0), confidence=0.9)
    first = tracker.update([upright], pose_estimator.predict(frame, [upright.bbox]), timestamp=1.0)
    second = tracker.update([lying], pose_estimator.predict(frame, [lying.bbox]), timestamp=1.1)

    assert first[0].track_id == second[0].track_id


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


def test_sub_threshold_ml_probability_has_no_risk_influence() -> None:
    scorer = RiskScorer(
        ml_weight=0.8,
        ml_decision_threshold=0.7,
        allow_ml_level_override=True,
        ema_alpha=1.0,
        downgrade_grace_sec=0.0,
    )
    rule_decision = RuleDecision(
        track_id=8,
        timestamp=1.0,
        rule_score=0.2,
        rule_level="LOW",
        reasons=[],
    )

    event = scorer.score(rule_decision, ml_probability=0.69)

    assert event.confidence == pytest.approx(0.2)
    assert event.risk_level == "LOW"
    assert "ml_high_probability" not in event.reasons


def test_above_threshold_ml_probability_can_influence_risk() -> None:
    scorer = RiskScorer(
        ml_weight=1.0,
        ml_decision_threshold=0.7,
        allow_ml_level_override=True,
        ema_alpha=1.0,
        downgrade_grace_sec=0.0,
    )
    rule_decision = RuleDecision(
        track_id=9,
        timestamp=1.0,
        rule_score=0.2,
        rule_level="LOW",
        reasons=[],
    )

    event = scorer.score(rule_decision, ml_probability=0.8)

    assert event.confidence == pytest.approx(0.8)
    assert event.risk_level == "HIGH"
    assert "ml_high_probability" in event.reasons


def test_shadow_mode_decision_remains_rule_only_with_high_ml_probability() -> None:
    scorer = RiskScorer(
        ml_weight=0.0,
        ml_decision_threshold=0.95,
        allow_ml_level_override=False,
        ema_alpha=1.0,
        downgrade_grace_sec=0.0,
    )
    rule_decision = RuleDecision(
        track_id=10,
        timestamp=1.0,
        rule_score=0.2,
        rule_level="LOW",
        reasons=[],
    )

    event = scorer.score(rule_decision, ml_probability=0.99)

    assert event.confidence == pytest.approx(0.2)
    assert event.risk_level == "LOW"
    assert event.event == "stable"


def test_checkpoint_threshold_is_used_unless_risk_config_overrides_it() -> None:
    class _TemporalModel:
        decision_threshold = 0.73

    inherited = _risk_config_with_model_threshold({"ml_weight": 0.2}, _TemporalModel())
    overridden = _risk_config_with_model_threshold(
        {"ml_weight": 0.2, "ml_decision_threshold": 0.9},
        _TemporalModel(),
    )

    assert inherited["ml_decision_threshold"] == pytest.approx(0.73)
    assert overridden["ml_decision_threshold"] == pytest.approx(0.9)


def test_history_buffer_spans_20fps_model_horizon_at_30fps() -> None:
    assert _history_buffer_length(sequence_len=10, source_fps=30.0, model_fps=20.0) == 15


def test_risk_downgrades_after_grace_window() -> None:
    t0 = time.time()
    scorer = RiskScorer(
        ml_weight=0.0,
        ema_alpha=1.0,
        downgrade_grace_sec=1.0,
        allow_ml_level_override=False,
    )

    high = RuleDecision(track_id=5, timestamp=t0, rule_score=0.85, rule_level="HIGH", reasons=["lean_instability"])
    low_soon = RuleDecision(track_id=5, timestamp=t0 + 0.2, rule_score=0.1, rule_level="LOW", reasons=[])
    low_later = RuleDecision(track_id=5, timestamp=t0 + 1.3, rule_score=0.1, rule_level="LOW", reasons=[])

    e1 = scorer.score(high, ml_probability=0.0)
    e2 = scorer.score(low_soon, ml_probability=0.0)
    e3 = scorer.score(low_later, ml_probability=0.0)

    assert e1.risk_level == "HIGH"
    assert e2.risk_level == "HIGH"
    assert e3.risk_level == "LOW"


def test_temporal_infer_interval_cache() -> None:
    class _CountingBackend:
        def __init__(self) -> None:
            self.calls = 0

        def predict(self, inputs):
            _ = inputs
            self.calls += 1
            return 0.42

    def _fv(ts: float) -> FeatureVector:
        return FeatureVector(
            track_id=10,
            timestamp=ts,
            center_of_mass=(100.0, 100.0),
            velocity=(1.0, 1.0),
            acceleration=(0.1, 0.1),
            joint_angles={"knee_l": 30.0},
            posture="standing",
            bed_zone_distance=50.0,
            lean_angle=5.0,
        )

    backend = _CountingBackend()
    model = TemporalRiskModel(backend=backend, sequence_len=8, infer_interval=3, min_infer_steps=2)
    seq = [_fv(float(i)) for i in range(8)]

    p1 = model.predict(seq, track_id=10)
    p2 = model.predict(seq, track_id=10)
    p3 = model.predict(seq, track_id=10)

    assert p1 == p2 == p3
    assert backend.calls == 2

