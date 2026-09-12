from __future__ import annotations

from temporal.rule_engine import RuleEngine
from utils.schemas import FeatureVector


def _feature(
    timestamp: float,
    center_y: float,
    *,
    vy: float = 0.0,
    ay: float = 0.0,
    posture: str = "standing",
    lean: float = 0.0,
    depth_y: float | None = 0.0,
    depth_ratio: float | None = 1.0,
    camera_motion: bool = False,
) -> FeatureVector:
    center_3d = None if depth_y is None else (0.0, depth_y, 2.0)
    return FeatureVector(
        track_id=1,
        timestamp=timestamp,
        center_of_mass=(100.0, center_y),
        velocity=(0.0, vy),
        acceleration=(0.0, ay),
        joint_angles={},
        posture=posture,
        bed_zone_distance=1_000.0,
        lean_angle=lean,
        pose_valid_ratio=0.9,
        body_height_px=200.0,
        center_of_mass_3d_m=center_3d,
        depth_valid_ratio=depth_ratio,
        camera_motion=camera_motion,
    )


def _engine(**overrides) -> RuleEngine:
    options = {
        "enable_sitting_edge_rule": False,
        "enable_lean_instability_rule": False,
        "enable_inactivity_rule": False,
        "enable_transition_rule": False,
        "require_fall_confirmation": True,
        "fall_min_track_frames": 5,
        "fall_require_depth": True,
        "fall_require_metric_drop": True,
        "fall_min_depth_valid_ratio": 0.6,
        "fall_confirm_lying_frames": 3,
        "fall_cooldown_sec": 5.0,
        "suppress_motion_sensitive_rules_on_camera_motion": True,
    }
    options.update(overrides)
    return RuleEngine(**options)


def test_single_frame_pose_jump_is_not_a_confirmed_fall() -> None:
    engine = _engine()
    for i in range(8):
        decision = engine.evaluate(_feature(i * 0.1, 100.0))

    decision = engine.evaluate(_feature(0.8, 170.0, vy=700.0, ay=5000.0))

    assert decision.rule_level == "LOW"
    assert "confirmed_fall" not in decision.reasons


def test_drop_followed_by_sustained_lying_confirms_once() -> None:
    engine = _engine()
    for i in range(8):
        engine.evaluate(_feature(i * 0.1, 100.0, depth_y=0.0))

    sequence = [
        _feature(0.8, 125.0, vy=250.0, posture="lying", lean=70.0, depth_y=0.12),
        _feature(0.9, 155.0, vy=300.0, posture="lying", lean=75.0, depth_y=0.28),
        _feature(1.0, 170.0, vy=150.0, posture="lying", lean=80.0, depth_y=0.35),
    ]
    decisions = [engine.evaluate(item) for item in sequence]

    assert decisions[-1].rule_level == "CRITICAL"
    assert "confirmed_fall" in decisions[-1].reasons
    assert "depth_validated" in decisions[-1].reasons

    cooldown = engine.evaluate(
        _feature(1.1, 172.0, vy=20.0, posture="lying", lean=80.0, depth_y=0.36)
    )
    assert cooldown.rule_level == "LOW"


def test_depth_required_mode_rejects_missing_depth() -> None:
    engine = _engine()
    for i in range(8):
        engine.evaluate(_feature(i * 0.1, 100.0, depth_y=None, depth_ratio=None))
    for i, center_y in enumerate((125.0, 155.0, 180.0), start=8):
        decision = engine.evaluate(
            _feature(
                i * 0.1,
                center_y,
                vy=300.0,
                ay=1500.0,
                posture="lying",
                lean=75.0,
                depth_y=None,
                depth_ratio=None,
            )
        )

    assert decision.rule_level == "LOW"


def test_cropped_pose_that_only_looks_lying_is_not_confirmed() -> None:
    engine = _engine()
    for i in range(8):
        engine.evaluate(_feature(i * 0.1, 100.0, depth_y=0.0))
    for i, (center_y, depth_y) in enumerate(
        ((125.0, 0.12), (155.0, 0.24), (175.0, 0.32)),
        start=8,
    ):
        decision = engine.evaluate(
            _feature(
                i * 0.1,
                center_y,
                vy=300.0,
                ay=1500.0,
                posture="lying",
                lean=12.0,
                depth_y=depth_y,
            )
        )

    assert decision.rule_level == "LOW"
    assert "confirmed_fall" not in decision.reasons


def test_camera_bump_clears_candidate_and_holds_settle_period() -> None:
    engine = _engine(fall_camera_settle_sec=1.0)
    for i in range(8):
        engine.evaluate(_feature(i * 0.1, 100.0))
    engine.evaluate(_feature(0.8, 160.0, vy=600.0, ay=4000.0, posture="lying", lean=75.0))
    engine.evaluate(_feature(0.9, 170.0, camera_motion=True, posture="lying", lean=75.0))
    decision = engine.evaluate(_feature(1.0, 180.0, vy=300.0, posture="lying", lean=75.0))

    assert decision.rule_level == "LOW"
    assert "confirmed_fall" not in decision.reasons
