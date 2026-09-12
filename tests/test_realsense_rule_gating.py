from __future__ import annotations

from temporal.rule_engine import RuleEngine
from utils.schemas import FeatureVector


def _feature(*, camera_motion: bool) -> FeatureVector:
    return FeatureVector(
        track_id=1,
        timestamp=10.0,
        center_of_mass=(100.0, 200.0),
        velocity=(0.0, 300.0),
        acceleration=(0.0, 1500.0),
        joint_angles={},
        posture="standing",
        bed_zone_distance=1_000.0,
        lean_angle=0.0,
        camera_motion=camera_motion,
    )


def test_camera_motion_suppresses_pixel_fall_rule_when_enabled() -> None:
    engine = RuleEngine(
        enable_fall_rule=True,
        suppress_motion_sensitive_rules_on_camera_motion=True,
    )

    decision = engine.evaluate(_feature(camera_motion=True))

    assert decision.rule_level == "LOW"
    assert "sudden_vertical_drop" not in decision.reasons


def test_stationary_camera_keeps_pixel_fall_rule_active() -> None:
    engine = RuleEngine(
        enable_fall_rule=True,
        suppress_motion_sensitive_rules_on_camera_motion=True,
    )

    decision = engine.evaluate(_feature(camera_motion=False))

    assert decision.rule_level == "CRITICAL"
    assert "sudden_vertical_drop" in decision.reasons


def test_legacy_rule_behavior_is_unchanged_by_default() -> None:
    engine = RuleEngine(enable_fall_rule=True)

    decision = engine.evaluate(_feature(camera_motion=True))

    assert decision.rule_level == "CRITICAL"
    assert "sudden_vertical_drop" in decision.reasons
