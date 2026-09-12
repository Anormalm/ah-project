from __future__ import annotations

import math

import numpy as np
import pytest

from features.feature_extractor import FeatureExtractor
from ingestion.frame_packet import MotionSample, PinholeIntrinsics, SensorFrame
from utils.schemas import TrackPose


def _track(
    *,
    track_id: int = 7,
    timestamp: float = 1.0,
    pixel: tuple[float, float] = (3.0, 4.0),
    confidence: float = 1.0,
) -> TrackPose:
    x, y = pixel
    return TrackPose(
        track_id=track_id,
        timestamp=timestamp,
        keypoints=[(x, y, confidence) for _ in range(17)],
    )


def _motion(xyz: tuple[float, float, float], timestamp_ms: float = 990.0) -> MotionSample:
    return MotionSample(
        xyz=xyz,
        device_timestamp_ms=timestamp_ms,
        timestamp_domain="hardware_clock",
    )


def _sensor(
    depth: np.ndarray,
    *,
    depth_scale_m: float = 0.001,
    accel_samples: list[MotionSample] | None = None,
    gyro_samples: list[MotionSample] | None = None,
    frame_number: int = 1,
) -> SensorFrame:
    height, width = depth.shape
    return SensorFrame(
        aligned_depth=depth,
        depth_scale_m=depth_scale_m,
        color_intrinsics=PinholeIntrinsics(
            width=width,
            height=height,
            fx=100.0,
            fy=100.0,
            ppx=2.0,
            ppy=2.0,
        ),
        infrared={},
        accel_samples=accel_samples or [],
        gyro_samples=gyro_samples or [],
        device_timestamp_ms=1000.0,
        frame_number=frame_number,
        usb_type="3.2",
    )


def _assert_finite_triplet(values: tuple[float, float, float] | None) -> None:
    assert values is not None
    assert all(math.isfinite(value) for value in values)


def test_depth_is_scaled_once_and_deprojected_with_color_intrinsics() -> None:
    depth = np.full((9, 9), 2000, dtype=np.uint16)
    extractor = FeatureExtractor()

    feature = extractor.extract(_track(pixel=(3.0, 4.0)), _sensor(depth))

    assert feature.center_of_mass_3d_m == pytest.approx((0.02, 0.04, 2.0))
    assert feature.velocity_3d_m_s == pytest.approx((0.0, 0.0, 0.0))
    assert feature.acceleration_3d_m_s2 == pytest.approx((0.0, 0.0, 0.0))
    assert feature.depth_valid_ratio == pytest.approx(1.0)


def test_depth_patch_median_ignores_holes_nonfinite_and_out_of_range_values() -> None:
    depth = np.full((9, 9), 2000.0, dtype=np.float32)
    # All four values are inside the 5x5 sampling patch centered on (3, 4).
    depth[4, 3] = 0.0
    depth[3, 3] = np.nan
    depth[4, 4] = 20_000.0  # 20 m: beyond the accepted 10 m maximum.
    depth[5, 3] = 50.0  # 0.05 m: below the accepted 0.1 m minimum.
    extractor = FeatureExtractor()

    feature = extractor.extract(_track(pixel=(3.0, 4.0)), _sensor(depth))

    assert feature.center_of_mass_3d_m == pytest.approx((0.02, 0.04, 2.0))
    assert feature.depth_valid_ratio == pytest.approx(1.0)
    _assert_finite_triplet(feature.center_of_mass_3d_m)


def test_missing_or_unusable_depth_produces_optional_values_without_nan() -> None:
    depth = np.zeros((9, 9), dtype=np.float32)
    depth[2:7, 1:6] = np.nan
    depth[2, 1] = 0.0
    depth[2, 2] = 50.0
    depth[2, 3] = 20_000.0
    extractor = FeatureExtractor()

    feature = extractor.extract(_track(pixel=(3.0, 4.0)), _sensor(depth))

    assert feature.center_of_mass_3d_m is None
    assert feature.velocity_3d_m_s is None
    assert feature.acceleration_3d_m_s2 is None
    assert feature.depth_valid_ratio == pytest.approx(0.0)


def test_low_confidence_keypoints_are_not_assigned_depth() -> None:
    depth = np.full((9, 9), 2000, dtype=np.uint16)
    extractor = FeatureExtractor(min_kpt_conf=0.2)

    feature = extractor.extract(
        _track(pixel=(3.0, 4.0), confidence=0.19),
        _sensor(depth),
    )

    assert feature.center_of_mass_3d_m is None
    assert feature.velocity_3d_m_s is None
    assert feature.acceleration_3d_m_s2 is None
    assert feature.depth_valid_ratio == pytest.approx(0.0)


def test_metric_velocity_and_acceleration_use_track_timestamps() -> None:
    depth = np.full((9, 9), 2000, dtype=np.uint16)
    extractor = FeatureExtractor()

    first = extractor.extract(
        _track(timestamp=1.0, pixel=(3.0, 4.0)),
        _sensor(depth, frame_number=1),
    )
    second = extractor.extract(
        _track(timestamp=1.1, pixel=(4.0, 4.0)),
        _sensor(depth, frame_number=2),
    )

    assert first.velocity_3d_m_s == pytest.approx((0.0, 0.0, 0.0))
    assert first.acceleration_3d_m_s2 == pytest.approx((0.0, 0.0, 0.0))
    assert second.center_of_mass_3d_m == pytest.approx((0.04, 0.04, 2.0))
    assert second.velocity_3d_m_s == pytest.approx((0.2, 0.0, 0.0))
    assert second.acceleration_3d_m_s2 == pytest.approx((2.0, 0.0, 0.0))


def test_metric_kinematics_state_is_isolated_by_track_id() -> None:
    depth = np.full((9, 9), 2000, dtype=np.uint16)
    extractor = FeatureExtractor()

    extractor.extract(_track(track_id=1, timestamp=1.0, pixel=(3.0, 4.0)), _sensor(depth))
    other_track = extractor.extract(
        _track(track_id=2, timestamp=1.1, pixel=(4.0, 4.0)),
        _sensor(depth, frame_number=2),
    )

    assert other_track.velocity_3d_m_s == pytest.approx((0.0, 0.0, 0.0))
    assert other_track.acceleration_3d_m_s2 == pytest.approx((0.0, 0.0, 0.0))


@pytest.mark.parametrize(
    ("accel_samples", "gyro_samples", "expected"),
    [
        ([], [], False),
        ([_motion((0.0, 9.80665, 0.0))], [_motion((0.0, 0.0, 0.0))], False),
        ([_motion((0.0, 9.80665, 0.0))], [_motion((0.36, 0.0, 0.0))], True),
        ([_motion((12.0, 0.0, 0.0))], [_motion((0.0, 0.0, 0.0))], True),
    ],
)
def test_camera_motion_uses_gyro_and_gravity_compensated_acceleration(
    accel_samples: list[MotionSample],
    gyro_samples: list[MotionSample],
    expected: bool,
) -> None:
    depth = np.full((9, 9), 2000, dtype=np.uint16)
    extractor = FeatureExtractor()

    feature = extractor.extract(
        _track(),
        _sensor(depth, accel_samples=accel_samples, gyro_samples=gyro_samples),
    )

    assert feature.camera_motion is expected


def test_camera_motion_flag_does_not_discard_metric_kinematics() -> None:
    depth = np.full((9, 9), 2000, dtype=np.uint16)
    moving_sensor = _sensor(
        depth,
        gyro_samples=[_motion((1.0, 0.0, 0.0))],
    )
    extractor = FeatureExtractor()

    extractor.extract(_track(timestamp=1.0, pixel=(3.0, 4.0)), moving_sensor)
    feature = extractor.extract(
        _track(timestamp=1.1, pixel=(4.0, 4.0)),
        _sensor(depth, gyro_samples=[_motion((1.0, 0.0, 0.0))], frame_number=2),
    )

    assert feature.camera_motion is True
    assert feature.velocity_3d_m_s == pytest.approx((0.2, 0.0, 0.0))
    assert feature.acceleration_3d_m_s2 == pytest.approx((2.0, 0.0, 0.0))


def test_legacy_extraction_without_sensor_data_remains_supported() -> None:
    feature = FeatureExtractor().extract(_track(), None)

    assert feature.center_of_mass_3d_m is None
    assert feature.velocity_3d_m_s is None
    assert feature.acceleration_3d_m_s2 is None
    assert feature.depth_valid_ratio is None
    assert feature.camera_motion is False
