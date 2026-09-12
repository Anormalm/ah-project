from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ingestion.frame_packet import SensorFrame
from utils.schemas import FeatureVector, TrackPose


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom <= 1e-6:
        return 0.0
    cos_angle = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


@dataclass
class _PrevState:
    center: tuple[float, float]
    velocity: tuple[float, float]
    timestamp: float
    center_3d_m: tuple[float, float, float] | None = None
    velocity_3d_m_s: tuple[float, float, float] | None = None


class FeatureExtractor:
    def __init__(
        self,
        bed_zones: list[tuple[float, float, float, float]] | None = None,
        min_kpt_conf: float = 0.2,
        camera_motion_gyro_threshold_rad_s: float = 0.35,
        camera_motion_accel_delta_threshold_m_s2: float = 2.0,
        kinematic_ema_alpha: float = 1.0,
        max_kinematic_gap_sec: float = 0.5,
        max_3d_speed_m_s: float = 5.0,
    ) -> None:
        self.bed_zones = bed_zones or []
        self.min_kpt_conf = min_kpt_conf
        self.camera_motion_gyro_threshold_rad_s = max(float(camera_motion_gyro_threshold_rad_s), 0.0)
        self.camera_motion_accel_delta_threshold_m_s2 = max(
            float(camera_motion_accel_delta_threshold_m_s2), 0.0
        )
        self.kinematic_ema_alpha = float(np.clip(kinematic_ema_alpha, 0.05, 1.0))
        self.max_kinematic_gap_sec = max(float(max_kinematic_gap_sec), 0.05)
        self.max_3d_speed_m_s = max(float(max_3d_speed_m_s), 0.1)
        self._prev: dict[int, _PrevState] = {}

    def _center_of_mass(self, keypoints: np.ndarray) -> tuple[float, float]:
        # A fixed torso subset is much less sensitive to hands/feet disappearing
        # than averaging every currently visible keypoint.
        torso = keypoints[[5, 6, 11, 12]]
        valid_torso = torso[torso[:, 2] >= self.min_kpt_conf]
        if valid_torso.shape[0] >= 2:
            return (float(valid_torso[:, 0].mean()), float(valid_torso[:, 1].mean()))
        valid = keypoints[keypoints[:, 2] >= self.min_kpt_conf]
        if valid.size == 0:
            return (0.0, 0.0)
        return (float(valid[:, 0].mean()), float(valid[:, 1].mean()))

    @staticmethod
    def _distance_to_rect(point: tuple[float, float], rect: tuple[float, float, float, float]) -> float:
        px, py = point
        x1, y1, x2, y2 = rect
        dx = max(x1 - px, 0.0, px - x2)
        dy = max(y1 - py, 0.0, py - y2)
        return float(np.hypot(dx, dy))

    def _posture(self, keypoints: np.ndarray, joint_angles: dict[str, float]) -> str:
        valid = keypoints[keypoints[:, 2] >= self.min_kpt_conf]
        if valid.shape[0] < 6:
            return "unknown"

        span_x = float(valid[:, 0].max() - valid[:, 0].min())
        span_y = float(valid[:, 1].max() - valid[:, 1].min())
        ratio = span_x / max(span_y, 1e-6)

        if ratio > 1.25:
            return "lying"

        knee = max(joint_angles.get("left_knee", 180.0), joint_angles.get("right_knee", 180.0))
        hip = max(joint_angles.get("left_hip", 180.0), joint_angles.get("right_hip", 180.0))
        if knee < 130 or hip < 130:
            return "sitting"
        return "standing"

    def _joint_angles(self, keypoints: np.ndarray) -> tuple[dict[str, float], float]:
        def p(idx: int) -> np.ndarray:
            return keypoints[idx, :2]

        angles = {
            "left_knee": _angle(p(11), p(13), p(15)),
            "right_knee": _angle(p(12), p(14), p(16)),
            "left_hip": _angle(p(5), p(11), p(13)),
            "right_hip": _angle(p(6), p(12), p(14)),
        }

        shoulder_mid = (p(5) + p(6)) / 2.0
        hip_mid = (p(11) + p(12)) / 2.0
        torso = shoulder_mid - hip_mid
        vertical = np.array([0.0, -1.0], dtype=np.float32)
        denom = np.linalg.norm(torso) * np.linalg.norm(vertical)
        lean_angle = 0.0 if denom <= 1e-6 else float(np.degrees(np.arccos(np.clip(np.dot(torso, vertical) / denom, -1.0, 1.0))))
        return angles, lean_angle

    def _kinematics(
        self,
        track_id: int,
        center: tuple[float, float],
        timestamp: float,
        center_3d_m: tuple[float, float, float] | None = None,
    ) -> tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float, float] | None,
        tuple[float, float, float] | None,
    ]:
        prev = self._prev.get(track_id)
        if prev is None:
            velocity_3d = (0.0, 0.0, 0.0) if center_3d_m is not None else None
            acceleration_3d = (0.0, 0.0, 0.0) if center_3d_m is not None else None
            self._prev[track_id] = _PrevState(
                center=center,
                velocity=(0.0, 0.0),
                timestamp=timestamp,
                center_3d_m=center_3d_m,
                velocity_3d_m_s=velocity_3d,
            )
            return (0.0, 0.0), (0.0, 0.0), velocity_3d, acceleration_3d

        dt = timestamp - prev.timestamp
        if dt <= 1e-6 or dt > self.max_kinematic_gap_sec:
            velocity_3d = (0.0, 0.0, 0.0) if center_3d_m is not None else None
            acceleration_3d = (0.0, 0.0, 0.0) if center_3d_m is not None else None
            self._prev[track_id] = _PrevState(
                center=center,
                velocity=(0.0, 0.0),
                timestamp=timestamp,
                center_3d_m=center_3d_m,
                velocity_3d_m_s=velocity_3d,
            )
            return (0.0, 0.0), (0.0, 0.0), velocity_3d, acceleration_3d

        alpha = self.kinematic_ema_alpha
        raw_vx = (center[0] - prev.center[0]) / dt
        raw_vy = (center[1] - prev.center[1]) / dt
        vx = alpha * raw_vx + (1.0 - alpha) * prev.velocity[0]
        vy = alpha * raw_vy + (1.0 - alpha) * prev.velocity[1]
        ax = (vx - prev.velocity[0]) / dt
        ay = (vy - prev.velocity[1]) / dt

        velocity_3d: tuple[float, float, float] | None = None
        acceleration_3d: tuple[float, float, float] | None = None
        if center_3d_m is not None:
            if prev.center_3d_m is None or prev.velocity_3d_m_s is None:
                velocity_3d = (0.0, 0.0, 0.0)
                acceleration_3d = (0.0, 0.0, 0.0)
            else:
                current = np.asarray(center_3d_m, dtype=np.float64)
                prior = np.asarray(prev.center_3d_m, dtype=np.float64)
                prior_velocity = np.asarray(prev.velocity_3d_m_s, dtype=np.float64)
                raw_velocity = (current - prior) / dt
                if float(np.linalg.norm(raw_velocity)) <= self.max_3d_speed_m_s:
                    velocity_array = alpha * raw_velocity + (1.0 - alpha) * prior_velocity
                    acceleration_array = (velocity_array - prior_velocity) / dt
                    velocity_3d = tuple(float(v) for v in velocity_array)
                    acceleration_3d = tuple(float(v) for v in acceleration_array)

        self._prev[track_id] = _PrevState(
            center=center,
            velocity=(float(vx), float(vy)),
            timestamp=timestamp,
            center_3d_m=center_3d_m,
            velocity_3d_m_s=velocity_3d,
        )
        return (float(vx), float(vy)), (float(ax), float(ay)), velocity_3d, acceleration_3d

    def _depth_features(
        self,
        keypoints: np.ndarray,
        sensor_frame: SensorFrame | None,
    ) -> tuple[tuple[float, float, float] | None, float | None]:
        if (
            sensor_frame is None
            or sensor_frame.aligned_depth is None
            or sensor_frame.depth_scale_m is None
            or sensor_frame.color_intrinsics is None
        ):
            return None, None

        depth = sensor_frame.aligned_depth
        if depth.ndim != 2:
            return None, None
        intrinsics = sensor_frame.color_intrinsics
        valid_keypoints = keypoints[keypoints[:, 2] >= self.min_kpt_conf]
        if valid_keypoints.size == 0:
            return None, 0.0

        height, width = depth.shape
        points_3d: list[tuple[float, float, float]] = []
        for x_value, y_value, _confidence in valid_keypoints:
            x = int(round(float(x_value)))
            y = int(round(float(y_value)))
            if x < 0 or y < 0 or x >= width or y >= height:
                continue
            x1, x2 = max(0, x - 2), min(width, x + 3)
            y1, y2 = max(0, y - 2), min(height, y + 3)
            patch_m = depth[y1:y2, x1:x2].astype(np.float64) * float(sensor_frame.depth_scale_m)
            valid_depth = patch_m[
                np.isfinite(patch_m) & (patch_m >= 0.1) & (patch_m <= 10.0)
            ]
            if valid_depth.size == 0:
                continue
            z = float(np.median(valid_depth))
            if intrinsics.fx <= 0.0 or intrinsics.fy <= 0.0:
                continue
            point_x = (float(x_value) - intrinsics.ppx) * z / intrinsics.fx
            point_y = (float(y_value) - intrinsics.ppy) * z / intrinsics.fy
            points_3d.append((point_x, point_y, z))

        valid_ratio = float(len(points_3d) / max(len(valid_keypoints), 1))
        if not points_3d:
            return None, valid_ratio
        center_3d = tuple(float(v) for v in np.mean(np.asarray(points_3d, dtype=np.float64), axis=0))
        return center_3d, valid_ratio

    def _camera_motion(
        self,
        sensor_frame: SensorFrame | None,
    ) -> tuple[bool, float | None, float | None]:
        if sensor_frame is None:
            return False, None, None

        gyro_peak = None
        if sensor_frame.gyro_samples:
            gyro_peak = max(float(np.linalg.norm(sample.xyz)) for sample in sensor_frame.gyro_samples)

        accel_delta_peak = None
        if sensor_frame.accel_samples:
            gravity = 9.80665
            accel_delta_peak = max(
                abs(float(np.linalg.norm(sample.xyz)) - gravity)
                for sample in sensor_frame.accel_samples
            )

        moving = bool(
            (gyro_peak is not None and gyro_peak >= self.camera_motion_gyro_threshold_rad_s)
            or (
                accel_delta_peak is not None
                and accel_delta_peak >= self.camera_motion_accel_delta_threshold_m_s2
            )
        )
        return moving, gyro_peak, accel_delta_peak

    def extract(self, track_pose: TrackPose, sensor_frame: SensorFrame | None = None) -> FeatureVector:
        keypoints = np.array(track_pose.keypoints, dtype=np.float32)
        valid_keypoints = keypoints[keypoints[:, 2] >= self.min_kpt_conf]
        pose_valid_ratio = float(len(valid_keypoints) / max(len(keypoints), 1))
        body_height_px = (
            float(valid_keypoints[:, 1].max() - valid_keypoints[:, 1].min())
            if len(valid_keypoints) >= 2
            else None
        )
        center = self._center_of_mass(keypoints)
        center_3d, depth_valid_ratio = self._depth_features(keypoints, sensor_frame)
        velocity, acceleration, velocity_3d, acceleration_3d = self._kinematics(
            track_pose.track_id,
            center,
            track_pose.timestamp,
            center_3d,
        )
        joint_angles, lean_angle = self._joint_angles(keypoints)
        posture = self._posture(keypoints, joint_angles)
        camera_motion, gyro_peak, accel_delta_peak = self._camera_motion(sensor_frame)

        if self.bed_zones:
            bed_dist = min(self._distance_to_rect(center, zone) for zone in self.bed_zones)
        else:
            bed_dist = 1e6

        return FeatureVector(
            track_id=track_pose.track_id,
            timestamp=track_pose.timestamp,
            center_of_mass=center,
            velocity=velocity,
            acceleration=acceleration,
            joint_angles=joint_angles,
            posture=posture,
            bed_zone_distance=float(bed_dist),
            lean_angle=float(lean_angle),
            pose_valid_ratio=pose_valid_ratio,
            body_height_px=body_height_px,
            center_of_mass_3d_m=center_3d,
            velocity_3d_m_s=velocity_3d,
            acceleration_3d_m_s2=acceleration_3d,
            depth_valid_ratio=depth_valid_ratio,
            camera_motion=camera_motion,
            camera_gyro_peak_rad_s=gyro_peak,
            camera_accel_delta_peak_m_s2=accel_delta_peak,
        )

