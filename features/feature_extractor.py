from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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


class FeatureExtractor:
    def __init__(self, bed_zones: list[tuple[float, float, float, float]] | None = None, min_kpt_conf: float = 0.2) -> None:
        self.bed_zones = bed_zones or []
        self.min_kpt_conf = min_kpt_conf
        self._prev: dict[int, _PrevState] = {}

    def _center_of_mass(self, keypoints: np.ndarray) -> tuple[float, float]:
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

    def _kinematics(self, track_id: int, center: tuple[float, float], timestamp: float) -> tuple[tuple[float, float], tuple[float, float]]:
        prev = self._prev.get(track_id)
        if prev is None:
            self._prev[track_id] = _PrevState(center=center, velocity=(0.0, 0.0), timestamp=timestamp)
            return (0.0, 0.0), (0.0, 0.0)

        dt = max(timestamp - prev.timestamp, 1e-6)
        vx = (center[0] - prev.center[0]) / dt
        vy = (center[1] - prev.center[1]) / dt
        ax = (vx - prev.velocity[0]) / dt
        ay = (vy - prev.velocity[1]) / dt
        self._prev[track_id] = _PrevState(center=center, velocity=(float(vx), float(vy)), timestamp=timestamp)
        return (float(vx), float(vy)), (float(ax), float(ay))

    def extract(self, track_pose: TrackPose) -> FeatureVector:
        keypoints = np.array(track_pose.keypoints, dtype=np.float32)
        center = self._center_of_mass(keypoints)
        velocity, acceleration = self._kinematics(track_pose.track_id, center, track_pose.timestamp)
        joint_angles, lean_angle = self._joint_angles(keypoints)
        posture = self._posture(keypoints, joint_angles)

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
        )

