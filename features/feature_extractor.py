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
    def __init__(
        self,
        bed_zones: list[tuple[float, float, float, float]] | None = None,
        min_kpt_conf: float = 0.2,
        center_ema_alpha: float = 0.35,
    ) -> None:
        self.bed_zones = bed_zones or []
        self.min_kpt_conf = min_kpt_conf
        self.center_ema_alpha = float(np.clip(center_ema_alpha, 0.05, 1.0))
        self._prev: dict[int, _PrevState] = {}

    def remove_track(self, track_id: int) -> None:
        self._prev.pop(int(track_id), None)

    def _center_of_mass(self, keypoints: np.ndarray) -> tuple[float, float]:
        # Motion must not jump when low-confidence face/ankle/wrist points appear
        # or disappear. Prefer the torso, which is both stable and relevant to
        # vertical fall motion, then fall back to all visible landmarks.
        torso_indices = [idx for idx in (5, 6, 11, 12) if idx < keypoints.shape[0]]
        torso = keypoints[torso_indices]
        torso = torso[torso[:, 2] >= self.min_kpt_conf]
        if torso.shape[0] >= 2:
            return (float(torso[:, 0].mean()), float(torso[:, 1].mean()))
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

        def visible(*indices: int) -> bool:
            return all(idx < keypoints.shape[0] and keypoints[idx, 2] >= self.min_kpt_conf for idx in indices)

        angles = {
            "left_knee": _angle(p(11), p(13), p(15)) if visible(11, 13, 15) else 180.0,
            "right_knee": _angle(p(12), p(14), p(16)) if visible(12, 14, 16) else 180.0,
            "left_hip": _angle(p(5), p(11), p(13)) if visible(5, 11, 13) else 180.0,
            "right_hip": _angle(p(6), p(12), p(14)) if visible(6, 12, 14) else 180.0,
        }

        if not visible(5, 6, 11, 12):
            return angles, 0.0
        shoulder_mid = (p(5) + p(6)) / 2.0
        hip_mid = (p(11) + p(12)) / 2.0
        torso = shoulder_mid - hip_mid
        vertical = np.array([0.0, -1.0], dtype=np.float32)
        denom = np.linalg.norm(torso) * np.linalg.norm(vertical)
        lean_angle = 0.0 if denom <= 1e-6 else float(np.degrees(np.arccos(np.clip(np.dot(torso, vertical) / denom, -1.0, 1.0))))
        return angles, lean_angle

    def _normalize_keypoints(self, keypoints: np.ndarray) -> list[tuple[float, float, float]]:
        """Return translation/scale-invariant COCO skeleton coordinates."""
        if keypoints.ndim != 2 or keypoints.shape[1] < 3:
            return []
        visible = keypoints[:, 2] >= self.min_kpt_conf
        if not np.any(visible):
            return [(0.0, 0.0, 0.0) for _ in range(keypoints.shape[0])]

        if keypoints.shape[0] > 12 and visible[11] and visible[12]:
            origin = (keypoints[11, :2] + keypoints[12, :2]) * 0.5
        elif keypoints.shape[0] > 6 and visible[5] and visible[6]:
            origin = (keypoints[5, :2] + keypoints[6, :2]) * 0.5
        else:
            origin = keypoints[visible, :2].mean(axis=0)

        visible_xy = keypoints[visible, :2]
        span = visible_xy.max(axis=0) - visible_xy.min(axis=0)
        scale = max(float(np.hypot(span[0], span[1])), 1.0)
        normalized = np.zeros((keypoints.shape[0], 3), dtype=np.float32)
        normalized[:, :2] = (keypoints[:, :2] - origin) / scale
        normalized[:, 2] = np.clip(keypoints[:, 2], 0.0, 1.0)
        normalized[~visible, :2] = 0.0
        return [tuple(float(v) for v in point) for point in normalized]

    def _pose_quality(self, keypoints: np.ndarray) -> float:
        if keypoints.ndim != 2 or keypoints.shape[0] == 0 or keypoints.shape[1] < 3:
            return 0.0
        confidence = np.clip(keypoints[:, 2], 0.0, 1.0)
        visible = confidence >= self.min_kpt_conf
        if not np.any(visible):
            return 0.0
        # Penalize fragments even when their few remaining joints are confident.
        visible_ratio = float(np.mean(visible))
        visible_confidence = float(np.mean(confidence[visible]))
        return float(np.clip(visible_ratio * visible_confidence, 0.0, 1.0))

    def _kinematics(
        self,
        track_id: int,
        center: tuple[float, float],
        timestamp: float,
    ) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        prev = self._prev.get(track_id)
        if prev is None:
            self._prev[track_id] = _PrevState(center=center, velocity=(0.0, 0.0), timestamp=timestamp)
            return center, (0.0, 0.0), (0.0, 0.0)

        alpha = self.center_ema_alpha
        smoothed = (
            alpha * center[0] + (1.0 - alpha) * prev.center[0],
            alpha * center[1] + (1.0 - alpha) * prev.center[1],
        )
        dt = max(timestamp - prev.timestamp, 1.0 / 120.0)
        vx = (smoothed[0] - prev.center[0]) / dt
        vy = (smoothed[1] - prev.center[1]) / dt
        ax = (vx - prev.velocity[0]) / dt
        ay = (vy - prev.velocity[1]) / dt
        self._prev[track_id] = _PrevState(center=smoothed, velocity=(float(vx), float(vy)), timestamp=timestamp)
        return smoothed, (float(vx), float(vy)), (float(ax), float(ay))

    def extract(self, track_pose: TrackPose) -> FeatureVector:
        keypoints = np.array(track_pose.keypoints, dtype=np.float32)
        raw_center = self._center_of_mass(keypoints)
        center, velocity, acceleration = self._kinematics(track_pose.track_id, raw_center, track_pose.timestamp)
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
            pose_quality=self._pose_quality(keypoints),
            normalized_keypoints=self._normalize_keypoints(keypoints),
        )
