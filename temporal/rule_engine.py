from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from utils.schemas import FeatureVector, RuleDecision


@dataclass(frozen=True)
class _FallSample:
    timestamp: float
    center_y: float
    center_3d_y_m: float | None
    vy: float
    ay: float
    posture: str
    lean_angle: float
    body_height_px: float | None


@dataclass
class _RuleState:
    sitting_start_ts: float | None = None
    lean_history: deque[float] = field(default_factory=lambda: deque(maxlen=12))
    posture_history: deque[str] = field(default_factory=lambda: deque(maxlen=24))
    transition_timestamps: deque[float] = field(default_factory=lambda: deque(maxlen=32))
    still_start_ts: float | None = None
    seen_frames: int = 0
    fall_history: deque[_FallSample] = field(default_factory=lambda: deque(maxlen=90))
    fall_candidate_ts: float | None = None
    last_fall_ts: float | None = None
    last_camera_motion_ts: float | None = None


class RuleEngine:
    _severity_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}

    def __init__(
        self,
        bed_edge_distance_px: float = 40.0,
        sitting_edge_seconds: float = 5.0,
        sudden_drop_vy: float = 250.0,
        sudden_drop_ay: float = 1200.0,
        lean_angle_deg: float = 30.0,
        lean_std_deg: float = 8.0,
        inactivity_speed_px_s: float = 8.0,
        inactivity_seconds: float = 20.0,
        transition_window_sec: float = 15.0,
        transition_threshold: int = 4,
        enable_sitting_edge_rule: bool = True,
        enable_fall_rule: bool = True,
        enable_lean_instability_rule: bool = True,
        enable_inactivity_rule: bool = True,
        enable_transition_rule: bool = True,
        suppress_motion_sensitive_rules_on_camera_motion: bool = False,
        require_fall_confirmation: bool = False,
        fall_motion_window_sec: float = 0.75,
        fall_confirmation_window_sec: float = 1.25,
        fall_min_track_frames: int = 8,
        fall_min_pose_valid_ratio: float = 0.5,
        fall_require_depth: bool = False,
        fall_require_metric_drop: bool = False,
        fall_min_depth_valid_ratio: float = 0.5,
        fall_min_drop_px: float = 45.0,
        fall_min_drop_body_ratio: float = 0.22,
        fall_min_drop_m: float = 0.25,
        fall_confirm_lying_frames: int = 3,
        fall_confirm_lean_angle_deg: float = 55.0,
        fall_camera_settle_sec: float = 1.0,
        fall_cooldown_sec: float = 8.0,
    ) -> None:
        self.bed_edge_distance_px = bed_edge_distance_px
        self.sitting_edge_seconds = sitting_edge_seconds
        self.sudden_drop_vy = sudden_drop_vy
        self.sudden_drop_ay = sudden_drop_ay
        self.lean_angle_deg = lean_angle_deg
        self.lean_std_deg = lean_std_deg
        self.inactivity_speed_px_s = inactivity_speed_px_s
        self.inactivity_seconds = inactivity_seconds
        self.transition_window_sec = transition_window_sec
        self.transition_threshold = transition_threshold
        self.enable_sitting_edge_rule = enable_sitting_edge_rule
        self.enable_fall_rule = enable_fall_rule
        self.enable_lean_instability_rule = enable_lean_instability_rule
        self.enable_inactivity_rule = enable_inactivity_rule
        self.enable_transition_rule = enable_transition_rule
        self.suppress_motion_sensitive_rules_on_camera_motion = suppress_motion_sensitive_rules_on_camera_motion
        self.require_fall_confirmation = bool(require_fall_confirmation)
        self.fall_motion_window_sec = max(float(fall_motion_window_sec), 0.1)
        self.fall_confirmation_window_sec = max(float(fall_confirmation_window_sec), 0.1)
        self.fall_min_track_frames = max(int(fall_min_track_frames), 1)
        self.fall_min_pose_valid_ratio = float(np.clip(fall_min_pose_valid_ratio, 0.0, 1.0))
        self.fall_require_depth = bool(fall_require_depth)
        self.fall_require_metric_drop = bool(fall_require_metric_drop)
        self.fall_min_depth_valid_ratio = float(np.clip(fall_min_depth_valid_ratio, 0.0, 1.0))
        self.fall_min_drop_px = max(float(fall_min_drop_px), 0.0)
        self.fall_min_drop_body_ratio = max(float(fall_min_drop_body_ratio), 0.0)
        self.fall_min_drop_m = max(float(fall_min_drop_m), 0.0)
        self.fall_confirm_lying_frames = max(int(fall_confirm_lying_frames), 1)
        self.fall_confirm_lean_angle_deg = max(float(fall_confirm_lean_angle_deg), 0.0)
        self.fall_camera_settle_sec = max(float(fall_camera_settle_sec), 0.0)
        self.fall_cooldown_sec = max(float(fall_cooldown_sec), 0.0)
        self._state: dict[int, _RuleState] = {}

    def _elevate(self, current_level: str, candidate: str) -> str:
        if self._severity_rank[candidate] > self._severity_rank[current_level]:
            return candidate
        return current_level

    def _confirmed_fall_reasons(
        self,
        feature: FeatureVector,
        state: _RuleState,
        camera_motion_suppressed: bool,
    ) -> list[str]:
        if camera_motion_suppressed:
            state.fall_history.clear()
            state.fall_candidate_ts = None
            return []

        if feature.pose_valid_ratio < self.fall_min_pose_valid_ratio:
            state.fall_candidate_ts = None
            return []

        depth_is_valid = bool(
            feature.center_of_mass_3d_m is not None
            and feature.depth_valid_ratio is not None
            and feature.depth_valid_ratio >= self.fall_min_depth_valid_ratio
        )
        if self.fall_require_depth and not depth_is_valid:
            state.fall_candidate_ts = None
            return []

        center_3d_y = feature.center_of_mass_3d_m[1] if depth_is_valid else None
        state.fall_history.append(
            _FallSample(
                timestamp=feature.timestamp,
                center_y=float(feature.center_of_mass[1]),
                center_3d_y_m=float(center_3d_y) if center_3d_y is not None else None,
                vy=float(feature.velocity[1]),
                ay=float(feature.acceleration[1]),
                posture=feature.posture,
                lean_angle=float(feature.lean_angle),
                body_height_px=feature.body_height_px,
            )
        )

        keep_sec = max(self.fall_motion_window_sec, self.fall_confirmation_window_sec) + 0.5
        while state.fall_history and feature.timestamp - state.fall_history[0].timestamp > keep_sec:
            state.fall_history.popleft()

        if state.seen_frames < self.fall_min_track_frames:
            return []
        if state.last_fall_ts is not None and feature.timestamp - state.last_fall_ts < self.fall_cooldown_sec:
            return []

        recent = [
            sample
            for sample in state.fall_history
            if feature.timestamp - sample.timestamp <= self.fall_motion_window_sec
        ]
        if len(recent) >= 3:
            first = recent[0]
            last = recent[-1]
            drop_px = last.center_y - first.center_y
            heights = [s.body_height_px for s in recent if s.body_height_px and s.body_height_px > 1.0]
            body_height = float(np.median(np.asarray(heights, dtype=np.float32))) if heights else None
            normalized_drop = drop_px / body_height if body_height else 0.0

            metric_drop = 0.0
            if first.center_3d_y_m is not None and last.center_3d_y_m is not None:
                metric_drop = last.center_3d_y_m - first.center_3d_y_m

            peak_vy = max(sample.vy for sample in recent)
            peak_ay = max(sample.ay for sample in recent)
            rapid_motion = peak_vy >= self.sudden_drop_vy or peak_ay >= self.sudden_drop_ay
            if self.fall_require_metric_drop:
                meaningful_drop = metric_drop >= self.fall_min_drop_m
            else:
                meaningful_drop = bool(
                    drop_px >= self.fall_min_drop_px
                    or normalized_drop >= self.fall_min_drop_body_ratio
                    or metric_drop >= self.fall_min_drop_m
                )
            if rapid_motion and meaningful_drop and state.fall_candidate_ts is None:
                state.fall_candidate_ts = feature.timestamp

        candidate_ts = state.fall_candidate_ts
        if candidate_ts is None:
            return []
        if feature.timestamp - candidate_ts > self.fall_confirmation_window_sec:
            state.fall_candidate_ts = None
            return []

        terminal_count = max(self.fall_confirm_lying_frames, 3)
        terminal = list(state.fall_history)[-terminal_count:]
        lying_frames = sum(sample.posture == "lying" for sample in terminal)
        leaned_frames = sum(sample.lean_angle >= self.fall_confirm_lean_angle_deg for sample in terminal)
        # Aspect-ratio posture alone is unreliable for close/cropped people.
        # Confirm horizontal posture only when the torso angle agrees.
        confirmed_posture = bool(
            lying_frames >= self.fall_confirm_lying_frames
            and leaned_frames >= max(2, self.fall_confirm_lying_frames - 1)
        )
        if not confirmed_posture:
            return []

        state.last_fall_ts = feature.timestamp
        state.fall_candidate_ts = None
        reasons = ["confirmed_fall", "rapid_downward_motion", "lying_after_drop"]
        if depth_is_valid:
            reasons.append("depth_validated")
        return reasons

    def evaluate(self, feature: FeatureVector) -> RuleDecision:
        state = self._state.setdefault(feature.track_id, _RuleState())
        state.seen_frames += 1
        level = "LOW"
        score = 0.1
        reasons: list[str] = []
        if feature.camera_motion:
            state.last_camera_motion_ts = feature.timestamp
        camera_settling = bool(
            state.last_camera_motion_ts is not None
            and feature.timestamp - state.last_camera_motion_ts < self.fall_camera_settle_sec
        )
        camera_motion_suppressed = bool(
            self.suppress_motion_sensitive_rules_on_camera_motion
            and (feature.camera_motion or camera_settling)
        )
        if camera_motion_suppressed:
            state.posture_history.clear()
            state.transition_timestamps.clear()
        else:
            state.posture_history.append(feature.posture)

        if self.enable_sitting_edge_rule and feature.posture == "sitting" and feature.bed_zone_distance <= self.bed_edge_distance_px:
            if state.sitting_start_ts is None:
                state.sitting_start_ts = feature.timestamp
            elapsed = feature.timestamp - state.sitting_start_ts
            if elapsed >= self.sitting_edge_seconds:
                level = self._elevate(level, "MEDIUM")
                score = max(score, 0.55)
                reasons.append("sitting_at_edge")
        else:
            state.sitting_start_ts = None

        if self.enable_fall_rule:
            if self.require_fall_confirmation:
                fall_reasons = self._confirmed_fall_reasons(
                    feature,
                    state,
                    camera_motion_suppressed=camera_motion_suppressed,
                )
                if fall_reasons:
                    level = self._elevate(level, "CRITICAL")
                    score = max(score, 0.96)
                    reasons.extend(fall_reasons)
            else:
                vy = feature.velocity[1]
                ay = feature.acceleration[1]
                if not camera_motion_suppressed and (
                    vy >= self.sudden_drop_vy or ay >= self.sudden_drop_ay
                ):
                    level = self._elevate(level, "CRITICAL")
                    score = max(score, 0.96)
                    reasons.append("sudden_vertical_drop")
                    if ay >= self.sudden_drop_ay:
                        reasons.append("high_vertical_acceleration")

        if camera_motion_suppressed:
            state.lean_history.clear()
        else:
            state.lean_history.append(feature.lean_angle)
            if self.enable_lean_instability_rule and len(state.lean_history) >= 5:
                lean_std = float(np.std(np.array(state.lean_history, dtype=np.float32)))
                if feature.lean_angle >= self.lean_angle_deg and lean_std >= self.lean_std_deg:
                    level = self._elevate(level, "HIGH")
                    score = max(score, 0.82)
                    reasons.append("lean_instability")

        if self.enable_transition_rule and len(state.posture_history) >= 2:
            prev = state.posture_history[-2]
            cur = state.posture_history[-1]
            if {prev, cur} == {"sitting", "standing"}:
                state.transition_timestamps.append(feature.timestamp)
            while state.transition_timestamps and feature.timestamp - state.transition_timestamps[0] > self.transition_window_sec:
                state.transition_timestamps.popleft()
            if len(state.transition_timestamps) >= self.transition_threshold:
                level = self._elevate(level, "HIGH")
                score = max(score, 0.78)
                reasons.append("repeated_sit_stand_transitions")

        if self.enable_inactivity_rule:
            speed = float(np.hypot(feature.velocity[0], feature.velocity[1]))
            if speed <= self.inactivity_speed_px_s and feature.posture in {"lying", "sitting"}:
                if state.still_start_ts is None:
                    state.still_start_ts = feature.timestamp
                inactivity_elapsed = feature.timestamp - state.still_start_ts
                if inactivity_elapsed >= self.inactivity_seconds:
                    level = self._elevate(level, "MEDIUM")
                    score = max(score, 0.62)
                    reasons.append("prolonged_inactivity")
            else:
                state.still_start_ts = None

        return RuleDecision(
            track_id=feature.track_id,
            timestamp=feature.timestamp,
            rule_score=score,
            rule_level=level,
            reasons=reasons,
        )

