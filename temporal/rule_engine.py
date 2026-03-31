from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from utils.schemas import FeatureVector, RuleDecision


@dataclass
class _RuleState:
    sitting_start_ts: float | None = None
    lean_history: deque[float] = field(default_factory=lambda: deque(maxlen=12))
    posture_history: deque[str] = field(default_factory=lambda: deque(maxlen=24))
    transition_timestamps: deque[float] = field(default_factory=lambda: deque(maxlen=32))
    still_start_ts: float | None = None


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
        self._state: dict[int, _RuleState] = {}

    def _elevate(self, current_level: str, candidate: str) -> str:
        if self._severity_rank[candidate] > self._severity_rank[current_level]:
            return candidate
        return current_level

    def evaluate(self, feature: FeatureVector) -> RuleDecision:
        state = self._state.setdefault(feature.track_id, _RuleState())
        level = "LOW"
        score = 0.1
        reasons: list[str] = []
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

        vy = feature.velocity[1]
        ay = feature.acceleration[1]
        if self.enable_fall_rule and (vy >= self.sudden_drop_vy or ay >= self.sudden_drop_ay):
            level = self._elevate(level, "CRITICAL")
            score = max(score, 0.96)
            reasons.append("sudden_vertical_drop")
            if ay >= self.sudden_drop_ay:
                reasons.append("high_vertical_acceleration")

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

