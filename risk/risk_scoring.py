from __future__ import annotations

from dataclasses import dataclass

from utils.schemas import RiskEvent, RuleDecision


@dataclass
class _RiskState:
    ema_score: float
    last_level: str
    last_level_change_ts: float


class RiskScorer:
    _rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}

    def __init__(
        self,
        ml_weight: float = 0.35,
        medium_threshold: float = 0.45,
        high_threshold: float = 0.7,
        critical_threshold: float = 0.9,
        allow_ml_level_override: bool = True,
        ema_alpha: float = 0.35,
        downgrade_grace_sec: float = 2.0,
    ) -> None:
        self.ml_weight = ml_weight
        self.medium_threshold = medium_threshold
        self.high_threshold = high_threshold
        self.critical_threshold = critical_threshold
        self.allow_ml_level_override = allow_ml_level_override
        self.ema_alpha = ema_alpha
        self.downgrade_grace_sec = downgrade_grace_sec
        self._state: dict[int, _RiskState] = {}

    @staticmethod
    def _level_from_prob(prob: float, medium: float, high: float, critical: float) -> str:
        if prob >= critical:
            return "CRITICAL"
        if prob >= high:
            return "HIGH"
        if prob >= medium:
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def _event_type(level: str, reasons: list[str]) -> str:
        if "sudden_vertical_drop" in reasons or level == "CRITICAL":
            return "fall_detected"
        if "lean_instability" in reasons or "repeated_sit_stand_transitions" in reasons or level == "HIGH":
            return "instability_risk"
        if "sitting_at_edge" in reasons:
            return "bed_exit_risk"
        if "prolonged_inactivity" in reasons:
            return "inactivity_risk"
        return "stable"

    def _stabilize(self, track_id: int, raw_score: float, raw_level: str, ts: float) -> tuple[float, str]:
        prev = self._state.get(track_id)
        if prev is None:
            self._state[track_id] = _RiskState(
                ema_score=raw_score,
                last_level=raw_level,
                last_level_change_ts=ts,
            )
            return raw_score, raw_level

        ema = self.ema_alpha * raw_score + (1.0 - self.ema_alpha) * prev.ema_score
        ema_level = self._level_from_prob(ema, self.medium_threshold, self.high_threshold, self.critical_threshold)

        # Avoid rapid downgrades while allowing immediate escalation.
        if self._rank[ema_level] < self._rank[prev.last_level] and (ts - prev.last_level_change_ts) < self.downgrade_grace_sec:
            stable_level = prev.last_level
            level_change_ts = prev.last_level_change_ts
        else:
            stable_level = ema_level if self._rank[ema_level] >= self._rank[raw_level] else raw_level
            level_change_ts = ts if stable_level != prev.last_level else prev.last_level_change_ts

        self._state[track_id] = _RiskState(
            ema_score=ema,
            last_level=stable_level,
            last_level_change_ts=level_change_ts,
        )
        return ema, stable_level

    def score(self, rule_decision: RuleDecision, ml_probability: float) -> RiskEvent:
        fused = (1.0 - self.ml_weight) * rule_decision.rule_score + self.ml_weight * ml_probability
        ml_level = self._level_from_prob(ml_probability, self.medium_threshold, self.high_threshold, self.critical_threshold)

        raw_level = rule_decision.rule_level
        if self.allow_ml_level_override and self._rank[ml_level] > self._rank[raw_level]:
            raw_level = ml_level

        confidence, final_level = self._stabilize(rule_decision.track_id, fused, raw_level, rule_decision.timestamp)

        reasons = list(rule_decision.reasons)
        if ml_probability >= self.high_threshold:
            reasons.append("ml_high_probability")

        return RiskEvent(
            track_id=rule_decision.track_id,
            risk_level=final_level,
            confidence=float(max(0.0, min(1.0, confidence))),
            timestamp=rule_decision.timestamp,
            event=self._event_type(final_level, reasons),
            reasons=reasons,
        )

