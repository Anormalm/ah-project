from __future__ import annotations

from utils.schemas import RiskEvent, RuleDecision


class RiskScorer:
    _rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}

    def __init__(
        self,
        ml_weight: float = 0.35,
        medium_threshold: float = 0.45,
        high_threshold: float = 0.7,
        critical_threshold: float = 0.9,
        allow_ml_level_override: bool = True,
    ) -> None:
        self.ml_weight = ml_weight
        self.medium_threshold = medium_threshold
        self.high_threshold = high_threshold
        self.critical_threshold = critical_threshold
        self.allow_ml_level_override = allow_ml_level_override

    @staticmethod
    def _level_from_prob(prob: float, medium: float, high: float, critical: float) -> str:
        if prob >= critical:
            return "CRITICAL"
        if prob >= high:
            return "HIGH"
        if prob >= medium:
            return "MEDIUM"
        return "LOW"

    def score(self, rule_decision: RuleDecision, ml_probability: float) -> RiskEvent:
        fused = (1.0 - self.ml_weight) * rule_decision.rule_score + self.ml_weight * ml_probability
        ml_level = self._level_from_prob(ml_probability, self.medium_threshold, self.high_threshold, self.critical_threshold)

        final_level = rule_decision.rule_level
        if self.allow_ml_level_override and self._rank[ml_level] > self._rank[final_level]:
            final_level = ml_level

        reasons = list(rule_decision.reasons)
        if ml_probability >= self.high_threshold:
            reasons.append("ml_high_probability")

        return RiskEvent(
            track_id=rule_decision.track_id,
            risk_level=final_level,
            confidence=float(max(0.0, min(1.0, fused))),
            timestamp=rule_decision.timestamp,
            reasons=reasons,
        )

