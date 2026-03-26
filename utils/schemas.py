from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Detection(BaseModel):
    bbox: tuple[float, float, float, float]
    confidence: float = Field(ge=0.0, le=1.0)
    class_id: int = 0
    class_name: str = "person"

    model_config = ConfigDict(frozen=True)


class PoseResult(BaseModel):
    bbox: tuple[float, float, float, float]
    keypoints: list[tuple[float, float, float]]

    model_config = ConfigDict(frozen=True)


class TrackPose(BaseModel):
    track_id: int
    keypoints: list[tuple[float, float, float]]
    timestamp: float

    model_config = ConfigDict(frozen=True)


class FeatureVector(BaseModel):
    track_id: int
    timestamp: float
    center_of_mass: tuple[float, float]
    velocity: tuple[float, float]
    joint_angles: dict[str, float]
    posture: Literal["lying", "sitting", "standing", "unknown"]
    bed_zone_distance: float
    lean_angle: float

    model_config = ConfigDict(frozen=True)


class RuleDecision(BaseModel):
    track_id: int
    timestamp: float
    rule_score: float = Field(ge=0.0, le=1.0)
    rule_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    reasons: list[str]


class RiskEvent(BaseModel):
    track_id: int
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp: float
    reasons: list[str] = Field(default_factory=list)


class AlertRecord(BaseModel):
    stream_id: str
    event: RiskEvent

