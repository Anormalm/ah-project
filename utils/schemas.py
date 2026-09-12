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
    acceleration: tuple[float, float]
    joint_angles: dict[str, float]
    posture: Literal["lying", "sitting", "standing", "unknown"]
    bed_zone_distance: float
    lean_angle: float
    pose_valid_ratio: float = Field(default=1.0, ge=0.0, le=1.0)
    body_height_px: float | None = Field(default=None, ge=0.0)
    center_of_mass_3d_m: tuple[float, float, float] | None = None
    velocity_3d_m_s: tuple[float, float, float] | None = None
    acceleration_3d_m_s2: tuple[float, float, float] | None = None
    depth_valid_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    camera_motion: bool = False
    camera_gyro_peak_rad_s: float | None = None
    camera_accel_delta_peak_m_s2: float | None = None

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
    event: Literal["stable", "inactivity_risk", "instability_risk", "bed_exit_risk", "fall_detected"] = "stable"
    reasons: list[str] = Field(default_factory=list)


class AlertRecord(BaseModel):
    stream_id: str
    event: RiskEvent

