from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True, slots=True)
class MotionSample:
    """One accelerometer or gyroscope sample in the device clock domain."""

    xyz: tuple[float, float, float]
    device_timestamp_ms: float
    timestamp_domain: str


@dataclass(frozen=True, slots=True)
class PinholeIntrinsics:
    """Minimal pinhole parameters needed to deproject aligned depth pixels."""

    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float


@dataclass(slots=True)
class SensorFrame:
    """Optional sensor data captured alongside a pipeline inference image.

    ``aligned_depth`` contains raw Z16 values. Multiply a non-zero value by
    ``depth_scale_m`` to convert it to metres. Infrared frames are keyed by
    their one-based RealSense stream index.
    """

    aligned_depth: Optional[np.ndarray] = None
    depth_scale_m: Optional[float] = None
    color_intrinsics: Optional[PinholeIntrinsics] = None
    infrared: dict[int, np.ndarray] = field(default_factory=dict)
    accel_samples: tuple[MotionSample, ...] = ()
    gyro_samples: tuple[MotionSample, ...] = ()
    device_timestamp_ms: Optional[float] = None
    frame_number: Optional[int] = None
    usb_type: Optional[str] = None
    primary_stream: str = "color"


@dataclass(slots=True)
class FramePacket:
    """A single image for inference plus optional synchronized sensor data.

    ``timestamp`` is host wall-clock time in seconds, preserving the semantics
    used by the existing tracking, rules, and alert pipeline. Device-clock
    timestamps live in ``sensor`` and must not be mixed with it directly.
    """

    frame: Optional[np.ndarray]
    timestamp: float
    sensor: Optional[SensorFrame] = None
