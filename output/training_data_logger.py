from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path

from utils.schemas import FeatureVector, RiskEvent


class TrainingDataLogger:
    _locks: dict[str, threading.Lock] = {}
    _locks_guard = threading.Lock()
    _process_session_id = uuid.uuid4().hex

    def __init__(self, path: str | None, session_id: str | None = None) -> None:
        self.path = Path(path) if path else None
        self.session_id = session_id or self._process_session_id
        self._lock: threading.Lock | None = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            key = str(self.path.resolve())
            with self._locks_guard:
                if key not in self._locks:
                    self._locks[key] = threading.Lock()
                self._lock = self._locks[key]

    def emit(
        self,
        stream_id: str,
        feature: FeatureVector,
        event: RiskEvent,
        ml_probability: float | None = None,
    ) -> None:
        if self.path is None:
            return
        speed = float((feature.velocity[0] ** 2 + feature.velocity[1] ** 2) ** 0.5)
        acc = float((feature.acceleration[0] ** 2 + feature.acceleration[1] ** 2) ** 0.5)
        payload = {
            "record_type": "frame_features",
            "session_id": self.session_id,
            "stream_id": stream_id,
            "track_id": feature.track_id,
            "timestamp": feature.timestamp,
            "center_of_mass": list(feature.center_of_mass),
            "speed": speed,
            "vy": float(feature.velocity[1]),
            "acc": acc,
            "lean": float(feature.lean_angle),
            "posture": feature.posture,
            "pose_valid_ratio": float(feature.pose_valid_ratio),
            "body_height_px": feature.body_height_px,
            "risk_level": event.risk_level,
            "event": event.event,
            "reasons": list(event.reasons),
            # Runtime risk is a model/rule prediction, not human ground truth.
            "weak_label": 1 if event.risk_level in {"HIGH", "CRITICAL"} else 0,
            "weak_label_confidence": float(event.confidence),
        }
        if feature.center_of_mass_3d_m is not None:
            payload["center_of_mass_3d_m"] = list(feature.center_of_mass_3d_m)
        if feature.velocity_3d_m_s is not None:
            payload["velocity_3d_m_s"] = list(feature.velocity_3d_m_s)
        if feature.acceleration_3d_m_s2 is not None:
            payload["acceleration_3d_m_s2"] = list(feature.acceleration_3d_m_s2)
        if feature.depth_valid_ratio is not None:
            payload["depth_valid_ratio"] = float(feature.depth_valid_ratio)
        payload["camera_motion"] = bool(feature.camera_motion)
        if feature.camera_gyro_peak_rad_s is not None:
            payload["camera_gyro_peak_rad_s"] = float(feature.camera_gyro_peak_rad_s)
        if feature.camera_accel_delta_peak_m_s2 is not None:
            payload["camera_accel_delta_peak_m_s2"] = float(feature.camera_accel_delta_peak_m_s2)
        if ml_probability is not None:
            payload["ml_probability"] = float(ml_probability)
        line = json.dumps(payload, separators=(",", ":"))
        lock = self._lock
        if lock is None:
            return
        with lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
