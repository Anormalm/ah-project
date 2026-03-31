from __future__ import annotations

import json
import threading
from pathlib import Path

from utils.schemas import FeatureVector, RiskEvent


class TrainingDataLogger:
    _locks: dict[str, threading.Lock] = {}
    _locks_guard = threading.Lock()

    def __init__(self, path: str | None) -> None:
        self.path = Path(path) if path else None
        self._lock: threading.Lock | None = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            key = str(self.path.resolve())
            with self._locks_guard:
                if key not in self._locks:
                    self._locks[key] = threading.Lock()
                self._lock = self._locks[key]

    def emit(self, stream_id: str, feature: FeatureVector, event: RiskEvent) -> None:
        if self.path is None:
            return
        speed = float((feature.velocity[0] ** 2 + feature.velocity[1] ** 2) ** 0.5)
        acc = float((feature.acceleration[0] ** 2 + feature.acceleration[1] ** 2) ** 0.5)
        payload = {
            "stream_id": stream_id,
            "track_id": feature.track_id,
            "timestamp": feature.timestamp,
            "speed": speed,
            "vy": float(feature.velocity[1]),
            "acc": acc,
            "lean": float(feature.lean_angle),
            "posture": feature.posture,
            "risk_level": event.risk_level,
            "label": 1 if event.risk_level in {"HIGH", "CRITICAL"} else 0,
        }
        line = json.dumps(payload, separators=(",", ":"))
        lock = self._lock
        if lock is None:
            return
        with lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
