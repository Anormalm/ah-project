from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path

from utils.schemas import FeatureVector, RiskEvent
from utils.async_jsonl import BoundedJSONLWriter


class TrainingDataLogger:
    _registry_lock = threading.Lock()
    _writers: dict[str, tuple[BoundedJSONLWriter, int]] = {}

    def __init__(self, path: str | None, queue_size: int = 4096, batch_size: int = 128) -> None:
        self.path = Path(path) if path else None
        self._writer: BoundedJSONLWriter | None = None
        self._registry_key: str | None = None
        self._closed = False
        self._session_id = uuid.uuid4().hex
        if self.path is not None:
            key = str(self.path.resolve())
            with self._registry_lock:
                entry = self._writers.get(key)
                if entry is None:
                    writer = BoundedJSONLWriter(self.path, queue_size=queue_size, batch_size=batch_size)
                    self._writers[key] = (writer, 1)
                else:
                    writer, refs = entry
                    self._writers[key] = (writer, refs + 1)
                self._writer = writer
                self._registry_key = key

    def emit(self, stream_id: str, feature: FeatureVector, event: RiskEvent) -> None:
        if self.path is None:
            return
        speed = float((feature.velocity[0] ** 2 + feature.velocity[1] ** 2) ** 0.5)
        acc = float((feature.acceleration[0] ** 2 + feature.acceleration[1] ** 2) ** 0.5)
        payload = {
            "stream_id": stream_id,
            "session_id": self._session_id,
            "track_id": feature.track_id,
            "timestamp": feature.timestamp,
            "speed": speed,
            "vy": float(feature.velocity[1]),
            "acc": acc,
            "lean": float(feature.lean_angle),
            "posture": feature.posture,
            "pose_quality": float(feature.pose_quality),
            "normalized_keypoints": feature.normalized_keypoints,
            "risk_level": event.risk_level,
            "label": 1 if event.risk_level in {"HIGH", "CRITICAL"} else 0,
            "label_source": "weak_rule",
        }
        line = json.dumps(payload, separators=(",", ":"))
        if self._writer is None:
            return
        self._writer.write(line)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        key = self._registry_key
        writer_to_close: BoundedJSONLWriter | None = None
        if key is not None:
            with self._registry_lock:
                entry = self._writers.get(key)
                if entry is not None:
                    writer, refs = entry
                    if refs <= 1:
                        self._writers.pop(key, None)
                        writer_to_close = writer
                    else:
                        self._writers[key] = (writer, refs - 1)
        if writer_to_close is not None:
            writer_to_close.close()
