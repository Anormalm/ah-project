from __future__ import annotations

import json
import math
import queue
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2

from output.clinical_dashboard import build_summary, create_dashboard_app
from utils.logger import setup_logger
from utils.schemas import AlertRecord, RiskEvent


class AlertManager:
    _rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
    _feedback_labels = frozenset({"confirmed_fall", "false_alarm", "non_fall_activity", "unclear"})

    def __init__(
        self,
        json_log_path: str,
        enable_api: bool = False,
        api_host: str = "0.0.0.0",
        api_port: int = 8000,
        frame_jpeg_quality: int = 80,
        logger_name: str = "alert_manager",
        feedback_log_path: str | None = None,
    ) -> None:
        self.log_path = Path(json_log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.feedback_log_path = Path(feedback_log_path) if feedback_log_path else None
        if self.feedback_log_path is not None:
            self.feedback_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.enable_api = enable_api
        self.api_host = api_host
        self.api_port = api_port
        self.frame_jpeg_quality = max(30, min(int(frame_jpeg_quality), 95))
        self.logger = setup_logger(name=logger_name)
        self._latest: deque[dict[str, Any]] = deque(maxlen=512)
        self._latest_frame_jpeg: dict[str, bytes] = {}
        self._frame_seq: dict[str, int] = {}
        self._stream_ids: set[str] = set()
        self._acked_tracks: dict[tuple[str, int], dict[str, Any]] = {}
        self._last_track_level: dict[tuple[str, int], str] = {}
        self._lock = threading.Lock()
        self._feedback_lock = threading.Lock()
        self._start_lock = threading.Lock()
        self._server_thread: threading.Thread | None = None
        self._subscribers: list[queue.Queue[dict[str, Any]]] = []

    def start(self) -> None:
        if self.enable_api:
            with self._start_lock:
                self._start_api_server()

    def register_stream(self, stream_id: str) -> None:
        with self._lock:
            self._stream_ids.add(stream_id)

    def emit(self, stream_id: str, event: RiskEvent) -> None:
        record = AlertRecord(stream_id=stream_id, event=event)
        payload = record.model_dump()

        level = event.risk_level
        line = json.dumps(payload, separators=(",", ":"))
        if level in {"HIGH", "CRITICAL"}:
            self.logger.warning(line)
        else:
            self.logger.info(line)

        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

        key = (stream_id, int(event.track_id))
        with self._lock:
            self._stream_ids.add(stream_id)
            prior_level = self._last_track_level.get(key, "LOW")
            if key in self._acked_tracks:
                acked_level = str(self._acked_tracks[key].get("level", "LOW"))
                escalated = self._rank.get(level, 0) > self._rank.get(acked_level, 0)
                crossed_severe = self._rank.get(prior_level, 0) < self._rank["HIGH"] <= self._rank.get(level, 0)
                if escalated or crossed_severe:
                    self._acked_tracks.pop(key, None)
            self._last_track_level[key] = level
            self._latest.append(payload)
            subscribers = list(self._subscribers)

        for sub in subscribers:
            try:
                sub.put_nowait(payload)
            except queue.Full:
                pass

    def get_latest(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            data = list(self._latest)
            acked = set(self._acked_tracks.keys())
        capped = max(1, min(limit, len(data) if data else 1))
        recent = data[-capped:]
        out: list[dict[str, Any]] = []
        for row in recent:
            event = row.get("event") or {}
            key = (str(row.get("stream_id", "unknown")), int(event.get("track_id", -1)))
            item = dict(row)
            item["acknowledged"] = key in acked
            out.append(item)
        return out

    def get_summary(self) -> dict[str, Any]:
        return build_summary(self.get_latest(limit=512), self.get_open_alerts(limit=512))

    def record_feedback(
        self,
        stream_id: str,
        track_id: int,
        timestamp: float,
        label: str,
    ) -> dict[str, Any]:
        normalized_stream_id = str(stream_id).strip()
        normalized_label = str(label).strip().lower()
        target_timestamp = float(timestamp)
        if not normalized_stream_id:
            raise ValueError("stream_id is required")
        if normalized_label not in self._feedback_labels:
            allowed = ", ".join(sorted(self._feedback_labels))
            raise ValueError(f"Unsupported feedback label. Expected one of: {allowed}")
        if not math.isfinite(target_timestamp) or target_timestamp <= 0.0:
            raise ValueError("timestamp must be a positive finite number")
        if self.feedback_log_path is None:
            raise RuntimeError("Feedback logging is disabled; configure output.feedback_log_path")

        payload = {
            "stream_id": normalized_stream_id,
            "track_id": int(track_id),
            "timestamp": target_timestamp,
            "label": normalized_label,
            "annotated_at": time.time(),
        }
        line = json.dumps(payload, separators=(",", ":"))
        with self._feedback_lock:
            with self.feedback_log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        return payload

    def get_stream_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._stream_ids)

    def publish_frame(self, stream_id: str, frame) -> None:
        quality = [int(cv2.IMWRITE_JPEG_QUALITY), self.frame_jpeg_quality]
        ok, encoded = cv2.imencode(".jpg", frame, quality)
        if not ok:
            return
        payload = encoded.tobytes()
        with self._lock:
            self._stream_ids.add(stream_id)
            self._latest_frame_jpeg[stream_id] = payload
            self._frame_seq[stream_id] = self._frame_seq.get(stream_id, 0) + 1

    def get_latest_frame(self, stream_id: str) -> tuple[bytes | None, int]:
        with self._lock:
            return self._latest_frame_jpeg.get(stream_id), self._frame_seq.get(stream_id, 0)

    def ack_track(self, stream_id: str, track_id: int) -> None:
        key = (stream_id, int(track_id))
        with self._lock:
            level = self._last_track_level.get(key, "LOW")
            self._acked_tracks[key] = {"ack_ts": time.time(), "level": level}

    def unack_track(self, stream_id: str, track_id: int) -> None:
        key = (stream_id, int(track_id))
        with self._lock:
            self._acked_tracks.pop(key, None)

    def get_open_alerts(self, limit: int = 100, min_level: str = "HIGH") -> list[dict[str, Any]]:
        rank_floor = self._rank.get(min_level.upper(), self._rank["HIGH"])
        rows = self.get_latest(limit=max(limit, 512))
        latest_by_track: dict[tuple[str, int], dict[str, Any]] = {}
        for row in rows:
            event = row.get("event") or {}
            key = (str(row.get("stream_id", "unknown")), int(event.get("track_id", -1)))
            current = latest_by_track.get(key)
            current_ts = float((current.get("event") or {}).get("timestamp", 0.0)) if current else float("-inf")
            row_ts = float(event.get("timestamp", 0.0))
            if current is None or row_ts >= current_ts:
                latest_by_track[key] = row

        filtered: list[dict[str, Any]] = []
        latest_rows = sorted(
            latest_by_track.values(),
            key=lambda row: float((row.get("event") or {}).get("timestamp", 0.0)),
            reverse=True,
        )
        for row in latest_rows:
            event = row.get("event") or {}
            level = str(event.get("risk_level", "LOW"))
            if self._rank.get(level, 0) < rank_floor:
                continue
            if bool(row.get("acknowledged", False)):
                continue
            filtered.append(row)
            if len(filtered) >= limit:
                break
        return filtered

    def subscribe(self) -> queue.Queue[dict[str, Any]]:
        q: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=256)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue[dict[str, Any]]) -> None:
        with self._lock:
            self._subscribers = [sub for sub in self._subscribers if sub is not q]

    def _start_api_server(self) -> None:
        if self._server_thread and self._server_thread.is_alive():
            return
        try:
            import uvicorn
        except ImportError:
            self.logger.error("uvicorn not installed, API alert endpoint disabled")
            return

        app = create_dashboard_app(self)

        def _run() -> None:
            uvicorn.run(app, host=self.api_host, port=self.api_port, log_level="warning")

        self._server_thread = threading.Thread(target=_run, daemon=True)
        self._server_thread.start()

