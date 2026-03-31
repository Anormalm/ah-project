from __future__ import annotations

import json
import queue
import threading
from collections import deque
from pathlib import Path
from typing import Any

import cv2

from output.clinical_dashboard import build_summary, create_dashboard_app
from utils.logger import setup_logger
from utils.schemas import AlertRecord, RiskEvent


class AlertManager:
    def __init__(
        self,
        json_log_path: str,
        enable_api: bool = False,
        api_host: str = "0.0.0.0",
        api_port: int = 8000,
        frame_jpeg_quality: int = 80,
        logger_name: str = "alert_manager",
    ) -> None:
        self.log_path = Path(json_log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.enable_api = enable_api
        self.api_host = api_host
        self.api_port = api_port
        self.frame_jpeg_quality = max(30, min(int(frame_jpeg_quality), 95))
        self.logger = setup_logger(name=logger_name)
        self._latest: deque[dict[str, Any]] = deque(maxlen=512)
        self._latest_frame_jpeg: dict[str, bytes] = {}
        self._frame_seq: dict[str, int] = {}
        self._stream_ids: set[str] = set()
        self._lock = threading.Lock()
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

        with self._lock:
            self._stream_ids.add(stream_id)
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
        capped = max(1, min(limit, len(data) if data else 1))
        return data[-capped:]

    def get_summary(self) -> dict[str, Any]:
        return build_summary(self.get_latest(limit=512))

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

