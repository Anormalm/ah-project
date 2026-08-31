from __future__ import annotations

import json
import queue
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2

from output.clinical_dashboard import build_summary, create_dashboard_app
from utils.async_jsonl import BoundedJSONLWriter
from utils.logger import setup_logger
from utils.schemas import AlertRecord, RiskEvent


class AlertManager:
    _rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}

    def __init__(
        self,
        json_log_path: str,
        enable_api: bool = False,
        api_host: str = "0.0.0.0",
        api_port: int = 8000,
        frame_jpeg_quality: int = 80,
        dedupe_window_sec: float = 1.0,
        emit_on_level_change_only: bool = True,
        logger_name: str = "alert_manager",
        log_queue_size: int = 2048,
        log_batch_size: int = 64,
        frame_queue_size: int = 2,
        allow_privacy_toggle: bool = False,
    ) -> None:
        self.log_path = Path(json_log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.enable_api = enable_api
        self.api_host = api_host
        self.api_port = api_port
        self.frame_jpeg_quality = max(30, min(int(frame_jpeg_quality), 95))
        self.dedupe_window_sec = max(0.0, float(dedupe_window_sec))
        self.emit_on_level_change_only = bool(emit_on_level_change_only)
        self.allow_privacy_toggle = bool(allow_privacy_toggle)
        self.logger = setup_logger(name=logger_name)
        self._log_writer = BoundedJSONLWriter(
            self.log_path,
            queue_size=log_queue_size,
            batch_size=log_batch_size,
        )
        self._latest: deque[dict[str, Any]] = deque(maxlen=512)
        self._latest_frame_jpeg: dict[str, bytes] = {}
        self._frame_seq: dict[str, int] = {}
        self._stream_ids: set[str] = set()
        self._stream_health: dict[str, dict[str, Any]] = {}
        self._stream_privacy: dict[str, dict[str, Any]] = {}
        self._acked_tracks: dict[tuple[str, int], dict[str, Any]] = {}
        self._last_track_level: dict[tuple[str, int], str] = {}
        self._last_emit_meta: dict[tuple[str, int], tuple[str, float]] = {}
        self._lock = threading.Lock()
        self._start_lock = threading.Lock()
        self._server_thread: threading.Thread | None = None
        self._server = None
        self._subscribers: list[queue.Queue[dict[str, Any]]] = []
        self._frame_queue: queue.Queue[tuple[str, Any] | object] = queue.Queue(maxsize=max(1, int(frame_queue_size)))
        self._frame_sentinel = object()
        self._frame_thread: threading.Thread | None = None
        self._closed = False

    def start(self) -> None:
        with self._start_lock:
            if self._closed:
                raise RuntimeError("AlertManager is closed")
            self._log_writer.start()
            if self._frame_thread is None or not self._frame_thread.is_alive():
                self._frame_thread = threading.Thread(target=self._frame_encode_loop, name="dashboard-jpeg", daemon=True)
                self._frame_thread.start()
        if self.enable_api:
            with self._start_lock:
                self._start_api_server()

    def register_stream(self, stream_id: str, privacy_mode: str | None = None) -> None:
        with self._lock:
            self._stream_ids.add(stream_id)
            self._stream_health.setdefault(stream_id, {"last_processed_ts": None, "fps": 0.0})
            if privacy_mode is not None:
                mode = str(privacy_mode or "none").lower()
                self._stream_privacy[stream_id] = {
                    "configured_mode": mode,
                    "enabled": mode not in {"none", "off", "false"},
                }

    def get_privacy_status(self, stream_id: str) -> dict[str, Any]:
        with self._lock:
            state = dict(self._stream_privacy.get(stream_id, {"configured_mode": "none", "enabled": False}))
        state["stream_id"] = stream_id
        state["toggle_allowed"] = self.allow_privacy_toggle
        state["effective_mode"] = state["configured_mode"] if state["enabled"] else "none"
        return state

    def set_privacy_enabled(self, stream_id: str, enabled: bool) -> dict[str, Any]:
        if not self.allow_privacy_toggle:
            raise PermissionError("Runtime privacy toggle is disabled by configuration")
        with self._lock:
            if stream_id not in self._stream_privacy:
                raise KeyError(f"Unknown stream: {stream_id}")
            configured = str(self._stream_privacy[stream_id]["configured_mode"])
            self._stream_privacy[stream_id]["enabled"] = bool(enabled) and configured not in {"none", "off", "false"}
        return self.get_privacy_status(stream_id)

    def get_effective_privacy_mode(self, stream_id: str) -> str:
        return str(self.get_privacy_status(stream_id)["effective_mode"])

    def update_stream_health(self, stream_id: str, fps: float, source) -> None:
        with self._lock:
            self._stream_ids.add(stream_id)
            self._stream_health[stream_id] = {
                "last_processed_ts": time.time(),
                "fps": float(fps),
                "frames_captured": int(getattr(source, "frames_captured", 0)),
                "frames_dropped": int(getattr(source, "frames_dropped", 0)),
                "reconnect_count": int(getattr(source, "reconnect_count", 0)),
            }

    def get_health(self, stale_after_sec: float = 10.0) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            streams = {key: dict(value) for key, value in self._stream_health.items()}
        stale = [
            stream_id
            for stream_id, state in streams.items()
            if state.get("last_processed_ts") is None or now - float(state["last_processed_ts"]) > stale_after_sec
        ]
        return {
            "status": "ok" if streams and not stale else "degraded",
            "generated_at": now,
            "stale_streams": stale,
            "streams": streams,
        }

    def emit(self, stream_id: str, event: RiskEvent) -> None:
        record = AlertRecord(stream_id=stream_id, event=event)
        payload = record.model_dump()

        level = event.risk_level
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

            should_emit = True
            last_meta = self._last_emit_meta.get(key)
            if self.emit_on_level_change_only and last_meta is not None:
                last_level, last_ts = last_meta
                if last_level == level and (event.timestamp - last_ts) < self.dedupe_window_sec:
                    should_emit = False

            if should_emit:
                self._last_emit_meta[key] = (level, float(event.timestamp))
            self._latest.append(payload)
            subscribers = list(self._subscribers) if should_emit else []

        if should_emit:
            line = json.dumps(payload, separators=(",", ":"))
            if level in {"HIGH", "CRITICAL"}:
                self.logger.warning(line)
            else:
                self.logger.debug(line)
            self._log_writer.write(line)

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

    def get_stream_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._stream_ids)

    def publish_frame(self, stream_id: str, frame) -> None:
        self.start()
        item = (stream_id, frame)
        try:
            self._frame_queue.put_nowait(item)
        except queue.Full:
            try:
                self._frame_queue.get_nowait()
                self._frame_queue.task_done()
            except queue.Empty:
                pass
            try:
                self._frame_queue.put_nowait(item)
            except queue.Full:
                pass

    def _frame_encode_loop(self) -> None:
        while True:
            item = self._frame_queue.get()
            if item is self._frame_sentinel:
                self._frame_queue.task_done()
                return
            stream_id, frame = item
            self._encode_and_store_frame(stream_id, frame)
            self._frame_queue.task_done()

    def _encode_and_store_frame(self, stream_id: str, frame) -> None:
        quality = [int(cv2.IMWRITE_JPEG_QUALITY), self.frame_jpeg_quality]
        ok, encoded = cv2.imencode(".jpg", frame, quality)
        if not ok:
            return
        payload = encoded.tobytes()
        with self._lock:
            self._stream_ids.add(stream_id)
            self._latest_frame_jpeg[stream_id] = payload
            self._frame_seq[stream_id] = self._frame_seq.get(stream_id, 0) + 1

    def retire_track(self, stream_id: str, track_id: int) -> None:
        key = (stream_id, int(track_id))
        with self._lock:
            self._last_track_level.pop(key, None)
            self._last_emit_meta.pop(key, None)

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
        filtered: list[dict[str, Any]] = []
        for row in reversed(rows):
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
        config = uvicorn.Config(app, host=self.api_host, port=self.api_port, log_level="warning")
        server = uvicorn.Server(config)
        self._server = server

        def _run() -> None:
            server.run()

        self._server_thread = threading.Thread(target=_run, daemon=True)
        self._server_thread.start()

    def close(self) -> None:
        with self._start_lock:
            if self._closed:
                return
            self._closed = True
            server = self._server
            frame_thread = self._frame_thread
        if server is not None:
            server.should_exit = True
        if self._server_thread is not None and self._server_thread.is_alive():
            self._server_thread.join(timeout=3.0)
        if frame_thread is not None and frame_thread.is_alive():
            while True:
                try:
                    self._frame_queue.put_nowait(self._frame_sentinel)
                    break
                except queue.Full:
                    try:
                        self._frame_queue.get_nowait()
                        self._frame_queue.task_done()
                    except queue.Empty:
                        break
            frame_thread.join(timeout=3.0)
        self._log_writer.close()
