from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class FramePacket:
    frame: Optional[np.ndarray]
    timestamp: float


class AsyncVideoSource:
    def __init__(
        self,
        source: str | int,
        target_fps: float = 15.0,
        buffer_size: int = 4,
        api_preference: int | None = None,
        startup_retries: int = 0,
        reconnect_interval_sec: float = 2.0,
        runtime_reconnect: bool = False,
        runtime_max_retries: int | None = None,
    ) -> None:
        self.source = source
        self.target_fps = max(target_fps, 0.0)
        self._interval = 1.0 / self.target_fps if self.target_fps > 0 else 0.0
        self._buffer: queue.Queue[FramePacket] = queue.Queue(maxsize=max(buffer_size, 1))
        self.api_preference = api_preference
        self.startup_retries = max(0, int(startup_retries))
        self.reconnect_interval_sec = max(0.1, float(reconnect_interval_sec))
        self.runtime_reconnect = bool(runtime_reconnect)
        self.runtime_max_retries = None if runtime_max_retries is None else max(0, int(runtime_max_retries))
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._capture: Optional[cv2.VideoCapture] = None
        self.frames_captured = 0
        self.frames_dropped = 0
        self.reconnect_count = 0
        self.last_error: str | None = None

    @property
    def is_running(self) -> bool:
        return self._running

    def _on_capture_opened(self, capture: cv2.VideoCapture) -> None:
        _ = capture

    def _new_capture(self) -> cv2.VideoCapture:
        if self.api_preference is None:
            return cv2.VideoCapture(self.source)
        return cv2.VideoCapture(self.source, self.api_preference)

    def _open_capture(self) -> bool:
        capture = self._new_capture()
        if not capture.isOpened():
            capture.release()
            self.last_error = "Unable to open video source"
            return False
        self._on_capture_opened(capture)
        self._capture = capture
        self.last_error = None
        return True

    def _release_capture(self) -> None:
        capture = self._capture
        self._capture = None
        if capture is not None:
            capture.release()

    def _wait_while_running(self, delay_sec: float) -> None:
        deadline = time.monotonic() + max(0.0, delay_sec)
        while self._running and time.monotonic() < deadline:
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        for attempt in range(self.startup_retries + 1):
            if self._open_capture():
                break
            if attempt < self.startup_retries:
                self._wait_while_running(self.reconnect_interval_sec)
        else:
            self._running = False
            raise RuntimeError(self.last_error or "Unable to open video source")
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _reconnect(self) -> bool:
        self._release_capture()
        retries = 0
        while self._running:
            if self.runtime_max_retries is not None and retries > self.runtime_max_retries:
                return False
            self._wait_while_running(self.reconnect_interval_sec)
            if not self._running:
                return False
            retries += 1
            if self._open_capture():
                self.reconnect_count += 1
                return True
        return False

    def _loop(self) -> None:
        last_emit = 0.0
        while self._running:
            capture = self._capture
            if capture is None:
                break
            ok, frame = capture.read()
            if not ok:
                self.last_error = "Video read failed"
                if self.runtime_reconnect and self._reconnect():
                    last_emit = 0.0
                    continue
                self._running = False
                break

            now_mono = time.monotonic()
            if self._interval > 0 and now_mono - last_emit < self._interval:
                self.frames_dropped += 1
                continue
            last_emit = now_mono

            self.frames_captured += 1
            packet = FramePacket(frame=frame, timestamp=time.time())
            if self._buffer.full():
                try:
                    self._buffer.get_nowait()
                    self.frames_dropped += 1
                except queue.Empty:
                    pass
            self._buffer.put_nowait(packet)

    def read(self, timeout: float = 0.2) -> FramePacket | None:
        if not self._running and self._buffer.empty():
            return None
        try:
            return self._buffer.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._running = False
        self._release_capture()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)


class WebcamSource(AsyncVideoSource):
    def __init__(
        self,
        camera_id: int = 0,
        target_fps: float = 15.0,
        buffer_size: int = 4,
        requested_fps: float | None = None,
        width: int | None = None,
        height: int | None = None,
        api_preference: int | None = None,
    ) -> None:
        super().__init__(source=camera_id, target_fps=target_fps, buffer_size=buffer_size, api_preference=api_preference)
        self.requested_fps = requested_fps
        self.width = width
        self.height = height

    def _on_capture_opened(self, capture: cv2.VideoCapture) -> None:
        if self.width is not None and self.width > 0:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        if self.height is not None and self.height > 0:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        if self.requested_fps is not None and self.requested_fps > 0:
            capture.set(cv2.CAP_PROP_FPS, float(self.requested_fps))


class VideoFileSource(AsyncVideoSource):
    def __init__(self, path: str, target_fps: float = 15.0, buffer_size: int = 4) -> None:
        super().__init__(source=path, target_fps=target_fps, buffer_size=buffer_size)


class GStreamerSource(AsyncVideoSource):
    def __init__(
        self,
        pipeline: str,
        target_fps: float = 0.0,
        buffer_size: int = 1,
        startup_retries: int = 0,
        reconnect_interval_sec: float = 2.0,
        runtime_reconnect: bool = False,
        runtime_max_retries: int | None = None,
    ) -> None:
        super().__init__(
            source=pipeline,
            target_fps=target_fps,
            buffer_size=buffer_size,
            api_preference=cv2.CAP_GSTREAMER,
            startup_retries=startup_retries,
            reconnect_interval_sec=reconnect_interval_sec,
            runtime_reconnect=runtime_reconnect,
            runtime_max_retries=runtime_max_retries,
        )


class DummyVideoSource:
    def __init__(self, width: int = 640, height: int = 360, target_fps: float = 15.0) -> None:
        self.width = width
        self.height = height
        self.target_fps = max(target_fps, 0.0)
        self._interval = 1.0 / self.target_fps if self.target_fps > 0 else 0.0
        self._running = False
        self._last_ts = 0.0
        self._frame_idx = 0
        self.frames_captured = 0
        self.frames_dropped = 0
        self.reconnect_count = 0

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        self._running = True
        self._last_ts = time.time()

    def read(self, timeout: float = 0.2) -> FramePacket | None:
        _ = timeout
        if not self._running:
            return None
        now = time.time()
        dt = now - self._last_ts
        if self._interval > 0 and dt < self._interval:
            time.sleep(self._interval - dt)
        self._last_ts = time.time()

        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        x = 50 + (self._frame_idx * 8) % (self.width - 120)
        y = 120 + int(40 * np.sin(self._frame_idx / 6))
        cv2.rectangle(frame, (x, y), (x + 60, y + 140), (255, 255, 255), -1)
        self._frame_idx += 1
        self.frames_captured += 1
        return FramePacket(frame=frame, timestamp=time.time())

    def stop(self) -> None:
        self._running = False


def create_video_source(
    source_type: str,
    source_path: str | int,
    fps: float,
    buffer_size: int,
    webcam_options: dict | None = None,
):
    normalized = source_type.lower()
    if normalized == "webcam":
        webcam_options = webcam_options or {}
        backend = str(webcam_options.get("backend", "auto")).lower()
        api_preference = cv2.CAP_V4L2 if backend == "v4l2" and hasattr(cv2, "CAP_V4L2") else None
        return WebcamSource(
            camera_id=int(source_path),
            target_fps=fps,
            buffer_size=buffer_size,
            requested_fps=webcam_options.get("requested_fps"),
            width=webcam_options.get("width"),
            height=webcam_options.get("height"),
            api_preference=api_preference,
        )
    if normalized == "video":
        return VideoFileSource(path=str(source_path), target_fps=fps, buffer_size=buffer_size)
    if normalized == "dummy":
        return DummyVideoSource(target_fps=fps)
    if normalized == "gstreamer":
        return GStreamerSource(pipeline=str(source_path), target_fps=fps, buffer_size=buffer_size)
    raise ValueError(f"Unsupported video source type: {source_type}")

