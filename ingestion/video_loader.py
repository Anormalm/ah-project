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
    def __init__(self, source: str | int, target_fps: float = 15.0, buffer_size: int = 4) -> None:
        self.source = source
        self.target_fps = max(target_fps, 0.0)
        self._interval = 1.0 / self.target_fps if self.target_fps > 0 else 0.0
        self._buffer: queue.Queue[FramePacket] = queue.Queue(maxsize=max(buffer_size, 1))
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._capture: Optional[cv2.VideoCapture] = None

    @property
    def is_running(self) -> bool:
        return self._running

    def _on_capture_opened(self, capture: cv2.VideoCapture) -> None:
        _ = capture

    def start(self) -> None:
        if self._running:
            return
        self._capture = cv2.VideoCapture(self.source)
        if not self._capture.isOpened():
            raise RuntimeError(f"Unable to open video source: {self.source}")
        self._on_capture_opened(self._capture)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        last_emit = 0.0
        while self._running:
            if self._capture is None:
                break
            ok, frame = self._capture.read()
            if not ok:
                self._running = False
                break

            now = time.time()
            if self._interval > 0 and now - last_emit < self._interval:
                continue
            last_emit = now

            packet = FramePacket(frame=frame, timestamp=now)
            if self._buffer.full():
                try:
                    self._buffer.get_nowait()
                except queue.Empty:
                    pass
            self._buffer.put(packet)

    def read(self, timeout: float = 0.2) -> FramePacket | None:
        if not self._running and self._buffer.empty():
            return None
        try:
            return self._buffer.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._capture is not None:
            self._capture.release()
            self._capture = None


class WebcamSource(AsyncVideoSource):
    def __init__(
        self,
        camera_id: int = 0,
        target_fps: float = 15.0,
        buffer_size: int = 4,
        requested_fps: float | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        super().__init__(source=camera_id, target_fps=target_fps, buffer_size=buffer_size)
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


class DummyVideoSource:
    def __init__(self, width: int = 640, height: int = 360, target_fps: float = 15.0) -> None:
        self.width = width
        self.height = height
        self.target_fps = max(target_fps, 0.0)
        self._interval = 1.0 / self.target_fps if self.target_fps > 0 else 0.0
        self._running = False
        self._last_ts = 0.0
        self._frame_idx = 0

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
        return WebcamSource(
            camera_id=int(source_path),
            target_fps=fps,
            buffer_size=buffer_size,
            requested_fps=webcam_options.get("requested_fps"),
            width=webcam_options.get("width"),
            height=webcam_options.get("height"),
        )
    if normalized == "video":
        return VideoFileSource(path=str(source_path), target_fps=fps, buffer_size=buffer_size)
    if normalized == "dummy":
        return DummyVideoSource(target_fps=fps)
    raise ValueError(f"Unsupported video source type: {source_type}")

