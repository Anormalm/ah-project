from __future__ import annotations

import queue
import threading
import time
from typing import Any

import numpy as np

from ingestion.video_loader import FramePacket


class RealSenseSource:
    """Latest-frame RGB-D source with lazy librealsense loading.

    Color is returned in BGR order for the existing OpenCV/pose pipeline. Depth
    is aligned to color and retained in the packet as raw uint16 values together
    with the device's metres-per-unit scale.
    """

    def __init__(
        self,
        serial: str | None = None,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        buffer_size: int = 1,
        frame_timeout_ms: int = 1000,
        enable_imu: bool = False,
        rs_module: Any | None = None,
    ) -> None:
        self.serial = None if serial in {None, "", "auto"} else str(serial)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.frame_timeout_ms = max(100, int(frame_timeout_ms))
        self.enable_imu = bool(enable_imu)
        self._rs = rs_module
        self._buffer: queue.Queue[FramePacket] = queue.Queue(maxsize=max(1, int(buffer_size)))
        self._pipeline = None
        self._align = None
        self._thread: threading.Thread | None = None
        self._running = False
        self.depth_scale_m: float | None = None
        self.frames_captured = 0
        self.frames_dropped = 0
        self.reconnect_count = 0
        self.last_error: str | None = None

    @property
    def is_running(self) -> bool:
        return self._running

    def _sdk(self):
        if self._rs is not None:
            return self._rs
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError(
                "RealSense input requires librealsense and pyrealsense2. "
                "Install them for the target platform before using a realsense stream."
            ) from exc
        self._rs = rs
        return rs

    def start(self) -> None:
        if self._running:
            return
        rs = self._sdk()
        pipeline = rs.pipeline()
        config = rs.config()
        if self.serial:
            config.enable_device(self.serial)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        if self.enable_imu:
            config.enable_stream(rs.stream.accel)
            config.enable_stream(rs.stream.gyro)

        try:
            profile = pipeline.start(config)
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale_m = float(depth_sensor.get_depth_scale())
        except Exception:
            try:
                pipeline.stop()
            except Exception:
                pass
            raise

        self._pipeline = pipeline
        self._align = rs.align(rs.stream.color)
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="realsense-capture", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            try:
                frames = self._pipeline.wait_for_frames(self.frame_timeout_ms)
                aligned = self._align.process(frames)
                color = aligned.get_color_frame()
                depth = aligned.get_depth_frame()
                if not color or not depth:
                    self.frames_dropped += 1
                    continue
                packet = FramePacket(
                    frame=np.asanyarray(color.get_data()).copy(),
                    depth_frame=np.asanyarray(depth.get_data()).copy(),
                    depth_scale_m=self.depth_scale_m,
                    timestamp=time.time(),
                )
                self.frames_captured += 1
                if self._buffer.full():
                    try:
                        self._buffer.get_nowait()
                        self.frames_dropped += 1
                    except queue.Empty:
                        pass
                self._buffer.put_nowait(packet)
                self.last_error = None
            except Exception as exc:
                if not self._running:
                    break
                self.last_error = str(exc)

    def read(self, timeout: float = 0.2) -> FramePacket | None:
        if not self._running and self._buffer.empty():
            return None
        try:
            return self._buffer.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._running = False
        pipeline = self._pipeline
        self._pipeline = None
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)


def create_realsense_source(source: str | int, options: dict[str, Any], buffer_size: int) -> RealSenseSource:
    serial = None if source in {"auto", "", 0, "0"} else str(source)
    return RealSenseSource(
        serial=serial,
        width=int(options.get("width", 640)),
        height=int(options.get("height", 480)),
        fps=int(options.get("fps", 30)),
        buffer_size=buffer_size,
        frame_timeout_ms=int(options.get("frame_timeout_ms", 1000)),
        enable_imu=bool(options.get("enable_imu", False)),
    )
