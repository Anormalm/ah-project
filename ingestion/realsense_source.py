from __future__ import annotations

import importlib
import logging
import queue
import re
import threading
import time
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Mapping, Optional

import numpy as np

from ingestion.frame_packet import FramePacket, MotionSample, PinholeIntrinsics, SensorFrame


LOGGER = logging.getLogger(__name__)


def _load_pyrealsense2() -> ModuleType:
    """Load the optional SDK only when a RealSense source is started."""

    try:
        return importlib.import_module("pyrealsense2")
    except ImportError as exc:
        raise RuntimeError(
            "RealSense input requires pyrealsense2. Install the optional "
            "RealSense dependency before using a realsense stream."
        ) from exc


def _as_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _positive_int(value: Any, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return parsed


def _positive_float(value: Any, name: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise ValueError(f"{name} must be greater than zero")
    return parsed


@dataclass(frozen=True, slots=True)
class VideoStreamSettings:
    enabled: bool
    width: int
    height: int
    fps: int


@dataclass(frozen=True, slots=True)
class InfraredStreamSettings:
    enabled: bool
    indices: tuple[int, ...]
    width: int
    height: int
    fps: int


@dataclass(frozen=True, slots=True)
class ImuStreamSettings:
    enabled: bool
    accel_enabled: bool
    accel_fps: int
    gyro_enabled: bool
    gyro_fps: int


@dataclass(frozen=True, slots=True)
class RealSenseSettings:
    color: VideoStreamSettings
    depth: VideoStreamSettings
    infrared: InfraredStreamSettings
    imu: ImuStreamSettings
    require_usb3: bool
    inference_stream: str
    frame_timeout_sec: float
    max_motion_samples: int

    @classmethod
    def from_options(
        cls,
        options: Mapping[str, Any] | None,
        target_fps: float,
    ) -> "RealSenseSettings":
        opts = _as_mapping(options, "RealSense options")
        default_fps = max(1, int(round(target_fps))) if target_fps > 0 else 30

        color_cfg = _as_mapping(opts.get("color"), "realsense.color")
        depth_cfg = _as_mapping(opts.get("depth"), "realsense.depth")
        infrared_cfg = _as_mapping(opts.get("infrared"), "realsense.infrared")
        imu_cfg = _as_mapping(opts.get("imu"), "realsense.imu")

        color = VideoStreamSettings(
            enabled=bool(color_cfg.get("enabled", True)),
            width=_positive_int(color_cfg.get("width", 640), "realsense.color.width"),
            height=_positive_int(color_cfg.get("height", 480), "realsense.color.height"),
            fps=_positive_int(color_cfg.get("fps", default_fps), "realsense.color.fps"),
        )
        depth = VideoStreamSettings(
            enabled=bool(depth_cfg.get("enabled", True)),
            width=_positive_int(depth_cfg.get("width", 640), "realsense.depth.width"),
            height=_positive_int(depth_cfg.get("height", 480), "realsense.depth.height"),
            fps=_positive_int(depth_cfg.get("fps", default_fps), "realsense.depth.fps"),
        )

        raw_indices = infrared_cfg.get("indices", (1, 2))
        if isinstance(raw_indices, int):
            raw_indices = (raw_indices,)
        if isinstance(raw_indices, (str, bytes)):
            raise TypeError("realsense.infrared.indices must be a sequence of integers")
        try:
            indices = tuple(dict.fromkeys(int(index) for index in raw_indices))
        except TypeError as exc:
            raise TypeError("realsense.infrared.indices must be a sequence of integers") from exc
        if any(index not in {1, 2} for index in indices):
            raise ValueError("D435i infrared stream indices must be 1 and/or 2")
        infrared = InfraredStreamSettings(
            enabled=bool(infrared_cfg.get("enabled", False)),
            indices=indices,
            width=_positive_int(infrared_cfg.get("width", 640), "realsense.infrared.width"),
            height=_positive_int(infrared_cfg.get("height", 480), "realsense.infrared.height"),
            fps=_positive_int(infrared_cfg.get("fps", default_fps), "realsense.infrared.fps"),
        )
        if infrared.enabled and not infrared.indices:
            raise ValueError("At least one infrared stream index is required when infrared is enabled")

        imu = ImuStreamSettings(
            enabled=bool(imu_cfg.get("enabled", True)),
            accel_enabled=bool(imu_cfg.get("accel_enabled", True)),
            accel_fps=_positive_int(imu_cfg.get("accel_fps", 100), "realsense.imu.accel_fps"),
            gyro_enabled=bool(imu_cfg.get("gyro_enabled", True)),
            gyro_fps=_positive_int(imu_cfg.get("gyro_fps", 200), "realsense.imu.gyro_fps"),
        )

        inference_stream = str(opts.get("inference_stream", "color")).strip().lower()
        if inference_stream != "color":
            raise ValueError(
                "Only 'color' is currently supported as the RealSense inference_stream; "
                "infrared is captured as auxiliary data to preserve one stable tracker coordinate system"
            )
        if not color.enabled:
            raise ValueError("The color stream must be enabled because it is the inference and depth-alignment target")

        return cls(
            color=color,
            depth=depth,
            infrared=infrared,
            imu=imu,
            require_usb3=bool(opts.get("require_usb3", True)),
            inference_stream=inference_stream,
            frame_timeout_sec=_positive_float(
                opts.get("frame_timeout_sec", 5.0),
                "realsense.frame_timeout_sec",
            ),
            max_motion_samples=_positive_int(
                opts.get("max_motion_samples", 2048),
                "realsense.max_motion_samples",
            ),
        )


@dataclass(frozen=True, slots=True)
class _FramesetEnvelope:
    frameset: Any
    host_timestamp: float
    accel_samples: tuple[MotionSample, ...]
    gyro_samples: tuple[MotionSample, ...]


class RealSenseSource:
    """Asynchronous D435i source with one inference image per video frameset.

    The SDK callback supplies synchronized video streams as framesets while
    accelerometer and gyroscope frames bypass video synchronization. Motion
    samples are accumulated between framesets, then attached to the next
    emitted packet. Alignment and NumPy copies happen on a worker thread so
    the SDK's sensor callback remains short.

    Supported ``options`` keys mirror the intended YAML structure::

        color: {enabled: true, width: 640, height: 480, fps: 30}
        depth: {enabled: true, width: 640, height: 480, fps: 30}
        infrared: {enabled: true, indices: [1, 2], width: 640, height: 480, fps: 30}
        imu: {enabled: true, accel_enabled: true, accel_fps: 100,
              gyro_enabled: true, gyro_fps: 200}
        require_usb3: true
        inference_stream: color
        frame_timeout_sec: 5.0
    """

    def __init__(
        self,
        source: str | int | None = "auto",
        target_fps: float = 30.0,
        buffer_size: int = 4,
        options: Mapping[str, Any] | None = None,
    ) -> None:
        option_map = _as_mapping(options, "RealSense options")
        configured_serial = option_map.get("serial")
        requested_source = configured_serial if source in {None, "", "auto"} else source
        self.serial = None if requested_source in {None, "", "auto"} else str(requested_source)
        self.target_fps = max(float(target_fps), 0.0)
        self.settings = RealSenseSettings.from_options(option_map, self.target_fps)
        self.buffer_size = max(int(buffer_size), 1)

        self._output: queue.Queue[FramePacket] = queue.Queue(maxsize=self.buffer_size)
        self._framesets: queue.Queue[_FramesetEnvelope] = queue.Queue(maxsize=1)
        self._motion_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._accept_frames = threading.Event()
        self._pending_accel: list[MotionSample] = []
        self._pending_gyro: list[MotionSample] = []

        self._rs: Optional[ModuleType] = None
        self._pipeline: Any = None
        self._pipeline_started = False
        self._align: Any = None
        self._worker: Optional[threading.Thread] = None
        self._depth_scale_m: Optional[float] = None
        self._color_intrinsics: Optional[PinholeIntrinsics] = None
        self._usb_type: Optional[str] = None
        self._device_info: dict[str, str] = {}
        self._last_frame_number: Optional[int] = None
        self._fatal_error: Optional[BaseException] = None

    @property
    def is_running(self) -> bool:
        with self._state_lock:
            return self._accept_frames.is_set() and self._fatal_error is None

    @property
    def last_error(self) -> Optional[BaseException]:
        with self._state_lock:
            return self._fatal_error

    @property
    def device_info(self) -> dict[str, str]:
        with self._state_lock:
            return dict(self._device_info)

    def start(self) -> None:
        with self._state_lock:
            if self._pipeline_started and self._accept_frames.is_set():
                return

            self._reset_for_start()
            rs = _load_pyrealsense2()
            pipeline = rs.pipeline()
            config = rs.config()
            self._configure_sdk_streams(rs, config)
            if self.serial is not None:
                config.enable_device(self.serial)

            try:
                resolved_profile = config.resolve(rs.pipeline_wrapper(pipeline))
                resolved_device = resolved_profile.get_device()
                resolved_info = self._collect_device_info(rs, resolved_device)
                self._validate_usb_connection(resolved_info.get("usb_type"))

                align = rs.align(rs.stream.color) if self.settings.depth.enabled else None
                self._rs = rs
                self._pipeline = pipeline
                self._align = align
                self._accept_frames.set()

                active_profile = pipeline.start(config, self._on_sdk_frame)
                self._pipeline_started = True
                active_device = active_profile.get_device()
                active_info = self._collect_device_info(rs, active_device)
                self._validate_usb_connection(active_info.get("usb_type"))
                self._device_info = active_info or resolved_info
                self._usb_type = self._device_info.get("usb_type")
                self._depth_scale_m = self._read_depth_scale(active_device)
                self._color_intrinsics = self._read_color_intrinsics(rs, active_profile)

                self._worker = threading.Thread(
                    target=self._worker_loop,
                    name="realsense-frame-worker",
                    daemon=True,
                )
                self._worker.start()
                LOGGER.info(
                    "RealSense started name=%s serial=%s firmware=%s usb=%s",
                    self._device_info.get("name", "unknown"),
                    self._device_info.get("serial", self.serial or "auto"),
                    self._device_info.get("firmware", "unknown"),
                    self._usb_type or "unknown",
                )
            except Exception:
                self._accept_frames.clear()
                if self._pipeline_started:
                    try:
                        pipeline.stop()
                    except Exception:
                        pass
                self._pipeline_started = False
                self._pipeline = None
                self._align = None
                self._rs = None
                raise

    def read(self, timeout: float = 0.2) -> FramePacket | None:
        try:
            return self._output.get(timeout=max(float(timeout), 0.0))
        except queue.Empty:
            with self._state_lock:
                error = self._fatal_error
                running = self._accept_frames.is_set()
            if error is not None and not running:
                raise RuntimeError(f"RealSense capture stopped: {error}") from error
            return None

    def stop(self) -> None:
        with self._state_lock:
            self._accept_frames.clear()
            pipeline = self._pipeline
            pipeline_started = self._pipeline_started
            worker = self._worker
            self._pipeline = None
            self._pipeline_started = False
            self._worker = None

        if pipeline is not None and pipeline_started:
            try:
                pipeline.stop()
            except RuntimeError:
                # Stopping an already-disconnected device should still be safe.
                pass

        if worker is not None and worker is not threading.current_thread() and worker.is_alive():
            worker.join(timeout=max(1.0, self.settings.frame_timeout_sec + 0.5))

        self._drain_queue(self._framesets)
        with self._motion_lock:
            self._pending_accel.clear()
            self._pending_gyro.clear()

        with self._state_lock:
            self._align = None
            self._rs = None
            self._depth_scale_m = None
            self._color_intrinsics = None
            self._last_frame_number = None

    def _reset_for_start(self) -> None:
        self._accept_frames.clear()
        self._output = queue.Queue(maxsize=self.buffer_size)
        self._framesets = queue.Queue(maxsize=1)
        with self._motion_lock:
            self._pending_accel.clear()
            self._pending_gyro.clear()
        self._fatal_error = None
        self._device_info = {}
        self._usb_type = None
        self._depth_scale_m = None
        self._color_intrinsics = None
        self._last_frame_number = None

    def _configure_sdk_streams(self, rs: ModuleType, config: Any) -> None:
        color = self.settings.color
        depth = self.settings.depth
        infrared = self.settings.infrared
        imu = self.settings.imu

        if color.enabled:
            config.enable_stream(
                rs.stream.color,
                color.width,
                color.height,
                rs.format.bgr8,
                color.fps,
            )
        if depth.enabled:
            config.enable_stream(
                rs.stream.depth,
                depth.width,
                depth.height,
                rs.format.z16,
                depth.fps,
            )
        if infrared.enabled:
            for index in infrared.indices:
                config.enable_stream(
                    rs.stream.infrared,
                    index,
                    infrared.width,
                    infrared.height,
                    rs.format.y8,
                    infrared.fps,
                )
        if imu.enabled and imu.accel_enabled:
            config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, imu.accel_fps)
        if imu.enabled and imu.gyro_enabled:
            config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, imu.gyro_fps)

    def _on_sdk_frame(self, frame: Any) -> None:
        if not self._accept_frames.is_set():
            return
        rs = self._rs
        if rs is None:
            return

        try:
            if frame.is_motion_frame():
                self._record_motion_sample(rs, frame)
                return
            if not frame.is_frameset():
                return

            with self._motion_lock:
                accel = tuple(self._pending_accel)
                gyro = tuple(self._pending_gyro)
                self._pending_accel.clear()
                self._pending_gyro.clear()

            envelope = _FramesetEnvelope(
                frameset=frame.as_frameset(),
                host_timestamp=time.time(),
                accel_samples=accel,
                gyro_samples=gyro,
            )
            self._offer_frameset(envelope)
        except Exception as exc:
            self._fail(exc)

    def _record_motion_sample(self, rs: ModuleType, frame: Any) -> None:
        profile = frame.get_profile()
        stream_type = profile.stream_type()
        if stream_type not in {rs.stream.accel, rs.stream.gyro}:
            return
        vector = frame.as_motion_frame().get_motion_data()
        sample = MotionSample(
            xyz=(float(vector.x), float(vector.y), float(vector.z)),
            device_timestamp_ms=float(frame.get_timestamp()),
            timestamp_domain=str(frame.get_frame_timestamp_domain()),
        )
        with self._motion_lock:
            target = self._pending_accel if stream_type == rs.stream.accel else self._pending_gyro
            target.append(sample)
            overflow = len(target) - self.settings.max_motion_samples
            if overflow > 0:
                del target[:overflow]

    def _offer_frameset(self, envelope: _FramesetEnvelope) -> None:
        current = envelope
        while self._accept_frames.is_set():
            try:
                self._framesets.put_nowait(current)
                return
            except queue.Full:
                try:
                    dropped = self._framesets.get_nowait()
                except queue.Empty:
                    continue
                # Preserve high-rate IMU samples even when an obsolete video
                # frameset is dropped to keep capture latency bounded.
                current = _FramesetEnvelope(
                    frameset=current.frameset,
                    host_timestamp=current.host_timestamp,
                    accel_samples=self._bounded_motion_merge(
                        dropped.accel_samples,
                        current.accel_samples,
                    ),
                    gyro_samples=self._bounded_motion_merge(
                        dropped.gyro_samples,
                        current.gyro_samples,
                    ),
                )

    def _bounded_motion_merge(
        self,
        older: tuple[MotionSample, ...],
        newer: tuple[MotionSample, ...],
    ) -> tuple[MotionSample, ...]:
        limit = self.settings.max_motion_samples
        combined = older + newer
        return combined[-limit:]

    def _worker_loop(self) -> None:
        last_frameset_at = time.monotonic()
        poll_timeout = min(0.2, self.settings.frame_timeout_sec)
        carried_accel: tuple[MotionSample, ...] = ()
        carried_gyro: tuple[MotionSample, ...] = ()
        while self._accept_frames.is_set():
            try:
                envelope = self._framesets.get(timeout=poll_timeout)
            except queue.Empty:
                if time.monotonic() - last_frameset_at >= self.settings.frame_timeout_sec:
                    self._fail(
                        TimeoutError(
                            f"no synchronized RealSense video frameset arrived for "
                            f"{self.settings.frame_timeout_sec:.1f}s"
                        )
                    )
                    return
                continue

            last_frameset_at = time.monotonic()
            envelope = _FramesetEnvelope(
                frameset=envelope.frameset,
                host_timestamp=envelope.host_timestamp,
                accel_samples=self._bounded_motion_merge(carried_accel, envelope.accel_samples),
                gyro_samples=self._bounded_motion_merge(carried_gyro, envelope.gyro_samples),
            )
            try:
                packet = self._packet_from_frameset(envelope)
            except Exception as exc:
                self._fail(exc)
                return
            if packet is None:
                # A frameset can transiently omit a required video member or
                # repeat a frame number. Keep its motion data for the next
                # complete, unique video packet instead of silently losing it.
                carried_accel = envelope.accel_samples
                carried_gyro = envelope.gyro_samples
                continue
            carried_accel = ()
            carried_gyro = ()
            self._offer_packet(packet)

    def _packet_from_frameset(self, envelope: _FramesetEnvelope) -> FramePacket | None:
        frameset = envelope.frameset
        color_frame = frameset.get_color_frame()
        if not color_frame:
            return None

        frame_number = int(color_frame.get_frame_number())
        if self._last_frame_number == frame_number:
            return None

        aligned_frameset = frameset
        if self.settings.depth.enabled:
            if self._align is None:
                raise RuntimeError("RealSense depth alignment was not initialized")
            aligned_frameset = self._align.process(frameset)
            color_frame = aligned_frameset.get_color_frame()
            depth_frame = aligned_frameset.get_depth_frame()
            if not color_frame or not depth_frame:
                return None
            aligned_depth = self._copy_frame_array(depth_frame)
        else:
            aligned_depth = None

        color_image = self._copy_frame_array(color_frame)
        infrared_images: dict[int, np.ndarray] = {}
        if self.settings.infrared.enabled:
            for index in self.settings.infrared.indices:
                infrared_frame = frameset.get_infrared_frame(index)
                if infrared_frame:
                    infrared_images[index] = self._copy_frame_array(infrared_frame)

        frame_number = int(color_frame.get_frame_number())
        self._last_frame_number = frame_number
        sensor = SensorFrame(
            aligned_depth=aligned_depth,
            depth_scale_m=self._depth_scale_m,
            color_intrinsics=self._color_intrinsics,
            infrared=infrared_images,
            accel_samples=envelope.accel_samples,
            gyro_samples=envelope.gyro_samples,
            device_timestamp_ms=float(color_frame.get_timestamp()),
            frame_number=frame_number,
            usb_type=self._usb_type,
            primary_stream=self.settings.inference_stream,
        )
        return FramePacket(
            frame=color_image,
            timestamp=envelope.host_timestamp,
            sensor=sensor,
        )

    @staticmethod
    def _copy_frame_array(frame: Any) -> np.ndarray:
        # RealSense's NumPy view aliases SDK-managed memory. Packets may outlive
        # the callback/frameset, so always detach it before crossing the queue.
        return np.array(np.asanyarray(frame.get_data()), copy=True, order="C")

    def _offer_packet(self, packet: FramePacket) -> None:
        while self._accept_frames.is_set():
            try:
                self._output.put_nowait(packet)
                return
            except queue.Full:
                try:
                    stale = self._output.get_nowait()
                except queue.Empty:
                    continue
                stale.frame = None
                stale.sensor = None

    def _read_depth_scale(self, device: Any) -> Optional[float]:
        if not self.settings.depth.enabled:
            return None
        scale = float(device.first_depth_sensor().get_depth_scale())
        if scale <= 0.0:
            raise RuntimeError(f"RealSense reported an invalid depth scale: {scale}")
        return scale

    def _read_color_intrinsics(self, rs: ModuleType, profile: Any) -> PinholeIntrinsics:
        stream_profile = profile.get_stream(rs.stream.color)
        intrinsics = stream_profile.as_video_stream_profile().get_intrinsics()
        return PinholeIntrinsics(
            width=int(intrinsics.width),
            height=int(intrinsics.height),
            fx=float(intrinsics.fx),
            fy=float(intrinsics.fy),
            ppx=float(intrinsics.ppx),
            ppy=float(intrinsics.ppy),
        )

    @staticmethod
    def _collect_device_info(rs: ModuleType, device: Any) -> dict[str, str]:
        fields = {
            "name": rs.camera_info.name,
            "serial": rs.camera_info.serial_number,
            "firmware": rs.camera_info.firmware_version,
            "usb_type": rs.camera_info.usb_type_descriptor,
        }
        result: dict[str, str] = {}
        for key, info_key in fields.items():
            try:
                if hasattr(device, "supports") and not device.supports(info_key):
                    continue
                result[key] = str(device.get_info(info_key))
            except (RuntimeError, TypeError, AttributeError):
                continue
        return result

    def _validate_usb_connection(self, descriptor: Optional[str]) -> None:
        if not self.settings.require_usb3:
            return
        if not descriptor:
            raise RuntimeError(
                "Unable to verify the RealSense USB link. Set require_usb3=false "
                "only if reduced-bandwidth operation is intentional."
            )
        match = re.search(r"(\d+)(?:\.(\d+))?", descriptor)
        if match is None or int(match.group(1)) < 3:
            raise RuntimeError(
                f"RealSense negotiated USB {descriptor}; all-feed capture requires "
                "a direct USB 3.x connection and a USB 3-capable data cable."
            )

    def _fail(self, error: BaseException) -> None:
        with self._state_lock:
            if self._fatal_error is None:
                self._fatal_error = error
            self._accept_frames.clear()

    @staticmethod
    def _drain_queue(target: queue.Queue[Any]) -> None:
        while True:
            try:
                target.get_nowait()
            except queue.Empty:
                return


def create_realsense_source(
    source: str | int | None,
    fps: float,
    buffer_size: int,
    options: Mapping[str, Any] | None = None,
) -> RealSenseSource:
    return RealSenseSource(
        source=source,
        target_fps=fps,
        buffer_size=buffer_size,
        options=options,
    )
