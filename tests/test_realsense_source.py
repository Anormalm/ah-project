from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import ingestion.realsense_source as realsense_source
from ingestion.frame_packet import FramePacket, MotionSample, PinholeIntrinsics
from ingestion.realsense_source import RealSenseSettings, RealSenseSource


class _RecordingConfig:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def enable_stream(self, *args) -> None:
        self.calls.append(args)


class _FakeVideoFrame:
    def __init__(self, data: np.ndarray, frame_number: int = 7, timestamp: float = 1234.5) -> None:
        self.data = data
        self.frame_number = frame_number
        self.timestamp = timestamp

    def __bool__(self) -> bool:
        return True

    def get_data(self) -> np.ndarray:
        return self.data

    def get_frame_number(self) -> int:
        return self.frame_number

    def get_timestamp(self) -> float:
        return self.timestamp


class _FakeFrameset:
    def __init__(
        self,
        color: _FakeVideoFrame,
        depth: _FakeVideoFrame | None = None,
        infrared: dict[int, _FakeVideoFrame] | None = None,
    ) -> None:
        self.color = color
        self.depth = depth
        self.infrared = infrared or {}

    def get_color_frame(self):
        return self.color

    def get_depth_frame(self):
        return self.depth

    def get_infrared_frame(self, index: int):
        return self.infrared.get(index)


class _FakeAlign:
    def __init__(self, aligned_frameset: _FakeFrameset) -> None:
        self.aligned_frameset = aligned_frameset
        self.inputs: list[object] = []

    def process(self, frameset):
        self.inputs.append(frameset)
        return self.aligned_frameset


def _fake_rs_namespace():
    return SimpleNamespace(
        stream=SimpleNamespace(
            color="color",
            depth="depth",
            infrared="infrared",
            accel="accel",
            gyro="gyro",
        ),
        format=SimpleNamespace(
            bgr8="bgr8",
            z16="z16",
            y8="y8",
            motion_xyz32f="motion_xyz32f",
        ),
    )


def test_realsense_settings_defaults_follow_d435i_safe_profile() -> None:
    settings = RealSenseSettings.from_options({}, target_fps=30.0)

    assert settings.color.enabled is True
    assert (settings.color.width, settings.color.height, settings.color.fps) == (640, 480, 30)
    assert settings.depth.enabled is True
    assert (settings.depth.width, settings.depth.height, settings.depth.fps) == (640, 480, 30)
    assert settings.infrared.enabled is False
    assert settings.infrared.indices == (1, 2)
    assert settings.imu.enabled is True
    assert settings.imu.accel_enabled is True
    assert settings.imu.accel_fps == 100
    assert settings.imu.gyro_enabled is True
    assert settings.imu.gyro_fps == 200
    assert settings.require_usb3 is True
    assert settings.inference_stream == "color"


def test_all_six_streams_are_enabled_with_requested_rates() -> None:
    source = RealSenseSource(
        options={
            "require_usb3": False,
            "infrared": {"enabled": True, "indices": [1, 2]},
            "imu": {"enabled": True, "accel_fps": 100, "gyro_fps": 200},
        }
    )
    fake_rs = _fake_rs_namespace()
    config = _RecordingConfig()

    source._configure_sdk_streams(fake_rs, config)

    assert config.calls == [
        ("color", 640, 480, "bgr8", 30),
        ("depth", 640, 480, "z16", 30),
        ("infrared", 1, 640, 480, "y8", 30),
        ("infrared", 2, 640, 480, "y8", 30),
        ("accel", "motion_xyz32f", 100),
        ("gyro", "motion_xyz32f", 200),
    ]


def test_usb2_is_rejected_unless_validation_is_disabled() -> None:
    source = RealSenseSource(options={"require_usb3": True})

    with pytest.raises(RuntimeError, match=r"USB 2\.1.*USB 3"):
        source._validate_usb_connection("2.1")
    with pytest.raises(RuntimeError, match="Unable to verify"):
        source._validate_usb_connection(None)
    source._validate_usb_connection("3.2")

    reduced_bandwidth_source = RealSenseSource(options={"require_usb3": False})
    reduced_bandwidth_source._validate_usb_connection("2.1")
    reduced_bandwidth_source._validate_usb_connection(None)


def test_dependency_is_loaded_only_when_start_is_called(monkeypatch) -> None:
    imports: list[str] = []

    def _missing_dependency(name: str):
        imports.append(name)
        raise ImportError(name)

    monkeypatch.setattr(realsense_source.importlib, "import_module", _missing_dependency)
    source = RealSenseSource(options={"require_usb3": False})

    assert imports == []
    with pytest.raises(RuntimeError, match="requires pyrealsense2"):
        source.start()
    assert imports == ["pyrealsense2"]
    assert source.is_running is False


def test_frameset_conversion_owns_arrays_and_keeps_sensor_metadata() -> None:
    source = RealSenseSource(
        buffer_size=1,
        options={
            "require_usb3": False,
            "infrared": {"enabled": True, "indices": [1, 2]},
        },
    )
    raw_color_data = np.full((2, 3, 3), 1, dtype=np.uint8)
    aligned_color_data = np.full((2, 3, 3), 2, dtype=np.uint8)
    aligned_depth_data = np.full((2, 3), 1500, dtype=np.uint16)
    infrared_1_data = np.full((2, 3), 11, dtype=np.uint8)
    infrared_2_data = np.full((2, 3), 22, dtype=np.uint8)

    raw_frameset = _FakeFrameset(
        color=_FakeVideoFrame(raw_color_data, frame_number=9),
        infrared={
            1: _FakeVideoFrame(infrared_1_data, frame_number=9),
            2: _FakeVideoFrame(infrared_2_data, frame_number=9),
        },
    )
    aligned_frameset = _FakeFrameset(
        color=_FakeVideoFrame(aligned_color_data, frame_number=9, timestamp=8765.0),
        depth=_FakeVideoFrame(aligned_depth_data, frame_number=9),
    )
    align = _FakeAlign(aligned_frameset)
    intrinsics = PinholeIntrinsics(width=3, height=2, fx=100.0, fy=101.0, ppx=1.0, ppy=0.5)
    accel = MotionSample((0.0, 9.8, 0.0), 8750.0, "hardware_clock")
    gyro = MotionSample((0.1, 0.2, 0.3), 8755.0, "hardware_clock")

    source._align = align
    source._depth_scale_m = 0.001
    source._color_intrinsics = intrinsics
    source._usb_type = "3.2"
    envelope = realsense_source._FramesetEnvelope(
        frameset=raw_frameset,
        host_timestamp=100.25,
        accel_samples=(accel,),
        gyro_samples=(gyro,),
    )

    packet = source._packet_from_frameset(envelope)

    assert packet is not None
    assert align.inputs == [raw_frameset]
    assert packet.timestamp == 100.25
    assert packet.sensor is not None
    assert packet.sensor.frame_number == 9
    assert packet.sensor.device_timestamp_ms == 8765.0
    assert packet.sensor.depth_scale_m == 0.001
    assert packet.sensor.color_intrinsics == intrinsics
    assert packet.sensor.usb_type == "3.2"
    assert packet.sensor.primary_stream == "color"
    assert packet.sensor.accel_samples == (accel,)
    assert packet.sensor.gyro_samples == (gyro,)
    assert set(packet.sensor.infrared) == {1, 2}
    np.testing.assert_array_equal(packet.frame, np.full((2, 3, 3), 2, dtype=np.uint8))
    np.testing.assert_array_equal(packet.sensor.aligned_depth, np.full((2, 3), 1500, dtype=np.uint16))

    aligned_color_data.fill(99)
    aligned_depth_data.fill(99)
    infrared_1_data.fill(99)
    infrared_2_data.fill(99)
    assert np.all(packet.frame == 2)
    assert np.all(packet.sensor.aligned_depth == 1500)
    assert np.all(packet.sensor.infrared[1] == 11)
    assert np.all(packet.sensor.infrared[2] == 22)
    assert source._packet_from_frameset(envelope) is None

    source._accept_frames.set()
    replacement = FramePacket(frame=np.zeros((1, 1, 3), dtype=np.uint8), timestamp=101.0)
    source._offer_packet(packet)
    source._offer_packet(replacement)
    assert packet.frame is None
    assert packet.sensor is None
    assert source.read(timeout=0.0) is replacement
    source.stop()


def test_motion_callback_attaches_samples_to_next_frameset() -> None:
    fake_rs = _fake_rs_namespace()
    source = RealSenseSource(options={"require_usb3": False})
    source._rs = fake_rs
    source._accept_frames.set()

    class _MotionFrame:
        def __init__(self, stream_type: str, xyz: tuple[float, float, float], timestamp: float) -> None:
            self.stream_type = stream_type
            self.xyz = xyz
            self.timestamp = timestamp

        def is_motion_frame(self) -> bool:
            return True

        def is_frameset(self) -> bool:
            return False

        def get_profile(self):
            return SimpleNamespace(stream_type=lambda: self.stream_type)

        def as_motion_frame(self):
            return SimpleNamespace(get_motion_data=lambda: SimpleNamespace(
                x=self.xyz[0],
                y=self.xyz[1],
                z=self.xyz[2],
            ))

        def get_timestamp(self) -> float:
            return self.timestamp

        def get_frame_timestamp_domain(self) -> str:
            return "hardware_clock"

    frameset_token = object()

    class _FramesetCallbackFrame:
        def is_motion_frame(self) -> bool:
            return False

        def is_frameset(self) -> bool:
            return True

        def as_frameset(self):
            return frameset_token

    source._on_sdk_frame(_MotionFrame("accel", (1.0, 2.0, 3.0), 10.0))
    source._on_sdk_frame(_MotionFrame("gyro", (4.0, 5.0, 6.0), 11.0))
    source._on_sdk_frame(_FramesetCallbackFrame())

    envelope = source._framesets.get_nowait()
    assert envelope.frameset is frameset_token
    assert envelope.accel_samples == (MotionSample((1.0, 2.0, 3.0), 10.0, "hardware_clock"),)
    assert envelope.gyro_samples == (MotionSample((4.0, 5.0, 6.0), 11.0, "hardware_clock"),)
    assert source._pending_accel == []
    assert source._pending_gyro == []
    source.stop()


def test_stop_is_idempotent() -> None:
    class _Pipeline:
        def __init__(self) -> None:
            self.stop_calls = 0

        def stop(self) -> None:
            self.stop_calls += 1

    source = RealSenseSource(options={"require_usb3": False})
    pipeline = _Pipeline()
    source._pipeline = pipeline
    source._pipeline_started = True
    source._accept_frames.set()

    source.stop()
    source.stop()

    assert pipeline.stop_calls == 1
    assert source.is_running is False


def test_pipeline_forwards_realsense_source_and_options(monkeypatch) -> None:
    from pipelines.main_pipeline import RiskDetectionPipeline

    captured = {}
    sentinel = object()

    def _factory(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr("ingestion.realsense_source.create_realsense_source", _factory)
    pipeline = SimpleNamespace(
        pipeline_cfg=SimpleNamespace(fps=30.0, buffer_size=2),
        stream=SimpleNamespace(type="realsense", source="012345678901"),
        cfg={"ingestion": {"realsense": {"require_usb3": True}}},
    )

    result = RiskDetectionPipeline._build_source(pipeline)

    assert result is sentinel
    assert captured == {
        "source": "012345678901",
        "fps": 30.0,
        "buffer_size": 2,
        "options": {"require_usb3": True},
    }
