from __future__ import annotations

import time

import numpy as np
import pytest

from ingestion.realsense_source import RealSenseSource, create_realsense_source
from pipelines.main_pipeline import StreamConfig
from run import load_config


def test_realsense_source_factory_defaults_to_first_device() -> None:
    source = create_realsense_source("auto", {"width": 848, "height": 480, "fps": 30}, buffer_size=1)
    assert source.serial is None
    assert (source.width, source.height, source.fps) == (848, 480, 30)
    assert source.enable_imu is False


def test_realsense_stream_type_is_valid() -> None:
    stream = StreamConfig(stream_id="d435i", type="realsense", source="auto")
    assert stream.type == "realsense"


def test_realsense_config_inherits_sota_profile() -> None:
    cfg = load_config("config/jetson_orin_nx_realsense_d435i.yaml")
    assert cfg["streams"][0]["type"] == "realsense"
    assert cfg["tracking"]["backend"] == "pose_kalman"
    assert cfg["ingestion"]["realsense"]["enable_imu"] is False


def test_missing_sdk_has_actionable_error(monkeypatch) -> None:
    source = RealSenseSource()
    monkeypatch.setitem(__import__("sys").modules, "pyrealsense2", None)
    with pytest.raises(RuntimeError, match="pyrealsense2"):
        source._sdk()


def test_rgbd_packet_is_aligned_and_buffered() -> None:
    color_data = np.full((2, 3, 3), 7, dtype=np.uint8)
    depth_data = np.full((2, 3), 1200, dtype=np.uint16)

    class _Frame:
        def __init__(self, data):
            self.data = data

        def get_data(self):
            return self.data

        def __bool__(self):
            return True

    class _Frames:
        def get_color_frame(self):
            return _Frame(color_data)

        def get_depth_frame(self):
            return _Frame(depth_data)

    class _Sensor:
        def get_depth_scale(self):
            return 0.001

    class _Device:
        def first_depth_sensor(self):
            return _Sensor()

    class _Profile:
        def get_device(self):
            return _Device()

    class _Pipeline:
        def __init__(self):
            self.stopped = False

        def start(self, config):
            return _Profile()

        def wait_for_frames(self, timeout):
            if self.stopped:
                raise RuntimeError("stopped")
            time.sleep(0.005)
            return _Frames()

        def stop(self):
            self.stopped = True

    class _Config:
        def enable_device(self, serial): pass
        def enable_stream(self, *args): pass

    class _Align:
        def process(self, frames): return frames

    class _RS:
        class stream:
            depth = "depth"
            color = "color"
            accel = "accel"
            gyro = "gyro"

        class format:
            z16 = "z16"
            bgr8 = "bgr8"

        pipeline = _Pipeline
        config = _Config

        @staticmethod
        def align(stream): return _Align()

    source = RealSenseSource(width=3, height=2, fps=30, rs_module=_RS)
    source.start()
    packet = source.read(timeout=0.5)
    source.stop()
    assert packet is not None
    assert np.array_equal(packet.frame, color_data)
    assert np.array_equal(packet.depth_frame, depth_data)
    assert packet.depth_scale_m == 0.001
