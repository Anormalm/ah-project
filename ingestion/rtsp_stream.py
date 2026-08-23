from __future__ import annotations

import cv2

from ingestion.video_loader import AsyncVideoSource


def _gst_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def build_jetson_rtsp_pipeline(
    rtsp_url: str,
    codec: str = "h264",
    latency_ms: int = 120,
    transport: str = "tcp",
) -> str:
    normalized_codec = codec.lower().replace(".", "")
    if normalized_codec in {"h265", "hevc"}:
        depay, parser = "rtph265depay", "h265parse"
    elif normalized_codec in {"h264", "avc"}:
        depay, parser = "rtph264depay", "h264parse"
    else:
        raise ValueError(f"Unsupported RTSP codec for Jetson GStreamer pipeline: {codec}")
    protocol = transport.lower()
    if protocol not in {"tcp", "udp"}:
        raise ValueError(f"Unsupported RTSP transport: {transport}")
    return " ! ".join(
        [
            f"rtspsrc location={_gst_quote(rtsp_url)} latency={max(0, int(latency_ms))} protocols={protocol} drop-on-latency=true",
            depay,
            parser,
            "nvv4l2decoder enable-max-performance=1",
            "nvvidconv",
            "video/x-raw,format=BGRx",
            "videoconvert",
            "video/x-raw,format=BGR",
            "appsink drop=true max-buffers=1 sync=false",
        ]
    )


class RTSPStreamSource(AsyncVideoSource):
    def __init__(
        self,
        rtsp_url: str,
        target_fps: float = 15.0,
        buffer_size: int = 4,
        reconnect_interval_sec: float = 2.0,
        max_retries: int = 5,
        use_jetson_gstreamer: bool = False,
        codec: str = "h264",
        latency_ms: int = 120,
        transport: str = "tcp",
        runtime_reconnect: bool = True,
        runtime_max_retries: int | None = None,
    ) -> None:
        source = (
            build_jetson_rtsp_pipeline(rtsp_url, codec=codec, latency_ms=latency_ms, transport=transport)
            if use_jetson_gstreamer
            else rtsp_url
        )
        api_preference = cv2.CAP_GSTREAMER if use_jetson_gstreamer else None
        super().__init__(
            source=source,
            target_fps=target_fps,
            buffer_size=buffer_size,
            api_preference=api_preference,
            startup_retries=max_retries,
            reconnect_interval_sec=reconnect_interval_sec,
            runtime_reconnect=runtime_reconnect,
            runtime_max_retries=runtime_max_retries,
        )


def create_rtsp_source(
    rtsp_url: str,
    fps: float,
    buffer_size: int,
    reconnect_interval_sec: float = 2.0,
    max_retries: int = 5,
    use_jetson_gstreamer: bool = False,
    codec: str = "h264",
    latency_ms: int = 120,
    transport: str = "tcp",
    runtime_reconnect: bool = True,
    runtime_max_retries: int | None = None,
) -> RTSPStreamSource:
    return RTSPStreamSource(
        rtsp_url=rtsp_url,
        target_fps=fps,
        buffer_size=buffer_size,
        reconnect_interval_sec=reconnect_interval_sec,
        max_retries=max_retries,
        use_jetson_gstreamer=use_jetson_gstreamer,
        codec=codec,
        latency_ms=latency_ms,
        transport=transport,
        runtime_reconnect=runtime_reconnect,
        runtime_max_retries=runtime_max_retries,
    )

