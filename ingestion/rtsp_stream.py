from __future__ import annotations

import time
from typing import Optional

from ingestion.video_loader import AsyncVideoSource


class RTSPStreamSource(AsyncVideoSource):
    def __init__(
        self,
        rtsp_url: str,
        target_fps: float = 15.0,
        buffer_size: int = 4,
        reconnect_interval_sec: float = 2.0,
        max_retries: int = 5,
    ) -> None:
        super().__init__(source=rtsp_url, target_fps=target_fps, buffer_size=buffer_size)
        self.reconnect_interval_sec = reconnect_interval_sec
        self.max_retries = max_retries

    def start(self) -> None:
        retries = 0
        while retries <= self.max_retries:
            try:
                super().start()
                return
            except RuntimeError:
                retries += 1
                time.sleep(self.reconnect_interval_sec)
        raise RuntimeError("Unable to connect RTSP stream after retries")


def create_rtsp_source(
    rtsp_url: str,
    fps: float,
    buffer_size: int,
    reconnect_interval_sec: float = 2.0,
    max_retries: int = 5,
) -> RTSPStreamSource:
    return RTSPStreamSource(
        rtsp_url=rtsp_url,
        target_fps=fps,
        buffer_size=buffer_size,
        reconnect_interval_sec=reconnect_interval_sec,
        max_retries=max_retries,
    )

