from __future__ import annotations

import json
import threading
from collections import deque
from pathlib import Path
from typing import Any

from utils.logger import setup_logger
from utils.schemas import AlertRecord, RiskEvent


class AlertManager:
    def __init__(
        self,
        json_log_path: str,
        enable_api: bool = False,
        api_host: str = "0.0.0.0",
        api_port: int = 8000,
        logger_name: str = "alert_manager",
    ) -> None:
        self.log_path = Path(json_log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.enable_api = enable_api
        self.api_host = api_host
        self.api_port = api_port
        self.logger = setup_logger(name=logger_name)
        self._latest: deque[dict[str, Any]] = deque(maxlen=256)
        self._lock = threading.Lock()
        self._server_thread: threading.Thread | None = None

    def start(self) -> None:
        if self.enable_api:
            self._start_api_server()

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
            self._latest.append(payload)

    def _start_api_server(self) -> None:
        if self._server_thread and self._server_thread.is_alive():
            return
        try:
            import uvicorn
            from fastapi import FastAPI
        except ImportError:
            self.logger.error("FastAPI/uvicorn not installed, API alert endpoint disabled")
            return

        app = FastAPI(title="Risk Alert API", version="1.0.0")

        @app.get("/health")
        def health() -> dict[str, str]:
            return {"status": "ok"}

        @app.get("/alerts")
        def alerts(limit: int = 50) -> list[dict[str, Any]]:
            with self._lock:
                data = list(self._latest)
            return data[-max(1, min(limit, 256)) :]

        def _run() -> None:
            uvicorn.run(app, host=self.api_host, port=self.api_port, log_level="warning")

        self._server_thread = threading.Thread(target=_run, daemon=True)
        self._server_thread.start()

