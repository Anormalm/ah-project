from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


try:
    import psutil
except ImportError:
    psutil = None


@dataclass
class FPSMonitor:
    alpha: float = 0.9
    _last_ts: float = field(default_factory=time.perf_counter)
    _ema_fps: float = 0.0

    def tick(self) -> float:
        now = time.perf_counter()
        dt = max(now - self._last_ts, 1e-6)
        self._last_ts = now
        fps = 1.0 / dt
        self._ema_fps = fps if self._ema_fps == 0 else self.alpha * self._ema_fps + (1 - self.alpha) * fps
        return self._ema_fps


@dataclass
class ModuleLatency:
    calls: int = 0
    total_ms: float = 0.0
    max_ms: float = 0.0

    def update(self, elapsed_ms: float) -> None:
        self.calls += 1
        self.total_ms += elapsed_ms
        self.max_ms = max(self.max_ms, elapsed_ms)

    @property
    def avg_ms(self) -> float:
        if self.calls == 0:
            return 0.0
        return self.total_ms / self.calls


class PerformanceTracker:
    def __init__(self) -> None:
        self._latencies: dict[str, ModuleLatency] = {}

    @contextmanager
    def track(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        yield
        elapsed_ms = (time.perf_counter() - start) * 1000
        bucket = self._latencies.setdefault(name, ModuleLatency())
        bucket.update(elapsed_ms)

    def summary(self) -> dict[str, dict[str, float]]:
        data: dict[str, dict[str, float]] = {}
        for name, metric in self._latencies.items():
            data[name] = {
                "avg_ms": round(metric.avg_ms, 3),
                "max_ms": round(metric.max_ms, 3),
                "calls": metric.calls,
            }
        return data

    @staticmethod
    def memory_usage_mb() -> float:
        if psutil is None:
            return -1.0
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)

