from __future__ import annotations

import queue
import threading
import time
from pathlib import Path


class BoundedJSONLWriter:
    """Batch JSONL writes off the inference thread with bounded memory use."""

    def __init__(
        self,
        path: str | Path,
        queue_size: int = 2048,
        batch_size: int = 64,
        flush_interval_sec: float = 0.5,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.batch_size = max(1, int(batch_size))
        self.flush_interval_sec = max(0.01, float(flush_interval_sec))
        self._queue: queue.Queue[str | object] = queue.Queue(maxsize=max(1, int(queue_size)))
        self._sentinel = object()
        self._state_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._closed = False

    def start(self) -> None:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("JSONL writer is closed")
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=self._run, name=f"jsonl:{self.path.name}", daemon=True)
            self._thread.start()

    def write(self, line: str) -> None:
        self.start()
        while True:
            try:
                self._queue.put(line, timeout=0.5)
                return
            except queue.Full:
                thread = self._thread
                if thread is None or not thread.is_alive():
                    raise RuntimeError(f"JSONL writer thread stopped unexpectedly: {self.path}")

    def _write_lines(self, lines: list[str]) -> None:
        if not lines:
            return
        with self._write_lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write("\n".join(lines))
                handle.write("\n")

    def _run(self) -> None:
        batch: list[str] = []
        deadline = time.monotonic() + self.flush_interval_sec
        while True:
            timeout = max(0.0, deadline - time.monotonic())
            try:
                item = self._queue.get(timeout=timeout)
            except queue.Empty:
                item = None

            if item is self._sentinel:
                self._queue.task_done()
                self._write_lines(batch)
                return
            if isinstance(item, str):
                batch.append(item)
                self._queue.task_done()

            now = time.monotonic()
            if len(batch) >= self.batch_size or now >= deadline:
                self._write_lines(batch)
                batch.clear()
                deadline = now + self.flush_interval_sec

    def close(self, timeout: float = 3.0) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            thread = self._thread
        if thread is None:
            return
        self._queue.put(self._sentinel)
        thread.join(timeout=max(0.0, float(timeout)))
