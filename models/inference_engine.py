from __future__ import annotations

from abc import ABC, abstractmethod
import threading
from typing import Any

import numpy as np


class InferenceEngine(ABC):
    @abstractmethod
    def predict(self, inputs: Any) -> Any:
        raise NotImplementedError


class SynchronizedInferenceEngine(InferenceEngine):
    """Serialize access to a shared backend and warm it only once."""

    def __init__(self, backend: InferenceEngine) -> None:
        self.backend = backend
        self._lock = threading.RLock()
        self._warmed = False

    def predict(self, inputs: Any) -> Any:
        with self._lock:
            return self.backend.predict(inputs)

    def warmup(self) -> None:
        with self._lock:
            if self._warmed:
                return
            warmup = getattr(self.backend, "warmup", None)
            if callable(warmup):
                warmup()
            self._warmed = True


class OnnxRuntimeEngine(InferenceEngine):
    def __init__(self, model_path: str, providers: list[str] | None = None) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required for OnnxRuntimeEngine") from exc
        self._ort = ort
        self.session = ort.InferenceSession(model_path, providers=providers or ["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, inputs: np.ndarray) -> list[np.ndarray]:
        return self.session.run(None, {self.input_name: inputs})


class TensorRTEngine(InferenceEngine):
    def __init__(self, engine_path: str) -> None:
        self.engine_path = engine_path
        self._runtime = None
        raise NotImplementedError("TensorRT runtime binding should be implemented on target device")

    def predict(self, inputs: Any) -> Any:
        raise NotImplementedError

