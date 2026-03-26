from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class InferenceEngine(ABC):
    @abstractmethod
    def predict(self, inputs: Any) -> Any:
        raise NotImplementedError


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

