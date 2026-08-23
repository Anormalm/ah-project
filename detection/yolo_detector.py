from __future__ import annotations

from typing import Sequence

import numpy as np

from models.inference_engine import InferenceEngine
from utils.schemas import Detection


class UltralyticsYOLOEngine(InferenceEngine):
    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        input_size: int | tuple[int, int] | list[int] = 640,
        conf_threshold: float = 0.4,
        person_class_id: int = 0,
        allow_cpu_fallback: bool = False,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics package is required for YOLO backend") from exc
        self._model = YOLO(model_path)
        self.model_path = model_path
        self._device = device
        self.input_size = input_size
        self.conf_threshold = float(conf_threshold)
        self.person_class_id = int(person_class_id)
        self.allow_cpu_fallback = bool(allow_cpu_fallback)
        self._is_tensorrt = str(model_path).lower().endswith(".engine")

    def predict(self, inputs: np.ndarray | list[np.ndarray]):
        kwargs = {
            "device": self._device,
            "verbose": False,
            "imgsz": self.input_size,
            "conf": self.conf_threshold,
            "classes": [self.person_class_id],
            "rect": False,
        }
        try:
            return self._model.predict(inputs, **kwargs)
        except Exception as exc:
            message = str(exc).lower()
            can_fallback = self.allow_cpu_fallback and not self._is_tensorrt and self._device != "cpu"
            if can_fallback and ("torchvision::nms" in message or "cuda" in message):
                self._device = "cpu"
                kwargs["device"] = "cpu"
                return self._model.predict(inputs, **kwargs)
            raise

    def warmup(self) -> None:
        if isinstance(self.input_size, (tuple, list)):
            height, width = self.input_size
        else:
            height = width = int(self.input_size)
        self.predict(np.zeros((height, width, 3), dtype=np.uint8))


class MockDetectionEngine(InferenceEngine):
    def predict(self, inputs: np.ndarray | list[np.ndarray]):
        if isinstance(inputs, list):
            return [self._single(frame) for frame in inputs]
        return [self._single(inputs)]

    @staticmethod
    def _single(frame: np.ndarray):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = w * 0.25, h * 0.2, w * 0.45, h * 0.8
        return [((x1, y1, x2, y2), 0.92, 0)]


class YOLOPersonDetector:
    def __init__(
        self,
        backend: InferenceEngine,
        conf_threshold: float = 0.4,
        person_class_id: int = 0,
    ) -> None:
        self.backend = backend
        self.conf_threshold = conf_threshold
        self.person_class_id = person_class_id

    def detect(self, frame: np.ndarray) -> list[Detection]:
        raw = self.backend.predict(frame)
        return self._parse_results(raw[0])

    def detect_batch(self, frames: Sequence[np.ndarray]) -> list[list[Detection]]:
        raw = self.backend.predict(list(frames))
        return [self._parse_results(item) for item in raw]

    def _parse_results(self, raw_item) -> list[Detection]:
        if isinstance(raw_item, list):
            detections: list[Detection] = []
            for bbox, conf, cls_id in raw_item:
                if int(cls_id) != self.person_class_id or conf < self.conf_threshold:
                    continue
                detections.append(
                    Detection(
                        bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                        confidence=float(conf),
                        class_id=int(cls_id),
                        class_name="person",
                    )
                )
            return detections

        detections = []
        boxes = raw_item.boxes
        if boxes is None:
            return detections
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy()
        for i in range(len(xyxy)):
            cls_id = int(cls[i])
            score = float(conf[i])
            if cls_id != self.person_class_id or score < self.conf_threshold:
                continue
            x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=score,
                    class_id=cls_id,
                    class_name="person",
                )
            )
        return detections

