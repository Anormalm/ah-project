from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from models.inference_engine import InferenceEngine
from utils.schemas import PoseResult


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0.0 else inter / denom


class MockPoseEngine(InferenceEngine):
    def predict(self, inputs: tuple[np.ndarray, list[tuple[float, float, float, float]]]) -> list[np.ndarray]:
        frame, bboxes = inputs
        _ = frame
        outputs: list[np.ndarray] = []
        for x1, y1, x2, y2 in bboxes:
            w = max(x2 - x1, 1.0)
            h = max(y2 - y1, 1.0)
            points = np.array(
                [
                    [x1 + 0.5 * w, y1 + 0.08 * h, 0.92],
                    [x1 + 0.4 * w, y1 + 0.1 * h, 0.88],
                    [x1 + 0.6 * w, y1 + 0.1 * h, 0.88],
                    [x1 + 0.35 * w, y1 + 0.22 * h, 0.85],
                    [x1 + 0.65 * w, y1 + 0.22 * h, 0.85],
                    [x1 + 0.32 * w, y1 + 0.38 * h, 0.82],
                    [x1 + 0.68 * w, y1 + 0.38 * h, 0.82],
                    [x1 + 0.45 * w, y1 + 0.35 * h, 0.9],
                    [x1 + 0.55 * w, y1 + 0.35 * h, 0.9],
                    [x1 + 0.42 * w, y1 + 0.55 * h, 0.86],
                    [x1 + 0.58 * w, y1 + 0.55 * h, 0.86],
                    [x1 + 0.4 * w, y1 + 0.72 * h, 0.84],
                    [x1 + 0.6 * w, y1 + 0.72 * h, 0.84],
                    [x1 + 0.38 * w, y1 + 0.9 * h, 0.82],
                    [x1 + 0.62 * w, y1 + 0.9 * h, 0.82],
                    [x1 + 0.37 * w, y1 + 0.98 * h, 0.8],
                    [x1 + 0.63 * w, y1 + 0.98 * h, 0.8],
                ],
                dtype=np.float32,
            )
            outputs.append(points)
        return outputs

    def predict_full(self, frame: np.ndarray) -> list[tuple[tuple[float, float, float, float], np.ndarray, float]]:
        h, w = frame.shape[:2]
        bbox = (w * 0.3, h * 0.15, w * 0.7, h * 0.95)
        kpts = self.predict((frame, [bbox]))[0]
        return [(bbox, kpts, 0.9)]


class UltralyticsPoseEngine(InferenceEngine):
    def __init__(self, model_path: str, device: str = "cpu", min_match_iou: float = 0.1, input_size: int = 256) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics package is required for ultralytics pose backend") from exc
        self._model = YOLO(model_path)
        self._device = device
        self.min_match_iou = min_match_iou
        self.input_size = input_size

    def _infer(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        try:
            result = self._model.predict(frame, device=self._device, imgsz=self.input_size, verbose=False)[0]
        except Exception as exc:
            message = str(exc).lower()
            if self._device != "cpu" and "torchvision::nms" in message:
                self._device = "cpu"
                result = self._model.predict(frame, device=self._device, imgsz=self.input_size, verbose=False)[0]
            else:
                raise
        if result.boxes is None or result.keypoints is None:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0, 17, 3), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )

        pose_boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        pose_conf = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
        pose_kpts = result.keypoints.data.detach().cpu().numpy().astype(np.float32)
        if pose_kpts.shape[-1] == 2:
            ones = np.ones((pose_kpts.shape[0], pose_kpts.shape[1], 1), dtype=np.float32)
            pose_kpts = np.concatenate([pose_kpts, ones], axis=2)
        return pose_boxes, pose_kpts, pose_conf

    def predict(self, inputs: tuple[np.ndarray, list[tuple[float, float, float, float]]]) -> list[np.ndarray]:
        frame, bboxes = inputs
        if not bboxes:
            return []

        pose_boxes, pose_kpts, _ = self._infer(frame)
        if pose_boxes.shape[0] == 0:
            return [np.zeros((17, 3), dtype=np.float32) for _ in bboxes]

        outputs: list[np.ndarray] = []
        for target in bboxes:
            best_idx = -1
            best_iou = 0.0
            for i, pb in enumerate(pose_boxes):
                iou = _bbox_iou(target, (float(pb[0]), float(pb[1]), float(pb[2]), float(pb[3])))
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_idx >= 0 and best_iou >= self.min_match_iou:
                outputs.append(pose_kpts[best_idx])
            else:
                outputs.append(np.zeros((17, 3), dtype=np.float32))
        return outputs

    def predict_full(self, frame: np.ndarray) -> list[tuple[tuple[float, float, float, float], np.ndarray, float]]:
        pose_boxes, pose_kpts, pose_conf = self._infer(frame)
        outputs: list[tuple[tuple[float, float, float, float], np.ndarray, float]] = []
        for box, kpts, conf in zip(pose_boxes, pose_kpts, pose_conf):
            outputs.append(((float(box[0]), float(box[1]), float(box[2]), float(box[3])), kpts, float(conf)))
        return outputs


class RTMOMMPoseEngine(InferenceEngine):
    def __init__(self, model_alias: str = "rtmo-m", device: str = "cpu", bbox_thr: float = 0.2) -> None:
        try:
            from mmpose.apis import MMPoseInferencer
        except ImportError as exc:
            raise RuntimeError(
                "mmpose is required for RTMO backend. Install with: "
                "pip install -U openmim && mim install mmengine mmcv mmpose"
            ) from exc

        self._inferencer = MMPoseInferencer(pose2d=model_alias, device=device)
        self.bbox_thr = bbox_thr

    @staticmethod
    def _to_kpts_xyc(keypoints: Any, scores: Any | None) -> np.ndarray:
        k = np.array(keypoints, dtype=np.float32)
        if k.ndim == 1:
            k = k.reshape(-1, 2)
        if k.ndim != 2:
            return np.zeros((17, 3), dtype=np.float32)
        if k.shape[1] >= 3:
            return k[:, :3].astype(np.float32)
        if scores is None:
            conf = np.ones((k.shape[0], 1), dtype=np.float32)
        else:
            s = np.array(scores, dtype=np.float32).reshape(-1, 1)
            if s.shape[0] != k.shape[0]:
                s = np.ones((k.shape[0], 1), dtype=np.float32)
            conf = s
        return np.concatenate([k[:, :2], conf], axis=1).astype(np.float32)

    @staticmethod
    def _to_bbox(instance: dict[str, Any], kpts: np.ndarray) -> tuple[float, float, float, float]:
        bbox = instance.get("bbox")
        if bbox is not None:
            b = np.array(bbox, dtype=np.float32).reshape(-1)
            if b.size >= 4:
                return float(b[0]), float(b[1]), float(b[2]), float(b[3])
        valid = kpts[kpts[:, 2] > 0.05]
        if valid.size == 0:
            return (0.0, 0.0, 0.0, 0.0)
        return (
            float(valid[:, 0].min()),
            float(valid[:, 1].min()),
            float(valid[:, 0].max()),
            float(valid[:, 1].max()),
        )

    def _infer(self, frame: np.ndarray) -> list[tuple[tuple[float, float, float, float], np.ndarray, float]]:
        out = next(self._inferencer(frame, return_vis=False, draw_bbox=False, show=False))
        predictions = out.get("predictions", [])
        if predictions and isinstance(predictions[0], list):
            instances = predictions[0]
        else:
            instances = predictions if isinstance(predictions, list) else []

        parsed: list[tuple[tuple[float, float, float, float], np.ndarray, float]] = []
        for inst in instances:
            if not isinstance(inst, dict):
                continue
            kpts = self._to_kpts_xyc(inst.get("keypoints", []), inst.get("keypoint_scores"))
            bbox = self._to_bbox(inst, kpts)
            bbox_score = float(inst.get("bbox_score", np.mean(kpts[:, 2]) if kpts.size else 0.0))
            if bbox_score < self.bbox_thr:
                continue
            parsed.append((bbox, kpts, bbox_score))
        return parsed

    def predict_full(self, frame: np.ndarray) -> list[tuple[tuple[float, float, float, float], np.ndarray, float]]:
        return self._infer(frame)

    def predict(self, inputs: tuple[np.ndarray, list[tuple[float, float, float, float]]]) -> list[np.ndarray]:
        frame, bboxes = inputs
        if not bboxes:
            return []
        full = self._infer(frame)
        if not full:
            return [np.zeros((17, 3), dtype=np.float32) for _ in bboxes]

        outputs: list[np.ndarray] = []
        full_boxes = [f[0] for f in full]
        for target in bboxes:
            best_idx = -1
            best_iou = 0.0
            for i, box in enumerate(full_boxes):
                iou = _bbox_iou(target, box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_idx >= 0 and best_iou >= 0.05:
                outputs.append(full[best_idx][1])
            else:
                outputs.append(np.zeros((17, 3), dtype=np.float32))
        return outputs


class MoveNetTorchEngine(InferenceEngine):
    def __init__(self, model_path: str, device: str = "cpu", input_size: int = 256) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch is required for MoveNetTorchEngine") from exc
        self._torch = torch
        self.device = device
        self.input_size = input_size
        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()

    def _preprocess(self, crop: np.ndarray):
        resized = cv2.resize(crop, (self.input_size, self.input_size))
        tensor = self._torch.from_numpy(resized).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def predict(self, inputs: tuple[np.ndarray, list[tuple[float, float, float, float]]]) -> list[np.ndarray]:
        frame, bboxes = inputs
        outputs: list[np.ndarray] = []
        h, w = frame.shape[:2]
        with self._torch.no_grad():
            for x1, y1, x2, y2 in bboxes:
                ix1 = max(int(x1), 0)
                iy1 = max(int(y1), 0)
                ix2 = min(int(x2), w - 1)
                iy2 = min(int(y2), h - 1)
                if ix2 <= ix1 or iy2 <= iy1:
                    outputs.append(np.zeros((17, 3), dtype=np.float32))
                    continue
                crop = frame[iy1:iy2, ix1:ix2]
                inp = self._preprocess(crop)
                pred = self.model(inp)
                arr = pred.squeeze().detach().cpu().numpy()
                if arr.shape[-1] != 3:
                    arr = arr.reshape(-1, 3)
                kp = np.zeros((arr.shape[0], 3), dtype=np.float32)
                kp[:, 0] = ix1 + arr[:, 1] * (ix2 - ix1)
                kp[:, 1] = iy1 + arr[:, 0] * (iy2 - iy1)
                kp[:, 2] = arr[:, 2]
                outputs.append(kp)
        return outputs


class PoseEstimator:
    def __init__(self, backend: InferenceEngine) -> None:
        self.backend = backend

    @staticmethod
    def _to_pose_result(bbox: tuple[float, float, float, float], kpts: np.ndarray) -> PoseResult:
        tuples = [(float(x), float(y), float(c)) for x, y, c in kpts]
        return PoseResult(bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])), keypoints=tuples)

    def predict(self, frame: np.ndarray, bboxes: list[tuple[float, float, float, float]]) -> list[PoseResult]:
        if not bboxes:
            return []
        raw = self.backend.predict((frame, bboxes))
        return [self._to_pose_result(bbox, kpts) for bbox, kpts in zip(bboxes, raw)]

    def predict_full_frame(self, frame: np.ndarray) -> list[PoseResult]:
        method = getattr(self.backend, "predict_full", None)
        if callable(method):
            raw: list[tuple[tuple[float, float, float, float], np.ndarray, float]] = method(frame)
            return [self._to_pose_result(bbox, kpts) for bbox, kpts, _ in raw]

        h, w = frame.shape[:2]
        return self.predict(frame, [(0.0, 0.0, float(w - 1), float(h - 1))])

