from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from utils.schemas import Detection, PoseResult, TrackPose

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


@dataclass
class _TrackState:
    track_id: int
    bbox: tuple[float, float, float, float]
    last_timestamp: float
    misses: int = 0
    keypoints_history: deque[list[tuple[float, float, float]]] = field(default_factory=lambda: deque(maxlen=32))


class ByteTrackLikeTracker:
    def __init__(self, iou_threshold: float = 0.3, max_misses: int = 20) -> None:
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self._next_id = 1
        self._tracks: dict[int, _TrackState] = {}
        self.last_removed_track_ids: list[int] = []

    @staticmethod
    def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter = inter_w * inter_h
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = area_a + area_b - inter
        if denom <= 0.0:
            return 0.0
        return inter / denom

    def _match(self, detections: list[Detection]) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        track_ids = list(self._tracks.keys())
        if not track_ids or not detections:
            return [], list(range(len(track_ids))), list(range(len(detections)))

        matrix = np.zeros((len(track_ids), len(detections)), dtype=np.float32)
        for i, tid in enumerate(track_ids):
            tbox = self._tracks[tid].bbox
            for j, det in enumerate(detections):
                matrix[i, j] = 1.0 - self._iou(tbox, det.bbox)

        matches: list[tuple[int, int]] = []
        unmatched_tracks = set(range(len(track_ids)))
        unmatched_dets = set(range(len(detections)))

        if linear_sum_assignment is not None:
            rows, cols = linear_sum_assignment(matrix)
            for r, c in zip(rows, cols):
                iou = 1.0 - matrix[r, c]
                if iou < self.iou_threshold:
                    continue
                matches.append((r, c))
                unmatched_tracks.discard(r)
                unmatched_dets.discard(c)
        else:
            for r in range(matrix.shape[0]):
                c = int(np.argmin(matrix[r]))
                iou = 1.0 - matrix[r, c]
                if iou >= self.iou_threshold and c in unmatched_dets and r in unmatched_tracks:
                    matches.append((r, c))
                    unmatched_tracks.discard(r)
                    unmatched_dets.discard(c)

        return matches, sorted(unmatched_tracks), sorted(unmatched_dets)

    def update(self, detections: list[Detection], poses: list[PoseResult], timestamp: float) -> list[TrackPose]:
        self.last_removed_track_ids = []
        pose_by_bbox = {pose.bbox: pose for pose in poses}
        track_ids = list(self._tracks.keys())
        matches, unmatched_track_idx, unmatched_det_idx = self._match(detections)

        results: list[TrackPose] = []

        for track_idx, det_idx in matches:
            tid = track_ids[track_idx]
            det = detections[det_idx]
            pose = pose_by_bbox.get(det.bbox)
            if pose is None:
                continue
            state = self._tracks[tid]
            state.bbox = det.bbox
            state.last_timestamp = timestamp
            state.misses = 0
            state.keypoints_history.append(pose.keypoints)
            results.append(TrackPose(track_id=tid, keypoints=pose.keypoints, timestamp=timestamp))

        for idx in unmatched_track_idx:
            tid = track_ids[idx]
            state = self._tracks[tid]
            state.misses += 1

        for det_idx in unmatched_det_idx:
            det = detections[det_idx]
            pose = pose_by_bbox.get(det.bbox)
            if pose is None:
                continue
            tid = self._next_id
            self._next_id += 1
            state = _TrackState(track_id=tid, bbox=det.bbox, last_timestamp=timestamp)
            state.keypoints_history.append(pose.keypoints)
            self._tracks[tid] = state
            results.append(TrackPose(track_id=tid, keypoints=pose.keypoints, timestamp=timestamp))

        stale_ids = [tid for tid, state in self._tracks.items() if state.misses > self.max_misses]
        for tid in stale_ids:
            del self._tracks[tid]
        self.last_removed_track_ids = stale_ids

        return results

    def get_track_history(self, track_id: int) -> list[list[tuple[float, float, float]]]:
        state = self._tracks.get(track_id)
        if state is None:
            return []
        return list(state.keypoints_history)


@dataclass
class _PoseKalmanState:
    track_id: int
    mean: np.ndarray
    covariance: np.ndarray
    last_timestamp: float
    keypoints: np.ndarray
    confidence: float
    misses: int = 0
    hits: int = 1
    keypoints_history: deque[list[tuple[float, float, float]]] = field(default_factory=lambda: deque(maxlen=64))


class PoseAwareKalmanTracker:
    """Edge-friendly multi-person tracker using motion, IoU and pose association.

    The two-pass confidence association follows ByteTrack's useful high/low score
    idea, while a constant-velocity Kalman filter and an OKS-like pose term make
    identities substantially more stable than IoU-only matching during crossings.
    """

    _COCO_SIGMAS = np.array(
        [0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
         0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089],
        dtype=np.float32,
    )

    def __init__(
        self,
        iou_threshold: float = 0.15,
        max_misses: int = 20,
        track_high_thresh: float = 0.35,
        track_low_thresh: float = 0.10,
        new_track_thresh: float = 0.40,
        pose_similarity_threshold: float = 0.20,
        max_center_distance: float = 1.25,
        iou_weight: float = 0.50,
        pose_weight: float = 0.40,
        motion_weight: float = 0.10,
        keypoint_ema_alpha: float = 0.70,
    ) -> None:
        if track_low_thresh > track_high_thresh:
            raise ValueError("track_low_thresh must be <= track_high_thresh")
        weights = np.array([iou_weight, pose_weight, motion_weight], dtype=np.float32)
        if np.any(weights < 0.0) or float(weights.sum()) <= 0.0:
            raise ValueError("association weights must be non-negative with a positive sum")
        weights /= weights.sum()

        self.iou_threshold = float(iou_threshold)
        self.max_misses = int(max_misses)
        self.track_high_thresh = float(track_high_thresh)
        self.track_low_thresh = float(track_low_thresh)
        self.new_track_thresh = float(new_track_thresh)
        self.pose_similarity_threshold = float(pose_similarity_threshold)
        self.max_center_distance = float(max_center_distance)
        self.iou_weight, self.pose_weight, self.motion_weight = (float(v) for v in weights)
        self.keypoint_ema_alpha = float(np.clip(keypoint_ema_alpha, 0.0, 1.0))
        self._next_id = 1
        self._tracks: dict[int, _PoseKalmanState] = {}
        self.last_removed_track_ids: list[int] = []

    @staticmethod
    def _bbox_to_xywh(bbox: tuple[float, float, float, float]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        return np.array(
            [(x1 + x2) * 0.5, (y1 + y2) * 0.5, max(x2 - x1, 1.0), max(y2 - y1, 1.0)],
            dtype=np.float32,
        )

    @staticmethod
    def _xywh_to_bbox(xywh: np.ndarray) -> tuple[float, float, float, float]:
        cx, cy, w, h = (float(v) for v in xywh[:4])
        w = max(w, 1.0)
        h = max(h, 1.0)
        return cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5

    @staticmethod
    def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        return ByteTrackLikeTracker._iou(a, b)

    @classmethod
    def _pose_similarity(
        cls,
        a: np.ndarray,
        b: np.ndarray,
        bbox: tuple[float, float, float, float],
    ) -> float:
        count = min(a.shape[0], b.shape[0], cls._COCO_SIGMAS.shape[0])
        if count == 0:
            return 0.0
        valid = (a[:count, 2] >= 0.2) & (b[:count, 2] >= 0.2)
        if int(valid.sum()) < 3:
            return 0.0
        x1, y1, x2, y2 = bbox
        area = max((x2 - x1) * (y2 - y1), 1.0)
        dist2 = np.square(a[:count, :2] - b[:count, :2]).sum(axis=1)
        denom = 2.0 * area * np.square(2.0 * cls._COCO_SIGMAS[:count]) + 1e-6
        score = np.exp(-dist2 / denom)
        confidence = np.minimum(a[:count, 2], b[:count, 2])
        weights = confidence[valid]
        return float(np.sum(score[valid] * weights) / max(float(weights.sum()), 1e-6))

    @staticmethod
    def _motion_distance(
        predicted: tuple[float, float, float, float],
        observed: tuple[float, float, float, float],
    ) -> float:
        p = PoseAwareKalmanTracker._bbox_to_xywh(predicted)
        o = PoseAwareKalmanTracker._bbox_to_xywh(observed)
        scale = max(float(np.hypot(p[2], p[3])), 1.0)
        return float(np.hypot(p[0] - o[0], p[1] - o[1]) / scale)

    @staticmethod
    def _init_filter(bbox: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray]:
        measurement = PoseAwareKalmanTracker._bbox_to_xywh(bbox)
        mean = np.concatenate([measurement, np.zeros(4, dtype=np.float32)])
        scale = max(float(measurement[2]), float(measurement[3]), 1.0)
        std = np.array([0.08 * scale] * 4 + [0.15 * scale] * 4, dtype=np.float32)
        return mean, np.diag(np.square(std)).astype(np.float32)

    @staticmethod
    def _predict_filter(state: _PoseKalmanState, timestamp: float) -> None:
        dt = float(np.clip(timestamp - state.last_timestamp, 1.0 / 120.0, 1.0))
        transition = np.eye(8, dtype=np.float32)
        transition[:4, 4:] = np.eye(4, dtype=np.float32) * dt
        scale = max(float(state.mean[2]), float(state.mean[3]), 1.0)
        process_std = np.array([0.025 * scale] * 4 + [0.05 * scale] * 4, dtype=np.float32)
        state.mean = transition @ state.mean
        state.mean[2:4] = np.maximum(state.mean[2:4], 1.0)
        state.covariance = transition @ state.covariance @ transition.T + np.diag(np.square(process_std))
        state.last_timestamp = timestamp

    @staticmethod
    def _update_filter(state: _PoseKalmanState, bbox: tuple[float, float, float, float]) -> None:
        measurement = PoseAwareKalmanTracker._bbox_to_xywh(bbox)
        observation = np.zeros((4, 8), dtype=np.float32)
        observation[:, :4] = np.eye(4, dtype=np.float32)
        scale = max(float(measurement[2]), float(measurement[3]), 1.0)
        measurement_cov = np.diag(np.square(np.array([0.05 * scale] * 4, dtype=np.float32)))
        innovation_cov = observation @ state.covariance @ observation.T + measurement_cov
        gain = state.covariance @ observation.T @ np.linalg.pinv(innovation_cov)
        state.mean = state.mean + gain @ (measurement - observation @ state.mean)
        state.mean[2:4] = np.maximum(state.mean[2:4], 1.0)
        state.covariance = (np.eye(8, dtype=np.float32) - gain @ observation) @ state.covariance

    def _association_cost(
        self,
        state: _PoseKalmanState,
        detection: Detection,
        pose: PoseResult,
    ) -> float:
        predicted_bbox = self._xywh_to_bbox(state.mean)
        iou = self._iou(predicted_bbox, detection.bbox)
        pose_score = self._pose_similarity(state.keypoints, np.asarray(pose.keypoints, dtype=np.float32), predicted_bbox)
        motion = self._motion_distance(predicted_bbox, detection.bbox)
        if motion > self.max_center_distance:
            return 1e6
        if iou < self.iou_threshold and pose_score < self.pose_similarity_threshold:
            return 1e6
        motion_cost = min(motion / max(self.max_center_distance, 1e-6), 1.0)
        cost = (
            self.iou_weight * (1.0 - iou)
            + self.pose_weight * (1.0 - pose_score)
            + self.motion_weight * motion_cost
        )
        return float(cost * (2.0 - detection.confidence))

    def _match(
        self,
        track_ids: list[int],
        detection_indices: list[int],
        detections: list[Detection],
        poses: list[PoseResult],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not track_ids or not detection_indices:
            return [], track_ids.copy(), detection_indices.copy()
        matrix = np.full((len(track_ids), len(detection_indices)), 1e6, dtype=np.float32)
        for row, track_id in enumerate(track_ids):
            state = self._tracks[track_id]
            for col, detection_index in enumerate(detection_indices):
                matrix[row, col] = self._association_cost(
                    state,
                    detections[detection_index],
                    poses[detection_index],
                )

        if linear_sum_assignment is None:
            assignments = []
            available_cols = set(range(matrix.shape[1]))
            for row in range(matrix.shape[0]):
                if not available_cols:
                    break
                col = min(available_cols, key=lambda idx: float(matrix[row, idx]))
                assignments.append((row, col))
                available_cols.remove(col)
        else:
            rows, cols = linear_sum_assignment(matrix)
            assignments = list(zip(rows.tolist(), cols.tolist()))

        matches: list[tuple[int, int]] = []
        unmatched_tracks = set(track_ids)
        unmatched_detections = set(detection_indices)
        for row, col in assignments:
            if float(matrix[row, col]) >= 1e5:
                continue
            track_id = track_ids[row]
            detection_index = detection_indices[col]
            matches.append((track_id, detection_index))
            unmatched_tracks.discard(track_id)
            unmatched_detections.discard(detection_index)
        return matches, sorted(unmatched_tracks), sorted(unmatched_detections)

    def _smooth_keypoints(self, state: _PoseKalmanState, pose: PoseResult) -> np.ndarray:
        current = np.asarray(pose.keypoints, dtype=np.float32)
        previous = state.keypoints
        if previous.shape != current.shape:
            return current
        smoothed = current.copy()
        valid = (current[:, 2] >= 0.2) & (previous[:, 2] >= 0.2)
        alpha = self.keypoint_ema_alpha
        smoothed[valid, :2] = alpha * current[valid, :2] + (1.0 - alpha) * previous[valid, :2]
        return smoothed

    def _apply_match(
        self,
        track_id: int,
        detection_index: int,
        detections: list[Detection],
        poses: list[PoseResult],
        timestamp: float,
    ) -> TrackPose:
        state = self._tracks[track_id]
        detection = detections[detection_index]
        pose = poses[detection_index]
        self._update_filter(state, detection.bbox)
        state.keypoints = self._smooth_keypoints(state, pose)
        state.confidence = float(detection.confidence)
        state.last_timestamp = timestamp
        state.misses = 0
        state.hits += 1
        keypoints = [tuple(float(v) for v in point) for point in state.keypoints]
        state.keypoints_history.append(keypoints)
        return TrackPose(track_id=track_id, keypoints=keypoints, timestamp=timestamp)

    def _new_track(self, detection: Detection, pose: PoseResult, timestamp: float) -> TrackPose:
        track_id = self._next_id
        self._next_id += 1
        mean, covariance = self._init_filter(detection.bbox)
        keypoint_array = np.asarray(pose.keypoints, dtype=np.float32)
        keypoints = [tuple(float(v) for v in point) for point in keypoint_array]
        state = _PoseKalmanState(
            track_id=track_id,
            mean=mean,
            covariance=covariance,
            last_timestamp=timestamp,
            keypoints=keypoint_array,
            confidence=float(detection.confidence),
        )
        state.keypoints_history.append(keypoints)
        self._tracks[track_id] = state
        return TrackPose(track_id=track_id, keypoints=keypoints, timestamp=timestamp)

    def update(self, detections: list[Detection], poses: list[PoseResult], timestamp: float) -> list[TrackPose]:
        self.last_removed_track_ids = []
        pose_by_bbox = {pose.bbox: pose for pose in poses}
        paired = [(det, pose_by_bbox.get(det.bbox)) for det in detections]
        paired = [(det, pose) for det, pose in paired if pose is not None]
        valid_detections = [item[0] for item in paired]
        valid_poses = [item[1] for item in paired]

        for state in self._tracks.values():
            self._predict_filter(state, timestamp)

        high_indices = [
            idx for idx, det in enumerate(valid_detections) if det.confidence >= self.track_high_thresh
        ]
        low_indices = [
            idx
            for idx, det in enumerate(valid_detections)
            if self.track_low_thresh <= det.confidence < self.track_high_thresh
        ]
        track_ids = list(self._tracks)
        high_matches, unmatched_tracks, unmatched_high = self._match(
            track_ids, high_indices, valid_detections, valid_poses
        )
        low_matches, unmatched_tracks, _ = self._match(
            unmatched_tracks, low_indices, valid_detections, valid_poses
        )

        results = [
            self._apply_match(tid, det_idx, valid_detections, valid_poses, timestamp)
            for tid, det_idx in high_matches + low_matches
        ]
        for track_id in unmatched_tracks:
            self._tracks[track_id].misses += 1

        for detection_index in unmatched_high:
            detection = valid_detections[detection_index]
            if detection.confidence >= self.new_track_thresh:
                results.append(self._new_track(detection, valid_poses[detection_index], timestamp))

        stale_ids = [track_id for track_id, state in self._tracks.items() if state.misses > self.max_misses]
        for track_id in stale_ids:
            del self._tracks[track_id]
        self.last_removed_track_ids = stale_ids
        return results

    def get_track_history(self, track_id: int) -> list[list[tuple[float, float, float]]]:
        state = self._tracks.get(track_id)
        return [] if state is None else list(state.keypoints_history)

