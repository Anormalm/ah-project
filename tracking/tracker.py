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

        return results

    def get_track_history(self, track_id: int) -> list[list[tuple[float, float, float]]]:
        state = self._tracks.get(track_id)
        if state is None:
            return []
        return list(state.keypoints_history)

