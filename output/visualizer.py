from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from utils.schemas import Detection, RiskEvent, TrackPose


COCO_SKELETON = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


@dataclass
class VisualizationConfig:
    enabled: bool = False
    show_bboxes: bool = True
    show_pose: bool = True
    show_tracks: bool = True
    show_risk: bool = True
    show_bed_zones: bool = True
    show_fps: bool = True
    keypoint_conf_threshold: float = 0.2
    window_scale: float = 1.0
    quit_key: str = "q"
    quit_keys: list[str] | None = None
    privacy_mode: str = "none"  # none | person_blur | person_pixelate | face_blur | face_pixelate | skeleton_only
    privacy_blur_kernel: int = 41
    privacy_pixelate_size: int = 14
    privacy_expand_ratio: float = 0.12


class Visualizer:
    _risk_colors = {
        "LOW": (80, 200, 80),
        "MEDIUM": (0, 215, 255),
        "HIGH": (0, 140, 255),
        "CRITICAL": (0, 0, 255),
    }

    def __init__(self, stream_id: str, cfg: VisualizationConfig) -> None:
        self.stream_id = stream_id
        self.cfg = cfg
        self.window_name = f"RiskView:{stream_id}"
        self._display_available = True
        self._last_output_frame: np.ndarray | None = None
        keys = cfg.quit_keys if cfg.quit_keys else [cfg.quit_key, "esc"]
        self._quit_keycodes = set()
        for key in keys:
            if key.lower() == "esc":
                self._quit_keycodes.add(27)
            elif key:
                self._quit_keycodes.add(ord(key.lower()[0]))

    def get_last_output_frame(self) -> np.ndarray | None:
        return self._last_output_frame

    @staticmethod
    def _clip_bbox(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def _expand_bbox(self, bbox: tuple[float, float, float, float], frame: np.ndarray) -> tuple[int, int, int, int] | None:
        x1, y1, x2, y2 = bbox
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        ex = w * self.cfg.privacy_expand_ratio
        ey = h * self.cfg.privacy_expand_ratio
        return self._clip_bbox(frame, (int(x1 - ex), int(y1 - ey), int(x2 + ex), int(y2 + ey)))

    @staticmethod
    def _ensure_odd(v: int) -> int:
        k = max(3, int(v))
        return k if k % 2 == 1 else k + 1

    def _blur_region(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
        x1, y1, x2, y2 = bbox
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return
        kernel = self._ensure_odd(self.cfg.privacy_blur_kernel)
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (kernel, kernel), 0)

    def _pixelate_region(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
        x1, y1, x2, y2 = bbox
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return
        h, w = roi.shape[:2]
        cell = max(4, int(self.cfg.privacy_pixelate_size))
        dw = max(1, w // cell)
        dh = max(1, h // cell)
        small = cv2.resize(roi, (dw, dh), interpolation=cv2.INTER_LINEAR)
        frame[y1:y2, x1:x2] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    def _person_regions(self, frame: np.ndarray, detections: list[Detection], tracks: list[TrackPose]) -> list[tuple[int, int, int, int]]:
        boxes: list[tuple[int, int, int, int]] = []
        for det in detections:
            box = self._expand_bbox(det.bbox, frame)
            if box is not None:
                boxes.append(box)

        if boxes:
            return boxes

        for track in tracks:
            pts = np.array(track.keypoints, dtype=np.float32)
            valid = pts[pts[:, 2] >= self.cfg.keypoint_conf_threshold]
            if valid.size == 0:
                continue
            bbox = (
                float(valid[:, 0].min()),
                float(valid[:, 1].min()),
                float(valid[:, 0].max()),
                float(valid[:, 1].max()),
            )
            box = self._expand_bbox(bbox, frame)
            if box is not None:
                boxes.append(box)
        return boxes

    def _face_region(self, frame: np.ndarray, track: TrackPose) -> tuple[int, int, int, int] | None:
        head_indices = [0, 1, 2, 3, 4]
        pts = np.array(track.keypoints, dtype=np.float32)
        if pts.shape[0] < 5:
            return None
        head = pts[head_indices]
        head = head[head[:, 2] >= self.cfg.keypoint_conf_threshold]
        if head.shape[0] < 2:
            return None
        x1, y1 = float(head[:, 0].min()), float(head[:, 1].min())
        x2, y2 = float(head[:, 0].max()), float(head[:, 1].max())
        w = max(12.0, x2 - x1)
        h = max(12.0, y2 - y1)
        margin_x = 0.9 * w
        margin_y = 1.0 * h
        return self._clip_bbox(frame, (int(x1 - margin_x), int(y1 - margin_y), int(x2 + margin_x), int(y2 + margin_y)))

    def _apply_privacy(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        tracks: list[TrackPose],
        privacy_mode_override: str | None = None,
    ) -> np.ndarray:
        mode = str(privacy_mode_override if privacy_mode_override is not None else self.cfg.privacy_mode or "none").lower()
        if mode in {"none", "off", "false"}:
            return frame.copy()

        if mode == "skeleton_only":
            return np.zeros_like(frame)

        canvas = frame.copy()
        if mode in {"person_blur", "person_pixelate"}:
            for box in self._person_regions(canvas, detections, tracks):
                if mode == "person_pixelate":
                    self._pixelate_region(canvas, box)
                else:
                    self._blur_region(canvas, box)
            return canvas

        if mode in {"face_blur", "face_pixelate"}:
            face_count = 0
            for track in tracks:
                face = self._face_region(canvas, track)
                if face is None:
                    continue
                face_count += 1
                if mode == "face_pixelate":
                    self._pixelate_region(canvas, face)
                else:
                    self._blur_region(canvas, face)
            # fallback: if faces are not detected reliably, anonymize full person regions
            if face_count == 0:
                for box in self._person_regions(canvas, detections, tracks):
                    self._blur_region(canvas, box)
            return canvas

        return frame.copy()

    def _draw_pose(self, frame: np.ndarray, keypoints: list[tuple[float, float, float]]) -> None:
        pts = np.array(keypoints, dtype=np.float32)

        for idx1, idx2 in COCO_SKELETON:
            if idx1 >= len(pts) or idx2 >= len(pts):
                continue
            x1, y1, c1 = pts[idx1]
            x2, y2, c2 = pts[idx2]
            if c1 < self.cfg.keypoint_conf_threshold or c2 < self.cfg.keypoint_conf_threshold:
                continue
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (120, 220, 255), 2)

        for x, y, c in pts:
            if c < self.cfg.keypoint_conf_threshold:
                continue
            cv2.circle(frame, (int(x), int(y)), 3, (255, 255, 255), -1)

    @staticmethod
    def _pose_center(keypoints: list[tuple[float, float, float]], min_conf: float) -> tuple[int, int]:
        pts = np.array(keypoints, dtype=np.float32)
        valid = pts[pts[:, 2] >= min_conf]
        if valid.size == 0:
            return (0, 0)
        return int(valid[:, 0].mean()), int(valid[:, 1].mean())

    def _compose_frame(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        tracks: list[TrackPose],
        risk_events: dict[int, RiskEvent],
        fps: float,
        bed_zones: list[tuple[float, float, float, float]] | None,
        privacy_mode_override: str | None = None,
    ) -> np.ndarray:
        canvas = self._apply_privacy(frame, detections, tracks, privacy_mode_override=privacy_mode_override)

        if self.cfg.show_bed_zones:
            for i, zone in enumerate(bed_zones or []):
                x1, y1, x2, y2 = [int(v) for v in zone]
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (180, 180, 60), 2)
                cv2.putText(canvas, f"BedZone{i}", (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 60), 1)

        if self.cfg.show_bboxes:
            for det in detections:
                x1, y1, x2, y2 = [int(v) for v in det.bbox]
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (60, 220, 60), 2)
                cv2.putText(
                    canvas,
                    f"person {det.confidence:.2f}",
                    (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (60, 220, 60),
                    1,
                )

        for track in tracks:
            if self.cfg.show_pose:
                self._draw_pose(canvas, track.keypoints)

            cx, cy = self._pose_center(track.keypoints, self.cfg.keypoint_conf_threshold)
            event = risk_events.get(track.track_id)
            risk_level = event.risk_level if event else "LOW"
            risk_conf = event.confidence if event else 0.0
            color = self._risk_colors.get(risk_level, (80, 200, 80))

            if self.cfg.show_tracks:
                cv2.putText(
                    canvas,
                    f"ID {track.track_id}",
                    (cx + 6, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )

            if self.cfg.show_risk:
                cv2.putText(
                    canvas,
                    f"{risk_level} {risk_conf:.2f}",
                    (cx + 6, cy + 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )

        if self.cfg.show_fps:
            cv2.putText(canvas, f"FPS {fps:.2f}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return canvas

    def render(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        tracks: list[TrackPose],
        risk_events: dict[int, RiskEvent],
        fps: float,
        bed_zones: list[tuple[float, float, float, float]] | None = None,
        privacy_mode_override: str | None = None,
    ) -> bool:
        canvas = self._compose_frame(
            frame=frame,
            detections=detections,
            tracks=tracks,
            risk_events=risk_events,
            fps=fps,
            bed_zones=bed_zones,
            privacy_mode_override=privacy_mode_override,
        )
        self._last_output_frame = canvas

        if not self.cfg.enabled or not self._display_available:
            return True

        display_canvas = canvas
        if self.cfg.window_scale != 1.0:
            h, w = display_canvas.shape[:2]
            display_canvas = cv2.resize(display_canvas, (int(w * self.cfg.window_scale), int(h * self.cfg.window_scale)))

        try:
            cv2.imshow(self.window_name, display_canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in self._quit_keycodes:
                return False
            visible = cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE)
            if visible < 1:
                return False
        except cv2.error:
            self._display_available = False

        return True

    def close(self) -> None:
        if not self.cfg.enabled:
            return
        try:
            cv2.destroyWindow(self.window_name)
        except cv2.error:
            pass
