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
        keys = cfg.quit_keys if cfg.quit_keys else [cfg.quit_key, "esc"]
        self._quit_keycodes = set()
        for key in keys:
            if key.lower() == "esc":
                self._quit_keycodes.add(27)
            elif key:
                self._quit_keycodes.add(ord(key.lower()[0]))

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

    def render(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        tracks: list[TrackPose],
        risk_events: dict[int, RiskEvent],
        fps: float,
        bed_zones: list[tuple[float, float, float, float]],
    ) -> bool:
        if not self.cfg.enabled or not self._display_available:
            return True

        canvas = frame

        if self.cfg.show_bed_zones:
            for i, zone in enumerate(bed_zones):
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

        if self.cfg.window_scale != 1.0:
            h, w = canvas.shape[:2]
            canvas = cv2.resize(canvas, (int(w * self.cfg.window_scale), int(h * self.cfg.window_scale)))

        try:
            cv2.imshow(self.window_name, canvas)
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

