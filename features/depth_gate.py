"""Aligned depth support and working-distance checks for person detections."""
from __future__ import annotations

import numpy as np


def filter_by_depth(packet, detections, poses, *, min_m=0.3, max_m=6.0, min_valid_fraction=0.2):
    if not (0 < min_m < max_m and 0 < min_valid_fraction <= 1):
        raise ValueError("Invalid depth gate range or valid fraction")
    depth, scale = packet.depth_frame, packet.depth_scale_m
    if depth is None or scale is None or not np.isfinite(scale) or scale <= 0:
        raise RuntimeError("Depth enforcement requires aligned depth and a valid metres/unit scale")
    if depth.ndim != 2 or depth.shape != packet.frame.shape[:2]:
        raise RuntimeError("Depth must be aligned to the color image")
    accepted = []
    measurements = []
    height, width = depth.shape
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        # Central region reduces background contamination; this is a range gate,
        # not person segmentation or floor-relative height estimation.
        dx, dy = (x2 - x1) * 0.25, (y2 - y1) * 0.25
        left, right = np.clip([int(x1 + dx), int(x2 - dx)], 0, width)
        top, bottom = np.clip([int(y1 + dy), int(y2 - dy)], 0, height)
        roi = depth[top:bottom, left:right].astype(np.float32) * scale
        valid = roi[np.isfinite(roi) & (roi > 0)]
        fraction = valid.size / max(roi.size, 1)
        distance = float(np.median(valid)) if valid.size else None
        keep = fraction >= min_valid_fraction and distance is not None and min_m <= distance <= max_m
        measurements.append({"distance_m": distance, "valid_fraction": fraction, "accepted": bool(keep)})
        if keep:
            accepted.append(detection)
    boxes = {d.bbox for d in accepted}
    return accepted, [p for p in poses if p.bbox in boxes], measurements
