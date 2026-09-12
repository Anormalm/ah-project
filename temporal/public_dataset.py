from __future__ import annotations

import csv
import json
import math
import os
import re
import tempfile
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Literal

import cv2
import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from features.feature_extractor import FeatureExtractor
from pose.pose_estimator import PoseEstimator
from tracking.tracker import ByteTrackLikeTracker
from utils.schemas import Detection, FeatureVector, PoseResult


class LabelInterval(BaseModel):
    """A half-open, strongly labelled interval.

    Use either frame coordinates (``start_frame <= frame < end_frame``) or
    timestamps in seconds (``start_time <= time < end_time``), never both.
    """

    model_config = ConfigDict(extra="forbid")

    label: Literal[0, 1]
    start_frame: int | None = Field(default=None, ge=0)
    end_frame: int | None = Field(default=None, gt=0)
    start_time: float | None = Field(default=None, ge=0.0)
    end_time: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def validate_coordinates(self) -> "LabelInterval":
        has_frames = self.start_frame is not None or self.end_frame is not None
        has_times = self.start_time is not None or self.end_time is not None
        if has_frames == has_times:
            raise ValueError("label interval must use exactly one of frame or time coordinates")
        if has_frames:
            if self.start_frame is None or self.end_frame is None:
                raise ValueError("frame interval requires start_frame and end_frame")
            if self.end_frame <= self.start_frame:
                raise ValueError("end_frame must be greater than start_frame")
        else:
            if self.start_time is None or self.end_time is None:
                raise ValueError("time interval requires start_time and end_time")
            if not math.isfinite(self.start_time) or not math.isfinite(self.end_time):
                raise ValueError("time interval boundaries must be finite")
            if self.end_time <= self.start_time:
                raise ValueError("end_time must be greater than start_time")
        return self


class PublicClip(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clip_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    source_type: Literal["video", "frames"] = "video"
    frames_glob: str = "*"
    split: Literal["train", "val"]
    group_id: str | int
    subject_id: str | int | None = None
    fps: float | None = Field(default=None, gt=0.0)
    default_label: Literal[0, 1] | None = None
    intervals: list[LabelInterval] = Field(default_factory=list)
    bbox_source: str | None = None
    bbox_format: Literal["jsonl_yolo_normalized"] | None = None
    weight: float = Field(default=1.0, gt=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_bbox_source(self) -> "PublicClip":
        if (self.bbox_source is None) != (self.bbox_format is None):
            raise ValueError("bbox_source and bbox_format must be set together")
        return self


class PublicDatasetManifest(BaseModel):
    """Data-only description of public clips and strong annotations."""

    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    dataset: str = Field(min_length=1)
    root: str = "."
    sequence_len: int = Field(default=10, ge=2)
    window_stride: int = Field(default=1, ge=1)
    negative_guard_frames: int = Field(default=10, ge=0)
    default_fps: float | None = Field(default=None, gt=0.0)
    resize_width: int | None = Field(default=None, gt=0)
    resize_height: int | None = Field(default=None, gt=0)
    min_keypoint_conf: float = Field(default=0.2, ge=0.0, le=1.0)
    min_pose_valid_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    min_pose_confidence: float = Field(default=0.2, ge=0.0, le=1.0)
    pose_input_size: int = Field(default=224, gt=0)
    bbox_crop_padding: float = Field(default=0.20, ge=0.0, le=1.0)
    person_selection: Literal["largest", "all"] = "largest"
    kinematic_ema_alpha: float = Field(default=0.30, ge=0.05, le=1.0)
    max_kinematic_gap_sec: float = Field(default=0.4, gt=0.0)
    clips: list[PublicClip] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_splits_and_groups(self) -> "PublicDatasetManifest":
        if (self.resize_width is None) != (self.resize_height is None):
            raise ValueError("resize_width and resize_height must be set together")
        clip_ids: set[str] = set()
        group_splits: dict[str, str] = {}
        seen_splits: set[str] = set()
        for clip in self.clips:
            if clip.clip_id in clip_ids:
                raise ValueError(f"duplicate clip_id: {clip.clip_id}")
            clip_ids.add(clip.clip_id)
            seen_splits.add(clip.split)
            group_id = str(clip.group_id)
            prior_split = group_splits.setdefault(group_id, clip.split)
            if prior_split != clip.split:
                raise ValueError(
                    f"group_id {group_id!r} appears in both {prior_split!r} and {clip.split!r}; "
                    "split whole subjects/recordings together"
                )
        if seen_splits != {"train", "val"}:
            raise ValueError("manifest must contain at least one train clip and one val clip")
        return self


@dataclass(frozen=True)
class _FrameFeature:
    frame_index: int
    timestamp: float
    label: int
    label_source: str
    values: dict[str, float | str | int]


@dataclass(frozen=True)
class _ClipResult:
    frames_read: int
    sequences: int
    positives: int
    negatives: int
    labelled_positive_frames: int
    labelled_negative_frames: int
    usable_positive_frames: int
    usable_negative_frames: int


def load_public_manifest(path: str | Path) -> tuple[PublicDatasetManifest, Path]:
    manifest_path = Path(path).resolve()
    raw = manifest_path.read_text(encoding="utf-8")
    if manifest_path.suffix.lower() == ".json":
        payload = json.loads(raw)
    else:
        payload = yaml.safe_load(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest must contain an object: {manifest_path}")
    return PublicDatasetManifest.model_validate(payload), manifest_path.parent


def write_public_manifest(manifest: PublicDatasetManifest, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.model_dump(mode="json", exclude_none=True)
    if output.suffix.lower() == ".json":
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _natural_sort_key(path: Path) -> list[int | str]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def _resolve_source(root: Path, source: str) -> Path:
    source_path = Path(source)
    return source_path.resolve() if source_path.is_absolute() else (root / source_path).resolve()


def _open_frames(
    clip: PublicClip,
    source_path: Path,
    default_fps: float | None,
) -> tuple[float, Iterator[tuple[int, np.ndarray]]]:
    if clip.source_type == "frames":
        fps = clip.fps or default_fps
        if fps is None:
            raise ValueError(f"frames clip {clip.clip_id!r} requires fps or manifest default_fps")
        if not source_path.is_dir():
            raise FileNotFoundError(f"Frame directory does not exist: {source_path}")
        extensions = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
        paths = sorted(
            (item for item in source_path.glob(clip.frames_glob) if item.suffix.lower() in extensions),
            key=_natural_sort_key,
        )
        if not paths:
            raise ValueError(
                f"No image frames matched {clip.frames_glob!r} for clip {clip.clip_id!r} in {source_path}"
            )

        def iter_images() -> Iterator[tuple[int, np.ndarray]]:
            for frame_index, frame_path in enumerate(paths):
                frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError(f"Unable to decode image frame: {frame_path}")
                yield frame_index, frame

        return float(fps), iter_images()

    if not source_path.is_file():
        raise FileNotFoundError(f"Video does not exist: {source_path}")
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to open video: {source_path}")
    detected_fps = float(capture.get(cv2.CAP_PROP_FPS))
    fps = clip.fps or (detected_fps if math.isfinite(detected_fps) and detected_fps > 0.0 else default_fps)
    if fps is None:
        capture.release()
        raise ValueError(f"Could not determine FPS for clip {clip.clip_id!r}; set clip.fps")

    def iter_video() -> Iterator[tuple[int, np.ndarray]]:
        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                yield frame_index, frame
                frame_index += 1
        finally:
            capture.release()

    return float(fps), iter_video()


def _interval_contains(interval: LabelInterval, frame_index: int, timestamp: float) -> bool:
    if interval.start_frame is not None and interval.end_frame is not None:
        return interval.start_frame <= frame_index < interval.end_frame
    assert interval.start_time is not None and interval.end_time is not None
    return interval.start_time <= timestamp < interval.end_time


def _strong_label(clip: PublicClip, frame_index: int, timestamp: float) -> tuple[int | None, str | None]:
    matches = [item.label for item in clip.intervals if _interval_contains(item, frame_index, timestamp)]
    if matches:
        labels = set(matches)
        if len(labels) != 1:
            raise ValueError(
                f"Conflicting strong labels in clip {clip.clip_id!r} at frame {frame_index} ({timestamp:.6f}s)"
            )
        return matches[0], "interval"
    if clip.default_label is not None:
        return clip.default_label, "clip_default"
    return None, None


def _load_frame_bboxes(clip: PublicClip, root: Path) -> dict[int, list[tuple[float, float, float, float]]]:
    """Load optional per-frame boxes without embedding thousands of rows in the manifest."""

    if clip.bbox_source is None:
        return {}
    path = _resolve_source(root, clip.bbox_source)
    if not path.is_file():
        raise FileNotFoundError(f"Bounding-box annotations do not exist: {path}")
    if clip.bbox_format != "jsonl_yolo_normalized":
        raise ValueError(f"Unsupported bbox_format for {clip.clip_id!r}: {clip.bbox_format}")

    by_frame: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON bounding-box annotation at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Bounding-box annotation must be an object at {path}:{line_number}")
            frame_index = _annotation_frame_index(row, line_number)
            objects = row.get("objects")
            raw_boxes = []
            if isinstance(objects, list):
                raw_boxes.extend(item.get("bbox_yolo") for item in objects if isinstance(item, dict))
            if not raw_boxes and row.get("bbox_yolo") is not None:
                raw_boxes.append(row["bbox_yolo"])
            for raw_box in raw_boxes:
                if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
                    raise ValueError(f"bbox_yolo must contain four values at {path}:{line_number}")
                box = tuple(float(value) for value in raw_box)
                if not all(math.isfinite(value) for value in box):
                    raise ValueError(f"bbox_yolo must be finite at {path}:{line_number}")
                cx, cy, width, height = box
                if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < width <= 1.0 and 0.0 < height <= 1.0):
                    raise ValueError(f"bbox_yolo is outside normalized bounds at {path}:{line_number}")
                by_frame[frame_index].append(box)
    return dict(by_frame)


def _bbox_guided_poses(
    pose_estimator: PoseEstimator,
    frame: np.ndarray,
    boxes: list[tuple[float, float, float, float]],
    padding: float,
) -> list[PoseResult]:
    """Run pose inference on annotation-guided crops and map results back to the frame."""

    frame_height, frame_width = frame.shape[:2]
    poses: list[PoseResult] = []
    for cx, cy, width, height in boxes:
        half_width = width * frame_width * (0.5 + padding)
        half_height = height * frame_height * (0.5 + padding)
        center_x = cx * frame_width
        center_y = cy * frame_height
        x1 = max(0, int(math.floor(center_x - half_width)))
        y1 = max(0, int(math.floor(center_y - half_height)))
        x2 = min(frame_width, int(math.ceil(center_x + half_width)))
        y2 = min(frame_height, int(math.ceil(center_y + half_height)))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        crop_poses = pose_estimator.predict_full_frame(frame[y1:y2, x1:x2])
        if not crop_poses:
            continue
        pose = max(
            crop_poses,
            key=lambda item: max(item.bbox[2] - item.bbox[0], 0.0)
            * max(item.bbox[3] - item.bbox[1], 0.0),
        )
        poses.append(
            PoseResult(
                bbox=(
                    pose.bbox[0] + x1,
                    pose.bbox[1] + y1,
                    pose.bbox[2] + x1,
                    pose.bbox[3] + y1,
                ),
                keypoints=[(x + x1, y + y1, confidence) for x, y, confidence in pose.keypoints],
            )
        )
    return poses


def _negative_is_guarded(
    clip: PublicClip,
    frame_index: int,
    fps: float,
    guard_frames: int,
) -> bool:
    for future_index in range(frame_index + 1, frame_index + guard_frames + 1):
        future_label, _ = _strong_label(clip, future_index, future_index / fps)
        if future_label == 1:
            return True
    return False


def _feature_values(feature: FeatureVector, frame_index: int) -> dict[str, float | str | int]:
    return {
        "frame_index": frame_index,
        "timestamp": float(feature.timestamp),
        "speed": float(np.hypot(feature.velocity[0], feature.velocity[1])),
        "vy": float(feature.velocity[1]),
        "acc": float(np.hypot(feature.acceleration[0], feature.acceleration[1])),
        "lean": float(feature.lean_angle),
        "posture": feature.posture,
    }


def _select_poses(poses: list[Any], selection: str) -> list[Any]:
    if selection == "all" or len(poses) <= 1:
        return poses
    return [
        max(
            poses,
            key=lambda pose: max(pose.bbox[2] - pose.bbox[0], 0.0)
            * max(pose.bbox[3] - pose.bbox[1], 0.0),
        )
    ]


def _pose_detections(poses: list[Any], threshold: float) -> tuple[list[Detection], list[Any]]:
    detections: list[Detection] = []
    accepted_poses: list[Any] = []
    for pose in poses:
        confidence = float(np.mean([point[2] for point in pose.keypoints])) if pose.keypoints else 0.0
        if confidence < threshold:
            continue
        detections.append(Detection(bbox=pose.bbox, confidence=confidence, class_name="person"))
        accepted_poses.append(pose)
    return detections, accepted_poses


def _sequence_record(
    manifest: PublicDatasetManifest,
    clip: PublicClip,
    source_path: Path,
    fps: float,
    track_id: int,
    window: deque[_FrameFeature],
    sequence_label: int,
) -> dict[str, Any]:
    first = window[0]
    last = window[-1]
    sources = sorted({item.label_source for item in window})
    label_source = sources[0] if len(sources) == 1 else "mixed_strong"
    return {
        "sequence": [item.values for item in window],
        "label": sequence_label,
        "weight": float(clip.weight),
        "provenance": {
            "dataset": manifest.dataset,
            "clip_id": clip.clip_id,
            "group_id": clip.group_id,
            "subject_id": clip.subject_id,
            "source": str(source_path),
            "split": clip.split,
            "track_id": track_id,
            "start_frame": first.frame_index,
            "end_frame": last.frame_index,
            "start_timestamp": first.timestamp,
            "end_timestamp": last.timestamp,
            "fps": fps,
            "label_source": label_source,
            "frame_labels": [item.label for item in window],
            "label_policy": "positive_endpoint_else_all_negative",
            "clip_metadata": clip.metadata,
        },
    }


def _process_clip(
    manifest: PublicDatasetManifest,
    clip: PublicClip,
    root: Path,
    pose_estimator: PoseEstimator,
    output,
) -> _ClipResult:
    source_path = _resolve_source(root, clip.source)
    fps, frames = _open_frames(clip, source_path, manifest.default_fps)
    frame_bboxes = _load_frame_bboxes(clip, root)
    tracker = ByteTrackLikeTracker(iou_threshold=0.15, max_misses=12, center_distance_threshold=0.45)
    extractor = FeatureExtractor(
        min_kpt_conf=manifest.min_keypoint_conf,
        kinematic_ema_alpha=manifest.kinematic_ema_alpha,
        max_kinematic_gap_sec=manifest.max_kinematic_gap_sec,
    )
    windows: dict[int, deque[_FrameFeature]] = defaultdict(
        lambda: deque(maxlen=manifest.sequence_len)
    )
    last_emitted_end: dict[int, int] = {}
    frames_read = sequences = positives = negatives = 0
    labelled_positive_frames = labelled_negative_frames = 0
    usable_positive_frames = usable_negative_frames = 0

    for frame_index, frame in frames:
        frames_read += 1
        # Dataset processing is deliberately independent of wall-clock and decode
        # speed. This makes kinematics reproducible across machines and reruns.
        timestamp = frame_index / fps
        label, label_source = _strong_label(clip, frame_index, timestamp)
        if label is None or label_source is None:
            for window in windows.values():
                window.clear()
            continue
        if label == 1:
            labelled_positive_frames += 1
        else:
            labelled_negative_frames += 1

        if manifest.resize_width is not None and manifest.resize_height is not None:
            frame = cv2.resize(
                frame,
                (manifest.resize_width, manifest.resize_height),
                interpolation=cv2.INTER_LINEAR,
            )

        annotation_boxes = frame_bboxes.get(frame_index, [])
        poses = _bbox_guided_poses(
            pose_estimator,
            frame,
            annotation_boxes,
            manifest.bbox_crop_padding,
        ) if annotation_boxes else []
        # Fall back to ordinary full-frame inference for generic manifests and
        # for the occasional annotation crop that still yields no pose.
        if not poses:
            poses = pose_estimator.predict_full_frame(frame)
        poses = _select_poses(poses, manifest.person_selection)
        detections, poses = _pose_detections(poses, manifest.min_pose_confidence)
        tracks = tracker.update(detections, poses, timestamp=timestamp)
        frame_is_usable = False
        for track in tracks:
            feature = extractor.extract(track)
            window = windows[track.track_id]
            if feature.pose_valid_ratio < manifest.min_pose_valid_ratio:
                window.clear()
                continue
            frame_is_usable = True
            if window and frame_index != window[-1].frame_index + 1:
                window.clear()
            window.append(
                _FrameFeature(
                    frame_index=frame_index,
                    timestamp=timestamp,
                    label=label,
                    label_source=label_source,
                    values=_feature_values(feature, frame_index),
                )
            )
            if len(window) < manifest.sequence_len:
                continue
            labels = [item.label for item in window]
            if label == 1:
                sequence_label = 1
            elif all(item == 0 for item in labels):
                if _negative_is_guarded(
                    clip,
                    frame_index=frame_index,
                    fps=fps,
                    guard_frames=manifest.negative_guard_frames,
                ):
                    continue
                sequence_label = 0
            else:
                # A post-fall/recovery window ending in no-fall is ambiguous for
                # this endpoint classifier until the full window is clean again.
                continue
            prior_end = last_emitted_end.get(track.track_id)
            if prior_end is not None and frame_index - prior_end < manifest.window_stride:
                continue
            output.write(
                json.dumps(
                    _sequence_record(
                        manifest=manifest,
                        clip=clip,
                        source_path=source_path,
                        fps=fps,
                        track_id=track.track_id,
                        window=window,
                        sequence_label=sequence_label,
                    ),
                    separators=(",", ":"),
                )
                + "\n"
            )
            last_emitted_end[track.track_id] = frame_index
            sequences += 1
            if sequence_label == 1:
                positives += 1
            else:
                negatives += 1

        if frame_is_usable:
            if label == 1:
                usable_positive_frames += 1
            else:
                usable_negative_frames += 1

    return _ClipResult(
        frames_read=frames_read,
        sequences=sequences,
        positives=positives,
        negatives=negatives,
        labelled_positive_frames=labelled_positive_frames,
        labelled_negative_frames=labelled_negative_frames,
        usable_positive_frames=usable_positive_frames,
        usable_negative_frames=usable_negative_frames,
    )


def _temporary_output(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    return handle, Path(handle.name)


def prepare_public_dataset(
    manifest: PublicDatasetManifest,
    manifest_dir: str | Path,
    pose_estimator: PoseEstimator,
    train_output: str | Path,
    val_output: str | Path,
    *,
    overwrite: bool = False,
    progress: Callable[[PublicClip, dict[str, int]], None] | None = None,
) -> dict[str, Any]:
    """Convert annotated public clips to trainer-compatible sequence JSONL."""

    train_path = Path(train_output).resolve()
    val_path = Path(val_output).resolve()
    if train_path == val_path:
        raise ValueError("train_output and val_output must be different files")
    for path in (train_path, val_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output already exists (pass overwrite=True): {path}")

    manifest_root = Path(manifest.root)
    root = (
        manifest_root.resolve()
        if manifest_root.is_absolute()
        else (Path(manifest_dir).resolve() / manifest_root).resolve()
    )
    train_handle, train_temp = _temporary_output(train_path)
    val_handle, val_temp = _temporary_output(val_path)
    handles = {"train": train_handle, "val": val_handle}
    totals: dict[str, dict[str, int]] = {
        "train": {
            "clips": 0,
            "frames": 0,
            "sequences": 0,
            "positive": 0,
            "negative": 0,
            "labelled_positive_frames": 0,
            "labelled_negative_frames": 0,
            "usable_positive_frames": 0,
            "usable_negative_frames": 0,
        },
        "val": {
            "clips": 0,
            "frames": 0,
            "sequences": 0,
            "positive": 0,
            "negative": 0,
            "labelled_positive_frames": 0,
            "labelled_negative_frames": 0,
            "usable_positive_frames": 0,
            "usable_negative_frames": 0,
        },
    }
    clip_summaries: list[dict[str, Any]] = []
    try:
        for clip in manifest.clips:
            result = _process_clip(manifest, clip, root, pose_estimator, handles[clip.split])
            split_total = totals[clip.split]
            split_total["clips"] += 1
            split_total["frames"] += result.frames_read
            split_total["sequences"] += result.sequences
            split_total["positive"] += result.positives
            split_total["negative"] += result.negatives
            split_total["labelled_positive_frames"] += result.labelled_positive_frames
            split_total["labelled_negative_frames"] += result.labelled_negative_frames
            split_total["usable_positive_frames"] += result.usable_positive_frames
            split_total["usable_negative_frames"] += result.usable_negative_frames
            clip_summary = {
                "clip_id": clip.clip_id,
                "split": clip.split,
                "group_id": clip.group_id,
                "frames": result.frames_read,
                "sequences": result.sequences,
                "positive": result.positives,
                "negative": result.negatives,
                "labelled_positive_frames": result.labelled_positive_frames,
                "labelled_negative_frames": result.labelled_negative_frames,
                "usable_positive_frames": result.usable_positive_frames,
                "usable_negative_frames": result.usable_negative_frames,
                "positive_frame_pose_coverage": (
                    result.usable_positive_frames / result.labelled_positive_frames
                    if result.labelled_positive_frames
                    else None
                ),
                "negative_frame_pose_coverage": (
                    result.usable_negative_frames / result.labelled_negative_frames
                    if result.labelled_negative_frames
                    else None
                ),
            }
            clip_summaries.append(clip_summary)
            if progress is not None:
                progress(clip, clip_summary)
        empty_splits = [split for split, total in totals.items() if total["sequences"] == 0]
        if empty_splits:
            raise ValueError(
                "No valid pose sequences were produced for split(s): " + ", ".join(empty_splits)
            )
        train_handle.close()
        val_handle.close()
        os.replace(train_temp, train_path)
        os.replace(val_temp, val_path)
    except BaseException:
        train_handle.close()
        val_handle.close()
        train_temp.unlink(missing_ok=True)
        val_temp.unlink(missing_ok=True)
        raise

    zero_positive_fall_clips = [
        item["clip_id"]
        for item in clip_summaries
        if item["labelled_positive_frames"] > 0 and item["positive"] == 0
    ]
    coverage: dict[str, dict[str, float | int]] = {}
    for split, total in totals.items():
        split_fall_clips = [
            item for item in clip_summaries
            if item["split"] == split and item["labelled_positive_frames"] > 0
        ]
        recovered_fall_clips = sum(item["positive"] > 0 for item in split_fall_clips)
        coverage[split] = {
            "fall_clips": len(split_fall_clips),
            "fall_clips_with_positive_sequences": recovered_fall_clips,
            "fall_clip_sequence_coverage": (
                recovered_fall_clips / len(split_fall_clips) if split_fall_clips else 1.0
            ),
            "positive_frame_pose_coverage": (
                total["usable_positive_frames"] / total["labelled_positive_frames"]
                if total["labelled_positive_frames"] else 1.0
            ),
            "negative_frame_pose_coverage": (
                total["usable_negative_frames"] / total["labelled_negative_frames"]
                if total["labelled_negative_frames"] else 1.0
            ),
        }
    quality_warnings = []
    if zero_positive_fall_clips:
        quality_warnings.append(
            f"{len(zero_positive_fall_clips)} labelled fall clips emitted no positive sequences; "
            "window-level recall excludes these clips"
        )

    return {
        "dataset": manifest.dataset,
        "sequence_len": manifest.sequence_len,
        "window_stride": manifest.window_stride,
        "train_output": str(train_path),
        "val_output": str(val_path),
        "splits": totals,
        "coverage": coverage,
        "zero_positive_fall_clips": zero_positive_fall_clips,
        "quality_warnings": quality_warnings,
        "clips": clip_summaries,
    }


def _caucafall_subject(path: Path, root: Path) -> int | None:
    for part in reversed(path.relative_to(root).parts):
        match = re.fullmatch(r"subject[._ -]*(\d+)", part, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _caucafall_frame_label(path: Path) -> int | None:
    labels: set[int] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        columns = line.split()
        if not columns:
            continue
        try:
            label = int(columns[0])
        except ValueError as exc:
            raise ValueError(f"Invalid YOLO class in {path}:{line_number}") from exc
        if label not in {0, 1}:
            raise ValueError(f"Unexpected CAUCAFall class {label} in {path}:{line_number}")
        labels.add(label)
    if not labels:
        return None
    if len(labels) > 1:
        # Conflicting objects make the frame unsuitable as sequence ground truth.
        return None
    return labels.pop()


def _frame_label_intervals(labels: list[int | None]) -> list[LabelInterval]:
    intervals: list[LabelInterval] = []
    run_label: int | None = None
    run_start = 0
    for index, label in enumerate([*labels, None]):
        if label == run_label:
            continue
        if run_label is not None:
            intervals.append(
                LabelInterval(label=run_label, start_frame=run_start, end_frame=index)
            )
        run_label = label
        run_start = index
    return intervals


def _annotation_frame_index(row: dict[str, Any], line_number: int) -> int:
    for key in ("frame_index", "frame", "index"):
        if row.get(key) is not None:
            return int(row[key])
    for key in ("annotation_filename", "filename", "name"):
        value = row.get(key)
        if value:
            match = re.search(r"(\d+)(?=\.[^.]+$|$)", str(value))
            if match:
                return int(match.group(1))
    # File order remains deterministic, but acquisition should normally retain
    # either the numeric suffix or an explicit source frame index.
    return line_number - 1


def _caucafall_jsonl_labels(path: Path, expected_frames: int) -> tuple[list[int | None], int]:
    by_frame: dict[int, set[int]] = defaultdict(set)
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON annotation at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Annotation must be an object at {path}:{line_number}")
            frame_index = _annotation_frame_index(row, line_number)
            raw_label = row.get("class_id", row.get("label", row.get("class")))
            if raw_label is None:
                by_frame.setdefault(frame_index, set())
                continue
            label = int(raw_label)
            if label not in {0, 1}:
                raise ValueError(f"Unexpected CAUCAFall class {label} at {path}:{line_number}")
            by_frame[frame_index].add(label)
    if not by_frame:
        raise ValueError(f"No frame annotations found in {path}")
    negative_indices = sorted(index for index in by_frame if index < 0)
    if negative_indices:
        raise ValueError(
            f"Negative CAUCAFall annotation indices in {path}: {negative_indices[:10]}"
        )
    extra_indices = {index for index in by_frame if index >= expected_frames}
    # Missing source label numbers are explicit None entries. The converter then
    # clears all track windows at those frames, so no sequence bridges the gap.
    labels: list[int | None] = [None] * expected_frames
    for index in sorted(by_frame):
        if index >= expected_frames:
            continue
        frame_labels = by_frame[index]
        labels[index] = next(iter(frame_labels)) if len(frame_labels) == 1 else None
    return labels, len(extra_indices)


def _caucafall_raw_labels(paths: list[Path], expected_frames: int) -> tuple[list[int | None], int]:
    labels: list[int | None] = [None] * expected_frames
    extra_frames: set[int] = set()
    numbered: list[tuple[int, Path]] = []
    for path in paths:
        match = re.search(r"(\d{6})(?=\.[^.]+$)", path.name)
        if not match:
            raise ValueError(
                f"Cannot align raw CAUCAFall label to a video frame; expected six-digit suffix: {path}"
            )
        numbered.append((int(match.group(1)), path))
    for number, path in numbered:
        # Official names (for example cas900001.txt) combine subject/activity
        # digits with a five-digit, one-based source-frame suffix.
        frame_index = (number % 100000) - 1
        if frame_index < 0:
            raise ValueError(f"Raw CAUCAFall label has a non-positive source frame: {path}")
        if frame_index >= expected_frames:
            extra_frames.add(frame_index)
            continue
        label = _caucafall_frame_label(path)
        if labels[frame_index] is not None and label != labels[frame_index]:
            labels[frame_index] = None
        else:
            labels[frame_index] = label
    return labels, len(extra_frames)


def caucafall_manifest_from_yolo_tree(
    dataset_root: str | Path,
    train_subjects: set[int],
    val_subjects: set[int],
    *,
    fps: float | None = None,
    negative_guard_frames: int = 10,
) -> PublicDatasetManifest:
    """Scan the official ``Subject.N/activity`` AVI + YOLO-label layout."""

    overlap = train_subjects & val_subjects
    if overlap:
        raise ValueError(f"Subjects cannot be in both train and val: {sorted(overlap)}")
    root = Path(dataset_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"CAUCAFall dataset root does not exist: {root}")
    if fps is not None and (not math.isfinite(fps) or fps <= 0.0):
        raise ValueError("FPS override must be finite and greater than zero")

    videos = sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".avi"),
        key=lambda path: str(path.relative_to(root)).lower(),
    )
    clips: list[PublicClip] = []
    unassigned: set[int] = set()
    for video in videos:
        subject = _caucafall_subject(video, root)
        if subject is None:
            raise ValueError(f"Could not infer Subject.N for video: {video}")
        if subject in train_subjects:
            split = "train"
        elif subject in val_subjects:
            split = "val"
        else:
            unassigned.add(subject)
            continue

        activity_dir = video.parent
        sibling_videos = [
            path for path in activity_dir.iterdir() if path.is_file() and path.suffix.lower() == ".avi"
        ]
        if len(sibling_videos) != 1:
            raise ValueError(f"Expected one AVI in CAUCAFall activity folder: {activity_dir}")
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Unable to open CAUCAFall video: {video}")
        video_frames = int(round(float(capture.get(cv2.CAP_PROP_FRAME_COUNT))))
        detected_fps = float(capture.get(cv2.CAP_PROP_FPS))
        capture.release()
        if video_frames <= 0:
            raise ValueError(f"Could not read frame count from CAUCAFall AVI: {video}")
        source_fps = fps or detected_fps
        if not math.isfinite(source_fps) or source_fps <= 0.0:
            raise ValueError(f"Could not read FPS from CAUCAFall AVI: {video}")

        consolidated_annotations = activity_dir / "annotations.jsonl"
        bbox_source = None
        bbox_format = None
        if consolidated_annotations.is_file():
            frame_labels, extra_annotation_frames = _caucafall_jsonl_labels(
                consolidated_annotations, video_frames
            )
            annotation_format = "caucafall_yolo_consolidated_jsonl"
            bbox_source = consolidated_annotations.relative_to(root).as_posix()
            bbox_format = "jsonl_yolo_normalized"
        else:
            label_paths = sorted(
                (
                    path
                    for path in activity_dir.iterdir()
                    if path.is_file()
                    and path.suffix.lower() == ".txt"
                    and path.name.lower() != "classes.txt"
                ),
                key=_natural_sort_key,
            )
            if not label_paths:
                raise ValueError(f"No annotations.jsonl or per-frame YOLO labels found beside {video}")
            frame_labels, extra_annotation_frames = _caucafall_raw_labels(
                label_paths, video_frames
            )
            annotation_format = "caucafall_yolo_per_frame"

        relative_video = video.relative_to(root).as_posix()
        clips.append(
            PublicClip(
                clip_id=relative_video[: -len(video.suffix)],
                source=relative_video,
                source_type="video",
                split=split,
                group_id=subject,
                subject_id=subject,
                fps=source_fps,
                intervals=_frame_label_intervals(frame_labels),
                bbox_source=bbox_source,
                bbox_format=bbox_format,
                metadata={
                    "annotation_format": annotation_format,
                    "activity": activity_dir.name,
                    "annotation_frames": len(frame_labels),
                    "unlabelled_frames": sum(label is None for label in frame_labels),
                    "extra_annotation_frames": extra_annotation_frames,
                    "source_fps": source_fps,
                },
            )
        )

    if unassigned:
        raise ValueError(f"CAUCAFall subjects are not assigned to train or val: {sorted(unassigned)}")
    if not clips:
        raise ValueError(f"No CAUCAFall AVI clips found under {root}")
    return PublicDatasetManifest(
        dataset="caucafall",
        root=str(root),
        sequence_len=10,
        window_stride=1,
        negative_guard_frames=negative_guard_frames,
        default_fps=fps,
        resize_width=640,
        resize_height=480,
        min_keypoint_conf=0.2,
        min_pose_valid_ratio=0.5,
        min_pose_confidence=0.2,
        pose_input_size=320,
        person_selection="largest",
        kinematic_ema_alpha=0.30,
        max_kinematic_gap_sec=0.4,
        clips=clips,
    )


def caucafall_manifest_from_omnifall(
    labels_csv: str | Path,
    video_root: str | Path,
    train_subjects: set[int],
    val_subjects: set[int],
    *,
    positive_labels: set[int] | None = None,
    video_extension: str = ".mp4",
) -> PublicDatasetManifest:
    """Build a CAUCAFall manifest from OmniFall's temporal annotation CSV.

    OmniFall uses class 1 for the fall transition and class 2 for the fallen
    state. Both are positive by default; other annotated states are negatives.
    Unannotated gaps remain excluded rather than receiving pseudo-labels.
    """

    overlap = train_subjects & val_subjects
    if overlap:
        raise ValueError(f"Subjects cannot be in both train and val: {sorted(overlap)}")
    positive = positive_labels or {1, 2}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with Path(labels_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"path", "label", "start", "end", "subject"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"OmniFall label CSV is missing columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            if str(row.get("dataset", "caucafall")).strip().lower() != "caucafall":
                continue
            try:
                path = str(row["path"]).strip().replace("\\", "/")
                subject = int(row["subject"])
                class_label = int(row["label"])
                start = float(row["start"])
                end = float(row["end"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid CAUCAFall annotation at row {row_number}: {exc}") from exc
            if not path:
                raise ValueError(f"Empty video path at row {row_number}")
            grouped[path].append(
                {
                    "subject": subject,
                    "interval": LabelInterval(
                        label=1 if class_label in positive else 0,
                        start_time=start,
                        end_time=end,
                    ),
                }
            )

    clips: list[PublicClip] = []
    unassigned: set[int] = set()
    extension = video_extension if video_extension.startswith(".") else f".{video_extension}"
    for path in sorted(grouped):
        rows = grouped[path]
        subjects = {int(row["subject"]) for row in rows}
        if len(subjects) != 1:
            raise ValueError(f"Video {path!r} has inconsistent subjects: {sorted(subjects)}")
        subject = subjects.pop()
        if subject in train_subjects:
            split = "train"
        elif subject in val_subjects:
            split = "val"
        else:
            unassigned.add(subject)
            continue
        clips.append(
            PublicClip(
                clip_id=path,
                source=f"{path}{extension}",
                source_type="video",
                split=split,
                group_id=subject,
                subject_id=subject,
                intervals=[row["interval"] for row in rows],
                metadata={"annotation_format": "omnifall"},
            )
        )
    if unassigned:
        raise ValueError(
            f"CAUCAFall subjects are not assigned to train or val: {sorted(unassigned)}"
        )
    if not clips:
        raise ValueError("No CAUCAFall rows found in the annotation CSV")
    return PublicDatasetManifest(
        dataset="caucafall",
        root=str(video_root),
        sequence_len=10,
        resize_width=640,
        resize_height=480,
        default_fps=23.0,
        pose_input_size=320,
        clips=clips,
    )
