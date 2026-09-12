from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from pose.pose_estimator import MockPoseEngine, PoseEstimator
from temporal.public_dataset import (
    LabelInterval,
    PublicClip,
    PublicDatasetManifest,
    caucafall_manifest_from_omnifall,
    caucafall_manifest_from_yolo_tree,
    prepare_public_dataset,
)
from temporal.training_data import load_sequence_dataset


def _write_frames(directory: Path, count: int) -> None:
    directory.mkdir(parents=True)
    for index in range(count):
        frame = np.zeros((80, 120, 3), dtype=np.uint8)
        cv2.rectangle(frame, (35 + index, 10), (75 + index, 72), (255, 255, 255), -1)
        assert cv2.imwrite(str(directory / f"frame_{index:04d}.png"), frame)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_avi(path: Path, frame_count: int) -> None:
    path.parent.mkdir(parents=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        23.0,
        (120, 80),
    )
    assert writer.isOpened()
    for index in range(frame_count):
        frame = np.full((80, 120, 3), index, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_preparer_uses_deterministic_timestamps_and_strong_windows(tmp_path: Path) -> None:
    _write_frames(tmp_path / "train_frames", 14)
    _write_frames(tmp_path / "val_frames", 20)
    manifest = PublicDatasetManifest(
        dataset="fixture",
        root=str(tmp_path),
        sequence_len=10,
        default_fps=20.0,
        clips=[
            PublicClip(
                clip_id="subject_1_normal",
                source="train_frames",
                source_type="frames",
                frames_glob="*.png",
                split="train",
                group_id="subject_1",
                subject_id=1,
                default_label=0,
            ),
            PublicClip(
                clip_id="subject_2_transition",
                source="val_frames",
                source_type="frames",
                frames_glob="*.png",
                split="val",
                group_id="subject_2",
                subject_id=2,
                intervals=[
                    LabelInterval(label=0, start_frame=0, end_frame=10),
                    LabelInterval(label=1, start_frame=10, end_frame=20),
                ],
            ),
        ],
    )
    train_path = tmp_path / "prepared" / "train.jsonl"
    val_path = tmp_path / "prepared" / "val.jsonl"
    estimator = PoseEstimator(MockPoseEngine())

    summary = prepare_public_dataset(
        manifest,
        tmp_path,
        estimator,
        train_path,
        val_path,
    )

    train_rows = _read_jsonl(train_path)
    val_rows = _read_jsonl(val_path)
    assert len(train_rows) == 5
    # A positive endpoint may include the preceding strongly-labelled motion;
    # negatives immediately before the fall are guarded out.
    assert [row["label"] for row in val_rows] == [1] * 10
    assert val_rows[0]["provenance"]["start_frame"] == 1
    assert val_rows[0]["provenance"]["end_frame"] == 10
    assert val_rows[0]["provenance"]["end_timestamp"] == pytest.approx(10 / 20.0)
    assert val_rows[-1]["provenance"]["start_frame"] == 10
    assert val_rows[-1]["provenance"]["end_frame"] == 19
    assert val_rows[-1]["provenance"]["label_source"] == "interval"
    assert summary["splits"]["val"]["positive"] == 10
    assert summary["splits"]["val"]["negative"] == 0
    assert summary["coverage"]["val"]["positive_frame_pose_coverage"] == 1.0
    assert summary["zero_positive_fall_clips"] == []
    assert summary["quality_warnings"] == []

    # The existing sequence loader ignores provenance but consumes the exact
    # five-feature payload without an adapter in the trainer.
    x_train, y_train, _ = load_sequence_dataset(str(train_path), sequence_len=10)
    assert x_train.shape == (5, 10, 5)
    assert y_train.tolist() == [0.0] * 5

    original = train_path.read_bytes(), val_path.read_bytes()
    prepare_public_dataset(
        manifest,
        tmp_path,
        estimator,
        train_path,
        val_path,
        overwrite=True,
    )
    assert original == (train_path.read_bytes(), val_path.read_bytes())


def test_manifest_rejects_group_leakage() -> None:
    with pytest.raises(ValueError, match="appears in both"):
        PublicDatasetManifest(
            dataset="bad_split",
            clips=[
                PublicClip(
                    clip_id="a",
                    source="a.mp4",
                    split="train",
                    group_id="same_subject",
                    default_label=0,
                ),
                PublicClip(
                    clip_id="b",
                    source="b.mp4",
                    split="val",
                    group_id="same_subject",
                    default_label=1,
                ),
            ],
        )


def test_caucafall_omnifall_labels_become_binary_subject_splits(tmp_path: Path) -> None:
    labels = tmp_path / "caucafall.csv"
    with labels.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "label", "start", "end", "subject", "cam", "dataset"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "path": "adl/WalkS1",
                    "label": 0,
                    "start": 0.0,
                    "end": 4.0,
                    "subject": 1,
                    "cam": 1,
                    "dataset": "caucafall",
                },
                {
                    "path": "side/FallLeftS9",
                    "label": 8,
                    "start": 0.0,
                    "end": 1.0,
                    "subject": 9,
                    "cam": 1,
                    "dataset": "caucafall",
                },
                {
                    "path": "side/FallLeftS9",
                    "label": 1,
                    "start": 1.0,
                    "end": 2.5,
                    "subject": 9,
                    "cam": 1,
                    "dataset": "caucafall",
                },
                {
                    "path": "side/FallLeftS9",
                    "label": 2,
                    "start": 2.5,
                    "end": 5.0,
                    "subject": 9,
                    "cam": 1,
                    "dataset": "caucafall",
                },
            ]
        )

    manifest = caucafall_manifest_from_omnifall(
        labels_csv=labels,
        video_root=tmp_path / "video",
        train_subjects={1},
        val_subjects={9},
    )

    assert manifest.sequence_len == 10
    assert [(clip.subject_id, clip.split) for clip in manifest.clips] == [(1, "train"), (9, "val")]
    fall_clip = next(clip for clip in manifest.clips if clip.subject_id == 9)
    assert fall_clip.source == "side/FallLeftS9.mp4"
    assert [interval.label for interval in fall_clip.intervals] == [0, 1, 1]


def test_caucafall_official_tree_uses_jsonl_frame_labels_and_keeps_gaps(tmp_path: Path) -> None:
    for subject, activity in ((1, "Walk"), (9, "Fall backwards")):
        folder = tmp_path / f"Subject.{subject}" / activity
        _write_avi(folder / f"clipS{subject}.avi", frame_count=12)
        rows = []
        for frame_index in range(12):
            if subject == 9 and frame_index == 4:
                # A missing annotation remains an explicit gap rather than
                # shifting every later source label by one frame.
                continue
            class_id = 1 if subject == 9 and frame_index >= 6 else 0
            rows.append(
                {
                    "subject": subject,
                    "activity": activity,
                    "frame_index": frame_index,
                    "source_frame_number": frame_index + 1,
                    "class_id": class_id,
                    "class_name": "fall" if class_id else "nofall",
                    "bbox_yolo": [0.5, 0.5, 0.4, 0.8],
                    "annotation_filename": f"cas{subject}{frame_index + 1:05d}.txt",
                }
            )
        if subject == 9:
            rows.append(
                {
                    "subject": subject,
                    "activity": activity,
                    "frame_index": 12,
                    "source_frame_number": 13,
                    "class_id": 1,
                    "class_name": "fall",
                    "bbox_yolo": [0.5, 0.5, 0.4, 0.8],
                    "annotation_filename": "extra_source_frame.txt",
                }
            )
        (folder / "annotations.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )

    manifest = caucafall_manifest_from_yolo_tree(
        dataset_root=tmp_path,
        train_subjects={1},
        val_subjects={9},
    )

    assert manifest.pose_input_size == 320
    assert (manifest.resize_width, manifest.resize_height) == (640, 480)
    assert manifest.default_fps is None
    fall_clip = next(clip for clip in manifest.clips if clip.subject_id == 9)
    assert fall_clip.fps == pytest.approx(23.0)
    assert fall_clip.metadata["unlabelled_frames"] == 1
    assert fall_clip.metadata["extra_annotation_frames"] == 1
    assert fall_clip.bbox_source == "Subject.9/Fall backwards/annotations.jsonl"
    assert fall_clip.bbox_format == "jsonl_yolo_normalized"
    assert [
        (interval.label, interval.start_frame, interval.end_frame)
        for interval in fall_clip.intervals
    ] == [(0, 0, 4), (0, 5, 6), (1, 6, 12)]


def test_preparer_uses_annotation_boxes_as_pose_crops_and_reports_coverage(tmp_path: Path) -> None:
    class RecordingMockPoseEngine(MockPoseEngine):
        def __init__(self) -> None:
            self.shapes: list[tuple[int, int]] = []

        def predict_full(self, frame: np.ndarray):
            self.shapes.append(frame.shape[:2])
            return super().predict_full(frame)

    for split in ("train", "val"):
        _write_frames(tmp_path / f"{split}_frames", 10)
        annotations = [
            {"frame_index": index, "bbox_yolo": [0.5, 0.5, 0.2, 0.5]}
            for index in range(10)
        ]
        (tmp_path / f"{split}_boxes.jsonl").write_text(
            "\n".join(json.dumps(row) for row in annotations) + "\n",
            encoding="utf-8",
        )

    manifest = PublicDatasetManifest(
        dataset="boxed_fixture",
        root=str(tmp_path),
        sequence_len=10,
        default_fps=20.0,
        clips=[
            PublicClip(
                clip_id=f"{split}_fall",
                source=f"{split}_frames",
                source_type="frames",
                frames_glob="*.png",
                split=split,
                group_id=split,
                default_label=1,
                bbox_source=f"{split}_boxes.jsonl",
                bbox_format="jsonl_yolo_normalized",
            )
            for split in ("train", "val")
        ],
    )
    engine = RecordingMockPoseEngine()
    summary = prepare_public_dataset(
        manifest,
        tmp_path,
        PoseEstimator(engine),
        tmp_path / "train.jsonl",
        tmp_path / "val.jsonl",
    )

    assert len(engine.shapes) == 20
    assert all(height < 80 and width < 120 for height, width in engine.shapes)
    assert summary["coverage"]["train"]["fall_clip_sequence_coverage"] == 1.0
    assert summary["coverage"]["val"]["positive_frame_pose_coverage"] == 1.0
    assert summary["zero_positive_fall_clips"] == []
