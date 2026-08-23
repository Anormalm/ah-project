from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field


POSTURE_MAP = {"unknown": 0.0, "standing": 0.2, "sitting": 0.6, "lying": 1.0}


class SequenceExample(BaseModel):
    sequence: list[Any]
    label: float = Field(ge=0.0, le=1.0)
    weight: float = Field(default=1.0, gt=0.0)


def _feature_row(item: list[float] | dict[str, Any]) -> list[float]:
    if isinstance(item, list):
        if len(item) != 5:
            raise ValueError(f"Expected 5 features per timestep, got {len(item)}")
        return [float(v) for v in item]

    posture_value = item.get("posture", 0.0)
    if isinstance(posture_value, str):
        posture_value = POSTURE_MAP.get(posture_value, 0.0)
    return [
        float(item.get("speed", 0.0)),
        float(item.get("vy", 0.0)),
        float(item.get("acc", 0.0)),
        float(item.get("lean", item.get("lean_angle", 0.0))),
        float(posture_value),
    ]


def _skeleton_row(item: Any, num_joints: int = 17) -> np.ndarray:
    raw = item.get("normalized_keypoints", item.get("keypoints", [])) if isinstance(item, dict) else item
    array = np.asarray(raw, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 3:
        return np.zeros((num_joints, 3), dtype=np.float32)
    if array.shape[0] < num_joints:
        array = np.pad(array, ((0, num_joints - array.shape[0]), (0, 0)))
    array = array[:num_joints]
    array[:, 2] = np.clip(array[:, 2], 0.0, 1.0)
    return array.astype(np.float32)


def _pad_or_trim(seq: list[Any], sequence_len: int, feature_shape: tuple[int, ...]) -> np.ndarray:
    if not seq:
        return np.zeros((sequence_len, *feature_shape), dtype=np.float32)
    trimmed = seq[-sequence_len:]
    if len(trimmed) < sequence_len:
        pad = [trimmed[0]] * (sequence_len - len(trimmed))
        trimmed = pad + trimmed
    return np.array(trimmed, dtype=np.float32)


def load_sequence_dataset(
    path: str,
    sequence_len: int = 16,
    feature_mode: str = "biomechanics",
    num_joints: int = 17,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows_x: list[np.ndarray] = []
    rows_y: list[float] = []
    rows_w: list[float] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = SequenceExample.model_validate(json.loads(line))
            if feature_mode == "skeleton":
                seq = [_skeleton_row(item, num_joints=num_joints) for item in record.sequence]
                rows_x.append(_pad_or_trim(seq, sequence_len, (num_joints, 3)))
            else:
                seq = [_feature_row(item) for item in record.sequence]
                rows_x.append(_pad_or_trim(seq, sequence_len, (5,)))
            rows_y.append(float(record.label))
            rows_w.append(float(record.weight))

    if not rows_x:
        raise ValueError(f"No training rows found in {path}")
    return np.stack(rows_x), np.array(rows_y, dtype=np.float32), np.array(rows_w, dtype=np.float32)


def load_frame_log_dataset(
    path: str,
    sequence_len: int = 16,
    min_positive_level: str = "HIGH",
    feature_mode: str = "biomechanics",
    num_joints: int = 17,
    weak_label_weight: float = 0.25,
    return_groups: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    level_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
    positive_rank = level_rank.get(min_positive_level.upper(), 2)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            subject_id = row.get("subject_id")
            if subject_id is not None:
                key = f"subject:{subject_id}"
            else:
                key = (
                    f"session:{row.get('session_id', 'legacy')}|"
                    f"stream:{row.get('stream_id', 'unknown')}|track:{int(row['track_id'])}"
                )
            grouped[key].append(row)

    sequences: list[np.ndarray] = []
    labels: list[float] = []
    weights: list[float] = []
    group_ids: list[str] = []

    for group_key, rows in grouped.items():
        rows.sort(key=lambda x: float(x.get("timestamp", 0.0)))
        window: deque[Any] = deque(maxlen=sequence_len)
        for row in rows:
            posture = row.get("posture", "unknown")
            if isinstance(posture, str):
                posture_scalar = POSTURE_MAP.get(posture, 0.0)
            else:
                posture_scalar = float(posture)
            if feature_mode == "skeleton":
                window.append(_skeleton_row(row, num_joints=num_joints))
            else:
                window.append(
                    [
                        float(row.get("speed", 0.0)),
                        float(row.get("vy", 0.0)),
                        float(row.get("acc", 0.0)),
                        float(row.get("lean", 0.0)),
                        float(posture_scalar),
                    ]
                )
            if len(window) < sequence_len:
                continue
            level = str(row.get("risk_level", "LOW")).upper()
            label = 1.0 if level_rank.get(level, 0) >= positive_rank else 0.0
            sequences.append(np.array(window, dtype=np.float32))
            labels.append(label)
            label_source = str(row.get("label_source", "unknown")).lower()
            is_curated = label_source in {"human", "curated", "clinical"}
            weights.append(1.0 if is_curated else float(weak_label_weight))
            group_ids.append(group_key)

    if not sequences:
        raise ValueError(f"No sequences could be built from {path}. Need at least {sequence_len} timesteps per track.")
    result = np.stack(sequences), np.array(labels, dtype=np.float32), np.array(weights, dtype=np.float32)
    if return_groups:
        return (*result, np.asarray(group_ids, dtype=str))
    return result


def split_dataset(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    n = x.shape[0]
    if n <= 1 or val_ratio <= 0:
        return (x, y, w), (x[:0], y[:0], w[:0])
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    split = max(1, min(n - 1, int(round(n * (1.0 - val_ratio)))))
    tr_idx = idx[:split]
    va_idx = idx[split:]
    return (x[tr_idx], y[tr_idx], w[tr_idx]), (x[va_idx], y[va_idx], w[va_idx])


def split_dataset_grouped(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    groups: np.ndarray,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if len(groups) != x.shape[0]:
        raise ValueError("groups length must match sample count")
    unique_groups = np.unique(groups)
    if unique_groups.size <= 1 or val_ratio <= 0.0:
        return (x, y, w), (x[:0], y[:0], w[:0])
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)
    validation_group_count = max(1, min(unique_groups.size - 1, int(round(unique_groups.size * val_ratio))))
    validation_groups = set(unique_groups[:validation_group_count].tolist())
    validation_mask = np.array([group in validation_groups for group in groups], dtype=bool)
    training_mask = ~validation_mask
    return (
        (x[training_mask], y[training_mask], w[training_mask]),
        (x[validation_mask], y[validation_mask], w[validation_mask]),
    )
