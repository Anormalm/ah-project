from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field


POSTURE_MAP = {"unknown": 0.0, "standing": 0.2, "sitting": 0.6, "lying": 1.0}
DatasetArrays = tuple[np.ndarray, np.ndarray, np.ndarray]


class SequenceExample(BaseModel):
    sequence: list[list[float] | dict[str, Any]]
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


def _pad_or_trim(seq: list[list[float]], sequence_len: int) -> np.ndarray:
    if not seq:
        return np.zeros((sequence_len, 5), dtype=np.float32)
    trimmed = seq[-sequence_len:]
    if len(trimmed) < sequence_len:
        pad = [trimmed[0]] * (sequence_len - len(trimmed))
        trimmed = pad + trimmed
    return np.array(trimmed, dtype=np.float32)


def load_sequence_dataset(path: str, sequence_len: int = 16) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows_x: list[np.ndarray] = []
    rows_y: list[float] = []
    rows_w: list[float] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = SequenceExample.model_validate(json.loads(line))
            seq = [_feature_row(item) for item in record.sequence]
            rows_x.append(_pad_or_trim(seq, sequence_len))
            rows_y.append(float(record.label))
            rows_w.append(float(record.weight))

    if not rows_x:
        raise ValueError(f"No training rows found in {path}")
    return np.stack(rows_x), np.array(rows_y, dtype=np.float32), np.array(rows_w, dtype=np.float32)


@dataclass(frozen=True)
class _WindowCandidate:
    features: np.ndarray
    weak_label: float
    session_id: str
    stream_id: str
    track_id: int
    end_timestamp: float


@dataclass(frozen=True)
class _FeedbackAnnotation:
    session_id: str | None
    stream_id: str | None
    track_id: int
    timestamp: float
    label: float
    weight: float
    line_number: int


def _parse_feedback_label(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in {"", "unclear", "unknown", "unsure", "unreviewed", "skip"}:
            return None
        if normalized in {"positive", "fall", "fall_detected", "confirmed_fall", "yes", "true"}:
            return 1.0
        if normalized in {
            "negative",
            "no_fall",
            "false_alarm",
            "non_fall_activity",
            "stable",
            "no",
            "false",
        }:
            return 0.0
        try:
            value = float(normalized)
        except ValueError as exc:
            raise ValueError(f"Unsupported feedback label: {value!r}") from exc
    label = float(value)
    if not math.isfinite(label) or not 0.0 <= label <= 1.0:
        raise ValueError(f"Feedback label must be between 0 and 1, got {value!r}")
    return label


def _feedback_is_reviewed(row: dict[str, Any]) -> bool:
    status = str(row.get("status", "")).strip().lower()
    if status in {"unclear", "unknown", "unreviewed", "pending", "skip"}:
        return False
    reviewed = row.get("reviewed", True)
    if isinstance(reviewed, str):
        return reviewed.strip().lower() in {"1", "true", "yes", "reviewed"}
    return bool(reviewed)


def _load_feedback_annotations(path: str) -> list[_FeedbackAnnotation]:
    annotations: list[_FeedbackAnnotation] = []
    feedback_path = Path(path)
    with feedback_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid feedback JSON at {feedback_path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Feedback row must be an object at {feedback_path}:{line_number}")
            if not _feedback_is_reviewed(row):
                continue
            try:
                label = _parse_feedback_label(row.get("label", row.get("human_label")))
                if label is None:
                    continue
                timestamp = float(row["timestamp"])
                if not math.isfinite(timestamp):
                    raise ValueError("timestamp must be finite")
                weight = float(row.get("weight", 1.0))
                if not math.isfinite(weight) or weight <= 0.0:
                    raise ValueError("weight must be finite and greater than zero")
                raw_track_id = row.get("track_id", -1)
                if raw_track_id is None or str(raw_track_id).strip().lower() in {"*", "all"}:
                    track_id = -1
                else:
                    track_id = int(raw_track_id)
                raw_session_id = row.get("session_id")
                raw_stream_id = row.get("stream_id")
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid feedback row at {feedback_path}:{line_number}: {exc}") from exc
            annotations.append(
                _FeedbackAnnotation(
                    session_id=None if raw_session_id in {None, ""} else str(raw_session_id),
                    stream_id=None if raw_stream_id in {None, ""} else str(raw_stream_id),
                    track_id=track_id,
                    timestamp=timestamp,
                    label=label,
                    weight=weight,
                    line_number=line_number,
                )
            )
    return annotations


def _weak_label_for_row(row: dict[str, Any], level_rank: dict[str, int], positive_rank: int) -> float:
    if "weak_label" in row:
        value = float(row["weak_label"])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"weak_label must be between 0 and 1, got {row['weak_label']!r}")
        return value
    level = str(row.get("risk_level", "LOW")).upper()
    return 1.0 if level_rank.get(level, 0) >= positive_rank else 0.0


def _apply_feedback(
    candidates: list[_WindowCandidate],
    annotations: list[_FeedbackAnnotation],
    max_seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if max_seconds < 0.0 or not math.isfinite(max_seconds):
        raise ValueError("feedback_max_seconds must be finite and non-negative")

    by_track: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        by_track[(candidate.session_id, candidate.stream_id, candidate.track_id)].append(index)

    # candidate index -> (precedence, human label, training weight). Exact-track
    # feedback wins over wildcard feedback, then the closest annotation wins,
    # with later JSONL rows resolving otherwise identical feedback deterministically.
    assignments: dict[int, tuple[tuple[int, float, int], float, float]] = {}
    for annotation in annotations:
        for (session_id, stream_id, track_id), indices in by_track.items():
            if annotation.session_id is not None and annotation.session_id != session_id:
                continue
            if annotation.stream_id is not None and annotation.stream_id != stream_id:
                continue
            if annotation.track_id != -1 and annotation.track_id != track_id:
                continue

            nearest_index = min(
                indices,
                key=lambda index: (
                    abs(candidates[index].end_timestamp - annotation.timestamp),
                    candidates[index].end_timestamp,
                ),
            )
            delta = abs(candidates[nearest_index].end_timestamp - annotation.timestamp)
            if delta > max_seconds:
                continue
            precedence = (
                1 if annotation.track_id != -1 else 0,
                -delta,
                annotation.line_number,
            )
            prior = assignments.get(nearest_index)
            if prior is None or precedence > prior[0]:
                assignments[nearest_index] = (precedence, annotation.label, annotation.weight)

    selected = sorted(assignments)
    if not selected:
        raise ValueError("No reviewed feedback matched a sequence window within the configured tolerance")
    return (
        np.stack([candidates[index].features for index in selected]),
        np.array([assignments[index][1] for index in selected], dtype=np.float32),
        np.array([assignments[index][2] for index in selected], dtype=np.float32),
    )


def load_frame_log_dataset(
    path: str,
    sequence_len: int = 16,
    min_positive_level: str = "HIGH",
    feedback_path: str | None = None,
    feedback_max_seconds: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    level_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
    positive_rank = level_rank.get(min_positive_level.upper(), 2)

    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("record_type") in {"feedback", "annotation"}:
                continue
            session_id = str(row.get("session_id") or "legacy")
            key = (session_id, str(row.get("stream_id", "unknown")), int(row["track_id"]))
            grouped[key].append(row)

    candidates: list[_WindowCandidate] = []

    for (session_id, stream_id, track_id), rows in grouped.items():
        rows.sort(key=lambda x: float(x.get("timestamp", 0.0)))
        window: deque[list[float]] = deque(maxlen=sequence_len)
        for row in rows:
            posture = row.get("posture", "unknown")
            if isinstance(posture, str):
                posture_scalar = POSTURE_MAP.get(posture, 0.0)
            else:
                posture_scalar = float(posture)
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
            candidates.append(
                _WindowCandidate(
                    features=np.array(window, dtype=np.float32),
                    weak_label=_weak_label_for_row(row, level_rank, positive_rank),
                    session_id=session_id,
                    stream_id=stream_id,
                    track_id=track_id,
                    end_timestamp=float(row.get("timestamp", 0.0)),
                )
            )

    if not candidates:
        raise ValueError(f"No sequences could be built from {path}. Need at least {sequence_len} timesteps per track.")
    if feedback_path is not None:
        return _apply_feedback(
            candidates,
            _load_feedback_annotations(feedback_path),
            max_seconds=float(feedback_max_seconds),
        )
    return (
        np.stack([candidate.features for candidate in candidates]),
        np.array([candidate.weak_label for candidate in candidates], dtype=np.float32),
        np.ones(len(candidates), dtype=np.float32),
    )


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


def prepare_train_validation_datasets(
    train: DatasetArrays,
    validation: DatasetArrays | None = None,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[DatasetArrays, DatasetArrays]:
    """Use an explicit validation dataset, or retain the legacy random split.

    Public-dataset adapters should split complete subjects or clips before they
    create overlapping sequence windows, then pass those files separately. This
    function deliberately leaves an explicit validation set untouched.
    """
    if validation is None:
        return split_dataset(*train, val_ratio=val_ratio, seed=seed)

    x_train, y_train, w_train = train
    x_val, y_val, w_val = validation
    _validate_dataset_arrays("training", x_train, y_train, w_train)
    _validate_dataset_arrays("validation", x_val, y_val, w_val)
    if x_train.shape[1:] != x_val.shape[1:]:
        raise ValueError(
            "Training and validation sequence shapes must match, got "
            f"{x_train.shape[1:]} and {x_val.shape[1:]}"
        )
    return train, validation


def _validate_dataset_arrays(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
) -> None:
    if x.ndim != 3:
        raise ValueError(f"{name.capitalize()} features must have shape [samples, timesteps, features]")
    sample_count = x.shape[0]
    if sample_count == 0:
        raise ValueError(f"{name.capitalize()} dataset must contain at least one sample")
    if y.shape != (sample_count,):
        raise ValueError(f"{name.capitalize()} labels must have one value per sample")
    if w.shape != (sample_count,):
        raise ValueError(f"{name.capitalize()} weights must have one value per sample")
