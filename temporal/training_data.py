from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field


POSTURE_MAP = {"unknown": 0.0, "standing": 0.2, "sitting": 0.6, "lying": 1.0}


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


def load_frame_log_dataset(path: str, sequence_len: int = 16, min_positive_level: str = "HIGH") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    level_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
    positive_rank = level_rank.get(min_positive_level.upper(), 2)

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = (str(row.get("stream_id", "unknown")), int(row["track_id"]))
            grouped[key].append(row)

    sequences: list[np.ndarray] = []
    labels: list[float] = []
    weights: list[float] = []

    for _, rows in grouped.items():
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
            level = str(row.get("risk_level", "LOW")).upper()
            label = 1.0 if level_rank.get(level, 0) >= positive_rank else 0.0
            sequences.append(np.array(window, dtype=np.float32))
            labels.append(label)
            weights.append(1.0)

    if not sequences:
        raise ValueError(f"No sequences could be built from {path}. Need at least {sequence_len} timesteps per track.")
    return np.stack(sequences), np.array(labels, dtype=np.float32), np.array(weights, dtype=np.float32)


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
