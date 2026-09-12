from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from output.training_data_logger import TrainingDataLogger
from scripts.train_temporal_gru import _weighted_bootstrap
from temporal.training_data import load_frame_log_dataset
from utils.schemas import FeatureVector, RiskEvent


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def _frame_rows(*, session: str, stream: str, track_id: int, count: int = 6) -> list[dict]:
    return [
        {
            "session_id": session,
            "stream_id": stream,
            "track_id": track_id,
            "timestamp": float(i),
            "speed": float(track_id * 100 + i),
            "vy": 1.0,
            "acc": 0.5,
            "lean": 8.0,
            "posture": "standing",
            "risk_level": "LOW",
            "weak_label": 0,
        }
        for i in range(count)
    ]


def test_logger_records_prediction_as_weak_label_with_session_and_quality(tmp_path: Path) -> None:
    path = tmp_path / "features.jsonl"
    logger = TrainingDataLogger(str(path), session_id="session-a")
    feature = FeatureVector(
        track_id=4,
        timestamp=10.0,
        center_of_mass=(3.0, 4.0),
        velocity=(3.0, 4.0),
        acceleration=(0.0, 2.0),
        joint_angles={},
        posture="standing",
        bed_zone_distance=50.0,
        lean_angle=5.0,
        center_of_mass_3d_m=(0.1, 0.2, 2.0),
        velocity_3d_m_s=(0.0, -0.4, 0.0),
        acceleration_3d_m_s2=(0.0, -2.0, 0.0),
        depth_valid_ratio=0.75,
        camera_motion=True,
        camera_gyro_peak_rad_s=0.5,
        camera_accel_delta_peak_m_s2=2.5,
    )
    event = RiskEvent(
        track_id=4,
        risk_level="HIGH",
        confidence=0.8,
        timestamp=10.0,
        event="instability_risk",
    )

    logger.emit("camera-a", feature, event, ml_probability=0.37)
    row = json.loads(path.read_text(encoding="utf-8"))

    assert "label" not in row
    assert row["weak_label"] == 1
    assert row["weak_label_confidence"] == 0.8
    assert row["session_id"] == "session-a"
    assert row["center_of_mass_3d_m"] == [0.1, 0.2, 2.0]
    assert row["depth_valid_ratio"] == 0.75
    assert row["camera_motion"] is True
    assert row["ml_probability"] == 0.37


def test_loader_without_feedback_keeps_legacy_risk_labels_and_five_features(tmp_path: Path) -> None:
    path = tmp_path / "legacy.jsonl"
    rows = _frame_rows(session="legacy", stream="s0", track_id=1, count=4)
    rows[2]["risk_level"] = "HIGH"
    rows[2].pop("weak_label")
    _write_jsonl(path, rows)

    x, y, weights = load_frame_log_dataset(str(path), sequence_len=3)

    assert x.shape == (2, 3, 5)
    assert y.tolist() == [1.0, 0.0]
    assert weights.tolist() == [1.0, 1.0]


def test_feedback_exact_and_wildcard_match_nearest_windows_and_skip_unclear(tmp_path: Path) -> None:
    frames_path = tmp_path / "frames.jsonl"
    feedback_path = tmp_path / "feedback.jsonl"
    rows = (
        _frame_rows(session="a", stream="s0", track_id=1)
        + _frame_rows(session="a", stream="s0", track_id=2)
        + _frame_rows(session="a", stream="other", track_id=3)
    )
    _write_jsonl(frames_path, rows)
    _write_jsonl(
        feedback_path,
        [
            {
                "session_id": "a",
                "stream_id": "s0",
                "track_id": 1,
                "timestamp": 3.1,
                "label": "confirmed_fall",
                "weight": 2.5,
            },
            {
                "session_id": "a",
                "stream_id": "s0",
                "track_id": -1,
                "timestamp": 4.1,
                "label": "non_fall_activity",
                "weight": 0.5,
            },
            {
                "session_id": "a",
                "stream_id": "s0",
                "track_id": 2,
                "timestamp": 5.0,
                "label": "unclear",
                "reviewed": True,
            },
            {
                "session_id": "a",
                "stream_id": "other",
                "track_id": 3,
                "timestamp": 4.0,
                "label": 1,
                "reviewed": False,
            },
        ],
    )

    x, y, weights = load_frame_log_dataset(
        str(frames_path),
        sequence_len=3,
        feedback_path=str(feedback_path),
        feedback_max_seconds=0.25,
    )

    # Terminal speed identifies track/window: 103=t1@3, 104=t1@4, 204=t2@4.
    observed = {
        float(sequence[-1, 0]): (float(label), float(weight))
        for sequence, label, weight in zip(x, y, weights)
    }
    assert observed == {103.0: (1.0, 2.5), 104.0: (0.0, 0.5), 204.0: (0.0, 0.5)}
    assert x.shape[-1] == 5


def test_exact_feedback_beats_wildcard_for_the_same_window(tmp_path: Path) -> None:
    frames_path = tmp_path / "frames.jsonl"
    feedback_path = tmp_path / "feedback.jsonl"
    _write_jsonl(frames_path, _frame_rows(session="a", stream="s0", track_id=1, count=4))
    _write_jsonl(
        feedback_path,
        [
            {"session_id": "a", "stream_id": "s0", "track_id": 1, "timestamp": 3.0, "label": 0},
            {"session_id": "a", "stream_id": "s0", "track_id": -1, "timestamp": 3.0, "label": 1},
        ],
    )

    _, y, _ = load_frame_log_dataset(
        str(frames_path),
        sequence_len=3,
        feedback_path=str(feedback_path),
        feedback_max_seconds=0.0,
    )

    assert y.tolist() == [0.0]


def test_weighted_bootstrap_uses_nonuniform_feedback_weights() -> None:
    x = np.zeros((100, 2, 5), dtype=np.float32)
    x[:, :, 0] = np.arange(100, dtype=np.float32)[:, None]
    y = np.zeros(100, dtype=np.float32)
    y[-1] = 1.0
    weights = np.ones(100, dtype=np.float32)
    weights[-1] = 10_000.0

    sampled_x, sampled_y = _weighted_bootstrap(x, y, weights, seed=7)

    assert sampled_x.shape[-1] == 5
    assert int(sampled_y.sum()) > 90
