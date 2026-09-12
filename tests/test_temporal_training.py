from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.train_temporal_gru import parse_args
import temporal.trainer as trainer_module
from temporal.temporal_model import TemporalModelMeta
from temporal.trainer import (
    GRURiskTrainer,
    TemporalRiskTrainer,
    TrainerConfig,
    _binary_metrics,
    select_threshold_for_max_fpr,
)
from temporal.training_data import (
    load_frame_log_dataset,
    prepare_train_validation_datasets,
    split_dataset,
)


def test_frame_log_to_sequences(tmp_path: Path) -> None:
    path = tmp_path / "train_features.jsonl"
    rows = []
    for i in range(24):
        rows.append(
            {
                "stream_id": "s0",
                "track_id": 1,
                "timestamp": float(i),
                "speed": 4.0 + i * 0.1,
                "vy": 1.0,
                "acc": 0.5,
                "lean": 8.0,
                "posture": "standing",
                "risk_level": "LOW" if i < 12 else "HIGH",
            }
        )
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    x, y, _ = load_frame_log_dataset(str(path), sequence_len=8, min_positive_level="HIGH")
    assert x.shape[0] > 0
    assert x.shape[1:] == (8, 5)
    assert set(np.unique(y).tolist()).issubset({0.0, 1.0})


def test_explicit_validation_dataset_is_not_randomly_mixed() -> None:
    train = (
        np.full((6, 4, 5), 11.0, dtype=np.float32),
        np.zeros(6, dtype=np.float32),
        np.ones(6, dtype=np.float32),
    )
    validation = (
        np.full((3, 4, 5), 99.0, dtype=np.float32),
        np.ones(3, dtype=np.float32),
        np.full(3, 2.0, dtype=np.float32),
    )

    train_out, validation_out = prepare_train_validation_datasets(
        train,
        validation=validation,
        val_ratio=0.99,
        seed=123,
    )

    for actual, expected in zip(train_out, train):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(validation_out, validation):
        np.testing.assert_array_equal(actual, expected)
    assert np.all(train_out[0] == 11.0)
    assert np.all(validation_out[0] == 99.0)


def test_explicit_validation_rejects_incompatible_sequence_shape() -> None:
    train = (
        np.zeros((4, 8, 5), dtype=np.float32),
        np.zeros(4, dtype=np.float32),
        np.ones(4, dtype=np.float32),
    )
    validation = (
        np.zeros((2, 16, 5), dtype=np.float32),
        np.zeros(2, dtype=np.float32),
        np.ones(2, dtype=np.float32),
    )

    with pytest.raises(ValueError, match="sequence shapes must match"):
        prepare_train_validation_datasets(train, validation=validation)


def test_legacy_random_split_is_preserved() -> None:
    x = np.arange(12 * 4 * 5, dtype=np.float32).reshape(12, 4, 5)
    y = np.arange(12, dtype=np.float32)
    w = np.ones(12, dtype=np.float32)

    expected = split_dataset(x, y, w, val_ratio=0.25, seed=7)
    actual = prepare_train_validation_datasets((x, y, w), val_ratio=0.25, seed=7)

    for actual_partition, expected_partition in zip(actual, expected):
        for actual_array, expected_array in zip(actual_partition, expected_partition):
            np.testing.assert_array_equal(actual_array, expected_array)


def test_training_cli_accepts_explicit_validation_input(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_temporal_gru.py",
            "--input",
            "train.jsonl",
            "--val-input",
            "validation.jsonl",
            "--max-fpr",
            "0.02",
        ],
    )

    args = parse_args()

    assert args.input == "train.jsonl"
    assert args.val_input == "validation.jsonl"
    assert args.max_fpr == pytest.approx(0.02)


def test_binary_metrics_include_confusion_counts_and_false_positive_rate() -> None:
    metrics = _binary_metrics(
        np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32),
        np.array([0.1, 0.8, 0.7, 0.2], dtype=np.float32),
        threshold=0.5,
    )

    assert metrics["tn"] == 1.0
    assert metrics["fp"] == 1.0
    assert metrics["fn"] == 1.0
    assert metrics["tp"] == 1.0
    assert metrics["specificity"] == pytest.approx(0.5)
    assert metrics["false_positive_rate"] == pytest.approx(0.5)
    assert metrics["decision_threshold"] == pytest.approx(0.5)


def test_threshold_calibration_maximizes_recall_under_fpr_cap() -> None:
    labels = np.array([0, 0, 0, 0, 1, 1, 1], dtype=np.float32)
    probabilities = np.array([0.1, 0.2, 0.3, 0.9, 0.4, 0.8, 0.95], dtype=np.float32)

    threshold, metrics = select_threshold_for_max_fpr(
        labels,
        probabilities,
        max_false_positive_rate=0.25,
    )

    assert threshold == pytest.approx(0.4)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["false_positive_rate"] == pytest.approx(0.25)


def test_threshold_calibration_requires_both_validation_classes() -> None:
    with pytest.raises(ValueError, match="both positive and negative"):
        select_threshold_for_max_fpr(
            np.zeros(4, dtype=np.float32),
            np.linspace(0.1, 0.4, 4, dtype=np.float32),
            max_false_positive_rate=0.02,
        )


def test_trainer_recomputes_metrics_from_restored_best_epoch(monkeypatch: pytest.MonkeyPatch) -> None:
    observed_probabilities: list[np.ndarray] = []

    def fake_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
        _ = y_true
        observed_probabilities.append(y_prob.copy())
        scores = [0.9, 0.1, 0.25]
        score = scores[len(observed_probabilities) - 1]
        return {
            "accuracy": score,
            "precision": score,
            "recall": score,
            "f1": score,
            "auc": score,
            "tn": 0.0,
            "fp": 0.0,
            "fn": 0.0,
            "tp": 0.0,
            "specificity": 0.0,
            "false_positive_rate": 0.0,
            "decision_threshold": threshold,
        }

    monkeypatch.setattr(trainer_module, "_binary_metrics", fake_metrics)
    rng = np.random.default_rng(9)
    x_train = rng.normal(size=(12, 8, 5)).astype(np.float32)
    y_train = np.array([0.0, 1.0] * 6, dtype=np.float32)
    x_val = rng.normal(size=(6, 8, 5)).astype(np.float32)
    y_val = np.array([0.0, 1.0] * 3, dtype=np.float32)
    trainer = TemporalRiskTrainer(
        TrainerConfig(
            epochs=2,
            batch_size=6,
            learning_rate=0.05,
            early_stop_patience=5,
            seed=9,
        )
    )

    metrics, artifact = trainer.train(x_train, y_train, x_val, y_val, TemporalModelMeta(sequence_len=8))

    assert len(observed_probabilities) == 3
    assert not np.allclose(observed_probabilities[0], observed_probabilities[1])
    np.testing.assert_allclose(observed_probabilities[2], observed_probabilities[0])
    assert metrics["f1"] == pytest.approx(0.25)
    assert artifact["best_epoch"] == 1
    assert artifact["decision_threshold"] == pytest.approx(0.5)


def test_trainer_saves_calibrated_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    observed_caps: list[float] = []

    def fake_calibration(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        max_false_positive_rate: float,
    ) -> tuple[float, dict[str, float]]:
        observed_caps.append(max_false_positive_rate)
        return 0.73, _binary_metrics(y_true, y_prob, threshold=0.73)

    monkeypatch.setattr(trainer_module, "select_threshold_for_max_fpr", fake_calibration)
    rng = np.random.default_rng(3)
    x_train = rng.normal(size=(6, 4, 5)).astype(np.float32)
    y_train = np.array([0.0, 1.0] * 3, dtype=np.float32)
    x_val = rng.normal(size=(4, 4, 5)).astype(np.float32)
    y_val = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    trainer = TemporalRiskTrainer(
        TrainerConfig(epochs=0, max_false_positive_rate=0.02, seed=3)
    )

    metrics, artifact = trainer.train(x_train, y_train, x_val, y_val, TemporalModelMeta(sequence_len=4))

    assert observed_caps == [pytest.approx(0.02)]
    assert metrics["decision_threshold"] == pytest.approx(0.73)
    assert artifact["decision_threshold"] == pytest.approx(0.73)


def test_gru_trainer_smoke(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    x = rng.normal(size=(64, 16, 5)).astype(np.float32)
    # label depends on lean feature average (index 3), makes learnable pattern
    y = (x[:, :, 3].mean(axis=1) > 0.0).astype(np.float32)
    w = np.ones_like(y, dtype=np.float32)

    (x_train, y_train, _), (x_val, y_val, _) = split_dataset(x, y, w, val_ratio=0.25, seed=42)

    trainer = GRURiskTrainer(
        TrainerConfig(
            epochs=2,
            batch_size=16,
            learning_rate=1e-3,
            weight_decay=1e-4,
            early_stop_patience=2,
            device="cpu",
            seed=42,
        )
    )
    metrics, artifact = trainer.train(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        model_meta=TemporalModelMeta(sequence_len=16),
    )
    out = tmp_path / "temporal_gru.pt"
    trainer.save(artifact, str(out))

    assert out.exists()
    assert "model_state_dict" in artifact
    assert artifact["decision_threshold"] == pytest.approx(0.5)
    assert metrics["decision_threshold"] == pytest.approx(0.5)
    assert 0.0 <= metrics["f1"] <= 1.0


def test_transformer_trainer_smoke(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(48, 16, 5)).astype(np.float32)
    y = (x[:, :, 1].mean(axis=1) + x[:, :, 3].mean(axis=1) > 0.1).astype(np.float32)
    w = np.ones_like(y, dtype=np.float32)
    (x_train, y_train, _), (x_val, y_val, _) = split_dataset(x, y, w, val_ratio=0.25, seed=7)

    trainer = TemporalRiskTrainer(
        TrainerConfig(
            epochs=1,
            batch_size=16,
            learning_rate=1e-3,
            early_stop_patience=1,
            device="cpu",
            seed=7,
            model_type="transformer_lite",
        )
    )
    metrics, artifact = trainer.train(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        model_meta=TemporalModelMeta(
            sequence_len=16,
            model_type="transformer_lite",
            hidden_size=16,
            num_layers=1,
            attention_heads=2,
            ff_mult=2,
        ),
    )
    out = tmp_path / "temporal_transformer_lite.pt"
    trainer.save(artifact, str(out))

    assert out.exists()
    assert artifact.get("model_type") == "transformer_lite"
    assert 0.0 <= metrics["accuracy"] <= 1.0
