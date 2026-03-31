from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from temporal.temporal_model import TemporalModelMeta
from temporal.trainer import GRURiskTrainer, TrainerConfig
from temporal.training_data import load_frame_log_dataset, split_dataset


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
    assert 0.0 <= metrics["f1"] <= 1.0

