from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from temporal.temporal_model import TemporalModelMeta
from temporal.trainer import GRURiskTrainer, TrainerConfig
from temporal.training_data import load_frame_log_dataset, load_sequence_dataset, split_dataset
from utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train temporal GRU risk model from JSONL dataset")
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument("--format", choices=["frame", "sequence"], default="frame", help="Dataset format")
    parser.add_argument("--output", default="models/temporal_gru.pt", help="Output model file")
    parser.add_argument("--metrics-out", default="output/temporal_train_metrics.json", help="Output metrics JSON path")
    parser.add_argument("--sequence-len", type=int, default=16, help="Sequence length")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--epochs", type=int, default=20, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold")
    parser.add_argument("--device", default="cpu", help="cpu | cuda")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--min-positive-level", default="HIGH", choices=["MEDIUM", "HIGH", "CRITICAL"], help="Positive cutoff for frame logs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger(name="train_temporal", level="INFO")

    if args.format == "sequence":
        x, y, w = load_sequence_dataset(args.input, sequence_len=args.sequence_len)
    else:
        x, y, w = load_frame_log_dataset(
            args.input,
            sequence_len=args.sequence_len,
            min_positive_level=args.min_positive_level,
        )
    _ = w

    (x_train, y_train, _), (x_val, y_val, _) = split_dataset(
        x,
        y,
        np.ones_like(y, dtype=float),
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    cfg = TrainerConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        early_stop_patience=args.patience,
        decision_threshold=args.threshold,
        device=args.device,
        seed=args.seed,
    )

    trainer = GRURiskTrainer(cfg)
    meta = TemporalModelMeta(sequence_len=args.sequence_len, input_size=int(x.shape[-1]))
    metrics, artifact = trainer.train(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        model_meta=meta,
    )
    trainer.save(artifact, args.output)

    payload = {
        "input_path": str(Path(args.input)),
        "format": args.format,
        "output_model": str(Path(args.output)),
        "samples_total": int(x.shape[0]),
        "samples_train": int(x_train.shape[0]),
        "samples_val": int(x_val.shape[0]),
        "sequence_len": args.sequence_len,
        "metrics": metrics,
    }
    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("saved model=%s metrics=%s", args.output, args.metrics_out)
    logger.info("metrics=%s", metrics)


if __name__ == "__main__":
    main()
