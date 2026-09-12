from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from temporal.temporal_model import (
    TemporalModelMeta,
    TorchGRUInferenceEngine,
    TorchTransformerLiteInferenceEngine,
)
from temporal.trainer import TemporalRiskTrainer, TrainerConfig, _binary_metrics
from temporal.training_data import (
    DatasetArrays,
    load_frame_log_dataset,
    load_sequence_dataset,
    prepare_train_validation_datasets,
)
from utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train temporal risk model (GRU or transformer-lite) from JSONL dataset")
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument(
        "--val-input",
        help="Optional pre-split validation JSONL path; keeps complete subjects/clips out of training",
    )
    parser.add_argument(
        "--test-input",
        help="Optional untouched test JSONL evaluated only after training and threshold calibration",
    )
    parser.add_argument("--format", choices=["frame", "sequence"], default="frame", help="Dataset format")
    parser.add_argument("--output", default="models/temporal_gru.pt", help="Output model file")
    parser.add_argument("--metrics-out", default="output/temporal_train_metrics.json", help="Output metrics JSON path")
    parser.add_argument("--sequence-len", type=int, default=16, help="Sequence length")
    parser.add_argument(
        "--model-fps",
        type=float,
        help="Training sample cadence stored in the checkpoint for live resampling",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--epochs", type=int, default=20, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold")
    parser.add_argument(
        "--max-fpr",
        type=float,
        help="Optionally calibrate the decision threshold on validation data to cap false-positive rate",
    )
    parser.add_argument("--device", default="cpu", help="cpu | cuda")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model-type", default="gru", choices=["gru", "transformer_lite"], help="Temporal model family")
    parser.add_argument("--hidden-size", type=int, default=32, help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=1, help="GRU or encoder layers")
    parser.add_argument("--attention-heads", type=int, default=2, help="Transformer attention heads")
    parser.add_argument("--ff-mult", type=int, default=2, help="Transformer feed-forward multiplier")
    parser.add_argument("--min-positive-level", default="HIGH", choices=["MEDIUM", "HIGH", "CRITICAL"], help="Positive cutoff for frame logs")
    parser.add_argument("--feedback", help="Optional human feedback annotations JSONL (frame format only)")
    parser.add_argument("--val-feedback", help="Optional validation feedback annotations JSONL (frame format only)")
    parser.add_argument(
        "--feedback-max-seconds",
        type=float,
        default=2.0,
        help="Maximum distance between feedback and a sequence-window end timestamp",
    )
    return parser.parse_args()


def _load_dataset(
    path: str,
    args: argparse.Namespace,
    feedback_path: str | None = None,
) -> DatasetArrays:
    if args.format == "sequence":
        if feedback_path:
            raise ValueError("Feedback annotations are supported only with --format frame")
        return load_sequence_dataset(path, sequence_len=args.sequence_len)
    return load_frame_log_dataset(
        path,
        sequence_len=args.sequence_len,
        min_positive_level=args.min_positive_level,
        feedback_path=feedback_path,
        feedback_max_seconds=args.feedback_max_seconds,
    )


def _weighted_bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply sample weights without changing the trainer's five-feature API."""
    if x.shape[0] == 0:
        return x, y
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (x.shape[0],):
        raise ValueError("Training weights must have one value per sample")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("Training weights must be finite and greater than zero")
    if np.allclose(weights, weights[0]):
        return x, y
    probabilities = weights / weights.sum()
    indices = np.random.default_rng(seed).choice(
        x.shape[0],
        size=x.shape[0],
        replace=True,
        p=probabilities,
    )
    return x[indices], y[indices]


def main() -> None:
    args = parse_args()
    logger = setup_logger(name="train_temporal", level="INFO")

    if args.val_feedback and not args.val_input:
        raise ValueError("--val-feedback requires --val-input")
    if args.max_fpr is not None and not args.val_input:
        raise ValueError("--max-fpr requires --val-input so calibration cannot use overlapping random windows")
    if args.model_fps is not None and args.model_fps <= 0.0:
        raise ValueError("--model-fps must be greater than zero")
    if args.format == "sequence" and (args.feedback or args.val_feedback):
        raise ValueError("Feedback annotations are supported only with --format frame")
    dataset_paths = [Path(path).resolve() for path in (args.input, args.val_input, args.test_input) if path]
    if len(dataset_paths) != len(set(dataset_paths)):
        raise ValueError("--input, --val-input, and --test-input must be different files")

    train_data = _load_dataset(args.input, args, feedback_path=args.feedback)
    validation_data = (
        _load_dataset(args.val_input, args, feedback_path=args.val_feedback)
        if args.val_input
        else None
    )
    test_data = _load_dataset(args.test_input, args) if args.test_input else None
    (x_train, y_train, w_train), (x_val, y_val, w_val) = prepare_train_validation_datasets(
        train_data,
        validation=validation_data,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    x_train_fit, y_train_fit = _weighted_bootstrap(x_train, y_train, w_train, seed=args.seed)

    cfg = TrainerConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        early_stop_patience=args.patience,
        decision_threshold=args.threshold,
        max_false_positive_rate=args.max_fpr,
        device=args.device,
        seed=args.seed,
        model_type=args.model_type,
    )

    trainer = TemporalRiskTrainer(cfg)
    meta = TemporalModelMeta(
        sequence_len=args.sequence_len,
        input_size=int(x_train.shape[-1]),
        hidden_size=int(args.hidden_size),
        num_layers=int(args.num_layers),
        model_type=args.model_type,
        attention_heads=int(args.attention_heads),
        ff_mult=int(args.ff_mult),
    )
    metrics, artifact = trainer.train(
        x_train=x_train_fit,
        y_train=y_train_fit,
        x_val=x_val,
        y_val=y_val,
        model_meta=meta,
    )
    if args.model_fps is not None:
        artifact["model_fps"] = float(args.model_fps)
    trainer.save(artifact, args.output)

    test_metrics = None
    test_samples = 0
    if test_data is not None:
        x_test, y_test, _ = test_data
        engine = (
            TorchTransformerLiteInferenceEngine(args.output, device=args.device)
            if args.model_type == "transformer_lite"
            else TorchGRUInferenceEngine(args.output, device=args.device)
        )
        test_probabilities = np.asarray([engine.predict(sequence) for sequence in x_test], dtype=np.float32)
        test_metrics = _binary_metrics(
            y_test,
            test_probabilities,
            threshold=float(artifact["decision_threshold"]),
        )
        test_samples = int(x_test.shape[0])

    payload = {
        "input_path": str(Path(args.input)),
        "validation_input_path": str(Path(args.val_input)) if args.val_input else None,
        "test_input_path": str(Path(args.test_input)) if args.test_input else None,
        "split_strategy": "explicit_files" if args.val_input else "random_windows",
        "format": args.format,
        "output_model": str(Path(args.output)),
        "samples_total": int(x_train.shape[0] + x_val.shape[0]),
        "samples_train": int(x_train.shape[0]),
        "samples_val": int(x_val.shape[0]),
        "samples_test": test_samples,
        "sequence_len": args.sequence_len,
        "model_fps": args.model_fps,
        "model_type": args.model_type,
        "decision_threshold": float(artifact["decision_threshold"]),
        "max_false_positive_rate": args.max_fpr,
        "best_epoch": int(artifact["best_epoch"]),
        "feedback_path": str(Path(args.feedback)) if args.feedback else None,
        "validation_feedback_path": str(Path(args.val_feedback)) if args.val_feedback else None,
        "feedback_max_seconds": args.feedback_max_seconds if args.feedback else None,
        "train_weight_min": float(w_train.min()) if w_train.size else None,
        "train_weight_max": float(w_train.max()) if w_train.size else None,
        "val_weight_min": float(w_val.min()) if w_val.size else None,
        "val_weight_max": float(w_val.max()) if w_val.size else None,
        "metrics": metrics,
        "test_metrics": test_metrics,
    }
    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("saved model=%s metrics=%s", args.output, args.metrics_out)
    logger.info("metrics=%s", metrics)


if __name__ == "__main__":
    main()
