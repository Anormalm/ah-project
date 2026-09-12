from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pose.pose_estimator import MockPoseEngine, PoseEstimator, UltralyticsPoseEngine
from temporal.public_dataset import load_public_manifest, prepare_public_dataset
from utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert strongly annotated public fall clips to sequence JSONL"
    )
    parser.add_argument("--manifest", required=True, help="JSON or YAML public dataset manifest")
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory for <dataset>_train.jsonl, <dataset>_val.jsonl, and summary JSON",
    )
    parser.add_argument(
        "--pose-backend",
        choices=["ultralytics_pose", "mock"],
        default="ultralytics_pose",
        help="Use mock only for pipeline smoke tests",
    )
    parser.add_argument("--model", default="models/yolo11n-pose.pt", help="Ultralytics pose weights")
    parser.add_argument(
        "--input-size",
        type=int,
        help="Override the manifest pose inference size",
    )
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing prepared outputs")
    return parser.parse_args()


def _device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    args = parse_args()
    logger = setup_logger(name="prepare_public_dataset", level="INFO")
    manifest, manifest_dir = load_public_manifest(args.manifest)
    device = _device(args.device)
    if args.pose_backend == "mock":
        backend = MockPoseEngine()
    else:
        backend = UltralyticsPoseEngine(
            model_path=args.model,
            device=device,
            input_size=args.input_size or manifest.pose_input_size,
        )
    pose_estimator = PoseEstimator(backend=backend)

    output_dir = Path(args.output_dir)
    train_output = output_dir / f"{manifest.dataset}_train.jsonl"
    val_output = output_dir / f"{manifest.dataset}_val.jsonl"
    summary_output = output_dir / f"{manifest.dataset}_preparation.json"

    def progress(clip, result) -> None:
        logger.info(
            "clip=%s split=%s frames=%d sequences=%d positive=%d negative=%d",
            clip.clip_id,
            clip.split,
            result["frames"],
            result["sequences"],
            result["positive"],
            result["negative"],
        )

    summary = prepare_public_dataset(
        manifest=manifest,
        manifest_dir=manifest_dir,
        pose_estimator=pose_estimator,
        train_output=train_output,
        val_output=val_output,
        overwrite=args.overwrite,
        progress=progress,
    )
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logger.info("prepared train=%s val=%s summary=%s", train_output, val_output, summary_output)


if __name__ == "__main__":
    main()
