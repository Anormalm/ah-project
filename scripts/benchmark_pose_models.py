from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark pose candidates on the target and select the first accuracy-priority model meeting latency"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "models/yolo26s-pose-fp16-640.engine",
            "models/yolo26n-pose-fp16-512.engine",
        ],
        help="Models in preferred accuracy order",
    )
    parser.add_argument("--source", default=None, help="Optional video containing representative ward frames")
    parser.add_argument("--imgsz", type=int, default=None, help="Override engine manifest input size")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--target-fps", type=float, default=15.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="output/pose_benchmark.json")
    return parser.parse_args()


def _representative_frames(source: str | None, count: int) -> list[np.ndarray]:
    if source is None:
        return []
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open benchmark source: {source}")
    frames: list[np.ndarray] = []
    try:
        while len(frames) < count:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    if not frames:
        raise RuntimeError("Benchmark source produced no frames")
    return frames


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _model_imgsz(model_path: Path, override: int | None) -> int | list[int]:
    if override is not None:
        return int(override)
    manifest_path = model_path.with_suffix(model_path.suffix + ".json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        imgsz = manifest.get("imgsz", 640)
        if isinstance(imgsz, list):
            return [int(value) for value in imgsz]
        return int(imgsz)
    return 640


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO
    import torch

    source_frames = _representative_frames(args.source, max(args.iterations, 1))
    use_cuda = str(args.device).startswith("cuda") and torch.cuda.is_available()
    rows: list[dict] = []

    for raw_model_path in args.models:
        model_path = Path(raw_model_path).resolve()
        if not model_path.exists():
            rows.append({"model": str(model_path), "status": "missing"})
            continue
        model_imgsz = _model_imgsz(model_path, args.imgsz)
        if source_frames:
            frames = source_frames
        else:
            if isinstance(model_imgsz, list):
                height, width = model_imgsz
            else:
                height = width = model_imgsz
            frames = [np.zeros((height, width, 3), dtype=np.uint8)]
        model = YOLO(str(model_path))
        for idx in range(max(0, args.warmup)):
            model.predict(
                frames[idx % len(frames)],
                device=args.device,
                imgsz=model_imgsz,
                rect=False,
                verbose=False,
            )
        if use_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        latencies: list[float] = []
        people: list[int] = []
        for idx in range(max(1, args.iterations)):
            frame = frames[idx % len(frames)]
            started = time.perf_counter()
            result = model.predict(
                frame,
                device=args.device,
                imgsz=model_imgsz,
                rect=False,
                verbose=False,
            )[0]
            if use_cuda:
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - started) * 1000.0)
            people.append(0 if result.boxes is None else len(result.boxes))

        p95 = _percentile(latencies, 95)
        rows.append(
            {
                "model": str(model_path),
                "status": "ok",
                "imgsz": model_imgsz,
                "iterations": len(latencies),
                "latency_ms": {
                    "p50": _percentile(latencies, 50),
                    "p95": p95,
                    "p99": _percentile(latencies, 99),
                },
                "throughput_fps_at_p95": 1000.0 / max(p95, 1e-6),
                "mean_people": float(np.mean(people)),
                "cuda_peak_memory_mb": (
                    float(torch.cuda.max_memory_allocated() / (1024 * 1024)) if use_cuda else 0.0
                ),
                "meets_target": p95 <= 1000.0 / max(args.target_fps, 1e-6),
            }
        )
        del model
        if use_cuda:
            torch.cuda.empty_cache()

    selected = next((row["model"] for row in rows if row.get("meets_target")), None)
    payload = {
        "selection_policy": "first model in accuracy-priority order whose p95 latency meets target_fps",
        "target_fps": args.target_fps,
        "source": "representative_video" if args.source else "synthetic_zero_frame",
        "selected_model": selected,
        "results": rows,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if selected is None:
        raise SystemExit("No candidate met the target; inspect the benchmark JSON and lower model/input size")


if __name__ == "__main__":
    main()
