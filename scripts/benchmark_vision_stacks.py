"""Replay identical RGB frames through each configured detector/pose stack."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from pose.pose_estimator import RTMOMMPoseEngine, UltralyticsPoseEngine
from scripts.benchmark_pose_models import _representative_frames


def build_backend(spec, device):
    if spec["backend"] == "ultralytics":
        return UltralyticsPoseEngine(spec["model"], device=device,
                                     input_size=spec.get("imgsz", 640),
                                     conf_threshold=spec.get("conf", 0.25))
    if spec["backend"] == "mmpose":
        return RTMOMMPoseEngine(spec["model"], device=device,
                               bbox_thr=spec.get("conf", 0.25),
                               pose_weights=spec.get("pose_weights"),
                               det_model=spec.get("det_model"),
                               det_weights=spec.get("det_weights"))
    raise ValueError(f"Unknown backend: {spec['backend']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="config/benchmark_vision.json")
    parser.add_argument("--source", required=True, help="Recorded representative RGB video")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="output/vision_benchmark.json")
    args = parser.parse_args()
    if args.frames < 1 or args.warmup < 0:
        parser.error("frames must be positive and warmup nonnegative")
    import torch
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; benchmark refuses CPU fallback")
    sync = (lambda: torch.cuda.synchronize(args.device)) if args.device.startswith("cuda") else (lambda: None)
    frames = _representative_frames(args.source, args.frames)
    digest = hashlib.sha256()
    for frame in frames:
        digest.update(frame.tobytes())
    specs = json.loads(Path(args.suite).read_text())
    rows = []
    for spec in specs:
        row = {"name": spec["name"], "spec": spec, "status": "error"}
        backend = None
        try:
            backend = build_backend(spec, args.device)
            for i in range(args.warmup):
                backend.predict_full(frames[i % len(frames)])
            sync()
            timings, counts, coverage = [], [], []
            for frame in frames:
                sync()
                start = time.perf_counter()
                poses = backend.predict_full(frame)
                sync()
                timings.append((time.perf_counter() - start) * 1000)
                counts.append(len(poses))
                for _, keypoints, _ in poses:
                    coverage.append(float(np.mean(keypoints[:, 2] >= 0.25)))
            row.update(status="ok", frames=len(frames),
                       p50_ms=float(np.percentile(timings, 50)),
                       p95_ms=float(np.percentile(timings, 95)),
                       p99_ms=float(np.percentile(timings, 99)),
                       fps=1000 / float(np.mean(timings)),
                       mean_people=float(np.mean(counts)),
                       mean_keypoint_coverage=float(np.mean(coverage)) if coverage else None)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            del backend
            gc.collect()
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = {"source": str(Path(args.source).resolve()), "frame_sha256": digest.hexdigest(),
              "device": args.device, "warmup": args.warmup,
              "scope": "Detector + pose including preprocessing and CPU output transfer; excludes capture, depth, tracking, rules and UI",
              "accuracy": "Not measured: coverage and detection counts are diagnostics, not accuracy. Labeled footage is required.",
              "results": rows}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    fields = ["name", "status", "frames", "p50_ms", "p95_ms", "p99_ms", "fps", "mean_people", "mean_keypoint_coverage", "error"]
    with output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    if any(row["status"] != "ok" for row in rows):
        raise SystemExit("Comparison incomplete; inspect per-model errors in the report")


if __name__ == "__main__":
    main()
