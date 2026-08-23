from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Jetson runtime dependencies and optionally warm up configured engines")
    parser.add_argument("--config", default="config/jetson_orin_nx_rtsp.yaml")
    parser.add_argument(
        "--skip-engine-check",
        action="store_true",
        help="Verify only platform dependencies; intended for first-time setup before export",
    )
    parser.add_argument("--skip-inference", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    failures: list[str] = []
    machine = platform.machine().lower()
    if machine not in {"aarch64", "arm64"}:
        failures.append(f"expected aarch64/arm64, found {machine}")

    import cv2
    import torch

    if "GStreamer:                   YES" not in cv2.getBuildInformation():
        failures.append("OpenCV was not built with GStreamer support")
    if not torch.cuda.is_available():
        failures.append("torch.cuda.is_available() is false")

    try:
        import tensorrt as trt

        print(f"TensorRT {trt.__version__}")
        tensorrt_version = trt.__version__
    except ImportError:
        tensorrt_version = None
        failures.append("TensorRT Python bindings are unavailable")

    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    project_root = cfg_path.parent.parent
    engine_paths: list[Path] = []
    for section in ("detection", "pose"):
        model_path = (cfg.get(section) or {}).get("model_path")
        if model_path and str(model_path).endswith(".engine"):
            path = Path(model_path)
            engine_paths.append(path if path.is_absolute() else project_root / path)
    if not args.skip_engine_check:
        for engine in engine_paths:
            if not engine.exists():
                failures.append(f"engine not found: {engine}")
                continue
            manifest_path = engine.with_suffix(engine.suffix + ".json")
            if not manifest_path.exists():
                failures.append(f"engine manifest not found: {manifest_path}")
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("engine_sha256") != _sha256(engine):
                failures.append(f"engine hash does not match manifest: {engine}")
            built_trt = str(manifest.get("tensorrt", "unknown"))
            if tensorrt_version is not None and built_trt != str(tensorrt_version):
                failures.append(f"TensorRT version mismatch for {engine.name}: built={built_trt} runtime={tensorrt_version}")

    if not failures and not args.skip_engine_check and not args.skip_inference:
        import numpy as np
        from ultralytics import YOLO

        for engine in engine_paths:
            pose_cfg = cfg["pose"]
            imgsz = pose_cfg.get("input_size", 384)
            if isinstance(imgsz, list):
                height, width = imgsz
            else:
                height = width = int(imgsz)
            model = YOLO(str(engine))
            model.predict(
                np.zeros((height, width, 3), dtype=np.uint8),
                device="cuda:0",
                imgsz=imgsz,
                rect=False,
                verbose=False,
            )
            print(f"warmup passed: {engine}")

    print(f"machine={machine} torch={torch.__version__} cuda={torch.version.cuda} opencv={cv2.__version__}")
    if failures:
        raise SystemExit("Jetson verification failed:\n- " + "\n- ".join(failures))
    print("Jetson runtime verification passed")


if __name__ == "__main__":
    main()
