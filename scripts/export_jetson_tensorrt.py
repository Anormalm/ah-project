from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
from pathlib import Path


def _parse_imgsz(value: str) -> int | tuple[int, int]:
    parts = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(parts) == 1 and parts[0] > 0:
        return parts[0]
    if len(parts) == 2 and min(parts) > 0:
        return parts[0], parts[1]
    raise argparse.ArgumentTypeError("imgsz must be N or H,W")


def _size_tag(imgsz: int | tuple[int, int]) -> str:
    return str(imgsz) if isinstance(imgsz, int) else f"{imgsz[0]}x{imgsz[1]}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_optional(path: str) -> str | None:
    candidate = Path(path)
    if not candidate.exists():
        return None
    return candidate.read_text(encoding="utf-8", errors="replace").strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a target-specific TensorRT engine on Jetson")
    parser.add_argument("--weights", default="models/yolo11n-pose.pt")
    parser.add_argument("--output", default=None)
    parser.add_argument("--imgsz", type=_parse_imgsz, default=384)
    parser.add_argument("--precision", choices=["fp16", "int8"], default="fp16")
    parser.add_argument("--data", default=None, help="Representative dataset YAML; required for INT8")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workspace", type=float, default=2.0, help="TensorRT builder workspace in GiB")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--device", default="0")
    parser.add_argument("--allow-non-jetson", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    machine = platform.machine().lower()
    if machine not in {"aarch64", "arm64"} and not args.allow_non_jetson:
        raise SystemExit("Refusing to build a non-portable Jetson engine off-device. Run this script on the target Orin NX.")
    if args.precision == "int8" and not args.data:
        raise SystemExit("--data is required for representative INT8 calibration")

    import torch
    from ultralytics import YOLO, __version__ as ultralytics_version

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; verify the JetPack PyTorch installation")

    weights = Path(args.weights).resolve()
    if not weights.exists():
        raise SystemExit(f"Weights not found: {weights}")

    model = YOLO(str(weights))
    export_args = {
        "format": "engine",
        "imgsz": args.imgsz,
        "batch": max(1, int(args.batch)),
        "workspace": max(0.25, float(args.workspace)),
        "dynamic": bool(args.dynamic),
        "device": args.device,
        "verbose": False,
    }
    # These names work across the Ultralytics 8.x versions used by JetPack 6.
    if args.precision == "fp16":
        export_args["half"] = True
    else:
        export_args["int8"] = True
        export_args["data"] = args.data

    exported = Path(str(model.export(**export_args))).resolve()
    default_name = f"{weights.stem}-{args.precision}-{_size_tag(args.imgsz)}.engine"
    destination = Path(args.output).resolve() if args.output else weights.with_name(default_name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if exported != destination:
        shutil.copy2(exported, destination)

    try:
        import tensorrt as trt

        tensorrt_version = trt.__version__
    except ImportError:
        tensorrt_version = "unknown"

    manifest = {
        "engine": str(destination),
        "engine_sha256": _sha256(destination),
        "source_weights": str(weights),
        "source_sha256": _sha256(weights),
        "precision": args.precision,
        "imgsz": args.imgsz,
        "batch": max(1, int(args.batch)),
        "dynamic": bool(args.dynamic),
        "machine": machine,
        "l4t_release": _read_optional("/etc/nv_tegra_release"),
        "os_release": _read_optional("/etc/os-release"),
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "tensorrt": tensorrt_version,
        "ultralytics": ultralytics_version,
    }
    manifest_path = destination.with_suffix(destination.suffix + ".json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"engine: {destination}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
