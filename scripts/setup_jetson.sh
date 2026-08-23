#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

python3 - <<'PY'
import cv2
import torch

if not torch.cuda.is_available():
    raise SystemExit("Install the NVIDIA PyTorch wheel matching JetPack before continuing")
if "GStreamer:                   YES" not in cv2.getBuildInformation():
    raise SystemExit("Use JetPack/system OpenCV with GStreamer support; do not install opencv-python")
print("JetPack PyTorch/OpenCV prerequisites passed")
PY

python3 -m venv --system-site-packages .venv-jetson
source .venv-jetson/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-jetson.txt
python -m pip install --no-deps ultralytics==8.4.32
python scripts/verify_jetson_runtime.py --skip-engine-check --skip-inference

echo "Jetson environment ready. Build the engine with scripts/export_jetson_tensorrt.py."
