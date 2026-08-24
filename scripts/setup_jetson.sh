#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

python3 -m venv --system-site-packages .venv-jetson
source .venv-jetson/bin/activate
python -m pip install --upgrade pip "setuptools<80" wheel

# JetPack 6.2.1 uses CUDA 12.6. Override this URL when targeting another
# JetPack/Python combination rather than installing the generic PyPI torch wheel.
torch_wheel_url="${JETSON_TORCH_WHEEL_URL:-https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl}"
if ! python -c 'import torch' >/dev/null 2>&1; then
  python -m pip install "$torch_wheel_url"
fi
python -m pip install "nvidia-cusparselt-cu12==0.6.2"
source scripts/activate_jetson_env.sh
python -m pip install -r requirements-jetson.txt
python -m pip install --no-deps ultralytics==8.4.52

if ! python - <<'PY' >/dev/null 2>&1
import torch
import torchvision
assert torch.cuda.is_available()
assert hasattr(torch.ops.torchvision, "nms")
PY
then
  scripts/build_jetson_torchvision.sh
fi

python - <<'PY'
import cv2
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA-enabled Jetson PyTorch validation failed")
if "GStreamer:                   YES" not in cv2.getBuildInformation():
    raise SystemExit("Use JetPack/system OpenCV with GStreamer support; do not install opencv-python")
print("JetPack PyTorch/OpenCV prerequisites passed")
PY

python scripts/verify_jetson_runtime.py --skip-engine-check --skip-inference

echo "Jetson environment ready. Build the YOLO26 pose engine with scripts/export_jetson_tensorrt.py."
