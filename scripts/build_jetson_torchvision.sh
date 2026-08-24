#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
source scripts/activate_jetson_env.sh

torchvision_version="${TORCHVISION_VERSION:-0.20.0}"
build_dir="$(mktemp -d /tmp/ah-torchvision.XXXXXX)"
trap 'rm -rf "$build_dir"' EXIT

git clone --depth 1 --branch "v${torchvision_version}" https://github.com/pytorch/vision.git "$build_dir"
env \
  FORCE_CUDA=1 \
  TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.7}" \
  MAX_JOBS="${MAX_JOBS:-4}" \
  BUILD_VERSION="$torchvision_version" \
  python -m pip install --no-build-isolation --no-deps "$build_dir"

python - <<'PY'
import torch
from torchvision.ops import nms

boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 9.0, 9.0]], device="cuda")
scores = torch.tensor([0.9, 0.8], device="cuda")
assert nms(boxes, scores, 0.5).cpu().tolist() == [0]
print("Jetson torchvision CUDA NMS passed")
PY
