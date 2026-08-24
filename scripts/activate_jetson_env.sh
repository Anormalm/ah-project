#!/usr/bin/env bash
# Source this file: source scripts/activate_jetson_env.sh

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="$project_dir/.venv-jetson"
if [[ ! -f "$venv_dir/bin/activate" ]]; then
  echo "Missing $venv_dir; run scripts/setup_jetson.sh first" >&2
  return 1 2>/dev/null || exit 1
fi

source "$venv_dir/bin/activate"
cusparselt_lib="$venv_dir/lib/python3.10/site-packages/cusparselt/lib"
if [[ -d "$cusparselt_lib" ]]; then
  export LD_LIBRARY_PATH="$cusparselt_lib:${LD_LIBRARY_PATH:-}"
fi
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"
export PATH="$CUDA_HOME/bin:$PATH"
