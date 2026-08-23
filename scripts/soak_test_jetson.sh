#!/usr/bin/env bash
set -euo pipefail

duration_sec="${1:-3600}"
config_path="${2:-config/jetson_orin_nx_sota.yaml}"
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_dir="$project_dir/output/soak-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$run_dir"
cd "$project_dir"

tegrastats --interval 1000 --logfile "$run_dir/tegrastats.log" &
tegrastats_pid=$!

cleanup() {
  kill "$tegrastats_pid" 2>/dev/null || true
  wait "$tegrastats_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

set +e
timeout --signal=TERM --kill-after=20 "${duration_sec}s" \
  .venv-jetson/bin/python run.py --config "$config_path" \
  2>&1 | tee "$run_dir/application.log"
run_status=${PIPESTATUS[0]}
set -e

if [[ "$run_status" -ne 0 && "$run_status" -ne 124 ]]; then
  echo "Soak test failed with status $run_status; logs: $run_dir" >&2
  exit "$run_status"
fi

echo "Soak test complete: $run_dir"
