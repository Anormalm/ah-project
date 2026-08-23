# AH Project - Real-Time Pose-Based Risk Detection

Production-grade, modular pipeline for real-time multi-person pose-based risk detection with privacy-first processing.

## What This Does

Pipeline flow:

`Camera/Video -> Detection -> Pose -> Tracking -> Feature Extraction -> Temporal Logic -> Risk Scoring -> Alerts`

- No raw frame storage to disk.
- In-memory processing only.
- Alert outputs as structured JSON.
- Optional live visualization with keypoints, IDs, and risk labels.
- Prototype mode runs without bed zones by default.
- Stabilized risk scoring with temporal smoothing/hysteresis to reduce alert flicker.
- Clinician-facing event labels: `fall_detected`, `instability_risk`, `inactivity_risk`, `bed_exit_risk`, `stable`.

## Tech Stack

- Python 3.10+
- OpenCV
- PyTorch
- Ultralytics YOLO
- NumPy + SciPy
- FastAPI (optional alert API)
- Pydantic

## Project Structure

- `run.py` - entrypoint
- `config/` - runtime profiles
- `pipelines/main_pipeline.py` - orchestrator
- `ingestion/` - webcam/video/RTSP input
- `detection/`, `pose/`, `tracking/`
- `features/`, `temporal/`, `risk/`
- `output/` - alerts + visualization
- `tests/` - unit tests

## Quick Start (PowerShell)

```powershell
cd "d:\AH Project"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
```

## 3 Starter Stacks (Different Pipelines)

### 1) Stack A - Ultralytics one-stage pose (fast prototype)

```powershell
python run.py --config config/stack1_ultralytics_pose_fast.yaml
```

### 2) Stack B - Ultralytics two-stage (detector + pose, balanced)

```powershell
python run.py --config config/stack2_ultralytics_twostage_balanced.yaml
```

### 3) Stack C - RTMO (SOTA track for ward-scale multi-person)

```powershell
python run.py --config config/stack3_rtmo_mmpose_ward6.yaml
```

### 4) Stack D - SOTA Temporal (Transformer Lite, low-latency)

```powershell
python run.py --config config/stack2_sota_transformer_realtime.yaml
```

For stacks A/B/C/D, open:
- `http://127.0.0.1:8000/dashboard`
- Includes live camera stream panel with pose overlays + live alert feed.

Stack B also writes training-ready feature logs to:
- `output/train_features_stack2.jsonl`

## Additional Profiles

### Standard baseline

```powershell
python run.py --config config/config.yaml
```

### Fall-only (target 30 FPS)

```powershell
python run.py --config config/fall_only_30fps.yaml
```

### Fast fall-only (optimized throughput)

```powershell
python run.py --config config/fall_only_fast.yaml
```

### Showcase clinical profile

```powershell
python run.py --config config/showcase_clinical.yaml
```

### Ward-6 RTMO profile

```powershell
python run.py --config config/ward6_rtmo_showcase.yaml
```

This profile uses one-stage multi-person pose (`RTMO`) and disables separate detection for better scaling when up to ~6 patients share one camera view.
- Minimal professional UI optimized for nurse-station triage.

## Download Model Weights

```powershell
python scripts/download_models.py
```

Downloads common Ultralytics weights into `models/`.

RTMO backend dependency install (one-time, for SOTA track):

```powershell
python -m pip install "numpy<2"
python -m pip install mmengine mmcv-lite mmpose xtcocotools
```

Windows note:
- Native Windows often fails for RTMO because `mmpose` may require `mmcv` ops (`mmcv._ext`) that are not reliably available without Linux/WSL builds.
- If you hit `Failed to build mmcv` or `No module named mmcv._ext`, use Stack A/B on Windows and run RTMO in WSL2/Ubuntu.

## Exit / Stop Controls

When visualization is enabled, you can stop with:

- `q`
- `Esc`
- Window close button `X`
- `Ctrl+C` in terminal

## Alerts Output

Alerts are written to JSONL logs (path depends on config):

- `output/alerts.jsonl`
- `output/alerts_fall_only.jsonl`
- `output/alerts_fall_fast.jsonl`
- `output/alerts_showcase.jsonl`

Sample event payload:

```json
{
  "track_id": 2,
  "risk_level": "HIGH",
  "confidence": 0.84,
  "event": "instability_risk",
  "timestamp": 1774937600.12,
  "reasons": ["lean_instability", "repeated_sit_stand_transitions"]
}
```

## Train Temporal Model (GRU or Transformer Lite)

1) Collect feature logs while running Stack B:

```powershell
python run.py --config config/stack2_ultralytics_twostage_balanced.yaml
```

2) Train a low-latency Transformer Lite model from collected logs:

```powershell
python scripts/train_temporal_gru.py `
  --input output/train_features_stack2.jsonl `
  --format frame `
  --model-type transformer_lite `
  --hidden-size 32 `
  --num-layers 1 `
  --attention-heads 2 `
  --ff-mult 2 `
  --sequence-len 16 `
  --epochs 25 `
  --batch-size 64 `
  --device cpu `
  --output models/temporal_transformer_lite.pt `
  --metrics-out output/temporal_train_metrics.json
```

3) Run with trained transformer temporal model:

```powershell
python run.py --config config/stack2_sota_transformer_realtime.yaml
```

Optional: train a GRU baseline instead:

```powershell
python scripts/train_temporal_gru.py `
  --input output/train_features_stack2.jsonl `
  --format frame `
  --model-type gru `
  --output models/temporal_gru.pt `
  --metrics-out output/temporal_train_metrics_gru.json

python run.py --config config/stack2_ultralytics_twostage_trained.yaml
```

Latency note:
- `temporal_model.infer_interval` controls how often temporal inference runs per track.
- Example: `infer_interval: 2` computes temporal ML every second frame and reuses cached probability in between to protect FPS.

Alert-fatigue control:
- `risk.high_consecutive_frames` and `risk.critical_consecutive_frames` require sustained severity before escalating.
- `output.alert_suppression.emit_on_level_change_only` reduces repeated duplicates.
- `output.alert_suppression.dedupe_window_sec` suppresses same-level repeats inside a short window.

Privacy display control:
- `output.visualization.privacy_mode` supports:
  - `none`
  - `person_blur` (recommended)
  - `person_pixelate`
  - `face_blur`
  - `face_pixelate`
  - `skeleton_only`
- Current Stack B/SOTA profiles default to `person_blur` for live UI anonymization.

## Run Tests

```powershell
python -m pytest -q
```

## Performance Notes

- `pipeline.fps` is capture target, not guaranteed processed FPS.
- Real processed FPS depends on model size, input resolution, and device.
- For higher speed, use `config/fall_only_fast.yaml` and reduce `pose.input_size` / camera resolution.

## Jetson / TensorRT Readiness

Use JetPack-provided CUDA, TensorRT, PyTorch, torchvision and OpenCV. Do not install
`requirements.txt` on Jetson because the PyPI `opencv-python`/PyTorch wheels can
replace NVIDIA's accelerated packages.

Recommended JetPack 6 bring-up:

```bash
cd /opt/ah-project
chmod +x scripts/setup_jetson.sh
./scripts/setup_jetson.sh

# Build the device-specific FP16 engine on the target Orin NX.
.venv-jetson/bin/python scripts/export_jetson_tensorrt.py \
  --weights models/yolo11n-pose.pt \
  --imgsz 384 \
  --precision fp16 \
  --output models/yolo11n-pose-fp16-384.engine

.venv-jetson/bin/python scripts/verify_jetson_runtime.py \
  --config config/jetson_orin_nx_rtsp.yaml
```

Set the RTSP URL outside Git and edit the codec/calibrated bed zones in
`config/jetson_orin_nx_rtsp.yaml`, then run:

```bash
export AH_RTSP_URL='rtsp://camera-host/path'
.venv-jetson/bin/python run.py --config config/jetson_orin_nx_rtsp.yaml
```

The Jetson profile:

- requires CUDA and never silently falls back to CPU;
- uses one-stage TensorRT pose to avoid duplicate detector inference;
- uses `nvv4l2decoder` through GStreamer for RTSP H.264/H.265 decode;
- keeps one pending frame, reconnects after runtime failures, and warms the engine before capture;
- batches JSONL writes, rate-limits/queues dashboard JPEG encoding, and disables training logs;
- binds the dashboard to localhost by default. Put authenticated TLS termination in front of it before remote access.

TensorRT engines are target-specific. Rebuild them on the deployed Orin NX after
JetPack/TensorRT upgrades. Start with FP16; use INT8 only with representative
calibration data and a measured pose/fall accuracy comparison.

For a systemd deployment, adapt `deploy/ah-project.service` to the actual user,
installation path and carrier-board environment before installing it.

Run a one-hour thermal/memory soak test after camera and model validation:

```bash
chmod +x scripts/soak_test_jetson.sh
scripts/soak_test_jetson.sh 3600 config/jetson_orin_nx_rtsp.yaml
```

Application metrics include capture/drop/reconnect counts and recent p50/p95/p99
module latency; the soak script records `tegrastats` alongside the application log.
