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

For stacks A/B/C, open:
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

## Train The Temporal GRU (PowerShell)

1) Collect feature logs while running Stack B:

```powershell
python run.py --config config/stack2_ultralytics_twostage_balanced.yaml
```

2) Train GRU from collected logs:

```powershell
python scripts/train_temporal_gru.py `
  --input output/train_features_stack2.jsonl `
  --format frame `
  --sequence-len 16 `
  --epochs 25 `
  --batch-size 64 `
  --device cpu `
  --output models/temporal_gru.pt `
  --metrics-out output/temporal_train_metrics.json
```

3) Run with trained temporal model:

```powershell
python run.py --config config/stack2_ultralytics_twostage_trained.yaml
```

## Run Tests

```powershell
python -m pytest -q
```

## Performance Notes

- `pipeline.fps` is capture target, not guaranteed processed FPS.
- Real processed FPS depends on model size, input resolution, and device.
- For higher speed, use `config/fall_only_fast.yaml` and reduce `pose.input_size` / camera resolution.

## Jetson / TensorRT Readiness

The code separates inference backends from pipeline logic. You can swap:

- PyTorch -> ONNX -> TensorRT

without changing orchestration modules.
