# AH Project - Real-Time Pose-Based Risk Detection

Production-grade, modular pipeline for real-time multi-person pose-based risk detection with privacy-first processing.

## What This Does

Pipeline flow:

`Camera/Video -> Detection -> Pose -> Tracking -> Feature Extraction -> Temporal Logic -> Risk Scoring -> Alerts`

- No raw frame storage to disk.
- In-memory processing only.
- Alert outputs as structured JSON.
- Optional live visualization with keypoints, IDs, and risk labels.

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
python -m pip install -r requirements.txt
python run.py --config config/config.yaml
```

## Config Profiles

### 1) Standard profile

```powershell
python run.py --config config/config.yaml
```

### 2) Fall-only (target 30 FPS)

```powershell
python run.py --config config/fall_only_30fps.yaml
```

### 3) Fast fall-only (optimized throughput)

```powershell
python run.py --config config/fall_only_fast.yaml
```

## Download Model Weights

```powershell
python scripts/download_models.py
```

Downloads common Ultralytics weights into `models/`.

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
