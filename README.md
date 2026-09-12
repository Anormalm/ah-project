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

### 5) Intel RealSense D435i - synchronized RGB, depth, IR, and IMU

Connect the camera directly over USB 3.x, then install the optional SDK binding and run:

```powershell
python -m pip install -r requirements-realsense.txt
python run.py --config config/d435i_all_feeds.yaml
```

The D435i profile performs pose inference once on RGB. Synchronized, color-aligned depth adds metric 3D motion features; accelerometer and gyroscope samples detect camera movement so motion-sensitive rules can be suppressed. Both infrared feeds are captured as auxiliary data without creating duplicate tracks or alerts.

The profile requires a verified USB 3.x link. A USB 2.x cable, hub, or port is rejected because full-feed capture is not reliable at that bandwidth.

For stacks A/B/C/D, open `http://127.0.0.1:8000/dashboard`. The D435i profile
uses `http://127.0.0.1:8001/dashboard` so it can run alongside another local
service already using port 8000.
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

1) Collect synchronized feature logs and dashboard feedback with the D435i
profile:

```powershell
python run.py --config config/d435i_all_feeds.yaml
```

This writes `output/train_features_d435i.jsonl` and
`output/feedback_d435i.jsonl`. Stack B can still collect weakly labeled feature
logs, but its configuration needs an `output.feedback_log_path` before the
dashboard can save reviews.

Feature-log predictions are stored as `weak_label`, not `label`: they are useful
for bootstrapping, but they are not ground truth. Each process run has a stable
`session_id`, and the log also retains depth coverage, camera-motion quality
signals, and available metric 3D position/velocity/acceleration. These extra
fields are kept for review and weighting; the temporal model input remains the
original five values (`speed`, `vy`, `acc`, `lean`, and `posture`).

For human-supervised training, enable `output.feedback_log_path` and use the
dashboard's **Confirm Fall**, **False Alarm**, and **Mark current activity normal** controls.
They append annotations such as:

```json
{"stream_id":"d435i","track_id":3,"timestamp":1774937600.12,"label":"confirmed_fall","annotated_at":1774937604.82}
```

- Dashboard labels map as follows: `confirmed_fall` is positive;
  `false_alarm` and `non_fall_activity` are negative; `unclear` is skipped.
  Numeric `0`/`1` and `no_fall`/`fall` are also accepted for imported reviews.
- `track_id: -1` applies the annotation to the nearest eligible window for every
  active track in the matching session and stream.
- `session_id` and `stream_id` may be omitted as wildcards, but including both
  prevents accidental matches across camera runs.
- Rows marked `reviewed: false`, or labels/statuses such as `unclear` and
  `unreviewed`, are skipped. Once `--feedback` is supplied, unmatched windows
  and weak labels are excluded from training.
- `weight` is optional, positive, and defaults to `1.0`.

The annotation timestamp is matched to the closest sequence-window end within
the inclusive tolerance set by `--feedback-max-seconds` (default: 2 seconds).

2) Train a low-latency Transformer Lite model from collected logs:

```powershell
python scripts/train_temporal_gru.py `
  --input output/train_features_d435i.jsonl `
  --format frame `
  --model-type transformer_lite `
  --hidden-size 32 `
  --num-layers 1 `
  --attention-heads 2 `
  --ff-mult 2 `
  --sequence-len 16 `
  --feedback output/feedback_d435i.jsonl `
  --feedback-max-seconds 2.0 `
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

### Train from CAUCAFall

The public-data path expects the official CAUCAFall `Subject.N/<Activity>`
tree with each AVI and its consolidated `annotations.jsonl`. The converter
uses the supplied person boxes as padded pose crops and reports any labelled
fall clip that still produced no usable positive sequence. Build a
subject-isolated manifest, extract the same five motion features used at
runtime, then train with an explicit validation file:

```powershell
python scripts/build_caucafall_manifest.py `
  --dataset-root data/raw/caucafall_v5 `
  --output config/caucafall_manifest.json

python scripts/prepare_public_dataset.py `
  --manifest config/caucafall_manifest.json `
  --output-dir data/processed `
  --device cpu

python scripts/train_temporal_gru.py `
  --input data/processed/caucafall_train.jsonl `
  --val-input data/processed/caucafall_val.jsonl `
  --format sequence `
  --sequence-len 10 `
  --model-fps 20 `
  --model-type gru `
  --epochs 25 `
  --max-fpr 0.0 `
  --output models/temporal_gru_caucafall.pt `
  --metrics-out output/temporal_gru_caucafall_metrics.json
```

For a final unbiased score, reserve a third set of whole subjects and pass its
prepared JSONL with `--test-input`; it is loaded only after early stopping and
threshold calibration. `--max-fpr` deliberately refuses a random window split
and therefore requires `--val-input`.

CAUCAFall is distributed under CC BY 4.0; retain the downloaded `SOURCE.json`
and attribution metadata. Public-data models should initially run with
`risk.ml_weight: 0.0` so their probabilities are logged without changing
depth-confirmed D435i alerts.

Latency note:
- `temporal_model.infer_interval` controls how often temporal inference runs per track.
- Example: `infer_interval: 2` computes temporal ML every second frame and reuses cached probability in between to protect FPS.

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
