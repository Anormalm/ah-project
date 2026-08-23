# Edge-SOTA model policy for Jetson Orin NX 16 GB

The production choice is a pipeline decision, not a single leaderboard score. The
recommended default is one-stage YOLO26s-pose at 640 px in TensorRT FP16, followed
by pose-aware tracking and the rule/temporal risk ensemble. It avoids a second
detector pass and leaves memory headroom for decode, tracking, API output and a
small temporal network.

Review every model/code license for the intended deployment. In particular,
Ultralytics publishes its repository under AGPL-3.0 and offers separate licensing
options; this project does not alter those obligations.

## Supported deployment tiers

| Tier | Pose model | Runtime profile | Intended use |
| --- | --- | --- | --- |
| Balanced accuracy | YOLO26s-pose, FP16, 640 | `jetson_orin_nx_sota.yaml` | One ward camera at a 15 FPS target |
| Throughput | YOLO26n-pose, FP16, 512 | `jetson_orin_nx_sota_multistream.yaml` | Multiple cameras or a tighter power budget |
| Validated temporal | Balanced vision + local ST-GCN | `jetson_orin_nx_sota_stgcn.yaml` | Only after site-specific training and held-out validation |
| Crowded-scene alternative | RTMO-s/m | Existing `rtmo_mmpose` adapter; production needs MMDeploy TensorRT integration | Heavy occlusion where COCO validation is insufficient |
| High keypoint-accuracy alternative | detector + RTMPose-m | Two-stage custom integration | When pose AP matters more than end-to-end latency |

Official references:

- YOLO26 pose and TensorRT export: https://docs.ultralytics.com/tasks/pose/
- Jetson deployment guidance: https://docs.ultralytics.com/guides/nvidia-jetson/
- RTMPose benchmarks and checkpoints: https://github.com/open-mmlab/mmpose/tree/main/configs/body_2d_keypoint/rtmpose
- MMPose TensorRT support matrix: https://mmdeploy.readthedocs.io/en/stable/04-supported-codebases/mmpose.html
- Skeleton action baselines: https://github.com/open-mmlab/mmaction2

## Target-side selection

Build both candidate engines on the exact Orin, then benchmark representative ward
footage. Candidate order is accuracy priority; the first model whose p95 latency
meets the requested FPS is selected.

```bash
.venv-jetson/bin/python scripts/benchmark_pose_models.py \
  --models models/yolo26s-pose-fp16-640.engine models/yolo26n-pose-fp16-512.engine \
  --source /data/representative-ward.mp4 \
  --target-fps 15 \
  --iterations 200
```

Do not compare only synthetic black frames: they verify the runtime but do not
represent crowded-scene post-processing or thermally sustained performance.

## Temporal model policy

The included ST-GCN-style network operates on translation/scale-normalized COCO
skeletons plus joint motion. It is intentionally compact for edge inference. It is
an architecture, not a universally valid fall detector: train it on consented,
site-representative data with subject-disjoint train/validation/test splits.

Rule-generated labels are marked `weak_rule` and receive 0.25 training weight by
default. Training refuses a weak-only frame log unless `--allow-weak-only` is
explicitly supplied. Human/curated labels should drive release decisions;
otherwise the model merely learns to imitate the rules.

```bash
python scripts/train_temporal_gru.py \
  --input output/train_features.jsonl \
  --format frame \
  --model-type stgcn \
  --sequence-len 32 \
  --hidden-size 32 \
  --num-layers 3 \
  --output models/temporal_stgcn.pt \
  --device cuda
```

Before training, add reviewed labels with `label_source: human` or
`label_source: curated`. Add a stable `subject_id` when available; the loader then
keeps each subject wholly within one split. Without it, splits are grouped by
capture session, stream and track.

Keep `allow_ml_level_override: false` until sensitivity, false-alert rate, time to
detection and calibration are accepted on a held-out clinical evaluation set.

## When to move to DeepStream

The current Python runtime is suitable for a small number of streams. For larger
camera counts, DeepStream provides batched decode/inference and NvDCF tracking;
NVIDIA recommends the PVA backend for NvDCF on Jetson to reserve GPU capacity.
That is a separate metadata-pipeline integration, not a drop-in configuration flag.
