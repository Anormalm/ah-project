from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from features.feature_extractor import FeatureExtractor
from run import load_config
from temporal.temporal_model import SkeletonSTGCNRiskNet, TemporalModelMeta, TorchSTGCNInferenceEngine
from temporal.trainer import TemporalRiskTrainer, TrainerConfig
from temporal.training_data import load_frame_log_dataset
from tracking.tracker import PoseAwareKalmanTracker
from utils.schemas import Detection, PoseResult, TrackPose


def _pose(bbox: tuple[float, float, float, float], confidence: float = 0.9) -> PoseResult:
    x1, y1, x2, y2 = bbox
    keypoints = []
    for index in range(17):
        x = x1 + (x2 - x1) * (0.35 + 0.03 * (index % 4))
        y = y1 + (y2 - y1) * (0.08 + 0.05 * index)
        keypoints.append((float(x), float(y), confidence))
    return PoseResult(bbox=bbox, keypoints=keypoints, confidence=confidence)


def test_sota_multistream_profile_inherits_balanced_profile(monkeypatch) -> None:
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("AH_RTSP_URL", "rtsp://camera.local/live")
    cfg = load_config(str(root / "config" / "jetson_orin_nx_sota_multistream.yaml"))
    assert cfg["pipeline"]["require_accelerator"] is True
    assert cfg["pipeline"]["fps"] == 20
    assert cfg["tracking"]["backend"] == "pose_kalman"
    assert cfg["pose"]["input_size"] == 512
    assert Path(cfg["pose"]["model_path"]).is_absolute()


def test_pose_tracker_uses_low_confidence_detection_to_preserve_identity() -> None:
    tracker = PoseAwareKalmanTracker(
        track_high_thresh=0.35,
        track_low_thresh=0.10,
        new_track_thresh=0.40,
        max_center_distance=2.0,
    )
    box1 = (20.0, 20.0, 80.0, 160.0)
    first = tracker.update([Detection(bbox=box1, confidence=0.9)], [_pose(box1)], timestamp=1.0)
    assert len(first) == 1

    box2 = (25.0, 22.0, 85.0, 162.0)
    second = tracker.update([Detection(bbox=box2, confidence=0.2)], [_pose(box2, 0.2)], timestamp=1.1)
    assert len(second) == 1
    assert second[0].track_id == first[0].track_id


def test_pose_tracker_does_not_spawn_from_low_confidence_detection() -> None:
    tracker = PoseAwareKalmanTracker(track_high_thresh=0.35, track_low_thresh=0.10, new_track_thresh=0.40)
    bbox = (10.0, 10.0, 70.0, 150.0)
    assert tracker.update([Detection(bbox=bbox, confidence=0.2)], [_pose(bbox, 0.2)], timestamp=1.0) == []


def test_skeleton_normalization_is_translation_and_scale_invariant() -> None:
    base = np.asarray(_pose((10.0, 20.0, 110.0, 220.0)).keypoints, dtype=np.float32)
    transformed = base.copy()
    transformed[:, :2] = transformed[:, :2] * 1.8 + np.array([200.0, 75.0], dtype=np.float32)
    extractor = FeatureExtractor(min_kpt_conf=0.2)
    f1 = extractor.extract(TrackPose(track_id=1, keypoints=[tuple(row) for row in base], timestamp=1.0))
    f2 = extractor.extract(TrackPose(track_id=2, keypoints=[tuple(row) for row in transformed], timestamp=1.0))
    assert np.allclose(
        np.asarray(f1.normalized_keypoints)[:, :2],
        np.asarray(f2.normalized_keypoints)[:, :2],
        atol=1e-5,
    )


def test_motion_center_ignores_peripheral_visibility_changes() -> None:
    keypoints = np.asarray(_pose((10.0, 20.0, 110.0, 220.0)).keypoints, dtype=np.float32)
    changed = keypoints.copy()
    # Face, wrists, knees and ankles can flicker at frame edges; torso stays put.
    for idx in (0, 1, 2, 3, 4, 7, 8, 13, 14, 15, 16):
        changed[idx, :2] += np.array([150.0, 120.0], dtype=np.float32)
        changed[idx, 2] = 0.0

    extractor = FeatureExtractor(min_kpt_conf=0.2, center_ema_alpha=1.0)
    first = extractor.extract(TrackPose(track_id=9, keypoints=[tuple(row) for row in keypoints], timestamp=1.0))
    second = extractor.extract(TrackPose(track_id=9, keypoints=[tuple(row) for row in changed], timestamp=1.04))

    assert np.allclose(first.center_of_mass, second.center_of_mass, atol=1e-5)
    assert np.allclose(second.velocity, (0.0, 0.0), atol=1e-5)


def test_pose_quality_penalizes_skeleton_fragments() -> None:
    full = np.asarray(_pose((10.0, 20.0, 110.0, 220.0), confidence=0.9).keypoints, dtype=np.float32)
    fragment = full.copy()
    fragment[7:, 2] = 0.0
    extractor = FeatureExtractor(min_kpt_conf=0.25)

    full_feature = extractor.extract(TrackPose(track_id=11, keypoints=[tuple(row) for row in full], timestamp=1.0))
    fragment_feature = extractor.extract(
        TrackPose(track_id=12, keypoints=[tuple(row) for row in fragment], timestamp=1.0)
    )

    assert full_feature.pose_quality > 0.8
    assert fragment_feature.pose_quality < 0.4


def test_skeleton_training_loader_and_weak_label_weight(tmp_path: Path) -> None:
    path = tmp_path / "skeleton.jsonl"
    rows = []
    for index in range(8):
        rows.append(
            {
                "stream_id": "ward",
                "track_id": 4,
                "timestamp": float(index),
                "normalized_keypoints": _pose((10.0, 20.0, 110.0, 220.0)).keypoints,
                "risk_level": "HIGH" if index >= 4 else "LOW",
                "label_source": "weak_rule",
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    x, y, weights = load_frame_log_dataset(
        str(path), sequence_len=4, feature_mode="skeleton", weak_label_weight=0.2
    )
    assert x.shape == (5, 4, 17, 3)
    assert y.shape == (5,)
    assert np.allclose(weights, 0.2)


def test_stgcn_forward_shape() -> None:
    torch = pytest.importorskip("torch")
    model = SkeletonSTGCNRiskNet(torch, hidden_size=8, num_layers=1, num_joints=17).model
    inputs = torch.rand(2, 8, 17, 3)
    outputs = model(inputs)
    assert tuple(outputs.shape) == (2, 1)
    assert bool(torch.all((outputs >= 0.0) & (outputs <= 1.0)))


def test_stgcn_normalization_preserves_pose_confidence() -> None:
    pytest.importorskip("torch")
    trainer = TemporalRiskTrainer(TrainerConfig(model_type="stgcn", epochs=1))
    inputs = np.zeros((2, 4, 17, 3), dtype=np.float32)
    inputs[..., 2] = 0.75
    normalized, _, mean, std = trainer._normalize(inputs, inputs.copy())
    assert mean[2] == 0.0
    assert std[2] == 1.0
    assert np.allclose(normalized[..., 2], 0.75)


def test_stgcn_checkpoint_round_trip(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    meta = TemporalModelMeta(model_type="stgcn", hidden_size=8, num_layers=1, sequence_len=8)
    model = SkeletonSTGCNRiskNet(torch, hidden_size=8, num_layers=1, num_joints=17).model
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_type": "stgcn",
        "input_size": 3,
        "hidden_size": 8,
        "num_layers": 1,
        "sequence_len": 8,
        "num_joints": 17,
        "dropout": meta.dropout,
        "feature_mean": [0.0, 0.0, 0.0],
        "feature_std": [1.0, 1.0, 1.0],
    }
    path = tmp_path / "stgcn.pt"
    torch.save(checkpoint, path)
    engine = TorchSTGCNInferenceEngine(str(path), device="cpu")
    inputs = np.zeros((8, 17, 3), dtype=np.float32)
    inputs[..., 2] = 0.9
    probability = engine.predict(inputs)
    assert 0.0 <= probability <= 1.0
