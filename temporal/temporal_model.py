from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from models.inference_engine import InferenceEngine
from utils.schemas import FeatureVector


@dataclass(frozen=True)
class TemporalModelMeta:
    input_size: int = 5
    hidden_size: int = 16
    num_layers: int = 1
    sequence_len: int = 16
    model_type: str = "gru"
    attention_heads: int = 2
    ff_mult: int = 2
    num_joints: int = 17
    dropout: float = 0.1


class GRURiskNet:
    def __init__(self, torch_module, input_size: int = 5, hidden_size: int = 16, num_layers: int = 1) -> None:
        nn = torch_module.nn

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x):
                out, _ = self.gru(x)
                logits = self.fc(out[:, -1, :])
                return torch_module.sigmoid(logits)

        self.model = _Model()


class TinyTransformerRiskNet:
    def __init__(
        self,
        torch_module,
        input_size: int = 5,
        hidden_size: int = 32,
        num_layers: int = 1,
        attention_heads: int = 2,
        ff_mult: int = 2,
    ) -> None:
        nn = torch_module.nn

        nhead = 1
        for cand in [attention_heads, 4, 2, 1]:
            if cand >= 1 and hidden_size % cand == 0:
                nhead = cand
                break

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.in_proj = nn.Linear(input_size, hidden_size)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=nhead,
                    dim_feedforward=max(hidden_size * ff_mult, hidden_size),
                    dropout=0.1,
                    activation="gelu",
                    batch_first=True,
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, num_layers))
                self.norm = nn.LayerNorm(hidden_size)
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x):
                z = self.in_proj(x)
                z = self.encoder(z)
                z = self.norm(z[:, -1, :])
                logits = self.fc(z)
                return torch_module.sigmoid(logits)

        self.model = _Model()


class SkeletonSTGCNRiskNet:
    """Compact ST-GCN-style classifier for normalized COCO skeleton windows."""

    _COCO_EDGES = (
        (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
        (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13),
        (13, 15), (12, 14), (14, 16),
    )

    def __init__(
        self,
        torch_module,
        hidden_size: int = 32,
        num_layers: int = 3,
        num_joints: int = 17,
        dropout: float = 0.1,
    ) -> None:
        nn = torch_module.nn
        adjacency = np.eye(num_joints, dtype=np.float32)
        for source, target in self._COCO_EDGES:
            if source < num_joints and target < num_joints:
                adjacency[source, target] = 1.0
                adjacency[target, source] = 1.0
        degree = np.clip(adjacency.sum(axis=1), 1.0, None)
        inv_sqrt = np.diag(1.0 / np.sqrt(degree))
        adjacency = inv_sqrt @ adjacency @ inv_sqrt

        class _GraphTemporalBlock(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.temporal = nn.Conv2d(
                    hidden_size,
                    hidden_size,
                    kernel_size=(3, 1),
                    padding=(1, 0),
                    groups=hidden_size,
                    bias=False,
                )
                self.channel = nn.Conv2d(hidden_size, hidden_size, kernel_size=1, bias=False)
                self.norm = nn.BatchNorm2d(hidden_size)
                self.activation = nn.GELU()
                self.dropout = nn.Dropout(dropout)

            def forward(self, x, graph):
                spatial = torch_module.einsum("btvc,vw->btwc", x, graph)
                z = spatial.permute(0, 3, 1, 2)
                z = self.dropout(self.activation(self.norm(self.channel(self.temporal(z)))))
                return x + z.permute(0, 2, 3, 1)

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("adjacency", torch_module.tensor(adjacency, dtype=torch_module.float32))
                self.input_projection = nn.Linear(5, hidden_size)
                self.blocks = nn.ModuleList([_GraphTemporalBlock() for _ in range(max(1, num_layers))])
                self.norm = nn.LayerNorm(hidden_size)
                self.head = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, 1),
                )

            def forward(self, x):
                if x.ndim != 4:
                    raise ValueError("ST-GCN input must have shape [batch,time,joints,channels]")
                coordinates = x[..., :2]
                confidence = x[..., 2:3].clamp(0.0, 1.0)
                motion = torch_module.zeros_like(coordinates)
                motion[:, 1:] = coordinates[:, 1:] - coordinates[:, :-1]
                z = self.input_projection(torch_module.cat([coordinates, confidence, motion], dim=-1))
                for block in self.blocks:
                    z = block(z, self.adjacency)
                z = self.norm(z)
                weights = confidence.expand_as(z)
                pooled = (z * weights).sum(dim=(1, 2)) / weights.sum(dim=(1, 2)).clamp_min(1e-6)
                return torch_module.sigmoid(self.head(pooled))

        self.model = _Model()


class HeuristicTemporalEngine(InferenceEngine):
    def predict(self, inputs: np.ndarray) -> float:
        features = inputs
        if features.size == 0:
            return 0.0
        speed = np.abs(features[:, 0]).mean()
        vy = np.abs(features[:, 1]).mean()
        acc = np.abs(features[:, 2]).mean()
        lean = features[:, 3].mean()
        posture = features[:, 4].mean()
        raw = 0.0020 * speed + 0.0025 * vy + 0.0009 * acc + 0.018 * lean + 0.25 * posture
        prob = 1.0 / (1.0 + np.exp(-raw + 1.2))
        return float(np.clip(prob, 0.0, 1.0))


class NullTemporalEngine(InferenceEngine):
    def predict(self, inputs: np.ndarray) -> float:
        _ = inputs
        return 0.0


def _build_torch_model(torch_module, meta: TemporalModelMeta):
    if meta.model_type == "stgcn":
        return SkeletonSTGCNRiskNet(
            torch_module=torch_module,
            hidden_size=meta.hidden_size,
            num_layers=meta.num_layers,
            num_joints=meta.num_joints,
            dropout=meta.dropout,
        ).model
    if meta.model_type == "transformer_lite":
        return TinyTransformerRiskNet(
            torch_module=torch_module,
            input_size=meta.input_size,
            hidden_size=meta.hidden_size,
            num_layers=meta.num_layers,
            attention_heads=meta.attention_heads,
            ff_mult=meta.ff_mult,
        ).model
    return GRURiskNet(
        torch_module=torch_module,
        input_size=meta.input_size,
        hidden_size=meta.hidden_size,
        num_layers=meta.num_layers,
    ).model


class TorchTemporalInferenceEngine(InferenceEngine):
    def __init__(self, model_path: str, device: str = "cpu", force_model_type: str | None = None) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch is required for torch temporal inference backends") from exc

        self._torch = torch
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(model_path, map_location=device)

        meta = TemporalModelMeta(model_type=force_model_type or "gru")
        state_dict = checkpoint
        feature_mean = None
        feature_std = None
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model_type = str(checkpoint.get("model_type", meta.model_type))
            if force_model_type is not None:
                model_type = force_model_type
            meta = TemporalModelMeta(
                input_size=int(checkpoint.get("input_size", 5)),
                hidden_size=int(checkpoint.get("hidden_size", 16)),
                num_layers=int(checkpoint.get("num_layers", 1)),
                sequence_len=int(checkpoint.get("sequence_len", 16)),
                model_type=model_type,
                attention_heads=int(checkpoint.get("attention_heads", 2)),
                ff_mult=int(checkpoint.get("ff_mult", 2)),
                num_joints=int(checkpoint.get("num_joints", 17)),
                dropout=float(checkpoint.get("dropout", 0.1)),
            )
            state_dict = checkpoint["model_state_dict"]
            feature_mean = checkpoint.get("feature_mean")
            feature_std = checkpoint.get("feature_std")

        self.model = _build_torch_model(torch, meta).to(device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.device = device
        self.meta = meta
        self.input_kind = "skeleton" if meta.model_type == "stgcn" else "biomechanics"

        if feature_mean is None or feature_std is None:
            self._feature_mean = None
            self._feature_std = None
        else:
            stat_shape = (1, 1, 1, -1) if self.input_kind == "skeleton" else (1, 1, -1)
            self._feature_mean = torch.tensor(feature_mean, dtype=torch.float32, device=device).reshape(stat_shape)
            self._feature_std = torch.tensor(feature_std, dtype=torch.float32, device=device).reshape(stat_shape)

    def predict(self, inputs: np.ndarray) -> float:
        features = inputs
        tensor = self._torch.from_numpy(features.astype(np.float32)).unsqueeze(0).to(self.device)
        if self._feature_mean is not None and self._feature_std is not None:
            tensor = (tensor - self._feature_mean) / self._feature_std.clamp_min(1e-6)
        with self._torch.no_grad():
            prob = self.model(tensor).squeeze().detach().cpu().item()
        return float(np.clip(prob, 0.0, 1.0))


class TorchGRUInferenceEngine(TorchTemporalInferenceEngine):
    def __init__(self, model_path: str, device: str = "cpu") -> None:
        super().__init__(model_path=model_path, device=device, force_model_type="gru")


class TorchTransformerLiteInferenceEngine(TorchTemporalInferenceEngine):
    def __init__(self, model_path: str, device: str = "cpu") -> None:
        super().__init__(model_path=model_path, device=device, force_model_type="transformer_lite")


class TorchSTGCNInferenceEngine(TorchTemporalInferenceEngine):
    def __init__(self, model_path: str, device: str = "cpu") -> None:
        super().__init__(model_path=model_path, device=device, force_model_type="stgcn")


class TemporalRiskModel:
    def __init__(
        self,
        backend: InferenceEngine,
        sequence_len: int = 16,
        infer_interval: int = 1,
        min_infer_steps: int = 2,
        min_pose_quality: float = 0.20,
    ) -> None:
        self.backend = backend
        self.sequence_len = sequence_len
        self.infer_interval = max(1, int(infer_interval))
        self.min_infer_steps = max(2, int(min_infer_steps))
        self.min_pose_quality = float(np.clip(min_pose_quality, 0.0, 1.0))
        self._infer_counter: dict[int, int] = {}
        self._cached_prob: dict[int, float] = {}

    def remove_track(self, track_id: int) -> None:
        tid = int(track_id)
        self._infer_counter.pop(tid, None)
        self._cached_prob.pop(tid, None)

    @staticmethod
    def _posture_to_scalar(posture: str) -> float:
        mapping = {"unknown": 0.0, "standing": 0.2, "sitting": 0.6, "lying": 1.0}
        return mapping.get(posture, 0.0)

    def _to_features(self, sequence: Sequence[FeatureVector]) -> np.ndarray:
        rows = []
        for f in sequence[-self.sequence_len :]:
            speed = float(np.hypot(f.velocity[0], f.velocity[1]))
            acc = float(np.hypot(f.acceleration[0], f.acceleration[1]))
            rows.append([
                speed,
                f.velocity[1],
                acc,
                f.lean_angle,
                self._posture_to_scalar(f.posture),
            ])
        if not rows:
            return np.zeros((0, 5), dtype=np.float32)
        return np.array(rows, dtype=np.float32)

    def _to_skeleton_features(self, sequence: Sequence[FeatureVector]) -> np.ndarray:
        rows: list[np.ndarray] = []
        num_joints = int(getattr(getattr(self.backend, "meta", None), "num_joints", 17))
        for feature in sequence[-self.sequence_len :]:
            keypoints = np.asarray(feature.normalized_keypoints, dtype=np.float32)
            if keypoints.ndim != 2 or keypoints.shape[1] != 3:
                keypoints = np.zeros((num_joints, 3), dtype=np.float32)
            if keypoints.shape[0] < num_joints:
                keypoints = np.pad(keypoints, ((0, num_joints - keypoints.shape[0]), (0, 0)))
            rows.append(keypoints[:num_joints])
        if not rows:
            return np.zeros((0, num_joints, 3), dtype=np.float32)
        return np.stack(rows).astype(np.float32)

    def predict(self, sequence: Sequence[FeatureVector], track_id: int | None = None) -> float:
        input_kind = str(getattr(self.backend, "input_kind", "biomechanics"))
        features = self._to_skeleton_features(sequence) if input_kind == "skeleton" else self._to_features(sequence)
        if features.shape[0] < self.min_infer_steps:
            return 0.0
        if input_kind == "skeleton" and float(features[..., 2].mean()) < self.min_pose_quality:
            return 0.0

        if track_id is None or self.infer_interval <= 1:
            return self.backend.predict(features)

        count = self._infer_counter.get(track_id, 0) + 1
        self._infer_counter[track_id] = count

        if count % self.infer_interval != 0 and track_id in self._cached_prob:
            return self._cached_prob[track_id]

        prob = float(self.backend.predict(features))
        self._cached_prob[track_id] = prob
        return prob
