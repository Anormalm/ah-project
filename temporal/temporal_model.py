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
        checkpoint = torch.load(model_path, map_location=device)

        meta = TemporalModelMeta(model_type=force_model_type or "gru")
        state_dict = checkpoint
        feature_mean = None
        feature_std = None
        decision_threshold = None
        model_fps = None
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
            )
            state_dict = checkpoint["model_state_dict"]
            feature_mean = checkpoint.get("feature_mean")
            feature_std = checkpoint.get("feature_std")
            decision_threshold = checkpoint.get("decision_threshold")
            model_fps = checkpoint.get("model_fps")

        self.model = _build_torch_model(torch, meta).to(device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.device = device
        self.meta = meta
        self.required_sequence_len = meta.sequence_len
        self.decision_threshold = None if decision_threshold is None else float(decision_threshold)
        self.model_fps = None if model_fps is None else float(model_fps)

        if feature_mean is None or feature_std is None:
            self._feature_mean = None
            self._feature_std = None
        else:
            self._feature_mean = torch.tensor(feature_mean, dtype=torch.float32, device=device).reshape(1, 1, -1)
            self._feature_std = torch.tensor(feature_std, dtype=torch.float32, device=device).reshape(1, 1, -1)

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


class TemporalRiskModel:
    def __init__(
        self,
        backend: InferenceEngine,
        sequence_len: int = 16,
        infer_interval: int = 1,
        min_infer_steps: int | None = None,
        model_fps: float | None = None,
    ) -> None:
        self.backend = backend
        checkpoint_sequence_len = getattr(backend, "required_sequence_len", None)
        if checkpoint_sequence_len is None:
            self.sequence_len = int(sequence_len)
            default_min_infer_steps = 2
        else:
            self.sequence_len = int(checkpoint_sequence_len)
            if self.sequence_len < 2:
                raise ValueError("checkpoint sequence_len must be at least 2")
            default_min_infer_steps = self.sequence_len

        self.infer_interval = max(1, int(infer_interval))
        requested_min_infer_steps = default_min_infer_steps if min_infer_steps is None else int(min_infer_steps)
        self.min_infer_steps = max(default_min_infer_steps, requested_min_infer_steps)

        checkpoint_model_fps = getattr(backend, "model_fps", None)
        selected_model_fps = checkpoint_model_fps if model_fps is None else model_fps
        if selected_model_fps is not None:
            selected_model_fps = float(selected_model_fps)
            if not np.isfinite(selected_model_fps) or selected_model_fps <= 0.0:
                raise ValueError("model_fps must be a finite positive number")
        self.model_fps = selected_model_fps
        self.decision_threshold = getattr(backend, "decision_threshold", None)
        self._infer_counter: dict[int, int] = {}
        self._cached_prob: dict[int, float] = {}

    @property
    def required_history_seconds(self) -> float | None:
        if self.model_fps is None:
            return None
        return (self.sequence_len - 1) / self.model_fps

    @staticmethod
    def _posture_to_scalar(posture: str) -> float:
        mapping = {"unknown": 0.0, "standing": 0.2, "sitting": 0.6, "lying": 1.0}
        return mapping.get(posture, 0.0)

    def _sample_sequence(self, sequence: Sequence[FeatureVector]) -> list[FeatureVector]:
        samples = list(sequence)
        if self.model_fps is None:
            return samples[-self.sequence_len :]
        if len(samples) < self.sequence_len:
            return []

        timestamps = np.asarray([sample.timestamp for sample in samples], dtype=np.float64)
        if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) < 0.0):
            return []

        end_timestamp = float(timestamps[-1])
        start_timestamp = end_timestamp - float(self.required_history_seconds)
        if timestamps[0] > start_timestamp + 1e-9:
            return []

        target_timestamps = start_timestamp + np.arange(self.sequence_len, dtype=np.float64) / self.model_fps
        right_indices = np.searchsorted(timestamps, target_timestamps, side="left")
        right_indices = np.clip(right_indices, 0, len(samples) - 1)
        left_indices = np.maximum(right_indices - 1, 0)
        choose_left = (
            np.abs(target_timestamps - timestamps[left_indices])
            <= np.abs(timestamps[right_indices] - target_timestamps)
        )
        nearest_indices = np.where(choose_left, left_indices, right_indices)
        return [samples[int(index)] for index in nearest_indices]

    def _to_features(self, sequence: Sequence[FeatureVector]) -> np.ndarray:
        rows = []
        for f in self._sample_sequence(sequence):
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

    def predict(self, sequence: Sequence[FeatureVector], track_id: int | None = None) -> float:
        features = self._to_features(sequence)
        if features.shape[0] < self.min_infer_steps:
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
