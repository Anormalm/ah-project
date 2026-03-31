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


class GRURiskNet:
    def __init__(self, torch_module, input_size: int = 5, hidden_size: int = 16, num_layers: int = 1) -> None:
        self._torch = torch_module
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


class TorchGRUInferenceEngine(InferenceEngine):
    def __init__(self, model_path: str, device: str = "cpu") -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch is required for TorchGRUInferenceEngine") from exc

        self._torch = torch
        checkpoint = torch.load(model_path, map_location=device)

        meta = TemporalModelMeta()
        state_dict = checkpoint
        feature_mean = None
        feature_std = None
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            meta = TemporalModelMeta(
                input_size=int(checkpoint.get("input_size", 5)),
                hidden_size=int(checkpoint.get("hidden_size", 16)),
                num_layers=int(checkpoint.get("num_layers", 1)),
                sequence_len=int(checkpoint.get("sequence_len", 16)),
            )
            feature_mean = checkpoint.get("feature_mean")
            feature_std = checkpoint.get("feature_std")

        wrapper = GRURiskNet(
            torch_module=torch,
            input_size=meta.input_size,
            hidden_size=meta.hidden_size,
            num_layers=meta.num_layers,
        )
        self.model = wrapper.model.to(device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.device = device
        self.meta = meta

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


class TemporalRiskModel:
    def __init__(self, backend: InferenceEngine, sequence_len: int = 16) -> None:
        self.backend = backend
        self.sequence_len = sequence_len

    @staticmethod
    def _posture_to_scalar(posture: str) -> float:
        mapping = {"unknown": 0.0, "standing": 0.2, "sitting": 0.6, "lying": 1.0}
        return mapping.get(posture, 0.0)

    def _to_features(self, sequence: Sequence[FeatureVector]) -> np.ndarray:
        rows = []
        for f in sequence[-self.sequence_len :]:
            speed = float(np.hypot(f.velocity[0], f.velocity[1]))
            acc = float(np.hypot(f.acceleration[0], f.acceleration[1]))
            rows.append(
                [
                    speed,
                    f.velocity[1],
                    acc,
                    f.lean_angle,
                    self._posture_to_scalar(f.posture),
                ]
            )
        if not rows:
            return np.zeros((0, 5), dtype=np.float32)
        return np.array(rows, dtype=np.float32)

    def predict(self, sequence: Sequence[FeatureVector]) -> float:
        features = self._to_features(sequence)
        if features.shape[0] < 2:
            return 0.0
        return self.backend.predict(features)
