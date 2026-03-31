from __future__ import annotations

from typing import Sequence

import numpy as np

from models.inference_engine import InferenceEngine
from utils.schemas import FeatureVector


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
            import torch.nn as nn
        except ImportError as exc:
            raise RuntimeError("torch is required for TorchGRUInferenceEngine") from exc

        self._torch = torch

        class _GRU(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gru = nn.GRU(input_size=5, hidden_size=16, num_layers=1, batch_first=True)
                self.fc = nn.Linear(16, 1)

            def forward(self, x):
                out, _ = self.gru(x)
                logits = self.fc(out[:, -1, :])
                return torch.sigmoid(logits)

        self.model = _GRU().to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.device = device

    def predict(self, inputs: np.ndarray) -> float:
        features = inputs
        tensor = self._torch.from_numpy(features.astype(np.float32)).unsqueeze(0).to(self.device)
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

