from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from temporal.temporal_model import GRURiskNet, TemporalRiskModel, TorchGRUInferenceEngine
from utils.schemas import FeatureVector


class _RecordingBackend:
    def __init__(
        self,
        *,
        required_sequence_len: int | None = None,
        model_fps: float | None = None,
    ) -> None:
        self.calls: list[np.ndarray] = []
        if required_sequence_len is not None:
            self.required_sequence_len = required_sequence_len
        if model_fps is not None:
            self.model_fps = model_fps

    def predict(self, inputs: np.ndarray) -> float:
        self.calls.append(inputs.copy())
        return 0.75


def _feature(timestamp: float) -> FeatureVector:
    return FeatureVector(
        track_id=1,
        timestamp=timestamp,
        center_of_mass=(0.0, 0.0),
        velocity=(timestamp + 1.0, 0.0),
        acceleration=(0.0, 0.0),
        joint_angles={},
        posture="standing",
        bed_zone_distance=0.0,
        lean_angle=timestamp,
    )


def test_checkpoint_sequence_len_blocks_prefix_inference() -> None:
    backend = _RecordingBackend(required_sequence_len=10)
    model = TemporalRiskModel(
        backend=backend,
        sequence_len=4,
        min_infer_steps=2,
    )

    sequence = [_feature(float(index)) for index in range(10)]
    for prefix_len in range(2, 10):
        assert model.predict(sequence[:prefix_len]) == 0.0

    assert backend.calls == []
    assert model.sequence_len == 10
    assert model.min_infer_steps == 10
    assert model.predict(sequence) == pytest.approx(0.75)
    assert backend.calls[0].shape == (10, 5)


def test_non_checkpoint_backend_keeps_legacy_two_step_default() -> None:
    backend = _RecordingBackend()
    model = TemporalRiskModel(backend=backend, sequence_len=8)

    assert model.predict([_feature(0.0)]) == 0.0
    assert model.predict([_feature(0.0), _feature(1.0)]) == pytest.approx(0.75)
    assert backend.calls[0].shape == (2, 5)


def test_live_30fps_history_is_resampled_to_explicit_20fps() -> None:
    backend = _RecordingBackend(required_sequence_len=10, model_fps=10.0)
    model = TemporalRiskModel(backend=backend, sequence_len=4, min_infer_steps=2, model_fps=20.0)

    short_history = [_feature(index / 30.0) for index in range(10)]
    assert model.predict(short_history) == 0.0
    assert backend.calls == []
    assert model.required_history_seconds == pytest.approx(0.45)

    history = [_feature(index / 30.0) for index in range(16)]
    assert model.predict(history) == pytest.approx(0.75)

    source_timestamps = np.arange(16, dtype=np.float64) / 30.0
    # Targets are 0.05 seconds apart; exact midpoint ties select the earlier
    # live frame so sampling remains deterministic.
    expected = source_timestamps[[1, 3, 4, 6, 7, 9, 10, 12, 13, 15]]
    np.testing.assert_allclose(backend.calls[0][:, 3], expected, atol=1e-6)


def test_torch_engine_exposes_checkpoint_runtime_metadata(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    state_dict = GRURiskNet(torch, input_size=5, hidden_size=4, num_layers=1).model.state_dict()
    checkpoint_path = tmp_path / "model.pt"
    torch.save(
        {
            "model_state_dict": state_dict,
            "input_size": 5,
            "hidden_size": 4,
            "num_layers": 1,
            "sequence_len": 10,
            "model_type": "gru",
            "decision_threshold": 0.73,
            "model_fps": 20.0,
        },
        checkpoint_path,
    )

    engine = TorchGRUInferenceEngine(str(checkpoint_path))

    assert engine.required_sequence_len == 10
    assert engine.decision_threshold == pytest.approx(0.73)
    assert engine.model_fps == pytest.approx(20.0)

    model = TemporalRiskModel(engine, sequence_len=4, min_infer_steps=2)
    assert model.sequence_len == 10
    assert model.min_infer_steps == 10
    assert model.decision_threshold == pytest.approx(0.73)


def test_legacy_torch_checkpoint_has_no_threshold_or_model_fps(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    state_dict = GRURiskNet(torch, input_size=5, hidden_size=4, num_layers=1).model.state_dict()
    checkpoint_path = tmp_path / "legacy.pt"
    torch.save(
        {
            "model_state_dict": state_dict,
            "input_size": 5,
            "hidden_size": 4,
            "num_layers": 1,
            "sequence_len": 10,
        },
        checkpoint_path,
    )

    engine = TorchGRUInferenceEngine(str(checkpoint_path))

    assert engine.decision_threshold is None
    assert engine.model_fps is None
