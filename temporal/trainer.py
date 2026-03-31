from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from temporal.temporal_model import GRURiskNet, TinyTransformerRiskNet, TemporalModelMeta


@dataclass
class TrainerConfig:
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stop_patience: int = 5
    decision_threshold: float = 0.5
    device: str = "cpu"
    seed: int = 42
    model_type: str = "gru"


def _set_seed(torch_module, seed: int) -> None:
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def _batch_indices(n: int, batch_size: int, rng: np.random.Generator):
    idx = np.arange(n)
    rng.shuffle(idx)
    for start in range(0, n, batch_size):
        yield idx[start : start + batch_size]


def _binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_true = y_true.astype(np.float32)
    y_prob = y_prob.astype(np.float32)
    y_pred = (y_prob >= threshold).astype(np.float32)

    tp = float(((y_pred == 1.0) & (y_true == 1.0)).sum())
    tn = float(((y_pred == 0.0) & (y_true == 0.0)).sum())
    fp = float(((y_pred == 1.0) & (y_true == 0.0)).sum())
    fn = float(((y_pred == 0.0) & (y_true == 1.0)).sum())

    precision = tp / max(tp + fp, 1e-9)
    recall = tp / max(tp + fn, 1e-9)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
    acc = (tp + tn) / max(tp + tn + fp + fn, 1e-9)
    auc = _roc_auc(y_true, y_prob)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "auc": auc}


def _roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    pos = y_true > 0.5
    neg = ~pos
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(y_prob)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_prob) + 1, dtype=np.float64)
    pos_rank_sum = ranks[pos].sum()
    auc = (pos_rank_sum - (n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg)
    return float(np.clip(auc, 0.0, 1.0))


def _build_model(torch_module, meta: TemporalModelMeta):
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


class TemporalRiskTrainer:
    def __init__(self, cfg: TrainerConfig) -> None:
        self.cfg = cfg
        import torch

        self._torch = torch
        _set_seed(torch, cfg.seed)

    def _normalize(self, x_train: np.ndarray, x_val: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mean = x_train.reshape(-1, x_train.shape[-1]).mean(axis=0)
        std = x_train.reshape(-1, x_train.shape[-1]).std(axis=0)
        std = np.clip(std, 1e-6, None)
        train_norm = (x_train - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
        val_norm = (x_val - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
        return train_norm.astype(np.float32), val_norm.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        model_meta: TemporalModelMeta | None = None,
    ) -> tuple[dict[str, float], dict]:
        torch = self._torch
        if model_meta is None:
            meta = TemporalModelMeta(
                input_size=int(x_train.shape[-1]),
                sequence_len=int(x_train.shape[1]),
                model_type=self.cfg.model_type,
            )
        else:
            if model_meta.model_type:
                meta = model_meta
            else:
                meta = TemporalModelMeta(
                    input_size=model_meta.input_size,
                    hidden_size=model_meta.hidden_size,
                    num_layers=model_meta.num_layers,
                    sequence_len=model_meta.sequence_len,
                    model_type=self.cfg.model_type,
                    attention_heads=model_meta.attention_heads,
                    ff_mult=model_meta.ff_mult,
                )

        x_train, x_val, feature_mean, feature_std = self._normalize(x_train, x_val if x_val.size > 0 else x_train[:1])

        model = _build_model(torch, meta).to(self.cfg.device)
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)

        train_x_t = torch.from_numpy(x_train).to(self.cfg.device)
        train_y_t = torch.from_numpy(y_train.reshape(-1, 1)).to(self.cfg.device)
        val_x_t = torch.from_numpy(x_val).to(self.cfg.device) if x_val.size > 0 else None

        best_state = None
        best_score = -1.0
        bad_epochs = 0
        rng = np.random.default_rng(self.cfg.seed)

        for _ in range(self.cfg.epochs):
            model.train()
            for batch_idx in _batch_indices(train_x_t.shape[0], self.cfg.batch_size, rng):
                bx = train_x_t[batch_idx]
                by = train_y_t[batch_idx]
                optimizer.zero_grad()
                pred = model(bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()

            model.eval()
            if val_x_t is None or x_val.size == 0:
                with torch.no_grad():
                    prob = model(train_x_t).detach().cpu().numpy().reshape(-1)
                    metrics = _binary_metrics(y_train, prob, threshold=self.cfg.decision_threshold)
            else:
                with torch.no_grad():
                    prob = model(val_x_t).detach().cpu().numpy().reshape(-1)
                    metrics = _binary_metrics(y_val, prob, threshold=self.cfg.decision_threshold)

            score = metrics["f1"]
            if score > best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.cfg.early_stop_patience:
                    break

        if best_state is None:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        artifact = {
            "model_state_dict": best_state,
            "input_size": meta.input_size,
            "hidden_size": meta.hidden_size,
            "num_layers": meta.num_layers,
            "sequence_len": meta.sequence_len,
            "model_type": meta.model_type,
            "attention_heads": meta.attention_heads,
            "ff_mult": meta.ff_mult,
            "feature_mean": feature_mean.tolist(),
            "feature_std": feature_std.tolist(),
        }
        final_metrics = metrics if "metrics" in locals() else {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "auc": 0.5}
        return final_metrics, artifact

    def save(self, artifact: dict, output_path: str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._torch.save(artifact, path)


class GRURiskTrainer(TemporalRiskTrainer):
    def __init__(self, cfg: TrainerConfig) -> None:
        if cfg.model_type != "gru":
            cfg.model_type = "gru"
        super().__init__(cfg)
