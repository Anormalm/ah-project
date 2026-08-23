from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

from pipelines.main_pipeline import MultiStreamRunner


def _resolve_runtime_paths(cfg: dict, cfg_path: Path) -> dict:
    candidate_root = cfg_path.parent.parent
    project_root = candidate_root if (candidate_root / "run.py").exists() else cfg_path.parent

    for section in ("detection", "pose", "temporal_model"):
        model_path = (cfg.get(section) or {}).get("model_path")
        if model_path and not Path(model_path).is_absolute():
            cfg[section]["model_path"] = str((project_root / model_path).resolve())

    output_cfg = cfg.get("output") or {}
    for key in ("json_log_path", "training_log_path"):
        value = output_cfg.get(key)
        if value and not Path(value).is_absolute():
            output_cfg[key] = str((project_root / value).resolve())

    for stream in cfg.get("streams", []):
        if stream.get("type") != "video":
            continue
        source = stream.get("source")
        if isinstance(source, str) and not Path(source).is_absolute():
            stream["source"] = str((project_root / source).resolve())
    return cfg


def _expand_environment(value):
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def load_config(path: str) -> dict:
    cfg_path = Path(path).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = _expand_environment(yaml.safe_load(f))
    return _resolve_runtime_paths(cfg, cfg_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time multi-person pose-based risk detection pipeline")
    parser.add_argument("--config", default="config/config.yaml", help="Path to YAML configuration file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    runner = MultiStreamRunner(cfg)
    runner.run()


if __name__ == "__main__":
    main()

