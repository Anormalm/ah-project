from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from pipelines.main_pipeline import MultiStreamRunner


def load_config(path: str) -> dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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

