from __future__ import annotations

import shutil
from pathlib import Path


def main() -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("ultralytics is not installed. Run: pip install ultralytics") from exc

    target_dir = Path("models")
    target_dir.mkdir(parents=True, exist_ok=True)

    model_names = ["yolo11n.pt", "yolo11n-pose.pt"]
    for name in model_names:
        model = YOLO(name)
        source = Path(str(model.ckpt_path))
        dest = target_dir / name
        if source.exists() and source.resolve() != dest.resolve():
            shutil.copy2(source, dest)
        elif source.exists() and not dest.exists():
            shutil.copy2(source, dest)
        print(f"ready: {dest}")


if __name__ == "__main__":
    main()

