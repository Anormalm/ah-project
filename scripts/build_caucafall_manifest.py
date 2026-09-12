from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from temporal.public_dataset import (
    caucafall_manifest_from_omnifall,
    caucafall_manifest_from_yolo_tree,
    write_public_manifest,
)


def _integer_set(value: str) -> set[int]:
    try:
        return {int(part.strip()) for part in value.split(",") if part.strip()}
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integer IDs") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a leak-free CAUCAFall manifest from official or OmniFall labels"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--dataset-root",
        help="Official CAUCAFall Subject.N/activity tree with AVI and per-frame TXT files",
    )
    source.add_argument("--labels", help="Alternative: OmniFall labels/caucafall.csv")
    parser.add_argument(
        "--video-root",
        help="With --labels: directory containing paths such as adl/WalkS1.mp4",
    )
    parser.add_argument("--output", default="config/caucafall_manifest.json")
    parser.add_argument(
        "--train-subjects",
        type=_integer_set,
        default=_integer_set("1,2,3,4,5,6,7,8"),
        help="Comma-separated subject IDs (default: 1-8)",
    )
    parser.add_argument(
        "--val-subjects",
        type=_integer_set,
        default=_integer_set("9,10"),
        help="Comma-separated subject IDs (default: 9,10)",
    )
    parser.add_argument(
        "--positive-labels",
        type=_integer_set,
        default=_integer_set("1,2"),
        help="OmniFall classes mapped to fall=1 (default: fall and fallen)",
    )
    parser.add_argument("--video-extension", default=".mp4")
    parser.add_argument(
        "--fps",
        type=float,
        help="Optional FPS override; by default each AVI's encoded FPS is used",
    )
    parser.add_argument(
        "--negative-guard-frames",
        type=int,
        default=10,
        help="Do not emit negative windows this many frames before a fall label",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset_root:
        manifest = caucafall_manifest_from_yolo_tree(
            dataset_root=args.dataset_root,
            train_subjects=args.train_subjects,
            val_subjects=args.val_subjects,
            fps=args.fps,
            negative_guard_frames=args.negative_guard_frames,
        )
    else:
        if not args.video_root:
            raise ValueError("--video-root is required with --labels")
        manifest = caucafall_manifest_from_omnifall(
            labels_csv=args.labels,
            video_root=args.video_root,
            train_subjects=args.train_subjects,
            val_subjects=args.val_subjects,
            positive_labels=args.positive_labels,
            video_extension=args.video_extension,
        )
    write_public_manifest(manifest, args.output)
    print(f"Wrote {len(manifest.clips)} clips to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
