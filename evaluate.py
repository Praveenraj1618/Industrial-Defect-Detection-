from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained detector on the held-out test split.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=ROOT / "runs" / "defect_detection" / "train" / "weights" / "best.pt",
    )
    parser.add_argument("--data", type=Path, default=ROOT / "dataset_yolo" / "data.yaml")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto", help="auto, 0 for NVIDIA GPU, or cpu.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = args.weights.resolve()
    data = args.data.resolve()

    if not weights.is_file():
        print(f"Model checkpoint not found: {weights}", file=sys.stderr)
        print("Run python train.py first.", file=sys.stderr)
        raise SystemExit(1)
    if not data.is_file():
        print(f"Dataset YAML not found: {data}", file=sys.stderr)
        print("Run python convert_dataset.py first.", file=sys.stderr)
        raise SystemExit(1)

    if args.device.lower() == "auto":
        device = 0 if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Weights: {weights}")
    print(f"Dataset: {data}")
    print(f"Split: {args.split}")
    print(f"Device: {device}")

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data),
        split=args.split,
        imgsz=args.imgsz,
        device=device,
        plots=True,
        project=str(ROOT / "runs" / "defect_detection" / "evaluation"),
        name=args.split,
        exist_ok=True,
    )

    box_metrics = getattr(metrics, "box", None)
    summary = {
        "weights": str(weights),
        "data": str(data),
        "split": args.split,
        "precision": float(box_metrics.mp) if box_metrics is not None else None,
        "recall": float(box_metrics.mr) if box_metrics is not None else None,
        "map50": float(box_metrics.map50) if box_metrics is not None else None,
        "map50_95": float(box_metrics.map) if box_metrics is not None else None,
        "results": {
            str(key): float(value)
            for key, value in getattr(metrics, "results_dict", {}).items()
            if isinstance(value, (int, float))
        },
    }

    output_dir = ROOT / "runs" / "defect_detection" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{args.split}_metrics.json"
    report_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\nEvaluation summary")
    for key in ("precision", "recall", "map50", "map50_95"):
        value = summary[key]
        print(f"{key}: {value:.4f}" if value is not None else f"{key}: unavailable")
    print(f"Full metrics: {report_path}")


if __name__ == "__main__":
    main()
