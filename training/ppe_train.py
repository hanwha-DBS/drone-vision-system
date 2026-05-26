"""
Phase 4: PPE (personal protective equipment) detection sub-task.

Trains a small detector specifically for high-visibility vests and hardhats.
At inference time the engine cross-references PPE detections with person
boxes — a person with a PPE hit gets confidence boosted, a person without
PPE in a blast zone gets flagged but with a separate "no_PPE" annotation.

Class layout (configs/ppe_dataset.yaml — user-provided):
    0: helmet
    1: vest
    2: no_helmet
    3: no_vest

Smaller, fast model (YOLO11s or YOLO11n) is preferred — PPE detection runs on
person crops, not full frames, so the model can be small.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPE sub-task detector.")
    parser.add_argument("--data", default="configs/ppe_dataset.yaml")
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="ppe_detection")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"PPE dataset yaml not found at {data_path}. "
            "Create it with classes: helmet, vest, no_helmet, no_vest."
        )
    Path(args.project).mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        pretrained=True,
        cache=False,
        mosaic=0.5,
        mixup=0.05,
        copy_paste=0.1,
    )


if __name__ == "__main__":
    main()
