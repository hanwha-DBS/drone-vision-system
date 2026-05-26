"""
Phase 2: detector fine-tuning.

Trains a custom detector for person/dump_truck/tanker_truck/truck plus
optional excavator/loader. Supports YOLO11 (default) and RT-DETRv2 backbones
through Ultralytics. Honors the model-strategy.md recommendations:
    - YOLO11m for the current GPU stack
    - RT-DETR-L when small-object recall is the priority

Example:
    python -m training.finetune_detector \
        --data configs/person_dump_tanker_dataset.yaml \
        --model yolo11m.pt --epochs 100 --imgsz 1280 --batch 8
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import RTDETR, YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune the detector for mine-site safety classes.")
    parser.add_argument("--data", default="configs/person_dump_tanker_dataset.yaml", help="Dataset yaml.")
    parser.add_argument("--model", default="yolo11m.pt", help="Base checkpoint (yolo11* or rtdetr-l.pt).")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="person_dump_tanker_finetune")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--rtdetr", action="store_true", help="Force RT-DETR architecture.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_path}")
    Path(args.project).mkdir(parents=True, exist_ok=True)

    is_rtdetr = args.rtdetr or "rtdetr" in args.model.lower()
    model_cls = RTDETR if is_rtdetr else YOLO
    model = model_cls(args.model)

    # Heavier augmentation for the drone domain.
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        workers=args.workers,
        patience=args.patience,
        pretrained=True,
        cache=False,
        close_mosaic=10,
        degrees=2.0,
        scale=0.35,
        fliplr=0.5,
        mosaic=0.6,
        mixup=0.10,
        copy_paste=0.30,    # critical for tiny-object recall
        hsv_h=0.015,
        hsv_s=0.55,
        hsv_v=0.35,
    )


if __name__ == "__main__":
    main()
