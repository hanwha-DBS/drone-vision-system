import argparse
import os
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a custom detector for person/dump_truck/tanker_truck/truck."
    )
    parser.add_argument(
        "--data",
        default="configs/person_dump_tanker_dataset.yaml",
        help="Path to dataset yaml.",
    )
    parser.add_argument(
        "--model",
        default="yolo11m.pt",
        help="Base model checkpoint. Use yolov8m.pt if YOLO11 weights are not available yet.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="person_dump_tanker_yolo11m")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_path}")

    os.makedirs(args.project, exist_ok=True)

    model = YOLO(args.model)
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
        mixup=0.05,
    )


if __name__ == "__main__":
    main()
