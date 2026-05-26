"""
Phase 5: TensorRT INT8 export for production inference.

Converts a trained Ultralytics checkpoint to a TensorRT engine with INT8
quantization. INT8 calibration requires a representative dataset (200-500
images that match the deployment distribution).

Usage:
    python -m training.export_tensorrt \
        --weights runs/train/person_dump_tanker_finetune/weights/best.pt \
        --calib-dir datasets/calibration_set \
        --imgsz 1280

The exported engine is platform/driver specific — re-export per machine.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Ultralytics model to TensorRT INT8.")
    parser.add_argument("--weights", required=True, help="Path to the .pt checkpoint.")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--calib-dir", default=None, help="Directory of calibration images (INT8).")
    parser.add_argument("--half", action="store_true", help="Export FP16 instead of INT8.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(weights)

    model = YOLO(str(weights))
    export_kwargs = {
        "format": "engine",
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": 0,
        "workspace": 4,
    }
    if args.half:
        export_kwargs["half"] = True
    else:
        export_kwargs["int8"] = True
        if args.calib_dir:
            export_kwargs["data"] = args.calib_dir

    output = model.export(**export_kwargs)
    print(f"[export] TensorRT engine written: {output}", flush=True)


if __name__ == "__main__":
    main()
