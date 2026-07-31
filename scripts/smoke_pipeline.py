"""Smoke test for the 5-layer detection pipeline.

No labeled eval set exists yet, so this is the minimal regression net:
  1. Unit-checks the WBF fusion logic with fixed boxes (no models needed).
  2. Builds the full pipeline from configs/runtime_detector.json and runs one
     snapshot through it (downloads weights on first run), printing detection
     counts, layer timings, and the active layer map.

Usage:
    python scripts/smoke_pipeline.py                 # WBF checks + pipeline run
    python scripts/smoke_pipeline.py --skip-models   # WBF checks only
    python scripts/smoke_pipeline.py --video path.mp4  # use a real frame
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np


def check_wbf() -> None:
    from detection.detector import merge_detections, merge_detections_wbf

    person_a = {"x1": 100, "y1": 100, "x2": 200, "y2": 200, "score": 0.6,
                "label": "person", "display_label": "person", "model_tag": "primary"}
    person_b = {"x1": 110, "y1": 105, "x2": 210, "y2": 205, "score": 0.4,
                "label": "person", "display_label": "person", "model_tag": "secondary"}
    lone_vehicle = {"x1": 500, "y1": 500, "x2": 640, "y2": 580, "score": 0.3,
                    "label": "vehicle", "display_label": "truck", "model_tag": "primary"}
    vehicle_a = {"x1": 800, "y1": 300, "x2": 950, "y2": 380, "score": 0.7,
                 "label": "vehicle", "display_label": "car", "model_tag": "primary"}
    vehicle_b = {"x1": 805, "y1": 305, "x2": 955, "y2": 385, "score": 0.5,
                 "label": "vehicle", "display_label": "dump_truck", "model_tag": "secondary"}

    fused = merge_detections_wbf(
        [person_a, person_b, lone_vehicle, vehicle_a, vehicle_b], iou_threshold=0.42
    )
    assert len(fused) == 3, f"expected 3 clusters, got {len(fused)}: {fused}"

    person = next(d for d in fused if d["label"] == "person")
    assert person["fused_from"] == 2, person
    # Two distinct models agreed -> bonus, never below the best member.
    assert 0.6 <= person["score"] <= 0.65, person
    # Coordinates are the score-weighted average, between the members.
    assert 100 <= person["x1"] <= 110 and 200 <= person["x2"] <= 210, person

    lone = next(d for d in fused if d["display_label"] == "truck")
    assert lone["score"] == 0.3 and "fused_from" not in lone, lone

    vehicle = next(d for d in fused if d["x1"] >= 700)
    # prefer_secondary_for_vehicle: display label comes from the secondary model.
    assert vehicle["display_label"] == "dump_truck", vehicle
    assert vehicle["score"] >= 0.7, vehicle

    # NMS path still works and keeps one winner per cluster.
    kept = merge_detections([person_a, person_b], iou_threshold=0.42)
    assert len(kept) == 1 and kept[0]["score"] == 0.6, kept

    print("[smoke] WBF unit checks passed")


def run_pipeline_once(video: str | None) -> None:
    import torch

    from detection.pipeline import build_pipeline

    config = json.loads((ROOT / "configs" / "runtime_detector.json").read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[smoke] building pipeline on device={device}")
    pipeline = build_pipeline(config, device)
    print(f"[smoke] primary={pipeline.primary_model_path} secondary={pipeline.secondary_model_path}")
    print(f"[smoke] layers={pipeline.describe_active_layers()}")

    if video:
        import cv2
        cap = cv2.VideoCapture(video)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise SystemExit(f"could not read a frame from {video}")
    else:
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 255, size=(1080, 1920, 3), dtype=np.uint8)

    pipeline.set_active_profile("accurate")
    pipeline.warmup()
    result = pipeline.run(frame, frame_index=0, interval_seconds=3)
    print(f"[smoke] mode={result.analysis_mode}")
    print(f"[smoke] raw={len(result.raw_detections)} confirmed={len(result.confirmed_detections)}")
    print("[smoke] timings_ms=" + json.dumps({k: round(v, 1) for k, v in result.layer_timings_ms.items()}))
    print("[smoke] pipeline run OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-models", action="store_true", help="run only the WBF unit checks")
    parser.add_argument("--video", default=None, help="optional video file to grab a real frame from")
    args = parser.parse_args()

    check_wbf()
    if not args.skip_models:
        run_pipeline_once(args.video)


if __name__ == "__main__":
    main()
