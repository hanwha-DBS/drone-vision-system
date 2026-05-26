"""
L1: Primary detector wrapper.

Pluggable interface over Ultralytics YOLO / RT-DETR with class normalization
and geometry filtering. Preserves the behavior of the legacy engine.py
predict_boxes_from_model function while making the model swappable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO, RTDETR


def _align_to_stride(value: int, stride: int = 32) -> int:
    value = max(stride, int(value))
    return ((value + stride - 1) // stride) * stride


def _clip_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> Dict[str, int]:
    return {
        "x1": int(max(0, min(width - 1, round(x1)))),
        "y1": int(max(0, min(height - 1, round(y1)))),
        "x2": int(max(0, min(width - 1, round(x2)))),
        "y2": int(max(0, min(height - 1, round(y2)))),
    }


@dataclass
class DetectorConfig:
    model_path: str
    target_width: int = 1792
    inference_imgsz: int = 1792
    confidence_threshold: float = 0.08
    architecture: str = "auto"  # "yolo" | "rtdetr" | "auto"
    half: bool = True
    device: str = "cuda"
    tag: str = "primary"


@dataclass
class FilterConfig:
    allowed_classes: Sequence[str] = field(default_factory=lambda: ("person", "car", "truck", "bus", "vehicle"))
    class_aliases: Dict[str, str] = field(default_factory=lambda: {
        "person": "person",
        "car": "vehicle",
        "truck": "vehicle",
        "bus": "vehicle",
        "dump_truck": "vehicle",
        "tanker_truck": "vehicle",
        "suv": "vehicle",
        "van": "vehicle",
        "excavator": "vehicle",
    })
    class_min_scores: Dict[str, float] = field(default_factory=lambda: {"person": 0.12, "vehicle": 0.10})
    geometry: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "person": {"min_area_ratio": 3e-6, "max_area_ratio": 0.05, "min_aspect_ratio": 0.08, "max_aspect_ratio": 1.6},
        "vehicle": {"min_area_ratio": 2e-5, "max_area_ratio": 0.30, "min_aspect_ratio": 0.18, "max_aspect_ratio": 7.2},
    })

    def normalize(self, raw_label: str) -> str:
        return self.class_aliases.get(raw_label, raw_label)

    def is_allowed(self, raw_label: str, normalized: str) -> bool:
        return raw_label in self.allowed_classes or normalized in self.allowed_classes

    def min_score(self, label: str) -> float:
        return self.class_min_scores.get(label, 0.0)

    def passes_geometry(self, box: Dict[str, int], label: str, frame_shape: Tuple[int, int, int]) -> bool:
        height, width = frame_shape[:2]
        area = max(1, (box["x2"] - box["x1"]) * (box["y2"] - box["y1"]))
        frame_area = max(1, width * height)
        area_ratio = area / frame_area
        box_width = max(1, box["x2"] - box["x1"])
        box_height = max(1, box["y2"] - box["y1"])
        aspect_ratio = box_width / box_height
        rules = self.geometry.get(label, self.geometry.get("vehicle", {}))
        if area_ratio < rules.get("min_area_ratio", 0.0):
            return False
        if area_ratio > rules.get("max_area_ratio", 1.0):
            return False
        if aspect_ratio < rules.get("min_aspect_ratio", 0.0):
            return False
        if aspect_ratio > rules.get("max_aspect_ratio", 999.0):
            return False
        return True


class L1Detector:
    """Single-model detector. Wrap multiple instances for ensemble."""

    def __init__(self, config: DetectorConfig, filter_config: FilterConfig):
        self.config = config
        self.filter = filter_config
        self.model = self._load_model()

    def _load_model(self):
        arch = self.config.architecture.lower()
        path_lower = self.config.model_path.lower()
        if arch == "auto":
            arch = "rtdetr" if "rtdetr" in path_lower else "yolo"
        if arch == "rtdetr":
            return RTDETR(self.config.model_path)
        return YOLO(self.config.model_path)

    def _resize_for_inference(self, frame: np.ndarray) -> Tuple[np.ndarray, float, float]:
        height, width = frame.shape[:2]
        target_width = self.config.target_width
        if width <= target_width:
            return frame, 1.0, 1.0
        scale = target_width / float(width)
        resized = cv2.resize(
            frame,
            (target_width, max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        return resized, width / float(resized.shape[1]), height / float(resized.shape[0])

    def predict(self, frame: np.ndarray) -> List[Dict]:
        resized, scale_x, scale_y = self._resize_for_inference(frame)
        imgsz = _align_to_stride(min(max(resized.shape[:2]), self.config.inference_imgsz))
        use_half = self.config.half and self.config.device.startswith("cuda") and torch.cuda.is_available()
        results = self.model.predict(
            source=resized,
            conf=self.config.confidence_threshold,
            imgsz=imgsz,
            verbose=False,
            device=self.config.device,
            half=use_half,
        )
        if not results:
            return []
        result = results[0]
        names = result.names or {}
        if result.boxes is None:
            return []

        detections: List[Dict] = []
        for xyxy, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            raw_label = str(names.get(int(cls.item()), int(cls.item())))
            normalized = self.filter.normalize(raw_label)
            if not self.filter.is_allowed(raw_label, normalized):
                continue
            x1, y1, x2, y2 = xyxy.tolist()
            box = _clip_box(x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y, frame.shape[1], frame.shape[0])
            box["score"] = float(conf.item())
            box["label"] = normalized
            box["display_label"] = raw_label
            box["model_tag"] = self.config.tag
            if box["score"] < self.filter.min_score(normalized):
                continue
            if not self.filter.passes_geometry(box, normalized, frame.shape):
                continue
            detections.append(box)
        return detections


def merge_detections(
    detections: List[Dict],
    iou_threshold: float = 0.42,
    prefer_secondary_for_vehicle: bool = True,
) -> List[Dict]:
    """Ensemble NMS that preserves per-source label and prefers secondary on vehicle ties."""

    def iou(a: Dict, b: Dict) -> float:
        ix1, iy1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
        ix2, iy2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = max(1, (a["x2"] - a["x1"]) * (a["y2"] - a["y1"]))
        area_b = max(1, (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]))
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    merged: List[Dict] = []
    for candidate in sorted(detections, key=lambda item: item["score"], reverse=True):
        keep = True
        for kept in merged:
            if kept["label"] != candidate["label"]:
                continue
            if iou(kept, candidate) < iou_threshold:
                continue
            if (
                prefer_secondary_for_vehicle
                and candidate["label"] == "vehicle"
                and candidate.get("model_tag") == "secondary"
                and kept.get("model_tag") != "secondary"
            ):
                kept.update(candidate)
            keep = False
            break
        if keep:
            merged.append(candidate)
    return merged
