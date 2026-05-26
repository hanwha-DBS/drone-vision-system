"""
L5: Active learning ROI capture.

Saves uncertain detections (confidence in the band where the model is most
likely to be wrong) and rejected VLM verifications to a queue directory.
A separate label-review step turns these into new training data, which is
the engine of the long-term accuracy improvement.

Output layout:
    <root>/
        rois/             cropped jpgs
        manifest.csv      one row per saved ROI
"""
from __future__ import annotations

import csv
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import cv2
import numpy as np


@dataclass
class ActiveLearningConfig:
    enabled: bool = True
    root_directory: str = "datasets/active_learning"
    confidence_min: float = 0.30
    confidence_max: float = 0.60
    crop_padding: float = 0.15
    save_vlm_rejected: bool = True
    save_unconfirmed_tracks: bool = True
    max_crops_per_snapshot: int = 6


class ActiveLearningSink:
    def __init__(self, config: ActiveLearningConfig):
        self.config = config
        self._lock = threading.Lock()
        self._initialized = False
        self._manifest_path: Optional[str] = None
        self._roi_dir: Optional[str] = None

    def _ensure_initialized(self) -> bool:
        if self._initialized:
            return True
        if not self.config.enabled:
            return False
        try:
            root = os.path.abspath(self.config.root_directory)
            self._roi_dir = os.path.join(root, "rois")
            os.makedirs(self._roi_dir, exist_ok=True)
            self._manifest_path = os.path.join(root, "manifest.csv")
            if not os.path.exists(self._manifest_path):
                with open(self._manifest_path, "w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow([
                        "saved_at",
                        "image_path",
                        "frame_index",
                        "label",
                        "score",
                        "reason",
                        "track_id",
                        "vlm_positive",
                        "vlm_negative",
                        "vlm_top_prompt",
                    ])
            self._initialized = True
            return True
        except Exception as exc:
            print(f"[ActiveLearning] init failed: {exc}", flush=True)
            return False

    def _crop(self, frame: np.ndarray, det: Dict) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        pad_x = int((det["x2"] - det["x1"]) * self.config.crop_padding)
        pad_y = int((det["y2"] - det["y1"]) * self.config.crop_padding)
        x1 = max(0, det["x1"] - pad_x)
        y1 = max(0, det["y1"] - pad_y)
        x2 = min(w, det["x2"] + pad_x)
        y2 = min(h, det["y2"] + pad_y)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    def _is_uncertain(self, det: Dict) -> bool:
        return self.config.confidence_min <= det["score"] <= self.config.confidence_max

    def submit(self, frame: np.ndarray, frame_index: int, all_detections: List[Dict], confirmed_detections: List[Dict]) -> int:
        if not self._ensure_initialized():
            return 0

        confirmed_keys = {(d.get("track_id"), id(d)) for d in confirmed_detections}
        candidates: List[Dict] = []
        for det in all_detections:
            reason: Optional[str] = None
            if self._is_uncertain(det):
                reason = "uncertain_confidence"
            elif self.config.save_vlm_rejected and det.get("vlm_verified") is False:
                reason = "vlm_rejected"
            elif self.config.save_unconfirmed_tracks and not det.get("track_confirmed", True):
                if (det.get("track_id"), id(det)) not in confirmed_keys:
                    reason = "track_unconfirmed"
            if reason is None:
                continue
            det_copy = dict(det)
            det_copy["_review_reason"] = reason
            candidates.append(det_copy)

        if not candidates:
            return 0

        candidates.sort(key=lambda d: abs(d["score"] - 0.45))
        candidates = candidates[: self.config.max_crops_per_snapshot]

        with self._lock:
            saved = 0
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            for idx, det in enumerate(candidates):
                crop = self._crop(frame, det)
                if crop is None or crop.size == 0:
                    continue
                filename = f"frame{frame_index:06d}_{timestamp}_{idx:02d}_{det.get('label', 'x')}.jpg"
                image_path = os.path.join(self._roi_dir, filename)
                if not cv2.imwrite(image_path, crop):
                    continue
                with open(self._manifest_path, "a", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow([
                        datetime.now().isoformat(timespec="seconds"),
                        image_path,
                        frame_index,
                        det.get("label", ""),
                        round(float(det.get("score", 0.0)), 4),
                        det.get("_review_reason", ""),
                        det.get("track_id", ""),
                        det.get("vlm_positive", ""),
                        det.get("vlm_negative", ""),
                        det.get("vlm_top_prompt", ""),
                    ])
                saved += 1
            return saved
