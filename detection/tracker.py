"""
L4: Temporal consistency layer.

Snapshot-based ByteTrack wrapper with N-of-M confirmation rule:
only emit an alarm when a track has been observed in at least N of the last
M snapshots. This removes single-snapshot false positives almost entirely,
because hallucinations rarely persist across multiple sampled frames.

We import ByteTrack lazily from `supervision` so the rest of the pipeline
keeps working if the dependency is missing — in that case we fall back to a
position-based hash tracker (less robust, but still produces track IDs and
N-of-M confirmation).
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple


@dataclass
class TrackerConfig:
    enabled: bool = True
    backend: str = "bytetrack"  # "bytetrack" | "fallback"
    track_activation_threshold: float = 0.15
    lost_track_buffer: int = 30          # number of snapshots a track can survive without re-detection
    minimum_matching_threshold: float = 0.6
    # N-of-M confirmation: a track must appear in `confirm_min_hits` out of
    # the last `confirm_window` snapshots to be marked confirmed.
    confirm_window: int = 5
    confirm_min_hits: int = 3
    # IoU threshold for the fallback position tracker.
    fallback_iou: float = 0.3


class _TrackState:
    __slots__ = ("track_id", "label", "history", "last_seen_at", "last_box", "first_seen_at")

    def __init__(self, track_id: int, label: str, frame_index: int, box: Dict):
        self.track_id = track_id
        self.label = label
        self.history: Deque[int] = deque(maxlen=64)
        self.history.append(frame_index)
        self.first_seen_at = frame_index
        self.last_seen_at = frame_index
        self.last_box = box


def _box_iou(a: Dict, b: Dict) -> float:
    ix1, iy1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    ix2, iy2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(1, (a["x2"] - a["x1"]) * (a["y2"] - a["y1"]))
    area_b = max(1, (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class TemporalTracker:
    """Wraps ByteTrack (or a fallback) and adds N-of-M confirmation."""

    def __init__(self, config: TrackerConfig):
        self.config = config
        self._bytetrack = None
        self._backend_available = False
        if config.enabled and config.backend == "bytetrack":
            self._bytetrack = self._try_load_bytetrack()
        self._tracks: Dict[int, _TrackState] = {}
        self._next_fallback_id = 1

    def _try_load_bytetrack(self):
        try:
            import supervision as sv  # noqa: F401
            from supervision.tracker.byte_tracker.core import ByteTrack

            tracker = ByteTrack(
                track_activation_threshold=self.config.track_activation_threshold,
                lost_track_buffer=self.config.lost_track_buffer,
                minimum_matching_threshold=self.config.minimum_matching_threshold,
                frame_rate=int(max(1, self.config.confirm_window * 2)),
            )
            self._backend_available = True
            print("[Tracker] ByteTrack (supervision) loaded", flush=True)
            return tracker
        except Exception as exc:
            print(f"[Tracker] ByteTrack unavailable, using fallback: {exc}", flush=True)
            self._backend_available = False
            return None

    def reset(self) -> None:
        self._tracks.clear()
        self._next_fallback_id = 1
        if self._bytetrack is not None:
            self._bytetrack = self._try_load_bytetrack()

    def _assign_with_bytetrack(self, detections: List[Dict]) -> List[Optional[int]]:
        if self._bytetrack is None or not detections:
            return [None] * len(detections)
        try:
            import numpy as np
            import supervision as sv

            xyxy = np.array([[d["x1"], d["y1"], d["x2"], d["y2"]] for d in detections], dtype=float)
            confidence = np.array([d["score"] for d in detections], dtype=float)
            class_id = np.array([0 if d["label"] == "person" else 1 for d in detections], dtype=int)
            sv_detections = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
            tracked = self._bytetrack.update_with_detections(sv_detections)
            # Map back by IoU between tracked.xyxy and detections.
            ids: List[Optional[int]] = [None] * len(detections)
            for t_idx, t_xyxy in enumerate(tracked.xyxy):
                if tracked.tracker_id is None:
                    break
                best_iou = 0.0
                best_det = None
                for d_idx, det in enumerate(detections):
                    box_a = {
                        "x1": float(t_xyxy[0]),
                        "y1": float(t_xyxy[1]),
                        "x2": float(t_xyxy[2]),
                        "y2": float(t_xyxy[3]),
                    }
                    score = _box_iou(box_a, det)
                    if score > best_iou:
                        best_iou = score
                        best_det = d_idx
                if best_det is not None and best_iou > 0.3:
                    ids[best_det] = int(tracked.tracker_id[t_idx])
            return ids
        except Exception as exc:
            print(f"[Tracker] bytetrack update failed: {exc}", flush=True)
            return [None] * len(detections)

    def _assign_with_fallback(self, detections: List[Dict]) -> List[Optional[int]]:
        ids: List[Optional[int]] = [None] * len(detections)
        used: set = set()
        for d_idx, det in enumerate(detections):
            best_iou = self.config.fallback_iou
            best_id: Optional[int] = None
            for track_id, state in self._tracks.items():
                if track_id in used:
                    continue
                if state.label != det["label"]:
                    continue
                score = _box_iou(state.last_box, det)
                if score >= best_iou:
                    best_iou = score
                    best_id = track_id
            if best_id is not None:
                ids[d_idx] = best_id
                used.add(best_id)
        # Assign new ids for unmatched detections.
        for d_idx, det in enumerate(detections):
            if ids[d_idx] is None:
                ids[d_idx] = self._next_fallback_id
                self._next_fallback_id += 1
        return ids

    def update(self, frame_index: int, detections: List[Dict]) -> List[Dict]:
        """Assign track IDs, update history, mark confirmed tracks."""
        if not self.config.enabled or not detections:
            return detections

        if self._backend_available:
            assigned = self._assign_with_bytetrack(detections)
            # Fill in any unmatched detections with fallback IDs.
            for idx, tid in enumerate(assigned):
                if tid is None:
                    assigned[idx] = self._next_fallback_id
                    self._next_fallback_id += 1
        else:
            assigned = self._assign_with_fallback(detections)

        confirmed_results: List[Dict] = []
        for det, track_id in zip(detections, assigned):
            if track_id is None:
                continue
            state = self._tracks.get(track_id)
            if state is None:
                state = _TrackState(track_id, det["label"], frame_index, det)
                self._tracks[track_id] = state
            else:
                state.history.append(frame_index)
                state.last_seen_at = frame_index
                state.last_box = det

            recent_window = [
                h for h in state.history
                if h > frame_index - self.config.confirm_window
            ]
            hits = len(recent_window)
            confirmed = hits >= self.config.confirm_min_hits
            track_age = frame_index - state.first_seen_at + 1

            det["track_id"] = track_id
            det["track_hits"] = hits
            det["track_age"] = track_age
            det["track_confirmed"] = confirmed
            confirmed_results.append(det)

        # Garbage-collect tracks that have been gone too long.
        cutoff = frame_index - self.config.lost_track_buffer
        stale = [tid for tid, st in self._tracks.items() if st.last_seen_at < cutoff]
        for tid in stale:
            del self._tracks[tid]

        return confirmed_results

    def filter_confirmed(self, detections: List[Dict]) -> List[Dict]:
        """Return only detections whose track has passed N-of-M confirmation."""
        if not self.config.enabled:
            return detections
        return [d for d in detections if d.get("track_confirmed", False)]
