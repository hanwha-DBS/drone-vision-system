"""
L2: Slicing Aided Hyper Inference (SAHI).

Tiles a high-resolution drone frame into overlapping crops, runs the L1
detector on each tile, then merges results in the original frame coordinates.
This is the single biggest improvement for tiny-object recall on drone footage.

We implement a lightweight tiling loop instead of taking a hard dependency on
the `sahi` package, because the detector wrapper already produces normalized
detections and a custom tiler is simpler to tune for our risk-aware filters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .detector import L1Detector, merge_detections, merge_detections_wbf


@dataclass
class SahiConfig:
    enabled: bool = True
    fusion: str = "wbf"  # "wbf" fuses tile-clipped boxes; "nms" keeps one winner
    tile_size: int = 768
    overlap: float = 0.25
    min_tile_score: float = 0.0
    nms_iou_threshold: float = 0.5
    max_tiles: int = 64


def _generate_tile_origins(width: int, height: int, tile_size: int, overlap: float, max_tiles: int) -> List[Tuple[int, int]]:
    stride = max(1, int(tile_size * (1.0 - overlap)))
    xs = list(range(0, max(1, width - tile_size + 1), stride))
    ys = list(range(0, max(1, height - tile_size + 1), stride))
    # ensure the last tile covers the right/bottom edges
    if not xs or xs[-1] + tile_size < width:
        xs.append(max(0, width - tile_size))
    if not ys or ys[-1] + tile_size < height:
        ys.append(max(0, height - tile_size))
    origins = [(x, y) for y in ys for x in xs]
    if len(origins) > max_tiles:
        # downsample uniformly if config max_tiles exceeded
        step = max(1, len(origins) // max_tiles)
        origins = origins[::step][:max_tiles]
    return origins


def run_sahi(
    frame: np.ndarray,
    detector: L1Detector,
    config: SahiConfig,
) -> List[Dict]:
    """Tile the frame, run detection per tile, merge into a single list."""
    if not config.enabled:
        return detector.predict(frame)

    height, width = frame.shape[:2]
    tile_size = min(config.tile_size, min(width, height))
    if tile_size >= max(width, height):
        return detector.predict(frame)

    origins = _generate_tile_origins(width, height, tile_size, config.overlap, config.max_tiles)
    all_detections: List[Dict] = []

    # Always include a full-frame pass — catches large objects (whole trucks)
    # that get clipped across multiple tiles.
    all_detections.extend(detector.predict(frame))

    for x0, y0 in origins:
        x1 = min(width, x0 + tile_size)
        y1 = min(height, y0 + tile_size)
        tile = frame[y0:y1, x0:x1]
        if tile.size == 0:
            continue
        # No TTA on tiles — the per-tile cost doubles for little gain;
        # TTA stays on the full-frame pass only.
        tile_detections = detector.predict(tile, use_tta=False)
        for det in tile_detections:
            if det["score"] < config.min_tile_score:
                continue
            shifted = dict(det)
            shifted["x1"] = int(det["x1"]) + x0
            shifted["y1"] = int(det["y1"]) + y0
            shifted["x2"] = int(det["x2"]) + x0
            shifted["y2"] = int(det["y2"]) + y0
            shifted["source"] = "tile"
            all_detections.append(shifted)

    merge_fn = merge_detections_wbf if config.fusion == "wbf" else merge_detections
    return merge_fn(all_detections, iou_threshold=config.nms_iou_threshold)
