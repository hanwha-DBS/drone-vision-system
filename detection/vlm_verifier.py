"""
L3: Vision-Language Model zero-shot verification.

Crop each candidate box and compare against positive vs hard-negative text
prompts using a SigLIP-style CLIP model. Detections whose top match falls
into a hard-negative class (rocks, shadows, machinery edge, dust) are
rejected before they reach the alarm pipeline.

The verifier is lazily initialized — the model and processor download on
first use, so the rest of the pipeline can run even on machines without
HuggingFace transformers installed (the verifier is then a no-op).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class VLMConfig:
    enabled: bool = True
    model_id: str = "google/siglip-base-patch16-224"
    device: str = "cuda"
    crop_padding: float = 0.10  # add 10% context around each box
    min_crop_size: int = 32
    # If the single highest-scoring prompt is a hard-negative, reject.
    # Off by default because argmax over many negative prompts is noisy —
    # we compare per-prompt averaged probabilities instead (see min_positive_margin).
    reject_if_top_is_negative: bool = False
    # Compare positive vs negative as PER-PROMPT averages (probability mass
    # normalized by the number of prompts in each group). Reject only when
    # the average negative score exceeds the average positive score by more
    # than this margin. Set to 0.0 to disable, or negative to be lenient.
    min_positive_margin: float = 0.0
    # Hard cap on how many detections get a (relatively expensive) SigLIP
    # forward pass per frame. With SAHI (L2) enabled the raw detection count
    # explodes, and verifying every box makes CPU inference hang. The budget
    # goes to the LOWEST-scoring candidates — those are the false-positive
    # suspects; high-confidence boxes are trusted as-is. 0 = no cap.
    max_verifications: int = 0
    # Detections at or above this score skip verification entirely (recall-safe
    # and saves the budget for dubious boxes). 0 disables the shortcut.
    auto_trust_above: float = 0.45
    # How many crops go through SigLIP in a single forward pass.
    batch_size: int = 16
    positive_prompts: Dict[str, List[str]] = field(default_factory=lambda: {
        "person": [
            "an aerial photo of a person wearing a high-visibility safety vest",
            "an aerial drone photo of a worker on the ground",
            "a person standing on rocky terrain seen from above",
        ],
        "vehicle": [
            "an aerial photo of a dump truck on a mine site",
            "an aerial photo of a tanker truck or fuel truck",
            "an aerial photo of an excavator or loader on a mine site",
            "an aerial photo of a pickup truck or SUV on a construction site",
        ],
    })
    hard_negative_prompts: List[str] = field(default_factory=lambda: [
        "a pile of rocks and gravel",
        "a long shadow on the ground",
        "an empty patch of dirt or sand",
        "the metal edge of a machine arm",
        "construction debris and rubble",
        "white gravel patch on the ground",
        "a tarp or plastic sheet",
        "a single rock or boulder on the ground seen from above",
        "crushed stone fragments at a quarry",
        "a stone casting a small shadow on dirt",
    ])
    skip_labels: Sequence[str] = field(default_factory=tuple)


class VLMVerifier:
    """SigLIP-based zero-shot crop classifier. Lazy-loaded."""

    def __init__(self, config: VLMConfig):
        self.config = config
        self._processor = None
        self._model = None
        self._available: Optional[bool] = None
        self._device = config.device

    def _ensure_loaded(self) -> bool:
        if self._available is not None:
            return self._available
        if not self.config.enabled:
            self._available = False
            return False
        try:
            import torch
            from transformers import AutoModel, AutoProcessor

            device = self.config.device
            if device.startswith("cuda") and not torch.cuda.is_available():
                device = "cpu"
            self._device = device
            self._processor = AutoProcessor.from_pretrained(self.config.model_id)
            self._model = AutoModel.from_pretrained(self.config.model_id).to(device).eval()
            self._available = True
            print(f"[VLM] SigLIP loaded: {self.config.model_id} on {device}", flush=True)
        except Exception as exc:
            print(f"[VLM] disabled — failed to load {self.config.model_id}: {exc}", flush=True)
            self._available = False
        return self._available

    def _crop_box(self, frame: np.ndarray, box: Dict) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        pad_x = int((box["x2"] - box["x1"]) * self.config.crop_padding)
        pad_y = int((box["y2"] - box["y1"]) * self.config.crop_padding)
        x1 = max(0, box["x1"] - pad_x)
        y1 = max(0, box["y1"] - pad_y)
        x2 = min(w, box["x2"] + pad_x)
        y2 = min(h, box["y2"] + pad_y)
        # Tiny boxes (distant rocks mistaken for people) are the false-positive
        # hot zone — skipping them would auto-trust exactly the wrong cases.
        # Grow the crop window around the box center to min_crop_size so the
        # model sees enough context to judge.
        min_size = int(self.config.min_crop_size)
        if (x2 - x1) < min_size:
            cx = (x1 + x2) // 2
            x1 = max(0, cx - min_size // 2)
            x2 = min(w, x1 + min_size)
            x1 = max(0, x2 - min_size)
        if (y2 - y1) < min_size:
            cy = (y1 + y2) // 2
            y1 = max(0, cy - min_size // 2)
            y2 = min(h, y1 + min_size)
            y1 = max(0, y2 - min_size)
        if (x2 - x1) < 2 or (y2 - y1) < 2:
            return None
        return frame[y1:y2, x1:x2]

    def _score_crops(self, crops: List[np.ndarray], label: str) -> List[Dict]:
        """Score a batch of same-label crops in a single SigLIP forward pass."""
        import torch
        from PIL import Image

        positive_prompts = self.config.positive_prompts.get(
            label,
            self.config.positive_prompts.get("vehicle", []),
        )
        prompts = list(positive_prompts) + list(self.config.hard_negative_prompts)
        if not prompts:
            return [
                {"positive": 1.0, "negative": 0.0, "top_is_negative": False, "top_prompt": ""}
                for _ in crops
            ]

        pil_images = [Image.fromarray(crop[..., ::-1]) for crop in crops]  # BGR -> RGB
        inputs = self._processor(text=prompts, images=pil_images, return_tensors="pt", padding="max_length")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
            probs_batch = torch.softmax(outputs.logits_per_image, dim=-1).cpu().tolist()

        n_pos = len(positive_prompts)
        n_neg = max(1, len(prompts) - n_pos)
        scores: List[Dict] = []
        for probs in probs_batch:
            positive_total = sum(probs[:n_pos])
            negative_total = sum(probs[n_pos:])
            top_idx = int(np.argmax(probs))
            scores.append({
                "positive": float(positive_total / max(1, n_pos)),
                "negative": float(negative_total / n_neg),
                "positive_total": float(positive_total),
                "negative_total": float(negative_total),
                "top_is_negative": top_idx >= n_pos,
                "top_prompt": prompts[top_idx],
            })
        return scores

    def verify(self, frame: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """Annotate each detection with verification scores; drop rejected ones."""
        if not self._ensure_loaded():
            for det in detections:
                det.setdefault("vlm_positive", None)
                det.setdefault("vlm_negative", None)
                det.setdefault("vlm_verified", True)
            return detections

        # Pick which detections get a SigLIP pass. High-confidence boxes are
        # trusted as-is — the false-positive suspects are the LOW-confidence
        # ones, so the verification budget goes to them first.
        trust_above = float(self.config.auto_trust_above or 0.0)
        candidates: List[int] = []
        crops: Dict[int, np.ndarray] = {}
        for idx, det in enumerate(detections):
            label = det.get("label", "")
            if label in self.config.skip_labels:
                det.setdefault("vlm_positive", None)
                det.setdefault("vlm_negative", None)
                det["vlm_verified"] = True
                continue
            if trust_above > 0 and float(det.get("score", 0.0)) >= trust_above:
                det.setdefault("vlm_positive", None)
                det.setdefault("vlm_negative", None)
                det["vlm_verified"] = True
                continue
            crop = self._crop_box(frame, det)
            if crop is None:
                det["vlm_verified"] = True  # too small to verify, trust L1
                continue
            candidates.append(idx)
            crops[idx] = crop

        # Bound the per-frame SigLIP workload. With SAHI the raw box count can
        # reach the hundreds; lowest scores first — those are most suspicious.
        cap = int(self.config.max_verifications or 0)
        if cap > 0 and len(candidates) > cap:
            candidates.sort(key=lambda i: float(detections[i].get("score", 0.0)))
            for idx in candidates[cap:]:
                detections[idx].setdefault("vlm_positive", None)
                detections[idx].setdefault("vlm_negative", None)
                detections[idx]["vlm_verified"] = True
            candidates = candidates[:cap]
            print(
                f"[VLM] capping verifications: {len(crops)} candidates -> lowest {cap} scored",
                flush=True,
            )

        by_label: Dict[str, List[int]] = {}
        for idx in candidates:
            by_label.setdefault(detections[idx].get("label", ""), []).append(idx)

        rejected: set = set()
        batch_size = max(1, int(self.config.batch_size or 1))
        for label, idxs in by_label.items():
            for start in range(0, len(idxs), batch_size):
                chunk = idxs[start:start + batch_size]
                try:
                    chunk_scores = self._score_crops([crops[i] for i in chunk], label)
                except Exception as exc:
                    print(f"[VLM] scoring failed: {exc}", flush=True)
                    for i in chunk:
                        detections[i]["vlm_verified"] = True
                    continue
                for i, scores in zip(chunk, chunk_scores):
                    det = detections[i]
                    det["vlm_positive"] = round(scores["positive"], 4)
                    det["vlm_negative"] = round(scores["negative"], 4)
                    det["vlm_top_prompt"] = scores["top_prompt"]
                    reject = False
                    if self.config.reject_if_top_is_negative and scores["top_is_negative"]:
                        reject = True
                    # Compare per-prompt averages: rejects only when the negative
                    # group on average outranks the positive group by more than margin.
                    if scores["negative"] - scores["positive"] > self.config.min_positive_margin:
                        reject = True
                    det["vlm_verified"] = not reject
                    if reject:
                        rejected.add(i)

        return [det for idx, det in enumerate(detections) if idx not in rejected]
