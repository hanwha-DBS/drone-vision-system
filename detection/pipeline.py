"""
5-Layer detection pipeline orchestrator.

Order:
    L1 detector(s)  →  L2 SAHI tiling  →  L3 VLM verification
                                          →  L4 ByteTrack + N-of-M
                                          →  L5 active-learning capture

Layers are individually toggleable via the runtime config — this keeps the
A/B comparison story clean (each layer can be disabled to measure its
contribution). A fast-sampling mode keeps the lightweight L1-only path for
sub-2s intervals where VLM/tracker overhead would be unacceptable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .active_learning import ActiveLearningConfig, ActiveLearningSink
from .detector import DetectorConfig, FilterConfig, L1Detector, create_detector, merge_detections, merge_detections_wbf
from .sahi_runner import SahiConfig, run_sahi
from .tracker import TemporalTracker, TrackerConfig
from .vlm_verifier import VLMConfig, VLMVerifier


class NullProgress:
    """No-op load-progress sink.

    Model construction reports which stage it is on so the UI can show a real
    checklist instead of a guessed animation. Callers that don't care (tests,
    scripts/smoke_pipeline.py) get this stub, so the reporting calls below
    never need a `if progress is not None` guard.
    """

    def begin(self, key: str, detail: Optional[str] = None) -> None: ...
    def done(self, key: str, detail: Optional[str] = None) -> None: ...
    def skip(self, key: str, reason: str = "") -> None: ...
    def fail(self, key: str, message: str) -> None: ...


@dataclass
class EnsembleConfig:
    enabled: bool = True
    fusion: str = "wbf"  # "wbf" | "nms"
    nms_iou_threshold: float = 0.42
    prefer_secondary_for_vehicle: bool = True


@dataclass
class PipelineConfig:
    primary: DetectorConfig
    filter: FilterConfig = field(default_factory=FilterConfig)
    secondary: Optional[DetectorConfig] = None
    fast: Optional[DetectorConfig] = None
    fast_interval_threshold_seconds: int = 2
    fast_disable_secondary: bool = True
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    sahi: SahiConfig = field(default_factory=SahiConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    active_learning: ActiveLearningConfig = field(default_factory=ActiveLearningConfig)
    require_confirmation_for_alarm: bool = True


@dataclass
class PipelineResult:
    raw_detections: List[Dict]
    confirmed_detections: List[Dict]
    analysis_mode: str
    layer_timings_ms: Dict[str, float] = field(default_factory=dict)
    active_learning_saved: int = 0


def _time_ms(t0: float) -> float:
    import time
    return (time.perf_counter() - t0) * 1000.0


class DetectionPipeline:
    """Runs all five layers on a single snapshot frame."""

    def __init__(self, config: PipelineConfig, progress: Optional["NullProgress"] = None):
        self.config = config
        self._progress = progress or NullProgress()

        self._progress.begin("primary_detector", config.primary.model_path)
        self._primary = create_detector(config.primary, config.filter)
        self._progress.done("primary_detector")

        if config.secondary and config.ensemble.enabled:
            self._progress.begin("secondary_detector", config.secondary.model_path)
            self._secondary = create_detector(config.secondary, config.filter)
            self._progress.done("secondary_detector")
        else:
            self._secondary = None
            self._progress.skip("secondary_detector", "설정에서 비활성")

        if config.fast and config.fast.model_path:
            self._progress.begin("fast_detector", config.fast.model_path)
            self._fast = create_detector(config.fast, config.filter)
            self._progress.done("fast_detector")
        else:
            self._fast = self._primary
            self._progress.skip("fast_detector", "기본 검출기 재사용")

        self._vlm = VLMVerifier(config.vlm)

        # ByteTrack pulls in `supervision` on construction — around a second,
        # and the last thing between "all detectors loaded" and "ready".
        self._progress.begin("tracker", config.tracker.backend)
        self._tracker = TemporalTracker(config.tracker)
        self._active_learning = ActiveLearningSink(config.active_learning)
        self._progress.done("tracker")

        self._snapshot_step = 0
        # UI profile (fast / balanced / accurate) that gates which layers run.
        # None falls back to the legacy interval-based behavior.
        self._active_profile: Optional[str] = None

    def set_active_profile(self, profile: Optional[str]) -> None:
        p = (profile or "").strip().lower()
        self._active_profile = p if p in ("fast", "balanced", "accurate") else None

    def _profile_plan(self, interval_seconds: int) -> Dict[str, bool]:
        """Resolve which layers a profile permits.

        fast      -> L1 + L3(VLM)
        balanced  -> L1 + L2(SAHI) + L3(VLM) + L4(Tracker)
        accurate  -> full stack L1~L5 (+ secondary/ensemble + VLM)
        None      -> legacy: interval decides fast vs full; all layers allowed.

        VLM runs in EVERY profile: it is the only gate against persistent
        look-alikes (rocks, crushed stone read as "person") — the tracker
        cannot reject those because they never leave the frame, and since the
        batched SigLIP pass costs well under a second it fits even the fast
        budget.
        """
        p = self._active_profile
        if p == "fast":
            return {"fast": True, "sahi": False, "vlm": True, "tracker": False, "secondary": False}
        if p == "balanced":
            return {"fast": False, "sahi": True, "vlm": True, "tracker": True, "secondary": False}
        if p == "accurate":
            return {"fast": False, "sahi": True, "vlm": True, "tracker": True, "secondary": True}
        return {
            "fast": interval_seconds <= self.config.fast_interval_threshold_seconds,
            "sahi": True, "vlm": True, "tracker": True, "secondary": True,
        }

    def reset_session(self) -> None:
        self._tracker.reset()
        self._snapshot_step = 0

    def warmup(self) -> None:
        """Force lazy loaders (SigLIP, ByteTrack) to initialize now so the
        first inference doesn't pay model download / import latency."""
        # Trigger VLM model download/load in the background thread that calls
        # warmup(); without this, the first snapshot blocks for ~30s while
        # SigLIP fetches weights.
        if self.config.vlm.enabled:
            self._progress.begin("vlm", self.config.vlm.model_id)
            try:
                if self._vlm._ensure_loaded():
                    self._progress.done("vlm")
                else:
                    # _ensure_loaded returns False on its own internal failure.
                    self._progress.fail("vlm", "VLM 로드 실패")
            except Exception as exc:
                print(f"[Pipeline] VLM warmup failed: {exc}", flush=True)
                self._progress.fail("vlm", str(exc))
        else:
            self._progress.skip("vlm", "설정에서 비활성")

    def describe_active_layers(self) -> Dict[str, bool]:
        # Effective layers = config-enabled AND permitted by the active profile.
        plan = self._profile_plan(999)
        return {
            "sahi": bool(self.config.sahi.enabled and plan["sahi"]),
            "vlm": bool(self.config.vlm.enabled and plan["vlm"]),
            "tracker": bool(self.config.tracker.enabled and plan["tracker"]),
            "active_learning": bool(self.config.active_learning.enabled and not plan["fast"]),
            "ensemble": bool(self.config.ensemble.enabled and self._secondary is not None and plan["secondary"]),
        }

    @property
    def primary_model_path(self) -> str:
        return self.config.primary.model_path

    @property
    def secondary_model_path(self) -> Optional[str]:
        return self.config.secondary.model_path if self.config.secondary else None

    def _l1_l2(self, frame: np.ndarray, detector: L1Detector, use_sahi: bool) -> List[Dict]:
        if use_sahi and self.config.sahi.enabled:
            return run_sahi(frame, detector, self.config.sahi)
        return detector.predict(frame)

    def run(self, frame: np.ndarray, frame_index: int, interval_seconds: int) -> PipelineResult:
        import time

        plan = self._profile_plan(interval_seconds)
        fast_mode = plan["fast"]
        use_secondary = self._secondary is not None and plan["secondary"]
        timings: Dict[str, float] = {}

        # L1 + L2 (no SAHI in fast mode to stay under time budget)
        t0 = time.perf_counter()
        if fast_mode:
            raw = self._fast.predict(frame)
            analysis_mode = (self._active_profile or "fast") + "-fast"
        else:
            raw = self._l1_l2(frame, self._primary, use_sahi=plan["sahi"])
            if use_secondary:
                raw_secondary = self._l1_l2(frame, self._secondary, use_sahi=plan["sahi"])
                raw.extend(raw_secondary)
            analysis_mode = self._active_profile or ("full-ensemble" if use_secondary else "primary-only")
        timings["l1_l2_ms"] = _time_ms(t0)

        # Ensemble merge (WBF or NMS across primary + secondary)
        t0 = time.perf_counter()
        if self.config.ensemble.enabled and use_secondary:
            merge_fn = merge_detections_wbf if self.config.ensemble.fusion == "wbf" else merge_detections
            raw = merge_fn(
                raw,
                iou_threshold=self.config.ensemble.nms_iou_threshold,
                prefer_secondary_for_vehicle=self.config.ensemble.prefer_secondary_for_vehicle,
            )
        timings["ensemble_ms"] = _time_ms(t0)

        # L3 VLM verification (skipped unless the profile permits it).
        # Note: runs in fast mode too — the profile plan, not fast_mode,
        # decides; false-positive filtering is needed at every speed tier.
        t0 = time.perf_counter()
        if plan["vlm"] and self.config.vlm.enabled:
            verified = self._vlm.verify(frame, raw)
        else:
            for det in raw:
                det.setdefault("vlm_verified", True)
            verified = raw
        timings["vlm_ms"] = _time_ms(t0)

        # L4 Tracker + N-of-M.
        # IMPORTANT: confirm_window is measured in snapshots (pipeline steps),
        # not in raw video-frame indices. With a 3s sampling interval at 30fps,
        # two consecutive snapshots are 90 video frames apart — so any
        # frame-index-based window of size ~5 would always be empty.
        self._snapshot_step += 1
        t0 = time.perf_counter()
        if self.config.tracker.enabled and plan["tracker"]:
            tracked = self._tracker.update(self._snapshot_step, verified)
            confirmed = (
                self._tracker.filter_confirmed(tracked)
                if self.config.require_confirmation_for_alarm
                else tracked
            )
        else:
            for det in verified:
                det.setdefault("track_confirmed", True)
            tracked = verified
            confirmed = verified
        timings["tracker_ms"] = _time_ms(t0)

        # L5 Active learning capture
        t0 = time.perf_counter()
        saved = 0
        if not fast_mode:
            saved = self._active_learning.submit(frame, frame_index, tracked, confirmed)
        timings["active_learning_ms"] = _time_ms(t0)

        return PipelineResult(
            raw_detections=tracked,
            confirmed_detections=confirmed,
            analysis_mode=analysis_mode,
            layer_timings_ms=timings,
            active_learning_saved=saved,
        )


def build_pipeline(runtime_config: Dict, device: str,
                   progress: Optional[NullProgress] = None) -> DetectionPipeline:
    """Translate the runtime_detector.json dict into a DetectionPipeline."""
    progress = progress or NullProgress()
    # Torch device names ("cuda"/"cpu") map onto the config's "gpu"/"cpu" sections.
    device_section = runtime_config["gpu" if device == "cuda" else "cpu"]
    filter_config = FilterConfig(
        allowed_classes=tuple(runtime_config.get("allowed_classes", FilterConfig().allowed_classes)),
        class_aliases=dict(runtime_config.get("class_aliases", FilterConfig().class_aliases)),
        class_min_scores=dict(runtime_config.get("class_min_scores", FilterConfig().class_min_scores)),
        geometry=dict(runtime_config.get("filters", FilterConfig().geometry)),
    )
    use_half = device == "cuda" and runtime_config.get("fp16_on_cuda", True)

    primary = DetectorConfig(
        model_path=device_section["model_path"],
        target_width=device_section["target_width"],
        inference_imgsz=device_section["inference_imgsz"],
        confidence_threshold=device_section["confidence_threshold"],
        architecture=device_section.get("architecture", "auto"),
        half=use_half,
        device=device,
        tag="primary",
        tta=bool(device_section.get("tta", False)),
    )
    secondary = None
    if device_section.get("secondary_model_path"):
        secondary = DetectorConfig(
            model_path=device_section["secondary_model_path"],
            target_width=device_section.get("secondary_target_width", device_section["target_width"]),
            inference_imgsz=device_section.get("secondary_inference_imgsz", device_section["inference_imgsz"]),
            confidence_threshold=device_section.get("secondary_confidence_threshold", device_section["confidence_threshold"]),
            architecture=device_section.get("secondary_architecture", "auto"),
            half=use_half,
            device=device,
            tag="secondary",
        )

    fast_section = runtime_config.get("fast_sampling", {})
    fast = None
    if fast_section.get("model_path"):
        fast = DetectorConfig(
            model_path=fast_section["model_path"],
            target_width=fast_section.get("target_width", primary.target_width),
            inference_imgsz=fast_section.get("inference_imgsz", primary.inference_imgsz),
            confidence_threshold=fast_section.get("confidence_threshold", primary.confidence_threshold),
            architecture=fast_section.get("architecture", "auto"),
            half=use_half,
            device=device,
            tag="fast",
        )

    ensemble_section = runtime_config.get("ensemble", {})
    ensemble = EnsembleConfig(
        enabled=bool(ensemble_section.get("enabled", True)),
        fusion=str(ensemble_section.get("fusion", "wbf")),
        nms_iou_threshold=float(ensemble_section.get("nms_iou_threshold", 0.42)),
        prefer_secondary_for_vehicle=bool(ensemble_section.get("prefer_secondary_for_vehicle", True)),
    )

    sahi_section = runtime_config.get("sahi", {})
    sahi_config = SahiConfig(
        enabled=bool(sahi_section.get("enabled", True)),
        fusion=str(sahi_section.get("fusion", "wbf")),
        tile_size=int(sahi_section.get("tile_size", 768)),
        overlap=float(sahi_section.get("overlap", 0.25)),
        min_tile_score=float(sahi_section.get("min_tile_score", 0.0)),
        nms_iou_threshold=float(sahi_section.get("nms_iou_threshold", 0.5)),
        max_tiles=int(sahi_section.get("max_tiles", 64)),
    )

    vlm_section = runtime_config.get("vlm", {})
    vlm_config = VLMConfig(
        enabled=bool(vlm_section.get("enabled", True)),
        model_id=str(vlm_section.get("model_id", "google/siglip-base-patch16-224")),
        device=device,
        crop_padding=float(vlm_section.get("crop_padding", 0.10)),
        min_crop_size=int(vlm_section.get("min_crop_size", 32)),
        reject_if_top_is_negative=bool(vlm_section.get("reject_if_top_is_negative", True)),
        min_positive_margin=float(vlm_section.get("min_positive_margin", 0.05)),
        max_verifications=int(vlm_section.get("max_verifications", 40)),
        auto_trust_above=float(vlm_section.get("auto_trust_above", 0.45)),
        batch_size=int(vlm_section.get("batch_size", 16)),
        positive_prompts=dict(vlm_section.get("positive_prompts", VLMConfig().positive_prompts)),
        hard_negative_prompts=list(vlm_section.get("hard_negative_prompts", VLMConfig().hard_negative_prompts)),
        skip_labels=tuple(vlm_section.get("skip_labels", ())),
    )

    tracker_section = runtime_config.get("tracker", {})
    tracker_config = TrackerConfig(
        enabled=bool(tracker_section.get("enabled", True)),
        backend=str(tracker_section.get("backend", "bytetrack")),
        track_activation_threshold=float(tracker_section.get("track_activation_threshold", 0.15)),
        lost_track_buffer=int(tracker_section.get("lost_track_buffer", 30)),
        minimum_matching_threshold=float(tracker_section.get("minimum_matching_threshold", 0.6)),
        confirm_window=int(tracker_section.get("confirm_window", 5)),
        confirm_min_hits=int(tracker_section.get("confirm_min_hits", 3)),
        fallback_iou=float(tracker_section.get("fallback_iou", 0.3)),
    )

    al_section = runtime_config.get("active_learning", {})
    al_config = ActiveLearningConfig(
        enabled=bool(al_section.get("enabled", True)),
        root_directory=str(al_section.get("root_directory", "datasets/active_learning")),
        confidence_min=float(al_section.get("confidence_min", 0.30)),
        confidence_max=float(al_section.get("confidence_max", 0.60)),
        crop_padding=float(al_section.get("crop_padding", 0.15)),
        save_vlm_rejected=bool(al_section.get("save_vlm_rejected", True)),
        save_unconfirmed_tracks=bool(al_section.get("save_unconfirmed_tracks", True)),
        max_crops_per_snapshot=int(al_section.get("max_crops_per_snapshot", 6)),
    )

    # ------------------------------------------------------------------
    # CPU safety net.
    #
    # SAHI + ensemble + VLM on CPU multiplies inference time by 10-30x and
    # makes the first snapshot effectively never finish. We auto-disable
    # the heavy layers unless the operator explicitly opts in via the
    # cpu_runtime override block.
    # ------------------------------------------------------------------
    cpu_overrides = runtime_config.get("cpu_runtime", {})
    if device != "cuda":
        if not bool(cpu_overrides.get("allow_sahi", False)):
            sahi_config.enabled = False
        if not bool(cpu_overrides.get("allow_vlm", False)):
            vlm_config.enabled = False
            # Claim the stage here rather than in warmup(), so the UI shows
            # *why* it was skipped (CPU auto-disable) instead of the generic
            # "disabled in config" reason.
            progress.skip("vlm", "CPU 모드 — 자동 비활성")
        if not bool(cpu_overrides.get("allow_secondary", False)):
            secondary = None
            ensemble.enabled = False
            progress.skip("secondary_detector", "CPU 모드 — 자동 비활성")
        # On CPU the N-of-M rule is fine to keep, but the user wants to see
        # boxes immediately, so we lower the default to 1-of-3 unless they
        # override.
        if "confirm_min_hits" not in tracker_section:
            tracker_config.confirm_min_hits = max(1, tracker_config.confirm_min_hits - 1)

    pipeline_config = PipelineConfig(
        primary=primary,
        filter=filter_config,
        secondary=secondary,
        fast=fast,
        fast_interval_threshold_seconds=int(fast_section.get("interval_threshold_seconds", 2)),
        fast_disable_secondary=bool(fast_section.get("disable_secondary", True)),
        ensemble=ensemble,
        sahi=sahi_config,
        vlm=vlm_config,
        tracker=tracker_config,
        active_learning=al_config,
        require_confirmation_for_alarm=bool(runtime_config.get("require_confirmation_for_alarm", True)),
    )
    pipeline = DetectionPipeline(pipeline_config, progress=progress)

    layers = pipeline.describe_active_layers()
    print(
        f"[Pipeline] device={device} layers="
        + ", ".join(f"{k}={'on' if v else 'off'}" for k, v in layers.items()),
        flush=True,
    )
    if device != "cuda":
        print(
            "[Pipeline] CPU mode — SAHI/VLM/secondary auto-disabled. "
            "Override with cpu_runtime.allow_sahi / allow_vlm / allow_secondary in config.",
            flush=True,
        )
    return pipeline
