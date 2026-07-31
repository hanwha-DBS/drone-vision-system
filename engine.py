"""
Drone Vision System — AI engine (5-Layer Safety Stack).

Responsibilities of this module (slimmed down vs. the legacy engine):
  - Load runtime configuration and build a DetectionPipeline.
  - Manage the ZMQ REQ/REP server that the Electron UI talks to.
  - Run a VideoAnalysisSession with 4 worker threads:
        video_stream_worker  -> reads frames from the source
        snapshot_scheduler   -> samples a frame every N seconds
        inference_worker     -> invokes the 5-layer pipeline
        result_handler       -> draws boxes, encodes JPEG, saves alarms

All detection logic now lives in the `detection/` package.
"""
from __future__ import annotations

import base64
import json
import math
import os
import queue
import re
import threading
import time
import traceback
from dataclasses import asdict
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional

import cv2
import zmq

# torch / ultralytics / the detection package cost ~4s warm and several times
# that on a cold start. Importing them here would push the whole cost in front
# of the ZMQ bind, so the UI could not even connect — let alone show progress —
# during the slowest part of startup. They are imported inside load_pipeline()
# instead, as the first reported loading stage.
if TYPE_CHECKING:
    from detection import DetectionPipeline, PipelineResult


DEFAULT_RUNTIME_CONFIG: Dict = {
    "profile_name": "snapshot-detection-v3-5layer",
    "cpu": {
        "model_path": "yolov8l.pt",
        "target_width": 1536,
        "confidence_threshold": 0.09,
        "inference_imgsz": 1536,
        "secondary_model_path": "rtdetr-l.pt",
        "secondary_target_width": 1536,
        "secondary_confidence_threshold": 0.08,
        "secondary_inference_imgsz": 1536,
    },
    "gpu": {
        "model_path": "yolov8l.pt",
        "target_width": 1792,
        "confidence_threshold": 0.08,
        "inference_imgsz": 1792,
        "secondary_model_path": "rtdetr-l.pt",
        "secondary_target_width": 1792,
        "secondary_confidence_threshold": 0.07,
        "secondary_inference_imgsz": 1792,
    },
    "fast_sampling": {
        "interval_threshold_seconds": 2,
        "model_path": "yolov8m.pt",
        "target_width": 1280,
        "confidence_threshold": 0.10,
        "inference_imgsz": 1280,
        "disable_secondary": True,
    },
    "allowed_classes": ["person", "car", "truck", "bus", "vehicle"],
    "class_aliases": {
        "person": "person",
        "car": "vehicle",
        "truck": "vehicle",
        "bus": "vehicle",
        "dump_truck": "vehicle",
        "tanker_truck": "vehicle",
        "suv": "vehicle",
        "van": "vehicle",
        "excavator": "vehicle",
    },
    "class_min_scores": {"person": 0.12, "vehicle": 0.10},
    "filters": {
        "person": {"min_area_ratio": 0.000003, "max_area_ratio": 0.05, "min_aspect_ratio": 0.08, "max_aspect_ratio": 1.6},
        "vehicle": {"min_area_ratio": 0.00002, "max_area_ratio": 0.30, "min_aspect_ratio": 0.18, "max_aspect_ratio": 7.2},
    },
    "ensemble": {"enabled": True, "nms_iou_threshold": 0.42, "prefer_secondary_for_vehicle": True},
    "sahi": {"enabled": True, "tile_size": 768, "overlap": 0.25, "nms_iou_threshold": 0.5, "max_tiles": 36},
    "vlm": {
        "enabled": True,
        "model_id": "google/siglip-base-patch16-224",
        "crop_padding": 0.10,
        "min_crop_size": 32,
        "reject_if_top_is_negative": False,
        "min_positive_margin": 0.0,
    },
    "tracker": {
        "enabled": True,
        "backend": "bytetrack",
        "track_activation_threshold": 0.15,
        "lost_track_buffer": 30,
        "minimum_matching_threshold": 0.6,
        "confirm_window": 5,
        "confirm_min_hits": 3,
    },
    "active_learning": {
        "enabled": True,
        "root_directory": "datasets/active_learning",
        "confidence_min": 0.30,
        "confidence_max": 0.60,
        "save_vlm_rejected": True,
        "save_unconfirmed_tracks": True,
        "max_crops_per_snapshot": 6,
    },
    "preview_jpeg_quality": 88,
    "preview_max_width": 1400,
    "sampling": {
        "default_interval_seconds": 3,
        "min_interval_seconds": 1,
        "max_interval_seconds": 10,
        "snapshot_queue_size": 2,
        "result_queue_size": 4,
        # File playback only: how much RAM the not-yet-analyzed snapshot backlog
        # may hold. Capture runs on the sampling clock and never waits for
        # inference, so a slow model grows this backlog; past the budget the
        # oldest pending snapshot is dropped (see snapshot_scheduler).
        "max_pending_snapshot_mb": 1536,
    },
    "event": {"enabled": True, "save_directory": "results/events", "max_recent_events": 200},
    "fp16_on_cuda": True,
    "cudnn_benchmark": True,
    # If True, only N-of-M-confirmed tracks raise alarms / events.
    # Default False — the operator wants every detection saved immediately;
    # confirmation is exposed as a UI badge ("✓") instead of a gating filter.
    "require_confirmation_for_alarm": False,
}


# Network stream schemes that OpenCV's VideoCapture can open directly
# (RTSP/RTMP/HTTP-MJPEG/HLS/UDP/SRT, etc.). These are not local files, so the
# `os.path.exists` check must be skipped for them.
_STREAM_SCHEME_RE = re.compile(
    r"^(rtsp|rtsps|rtmp|rtmps|http|https|udp|tcp|srt|mms|mmsh)://",
    re.IGNORECASE,
)


def is_stream_source(path: Optional[str]) -> bool:
    return bool(path) and bool(_STREAM_SCHEME_RE.match(str(path)))


def _deep_merge(base: Dict, override: Dict) -> Dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs", "runtime_detector.json")


def load_runtime_config() -> Dict:
    if not os.path.exists(CONFIG_PATH):
        return DEFAULT_RUNTIME_CONFIG
    with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    return _deep_merge(DEFAULT_RUNTIME_CONFIG, loaded)


# Resolved during the device_probe loading stage (needs torch). Everything
# that reads it treats None as "not yet known" and falls back to the CPU
# config section, which is also the safe default for the settings view.
device: Optional[str] = None
runtime_config = load_runtime_config()

SAMPLING_CONFIG = runtime_config["sampling"]
EVENT_CONFIG = runtime_config["event"]
PREVIEW_JPEG_QUALITY = int(runtime_config["preview_jpeg_quality"])
PREVIEW_MAX_WIDTH = int(runtime_config["preview_max_width"])

# Cap on the capture/analysis gauge history. One char per capture, so 3600
# covers a 3-hour session at a 3s interval and costs ~3.6KB per state poll.
SNAPSHOT_MARK_LIMIT = 3600

# Memory ceiling for the pending-snapshot backlog on file playback (bytes).
PENDING_SNAPSHOT_BUDGET = int(SAMPLING_CONFIG.get("max_pending_snapshot_mb", 1536)) * 1024 * 1024

# Delay before the sampler takes its first frame (see snapshot_scheduler); the
# expected-slot count has to start from the same offset to line up.
SNAPSHOT_FIRST_DELAY = 0.35

pipeline: Optional[DetectionPipeline] = None
model_loaded = False
model_loading = False

video_lock = threading.Lock()
session_lock = threading.Lock()
current_session: Optional["VideoAnalysisSession"] = None

video_state: Dict = {
    "status": "idle",
    "message": "video idle",
    "video_path": None,
    "is_stream": False,
    "live_frame_base64": None,
    "completed": False,
    "started_at": None,
    "device": device,
    "profile_name": runtime_config.get("profile_name"),
    "model_path": None,
    "secondary_model_path": None,
    "analysis_interval_seconds": 0,
    "processed_frames": 0,
    "total_frames": 0,
    "progress": 0.0,
    "paused": False,
    "source_fps": 0.0,
    "stream_fps": 0.0,
    "inference_fps": 0.0,
    "current_video_time": 0.0,
    "frame_width": 0,
    "frame_height": 0,
    "snapshot_count": 0,
    "analysis_count": 0,
    "snapshot_marks": "",
    "expected_snapshot_total": 0,
    "latest_snapshot_base64": None,
    "latest_snapshot_timestamp": None,
    "latest_snapshot_frame_index": 0,
    # Capture order (1-based) of each panel. The RAW panel follows the sampling
    # clock while the analyzed panel trails it by the inference backlog, so both
    # say which capture they are showing.
    "latest_snapshot_seq": 0,
    "latest_result_base64": None,
    "latest_result": None,
    "latest_result_frame_index": 0,
    "latest_result_timestamp": None,
    "latest_result_seq": 0,
    "latest_detections": [],
    "analysis_pending": False,
    # Sampled frames waiting for inference, and captures given up on because the
    # backlog outgrew its memory budget. Capture never waits for analysis, so the
    # UI needs these to explain why analysis trails the capture count.
    "analysis_backlog": 0,
    "dropped_snapshots": 0,
    # True while a frame is inside pipeline.run(). The backlog alone reads empty
    # for the last frame of a clip, which made the UI claim playback was still
    # running for the length of one inference after the video had ended.
    "analysis_in_flight": False,
    "analysis_mode": None,
    "current_persons": 0,
    "current_vehicle_like_objects": 0,
    "max_persons": 0,
    "max_vehicle_like_objects": 0,
    "risk_level": "clear",
    "risk_summary": "현재 감지된 사람이나 차량이 없습니다.",
    "event_count": 0,
    "latest_event_path": None,
    "recent_events": [],
    "result_video_path": None,
    "pending_api_events": 0,
    "event_directory": None,
    "active_learning_saved": 0,
    "layer_timings_ms": {},
}


def set_video_state(**updates) -> None:
    with video_lock:
        video_state.update(updates)


def snapshot_video_state() -> Dict:
    with video_lock:
        state = dict(video_state)
        if isinstance(state.get("latest_result"), dict):
            state["latest_result"] = dict(state["latest_result"])
        state["recent_events"] = [dict(item) for item in state.get("recent_events", [])]
        state["layer_timings_ms"] = dict(state.get("layer_timings_ms", {}))
        return state


# ===========================================================================
# Model loading progress
#
# Startup is a fixed, knowable sequence of stages, so the UI can show a real
# checklist instead of a timer-driven guess. Weights are relative costs
# measured on this project's venv (warm import cache):
#
#   runtime import  ~4.1s   (cv2 0.14 + torch 1.65 + detection 2.28)
#   primary/fast    local .pt weights, 53MB / 44MB
#   secondary/vlm   pull `transformers` (~16.5s on first import) plus a
#                   HuggingFace download on first run — by far the largest
#                   stages, and both auto-disabled on CPU.
#
# Skipped stages drop out of the denominator, so a CPU start legitimately
# reaches 100% without ever touching the two heavy stages.
# ===========================================================================
LOADING_STAGE_SPECS = (
    ("runtime_import", "런타임 로드", "torch · ultralytics", 12.0),
    ("device_probe", "실행 디바이스 판정", "", 1.0),
    ("primary_detector", "기본 검출기", "", 10.0),
    ("secondary_detector", "보조 검출기", "D-FINE", 30.0),
    ("fast_detector", "고속 검출기", "", 8.0),
    ("tracker", "추적기 · 학습 데이터 수집", "ByteTrack", 4.0),
    ("vlm", "VLM 검증기", "SigLIP2", 25.0),
)


class LoadingProgress:
    """Thread-safe record of where model loading currently is.

    Written by the loader thread (and, via the same interface, by
    detection.pipeline while it constructs detectors); read by the ZMQ
    handler thread on every `status` request.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stages = [
            {"key": key, "label": label, "detail": detail, "weight": weight,
             "status": "pending", "ms": 0}
            for key, label, detail, weight in LOADING_STAGE_SPECS
        ]
        self._by_key = {stage["key"]: stage for stage in self._stages}
        self._started_at = time.time()
        self._finished_at: Optional[float] = None
        self._running_key: Optional[str] = None
        self._running_since = 0.0
        self._sub_fraction = 0.0
        self._sub_detail: Optional[str] = None
        self._error: Optional[str] = None
        # Model ids that are not in the HF cache yet, i.e. this really is a
        # first run and the long stages will include a download.
        self.downloads_pending: List[str] = []

    # -- writer side (loader thread) ---------------------------------------
    def begin(self, key: str, detail: Optional[str] = None) -> None:
        with self._lock:
            stage = self._by_key.get(key)
            if stage is None or stage["status"] != "pending":
                return
            stage["status"] = "running"
            if detail:
                stage["detail"] = detail
            self._running_key = key
            self._running_since = time.perf_counter()
            self._sub_fraction = 0.0
            self._sub_detail = None
        print(f"[AI Engine] loading: {key}" + (f" ({detail})" if detail else ""), flush=True)

    def _settle(self, key: str, status: str, detail: Optional[str]) -> None:
        with self._lock:
            stage = self._by_key.get(key)
            if stage is None or stage["status"] in ("done", "skipped", "failed"):
                return
            was_running = stage["status"] == "running"
            stage["status"] = status
            if detail:
                stage["detail"] = detail
            if was_running:
                stage["ms"] = int((time.perf_counter() - self._running_since) * 1000)
                self._running_key = None
                self._sub_fraction = 0.0
                self._sub_detail = None

    def done(self, key: str, detail: Optional[str] = None) -> None:
        self._settle(key, "done", detail)

    def skip(self, key: str, reason: str = "") -> None:
        self._settle(key, "skipped", reason or None)

    def fail(self, key: str, message: str) -> None:
        # A single non-fatal stage failure (e.g. VLM weights unreachable). The
        # pipeline still runs without that layer, so this is not `set_error`.
        self._settle(key, "failed", message)

    def set_sub_progress(self, fraction: Optional[float], detail: Optional[str] = None) -> None:
        with self._lock:
            if fraction is None:
                self._sub_fraction = 0.0
                self._sub_detail = None
                return
            self._sub_fraction = max(0.0, min(1.0, float(fraction)))
            self._sub_detail = detail

    def set_error(self, message: str) -> None:
        with self._lock:
            self._error = message
            if self._running_key:
                stage = self._by_key[self._running_key]
                if stage["status"] == "running":
                    stage["status"] = "failed"
                    stage["ms"] = int((time.perf_counter() - self._running_since) * 1000)
                self._running_key = None

    def finish(self) -> None:
        with self._lock:
            self._finished_at = time.time()

    # -- reader side (ZMQ thread) ------------------------------------------
    def snapshot(self) -> Dict:
        with self._lock:
            active = [s for s in self._stages if s["status"] != "skipped"]
            total_weight = sum(s["weight"] for s in active) or 1.0
            accumulated = sum(
                s["weight"] for s in active if s["status"] in ("done", "failed")
            )
            if self._running_key:
                accumulated += self._by_key[self._running_key]["weight"] * self._sub_fraction

            complete = self._finished_at is not None
            percent = 100.0 if complete else min(99.0, 100.0 * accumulated / total_weight)

            if self._error:
                state = "error"
            elif complete:
                state = "ready"
            else:
                state = "loading"

            running = self._by_key.get(self._running_key) if self._running_key else None
            return {
                "state": state,
                "percent": round(percent, 1),
                "stage": self._running_key,
                "stage_label": running["label"] if running else None,
                "stage_detail": self._sub_detail or (running["detail"] if running else None),
                "elapsed_ms": int(((self._finished_at or time.time()) - self._started_at) * 1000),
                "error": self._error,
                "downloads_pending": list(self.downloads_pending),
                "stages": [
                    {"key": s["key"], "label": s["label"], "detail": s["detail"],
                     "status": s["status"], "ms": s["ms"]}
                    for s in self._stages
                ],
            }


loading_progress = LoadingProgress()


def _hf_cache_has(model_id: str) -> bool:
    """True if `model_id` is already in the local HuggingFace cache.

    Used to tell a first-run install (weights must be downloaded, minutes)
    apart from a warm restart (seconds) before either has started.
    """
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        folder = "models--" + model_id.replace("/", "--")
        return os.path.isdir(os.path.join(HF_HUB_CACHE, folder))
    except Exception:
        return True  # unknown -> don't promise a long download


def _install_hf_download_hook(progress: LoadingProgress) -> None:
    """Route huggingface_hub's tqdm into `progress` as sub-stage progress.

    Without this the download stages are a dead spinner for minutes on a first
    run. Best-effort: hub internals are not a stable API, so any failure here
    just leaves the stage indeterminate.
    """
    try:
        import huggingface_hub.file_download as file_download

        base_tqdm = file_download.tqdm
        if getattr(base_tqdm, "_dvs_hooked", False):
            return

        class _ReportingTqdm(base_tqdm):  # type: ignore[misc, valid-type]
            _dvs_hooked = True

            def update(self, n=1):
                result = super().update(n)
                try:
                    if self.total:
                        progress.set_sub_progress(
                            self.n / self.total,
                            f"{self.n / 1e6:.0f} / {self.total / 1e6:.0f} MB 다운로드",
                        )
                except Exception:
                    pass
                return result

            def close(self):
                try:
                    progress.set_sub_progress(None)
                except Exception:
                    pass
                return super().close()

        file_download.tqdm = _ReportingTqdm
    except Exception as exc:
        print(f"[AI Engine] hf progress hook unavailable: {exc}", flush=True)


def load_pipeline() -> None:
    global pipeline, model_loaded, model_loading, device
    try:
        model_loading = True

        # --- stage 1: heavy runtime import (deferred from module import) ---
        loading_progress.begin("runtime_import")
        import torch  # noqa: PLC0415 — deliberately deferred, see module header
        from detection import build_pipeline  # noqa: PLC0415

        loading_progress.done("runtime_import")

        # --- stage 2: device probe ---
        loading_progress.begin("device_probe")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and runtime_config.get("cudnn_benchmark", True):
            try:
                torch.backends.cudnn.benchmark = True
            except Exception:
                pass
        set_video_state(device=device)

        # Only the GPU profile pulls the two HuggingFace models, so what still
        # needs downloading can only be answered once the device is known.
        if device == "cuda":
            candidates = [
                runtime_config.get("gpu", {}).get("secondary_model_path"),
                runtime_config.get("vlm", {}).get("model_id"),
            ]
            loading_progress.downloads_pending = [
                model_id for model_id in candidates
                if model_id and "/" in model_id and not _hf_cache_has(model_id)
            ]
            _install_hf_download_hook(loading_progress)
        loading_progress.done("device_probe", device.upper())
        print(f"[AI Engine] device: {device}", flush=True)
        print(f"[AI Engine] profile: {runtime_config.get('profile_name')}", flush=True)

        # --- stages 3-5: detectors (reported from inside build_pipeline) ---
        pipeline = build_pipeline(runtime_config, device, progress=loading_progress)
        print(f"[AI Engine] primary: {pipeline.primary_model_path}", flush=True)
        print(f"[AI Engine] secondary: {pipeline.secondary_model_path}", flush=True)

        # --- stage 6: VLM (reported from inside warmup) ---
        print("[AI Engine] warming up heavy layers...", flush=True)
        pipeline.warmup()

        model_loaded = True
        loading_progress.finish()
        print("[AI Engine] pipeline ready", flush=True)
    except Exception as exc:
        # Previously this only logged, leaving model_loaded False forever while
        # the UI kept animating a progress bar that would never finish.
        loading_progress.set_error(str(exc))
        print(f"[AI Engine ERR] pipeline load failed: {exc}", flush=True)
        traceback.print_exc()
    finally:
        model_loading = False


def encode_image_base64(frame) -> Optional[str]:
    preview = frame
    height, width = frame.shape[:2]
    if PREVIEW_MAX_WIDTH and width > PREVIEW_MAX_WIDTH:
        scale = PREVIEW_MAX_WIDTH / float(width)
        preview = cv2.resize(
            frame,
            (PREVIEW_MAX_WIDTH, max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, buffer = cv2.imencode(
        ".jpg",
        preview,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(PREVIEW_JPEG_QUALITY)],
    )
    if not ok:
        return None
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def get_box_color(label: str):
    if label == "person":
        return (0, 255, 0)
    return (0, 180, 255)


def draw_detections(frame, detections: List[Dict]):
    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        label = det.get("display_label", det.get("label", ""))
        score = det.get("score", 0.0)
        color = get_box_color(det.get("label", ""))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        track_id = det.get("track_id")
        suffix = f" #{track_id}" if track_id is not None else ""

        # Label with a filled background plate for legibility against busy
        # backgrounds. Font scale bumped ~10% (0.58 -> 0.64).
        text = f"{label} {score:.2f}{suffix}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.64
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        ty = max(y1 - 8, th + 6)
        cv2.rectangle(
            annotated,
            (x1, ty - th - 6),
            (x1 + tw + 10, ty + baseline - 1),
            color,
            -1,
        )
        # Pick black/white text based on plate luminance for max contrast.
        b, g, r = (int(c) for c in color)
        luminance = 0.114 * b + 0.587 * g + 0.299 * r
        text_color = (20, 20, 20) if luminance > 150 else (255, 255, 255)
        cv2.putText(
            annotated,
            text,
            (x1 + 5, ty - 2),
            font,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )
    return annotated


def infer_risk(detections: List[Dict]):
    has_person = any(d["label"] == "person" for d in detections)
    has_vehicle = any(d["label"] == "vehicle" for d in detections)
    if has_person and has_vehicle:
        return has_person, has_vehicle, "critical", "사람과 차량이 동시에 감지되었습니다."
    if has_person:
        return has_person, has_vehicle, "warning", "사람이 감지되었습니다."
    if has_vehicle:
        return has_person, has_vehicle, "notice", "차량이 감지되었습니다."
    return has_person, has_vehicle, "clear", "현재 감지된 사람이나 차량이 없습니다."


def serialize_detections(detections: List[Dict]) -> List[Dict]:
    payload = []
    for item in detections:
        payload.append({
            "label": item["label"],
            "display_label": item.get("display_label", item["label"]),
            "score": round(float(item["score"]), 4),
            "x1": item["x1"],
            "y1": item["y1"],
            "x2": item["x2"],
            "y2": item["y2"],
            "track_id": item.get("track_id"),
            "track_hits": item.get("track_hits"),
            "track_confirmed": item.get("track_confirmed"),
            "vlm_positive": item.get("vlm_positive"),
            "vlm_negative": item.get("vlm_negative"),
        })
    return payload


def build_result_payload(timestamp_text: str, detections: List[Dict], image_path: str, risk_level: str, has_person: bool, has_vehicle: bool) -> Dict:
    persons = sum(1 for d in detections if d["label"] == "person")
    vehicles = sum(1 for d in detections if d["label"] == "vehicle")
    return {
        "timestamp": timestamp_text,
        "detections": serialize_detections(detections),
        "image_path": image_path,
        "risk_level": risk_level,
        "has_person": has_person,
        "has_vehicle": has_vehicle,
        "person_count": persons,
        "vehicle_count": vehicles,
    }


def put_latest(target_queue: "queue.Queue", item) -> None:
    while True:
        try:
            target_queue.put_nowait(item)
            return
        except queue.Full:
            try:
                target_queue.get_nowait()
            except queue.Empty:
                return


def frame_nbytes(frame) -> int:
    try:
        return int(frame.nbytes)
    except Exception:
        return 0


class VideoAnalysisSession:
    def __init__(self, video_path: str, interval_seconds: int):
        min_interval = int(SAMPLING_CONFIG["min_interval_seconds"])
        max_interval = int(SAMPLING_CONFIG["max_interval_seconds"])
        self.video_path = video_path
        self.is_stream = is_stream_source(video_path)
        self.interval_seconds = max(min_interval, min(max_interval, int(interval_seconds)))
        self.stop_event = threading.Event()
        # Set alongside stop_event when the not-yet-analyzed backlog should be
        # thrown away instead of finished (see request_stop).
        self.abandon_event = threading.Event()
        # Stage handoffs: each consumer keeps draining until its *producer* is
        # gone, not until the session is asked to stop. Keying the drain off
        # stop_event instead let the result handler quit the moment the queue
        # looked empty at end-of-file, throwing away every analysis that
        # finished after playback ended.
        self.capture_done_event = threading.Event()
        self.analysis_done_event = threading.Event()
        # Pause/seek control for file playback (no-op for live streams).
        self.pause_event = threading.Event()
        self.seek_lock = threading.Lock()
        self.seek_request: Optional[int] = None
        # When True (manual file analysis), reaching the end pauses at the last
        # frame and keeps the session alive so the user can scrub back, resume,
        # or capture — instead of terminating the session.
        self.pause_at_end = False
        self.analysis_profile: Optional[str] = None
        self.final_lock = threading.Lock()
        self.frame_lock = threading.Lock()
        self.finalized = False
        self.final_status = "completed"
        self.final_message = "video analysis completed"

        self.capture = None
        self.latest_frame = None
        self.latest_frame_index = 0
        self.latest_frame_time = 0.0
        self.source_fps = 0.0
        self.stream_fps = 0.0
        self.total_frames = 0
        self.frame_width = 0
        self.frame_height = 0

        # Live streams keep the small "freshest only" queue — they can never
        # catch up, so a stale frame is worse than a dropped one.
        #
        # File playback used a bounded queue that the sampler *blocked* on when
        # full, which coupled the capture cadence to inference latency: with a
        # 3s interval and ~6s of inference, a 90s clip yielded ~16 captures
        # instead of ~30, and the UI looked like "shoot -> wait -> shoot". Both
        # queues are unbounded here so capture stays on the sampling clock and
        # inference simply trails behind, draining after playback ends. Memory
        # is bounded by PENDING_SNAPSHOT_BUDGET instead (see snapshot_scheduler).
        _snap_qsize = int(SAMPLING_CONFIG["snapshot_queue_size"])
        self.snapshot_queue: "queue.Queue" = queue.Queue(maxsize=_snap_qsize if self.is_stream else 0)
        self.result_queue: "queue.Queue" = queue.Queue(
            maxsize=int(SAMPLING_CONFIG["result_queue_size"]) if self.is_stream else 0
        )
        self.api_event_queue: "queue.Queue" = queue.Queue(maxsize=64)

        # Bytes of decoded frames sitting in snapshot_queue, so the sampler can
        # enforce a memory budget without blocking on inference.
        self.pending_lock = threading.Lock()
        self.pending_bytes = 0
        self.dropped_snapshots = 0

        self.snapshot_count = 0
        self.analysis_count = 0
        self.max_persons = 0
        self.max_vehicles = 0
        self.inference_started_at: Optional[float] = None
        self.active_learning_saved = 0

        self.recent_events: List[Dict] = []
        self.latest_event_path: Optional[str] = None
        self.event_count = 0
        self.threads: List[threading.Thread] = []

        # One character per capture slot — the capture/analysis gauge renders
        # this directly. Kept as a compact string (not a list of dicts) because
        # the whole video_state is serialized on every 250ms UI poll; per-cell
        # details come from recent_events instead.
        #   0 = clear  1 = vehicle  2 = person  3 = person+vehicle  4 = skipped
        # File playback indexes by capture order so a slot that never got
        # analyzed stays visible; streams append per analysis (they drop
        # constantly by design, so capture order carries no meaning there).
        self.snapshot_marks: List[str] = []
        self.snapshot_mark_offset = 0  # capture seq of snapshot_marks[0]
        # Total snapshots this clip will produce, known up front for files
        # (frames / fps / interval). 0 for live streams — no end to divide by.
        self.expected_snapshot_total = 0

    def update_latest_frame(self, frame, frame_index: int, frame_time: float) -> None:
        with self.frame_lock:
            self.latest_frame = frame.copy()
            self.latest_frame_index = frame_index
            self.latest_frame_time = frame_time

    def get_latest_frame_snapshot(self):
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            return {
                "frame": self.latest_frame.copy(),
                "frame_index": self.latest_frame_index,
                "video_time": self.latest_frame_time,
            }

    def enqueue_snapshot(self, snapshot: Dict) -> None:
        """Queue a sampled frame for analysis without ever waiting on inference.

        File playback only. The backlog is capped by bytes, not by item count:
        past the budget the oldest still-unanalyzed snapshot is discarded so
        sampling keeps covering the rest of the clip and the freshest evidence
        survives. Dropped slots are reported, never silently swallowed.
        """
        size = frame_nbytes(snapshot["frame"])
        with self.pending_lock:
            while self.pending_bytes + size > PENDING_SNAPSHOT_BUDGET and self.pending_bytes > 0:
                try:
                    stale = self.snapshot_queue.get_nowait()
                except queue.Empty:
                    break
                self.pending_bytes = max(0, self.pending_bytes - frame_nbytes(stale["frame"]))
                self.dropped_snapshots += 1
                print(
                    f"[Pipeline] !! dropped pending snapshot seq={stale.get('seq')} "
                    f"(backlog over {PENDING_SNAPSHOT_BUDGET // (1024 * 1024)}MB, "
                    f"total dropped={self.dropped_snapshots})",
                    flush=True,
                )
            self.pending_bytes += size
        self.snapshot_queue.put_nowait(snapshot)

    def release_snapshot(self, snapshot: Dict) -> None:
        with self.pending_lock:
            self.pending_bytes = max(0, self.pending_bytes - frame_nbytes(snapshot["frame"]))

    def record_snapshot_mark(self, seq: Optional[int], mark: str) -> None:
        """Place a result's gauge cell at its capture position (see snapshot_marks)."""
        marks = self.snapshot_marks
        if seq is None:
            marks.append(mark)
        else:
            index = seq - self.snapshot_mark_offset
            if index < 0:
                return  # already trimmed out of the gauge window
            # Results arrive in capture order, so any still-empty earlier slot
            # is one that was dropped rather than one that is still running.
            while len(marks) <= index:
                marks.append("4")
            marks[index] = mark
        if len(marks) > SNAPSHOT_MARK_LIMIT:
            overflow = len(marks) - SNAPSHOT_MARK_LIMIT
            del marks[:overflow]
            self.snapshot_mark_offset += overflow

    def request_seek(self, frame_target: int) -> None:
        hi = max(0, self.total_frames - 1) if self.total_frames else frame_target
        with self.seek_lock:
            self.seek_request = max(0, min(int(frame_target), hi))

    def pop_seek_request(self) -> Optional[int]:
        with self.seek_lock:
            target = self.seek_request
            self.seek_request = None
            return target

    def request_stop(self, status: str, message: str, drain: bool = False) -> None:
        """Ask the session to end.

        `drain=True` (reaching the end of a file) lets the workers finish the
        snapshots already captured — with capture decoupled from inference the
        backlog is where the tail of the clip lives, so throwing it away would
        lose real analysis. Every other stop — user stop, a new video, an error
        — abandons it, so pressing 중지 does not sit through a queue of
        multi-second inferences before the session actually lets go.
        """
        with self.final_lock:
            if self.finalized:
                return
            self.final_status = status
            self.final_message = message
        if not drain:
            self.abandon_event.set()
            self.discard_backlog()
        self.stop_event.set()

    def keep_working(self, work_queue: "queue.Queue", upstream_done: threading.Event) -> bool:
        """True while a worker should keep pulling from `work_queue`."""
        if self.abandon_event.is_set():
            return False
        return not upstream_done.is_set() or not work_queue.empty()

    def discard_backlog(self) -> None:
        for work_queue in (self.snapshot_queue, self.result_queue):
            while True:
                try:
                    work_queue.get_nowait()
                except queue.Empty:
                    break
        with self.pending_lock:
            self.pending_bytes = 0

    def finalize(self) -> None:
        with self.final_lock:
            if self.finalized:
                return
            self.finalized = True
            status = self.final_status
            message = self.final_message
        # Close the gauge row out at the real capture count. Captures with no
        # result — dropped for memory, or abandoned by a stop — are the trailing
        # slots nothing will ever fill, and they would otherwise keep pulsing
        # "분석 대기" after the session ended.
        if not self.is_stream:
            slots = self.snapshot_count - self.snapshot_mark_offset
            while len(self.snapshot_marks) < slots:
                self.snapshot_marks.append("4")
        set_video_state(
            status=status,
            message=message,
            completed=True,
            snapshot_marks="".join(self.snapshot_marks),
            analysis_backlog=0,
            analysis_in_flight=False,
        )
        with session_lock:
            global current_session
            if current_session is self:
                current_session = None


def video_stream_worker(session: VideoAnalysisSession) -> None:
    capture = None
    try:
        capture = cv2.VideoCapture(session.video_path)
        session.capture = capture
        if not capture.isOpened():
            raise RuntimeError("video open failed")

        # Live network streams have no fixed length, so frame count / progress
        # are meaningless — leave total_frames at 0 and skip real-time pacing.
        session.total_frames = 0 if session.is_stream else int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        session.source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        session.frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        session.frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        frame_interval = 1.0 / session.source_fps if session.source_fps > 0 else (1.0 / 30.0)

        # Files have a known length, so the gauge can draw every slot up front
        # and fill it in. Streams get 0 and fall back to a scrolling timeline.
        # Samples land at SNAPSHOT_FIRST_DELAY + k*interval while that is still
        # inside the clip, so the count is ceil((duration - first_delay)/interval)
        # — a plain duration/interval leaves a slot that can never be filled.
        if not session.is_stream and session.total_frames > 0 and session.source_fps > 0:
            duration_seconds = session.total_frames / session.source_fps
            usable = max(0.0, duration_seconds - SNAPSHOT_FIRST_DELAY)
            session.expected_snapshot_total = max(
                1, math.ceil(usable / max(1, session.interval_seconds))
            )

        started_at = time.perf_counter()
        fps_window_started_at = time.perf_counter()
        fps_window_frames = 0
        frame_index = 0
        consecutive_failures = 0
        last_live_emit = 0.0
        live_emit_interval = 1.0 / 6.0  # ~6 fps live preview for stream sources

        set_video_state(
            status="processing",
            message="stream running" if session.is_stream else "video stream running",
            video_path=session.video_path,
            is_stream=session.is_stream,
            completed=False,
            started_at=time.time(),
            profile_name=runtime_config.get("profile_name"),
            model_path=pipeline.primary_model_path if pipeline else None,
            secondary_model_path=pipeline.secondary_model_path if pipeline else None,
            analysis_interval_seconds=session.interval_seconds,
            total_frames=session.total_frames,
            source_fps=session.source_fps,
            frame_width=session.frame_width,
            frame_height=session.frame_height,
            expected_snapshot_total=session.expected_snapshot_total,
        )

        while not session.stop_event.is_set():
            # --- Seek (file only): jump the decoder to a requested frame and
            #     re-anchor real-time pacing so playback continues from there. ---
            if not session.is_stream:
                seek_target = session.pop_seek_request()
                if seek_target is not None:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, float(seek_target))
                    frame_index = seek_target
                    started_at = time.perf_counter() - frame_index * frame_interval

            # --- Pause (file only): hold the current position without advancing,
            #     while still honoring seeks so the user can scrub, then resume. ---
            if session.pause_event.is_set() and not session.is_stream:
                set_video_state(paused=True)
                while session.pause_event.is_set() and not session.stop_event.is_set():
                    seek_target = session.pop_seek_request()
                    if seek_target is not None:
                        capture.set(cv2.CAP_PROP_POS_FRAMES, float(seek_target))
                        frame_index = seek_target
                        ok, frame = capture.read()
                        if ok:
                            frame_index += 1
                            frame_time = (frame_index - 1) * frame_interval
                            session.update_latest_frame(frame, frame_index, frame_time)
                            progress = (frame_index / session.total_frames * 100.0) if session.total_frames else 0.0
                            set_video_state(
                                processed_frames=frame_index,
                                progress=min(progress, 100.0),
                                current_video_time=frame_time,
                            )
                    session.stop_event.wait(0.05)
                if session.stop_event.is_set():
                    break
                # Resumed — re-anchor pacing to the (possibly seeked) position
                # and flip back to processing (it may have been "completed").
                started_at = time.perf_counter() - frame_index * frame_interval
                set_video_state(paused=False, status="processing", message="video stream running")

            ok, frame = capture.read()
            if not ok:
                if session.is_stream:
                    # Tolerate transient drops on a live stream; only give up
                    # after a sustained outage (~9s at 0.1s/retry).
                    consecutive_failures += 1
                    if consecutive_failures > 90:
                        session.request_stop("error", "stream disconnected")
                        break
                    if session.stop_event.wait(0.1):
                        break
                    continue
                # End of file. For manual playback, pause at the last frame and
                # keep the session alive so the user can scrub back / resume /
                # capture; the top-of-loop pause block handles the waiting.
                if session.pause_at_end:
                    if not session.pause_event.is_set():
                        session.pause_event.set()
                        set_video_state(
                            status="completed",
                            message="video analysis completed",
                            paused=True,
                            progress=100.0,
                        )
                    if session.stop_event.wait(0.1):
                        break
                    continue
                # Playback is over, but the sampled tail of the clip may still be
                # queued — let the analysis workers finish it before the session
                # reports completion.
                session.request_stop("completed", "video analysis completed", drain=True)
                break
            consecutive_failures = 0

            frame_index += 1
            frame_time = (frame_index - 1) * frame_interval
            session.update_latest_frame(frame, frame_index, frame_time)

            # The HTML <video> element can play local files directly, but it
            # cannot open RTSP/UDP/etc., so for stream sources we push decoded
            # frames to the UI as throttled JPEG previews.
            if session.is_stream:
                now_live = time.perf_counter()
                if now_live - last_live_emit >= live_emit_interval:
                    last_live_emit = now_live
                    set_video_state(live_frame_base64=encode_image_base64(frame))

            fps_window_frames += 1
            now = time.perf_counter()
            if now - fps_window_started_at >= 1.0:
                session.stream_fps = fps_window_frames / max(now - fps_window_started_at, 1e-6)
                fps_window_started_at = now
                fps_window_frames = 0

            progress = (frame_index / session.total_frames * 100.0) if session.total_frames else 0.0
            set_video_state(
                processed_frames=frame_index,
                progress=min(progress, 100.0),
                current_video_time=frame_time,
                stream_fps=session.stream_fps,
                total_frames=session.total_frames,
                source_fps=session.source_fps,
            )

            # File playback is paced to real time; live streams self-pace on read().
            if not session.is_stream:
                target_time = started_at + frame_index * frame_interval
                sleep_seconds = target_time - time.perf_counter()
                if sleep_seconds > 0:
                    session.stop_event.wait(sleep_seconds)
    except Exception:
        session.request_stop("error", traceback.format_exc())
    finally:
        if capture is not None:
            capture.release()


def snapshot_scheduler(session: VideoAnalysisSession) -> None:
    try:
        next_sample_at = time.perf_counter() + SNAPSHOT_FIRST_DELAY
        last_sampled_frame_index = -1
        while not session.stop_event.is_set():
            # While paused, don't sample — resume sampling shortly after unpause
            # so analysis picks up the (possibly seeked) position.
            if session.pause_event.is_set():
                if session.stop_event.wait(0.1):
                    break
                next_sample_at = time.perf_counter() + 0.2
                continue

            wait_seconds = next_sample_at - time.perf_counter()
            if wait_seconds > 0 and session.stop_event.wait(wait_seconds):
                break

            snapshot = session.get_latest_frame_snapshot()
            if snapshot is None:
                next_sample_at += session.interval_seconds
                continue

            # The sampler runs on the wall clock while playback can fall behind
            # real time (4K decode on CPU), so "the current frame" is sometimes
            # the one just analyzed. Wait for a genuinely new frame rather than
            # spend another multi-second inference on the same image.
            if snapshot["frame_index"] == last_sampled_frame_index:
                next_sample_at = time.perf_counter() + 0.1
                continue
            last_sampled_frame_index = snapshot["frame_index"]
            next_sample_at += session.interval_seconds

            snapshot["captured_at"] = datetime.now().isoformat(timespec="seconds")
            snapshot["seq"] = session.snapshot_count  # 0-based capture order
            # Encode before handing the frame off, so the preview can never race
            # the inference worker reading the same array.
            snapshot_base64 = encode_image_base64(snapshot["frame"])
            if session.is_stream:
                # Live: keep only the freshest sample (can't fall behind real time).
                put_latest(session.snapshot_queue, snapshot)
            else:
                # File: hand off and immediately go back to waiting for the next
                # interval. Inference runs behind us in the background and keeps
                # draining after playback ends, so the sampling cadence is the
                # clip's alone — it is never charged for inference latency.
                session.enqueue_snapshot(snapshot)
            session.snapshot_count += 1

            # The up-front slot count divides the clip duration by the interval,
            # but sampling runs on the wall clock while playback can fall behind
            # real time (4K decode on CPU), producing more samples than the
            # clip's length predicts — the gauge would read "17 / 15". Extend
            # the total by exactly the overflow. Rate-based extrapolation was
            # tried and rejected: early on it divides by a tiny coverage figure
            # and guesses wildly (1 sample at 7% -> "18 slots").
            session.expected_snapshot_total = max(
                session.expected_snapshot_total, session.snapshot_count
            )

            # Publish the capture the moment it is taken. This used to wait and
            # be republished by result_handler together with its overlay, so the
            # two panels always showed the same frame — but that made the RAW
            # panel advance once per *analysis*, which is exactly the "capture,
            # wait, capture" the sampler no longer does. The panels now carry
            # their own capture numbers (RAW #30 / ANALYZED #18) instead of being
            # locked to each other.
            set_video_state(
                snapshot_count=session.snapshot_count,
                expected_snapshot_total=session.expected_snapshot_total,
                analysis_interval_seconds=session.interval_seconds,
                analysis_pending=True,
                analysis_backlog=session.snapshot_queue.qsize(),
                dropped_snapshots=session.dropped_snapshots,
                latest_snapshot_base64=snapshot_base64,
                latest_snapshot_timestamp=snapshot["captured_at"],
                latest_snapshot_frame_index=snapshot["frame_index"],
                latest_snapshot_seq=session.snapshot_count,  # 1-based for display
            )
    except Exception:
        session.request_stop("error", traceback.format_exc())
    finally:
        # No more captures will arrive — the inference worker may now finish the
        # backlog and exit (see keep_working).
        session.capture_done_event.set()


def inference_worker(session: VideoAnalysisSession) -> None:
    try:
        assert pipeline is not None
        while session.keep_working(session.snapshot_queue, session.capture_done_event):
            try:
                snapshot = session.snapshot_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            # Live streams: drain to the freshest queued snapshot so we never
            # analyze stale frames (real-time priority). File playback: process
            # every sampled frame FIFO so the analysis count matches the capture
            # count when the clip ends.
            if session.is_stream:
                while True:
                    try:
                        snapshot = session.snapshot_queue.get_nowait()
                    except queue.Empty:
                        break
            else:
                # Off the queue = off the backlog budget, so the sampler can
                # keep capturing while this frame is being analyzed.
                session.release_snapshot(snapshot)

            print(
                f"[Pipeline] -> starting frame={snapshot['frame_index']} "
                f"size={snapshot['frame'].shape[1]}x{snapshot['frame'].shape[0]} "
                f"interval={session.interval_seconds}s",
                flush=True,
            )
            set_video_state(analysis_in_flight=True)
            t_start = time.perf_counter()
            pipeline_result: PipelineResult = pipeline.run(
                snapshot["frame"],
                snapshot["frame_index"],
                session.interval_seconds,
            )
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            set_video_state(analysis_in_flight=False)

            if session.inference_started_at is None:
                session.inference_started_at = time.perf_counter()
            session.analysis_count += 1
            session.active_learning_saved += pipeline_result.active_learning_saved

            timings = pipeline_result.layer_timings_ms
            print(
                f"[Pipeline] <- done snap={session.analysis_count} "
                f"frame={snapshot['frame_index']} "
                f"mode={pipeline_result.analysis_mode} "
                f"raw={len(pipeline_result.raw_detections)} "
                f"confirmed={len(pipeline_result.confirmed_detections)} "
                f"total={elapsed_ms:.0f}ms "
                f"L1L2:{timings.get('l1_l2_ms', 0):.0f} "
                f"VLM:{timings.get('vlm_ms', 0):.0f} "
                f"Track:{timings.get('tracker_ms', 0):.0f}",
                flush=True,
            )

            enriched = dict(snapshot)
            # display detections = raw (post-VLM, with track info) so the UI
            # shows real-time progress; alarm/event detections = confirmed
            # (N-of-M filtered). Until N-of-M is met, we still draw boxes but
            # do not raise an alarm.
            enriched["display_detections"] = pipeline_result.raw_detections
            enriched["confirmed_detections"] = pipeline_result.confirmed_detections
            enriched["analysis_mode"] = pipeline_result.analysis_mode
            enriched["layer_timings_ms"] = pipeline_result.layer_timings_ms
            enriched["is_final_result"] = True
            if session.is_stream:
                # Live: dropping a rendered result is preferable to lagging.
                put_latest(session.result_queue, enriched)
            else:
                # File: never drop, never block. A dropped result means the
                # analysis happened but no event image was written, so
                # analysis_count and the archive would silently disagree with no
                # way to tell it from a clean frame. The result handler (encode +
                # imwrite) is far faster than inference, so this queue stays
                # shallow even though it is unbounded.
                session.result_queue.put_nowait(enriched)
    except Exception:
        session.request_stop("error", traceback.format_exc())
    finally:
        session.analysis_done_event.set()
        set_video_state(analysis_in_flight=False)


def result_handler(session: VideoAnalysisSession) -> None:
    try:
        save_dir = os.path.abspath(EVENT_CONFIG["save_directory"])
        os.makedirs(save_dir, exist_ok=True)

        while session.keep_working(session.result_queue, session.analysis_done_event):
            try:
                result = session.result_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            raw_frame = result["frame"]
            display_detections = result.get("display_detections") or result.get("detections", [])
            confirmed_detections = result.get("confirmed_detections", display_detections)

            # Save / count / risk on DISPLAY detections — the user's rule is
            # "save the event the moment any object is detected." The
            # `track_confirmed` flag is still propagated per-detection so
            # the UI can paint a ✓ on stable tracks, but it does not gate
            # the alarm path.
            annotated = draw_detections(raw_frame, display_detections)
            annotated_base64 = encode_image_base64(annotated)

            has_person, has_vehicle, risk_level, risk_summary = infer_risk(display_detections)
            persons = sum(1 for d in display_detections if d["label"] == "person")
            vehicles = sum(1 for d in display_detections if d["label"] == "vehicle")
            confirmed_persons = sum(1 for d in confirmed_detections if d["label"] == "person")
            confirmed_vehicles = sum(1 for d in confirmed_detections if d["label"] == "vehicle")
            object_labels: List[str] = []
            for det in display_detections:
                display = det.get("display_label", det["label"])
                if display not in object_labels:
                    object_labels.append(display)

            session.max_persons = max(session.max_persons, persons)
            session.max_vehicles = max(session.max_vehicles, vehicles)

            # Gauge cell for this analysis. "clear" is recorded too — that is
            # precisely the case the event archive cannot show, and the reason
            # analysed-count and saved-image-count legitimately differ.
            session.record_snapshot_mark(
                None if session.is_stream else result.get("seq"),
                {"clear": "0", "notice": "1", "warning": "2", "critical": "3"}.get(risk_level, "0"),
            )

            image_path = ""
            if EVENT_CONFIG.get("enabled", True) and (has_person or has_vehicle):
                image_name = f"event_{result['frame_index']:06d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                image_path = os.path.join(save_dir, image_name)
                cv2.imwrite(image_path, annotated)
                session.event_count += 1
                session.latest_event_path = image_path
                event_payload = {
                    "timestamp": result["captured_at"],
                    "frame_index": result["frame_index"],
                    "risk_level": risk_level,
                    "persons": persons,
                    "vehicles": vehicles,
                    "confirmed_persons": confirmed_persons,
                    "confirmed_vehicles": confirmed_vehicles,
                    "labels": object_labels,
                    "path": image_path,
                    "track_ids": sorted({d.get("track_id") for d in display_detections if d.get("track_id") is not None}),
                }
                session.recent_events.insert(0, event_payload)
                session.recent_events = session.recent_events[: int(EVENT_CONFIG.get("max_recent_events", 18))]
                put_latest(session.api_event_queue, event_payload)

            payload = build_result_payload(
                result["captured_at"],
                display_detections,
                image_path,
                risk_level,
                has_person,
                has_vehicle,
            )
            payload["confirmed_count"] = len(confirmed_detections)
            payload["confirmed_persons"] = confirmed_persons
            payload["confirmed_vehicles"] = confirmed_vehicles
            payload["analysis_mode"] = result.get("analysis_mode")
            payload["frame_index"] = result["frame_index"]
            payload["is_final_result"] = bool(result.get("is_final_result", True))
            payload["layer_timings_ms"] = result.get("layer_timings_ms", {})

            elapsed = max(time.perf_counter() - (session.inference_started_at or time.perf_counter()), 1e-6)
            inference_fps = session.analysis_count / elapsed if session.inference_started_at else 0.0

            set_video_state(
                analysis_count=session.analysis_count,
                snapshot_marks="".join(session.snapshot_marks),
                expected_snapshot_total=session.expected_snapshot_total,
                inference_fps=inference_fps,
                analysis_pending=not session.snapshot_queue.empty(),
                analysis_backlog=session.snapshot_queue.qsize(),
                dropped_snapshots=session.dropped_snapshots,
                event_count=session.event_count,
                latest_event_path=session.latest_event_path,
                recent_events=list(session.recent_events),
                pending_api_events=session.api_event_queue.qsize(),
                event_directory=save_dir,
                latest_result_base64=annotated_base64,
                latest_result=payload,
                latest_result_frame_index=result["frame_index"],
                latest_result_timestamp=result["captured_at"],
                # Which capture this overlay belongs to — it trails the RAW
                # panel by the backlog, so the UI labels both.
                latest_result_seq=(result["seq"] + 1) if result.get("seq") is not None else None,
                latest_detections=payload["detections"],
                analysis_mode=result.get("analysis_mode"),
                layer_timings_ms=result.get("layer_timings_ms", {}),
                current_persons=persons,
                current_vehicle_like_objects=vehicles,
                max_persons=session.max_persons,
                max_vehicle_like_objects=session.max_vehicles,
                risk_level=risk_level,
                risk_summary=risk_summary,
                active_learning_saved=session.active_learning_saved,
            )
    except Exception:
        session.request_stop("error", traceback.format_exc())


def session_monitor(session: VideoAnalysisSession) -> None:
    for thread in session.threads:
        thread.join()
    session.finalize()


def start_video_job(video_path: str, interval_seconds: int, pause_at_end: bool = False,
                    analysis_profile: Optional[str] = None) -> Dict:
    global current_session
    if pipeline is None:
        return {"status": "error", "msg": "pipeline not ready"}

    # A previous session may still be winding down (stop only *requests* a stop;
    # the worker threads, especially inference mid-run, take a moment to exit).
    # Stop it and wait for it to finalize so selecting a new video right after
    # stopping doesn't get rejected as "already running".
    with session_lock:
        existing = current_session
    if existing is not None:
        existing.request_stop("stopped", "superseded by new video")
        for thread in existing.threads:
            thread.join(timeout=8.0)
        existing.finalize()  # idempotent; clears current_session if still set

    with session_lock:
        if current_session is not None:
            return {"status": "error", "msg": "another video analysis job is already running"}
        session = VideoAnalysisSession(video_path, interval_seconds)
        session.pause_at_end = bool(pause_at_end) and not session.is_stream
        session.analysis_profile = analysis_profile
        current_session = session

    pipeline.reset_session()  # clear tracker state between videos
    pipeline.set_active_profile(analysis_profile)  # fast / balanced / accurate -> layer plan

    worker_specs = [
        (video_stream_worker, "video_stream_worker"),
        (snapshot_scheduler, "snapshot_scheduler"),
        (inference_worker, "inference_worker"),
        (result_handler, "result_handler"),
    ]
    session.threads = [
        threading.Thread(target=target, args=(session,), name=name, daemon=True)
        for target, name in worker_specs
    ]
    for thread in session.threads:
        thread.start()

    monitor_thread = threading.Thread(
        target=session_monitor, args=(session,), name="session_monitor", daemon=True
    )
    monitor_thread.start()

    set_video_state(
        status="processing",
        message="sampling analysis started",
        video_path=video_path,
        is_stream=session.is_stream,
        live_frame_base64=None,
        completed=False,
        profile_name=runtime_config.get("profile_name"),
        model_path=pipeline.primary_model_path,
        secondary_model_path=pipeline.secondary_model_path,
        analysis_interval_seconds=session.interval_seconds,
        paused=False,
        latest_snapshot_base64=None,
        latest_snapshot_timestamp=None,
        latest_snapshot_frame_index=0,
        latest_snapshot_seq=0,
        latest_result_base64=None,
        latest_result=None,
        latest_result_frame_index=0,
        latest_result_timestamp=None,
        latest_result_seq=0,
        latest_detections=[],
        analysis_pending=False,
        analysis_mode=None,
        layer_timings_ms={},
        current_persons=0,
        current_vehicle_like_objects=0,
        max_persons=0,
        max_vehicle_like_objects=0,
        risk_level="clear",
        risk_summary="현재 감지된 사람이나 차량이 없습니다.",
        event_count=0,
        latest_event_path=None,
        recent_events=[],
        event_directory=os.path.abspath(EVENT_CONFIG["save_directory"]),
        snapshot_count=0,
        analysis_count=0,
        snapshot_marks="",
        expected_snapshot_total=0,
        analysis_backlog=0,
        dropped_snapshots=0,
        analysis_in_flight=False,
        pending_api_events=0,
        progress=0.0,
        processed_frames=0,
        total_frames=0,
        source_fps=0.0,
        stream_fps=0.0,
        inference_fps=0.0,
        current_video_time=0.0,
        frame_width=0,
        frame_height=0,
        started_at=time.time(),
        active_learning_saved=0,
    )

    return {
        "status": "accepted",
        "msg": "video analysis started",
        "analysis_interval_seconds": session.interval_seconds,
    }


def stop_video_job() -> Dict:
    with session_lock:
        session = current_session
    if session is None:
        return {"status": "ok", "msg": "no active session"}
    session.request_stop("stopped", "user stopped video analysis")
    return {"status": "ok", "msg": "stop requested"}


def pause_video_job(paused: bool) -> Dict:
    with session_lock:
        session = current_session
    if session is None:
        return {"status": "error", "msg": "no active session"}
    if session.is_stream:
        return {"status": "error", "msg": "live streams cannot be paused"}
    if paused:
        session.pause_event.set()
    else:
        session.pause_event.clear()
    set_video_state(paused=bool(paused))
    return {"status": "ok", "paused": bool(paused)}


def analyze_image_job(image_path: str, analysis_profile: Optional[str] = None) -> Dict:
    """One-shot analysis of a single still image with the configured pipeline.

    Independent of the video session: reads the image, runs the pipeline with
    the selected profile (fast / balanced / accurate), and returns the annotated
    result so the UI can show raw + analyzed in the snapshot panel.
    """
    if pipeline is None:
        return {"status": "error", "msg": "pipeline not ready"}
    if not image_path or not os.path.exists(image_path):
        return {"status": "error", "msg": "image file not found"}
    frame = cv2.imread(image_path)
    if frame is None:
        return {"status": "error", "msg": "이미지를 읽을 수 없습니다"}

    # Fresh tracker state; profile decides the layer plan (large interval keeps
    # the legacy path on the full stack when no profile is given).
    pipeline.reset_session()
    pipeline.set_active_profile(analysis_profile)
    result = pipeline.run(frame, 0, 999)
    detections = result.raw_detections
    has_person, has_vehicle, risk_level, risk_summary = infer_risk(detections)
    annotated_b64 = encode_image_base64(draw_detections(frame, detections))
    payload = build_result_payload(
        datetime.now().isoformat(timespec="seconds"),
        detections,
        image_path,
        risk_level,
        has_person,
        has_vehicle,
    )
    return {
        "status": "ok",
        "annotated_base64": annotated_b64,
        "detections": payload["detections"],
        "person_count": payload.get("person_count", sum(1 for d in detections if d["label"] == "person")),
        "vehicle_count": payload.get("vehicle_count", sum(1 for d in detections if d["label"] == "vehicle")),
        "risk_level": risk_level,
        "risk_summary": risk_summary,
        "analysis_mode": result.analysis_mode,
    }


def capture_frame_job() -> Dict:
    """Capture the current (paused) frame and push it straight into analysis."""
    with session_lock:
        session = current_session
    if session is None:
        return {"status": "error", "msg": "no active session"}
    snapshot = session.get_latest_frame_snapshot()
    if snapshot is None:
        return {"status": "error", "msg": "no frame available yet"}
    snapshot["captured_at"] = datetime.now().isoformat(timespec="seconds")
    snapshot["seq"] = session.snapshot_count
    preview = encode_image_base64(snapshot["frame"])
    if session.is_stream:
        put_latest(session.snapshot_queue, snapshot)
    else:
        session.enqueue_snapshot(snapshot)
    session.snapshot_count += 1
    set_video_state(
        latest_snapshot_base64=preview,
        latest_snapshot_timestamp=snapshot["captured_at"],
        latest_snapshot_frame_index=snapshot["frame_index"],
        latest_snapshot_seq=session.snapshot_count,
        snapshot_count=session.snapshot_count,
        analysis_pending=True,
    )
    return {"status": "ok", "frame_index": snapshot["frame_index"]}


def seek_video_job(fraction: float) -> Dict:
    with session_lock:
        session = current_session
    if session is None:
        return {"status": "error", "msg": "no active session"}
    if session.is_stream or not session.total_frames:
        return {"status": "error", "msg": "this source is not seekable"}
    frac = max(0.0, min(1.0, float(fraction)))
    target = int(frac * (session.total_frames - 1))
    session.request_seek(target)
    return {"status": "ok", "target_frame": target, "total_frames": session.total_frames}


# ============ Runtime settings (UI-editable external variables) ============
def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def get_current_settings() -> Dict:
    """Return the operator-editable subset of the runtime config."""
    cms = runtime_config.get("class_min_scores", {})
    dev = runtime_config.get("gpu" if device == "cuda" else "cpu", {})
    sampling = runtime_config.get("sampling", {})
    return {
        "device": device,
        "person_min_score": float(cms.get("person", 0.12)),
        "vehicle_min_score": float(cms.get("vehicle", 0.10)),
        "model_confidence": float(dev.get("confidence_threshold", 0.08)),
        "default_interval_seconds": int(sampling.get("default_interval_seconds", 3)),
        "min_interval_seconds": int(sampling.get("min_interval_seconds", 1)),
        "max_interval_seconds": int(sampling.get("max_interval_seconds", 10)),
        "require_confirmation_for_alarm": bool(runtime_config.get("require_confirmation_for_alarm", False)),
        "event_enabled": bool(runtime_config.get("event", {}).get("enabled", True)),
        "vlm_enabled": bool(runtime_config.get("vlm", {}).get("enabled", True)),
        "vlm_strict": bool(runtime_config.get("vlm", {}).get("reject_if_top_is_negative", False)),
        "preview_jpeg_quality": int(runtime_config.get("preview_jpeg_quality", 88)),
        "cpu_allow_sahi": bool(runtime_config.get("cpu_runtime", {}).get("allow_sahi", False)),
        "cpu_allow_vlm": bool(runtime_config.get("cpu_runtime", {}).get("allow_vlm", False)),
    }


def _persist_runtime_config() -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as handle:
        json.dump(runtime_config, handle, ensure_ascii=False, indent=2)


def _effective_layer_enabled(layer: str) -> bool:
    """Compute whether a heavy layer should actually run on the live pipeline.

    On GPU the layer follows its own config flag. On CPU it additionally
    requires the operator to opt in via the cpu_runtime override block,
    mirroring the CPU safety net in detection/pipeline.build_pipeline().
    """
    if layer == "sahi":
        base = bool(runtime_config.get("sahi", {}).get("enabled", True))
        allow_key = "allow_sahi"
    else:  # "vlm"
        base = bool(runtime_config.get("vlm", {}).get("enabled", True))
        allow_key = "allow_vlm"
    if device == "cuda":
        return base
    return base and bool(runtime_config.get("cpu_runtime", {}).get(allow_key, False))


def apply_settings(updates: Dict) -> Dict:
    """Validate, hot-apply to the live pipeline, and persist editable settings.

    Every exposed field is read from its config object at inference time, so
    mutating the running pipeline's objects takes effect on the next analyzed
    frame — no model reload required.
    """
    global PREVIEW_JPEG_QUALITY

    if "person_min_score" in updates:
        v = _clamp(float(updates["person_min_score"]), 0.0, 1.0)
        runtime_config.setdefault("class_min_scores", {})["person"] = v
        if pipeline:
            pipeline.config.filter.class_min_scores["person"] = v

    if "vehicle_min_score" in updates:
        v = _clamp(float(updates["vehicle_min_score"]), 0.0, 1.0)
        runtime_config.setdefault("class_min_scores", {})["vehicle"] = v
        if pipeline:
            pipeline.config.filter.class_min_scores["vehicle"] = v

    if "model_confidence" in updates:
        v = _clamp(float(updates["model_confidence"]), 0.01, 0.9)
        for sec in ("cpu", "gpu"):
            if sec in runtime_config:
                runtime_config[sec]["confidence_threshold"] = v
                if "secondary_confidence_threshold" in runtime_config[sec]:
                    runtime_config[sec]["secondary_confidence_threshold"] = v
        if "fast_sampling" in runtime_config:
            runtime_config["fast_sampling"]["confidence_threshold"] = v
        if pipeline:
            for det in (pipeline._primary, pipeline._secondary, pipeline._fast):
                if det is not None:
                    det.config.confidence_threshold = v

    if "default_interval_seconds" in updates:
        lo = int(runtime_config["sampling"].get("min_interval_seconds", 1))
        hi = int(runtime_config["sampling"].get("max_interval_seconds", 10))
        v = int(_clamp(int(updates["default_interval_seconds"]), lo, hi))
        runtime_config["sampling"]["default_interval_seconds"] = v

    if "require_confirmation_for_alarm" in updates:
        v = bool(updates["require_confirmation_for_alarm"])
        runtime_config["require_confirmation_for_alarm"] = v
        if pipeline:
            pipeline.config.require_confirmation_for_alarm = v

    if "event_enabled" in updates:
        # EVENT_CONFIG is the same dict object as runtime_config["event"].
        runtime_config.setdefault("event", {})["enabled"] = bool(updates["event_enabled"])

    if "vlm_enabled" in updates:
        v = bool(updates["vlm_enabled"])
        runtime_config.setdefault("vlm", {})["enabled"] = v
        if pipeline:
            pipeline.config.vlm.enabled = _effective_layer_enabled("vlm")

    if "vlm_strict" in updates:
        v = bool(updates["vlm_strict"])
        runtime_config.setdefault("vlm", {})["reject_if_top_is_negative"] = v
        if pipeline:
            pipeline.config.vlm.reject_if_top_is_negative = v

    if "preview_jpeg_quality" in updates:
        v = int(_clamp(int(updates["preview_jpeg_quality"]), 40, 100))
        runtime_config["preview_jpeg_quality"] = v
        PREVIEW_JPEG_QUALITY = v

    # CPU force-enable overrides. On CPU the heavy layers are auto-disabled at
    # build time; toggling these lets the operator turn them back on (slow but
    # supported). On GPU they are no-ops — the layers already follow their own
    # config flags. Both are hot-applied so the next snapshot reflects them.
    if "cpu_allow_sahi" in updates:
        runtime_config.setdefault("cpu_runtime", {})["allow_sahi"] = bool(updates["cpu_allow_sahi"])
        if pipeline:
            pipeline.config.sahi.enabled = _effective_layer_enabled("sahi")

    if "cpu_allow_vlm" in updates:
        runtime_config.setdefault("cpu_runtime", {})["allow_vlm"] = bool(updates["cpu_allow_vlm"])
        if pipeline:
            pipeline.config.vlm.enabled = _effective_layer_enabled("vlm")

    _persist_runtime_config()
    return get_current_settings()


def main() -> None:
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    try:
        socket.bind("tcp://127.0.0.1:5555")
    except zmq.ZMQError as exc:
        # Almost always EADDRINUSE: a previous engine process didn't exit
        # cleanly (app force-closed / crashed) and still holds the port, or a
        # second app instance is starting. Emit a clear, parseable line the
        # Electron side can surface instead of a cryptic traceback, then exit.
        print(
            f"[AI Engine ERR] port 5555 bind failed ({exc}). "
            "An old engine process is likely still running and holding the port. "
            "Close it (Task Manager -> python.exe) or restart the PC, then relaunch.",
            flush=True,
        )
        socket.close(0)
        context.term()
        raise SystemExit(1)

    print("[AI Engine] READY", flush=True)
    threading.Thread(target=load_pipeline, daemon=True).start()

    while True:
        try:
            message = socket.recv_string()
            data = json.loads(message)
            req_type = data.get("type")

            if req_type == "status":
                layers = pipeline.describe_active_layers() if pipeline else {}
                socket.send_string(json.dumps({
                    "status": "ok",
                    "model_loaded": model_loaded,
                    "model_loading": model_loading,
                    "loading": loading_progress.snapshot(),
                    "device": device,
                    "model_path": pipeline.primary_model_path if pipeline else None,
                    "secondary_model_path": pipeline.secondary_model_path if pipeline else None,
                    "profile_name": runtime_config.get("profile_name"),
                    "allowed_classes": sorted(runtime_config.get("allowed_classes", [])),
                    "default_interval_seconds": SAMPLING_CONFIG["default_interval_seconds"],
                    "layers": layers,
                }))
                continue

            if req_type == "ping":
                socket.send_string(json.dumps({"status": "pong"}))
                continue

            if req_type == "video_status":
                socket.send_string(json.dumps(snapshot_video_state()))
                continue

            if req_type == "stop_video":
                socket.send_string(json.dumps(stop_video_job()))
                continue

            if req_type == "pause_video":
                socket.send_string(json.dumps(pause_video_job(bool(data.get("paused", True)))))
                continue

            if req_type == "seek_video":
                socket.send_string(json.dumps(seek_video_job(data.get("fraction", 0.0))))
                continue

            if req_type == "capture_frame":
                socket.send_string(json.dumps(capture_frame_job()))
                continue

            if req_type == "get_settings":
                socket.send_string(json.dumps({"status": "ok", "settings": get_current_settings()}))
                continue

            if req_type == "update_settings":
                try:
                    applied = apply_settings(data.get("settings", {}) or {})
                    socket.send_string(json.dumps({"status": "ok", "settings": applied}))
                except Exception as exc:
                    print(f"[AI Engine ERR] update_settings\n{traceback.format_exc()}", flush=True)
                    socket.send_string(json.dumps({"status": "error", "msg": str(exc)}))
                continue

            if not model_loaded:
                # Distinguish "not yet" from "never" — a caller that retries
                # (the UI, or an unattended trigger) must not spin forever on a
                # load that already failed.
                progress = loading_progress.snapshot()
                if progress["state"] == "error":
                    msg = f"model load failed: {progress['error']}"
                else:
                    msg = f"model is still loading ({progress['percent']:.0f}%)"
                socket.send_string(json.dumps({
                    "status": "error",
                    "msg": msg,
                    "retryable": progress["state"] != "error",
                    "loading": progress,
                }))
                continue

            if req_type == "start_video":
                video_path = data.get("video_path")
                interval_seconds = data.get(
                    "analysis_interval_seconds",
                    SAMPLING_CONFIG["default_interval_seconds"],
                )
                if not video_path:
                    socket.send_string(json.dumps({"status": "error", "msg": "video source is required"}))
                    continue
                if not is_stream_source(video_path) and not os.path.exists(video_path):
                    socket.send_string(json.dumps({"status": "error", "msg": "video file not found"}))
                    continue
                source_kind = "stream" if is_stream_source(video_path) else "file"
                pause_at_end = bool(data.get("pause_at_end", False))
                analysis_profile = data.get("analysis_profile")
                print(f"[AI Engine] start {source_kind}: {video_path} / interval={interval_seconds}s / profile={analysis_profile}", flush=True)
                socket.send_string(json.dumps(start_video_job(video_path, int(interval_seconds), pause_at_end, analysis_profile)))
                continue

            if req_type == "analyze_image":
                socket.send_string(json.dumps(analyze_image_job(data.get("image_path"), data.get("analysis_profile"))))
                continue

            socket.send_string(json.dumps({"status": "error", "msg": "unknown request"}))
        except Exception as exc:
            err_msg = traceback.format_exc()
            print(f"[AI Engine ERR]\n{err_msg}", flush=True)
            socket.send_string(json.dumps({"status": "error", "msg": str(exc)}))


if __name__ == "__main__":
    main()
