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
import os
import queue
import threading
import time
import traceback
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Optional

import cv2
import torch
import zmq

from detection import DetectionPipeline, PipelineResult, build_pipeline


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
    },
    "event": {"enabled": True, "save_directory": "results/events", "max_recent_events": 18},
    "fp16_on_cuda": True,
    "cudnn_benchmark": True,
    # If True, only N-of-M-confirmed tracks raise alarms / events.
    # Default False — the operator wants every detection saved immediately;
    # confirmation is exposed as a UI badge ("✓") instead of a gating filter.
    "require_confirmation_for_alarm": False,
}


def _deep_merge(base: Dict, override: Dict) -> Dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_runtime_config() -> Dict:
    config_path = os.path.join(os.path.dirname(__file__), "configs", "runtime_detector.json")
    if not os.path.exists(config_path):
        return DEFAULT_RUNTIME_CONFIG
    with open(config_path, "r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    return _deep_merge(DEFAULT_RUNTIME_CONFIG, loaded)


device = "cuda" if torch.cuda.is_available() else "cpu"
runtime_config = load_runtime_config()

if device == "cuda" and runtime_config.get("cudnn_benchmark", True):
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass

SAMPLING_CONFIG = runtime_config["sampling"]
EVENT_CONFIG = runtime_config["event"]
PREVIEW_JPEG_QUALITY = int(runtime_config["preview_jpeg_quality"])
PREVIEW_MAX_WIDTH = int(runtime_config["preview_max_width"])

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
    "source_fps": 0.0,
    "stream_fps": 0.0,
    "inference_fps": 0.0,
    "current_video_time": 0.0,
    "frame_width": 0,
    "frame_height": 0,
    "snapshot_count": 0,
    "analysis_count": 0,
    "latest_snapshot_base64": None,
    "latest_snapshot_timestamp": None,
    "latest_snapshot_frame_index": 0,
    "latest_result_base64": None,
    "latest_result": None,
    "latest_result_frame_index": 0,
    "latest_detections": [],
    "analysis_pending": False,
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


def load_pipeline() -> None:
    global pipeline, model_loaded, model_loading
    try:
        model_loading = True
        print(f"[AI Engine] device: {device}", flush=True)
        print(f"[AI Engine] profile: {runtime_config.get('profile_name')}", flush=True)
        pipeline = build_pipeline(runtime_config, device)
        print(f"[AI Engine] primary: {pipeline.primary_model_path}", flush=True)
        print(f"[AI Engine] secondary: {pipeline.secondary_model_path}", flush=True)
        print("[AI Engine] warming up heavy layers...", flush=True)
        pipeline.warmup()
        model_loaded = True
        print("[AI Engine] pipeline ready", flush=True)
    except Exception as exc:
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
        cv2.putText(
            annotated,
            f"{label} {score:.2f}{suffix}",
            (x1, max(y1 - 8, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
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


class VideoAnalysisSession:
    def __init__(self, video_path: str, interval_seconds: int):
        min_interval = int(SAMPLING_CONFIG["min_interval_seconds"])
        max_interval = int(SAMPLING_CONFIG["max_interval_seconds"])
        self.video_path = video_path
        self.interval_seconds = max(min_interval, min(max_interval, int(interval_seconds)))
        self.stop_event = threading.Event()
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

        self.snapshot_queue: "queue.Queue" = queue.Queue(maxsize=int(SAMPLING_CONFIG["snapshot_queue_size"]))
        self.result_queue: "queue.Queue" = queue.Queue(maxsize=int(SAMPLING_CONFIG["result_queue_size"]))
        self.api_event_queue: "queue.Queue" = queue.Queue(maxsize=64)

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

    def request_stop(self, status: str, message: str) -> None:
        with self.final_lock:
            if self.finalized:
                return
            self.final_status = status
            self.final_message = message
        self.stop_event.set()

    def finalize(self) -> None:
        with self.final_lock:
            if self.finalized:
                return
            self.finalized = True
            status = self.final_status
            message = self.final_message
        set_video_state(status=status, message=message, completed=True)
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

        session.total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        session.source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        session.frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        session.frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        frame_interval = 1.0 / session.source_fps if session.source_fps > 0 else (1.0 / 30.0)
        started_at = time.perf_counter()
        fps_window_started_at = time.perf_counter()
        fps_window_frames = 0
        frame_index = 0

        set_video_state(
            status="processing",
            message="video stream running",
            video_path=session.video_path,
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
        )

        while not session.stop_event.is_set():
            ok, frame = capture.read()
            if not ok:
                session.request_stop("completed", "video analysis completed")
                break

            frame_index += 1
            frame_time = (frame_index - 1) * frame_interval
            session.update_latest_frame(frame, frame_index, frame_time)

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
        next_sample_at = time.perf_counter() + 0.35
        while not session.stop_event.is_set():
            wait_seconds = next_sample_at - time.perf_counter()
            if wait_seconds > 0 and session.stop_event.wait(wait_seconds):
                break

            snapshot = session.get_latest_frame_snapshot()
            next_sample_at += session.interval_seconds
            if snapshot is None:
                continue

            snapshot["captured_at"] = datetime.now().isoformat(timespec="seconds")
            snapshot_preview = encode_image_base64(snapshot["frame"])
            put_latest(session.snapshot_queue, snapshot)
            session.snapshot_count += 1

            set_video_state(
                latest_snapshot_base64=snapshot_preview,
                latest_snapshot_timestamp=snapshot["captured_at"],
                latest_snapshot_frame_index=snapshot["frame_index"],
                snapshot_count=session.snapshot_count,
                analysis_interval_seconds=session.interval_seconds,
                analysis_pending=True,
            )
    except Exception:
        session.request_stop("error", traceback.format_exc())


def inference_worker(session: VideoAnalysisSession) -> None:
    try:
        assert pipeline is not None
        while not session.stop_event.is_set() or not session.snapshot_queue.empty():
            try:
                snapshot = session.snapshot_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            # Drain any backed-up snapshots — only the freshest one matters.
            while True:
                try:
                    snapshot = session.snapshot_queue.get_nowait()
                except queue.Empty:
                    break

            print(
                f"[Pipeline] -> starting frame={snapshot['frame_index']} "
                f"size={snapshot['frame'].shape[1]}x{snapshot['frame'].shape[0]} "
                f"interval={session.interval_seconds}s",
                flush=True,
            )
            t_start = time.perf_counter()
            pipeline_result: PipelineResult = pipeline.run(
                snapshot["frame"],
                snapshot["frame_index"],
                session.interval_seconds,
            )
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0

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
            put_latest(session.result_queue, enriched)
    except Exception:
        session.request_stop("error", traceback.format_exc())


def result_handler(session: VideoAnalysisSession) -> None:
    try:
        save_dir = os.path.abspath(EVENT_CONFIG["save_directory"])
        os.makedirs(save_dir, exist_ok=True)

        while not session.stop_event.is_set() or not session.result_queue.empty():
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
            raw_base64 = encode_image_base64(raw_frame)
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
                inference_fps=inference_fps,
                analysis_pending=not session.snapshot_queue.empty(),
                event_count=session.event_count,
                latest_event_path=session.latest_event_path,
                recent_events=list(session.recent_events),
                pending_api_events=session.api_event_queue.qsize(),
                event_directory=save_dir,
                latest_snapshot_base64=raw_base64,
                latest_snapshot_timestamp=result["captured_at"],
                latest_snapshot_frame_index=result["frame_index"],
                latest_result_base64=annotated_base64,
                latest_result=payload,
                latest_result_frame_index=result["frame_index"],
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


def start_video_job(video_path: str, interval_seconds: int) -> Dict:
    global current_session
    if pipeline is None:
        return {"status": "error", "msg": "pipeline not ready"}

    with session_lock:
        if current_session is not None:
            return {"status": "error", "msg": "another video analysis job is already running"}
        session = VideoAnalysisSession(video_path, interval_seconds)
        current_session = session

    pipeline.reset_session()  # clear tracker state between videos

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
        completed=False,
        profile_name=runtime_config.get("profile_name"),
        model_path=pipeline.primary_model_path,
        secondary_model_path=pipeline.secondary_model_path,
        analysis_interval_seconds=session.interval_seconds,
        latest_snapshot_base64=None,
        latest_snapshot_timestamp=None,
        latest_snapshot_frame_index=0,
        latest_result_base64=None,
        latest_result=None,
        latest_result_frame_index=0,
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


def main() -> None:
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://127.0.0.1:5555")

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

            if not model_loaded:
                socket.send_string(json.dumps({"status": "error", "msg": "model is still loading"}))
                continue

            if req_type == "start_video":
                video_path = data.get("video_path")
                interval_seconds = data.get(
                    "analysis_interval_seconds",
                    SAMPLING_CONFIG["default_interval_seconds"],
                )
                if not video_path or not os.path.exists(video_path):
                    socket.send_string(json.dumps({"status": "error", "msg": "video file not found"}))
                    continue
                print(f"[AI Engine] start video: {video_path} / interval={interval_seconds}s", flush=True)
                socket.send_string(json.dumps(start_video_job(video_path, int(interval_seconds))))
                continue

            socket.send_string(json.dumps({"status": "error", "msg": "unknown request"}))
        except Exception as exc:
            err_msg = traceback.format_exc()
            print(f"[AI Engine ERR]\n{err_msg}", flush=True)
            socket.send_string(json.dumps({"status": "error", "msg": str(exc)}))


if __name__ == "__main__":
    main()
