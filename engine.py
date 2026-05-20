import base64
import json
import os
import queue
import threading
import time
import traceback
from datetime import datetime

import cv2
import torch
import zmq
from ultralytics import YOLO


DEFAULT_RUNTIME_CONFIG = {
    "profile_name": "snapshot-detection-v2",
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
    },
    "class_min_scores": {
        "person": 0.12,
        "vehicle": 0.10,
    },
    "filters": {
        "person": {
            "min_area_ratio": 0.000003,
            "max_area_ratio": 0.05,
            "min_aspect_ratio": 0.08,
            "max_aspect_ratio": 1.6,
        },
        "vehicle": {
            "min_area_ratio": 0.00002,
            "max_area_ratio": 0.30,
            "min_aspect_ratio": 0.18,
            "max_aspect_ratio": 7.2,
        },
    },
    "ensemble": {
        "enabled": True,
        "nms_iou_threshold": 0.42,
        "prefer_secondary_for_vehicle": True,
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
    "event": {
        "enabled": True,
        "save_directory": "results/events",
        "max_recent_events": 18,
    },
    "fp16_on_cuda": True,
    "cudnn_benchmark": True,    
}


device = "cuda" if torch.cuda.is_available() else "cpu"
runtime_config = None
primary_detection_model = None
secondary_detection_model = None
fast_detection_model = None
model_loaded = False
model_loading = False

video_lock = threading.Lock()
session_lock = threading.Lock()
current_session = None

video_state = {
    "status": "idle",
    "message": "video idle",
    "video_path": None,
    "completed": False,
    "started_at": None,
    "device": device,
    "profile_name": None,
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
}


if device == "cuda" and DEFAULT_RUNTIME_CONFIG["cudnn_benchmark"]:
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass


def load_runtime_config():
    config_path = os.path.join(os.path.dirname(__file__), "configs", "runtime_detector.json")
    if not os.path.exists(config_path):
        return DEFAULT_RUNTIME_CONFIG

    with open(config_path, "r", encoding="utf-8") as handle:
        loaded = json.load(handle)

    merged = dict(DEFAULT_RUNTIME_CONFIG)
    merged.update(
        {
            key: value
            for key, value in loaded.items()
            if key
            not in {
                "cpu",
                "gpu",
                "sampling",
                "event",
                "filters",
                "class_min_scores",
                "ensemble",
                "fast_sampling",
            }
        }
    )
    merged["cpu"] = dict(DEFAULT_RUNTIME_CONFIG["cpu"], **loaded.get("cpu", {}))
    merged["gpu"] = dict(DEFAULT_RUNTIME_CONFIG["gpu"], **loaded.get("gpu", {}))
    merged["sampling"] = dict(DEFAULT_RUNTIME_CONFIG["sampling"], **loaded.get("sampling", {}))
    merged["event"] = dict(DEFAULT_RUNTIME_CONFIG["event"], **loaded.get("event", {}))
    merged["class_min_scores"] = dict(
        DEFAULT_RUNTIME_CONFIG["class_min_scores"],
        **loaded.get("class_min_scores", {}),
    )
    merged["ensemble"] = dict(DEFAULT_RUNTIME_CONFIG["ensemble"], **loaded.get("ensemble", {}))
    merged["fast_sampling"] = dict(
        DEFAULT_RUNTIME_CONFIG["fast_sampling"],
        **loaded.get("fast_sampling", {}),
    )

    filters = {}
    default_filters = DEFAULT_RUNTIME_CONFIG["filters"]
    loaded_filters = loaded.get("filters", {})
    for label, rules in default_filters.items():
        filters[label] = dict(rules, **loaded_filters.get(label, {}))
    for label, rules in loaded_filters.items():
        if label not in filters:
            filters[label] = dict(rules)
    merged["filters"] = filters
    return merged


runtime_config = load_runtime_config()
active_device_config = runtime_config[device]
MODEL_PATH = active_device_config["model_path"]
TARGET_WIDTH = active_device_config["target_width"]
CONFIDENCE_THRESHOLD = active_device_config["confidence_threshold"]
INFERENCE_IMGSZ = active_device_config["inference_imgsz"]
SECONDARY_MODEL_PATH = active_device_config.get("secondary_model_path")
SECONDARY_TARGET_WIDTH = active_device_config.get("secondary_target_width", TARGET_WIDTH)
SECONDARY_CONFIDENCE_THRESHOLD = active_device_config.get(
    "secondary_confidence_threshold",
    CONFIDENCE_THRESHOLD,
)
SECONDARY_INFERENCE_IMGSZ = active_device_config.get("secondary_inference_imgsz", INFERENCE_IMGSZ)
PREVIEW_JPEG_QUALITY = runtime_config["preview_jpeg_quality"]
PREVIEW_MAX_WIDTH = runtime_config["preview_max_width"]
ALLOWED_CLASSES = set(runtime_config["allowed_classes"])
CLASS_ALIASES = dict(runtime_config["class_aliases"])
CLASS_MIN_SCORES = dict(runtime_config["class_min_scores"])
FILTER_CONFIG = dict(runtime_config["filters"])
SAMPLING_CONFIG = dict(runtime_config["sampling"])
EVENT_CONFIG = dict(runtime_config["event"])
ENSEMBLE_CONFIG = dict(runtime_config["ensemble"])
FAST_SAMPLING_CONFIG = dict(runtime_config["fast_sampling"])
USE_FP16 = bool(device == "cuda" and runtime_config["fp16_on_cuda"])
FAST_MODEL_PATH = FAST_SAMPLING_CONFIG["model_path"]
FAST_TARGET_WIDTH = FAST_SAMPLING_CONFIG["target_width"]
FAST_CONFIDENCE_THRESHOLD = FAST_SAMPLING_CONFIG["confidence_threshold"]
FAST_INFERENCE_IMGSZ = FAST_SAMPLING_CONFIG["inference_imgsz"]
FAST_INTERVAL_THRESHOLD = int(FAST_SAMPLING_CONFIG["interval_threshold_seconds"])
FAST_DISABLE_SECONDARY = bool(FAST_SAMPLING_CONFIG["disable_secondary"])

print(f"[AI Engine] device: {device}", flush=True)
print(f"[AI Engine] profile: {runtime_config['profile_name']}", flush=True)
print(f"[AI Engine] primary model: {MODEL_PATH}", flush=True)
print(f"[AI Engine] secondary model: {SECONDARY_MODEL_PATH}", flush=True)
print(f"[AI Engine] fast model: {FAST_MODEL_PATH}", flush=True)
print(f"[AI Engine] fp16: {USE_FP16}", flush=True)



def set_video_state(**updates):
    with video_lock:
        video_state.update(updates)


def snapshot_video_state():
    with video_lock:
        state = dict(video_state)
        if isinstance(state.get("latest_result"), dict):
            state["latest_result"] = dict(state["latest_result"])
        state["recent_events"] = [dict(item) for item in state.get("recent_events", [])]
        return state


def align_to_stride(value, stride=32):
    value = max(stride, int(value))
    return ((value + stride - 1) // stride) * stride


def clip_box(x1, y1, x2, y2, width, height):
    return {
        "x1": int(max(0, min(width - 1, round(x1)))),
        "y1": int(max(0, min(height - 1, round(y1)))),
        "x2": int(max(0, min(width - 1, round(x2)))),
        "y2": int(max(0, min(height - 1, round(y2)))),
    }


def load_model():
    global primary_detection_model, secondary_detection_model, fast_detection_model, model_loaded, model_loading

    try:
        model_loading = True
        print(f"[AI Engine] loading primary model: {MODEL_PATH}", flush=True)
        primary_detection_model = YOLO(MODEL_PATH)

        if FAST_MODEL_PATH and FAST_MODEL_PATH != MODEL_PATH and os.path.exists(FAST_MODEL_PATH):
            print(f"[AI Engine] loading fast model: {FAST_MODEL_PATH}", flush=True)
            fast_detection_model = YOLO(FAST_MODEL_PATH)
        else:
            fast_detection_model = primary_detection_model

        if ENSEMBLE_CONFIG.get("enabled") and SECONDARY_MODEL_PATH and os.path.exists(SECONDARY_MODEL_PATH):
            print(f"[AI Engine] loading secondary model: {SECONDARY_MODEL_PATH}", flush=True)
            secondary_detection_model = YOLO(SECONDARY_MODEL_PATH)
        else:
            secondary_detection_model = None

        model_loaded = True
        print("[AI Engine] model stack ready", flush=True)
    except Exception as exc:
        print(f"[AI Engine ERR] model load failed: {exc}", flush=True)
        traceback.print_exc()
    finally:
        model_loading = False


def resize_for_inference(frame, target_width):
    height, width = frame.shape[:2]
    if width <= target_width:
        return frame, 1.0, 1.0

    scale = target_width / float(width)
    resized = cv2.resize(
        frame,
        (target_width, max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, width / float(resized.shape[1]), height / float(resized.shape[0])


def normalize_label(raw_label):
    return CLASS_ALIASES.get(raw_label, raw_label)


def is_allowed_detection(raw_label, normalized_label):
    return raw_label in ALLOWED_CLASSES or normalized_label in ALLOWED_CLASSES


def select_min_score(label):
    return CLASS_MIN_SCORES.get(label, 0.0)


def select_filter_rules(label):
    return FILTER_CONFIG.get(label, FILTER_CONFIG.get("vehicle", {}))


def passes_geometry_filter(box, label, frame_shape):
    height, width = frame_shape[:2]
    area = max(1, (box["x2"] - box["x1"]) * (box["y2"] - box["y1"]))
    frame_area = max(1, width * height)
    area_ratio = area / frame_area
    box_width = max(1, box["x2"] - box["x1"])
    box_height = max(1, box["y2"] - box["y1"])
    aspect_ratio = box_width / box_height

    rules = select_filter_rules(label)
    if area_ratio < rules.get("min_area_ratio", 0.0):
        return False
    if area_ratio > rules.get("max_area_ratio", 1.0):
        return False
    if aspect_ratio < rules.get("min_aspect_ratio", 0.0):
        return False
    if aspect_ratio > rules.get("max_aspect_ratio", 999.0):
        return False
    return True


def predict_boxes_from_model(frame, model, conf_threshold, target_width, inference_imgsz, model_tag):
    if model is None:
        return []

    resized_frame, scale_x, scale_y = resize_for_inference(frame, target_width)
    imgsz = align_to_stride(min(max(resized_frame.shape[:2]), inference_imgsz))
    results = model.predict(
        source=resized_frame,
        conf=conf_threshold,
        imgsz=imgsz,
        verbose=False,
        device=device,
        half=USE_FP16,
    )

    if not results:
        return []

    result = results[0]
    names = result.names or {}
    if result.boxes is None:
        return []

    detections = []
    for xyxy, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
        raw_label = str(names.get(int(cls.item()), int(cls.item())))
        normalized_label = normalize_label(raw_label)
        if not is_allowed_detection(raw_label, normalized_label):
            continue

        x1, y1, x2, y2 = xyxy.tolist()
        box = clip_box(
            x1 * scale_x,
            y1 * scale_y,
            x2 * scale_x,
            y2 * scale_y,
            frame.shape[1],
            frame.shape[0],
        )
        box["score"] = float(conf.item())
        box["label"] = normalized_label
        box["display_label"] = raw_label
        box["model_tag"] = model_tag

        if box["score"] < select_min_score(normalized_label):
            continue
        if not passes_geometry_filter(box, normalized_label, frame.shape):
            continue

        detections.append(box)

    return detections


def compute_iou(box_a, box_b):
    inter_x1 = max(box_a["x1"], box_b["x1"])
    inter_y1 = max(box_a["y1"], box_b["y1"])
    inter_x2 = min(box_a["x2"], box_b["x2"])
    inter_y2 = min(box_a["y2"], box_b["y2"])
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = max(1, (box_a["x2"] - box_a["x1"]) * (box_a["y2"] - box_a["y1"]))
    area_b = max(1, (box_b["x2"] - box_b["x1"]) * (box_b["y2"] - box_b["y1"]))
    union_area = area_a + area_b - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def merge_detections(detections):
    merged = []
    iou_threshold = float(ENSEMBLE_CONFIG.get("nms_iou_threshold", 0.42))
    prefer_secondary_vehicle = bool(ENSEMBLE_CONFIG.get("prefer_secondary_for_vehicle", True))

    for candidate in sorted(detections, key=lambda item: item["score"], reverse=True):
        keep = True
        for kept in merged:
            if kept["label"] != candidate["label"]:
                continue
            if compute_iou(kept, candidate) < iou_threshold:
                continue

            if (
                prefer_secondary_vehicle
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


def predict_boxes(
    frame,
    use_secondary=True,
    primary_model_override=None,
    primary_confidence_override=None,
    primary_target_width_override=None,
    primary_inference_imgsz_override=None,
):
    primary_model = primary_model_override or primary_detection_model
    primary_confidence = (
        CONFIDENCE_THRESHOLD if primary_confidence_override is None else primary_confidence_override
    )
    primary_target_width = TARGET_WIDTH if primary_target_width_override is None else primary_target_width_override
    primary_inference_imgsz = (
        INFERENCE_IMGSZ if primary_inference_imgsz_override is None else primary_inference_imgsz_override
    )

    detections = predict_boxes_from_model(
        frame,
        primary_model,
        primary_confidence,
        primary_target_width,
        primary_inference_imgsz,
        "primary",
    )

    if use_secondary and secondary_detection_model is not None:
        detections.extend(
            predict_boxes_from_model(
                frame,
                secondary_detection_model,
                SECONDARY_CONFIDENCE_THRESHOLD,
                SECONDARY_TARGET_WIDTH,
                SECONDARY_INFERENCE_IMGSZ,
                "secondary",
            )
        )

    return merge_detections(detections)


def get_box_color(label):
    if label == "person":
        return (0, 255, 0)
    return (0, 180, 255)


def draw_detections(frame, detections):
    annotated = frame.copy()
    for detection in detections:
        x1 = detection["x1"]
        y1 = detection["y1"]
        x2 = detection["x2"]
        y2 = detection["y2"]
        label = detection.get("display_label", detection["label"])
        score = detection["score"]
        color = get_box_color(detection["label"])

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            f"{label} {score:.2f}",
            (x1, max(y1 - 8, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def encode_image_base64(frame):
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


def infer_risk(detections):
    has_person = any(item["label"] == "person" for item in detections)
    has_vehicle = any(item["label"] == "vehicle" for item in detections)

    if has_person and has_vehicle:
        return has_person, has_vehicle, "critical", "사람과 차량이 동시에 감지되었습니다."
    if has_person:
        return has_person, has_vehicle, "warning", "사람이 감지되었습니다."
    if has_vehicle:
        return has_person, has_vehicle, "notice", "차량이 감지되었습니다."
    return has_person, has_vehicle, "clear", "현재 감지된 사람이나 차량이 없습니다."


def serialize_detections(detections):
    payload = []
    for item in detections:
        payload.append(
            {
                "label": item["label"],
                "display_label": item.get("display_label", item["label"]),
                "score": round(float(item["score"]), 4),
                "x1": item["x1"],
                "y1": item["y1"],
                "x2": item["x2"],
                "y2": item["y2"],
            }
        )
    return payload


def build_result_payload(timestamp_text, detections, image_path, risk_level, has_person, has_vehicle):
    person_count = sum(1 for item in detections if item["label"] == "person")
    vehicle_count = sum(1 for item in detections if item["label"] == "vehicle")
    return {
        "timestamp": timestamp_text,
        "detections": serialize_detections(detections),
        "image_path": image_path,
        "risk_level": risk_level,
        "has_person": has_person,
        "has_vehicle": has_vehicle,
        "person_count": person_count,
        "vehicle_count": vehicle_count,
    }


def should_use_fast_interval_mode(session):
    return session.interval_seconds <= FAST_INTERVAL_THRESHOLD


def put_latest(target_queue, item):
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
    def __init__(self, video_path, interval_seconds):
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

        self.snapshot_queue = queue.Queue(maxsize=int(SAMPLING_CONFIG["snapshot_queue_size"]))
        self.result_queue = queue.Queue(maxsize=int(SAMPLING_CONFIG["result_queue_size"]))
        self.api_event_queue = queue.Queue(maxsize=64)

        self.snapshot_count = 0
        self.analysis_count = 0
        self.max_persons = 0
        self.max_vehicles = 0
        self.inference_started_at = None

        self.recent_events = []
        self.latest_event_path = None
        self.event_count = 0

        self.threads = []

    def update_latest_frame(self, frame, frame_index, frame_time):
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

    def request_stop(self, status, message):
        with self.final_lock:
            if self.finalized:
                return
            self.final_status = status
            self.final_message = message
        self.stop_event.set()

    def finalize(self):
        with self.final_lock:
            if self.finalized:
                return
            self.finalized = True
            status = self.final_status
            message = self.final_message

        set_video_state(
            status=status,
            message=message,
            completed=True,
        )

        with session_lock:
            global current_session
            if current_session is self:
                current_session = None


def video_stream_worker(session):
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
            profile_name=runtime_config["profile_name"],
            model_path=MODEL_PATH,
            secondary_model_path=SECONDARY_MODEL_PATH,            
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


def snapshot_scheduler(session):
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


def inference_worker(session):
    try:
        while not session.stop_event.is_set() or not session.snapshot_queue.empty():
            try:
                snapshot = session.snapshot_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            dropped_snapshots = 0
            while True:
                try:
                    snapshot = session.snapshot_queue.get_nowait()
                    dropped_snapshots += 1
                except queue.Empty:
                    break

            fast_interval_mode = should_use_fast_interval_mode(session)
            
            quick_snapshot = dict(snapshot)
            quick_snapshot["detections"] = predict_boxes(
                snapshot["frame"],
                use_secondary=False,
                primary_model_override=fast_detection_model,
                primary_confidence_override=FAST_CONFIDENCE_THRESHOLD,
                primary_target_width_override=FAST_TARGET_WIDTH,
                primary_inference_imgsz_override=FAST_INFERENCE_IMGSZ,
            )
            quick_snapshot["analysis_mode"] = "fast-interval" if fast_interval_mode else "quick-preview"
            quick_snapshot["is_final_result"] = fast_interval_mode
            put_latest(session.result_queue, quick_snapshot)

            if session.inference_started_at is None:
                session.inference_started_at = time.perf_counter()
            session.analysis_count += 1

            if fast_interval_mode:
                continue
            if dropped_snapshots > 0 or not session.snapshot_queue.empty():
                continue

            use_secondary = secondary_detection_model is not None and not FAST_DISABLE_SECONDARY
            refined_snapshot = dict(snapshot)
            refined_snapshot["detections"] = predict_boxes(refined_snapshot["frame"], use_secondary=use_secondary)
            refined_snapshot["analysis_mode"] = "full-ensemble" if use_secondary else "primary-only"
            
            refined_snapshot["is_final_result"] = True
            put_latest(session.result_queue, refined_snapshot)
    except Exception:
        session.request_stop("error", traceback.format_exc())


def result_handler(session):
    try:
        save_dir = os.path.abspath(EVENT_CONFIG["save_directory"])
        os.makedirs(save_dir, exist_ok=True)

        while not session.stop_event.is_set() or not session.result_queue.empty():
            try:
                result = session.result_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            raw_frame = result["frame"]
            detections = result["detections"]
            annotated = draw_detections(raw_frame, detections)
            raw_base64 = encode_image_base64(raw_frame)
            annotated_base64 = encode_image_base64(annotated)

            has_person, has_vehicle, risk_level, risk_summary = infer_risk(detections)
            persons = sum(1 for item in detections if item["label"] == "person")
            vehicles = sum(1 for item in detections if item["label"] == "vehicle")
            object_labels = []
            for item in detections:
                display_label = item.get("display_label", item["label"])
                if display_label not in object_labels:
                    object_labels.append(display_label)

            session.max_persons = max(session.max_persons, persons)
            session.max_vehicles = max(session.max_vehicles, vehicles)

            image_path = ""
            if EVENT_CONFIG["enabled"] and (has_person or has_vehicle):
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
                    "labels": object_labels,
                    "path": image_path,
                }
                session.recent_events.insert(0, event_payload)
                session.recent_events = session.recent_events[: int(EVENT_CONFIG["max_recent_events"])]
                put_latest(session.api_event_queue, event_payload)

            payload = build_result_payload(
                result["captured_at"],
                detections,
                image_path,
                risk_level,
                has_person,
                has_vehicle,
            )
            payload["analysis_mode"] = result.get("analysis_mode")
            payload["frame_index"] = result["frame_index"]
            payload["is_final_result"] = bool(result.get("is_final_result", True))            

            elapsed = max(time.perf_counter() - session.inference_started_at, 1e-6)
            inference_fps = session.analysis_count / elapsed if session.inference_started_at else 0.0

            state_snapshot = snapshot_video_state()
            latest_result_frame_index = int(state_snapshot.get("latest_result_frame_index") or 0)
            latest_snapshot_frame_index = int(state_snapshot.get("latest_snapshot_frame_index") or 0)
            should_publish_visual = result["frame_index"] >= latest_result_frame_index
            is_stale_for_current_snapshot = result["frame_index"] < latest_snapshot_frame_index

            state_updates = {
                "analysis_count": session.analysis_count,
                "inference_fps": inference_fps,
                "analysis_pending": not session.snapshot_queue.empty(),                
                "event_count": session.event_count,
                "latest_event_path": session.latest_event_path,
                "recent_events": list(session.recent_events),
                "pending_api_events": session.api_event_queue.qsize(),
                "event_directory": save_dir,
            }

            if should_publish_visual and not is_stale_for_current_snapshot:
                state_updates.update(
                    {
                        "latest_snapshot_base64": raw_base64,
                        "latest_snapshot_timestamp": result["captured_at"],
                        "latest_snapshot_frame_index": result["frame_index"],
                        "latest_result_base64": annotated_base64,
                        "latest_result": payload,
                        "latest_result_frame_index": result["frame_index"],
                        "latest_detections": payload["detections"],
                        "analysis_mode": result.get("analysis_mode"),
                        "current_persons": persons,
                        "current_vehicle_like_objects": vehicles,
                        "max_persons": session.max_persons,
                        "max_vehicle_like_objects": session.max_vehicles,
                        "risk_level": risk_level,
                        "risk_summary": risk_summary,
                    }
                )

            set_video_state(**state_updates)
    except Exception:
        session.request_stop("error", traceback.format_exc())


def session_monitor(session):
    for thread in session.threads:
        thread.join()
    session.finalize()


def start_video_job(video_path, interval_seconds):
    global current_session

    with session_lock:
        if current_session is not None:
            return {"status": "error", "msg": "another video analysis job is already running"}

        session = VideoAnalysisSession(video_path, interval_seconds)
        current_session = session

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
        target=session_monitor,
        args=(session,),
        name="session_monitor",
        daemon=True,
    )
    monitor_thread.start()

    set_video_state(
        status="processing",
        message="sampling analysis started",
        video_path=video_path,
        completed=False,
        pprofile_name=runtime_config["profile_name"],
        model_path=MODEL_PATH,
        secondary_model_path=SECONDARY_MODEL_PATH,        
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
    )

    return {
        "status": "accepted",
        "msg": "video analysis started",
        "analysis_interval_seconds": session.interval_seconds,        
    }


def stop_video_job():
    with session_lock:
        session = current_session

    if session is None:
        return {"status": "ok", "msg": "no active session"}

    session.request_stop("stopped", "user stopped video analysis")
    return {"status": "ok", "msg": "stop requested"}


context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://127.0.0.1:5555")

print("[AI Engine] READY", flush=True)
threading.Thread(target=load_model, daemon=True).start()


while True:
    try:
        message = socket.recv_string()
        data = json.loads(message)
        req_type = data.get("type")

        if req_type == "status":
            socket.send_string(
                json.dumps(
                    {
                        "status": "ok",
                        "model_loaded": model_loaded,
                        "model_loading": model_loading,
                        "device": device,
                        "model_path": MODEL_PATH,
                        "secondary_model_path": SECONDARY_MODEL_PATH,
                        "profile_name": runtime_config["profile_name"],                        
                        "allowed_classes": sorted(ALLOWED_CLASSES),
                        "default_interval_seconds": SAMPLING_CONFIG["default_interval_seconds"],
                    }
                )
            )
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

            print(
                f"[AI Engine] start video: {video_path} / interval={interval_seconds}s",
                flush=True,
            )
            socket.send_string(json.dumps(start_video_job(video_path, interval_seconds)))
            continue

        socket.send_string(json.dumps({"status": "error", "msg": "unknown request"}))
    except Exception as exc:
        err_msg = traceback.format_exc()
        print(f"[AI Engine ERR]\n{err_msg}", flush=True)
        socket.send_string(json.dumps({"status": "error", "msg": str(exc)}))
