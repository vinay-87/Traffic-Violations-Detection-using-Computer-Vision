"""Configuration Module for Traffic Violation Detection System

Keeps all tunable parameters in one place so we don't have magic
numbers scattered across the codebase. Most of these were found
through trial-and-error on Bengaluru traffic footage.
"""
import os
from pathlib import Path

# ─── Base Paths ───────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent.resolve()
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EVIDENCE_DIR = PROCESSED_DIR / "evidence"
MODELS_DIR = BASE_DIR / "models"
SAMPLE_DIR = DATA_DIR / "sample_videos"

# Create directories on import
for _d in [RAW_DIR, PROCESSED_DIR, EVIDENCE_DIR, MODELS_DIR, SAMPLE_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ─── Detection Settings ──────────────────────────────────────
DETECTION = {
    "model_path": str(MODELS_DIR / "yolov8n.pt"),
    "input_size": 640,
    "confidence_threshold": 0.40,
    "iou_threshold": 0.50,
    "device": "",  # "" = auto (GPU if available, else CPU)
    "max_detections": 300,
    # COCO class IDs we care about
    "classes_of_interest": [0, 1, 2, 3, 5, 7, 9, 11],
}

# COCO class names (YOLOv8 default)
COCO_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    9: "traffic_light",
    11: "stop_sign",
}

# Vehicle class IDs in COCO
VEHICLE_CLASS_IDS = [1, 2, 3, 5, 7]  # bicycle, car, motorcycle, bus, truck
MOTORIZED_VEHICLE_IDS = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# ─── Tracking Settings ───────────────────────────────────────
TRACKING = {
    "max_age": 30,
    "n_init": 3,
    "max_iou_distance": 0.7,
    "embedder": "mobilenet",
    "half": True,
    "bgr": True,
    "max_trajectory_len": 60,
}

# ─── Violation Detection Rules ────────────────────────────────
VIOLATIONS = {
    "helmet": {
        "enabled": True,
        "motorcycle_class_id": 3,
        "person_class_id": 0,
        "iou_threshold": 0.3,
        "head_ratio": 0.30,       # top 30% of rider bbox is roughly head
        "temporal_frames": 4,     # need 4 consecutive frames to confirm
        "min_confidence": 0.40,   # below this the detection is too noisy
    },
    "triple_riding": {
        "enabled": True,
        "motorcycle_class_id": 3,
        "person_class_id": 0,
        "min_overlap": 0.25,
        "max_persons": 2,
        "temporal_frames": 4,
    },
    "red_light": {
        "enabled": True,
        "traffic_light_class_id": 9,
        "stop_line_y_ratio": 0.65,   # stop line at 65% of frame height
        # red in HSV wraps around 0/180 so we need two ranges
        "red_hsv_lower1": (0, 100, 100),
        "red_hsv_upper1": (10, 255, 255),
        "red_hsv_lower2": (160, 100, 100),
        "red_hsv_upper2": (180, 255, 255),
        "red_ratio_threshold": 0.10,
        "temporal_frames": 3,
    },
    "stop_line": {
        "enabled": True,
        "stop_line_y_ratio": 0.65,
        "cross_threshold": 0.3,
        "temporal_frames": 3,
    },
    "wrong_side": {
        "enabled": True,
        "expected_direction": "left_to_right",
        "expected_angle": 0,       # 0 deg = moving rightward on screen
        "angle_tolerance": 90,     # anything beyond this is wrong side
        "min_trajectory_len": 12,
        "temporal_frames": 6,
    },
    "seatbelt": {
        "enabled": True,
        # only check 4-wheelers, obviously bikes don't have seatbelts
        "car_class_ids": [2, 5, 7],
        "person_class_id": 0,
        "driver_region": "front_left",  # India = right-hand drive
        "temporal_frames": 5,
    },
    "illegal_parking": {
        "enabled": True,
        "stationary_threshold_px": 8,
        "stationary_min_frames": 45,  # ~1.5 sec at 30fps
        "no_parking_zones": [],
        "temporal_frames": 45,
    },
}

# ─── OCR Settings ─────────────────────────────────────────────
OCR = {
    "use_gpu": False,  # Safe default; auto-upgraded if GPU found
    "languages": ["en"],
    "min_confidence": 0.4,
    "plate_format_regex": r"^[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{4}$",
    "character_corrections": {
        "O": "0", "I": "1", "Z": "2", "S": "5",
        "B": "8", "Q": "0", "G": "6", "D": "0",
    },
    "preprocess": {
        "clahe_clip_limit": 2.0,
        "clahe_grid": (8, 8),
        "bilateral_d": 9,
        "bilateral_sigma_color": 75,
        "bilateral_sigma_space": 75,
        "target_height": 128,
        "max_width": 320,
    },
}

# ─── Evidence Generation ──────────────────────────────────────
EVIDENCE = {
    "output_dir": str(EVIDENCE_DIR),
    "bounding_box_thickness": 2,
    "violation_color": (0, 0, 255),      # BGR: Red
    "compliance_color": (0, 200, 0),     # BGR: Green
    "warning_color": (0, 200, 255),      # BGR: Yellow/Orange
    "info_color": (255, 200, 0),         # BGR: Cyan
    "font_scale": 0.55,
    "font_thickness": 2,
    "snapshot_quality": 95,
    "clip_duration_sec": 5,
    "buffer_size_frames": 300,           # 10s at 30fps
}

# ─── Database ─────────────────────────────────────────────────
DATABASE = {
    "url": f"sqlite:///{DATA_DIR / 'violations.db'}",
    "echo": False,
}

# ─── Dashboard ────────────────────────────────────────────────
DASHBOARD = {
    "title": "AI Traffic Violation Detection System",
    "page_icon": "🚦",
    "layout": "wide",
}

# ─── API ──────────────────────────────────────────────────────
API = {
    "host": "0.0.0.0",
    "port": 8000,
}

# ─── Video ────────────────────────────────────────────────────
VIDEO = {
    "default_fps": 30,
    "display_size": (1280, 720),
}
