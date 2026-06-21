"""
Utility Functions for Traffic Violation Detection System
"""
import cv2
import numpy as np
import uuid
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

logger = logging.getLogger(__name__)


# ─── ID / Timestamp Helpers ───────────────────────────────────

def generate_violation_id() -> str:
    """Generate unique violation ID."""
    return f"VIO_{uuid.uuid4().hex[:12].upper()}"


def get_timestamp() -> str:
    """Get current UTC ISO timestamp."""
    return datetime.utcnow().isoformat() + "Z"


def format_timestamp(ts: datetime) -> str:
    """Format datetime for human display."""
    return ts.strftime("%Y-%m-%d %H:%M:%S")


# ─── Geometry Helpers ─────────────────────────────────────────

def calculate_iou(box1: Tuple, box2: Tuple) -> float:
    """Calculate Intersection over Union between two (x1,y1,x2,y2) boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1[:4]
    x1_2, y1_2, x2_2, y2_2 = box2[:4]

    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = max(0, x2_1 - x1_1) * max(0, y2_1 - y1_1)
    box2_area = max(0, x2_2 - x1_2) * max(0, y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def calculate_ioa(inner_box: Tuple, outer_box: Tuple) -> float:
    """Intersection over Area of the inner box (how much of inner is inside outer)."""
    x1_1, y1_1, x2_1, y2_1 = inner_box[:4]
    x1_2, y1_2, x2_2, y2_2 = outer_box[:4]

    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    inner_area = max(0, x2_1 - x1_1) * max(0, y2_1 - y1_1)

    return inter_area / inner_area if inner_area > 0 else 0.0


def calculate_centroid(bbox: Tuple) -> Tuple[int, int]:
    """Calculate centroid of bounding box (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = bbox[:4]
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def get_bbox_area(bbox: Tuple) -> int:
    """Calculate area of bounding box."""
    x1, y1, x2, y2 = bbox[:4]
    return max(0, int(x2 - x1)) * max(0, int(y2 - y1))


def is_point_in_polygon(point: Tuple[int, int], polygon: List[Tuple[int, int]]) -> bool:
    """Ray-casting algorithm to check if point is inside polygon."""
    x, y = point
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def calculate_angle(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """Calculate angle in degrees from p1 to p2. 0=right, 90=down, 180=left."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = np.degrees(np.arctan2(dy, dx))
    return angle if angle >= 0 else angle + 360


# ─── Drawing Helpers ──────────────────────────────────────────

def add_text_with_background(
    frame: np.ndarray, text: str, position: Tuple[int, int],
    font_scale: float = 0.55, color: Tuple = (255, 255, 255),
    bg_color: Tuple = (0, 0, 0), thickness: int = 1,
    padding: int = 4
) -> np.ndarray:
    """Draw text with a solid background rectangle."""
    x, y = int(position[0]), int(position[1])
    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    cv2.rectangle(
        frame,
        (x - padding, y - th - padding),
        (x + tw + padding, y + baseline + padding),
        bg_color, -1
    )
    cv2.putText(
        frame, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness
    )
    return frame


# ─── Data Helpers ─────────────────────────────────────────────

def save_json(data: Dict, filepath: str):
    """Save dictionary as JSON file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: str) -> Dict:
    """Load JSON file to dictionary."""
    with open(filepath, "r") as f:
        return json.load(f)


# ─── Circular Buffer ─────────────────────────────────────────

class CircularBuffer:
    """Fixed-size circular buffer for frame history."""

    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self._buf: list = []
        self._idx: int = 0

    def add(self, item):
        if len(self._buf) < self.maxsize:
            self._buf.append(item)
        else:
            self._buf[self._idx] = item
        self._idx = (self._idx + 1) % self.maxsize

    def get_all(self) -> list:
        if len(self._buf) < self.maxsize:
            return list(self._buf)
        return self._buf[self._idx:] + self._buf[:self._idx]

    def __len__(self):
        return len(self._buf)

    def clear(self):
        self._buf.clear()
        self._idx = 0
