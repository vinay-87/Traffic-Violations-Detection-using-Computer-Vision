"""
Traffic Violation Detection Engine

This is the core intelligence layer. Each violation type has its own
detection method, and we use temporal confirmation buffers to avoid
flagging one-off misdetections as real violations. The idea is that
if something looks like a violation for N consecutive frames, it
probably is one.
"""
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
import logging

import supervision as sv

from src.config import VIOLATIONS, COCO_CLASSES, MOTORIZED_VEHICLE_IDS
from src.utils import (
    calculate_iou, calculate_ioa, calculate_centroid,
    get_bbox_area, generate_violation_id, get_timestamp,
)

logger = logging.getLogger(__name__)


# ─── Data Structures ──────────────────────────────────────────

@dataclass
class Violation:
    """Complete violation record."""
    violation_id: str
    violation_type: str
    timestamp: str
    confidence: float
    vehicle_type: str
    vehicle_bbox: Tuple[int, int, int, int]
    track_id: Optional[int] = None
    plate_number: Optional[str] = None
    plate_confidence: Optional[float] = None
    plate_bbox: Optional[Tuple[int, int, int, int]] = None
    violation_bbox: Optional[Tuple[int, int, int, int]] = None
    details: Dict = field(default_factory=dict)
    snapshot_path: Optional[str] = None
    video_clip_path: Optional[str] = None
    camera_id: str = "CAM_001"
    location: Optional[Dict] = None
    inference_time_ms: float = 0.0
    model_version: str = "v1.0"
    frame_idx: int = 0

    def to_dict(self) -> Dict:
        return {
            "violation_id": self.violation_id,
            "violation_type": self.violation_type,
            "timestamp": self.timestamp,
            "confidence": round(self.confidence, 4),
            "vehicle_type": self.vehicle_type,
            "vehicle_bbox": list(self.vehicle_bbox) if self.vehicle_bbox else None,
            "track_id": self.track_id,
            "plate_number": self.plate_number,
            "plate_confidence": round(self.plate_confidence, 4) if self.plate_confidence else None,
            "violation_bbox": list(self.violation_bbox) if self.violation_bbox else None,
            "details": self.details,
            "snapshot_path": self.snapshot_path,
            "video_clip_path": self.video_clip_path,
            "camera_id": self.camera_id,
            "location": self.location,
            "inference_time_ms": round(self.inference_time_ms, 2),
            "model_version": self.model_version,
        }


VEHICLE_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 1: "bicycle"}


# ─── Main Engine ──────────────────────────────────────────────

class ViolationEngine:
    """Rule-based violation detection with temporal confirmation."""

    def __init__(self, config: Optional[Dict] = None):
        self.cfg = config or VIOLATIONS
        # temporal_buffers[vtype][key] = {"first": frame_idx, "count": N}
        self._tbuf: Dict[str, Dict[str, Dict]] = {
            k: {} for k in self.cfg
        }
        # Set of already-fired violation keys to avoid duplicates
        self._fired: set = set()

    # ─── Public API ───────────────────────────────────────────

    def detect_violations(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        tracks: sv.Detections,
        frame_idx: int = 0,
        tracker=None,
    ) -> List[Violation]:
        """
        Main entry: analyse one frame and return confirmed violations.
        `tracker` is the VehicleTracker instance (needed for trajectory queries).
        """
        h, w = frame.shape[:2]
        violations: List[Violation] = []

        if self.cfg["helmet"]["enabled"]:
            violations.extend(
                self._check_helmet(frame, detections, tracks, frame_idx)
            )
        if self.cfg["triple_riding"]["enabled"]:
            violations.extend(
                self._check_triple_riding(detections, tracks, frame_idx)
            )
        if self.cfg["red_light"]["enabled"]:
            violations.extend(
                self._check_red_light(frame, detections, tracks, frame_idx, h, w)
            )
        if self.cfg["stop_line"]["enabled"]:
            violations.extend(
                self._check_stop_line(detections, tracks, frame_idx, h, w)
            )
        if self.cfg["wrong_side"]["enabled"] and tracker is not None:
            violations.extend(
                self._check_wrong_side(tracks, frame_idx, tracker)
            )
        if self.cfg["seatbelt"]["enabled"]:
            violations.extend(
                self._check_seatbelt(frame, detections, tracks, frame_idx)
            )
        if self.cfg["illegal_parking"]["enabled"] and tracker is not None:
            violations.extend(
                self._check_illegal_parking(tracks, frame_idx, tracker)
            )

        return violations

    def reset(self):
        self._tbuf = {k: {} for k in self.cfg}
        self._fired.clear()

    # ─── 1. Helmet Non-Compliance ─────────────────────────────

    def _check_helmet(self, frame, detections, tracks, fidx):
        violations = []
        cfg = self.cfg["helmet"]
        if len(detections) == 0:
            return violations

        # Get motorcycles and persons
        mot_mask = detections.class_id == cfg["motorcycle_class_id"]
        per_mask = detections.class_id == cfg["person_class_id"]
        motorcycles = detections[mot_mask]
        persons = detections[per_mask]

        if len(motorcycles) == 0 or len(persons) == 0:
            return violations

        for mi in range(len(motorcycles)):
            mbbox = tuple(motorcycles.xyxy[mi])
            mconf = float(motorcycles.confidence[mi])

            # Find riders overlapping this motorcycle
            for pi in range(len(persons)):
                pbbox = tuple(persons.xyxy[pi])
                pconf = float(persons.confidence[pi])
                overlap = calculate_ioa(pbbox, mbbox)

                if overlap < cfg["iou_threshold"]:
                    continue

                # Extract head region (top portion of person bbox)
                x1, y1, x2, y2 = map(int, pbbox)
                head_h = int((y2 - y1) * cfg["head_ratio"])
                head_y2 = y1 + max(head_h, 10)
                head_x1 = max(0, x1)
                head_y1 = max(0, y1)
                head_x2 = min(frame.shape[1], x2)
                head_y2_c = min(frame.shape[0], head_y2)
                head_region = frame[head_y1:head_y2_c, head_x1:head_x2]

                if head_region.size == 0:
                    continue

                has_helmet = self._analyse_helmet_region(head_region)

                if not has_helmet:
                    key = f"helmet_m{mi}_p{pi}"
                    if self._temporal_confirm("helmet", key, fidx, cfg["temporal_frames"]):
                        violations.append(Violation(
                            violation_id=generate_violation_id(),
                            violation_type="helmet_non_compliance",
                            timestamp=get_timestamp(),
                            confidence=min(mconf, pconf) * 0.90,
                            vehicle_type="motorcycle",
                            vehicle_bbox=tuple(int(v) for v in mbbox),
                            violation_bbox=(head_x1, head_y1, head_x2, head_y2_c),
                            details={"helmet_worn": False, "method": "head_region_analysis"},
                            frame_idx=fidx,
                        ))
        # Heuristic: if someone on a motorcycle doesn't have a
        # helmet-like shape on their head, we flag it. Not 100%
        # accurate (dark hair can look like a helmet) but temporal
        # confirmation across multiple frames brings precision up.
        return violations

    def _analyse_helmet_region(self, region: np.ndarray) -> bool:
        """
        Heuristic helmet detection on the head crop.
        Helmets tend to be a large, uniform, rounded, dark/colored object.
        No helmet = hair/skin tones, less uniform, smaller filled area.
        """
        if region.size == 0 or region.shape[0] < 5 or region.shape[1] < 5:
            return False

        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        # 1) Edge density — helmets have smooth surfaces (low edge density)
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.count_nonzero(edges) / edges.size

        # 2) Color uniformity — helmets are usually one solid color
        h, s, v = cv2.split(hsv)
        sat_std = float(np.std(s))
        val_mean = float(np.mean(v))

        # 3) Circular shape via contours
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        has_round = False
        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                if circularity > 0.4 and area > (region.shape[0] * region.shape[1] * 0.15):
                    has_round = True
                    break

        # Decision: helmet if smooth surface + uniform color + roundish shape
        helmet_score = 0
        if edge_ratio < 0.15:
            helmet_score += 1
        if sat_std < 40:
            helmet_score += 1
        if has_round:
            helmet_score += 1
        if val_mean > 40:
            helmet_score += 0.5

        # Overall: if 2+ indicators say helmet, we trust it.
        # This isn't perfect but works decently on real footage.
        return helmet_score >= 2.0

    # ─── 2. Triple Riding ─────────────────────────────────────

    def _check_triple_riding(self, detections, tracks, fidx):
        violations = []
        cfg = self.cfg["triple_riding"]
        if len(detections) == 0:
            return violations

        mot_mask = detections.class_id == cfg["motorcycle_class_id"]
        per_mask = detections.class_id == cfg["person_class_id"]
        motorcycles = detections[mot_mask]
        persons = detections[per_mask]

        if len(motorcycles) == 0 or len(persons) < 3:
            return violations

        for mi in range(len(motorcycles)):
            mbbox = tuple(motorcycles.xyxy[mi])
            mconf = float(motorcycles.confidence[mi])
            # Expand motorcycle bbox slightly for overlap check
            mx1, my1, mx2, my2 = mbbox
            mw, mh = mx2 - mx1, my2 - my1
            expanded = (mx1 - mw * 0.15, my1 - mh * 0.3,
                        mx2 + mw * 0.15, my2)

            rider_count = 0
            rider_bboxes = []
            for pi in range(len(persons)):
                pbbox = tuple(persons.xyxy[pi])
                overlap = calculate_ioa(pbbox, expanded)
                if overlap > cfg["min_overlap"]:
                    rider_count += 1
                    rider_bboxes.append(tuple(int(v) for v in pbbox))

            if rider_count > cfg["max_persons"]:
                key = f"triple_m{mi}"
                if self._temporal_confirm("triple_riding", key, fidx,
                                          cfg["temporal_frames"]):
                    violations.append(Violation(
                        violation_id=generate_violation_id(),
                        violation_type="triple_riding",
                        timestamp=get_timestamp(),
                        confidence=mconf * 0.92,
                        vehicle_type="motorcycle",
                        vehicle_bbox=tuple(int(v) for v in mbbox),
                        details={
                            "rider_count": rider_count,
                            "max_allowed": cfg["max_persons"],
                        },
                        frame_idx=fidx,
                    ))
        return violations

    # ─── 3. Red Light Violation ───────────────────────────────

    def _check_red_light(self, frame, detections, tracks, fidx, fh, fw):
        violations = []
        cfg = self.cfg["red_light"]
        if len(detections) == 0:
            return violations

        # Check if red light is active
        red_detected = self._is_red_light(frame, detections, cfg)
        if not red_detected:
            return violations

        stop_y = int(fh * cfg["stop_line_y_ratio"])

        # Check vehicles past stop line
        veh_mask = np.isin(detections.class_id, MOTORIZED_VEHICLE_IDS)
        vehicles = detections[veh_mask]

        for vi in range(len(vehicles)):
            vbbox = tuple(vehicles.xyxy[vi])
            vconf = float(vehicles.confidence[vi])
            vcid = int(vehicles.class_id[vi])
            cx, cy = calculate_centroid(vbbox)

            if cy > stop_y:
                key = f"redlight_v{vi}_{int(vbbox[0])}"
                if self._temporal_confirm("red_light", key, fidx,
                                          cfg["temporal_frames"]):
                    violations.append(Violation(
                        violation_id=generate_violation_id(),
                        violation_type="red_light_violation",
                        timestamp=get_timestamp(),
                        confidence=vconf * 0.93,
                        vehicle_type=VEHICLE_NAMES.get(vcid, "vehicle"),
                        vehicle_bbox=tuple(int(v) for v in vbbox),
                        details={
                            "stop_line_y": stop_y,
                            "vehicle_centroid_y": cy,
                        },
                        frame_idx=fidx,
                    ))
        return violations

    def _is_red_light(self, frame, detections, cfg):
        """Check if any detected traffic light is showing red."""
        tl_mask = detections.class_id == cfg["traffic_light_class_id"]
        tls = detections[tl_mask]

        if len(tls) > 0:
            for bbox in tls.xyxy:
                x1, y1, x2, y2 = map(int, bbox)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                # Divide traffic light into thirds — top = red
                th = roi.shape[0] // 3
                top_third = roi[:max(th, 1), :]
                if top_third.size == 0:
                    continue

                hsv = cv2.cvtColor(top_third, cv2.COLOR_BGR2HSV)
                m1 = cv2.inRange(hsv,
                                 np.array(cfg["red_hsv_lower1"]),
                                 np.array(cfg["red_hsv_upper1"]))
                m2 = cv2.inRange(hsv,
                                 np.array(cfg["red_hsv_lower2"]),
                                 np.array(cfg["red_hsv_upper2"]))
                red_ratio = (np.count_nonzero(m1) + np.count_nonzero(m2)) / m1.size
                if red_ratio > cfg["red_ratio_threshold"]:
                    return True

        # Fallback: scan top-left region of frame for red circle
        h, w = frame.shape[:2]
        top_region = frame[:int(h * 0.35), :]
        if top_region.size > 0:
            hsv = cv2.cvtColor(top_region, cv2.COLOR_BGR2HSV)
            m1 = cv2.inRange(hsv, np.array(cfg["red_hsv_lower1"]),
                             np.array(cfg["red_hsv_upper1"]))
            m2 = cv2.inRange(hsv, np.array(cfg["red_hsv_lower2"]),
                             np.array(cfg["red_hsv_upper2"]))
            ratio = (np.count_nonzero(m1) + np.count_nonzero(m2)) / m1.size
            if ratio > 0.03:
                return True

        return False

    # ─── 4. Stop Line Violation ───────────────────────────────

    def _check_stop_line(self, detections, tracks, fidx, fh, fw):
        violations = []
        cfg = self.cfg["stop_line"]
        if len(detections) == 0:
            return violations

        stop_y = int(fh * cfg["stop_line_y_ratio"])

        veh_mask = np.isin(detections.class_id, MOTORIZED_VEHICLE_IDS)
        vehicles = detections[veh_mask]

        for vi in range(len(vehicles)):
            vbbox = tuple(vehicles.xyxy[vi])
            vconf = float(vehicles.confidence[vi])
            vcid = int(vehicles.class_id[vi])
            _, cy = calculate_centroid(vbbox)
            veh_h = vbbox[3] - vbbox[1]

            if veh_h < 1:
                continue
            cross_ratio = (cy - stop_y) / veh_h
            if cross_ratio > cfg["cross_threshold"]:
                key = f"stopline_v{vi}_{int(vbbox[0])}"
                if self._temporal_confirm("stop_line", key, fidx,
                                          cfg["temporal_frames"]):
                    violations.append(Violation(
                        violation_id=generate_violation_id(),
                        violation_type="stop_line_violation",
                        timestamp=get_timestamp(),
                        confidence=vconf * 0.88,
                        vehicle_type=VEHICLE_NAMES.get(vcid, "vehicle"),
                        vehicle_bbox=tuple(int(v) for v in vbbox),
                        details={"cross_ratio": round(cross_ratio, 3)},
                        frame_idx=fidx,
                    ))
        return violations

    # ─── 5. Wrong-Side Driving ────────────────────────────────

    def _check_wrong_side(self, tracks, fidx, tracker):
        violations = []
        cfg = self.cfg["wrong_side"]
        if tracks.tracker_id is None or len(tracks) == 0:
            return violations

        veh_mask = np.isin(tracks.class_id, MOTORIZED_VEHICLE_IDS)
        vehicles = tracks[veh_mask]

        if vehicles.tracker_id is None:
            return violations

        expected = cfg["expected_angle"]
        tolerance = cfg["angle_tolerance"]

        for vi in range(len(vehicles)):
            tid = int(vehicles.tracker_id[vi])
            angle = tracker.get_direction_angle(tid, cfg["min_trajectory_len"])
            if angle is None:
                continue

            # Calculate angular difference
            diff = abs(angle - expected)
            if diff > 180:
                diff = 360 - diff

            if diff > tolerance:
                key = f"wrongside_{tid}"
                if self._temporal_confirm("wrong_side", key, fidx,
                                          cfg["temporal_frames"]):
                    vbbox = tuple(vehicles.xyxy[vi])
                    vcid = int(vehicles.class_id[vi])
                    vconf = float(vehicles.confidence[vi])
                    violations.append(Violation(
                        violation_id=generate_violation_id(),
                        violation_type="wrong_side_driving",
                        timestamp=get_timestamp(),
                        confidence=vconf * 0.80,
                        vehicle_type=VEHICLE_NAMES.get(vcid, "vehicle"),
                        vehicle_bbox=tuple(int(v) for v in vbbox),
                        track_id=tid,
                        details={
                            "detected_angle": round(angle, 1),
                            "expected_angle": expected,
                            "deviation": round(diff, 1),
                        },
                        frame_idx=fidx,
                    ))
        return violations

    # ─── 6. Seatbelt Non-Compliance ───────────────────────────

    def _check_seatbelt(self, frame, detections, tracks, fidx):
        violations = []
        cfg = self.cfg["seatbelt"]
        if len(detections) == 0:
            return violations

        car_mask = np.isin(detections.class_id, cfg["car_class_ids"])
        cars = detections[car_mask]

        for ci in range(len(cars)):
            cbbox = tuple(cars.xyxy[ci])
            cconf = float(cars.confidence[ci])
            ccid = int(cars.class_id[ci])
            x1, y1, x2, y2 = map(int, cbbox)
            cw, ch = x2 - x1, y2 - y1

            if cw < 40 or ch < 40:
                continue

            # Driver region: front-left for India (right-hand drive)
            # = left half, top half of car bbox
            dx1, dy1 = x1, y1
            dx2, dy2 = x1 + cw // 2, y1 + ch // 2
            dx1, dy1 = max(0, dx1), max(0, dy1)
            dx2 = min(frame.shape[1], dx2)
            dy2 = min(frame.shape[0], dy2)
            driver_roi = frame[dy1:dy2, dx1:dx2]

            if driver_roi.size == 0 or driver_roi.shape[0] < 10:
                continue

            has_seatbelt = self._detect_seatbelt(driver_roi)

            if not has_seatbelt:
                key = f"seatbelt_c{ci}_{x1}"
                if self._temporal_confirm("seatbelt", key, fidx,
                                          cfg["temporal_frames"]):
                    violations.append(Violation(
                        violation_id=generate_violation_id(),
                        violation_type="seatbelt_non_compliance",
                        timestamp=get_timestamp(),
                        confidence=cconf * 0.78,
                        vehicle_type=VEHICLE_NAMES.get(ccid, "car"),
                        vehicle_bbox=tuple(int(v) for v in cbbox),
                        violation_bbox=(dx1, dy1, dx2, dy2),
                        details={"seatbelt_worn": False},
                        frame_idx=fidx,
                    ))
        return violations

    def _detect_seatbelt(self, roi: np.ndarray) -> bool:
        """Detect diagonal seatbelt strap via edge + line analysis."""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 40, 120)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25,
                                minLineLength=15, maxLineGap=8)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x2 - x1) < 1:
                    continue
                angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                # Seatbelt = diagonal strap ~25-65 degrees
                if 20 < angle < 70:
                    length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    if length > min(roi.shape[:2]) * 0.3:
                        return True
        # The seatbelt check is admittedly the weakest of the seven.
        # In practice you'd want a dedicated classifier trained on
        # driver-cabin crops, but this edge-based approach is a
        # reasonable first pass for the prototype.
        return False

    # ─── 7. Illegal Parking ───────────────────────────────────

    def _check_illegal_parking(self, tracks, fidx, tracker):
        violations = []
        cfg = self.cfg["illegal_parking"]
        if tracks.tracker_id is None or len(tracks) == 0:
            return violations

        veh_mask = np.isin(tracks.class_id, MOTORIZED_VEHICLE_IDS)
        vehicles = tracks[veh_mask]
        if vehicles.tracker_id is None:
            return violations

        for vi in range(len(vehicles)):
            tid = int(vehicles.tracker_id[vi])
            is_parked = tracker.is_stationary(
                tid,
                threshold_px=cfg["stationary_threshold_px"],
                min_frames=cfg["stationary_min_frames"],
            )
            if not is_parked:
                continue

            # Check no-parking zones if configured
            vbbox = tuple(vehicles.xyxy[vi])
            cx, cy = calculate_centroid(vbbox)
            in_zone = False

            if cfg["no_parking_zones"]:
                from src.utils import is_point_in_polygon
                for zone in cfg["no_parking_zones"]:
                    if is_point_in_polygon((cx, cy), zone):
                        in_zone = True
                        break
                if not in_zone:
                    continue

            key = f"parking_{tid}"
            if self._temporal_confirm("illegal_parking", key, fidx,
                                      cfg["temporal_frames"]):
                vcid = int(vehicles.class_id[vi])
                vconf = float(vehicles.confidence[vi])
                violations.append(Violation(
                    violation_id=generate_violation_id(),
                    violation_type="illegal_parking",
                    timestamp=get_timestamp(),
                    confidence=vconf * 0.75,
                    vehicle_type=VEHICLE_NAMES.get(vcid, "vehicle"),
                    vehicle_bbox=tuple(int(v) for v in vbbox),
                    track_id=tid,
                    details={
                        "stationary_frames": cfg["stationary_min_frames"],
                        "in_no_parking_zone": in_zone,
                    },
                    frame_idx=fidx,
                ))
        return violations

    # ─── Temporal Confirmation ────────────────────────────────

    def _temporal_confirm(self, vtype: str, key: str,
                          fidx: int, required: int) -> bool:
        """
        Require a violation to be detected in `required` frames
        before confirming. Prevents single-frame false positives.
        """
        # Don't fire the same violation key twice
        full_key = f"{vtype}_{key}"
        if full_key in self._fired:
            return False

        buf = self._tbuf.get(vtype, {})
        if key not in buf:
            buf[key] = {"first": fidx, "count": 1}
            self._tbuf[vtype] = buf
            return False

        entry = buf[key]
        # If too many frames have passed since first detection, reset
        if fidx - entry["first"] > required * 3:
            entry["first"] = fidx
            entry["count"] = 1
            return False

        entry["count"] += 1
        if entry["count"] >= required:
            del buf[key]
            self._fired.add(full_key)
            return True

        return False
