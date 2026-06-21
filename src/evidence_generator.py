"""
Evidence Generation Module
Annotated snapshots, video clips, and JSON metadata for each violation.
"""
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from datetime import datetime
import json
import logging

import supervision as sv

from src.config import EVIDENCE
from src.violation_engine import Violation
from src.utils import CircularBuffer, add_text_with_background

logger = logging.getLogger(__name__)


class EvidenceGenerator:
    """Generate professional evidence packages for traffic violations."""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir or EVIDENCE["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.viol_color = EVIDENCE["violation_color"]
        self.ok_color = EVIDENCE["compliance_color"]
        self.warn_color = EVIDENCE["warning_color"]
        self.info_color = EVIDENCE["info_color"]
        self.font_scale = EVIDENCE["font_scale"]
        self.font_thick = EVIDENCE["font_thickness"]
        self.snap_quality = EVIDENCE["snapshot_quality"]
        self.clip_dur = EVIDENCE["clip_duration_sec"]

        # Circular buffer for video clips
        buf_size = EVIDENCE["buffer_size_frames"]
        self.frame_buf = CircularBuffer(buf_size)
        self.ts_buf = CircularBuffer(buf_size)

        logger.info(f"EvidenceGenerator → {self.output_dir}")

    # ─── Public API ───────────────────────────────────────────

    def add_frame_to_buffer(self, frame: np.ndarray, timestamp: float):
        """Push frame into ring buffer for clip generation."""
        self.frame_buf.add(frame.copy())
        self.ts_buf.add(timestamp)

    def generate_evidence(
        self,
        frame: np.ndarray,
        violation: Violation,
        detections: Optional[sv.Detections] = None,
    ) -> Violation:
        """Create snapshot + clip + JSON for a violation, return updated Violation."""
        ts_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        base = f"{violation.violation_type}_{ts_str}_{violation.violation_id[-6:]}"

        # 1. Annotated snapshot
        snap_path = self.output_dir / f"{base}.jpg"
        annotated = self.annotate_frame(frame, [violation], detections)
        cv2.imwrite(str(snap_path), annotated,
                    [cv2.IMWRITE_JPEG_QUALITY, self.snap_quality])
        violation.snapshot_path = str(snap_path)

        # 2. Video clip from buffer
        clip_path = self.output_dir / f"{base}.mp4"
        self._write_clip(clip_path)
        violation.video_clip_path = str(clip_path)

        # 3. JSON metadata
        meta_path = self.output_dir / f"{base}.json"
        self._write_meta(meta_path, violation)

        return violation

    # ─── Annotation ───────────────────────────────────────────

    def annotate_frame(
        self,
        frame: np.ndarray,
        violations: List[Violation],
        detections: Optional[sv.Detections] = None,
    ) -> np.ndarray:
        """Draw professional annotations on a frame copy."""
        out = frame.copy()
        h, w = out.shape[:2]

        # Draw all detections in green (compliant)
        if detections is not None and len(detections) > 0:
            for bbox, cid, conf in zip(
                detections.xyxy, detections.class_id, detections.confidence
            ):
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(out, (x1, y1), (x2, y2), self.ok_color, 1)

        # Draw violations in red with labels
        for v in violations:
            if v.vehicle_bbox:
                bx1, by1, bx2, by2 = map(int, v.vehicle_bbox)
                cv2.rectangle(out, (bx1, by1), (bx2, by2), self.viol_color, 3)

                label = v.violation_type.replace("_", " ").upper()
                conf_str = f"{v.confidence:.0%}"
                tag = f"{label} | {conf_str}"

                # Label background
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX,
                                              self.font_scale, self.font_thick)
                cv2.rectangle(out, (bx1, by1 - th - 10), (bx1 + tw + 8, by1),
                              self.viol_color, -1)
                cv2.putText(out, tag, (bx1 + 4, by1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, self.font_scale,
                            (255, 255, 255), self.font_thick)

            # Violation-specific sub-region
            if v.violation_bbox:
                vx1, vy1, vx2, vy2 = map(int, v.violation_bbox)
                cv2.rectangle(out, (vx1, vy1), (vx2, vy2), self.warn_color, 2)

            # Plate overlay
            if v.plate_number:
                plate_label = f"PLATE: {v.plate_number}"
                bx = int(v.vehicle_bbox[0]) if v.vehicle_bbox else 10
                by = int(v.vehicle_bbox[3]) + 18 if v.vehicle_bbox else 30
                add_text_with_background(out, plate_label, (bx, by),
                                         font_scale=0.5, color=(255, 255, 0),
                                         bg_color=(80, 0, 80))

        # Header bar
        cv2.rectangle(out, (0, 0), (w, 36), (20, 20, 20), -1)
        ts_now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(out,
                    f"AI Traffic Violation Detection | {ts_now} UTC",
                    (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 220, 0), 1)

        if violations:
            alert = f"ALERT: {len(violations)} VIOLATION(S)"
            (aw, _), _ = cv2.getTextSize(alert, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(out, (w - aw - 16, 0), (w, 36), self.viol_color, -1)
            cv2.putText(out, alert, (w - aw - 8, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Footer watermark
        wm = "Flipkart Gridlock 2.0 | AI Traffic Enforcement"
        cv2.putText(out, wm, (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (180, 180, 180), 1)

        return out

    # ─── Internal ─────────────────────────────────────────────

    def _write_clip(self, path: Path):
        """Write buffered frames to an MP4 clip."""
        frames = self.frame_buf.get_all()
        if not frames:
            return
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, 30, (w, h))
        for f in frames:
            writer.write(f)
        writer.release()

    def _write_meta(self, path: Path, v: Violation):
        """Save violation metadata JSON."""
        meta = {
            "violation_id": v.violation_id,
            "type": v.violation_type,
            "timestamp": v.timestamp,
            "confidence": v.confidence,
            "camera_id": v.camera_id,
            "location": v.location or {"lat": 12.9716, "lng": 77.5946,
                                        "name": "Bengaluru"},
            "vehicle": {
                "type": v.vehicle_type,
                "plate_number": v.plate_number,
                "plate_confidence": v.plate_confidence,
            },
            "details": v.details,
            "evidence": {
                "snapshot": v.snapshot_path,
                "video_clip": v.video_clip_path,
            },
            "processing": {
                "model_version": v.model_version,
                "inference_time_ms": v.inference_time_ms,
            },
        }
        with open(path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

    def generate_daily_report(self, date: Optional[str] = None) -> Dict:
        """Compile daily statistics from evidence JSON files."""
        if date is None:
            date = datetime.utcnow().strftime("%Y-%m-%d")
        by_type: Dict[str, int] = {}
        total = 0
        for jf in self.output_dir.glob("*.json"):
            try:
                with open(jf) as f:
                    d = json.load(f)
                if d.get("timestamp", "").startswith(date):
                    vt = d["type"]
                    by_type[vt] = by_type.get(vt, 0) + 1
                    total += 1
            except Exception:
                continue
        return {"date": date, "total": total, "by_type": by_type}
