#!/usr/bin/env python3
"""
AI Traffic Violation Detection — Demo & Processing Script
Flipkart Gridlock 2.0

Modes:
  python demo.py --demo          # Simulated demo (no GPU needed)
  python demo.py --video FILE    # Process a real traffic video
  python demo.py --demo --video FILE  # Process real video in demo-friendly mode
"""
import argparse
import sys
import time
import os
from pathlib import Path
from collections import Counter
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
from tqdm import tqdm
import logging

from src.config import EVIDENCE, EVIDENCE_DIR
from src.preprocessor import ImagePreprocessor
from src.database import ViolationDB
from src.utils import generate_violation_id, get_timestamp, CircularBuffer
from src.analytics import ViolationHeatmapGenerator, CongestionAnalyzer, RiskScoreCalculator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ─── Synthetic Demo Video Generator ──────────────────────────

def create_demo_video(output_path: str, duration_sec: int = 15, fps: int = 30):
    """
    Render a synthetic traffic scene. Used as visual background when
    no real video is available.
    """
    logger.info(f"Rendering demo scene ({duration_sec}s at {fps}fps)...")
    W, H = 1280, 720
    total = duration_sec * fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    for fi in range(total):
        frame = np.zeros((H, W, 3), dtype=np.uint8)

        # Sky gradient
        for y in range(300):
            b = int(235 - y * 0.3)
            g = int(206 - y * 0.2)
            r = int(135 + y * 0.1)
            frame[y, :] = (max(0, min(b, 255)),
                           max(0, min(g, 255)),
                           max(0, min(r, 255)))

        # Road surface
        cv2.rectangle(frame, (0, 300), (W, H), (70, 70, 70), -1)
        # Centre divider
        cv2.rectangle(frame, (0, 498), (W, 502), (200, 200, 200), -1)
        # Dashed lane markings
        for x in range(0, W, 120):
            cv2.rectangle(frame, (x, 398), (x + 60, 402), (255, 255, 255), -1)

        # Stop line (yellow)
        stop_y = int(H * 0.65)
        cv2.rectangle(frame, (180, stop_y - 2), (W - 180, stop_y + 2),
                      (0, 230, 255), -1)

        # Traffic light housing
        tl_phase = "red" if fi < total * 0.55 else "green"
        cv2.rectangle(frame, (575, 80), (705, 270), (40, 40, 40), -1)
        cv2.rectangle(frame, (575, 80), (705, 270), (120, 120, 120), 2)
        red_c = (0, 0, 255) if tl_phase == "red" else (0, 0, 70)
        cv2.circle(frame, (640, 125), 28, red_c, -1)
        cv2.circle(frame, (640, 178), 28, (0, 120, 120), -1)
        green_c = (0, 255, 0) if tl_phase == "green" else (0, 70, 0)
        cv2.circle(frame, (640, 231), 28, green_c, -1)

        # Animated vehicles (drawn as realistic-ish shapes)
        # Car crossing stop line during red
        car_x = 250 + int(fi * 2.2)
        if car_x < W + 200:
            _draw_car(frame, car_x, 440, (180, 60, 30), "KA01AB1234")

        # Motorcycle with helmetless rider
        bike_x = 120 + int(fi * 1.6)
        if bike_x < W + 100:
            _draw_motorcycle(frame, bike_x, 530, helmet=False)

        # Triple riding motorcycle
        if fi > 50:
            tx = 780 + int((fi - 50) * 1.4)
            if tx < W + 120:
                _draw_motorcycle(frame, tx, 550, helmet=False, riders=3)

        # Stationary (parked) vehicle
        _draw_car(frame, 940, 580, (50, 120, 180), "PARKED")

        # Timestamp
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"CAM_001 | {ts} | Frame {fi}/{total}",
                    (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (180, 180, 180), 1)

        writer.write(frame)

    writer.release()
    logger.info(f"Demo video saved to {output_path}")


def _draw_car(frame, x, y, color, plate_text):
    """Draw a stylised car shape."""
    cv2.rectangle(frame, (x, y), (x + 150, y + 85), color, -1)
    cv2.rectangle(frame, (x + 5, y + 2), (x + 65, y + 35), (180, 160, 100), -1)
    cv2.rectangle(frame, (x + 85, y + 2), (x + 145, y + 35), (180, 160, 100), -1)
    cv2.circle(frame, (x + 25, y + 82), 12, (30, 30, 30), -1)
    cv2.circle(frame, (x + 125, y + 82), 12, (30, 30, 30), -1)
    if plate_text:
        cv2.rectangle(frame, (x + 30, y + 62), (x + 120, y + 78),
                      (240, 240, 240), -1)
        cv2.putText(frame, plate_text, (x + 33, y + 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)


def _draw_motorcycle(frame, x, y, helmet=True, riders=1):
    """Draw a stylised motorcycle with riders."""
    cv2.rectangle(frame, (x, y), (x + 70, y + 45), (40, 40, 140), -1)
    cv2.circle(frame, (x + 12, y + 42), 10, (25, 25, 25), -1)
    cv2.circle(frame, (x + 58, y + 42), 10, (25, 25, 25), -1)

    for ri in range(riders):
        rx = x + 10 + ri * 22
        # Body
        cv2.rectangle(frame, (rx, y - 50), (rx + 20, y), (80 + ri * 30, 130, 170), -1)
        # Head
        head_color = (120, 120, 120) if helmet else (160, 190, 220)
        cv2.circle(frame, (rx + 10, y - 60), 11, head_color, -1)
        if not helmet:
            cv2.circle(frame, (rx + 10, y - 66), 6, (40, 30, 20), -1)


# ─── Simulated Demo Processing ───────────────────────────────

def run_simulated_demo(output_dir: Path, save_video: bool = True):
    """
    Run a complete demo using simulated violations overlaid on
    a synthetic traffic scene. This showcases all system capabilities
    without needing a real video or GPU.
    """
    from src.violation_engine import Violation
    from src.evidence_generator import EvidenceGenerator

    evidence_dir = output_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence = EvidenceGenerator(str(evidence_dir))
    db = ViolationDB()
    heatmap = ViolationHeatmapGenerator()
    congestion = CongestionAnalyzer()

    # Generate demo video
    video_path = str(output_dir / "demo_traffic.mp4")
    create_demo_video(video_path)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if save_video:
        out_path = str(output_dir / "annotated_output.mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (fw, fh))

    violations_all = []
    frame_idx = 0

    # Pre-defined simulated violations at specific frames
    scheduled = _get_scheduled_violations(total_frames, fh, fw)

    pbar = tqdm(total=total_frames, desc="Processing", unit="frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        violations_now = []

        # Check scheduled violations for this frame
        for sv_item in scheduled:
            if sv_item["trigger_frame"] == frame_idx:
                v = Violation(
                    violation_id=generate_violation_id(),
                    violation_type=sv_item["type"],
                    timestamp=get_timestamp(),
                    confidence=sv_item["confidence"],
                    vehicle_type=sv_item["vehicle_type"],
                    vehicle_bbox=sv_item["bbox"],
                    violation_bbox=sv_item.get("viol_bbox"),
                    plate_number=sv_item.get("plate"),
                    plate_confidence=sv_item.get("plate_conf", 0.85),
                    details=sv_item.get("details", {}),
                    frame_idx=frame_idx,
                )
                violations_now.append(v)

                # Record on heatmap
                cx = (sv_item["bbox"][0] + sv_item["bbox"][2]) // 2
                cy = (sv_item["bbox"][1] + sv_item["bbox"][3]) // 2
                heatmap.record_violation(cx, cy, sv_item["type"])

        # Save violations
        for v in violations_now:
            evidence.add_frame_to_buffer(frame, frame_idx / fps)
            v = evidence.generate_evidence(frame, v)
            violations_all.append(v)
            db.save_violation(v)
            logger.info(
                f"VIOLATION DETECTED: {v.violation_type} | "
                f"{v.vehicle_type} | Plate: {v.plate_number or 'N/A'} | "
                f"Conf: {v.confidence:.0%}"
            )

        if not violations_now:
            evidence.add_frame_to_buffer(frame, frame_idx / fps)

        # Annotate
        annotated = evidence.annotate_frame(frame, violations_now)

        # HUD overlay
        cv2.putText(annotated, f"Frame: {frame_idx}/{total_frames}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(annotated, f"Total Violations: {len(violations_all)}",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        if writer:
            writer.write(annotated)

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    if writer:
        writer.release()

    # Save heatmap
    heatmap_path = str(output_dir / "violation_heatmap.jpg")
    bg_frame = np.zeros((fh, fw, 3), dtype=np.uint8)
    bg_frame[:] = (50, 50, 50)
    heatmap.save_heatmap(heatmap_path, bg_frame)

    # Print summary
    _print_summary(violations_all, frame_idx, output_dir)
    return violations_all


def _get_scheduled_violations(total_frames, fh, fw):
    """Pre-schedule violations at specific frames for the demo."""
    stop_y = int(fh * 0.65)
    return [
        {
            "trigger_frame": int(total_frames * 0.15),
            "type": "helmet_non_compliance",
            "confidence": 0.87,
            "vehicle_type": "motorcycle",
            "bbox": (200, 475, 290, 580),
            "viol_bbox": (220, 470, 270, 500),
            "plate": "KA05MH3921",
            "plate_conf": 0.82,
            "details": {"helmet_worn": False, "method": "head_region_analysis"},
        },
        {
            "trigger_frame": int(total_frames * 0.25),
            "type": "red_light_violation",
            "confidence": 0.93,
            "vehicle_type": "car",
            "bbox": (500, 420, 660, 520),
            "plate": "KA01AB1234",
            "plate_conf": 0.91,
            "details": {"stop_line_y": stop_y, "signal_state": "red"},
        },
        {
            "trigger_frame": int(total_frames * 0.35),
            "type": "triple_riding",
            "confidence": 0.89,
            "vehicle_type": "motorcycle",
            "bbox": (650, 490, 760, 595),
            "plate": "KA03EF5678",
            "plate_conf": 0.77,
            "details": {"rider_count": 3, "max_allowed": 2},
        },
        {
            "trigger_frame": int(total_frames * 0.42),
            "type": "stop_line_violation",
            "confidence": 0.85,
            "vehicle_type": "car",
            "bbox": (350, 440, 510, 530),
            "plate": "KA02CD9012",
            "plate_conf": 0.88,
            "details": {"cross_ratio": 0.45},
        },
        {
            "trigger_frame": int(total_frames * 0.50),
            "type": "seatbelt_non_compliance",
            "confidence": 0.78,
            "vehicle_type": "car",
            "bbox": (380, 410, 540, 500),
            "plate": "KA04GH3456",
            "plate_conf": 0.80,
            "details": {"seatbelt_worn": False},
        },
        {
            "trigger_frame": int(total_frames * 0.62),
            "type": "wrong_side_driving",
            "confidence": 0.82,
            "vehicle_type": "motorcycle",
            "bbox": (700, 350, 780, 430),
            "plate": "KA09JK7890",
            "plate_conf": 0.73,
            "details": {"detected_angle": 195.3, "expected_angle": 0,
                        "deviation": 164.7},
        },
        {
            "trigger_frame": int(total_frames * 0.75),
            "type": "illegal_parking",
            "confidence": 0.76,
            "vehicle_type": "car",
            "bbox": (940, 580, 1100, 660),
            "plate": "KA51MN2345",
            "plate_conf": 0.85,
            "details": {"stationary_frames": 90, "in_no_parking_zone": True},
        },
        {
            "trigger_frame": int(total_frames * 0.85),
            "type": "helmet_non_compliance",
            "confidence": 0.91,
            "vehicle_type": "motorcycle",
            "bbox": (150, 510, 230, 590),
            "viol_bbox": (160, 480, 210, 510),
            "plate": "KA12PQ6789",
            "plate_conf": 0.79,
            "details": {"helmet_worn": False, "method": "head_region_analysis"},
        },
    ]


# ─── Real Video Processing ───────────────────────────────────

def run_real_video(video_path: str, output_dir: Path, save_video: bool = True,
                   display: bool = False, device: str = "", conf: float = 0.4,
                   skip_ocr: bool = False):
    """Process a real traffic video with the full AI pipeline."""
    from src.detector import TrafficDetector
    from src.tracker import VehicleTracker
    from src.violation_engine import ViolationEngine
    from src.ocr_engine import LicensePlateRecognizer
    from src.evidence_generator import EvidenceGenerator

    evidence_dir = output_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading AI components...")
    t0 = time.time()
    detector = TrafficDetector(conf_thresh=conf, device=device)
    tracker = VehicleTracker()
    engine = ViolationEngine()
    ocr = None if skip_ocr else LicensePlateRecognizer()
    evidence = EvidenceGenerator(str(evidence_dir))
    preprocessor = ImagePreprocessor()
    db = ViolationDB()
    heatmap = ViolationHeatmapGenerator()
    congestion = CongestionAnalyzer()
    risk = RiskScoreCalculator()
    logger.info(f"Components ready in {time.time() - t0:.1f}s")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open: {video_path}")
        return []

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Video: {fw}x{fh} @ {fps}fps, {total_frames} frames")

    writer = None
    if save_video:
        out_path = str(output_dir / "annotated_output.mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (fw, fh))

    violations_all = []
    frame_idx = 0
    pbar = tqdm(total=total_frames, desc="Processing", unit="frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed, _ = preprocessor.preprocess_frame(frame)
        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame)

        violations = engine.detect_violations(
            frame, detections, tracks, frame_idx, tracker=tracker
        )

        # Congestion analysis
        cong = congestion.analyse_frame(detections, tracks, tracker,
                                        frame.shape)

        # OCR + evidence for violations
        for v in violations:
            if v.vehicle_bbox and ocr:
                try:
                    pimg, pbbox = ocr.detect_plate_region(frame, v.vehicle_bbox)
                    if pimg is not None:
                        text, cf = ocr.recognize(pimg)
                        v.plate_number = text
                        v.plate_confidence = cf
                        v.plate_bbox = pbbox
                except Exception:
                    pass

            evidence.add_frame_to_buffer(frame, frame_idx / fps)
            v = evidence.generate_evidence(frame, v, detections)
            violations_all.append(v)
            db.save_violation(v)

            # Heatmap + risk
            cx = (v.vehicle_bbox[0] + v.vehicle_bbox[2]) // 2
            cy = (v.vehicle_bbox[1] + v.vehicle_bbox[3]) // 2
            heatmap.record_violation(cx, cy, v.violation_type)
            if v.track_id is not None:
                risk.record_violation(v.track_id, v.violation_type)

            logger.info(
                f"VIOLATION: {v.violation_type} | {v.vehicle_type} | "
                f"Plate: {v.plate_number or 'N/A'} | Conf: {v.confidence:.0%}"
            )

        if not violations:
            evidence.add_frame_to_buffer(frame, frame_idx / fps)

        # Annotate
        annotated = evidence.annotate_frame(frame, violations, detections)
        _draw_hud(annotated, frame_idx, total_frames, len(violations_all), cong)

        if display:
            cv2.imshow("AI Traffic Violation Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if writer:
            writer.write(annotated)

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    if writer:
        writer.release()
    try:
        cv2.destroyAllWindows()
    except:
        pass

    # Save heatmap
    heatmap.save_heatmap(str(output_dir / "violation_heatmap.jpg"))

    _print_summary(violations_all, frame_idx, output_dir)
    return violations_all


def _draw_hud(frame, fidx, total, v_count, cong):
    """Overlay HUD info on the annotated frame."""
    cv2.putText(frame, f"Frame: {fidx}/{total}",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, f"Violations: {v_count}",
                (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    level = cong.get("level", "N/A")
    score = cong.get("score", 0)
    color = (0, 255, 0) if score < 30 else (0, 200, 255) if score < 60 else (0, 0, 255)
    cv2.putText(frame, f"Congestion: {level} ({score}%)",
                (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def _print_summary(violations, frames, output_dir):
    """Print final processing summary."""
    logger.info("=" * 60)
    logger.info("  PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Frames processed : {frames}")
    logger.info(f"  Violations found : {len(violations)}")

    if violations:
        counts = Counter(v.violation_type for v in violations)
        logger.info("  Breakdown:")
        for vt, c in counts.most_common():
            logger.info(f"    {vt.replace('_', ' ').title()}: {c}")

        plates = [v.plate_number for v in violations if v.plate_number]
        if plates:
            logger.info(f"  Plates recognised: {len(set(plates))}")
            for p in sorted(set(plates)):
                logger.info(f"    {p}")

    logger.info(f"  Evidence folder  : {output_dir / 'evidence'}")
    logger.info(f"  Heatmap          : {output_dir / 'violation_heatmap.jpg'}")
    logger.info("=" * 60)


# ─── CLI Entry Point ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AI Traffic Violation Detection — Flipkart Gridlock 2.0",
    )
    parser.add_argument("--video", type=str, help="Path to input video")
    parser.add_argument("--output", default="./output", help="Output directory")
    parser.add_argument("--demo", action="store_true",
                        help="Run simulated demo (no GPU required)")
    parser.add_argument("--save-video", action="store_true",
                        help="Save annotated output video")
    parser.add_argument("--display", action="store_true",
                        help="Show real-time display window")
    parser.add_argument("--no-ocr", action="store_true",
                        help="Skip OCR processing")
    parser.add_argument("--device", default="",
                        help="Inference device: '' (auto), '0' (GPU), 'cpu'")
    parser.add_argument("--conf", type=float, default=0.40,
                        help="Detection confidence threshold")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Banner
    logger.info("=" * 60)
    logger.info("  AI Traffic Violation Detection System")
    logger.info("  Flipkart Gridlock 2.0")
    logger.info("=" * 60)

    if args.demo and not args.video:
        # Pure simulation demo
        run_simulated_demo(output_dir, save_video=args.save_video)
    elif args.video:
        if not Path(args.video).exists():
            logger.error(f"Video file not found: {args.video}")
            return 1
        run_real_video(
            args.video, output_dir,
            save_video=args.save_video,
            display=args.display,
            device=args.device,
            conf=args.conf,
            skip_ocr=args.no_ocr,
        )
    else:
        logger.error("Specify --demo or --video <path>")
        return 1

    logger.info("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
