"""
Advanced Analytics Module
Violation heatmaps, congestion scoring, risk zone identification,
and trend analysis — features that set this prototype apart.
"""
import cv2
import numpy as np
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import json
import logging
from pathlib import Path

from src.config import EVIDENCE

logger = logging.getLogger(__name__)


class ViolationHeatmapGenerator:
    """
    Build spatial heatmaps showing where violations concentrate.
    Useful for traffic police to identify high-risk intersections.
    """

    def __init__(self, frame_width: int = 1280, frame_height: int = 720):
        self.w = frame_width
        self.h = frame_height
        self.accumulator = np.zeros((frame_height, frame_width), dtype=np.float32)
        self.type_accumulators: Dict[str, np.ndarray] = {}

    def record_violation(self, cx: int, cy: int, violation_type: str,
                         radius: int = 40):
        """Add a Gaussian blob at the violation location."""
        cx = max(0, min(cx, self.w - 1))
        cy = max(0, min(cy, self.h - 1))

        # Create Gaussian kernel
        size = radius * 2 + 1
        kernel = cv2.getGaussianKernel(size, radius / 3)
        blob = kernel @ kernel.T

        # Place on accumulator
        x1 = max(0, cx - radius)
        y1 = max(0, cy - radius)
        x2 = min(self.w, cx + radius + 1)
        y2 = min(self.h, cy + radius + 1)

        bx1 = radius - (cx - x1)
        by1 = radius - (cy - y1)
        bx2 = bx1 + (x2 - x1)
        by2 = by1 + (y2 - y1)

        self.accumulator[y1:y2, x1:x2] += blob[by1:by2, bx1:bx2]

        # Per-type accumulator
        if violation_type not in self.type_accumulators:
            self.type_accumulators[violation_type] = np.zeros(
                (self.h, self.w), dtype=np.float32
            )
        self.type_accumulators[violation_type][y1:y2, x1:x2] += blob[by1:by2, bx1:bx2]

    def render_heatmap(self, background: Optional[np.ndarray] = None,
                       alpha: float = 0.55) -> np.ndarray:
        """Render the accumulated heatmap as a colour overlay."""
        if background is None:
            background = np.zeros((self.h, self.w, 3), dtype=np.uint8)

        if self.accumulator.max() < 0.001:
            return background.copy()

        norm = cv2.normalize(self.accumulator, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)
        return cv2.addWeighted(background, 1 - alpha, heatmap, alpha, 0)

    def save_heatmap(self, output_path: str,
                     background: Optional[np.ndarray] = None):
        """Save heatmap image to disk."""
        img = self.render_heatmap(background)
        cv2.imwrite(output_path, img)
        logger.info(f"Heatmap saved: {output_path}")

    def get_hotspots(self, top_n: int = 5) -> List[Dict]:
        """Find the top-N violation hotspot regions."""
        if self.accumulator.max() < 0.001:
            return []

        blur = cv2.GaussianBlur(self.accumulator, (51, 51), 0)
        spots = []
        temp = blur.copy()

        for _ in range(top_n):
            _, max_val, _, max_loc = cv2.minMaxLoc(temp)
            if max_val < 0.01:
                break
            spots.append({
                "x": int(max_loc[0]),
                "y": int(max_loc[1]),
                "intensity": round(float(max_val), 4),
            })
            # Suppress this region
            cv2.circle(temp, max_loc, 60, 0, -1)

        return spots


class CongestionAnalyzer:
    """
    Measure traffic density and flow characteristics per frame.
    Outputs a congestion score (0-100) based on vehicle count,
    speed distribution, and spatial density.
    """

    def __init__(self, grid_cells: Tuple[int, int] = (4, 3)):
        self.grid_cols, self.grid_rows = grid_cells
        self.history: List[Dict] = []
        self.max_history = 300  # ~10 sec at 30fps

    def analyse_frame(self, detections, tracks, tracker,
                      frame_shape: Tuple[int, int]) -> Dict:
        """
        Compute congestion metrics for a single frame.
        Returns dict with score, density, avg_speed, etc.
        """
        h, w = frame_shape[:2]
        vehicle_ids = [2, 3, 5, 7]  # car, motorcycle, bus, truck

        # Count vehicles
        if len(detections) == 0:
            result = {
                "score": 0, "vehicle_count": 0, "density": 0.0,
                "avg_speed": 0.0, "grid_density": [], "level": "free_flow",
            }
            self._push(result)
            return result

        veh_mask = np.isin(detections.class_id, vehicle_ids)
        veh_count = int(np.sum(veh_mask))

        # Spatial density (vehicles per grid cell)
        cell_w = w / self.grid_cols
        cell_h = h / self.grid_rows
        grid = np.zeros((self.grid_rows, self.grid_cols), dtype=int)

        vehicles = detections[veh_mask]
        for bbox in vehicles.xyxy:
            cx = int((bbox[0] + bbox[2]) / 2)
            cy = int((bbox[1] + bbox[3]) / 2)
            gc = min(int(cx / cell_w), self.grid_cols - 1)
            gr = min(int(cy / cell_h), self.grid_rows - 1)
            grid[gr, gc] += 1

        density = float(np.mean(grid[grid > 0])) if grid.max() > 0 else 0.0

        # Average speed from tracker
        speeds = []
        if (tracks is not None and len(tracks) > 0
                and tracks.tracker_id is not None):
            for tid in tracks.tracker_id:
                spd = tracker.get_speed_px_per_frame(int(tid), window=10)
                if spd > 0:
                    speeds.append(spd)

        avg_speed = float(np.mean(speeds)) if speeds else 0.0

        # Congestion score (0-100)
        # High vehicle count + low speed + high density = congested
        count_factor = min(veh_count / 20, 1.0) * 40
        speed_factor = max(0, 1 - avg_speed / 15) * 30 if speeds else 15
        density_factor = min(density / 5, 1.0) * 30
        score = min(100, int(count_factor + speed_factor + density_factor))

        level = "free_flow"
        if score > 70:
            level = "heavy"
        elif score > 45:
            level = "moderate"
        elif score > 20:
            level = "light"

        result = {
            "score": score,
            "vehicle_count": veh_count,
            "density": round(density, 2),
            "avg_speed": round(avg_speed, 2),
            "grid_density": grid.tolist(),
            "level": level,
        }
        self._push(result)
        return result

    def get_trend(self, window: int = 30) -> Dict:
        """Return congestion trend over recent frames."""
        if len(self.history) < 2:
            return {"trend": "stable", "change": 0}

        recent = self.history[-window:]
        scores = [r["score"] for r in recent]
        if len(scores) < 2:
            return {"trend": "stable", "change": 0}

        first_half = np.mean(scores[:len(scores) // 2])
        second_half = np.mean(scores[len(scores) // 2:])
        change = second_half - first_half

        if change > 5:
            trend = "increasing"
        elif change < -5:
            trend = "decreasing"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "change": round(float(change), 1),
            "current_avg": round(float(np.mean(scores[-10:])), 1),
        }

    def _push(self, result):
        self.history.append(result)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]


class RiskScoreCalculator:
    """
    Assign a compound risk score to each tracked vehicle based on
    multiple factors: violation history, speed, trajectory erraticism.
    """

    def __init__(self):
        self.vehicle_violations: Dict[int, List[str]] = defaultdict(list)
        self.vehicle_scores: Dict[int, float] = {}

    def record_violation(self, track_id: int, violation_type: str):
        """Log a violation against a tracked vehicle."""
        self.vehicle_violations[track_id].append(violation_type)

    def compute_score(self, track_id: int, tracker,
                      speed_limit_px: float = 12.0) -> float:
        """
        Compute risk score (0-100) for a vehicle.
        Factors: violation count, speed, trajectory smoothness.
        """
        score = 0.0

        # Factor 1: violation history (each violation adds risk)
        violation_weights = {
            "helmet_non_compliance": 15,
            "triple_riding": 20,
            "red_light_violation": 25,
            "stop_line_violation": 10,
            "wrong_side_driving": 30,
            "seatbelt_non_compliance": 12,
            "illegal_parking": 8,
        }
        for vtype in self.vehicle_violations.get(track_id, []):
            score += violation_weights.get(vtype, 10)

        # Factor 2: speed (over speed limit = risky)
        speed = tracker.get_speed_px_per_frame(track_id)
        if speed > speed_limit_px:
            score += min((speed - speed_limit_px) * 5, 25)

        # Factor 3: trajectory erraticism (high variance = risky)
        traj = tracker.get_trajectory(track_id)
        if len(traj) > 5:
            pts = np.array(traj[-20:])
            if len(pts) > 2:
                # Compute curvature variance
                diffs = np.diff(pts, axis=0)
                angles = np.arctan2(diffs[:, 1], diffs[:, 0])
                if len(angles) > 1:
                    angle_var = float(np.var(np.diff(angles)))
                    score += min(angle_var * 50, 20)

        score = min(100, score)
        self.vehicle_scores[track_id] = score
        return score

    def get_high_risk_vehicles(self, threshold: float = 40) -> List[Dict]:
        """Return all vehicles above the risk threshold."""
        return [
            {"track_id": tid, "risk_score": round(s, 1),
             "violations": self.vehicle_violations.get(tid, [])}
            for tid, s in sorted(
                self.vehicle_scores.items(), key=lambda x: x[1], reverse=True
            )
            if s >= threshold
        ]
