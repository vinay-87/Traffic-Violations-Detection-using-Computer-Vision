"""
Multi-Object Tracking — DeepSORT integration

Keeps track of individual vehicles across frames and stores their
centroid history so we can analyse movement patterns (direction,
speed, stationarity) for violation detection.
"""
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging

from deep_sort_realtime.deepsort_tracker import DeepSort
import supervision as sv

from src.config import TRACKING
from src.utils import calculate_centroid, calculate_iou

logger = logging.getLogger(__name__)


class VehicleTracker:
    """DeepSORT-based multi-object tracker with trajectory history."""

    def __init__(self):
        self.tracker = DeepSort(
            max_age=TRACKING["max_age"],
            n_init=TRACKING["n_init"],
            max_iou_distance=TRACKING["max_iou_distance"],
            embedder=TRACKING["embedder"],
            half=TRACKING["half"],
            bgr=TRACKING["bgr"],
        )
        self.max_traj = TRACKING["max_trajectory_len"]

        # Per-track history
        self.trajectories: Dict[int, List[Tuple[int, int]]] = {}
        self.track_classes: Dict[int, int] = {}

        logger.info("VehicleTracker ready")

    def update(self, detections: sv.Detections,
               frame: np.ndarray) -> sv.Detections:
        """
        Feed new detections into DeepSORT and return tracked detections
        with `tracker_id` populated.
        """
        if len(detections) == 0:
            self.tracker.update_tracks([], frame=frame)
            return sv.Detections.empty()

        # Build the detection list that deep-sort-realtime expects:
        # list of ( [x1, y1, w, h], confidence, class_id )
        raw_dets = []
        for bbox, conf, cid in zip(
            detections.xyxy, detections.confidence, detections.class_id
        ):
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            raw_dets.append(([x1, y1, w, h], float(conf), int(cid)))

        tracks = self.tracker.update_tracks(raw_dets, frame=frame)

        # Collect confirmed tracks
        bboxes, confs, cids, tids = [], [], [], []
        for t in tracks:
            if not t.is_confirmed():
                continue
            ltrb = t.to_ltrb()  # [x1, y1, x2, y2]
            tid = t.track_id

            # Match to nearest original detection for class & conf
            best_cid, best_conf = self._match_detection(ltrb, detections)

            # Update trajectory
            cx, cy = calculate_centroid(tuple(ltrb))
            hist = self.trajectories.setdefault(tid, [])
            hist.append((cx, cy))
            if len(hist) > self.max_traj:
                self.trajectories[tid] = hist[-self.max_traj:]
            self.track_classes[tid] = best_cid

            bboxes.append(ltrb)
            confs.append(best_conf)
            cids.append(best_cid)
            tids.append(tid)

        if not bboxes:
            return sv.Detections.empty()

        return sv.Detections(
            xyxy=np.array(bboxes, dtype=np.float32),
            confidence=np.array(confs, dtype=np.float32),
            class_id=np.array(cids, dtype=int),
            tracker_id=np.array(tids, dtype=int),
        )

    # ─── Trajectory Analysis ──────────────────────────────────

    def get_trajectory(self, track_id: int) -> List[Tuple[int, int]]:
        """Return centroid history for a track."""
        return self.trajectories.get(track_id, [])

    def get_direction_angle(self, track_id: int,
                            min_len: int = 5) -> Optional[float]:
        """
        Compute overall movement direction in degrees.
        0 = rightward, 90 = downward, 180 = leftward.
        Returns None if insufficient history.
        """
        traj = self.get_trajectory(track_id)
        if len(traj) < min_len:
            return None
        p1 = traj[0]
        p2 = traj[-1]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angle = np.degrees(np.arctan2(dy, dx))
        return float(angle if angle >= 0 else angle + 360)

    def is_stationary(self, track_id: int,
                      threshold_px: float = 8,
                      min_frames: int = 10) -> bool:
        """Check if track has barely moved over recent frames."""
        traj = self.get_trajectory(track_id)
        if len(traj) < min_frames:
            return False
        recent = np.array(traj[-min_frames:])
        var = np.var(recent, axis=0)
        return float(max(var)) < threshold_px ** 2

    def get_speed_px_per_frame(self, track_id: int,
                               window: int = 10) -> float:
        """Average pixel displacement per frame over last `window` frames."""
        traj = self.get_trajectory(track_id)
        if len(traj) < 2:
            return 0.0
        recent = traj[-window:]
        total = sum(
            np.sqrt((recent[i][0] - recent[i - 1][0]) ** 2 +
                     (recent[i][1] - recent[i - 1][1]) ** 2)
            for i in range(1, len(recent))
        )
        return total / (len(recent) - 1)

    def reset(self):
        """Reset all tracker state."""
        self.trajectories.clear()
        self.track_classes.clear()
        self.tracker = DeepSort(
            max_age=TRACKING["max_age"],
            n_init=TRACKING["n_init"],
            max_iou_distance=TRACKING["max_iou_distance"],
            embedder=TRACKING["embedder"],
            half=TRACKING["half"],
            bgr=TRACKING["bgr"],
        )

    # ─── Internal ─────────────────────────────────────────────

    def _match_detection(self, track_bbox, detections: sv.Detections):
        """Find the original detection closest to a track bbox."""
        best_iou = 0
        best_cid = 0
        best_conf = 0.5
        for bbox, conf, cid in zip(
            detections.xyxy, detections.confidence, detections.class_id
        ):
            iou = calculate_iou(tuple(track_bbox), tuple(bbox))
            if iou > best_iou:
                best_iou = iou
                best_cid = int(cid)
                best_conf = float(conf)
        return best_cid, best_conf
