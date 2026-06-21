"""
YOLOv8 Object Detector Wrapper

Wraps Ultralytics YOLOv8 so the rest of the system doesn't need
to know about YOLO internals. The model downloads automatically
on first run (~6MB for the nano variant).
"""
import cv2
import numpy as np
from typing import List, Optional, Dict, Tuple
from pathlib import Path
import logging

from ultralytics import YOLO
import supervision as sv

from src.config import DETECTION, COCO_CLASSES, MOTORIZED_VEHICLE_IDS

logger = logging.getLogger(__name__)


class TrafficDetector:
    """YOLOv8-based traffic object detector."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_thresh: Optional[float] = None,
        iou_thresh: Optional[float] = None,
        device: Optional[str] = None,
    ):
        self.conf_thresh = conf_thresh or DETECTION["confidence_threshold"]
        self.iou_thresh = iou_thresh or DETECTION["iou_threshold"]
        self.device = device if device is not None else DETECTION["device"]
        self.input_size = DETECTION["input_size"]

        # Load model — auto-downloads if missing
        model_path = model_path or DETECTION["model_path"]
        logger.info(f"Loading YOLOv8 model from {model_path}")
        try:
            self.model = YOLO(model_path)
        except Exception:
            logger.warning("Custom model not found, downloading default yolov8n.pt")
            self.model = YOLO("yolov8n.pt")

        # Supervision annotators
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
        logger.info("TrafficDetector ready")

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """Run detection on a single frame, return supervision Detections."""
        results = self.model(
            frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            device=self.device if self.device else None,
            verbose=False,
        )[0]

        detections = sv.Detections.from_ultralytics(results)

        # Filter to classes of interest
        if len(detections) > 0:
            mask = np.isin(detections.class_id,
                           DETECTION["classes_of_interest"])
            detections = detections[mask]

        return detections

    def draw_detections(self, frame: np.ndarray,
                        detections: sv.Detections) -> np.ndarray:
        """Draw bounding boxes with labels on frame copy."""
        if len(detections) == 0:
            return frame.copy()

        labels = [
            f"{COCO_CLASSES.get(cid, f'cls{cid}')} {conf:.2f}"
            for cid, conf in zip(detections.class_id, detections.confidence)
        ]
        out = self.box_annotator.annotate(scene=frame.copy(),
                                          detections=detections)
        out = self.label_annotator.annotate(scene=out, detections=detections,
                                            labels=labels)
        return out

    # ─── Filtering Helpers ────────────────────────────────────

    def get_class_name(self, class_id: int) -> str:
        return COCO_CLASSES.get(class_id, f"class_{class_id}")

    def filter_by_class(self, detections: sv.Detections,
                        class_ids: List[int]) -> sv.Detections:
        """Return only detections matching given class IDs."""
        if len(detections) == 0:
            return detections
        mask = np.isin(detections.class_id, class_ids)
        return detections[mask]

    def get_vehicles(self, detections: sv.Detections) -> sv.Detections:
        return self.filter_by_class(detections, MOTORIZED_VEHICLE_IDS)

    def get_persons(self, detections: sv.Detections) -> sv.Detections:
        return self.filter_by_class(detections, [0])

    def get_motorcycles(self, detections: sv.Detections) -> sv.Detections:
        return self.filter_by_class(detections, [3])

    def get_traffic_lights(self, detections: sv.Detections) -> sv.Detections:
        return self.filter_by_class(detections, [9])
