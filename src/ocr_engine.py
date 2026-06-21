"""
License Plate Recognition — EasyOCR wrapper for Indian plates

Indian number plates follow the format KA01AB1234 (state-district-
series-number) but are notoriously hard to read from CCTV footage
due to low resolution, dirt, and non-standard fonts. We preprocess
heavily and apply character-level corrections (O→0, I→1 etc).
"""
import cv2
import numpy as np
import re
from typing import Tuple, Optional, List
import logging

from src.config import OCR
from src.preprocessor import ImagePreprocessor

logger = logging.getLogger(__name__)

# Lazy import
_easyocr_reader = None


def _get_reader():
    """Lazy-init EasyOCR reader (heavy, only load once)."""
    global _easyocr_reader
    if _easyocr_reader is not None:
        return _easyocr_reader
    try:
        import easyocr
        logger.info("Initializing EasyOCR...")
        _easyocr_reader = easyocr.Reader(
            OCR["languages"],
            gpu=OCR["use_gpu"],
            verbose=False,
        )
        logger.info("EasyOCR ready")
        return _easyocr_reader
    except Exception as e:
        logger.warning(f"EasyOCR unavailable: {e}")
        return None


class LicensePlateRecognizer:
    """Automatic Number Plate Recognition for Indian vehicles."""

    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        self.plate_regex = re.compile(OCR["plate_format_regex"])
        self.corrections = OCR["character_corrections"]
        logger.info("LicensePlateRecognizer ready")

    def detect_plate_region(
        self, frame: np.ndarray,
        vehicle_bbox: Tuple[int, int, int, int],
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        """
        Crop likely plate region from vehicle bounding box.
        Indian plates are typically in lower 30-50% of vehicle rear.
        """
        x1, y1, x2, y2 = map(int, vehicle_bbox)
        vh, vw = y2 - y1, x2 - x1
        h, w = frame.shape[:2]

        # Plate region: lower-center of vehicle
        py1 = y1 + int(vh * 0.55)
        py2 = min(h, y1 + int(vh * 0.90))
        px1 = max(0, x1 + int(vw * 0.15))
        px2 = min(w, x1 + int(vw * 0.85))

        if py2 <= py1 or px2 <= px1:
            return None, None

        plate_crop = frame[py1:py2, px1:px2]
        if plate_crop.size == 0:
            return None, None

        # Validate aspect ratio (Indian plates ≈ 2:1 to 5:1)
        ph, pw = plate_crop.shape[:2]
        ar = pw / ph if ph > 0 else 0
        if 1.5 < ar < 6.0:
            return plate_crop, (px1, py1, px2, py2)

        # Fallback: contour-based plate search
        return self._find_plate_contours(plate_crop, (px1, py1))

    def recognize(self, plate_image: np.ndarray) -> Tuple[str, float]:
        """
        Run OCR on plate image.
        Returns (plate_text, confidence).
        """
        if plate_image is None or plate_image.size == 0:
            return "", 0.0

        # Preprocess
        processed = self.preprocessor.preprocess_for_ocr(plate_image)

        # Ensure uint8 for EasyOCR
        if processed.dtype != np.uint8:
            processed = (processed * 255).astype(np.uint8) if processed.max() <= 1.0 else processed.astype(np.uint8)

        # Convert to 3-channel if grayscale
        if len(processed.shape) == 2:
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

        reader = _get_reader()
        if reader is not None:
            try:
                results = reader.readtext(
                    processed, detail=1, paragraph=False,
                    contrast_ths=0.1, adjust_contrast=0.5,
                    text_threshold=0.4, low_text=0.3,
                )
                if results:
                    texts = []
                    confs = []
                    for _, text, conf in results:
                        texts.append(text)
                        confs.append(conf)

                    raw = "".join(texts).upper().replace(" ", "").replace("-", "")
                    corrected = self._correct(raw)
                    avg_conf = float(np.mean(confs))

                    if self.plate_regex.match(corrected):
                        return corrected, avg_conf
                    return corrected, avg_conf * 0.7
            except Exception as e:
                logger.warning(f"OCR error: {e}")

        return "", 0.0

    def batch_recognize(self, images: List[np.ndarray]) -> List[Tuple[str, float]]:
        return [self.recognize(img) for img in images]

    # ─── Internal ─────────────────────────────────────────────

    def _correct(self, text: str) -> str:
        """Apply character substitution heuristics."""
        out = []
        for ch in text:
            if ch in self.corrections:
                out.append(self.corrections[ch])
            elif ch.isalnum():
                out.append(ch)
        return "".join(out)

    def _find_plate_contours(self, region, offset):
        """Find plate-like rectangle via contour analysis."""
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        best_plate, best_bbox, best_score = None, None, 0

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            ar = w / h if h > 0 else 0
            area = w * h
            if 1.5 < ar < 6.0 and area > 400:
                score = area * (1 - abs(ar - 3.5) / 3.5)
                if score > best_score:
                    best_score = score
                    best_plate = region[y:y + h, x:x + w]
                    best_bbox = (offset[0] + x, offset[1] + y,
                                 offset[0] + x + w, offset[1] + y + h)

        return best_plate, best_bbox
