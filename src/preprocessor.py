"""
Image Preprocessing Module
Handles frame enhancement, normalization, and OCR-specific preprocessing.
"""
import cv2
import numpy as np
from typing import Tuple, Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Comprehensive image preprocessing pipeline for traffic analysis."""

    def __init__(self, target_size: Tuple[int, int] = (640, 640)):
        self.target_size = target_size
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # ─── Main Pipeline ────────────────────────────────────────

    def preprocess_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Full preprocessing pipeline for a single frame.
        Returns (processed_frame, metadata).
        """
        metadata = {
            "original_shape": frame.shape,
            "steps": [],
        }
        out = frame.copy()

        # 1. Noise reduction (bilateral preserves edges)
        out = self.remove_noise(out)
        metadata["steps"].append("noise_reduction")

        # 2. Contrast enhancement
        out = self.enhance_contrast(out)
        metadata["steps"].append("contrast_enhancement")

        # 3. Low-light enhancement if needed
        if self._is_low_light(out):
            out = self.enhance_low_light(out)
            metadata["steps"].append("low_light_enhancement")

        # 4. Shadow reduction
        out = self.reduce_shadows(out)
        metadata["steps"].append("shadow_reduction")

        return out, metadata

    # ─── Individual Processors ────────────────────────────────

    def remove_noise(self, image: np.ndarray) -> np.ndarray:
        """Bilateral filter: removes noise while keeping edges sharp."""
        return cv2.bilateralFilter(image, d=7, sigmaColor=50, sigmaSpace=50)

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """CLAHE on L-channel of LAB colour space."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    def enhance_low_light(self, image: np.ndarray) -> np.ndarray:
        """Gamma correction + brightness boost for dark frames."""
        gamma = 0.7
        table = np.array(
            [((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]
        ).astype("uint8")
        image = cv2.LUT(image, table)
        return cv2.convertScaleAbs(image, alpha=1.0, beta=25)

    def reduce_shadows(self, image: np.ndarray) -> np.ndarray:
        """Brighten shadow regions identified via HSV value channel."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        shadow_mask = cv2.inRange(hsv, (0, 0, 0), (180, 80, 80))
        v_bright = cv2.add(v, 35, mask=shadow_mask)
        v = np.where(shadow_mask > 0, v_bright, v).astype(np.uint8)
        return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

    def letterbox_resize(self, image: np.ndarray,
                         target_size: Optional[Tuple[int, int]] = None
                         ) -> Tuple[np.ndarray, Dict]:
        """Resize maintaining aspect ratio with letterbox padding."""
        if target_size is None:
            target_size = self.target_size
        th, tw = target_size
        h, w = image.shape[:2]
        scale = min(tw / w, th / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        padded = np.full((th, tw, 3), 114, dtype=np.uint8)
        pt, pl = (th - nh) // 2, (tw - nw) // 2
        padded[pt:pt + nh, pl:pl + nw] = resized
        return padded, {"scale": scale, "pad_top": pt, "pad_left": pl}

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize pixel values to [0, 1]."""
        return image.astype(np.float32) / 255.0

    # ─── OCR-specific preprocessing ──────────────────────────

    def preprocess_for_ocr(self, plate_image: np.ndarray) -> np.ndarray:
        """Specialized pipeline for license plate OCR."""
        if plate_image is None or plate_image.size == 0:
            return plate_image

        # Grayscale
        gray = (
            cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
            if len(plate_image.shape) == 3
            else plate_image.copy()
        )

        # CLAHE
        gray = self.clahe.apply(gray)

        # Sharpen
        gaussian = cv2.GaussianBlur(gray, (0, 0), 3)
        gray = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)

        # Bilateral filter
        gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

        # Resize to standard height
        h, w = gray.shape
        if h > 0:
            target_h = 128
            target_w = min(int(w * target_h / h), 320)
            gray = cv2.resize(gray, (target_w, target_h),
                              interpolation=cv2.INTER_CUBIC)

        return gray

    # ─── Internal Helpers ─────────────────────────────────────

    def _is_low_light(self, image: np.ndarray, threshold: float = 80.0) -> bool:
        """Check if image is low-light based on mean brightness."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray)) < threshold
