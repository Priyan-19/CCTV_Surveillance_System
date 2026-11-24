"""Snapshot and visual crop extractor for Evidence Collection.

Extracts person bounding boxes, associated object bounding boxes, and
face snapshots with margin expansions, saving them as high-quality JPEGs.
Conforms to docs/05_THREAT_ENGINE_SPEC.md, docs/07_DATABASE_SCHEMA.md,
and docs/11_SECURITY_PRIVACY.md (proportionality principle).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CropExtractor:
    """Extracts and saves photographic evidence crops for incident subjects."""

    @staticmethod
    def extract_crop(
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
        save_path: Path,
        margin_pct: float = 0.05,
        jpeg_quality: int = 95,
    ) -> Optional[str]:
        """Crop bounding box region from a frame and save as high-quality JPEG.

        Args:
            frame: Full BGR video frame.
            bbox: (x1, y1, x2, y2) bounding box in pixel coordinates.
            save_path: Destination file path.
            margin_pct: Optional percentage margin expansion around the box.
            jpeg_quality: Output JPEG compression quality (1-100).

        Returns:
            Normalized file path string if successful, or None on failure.
        """
        if frame is None or frame.size == 0:
            return None

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox

        # Margin expansion
        box_w = x2 - x1
        box_h = y2 - y1
        mx = box_w * margin_pct
        my = box_h * margin_pct

        cx1 = max(0, int(round(x1 - mx)))
        cy1 = max(0, int(round(y1 - my)))
        cx2 = min(w, int(round(x2 + mx)))
        cy2 = min(h, int(round(y2 + my)))

        if cx2 <= cx1 or cy2 <= cy1:
            return None

        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None

        save_path.parent.mkdir(parents=True, exist_ok=True)
        success = cv2.imwrite(
            str(save_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        )
        if success:
            return str(save_path).replace("\\", "/")
        return None

    @staticmethod
    def save_face_crop(
        face_img: np.ndarray,
        save_path: Path,
        jpeg_quality: int = 95,
    ) -> Optional[str]:
        """Save a pre-cropped aligned face image (from FaceWorker or detector).

        Args:
            face_img: Aligned BGR face crop.
            save_path: Destination file path.
            jpeg_quality: Output JPEG quality.

        Returns:
            Normalized file path string if successful, or None.
        """
        if face_img is None or face_img.size == 0:
            return None

        save_path.parent.mkdir(parents=True, exist_ok=True)
        success = cv2.imwrite(
            str(save_path), face_img, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        )
        if success:
            return str(save_path).replace("\\", "/")
        return None
