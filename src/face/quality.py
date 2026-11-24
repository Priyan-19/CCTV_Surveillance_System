"""Face quality verification and filtering module for Phase 3.

Filters out low-resolution, blurred, or occluded face detections before
embedding extraction and matching, preventing noisy embeddings from degrading
identity assignments (conforming to docs/10_TEST_PLAN.md T-07).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np


@dataclass
class QualityResult:
    """Represents quality assessment outcome for a face crop."""

    is_valid: bool
    reason: str
    blur_score: float
    width: int
    height: int
    det_score: float

    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "reason": self.reason,
            "blur_score": round(self.blur_score, 2),
            "dimensions": f"{self.width}x{self.height}",
            "det_score": round(self.det_score, 3),
        }


class FaceQualityFilter:
    """Evaluates face image candidates against geometric and clarity criteria."""

    def __init__(
        self,
        min_face_size: int = 40,
        min_blur_score: float = 35.0,
        min_det_score: float = 0.55,
    ) -> None:
        """Initialize quality filter parameters.

        Args:
            min_face_size: Minimum width and height of face bounding box in pixels.
            min_blur_score: Minimum variance of Laplacian on grayscale face crop.
            min_det_score: Minimum detection probability from face detector.
        """
        self.min_face_size = min_face_size
        self.min_blur_score = min_blur_score
        self.min_det_score = min_det_score

    def evaluate(
        self,
        face_img: np.ndarray,
        det_score: float = 1.0,
        face_bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> QualityResult:
        """Evaluate a face crop.

        Args:
            face_img: BGR image crop of the face.
            det_score: Confidence returned by the face detector.
            face_bbox: Optional (x1, y1, x2, y2) bounding box in frame coordinates.

        Returns:
            QualityResult with is_valid=True if all thresholds pass.
        """
        if face_img is None or face_img.size == 0:
            return QualityResult(
                is_valid=False,
                reason="empty_image",
                blur_score=0.0,
                width=0,
                height=0,
                det_score=float(det_score),
            )

        h, w = face_img.shape[:2]

        # 1. Detection score check
        if det_score < self.min_det_score:
            return QualityResult(
                is_valid=False,
                reason=f"low_detection_confidence_{det_score:.2f}_below_{self.min_det_score:.2f}",
                blur_score=0.0,
                width=w,
                height=h,
                det_score=float(det_score),
            )

        # 2. Geometric size check
        if w < self.min_face_size or h < self.min_face_size:
            return QualityResult(
                is_valid=False,
                reason=f"low_resolution_{w}x{h}_below_{self.min_face_size}px",
                blur_score=0.0,
                width=w,
                height=h,
                det_score=float(det_score),
            )

        # 3. Blur check using Laplacian variance
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if face_img.ndim == 3 else face_img
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            blur_var = float(laplacian.var())
        except Exception:
            blur_var = 0.0

        if blur_var < self.min_blur_score:
            return QualityResult(
                is_valid=False,
                reason=f"blurred_face_{blur_var:.1f}_below_{self.min_blur_score:.1f}",
                blur_score=blur_var,
                width=w,
                height=h,
                det_score=float(det_score),
            )

        return QualityResult(
            is_valid=True,
            reason="quality_passed",
            blur_score=blur_var,
            width=w,
            height=h,
            det_score=float(det_score),
        )
