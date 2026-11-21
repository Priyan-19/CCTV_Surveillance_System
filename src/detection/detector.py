"""Object detection wrapper module.

Provides a model-agnostic interface for deep learning detectors.
Runs YOLO models with FP16 precision on CUDA, filters detections by
configured class IDs, and constructs standardized Detection objects.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set
import numpy as np
import torch
from ultralytics import YOLO

from src.common.types import Detection

logger = logging.getLogger("detector")


class ObjectDetector:
    """YOLO-based object detector with GPU acceleration, class filtering, and warm-up."""

    def __init__(
        self,
        model_name: str = "yolo26s.pt",
        device: str = "cuda:0",
        half_precision: bool = True,
        imgsz: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.70,
        tracked_classes: Optional[Sequence[int]] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.half_precision = half_precision
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Default surveillance COCO classes:
        # 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck,
        # 24: backpack, 26: handbag, 28: suitcase
        self.tracked_classes: Set[int] = (
            set(tracked_classes)
            if tracked_classes is not None
            else {0, 1, 2, 3, 5, 7, 24, 26, 28}
        )

        self._load_and_warmup()

    def _load_and_warmup(self) -> None:
        """Load YOLO model weights onto CUDA and perform inference warm-up."""
        logger.info("Initializing detector with model: '%s' on device: '%s'...", self.model_name, self.device)
        self.model = YOLO(self.model_name)

        # Place model on device
        self.model.to(self.device)

        # Put model into FP16 half precision if requested and supported
        if self.half_precision and "cuda" in str(self.device) and torch.cuda.is_available():
            try:
                self.model.model.half()
                logger.info("Model weights placed into FP16 half-precision.")
            except Exception as e:
                logger.warning("Could not set model.model.half(): %s", e)

        # Map class IDs to class names from model metadata
        self.class_names: Dict[int, str] = self.model.names if hasattr(self.model, "names") else {}

        # Warm-up pass to eliminate initial CUDA kernel compilation latency
        if torch.cuda.is_available() and "cuda" in str(self.device):
            logger.info("Running CUDA warm-up inference pass (%d px)...", self.imgsz)
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            _ = self.model.predict(
                dummy,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )
            torch.cuda.synchronize()
            logger.info("Detector warm-up completed successfully.")

    def detect(
        self,
        frame: np.ndarray,
        camera_id: str,
        frame_id: int,
        timestamp: float,
    ) -> List[Detection]:
        """Perform object detection on an image frame and return filtered Detection objects."""
        # Ultralytics predict
        results = self.model.predict(
            frame,
            imgsz=self.imgsz,
            device=self.device,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=list(self.tracked_classes) if self.tracked_classes else None,
            verbose=False,
        )

        detections: List[Detection] = []
        if not results or len(results) == 0:
            return detections

        boxes_obj = results[0].boxes
        if boxes_obj is None or len(boxes_obj) == 0:
            return detections

        # Extract coordinates, confidences, and class IDs
        xyxy = boxes_obj.xyxy.cpu().numpy()
        confs = boxes_obj.conf.cpu().numpy()
        classes = boxes_obj.cls.cpu().numpy().astype(int)

        for i in range(len(xyxy)):
            cls_id = int(classes[i])
            # Strict class filtering
            if self.tracked_classes and cls_id not in self.tracked_classes:
                continue

            conf = float(confs[i])
            x1, y1, x2, y2 = xyxy[i]
            cls_name = self.class_names.get(cls_id, f"class_{cls_id}")

            det = Detection(
                camera_id=camera_id,
                frame_id=frame_id,
                timestamp=timestamp,
                class_id=cls_id,
                class_name=cls_name,
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                confidence=conf,
            )
            detections.append(det)

        return detections
