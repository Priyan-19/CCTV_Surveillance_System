"""Object tracking wrapper module.

Provides a unified Tracker interface conforming to docs/04_DATA_FLOW.md.
Implements:
1. UltralyticsByteTracker: GPU-accelerated ByteTrack with persist=True using configs/bytetrack.yaml.
2. Lightweight centroid/IoU tracker for offline unit tests and CPU verification.
Tracks maintain persistent motion histories (last N centroids), first_seen, and last_seen timestamps.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple
import numpy as np
import torch
from ultralytics import YOLO

from src.common.types import Detection, Track

logger = logging.getLogger("tracker")


class BaseTracker(ABC):
    """Abstract Base Class for multi-object trackers."""

    @abstractmethod
    def update(
        self,
        frame: np.ndarray,
        camera_id: str,
        frame_id: int,
        timestamp: float,
    ) -> Tuple[List[Detection], List[Track]]:
        """Process a new frame and return both raw detections and active tracks."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal tracker state and trajectory histories."""
        pass


class ByteTrackTracker(BaseTracker):
    """Production tracker wrapping Ultralytics ByteTrack with configurable parameters."""

    def __init__(
        self,
        model_name: str = "yolo26s.pt",
        tracker_config: str = "configs/bytetrack.yaml",
        device: str = "cuda:0",
        half_precision: bool = True,
        imgsz: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.70,
        tracked_classes: Optional[Sequence[int]] = None,
        max_trail_length: int = 30,
    ) -> None:
        self.model_name = model_name
        self.tracker_config = tracker_config
        self.device = device
        self.half_precision = half_precision
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_trail_length = max_trail_length

        self.tracked_classes: Set[int] = (
            set(tracked_classes)
            if tracked_classes is not None
            else {0, 1, 2, 3, 5, 7, 24, 26, 28}
        )

        # Persistent track state: track_id -> Track
        self._active_tracks: Dict[int, Track] = {}

        self._init_model()

    def _init_model(self) -> None:
        """Initialize YOLO model for tracking."""
        logger.info(
            "Initializing ByteTrack tracker (config: '%s') using model: '%s' on %s...",
            self.tracker_config,
            self.model_name,
            self.device,
        )

        cfg_path = Path(self.tracker_config)
        if not cfg_path.exists():
            raise FileNotFoundError(f"ByteTrack configuration file not found at: {cfg_path.resolve()}")

        self.model = YOLO(self.model_name)
        self.model.to(self.device)

        if self.half_precision and "cuda" in str(self.device) and torch.cuda.is_available():
            try:
                self.model.model.half()
            except Exception:
                pass

        self.class_names: Dict[int, str] = self.model.names if hasattr(self.model, "names") else {}

        # Log confirmation of tracker selection
        logger.info("Confirmed active tracker: ByteTrack (loaded from %s)", self.tracker_config)

    def update(
        self,
        frame: np.ndarray,
        camera_id: str,
        frame_id: int,
        timestamp: float,
    ) -> Tuple[List[Detection], List[Track]]:
        """Run YOLO inference and ByteTrack association."""
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_config,
            imgsz=self.imgsz,
            device=self.device,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=list(self.tracked_classes) if self.tracked_classes else None,
            verbose=False,
        )

        detections: List[Detection] = []
        current_tracks: List[Track] = []

        if not results or len(results) == 0:
            return detections, current_tracks

        boxes_obj = results[0].boxes
        if boxes_obj is None or len(boxes_obj) == 0:
            return detections, current_tracks

        xyxy = boxes_obj.xyxy.cpu().numpy()
        confs = boxes_obj.conf.cpu().numpy()
        classes = boxes_obj.cls.cpu().numpy().astype(int)
        ids = boxes_obj.id.cpu().numpy().astype(int) if boxes_obj.id is not None else None

        for i in range(len(xyxy)):
            cls_id = int(classes[i])
            if self.tracked_classes and cls_id not in self.tracked_classes:
                continue

            conf = float(confs[i])
            x1, y1, x2, y2 = xyxy[i]
            bbox_tuple = (float(x1), float(y1), float(x2), float(y2))
            cls_name = self.class_names.get(cls_id, f"class_{cls_id}")

            # 1. Record raw detection
            det = Detection(
                camera_id=camera_id,
                frame_id=frame_id,
                timestamp=timestamp,
                class_id=cls_id,
                class_name=cls_name,
                bbox=bbox_tuple,
                confidence=conf,
            )
            detections.append(det)

            # 2. Record track if tracker assigned an ID
            if ids is not None and i < len(ids):
                track_id = int(ids[i])
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                if track_id in self._active_tracks:
                    existing = self._active_tracks[track_id]
                    existing.bbox = bbox_tuple
                    existing.confidence = conf
                    existing.last_seen = timestamp
                    existing.path.append((cx, cy))
                    if len(existing.path) > self.max_trail_length:
                        existing.path = existing.path[-self.max_trail_length:]
                    current_tracks.append(existing)
                else:
                    new_track = Track(
                        camera_id=camera_id,
                        track_id=track_id,
                        class_id=cls_id,
                        class_name=cls_name,
                        bbox=bbox_tuple,
                        confidence=conf,
                        path=[(cx, cy)],
                        first_seen=timestamp,
                        last_seen=timestamp,
                    )
                    self._active_tracks[track_id] = new_track
                    current_tracks.append(new_track)

        return detections, current_tracks

    def reset(self) -> None:
        """Clear all active tracks and reset tracker state."""
        self._active_tracks.clear()
        if hasattr(self.model, "predictor") and self.model.predictor is not None:
            if hasattr(self.model.predictor, "trackers"):
                self.model.predictor.trackers = None


class SyntheticDetectionTracker(BaseTracker):
    """Pure Python tracker for unit testing without GPU or neural network dependencies.

    Uses simple IoU matching between consecutive frames to maintain Track IDs.
    """

    def __init__(self, iou_thresh: float = 0.3, max_trail_length: int = 30) -> None:
        self.iou_thresh = iou_thresh
        self.max_trail_length = max_trail_length
        self._next_id: int = 1
        self._tracks: Dict[int, Track] = {}

    @staticmethod
    def _compute_iou(b1: Tuple[float, float, float, float], b2: Tuple[float, float, float, float]) -> float:
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union_area = area1 + area2 - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def update_with_detections(
        self,
        detections: List[Detection],
        camera_id: str,
        frame_id: int,
        timestamp: float,
    ) -> List[Track]:
        """Update tracker state directly from a list of detections."""
        updated_tracks: List[Track] = []
        matched_det_indices = set()
        matched_track_ids = set()

        for track_id, track in list(self._tracks.items()):
            best_iou = 0.0
            best_det_idx = -1
            for d_idx, det in enumerate(detections):
                if d_idx in matched_det_indices:
                    continue
                if det.class_id != track.class_id:
                    continue
                iou = self._compute_iou(track.bbox, det.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = d_idx

            if best_iou >= self.iou_thresh and best_det_idx >= 0:
                det = detections[best_det_idx]
                track.bbox = det.bbox
                track.confidence = det.confidence
                track.last_seen = timestamp
                track.path.append(det.centre)
                if len(track.path) > self.max_trail_length:
                    track.path = track.path[-self.max_trail_length:]
                matched_det_indices.add(best_det_idx)
                matched_track_ids.add(track_id)
                updated_tracks.append(track)

        # Unmatched detections become new tracks
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_det_indices:
                new_track = Track(
                    camera_id=camera_id,
                    track_id=self._next_id,
                    class_id=det.class_id,
                    class_name=det.class_name,
                    bbox=det.bbox,
                    confidence=det.confidence,
                    path=[det.centre],
                    first_seen=timestamp,
                    last_seen=timestamp,
                )
                self._tracks[self._next_id] = new_track
                self._next_id += 1
                updated_tracks.append(new_track)

        return updated_tracks

    def update(
        self,
        frame: np.ndarray,
        camera_id: str,
        frame_id: int,
        timestamp: float,
    ) -> Tuple[List[Detection], List[Track]]:
        return [], []

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
