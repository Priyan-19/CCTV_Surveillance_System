"""Data structures and types conforming to docs/04_DATA_FLOW.md.

Defines the core data contracts for detections, tracks, frame results,
and pipeline performance timings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class Detection:
    """Represents a single bounding box detection from the object detector.

    Attributes:
        camera_id: Identifier of the capturing camera/source stream.
        frame_id: Monotonically increasing frame index.
        timestamp: Unix epoch timestamp in seconds.
        class_id: Numeric integer class ID (e.g. COCO class 0 = person).
        class_name: Human-readable class label (e.g. 'person', 'backpack').
        bbox: Bounding box tuple (x1, y1, x2, y2) in pixel coordinates.
        confidence: Detection probability score between 0.0 and 1.0.
    """

    camera_id: str
    frame_id: int
    timestamp: float
    class_id: int
    class_name: str
    bbox: Tuple[float, float, float, float]
    confidence: float

    @property
    def centre(self) -> Tuple[float, float]:
        """Compute the (cx, cy) centroid of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def width(self) -> float:
        """Width of the bounding box."""
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        """Height of the bounding box."""
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> float:
        """Pixel area of the bounding box."""
        return self.width * self.height


@dataclass
class Track:
    """Represents an active multi-frame object trajectory produced by the tracker.

    Attributes:
        camera_id: Identifier of the capturing camera stream.
        track_id: Unique integer identifier assigned by the tracker.
        class_id: Detected class index.
        class_name: Human-readable class name.
        bbox: Current bounding box (x1, y1, x2, y2).
        confidence: Confidence score of the most recent associated detection.
        path: History of centroids [(cx, cy), ...] ordered chronologically.
        first_seen: Unix epoch timestamp when track was first initiated.
        last_seen: Unix epoch timestamp of the most recent track observation.
        zone_ids: List of spatial zone IDs containing this track (Phase 2 placeholder).
        dwell_seconds: Duration in seconds spent inside current zone (Phase 2 placeholder).
    """

    camera_id: str
    track_id: int
    class_id: int
    class_name: str
    bbox: Tuple[float, float, float, float]
    confidence: float
    path: List[Tuple[float, float]] = field(default_factory=list)
    first_seen: float = 0.0
    last_seen: float = 0.0
    # Phase 2 spatial reasoning placeholders:
    zone_ids: List[str] = field(default_factory=list)
    dwell_seconds: float = 0.0

    @property
    def centre(self) -> Tuple[float, float]:
        """Current centroid of the track."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def lifespan_seconds(self) -> float:
        """Total duration this track has persisted."""
        return max(0.0, self.last_seen - self.first_seen)


@dataclass
class FrameResult:
    """Aggregated output for a single processed frame across all pipeline stages.

    Attributes:
        frame: The raw or annotated image frame (OpenCV BGR numpy array).
        frame_id: Frame sequence index.
        timestamp: Timestamp of frame acquisition.
        detections: List of raw detections before tracking.
        tracks: List of active Track instances in this frame.
        timings: Dictionary mapping stage names to latency in milliseconds
                 (e.g. {'read_ms': 1.2, 'detect_ms': 18.4, 'track_ms': 4.1}).
    """

    frame: np.ndarray
    frame_id: int
    timestamp: float
    detections: List[Detection] = field(default_factory=list)
    tracks: List[Track] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)
