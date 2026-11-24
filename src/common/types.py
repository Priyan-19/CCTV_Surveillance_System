"""Data structures and types conforming to docs/04_DATA_FLOW.md and docs/05_THREAT_ENGINE_SPEC.md.

Defines the core data contracts for detections, tracks, frame results,
security events, spatial zones, virtual lines, and pipeline timings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class EventType(str, Enum):
    """Standard security event types from docs/05_THREAT_ENGINE_SPEC.md."""

    RESTRICTED_ZONE_ENTRY = "RESTRICTED_ZONE_ENTRY"
    LINE_CROSSING = "LINE_CROSSING"
    PROLONGED_STAY = "PROLONGED_STAY"
    LOITERING = "LOITERING"
    ABANDONED_OBJECT = "ABANDONED_OBJECT"
    OCCUPANCY_EXCEEDED = "OCCUPANCY_EXCEEDED"
    # Reserved for future phases:
    FIGHTING = "FIGHTING"
    RUNNING = "RUNNING"
    UNKNOWN_PERSON = "UNKNOWN_PERSON"


class Severity(str, Enum):
    """Event severity levels matching docs/05_THREAT_ENGINE_SPEC.md."""

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class IncidentStatus(str, Enum):
    """Human workflow status matching docs/05 and docs/07."""

    NEW = "new"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    FALSE_ALARM = "false_alarm"


class EvidenceType(str, Enum):
    """Evidence types strictly matching docs/07_DATABASE_SCHEMA.md."""

    PERSON_IMAGE = "person_image"
    FACE_IMAGE = "face_image"
    OBJECT_IMAGE = "object_image"
    VIDEO_CLIP = "video_clip"


class IdentityStatus(str, Enum):
    """Identity classification state for tracked persons (Phase 3)."""

    PENDING = "PENDING"
    KNOWN = "KNOWN"
    UNKNOWN = "UNKNOWN"


@dataclass
class Identity:
    """Represents a computed or consolidated identity result."""

    status: IdentityStatus = IdentityStatus.PENDING
    person_id: Optional[int] = None
    name: Optional[str] = None
    confidence: float = 0.0
    margin: float = 0.0
    observation_count: int = 0
    reason: Optional[str] = None
    face_bbox: Optional[Tuple[float, float, float, float]] = None


@dataclass
class Detection:
    """Represents a single bounding box detection from the object detector.

    Attributes:
        camera_id: Identifier of the capturing camera/source stream.
        frame_id: Monotonically increasing frame index.
        timestamp: Timestamp in seconds (video-time or capture time).
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
    def bottom_center(self) -> Tuple[float, float]:
        """Compute bottom-centre (feet contact point) on ground plane."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y2)

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class Track:
    """Represents an active multi-frame object trajectory produced by the tracker."""

    camera_id: str
    track_id: int
    class_id: int
    class_name: str
    bbox: Tuple[float, float, float, float]
    confidence: float
    path: List[Tuple[float, float]] = field(default_factory=list)
    first_seen: float = 0.0
    last_seen: float = 0.0
    # Spatial & temporal rule state:
    zone_ids: List[str] = field(default_factory=list)
    dwell_seconds: float = 0.0
    # Phase 3 Face Recognition state:
    person_id: Optional[int] = None
    person_name: Optional[str] = None
    identity_status: IdentityStatus = IdentityStatus.PENDING
    identity_confidence: float = 0.0
    identity_margin: float = 0.0

    @property
    def centre(self) -> Tuple[float, float]:
        """Current centroid of the track."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def bottom_center(self) -> Tuple[float, float]:
        """Current bottom-centre (feet contact point) of the track."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y2)

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def lifespan_seconds(self) -> float:
        return max(0.0, self.last_seen - self.first_seen)


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert NumPy scalars/arrays to native Python primitives for JSON."""
    if isinstance(obj, (np.floating, float)):
        return round(float(obj), 3)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return [_sanitize_for_json(v) for v in obj.tolist()]
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    return obj


@dataclass
class Event:
    """Represents a validated security event produced by the Rule Engine.

    Conforms to docs/04_DATA_FLOW.md Section 6.
    """

    event_id: str
    camera_id: str
    track_id: Optional[int]
    event_type: EventType
    timestamp: float
    severity: Severity
    confidence: float
    location: Dict[str, Any]  # {'zone_id': ..., 'zone_name': ..., 'coordinates': (x, y)}
    details: Dict[str, Any] = field(default_factory=dict)
    person_id: Optional[Union[int, str]] = None  # Populated in Phase 3
    object_id: Optional[Union[str, int]] = None  # For associated bags/objects

    def to_dict(self) -> Dict[str, Any]:
        """Serialize event to a JSON-compatible dictionary."""
        return {
            "event_id": self.event_id,
            "camera_id": self.camera_id,
            "track_id": self.track_id,
            "person_id": self.person_id,
            "object_id": self.object_id,
            "event_type": self.event_type.value,
            "timestamp": round(float(self.timestamp), 3),
            "severity": self.severity.value,
            "confidence": round(float(self.confidence), 3),
            "location": _sanitize_for_json(self.location),
            "details": _sanitize_for_json(self.details),
        }


@dataclass
class FrameResult:
    """Aggregated output for a single processed frame across all pipeline stages."""

    frame: np.ndarray
    frame_id: int
    timestamp: float
    detections: List[Detection] = field(default_factory=list)
    tracks: List[Track] = field(default_factory=list)
    events: List[Event] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)


@dataclass
class ScoreExplanation:
    """Detailed score explanation breakdown for auditability (docs/05)."""

    base_score: int
    unique_events: List[str]
    modifiers_applied: List[Dict[str, Any]]
    unclamped_score: int
    final_score: int
    threat_level: str
    explanation_lines: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_score": self.base_score,
            "unique_events": self.unique_events,
            "modifiers_applied": self.modifiers_applied,
            "unclamped_score": self.unclamped_score,
            "final_score": self.final_score,
            "threat_level": self.threat_level,
            "explanation_lines": self.explanation_lines,
        }


@dataclass
class Evidence:
    """Represents a collected visual evidence artifact (docs/07)."""

    id: Optional[int] = None
    incident_id: Optional[int] = None
    type: str = EvidenceType.PERSON_IMAGE.value
    file_path: str = ""
    timestamp: float = 0.0
    location: Dict[str, Any] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "incident_id": self.incident_id,
            "type": self.type,
            "file_path": self.file_path,
            "timestamp": round(float(self.timestamp), 3),
            "location": _sanitize_for_json(self.location),
            "details": _sanitize_for_json(self.details),
        }


@dataclass
class Incident:
    """Represents a correlated security incident (docs/05 and docs/07)."""

    id: Optional[int] = None
    incident_no: str = ""
    camera_id: str = ""
    place: str = ""
    subject: Dict[str, Any] = field(default_factory=dict)
    status: str = IncidentStatus.NEW.value
    activities: List[str] = field(default_factory=list)
    threat_level: str = Severity.LOW.value
    threat_score: int = 0
    start_time: float = 0.0
    last_event_time: float = 0.0
    closed_at: Optional[float] = None
    model_versions: Dict[str, str] = field(default_factory=dict)
    keywords: str = ""
    events: List[Event] = field(default_factory=list)
    evidence: List[Evidence] = field(default_factory=list)
    score_history: List[Dict[str, Any]] = field(default_factory=list)
    explanation: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    is_open: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "incident_no": self.incident_no,
            "camera_id": self.camera_id,
            "place": self.place,
            "subject": _sanitize_for_json(self.subject),
            "status": self.status,
            "activities": self.activities,
            "threat_level": self.threat_level,
            "threat_score": self.threat_score,
            "start_time": round(float(self.start_time), 3),
            "last_event_time": round(float(self.last_event_time), 3),
            "closed_at": round(float(self.closed_at), 3) if self.closed_at is not None else None,
            "model_versions": self.model_versions,
            "keywords": self.keywords,
            "events": [e.to_dict() for e in self.events],
            "evidence": [ev.to_dict() for ev in self.evidence],
            "score_history": self.score_history,
            "explanation": self.explanation,
            "details": _sanitize_for_json(self.details),
            "is_open": self.is_open,
        }


@dataclass
class Alert:
    """Represents an alert notification dispatched to console, outbox, or operators (docs/05 Section 7)."""

    alert_id: str
    incident_no: str
    threat_level: str
    threat_score: int
    camera_id: str
    place: str
    timestamp: float
    message: str
    evidence_paths: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "incident_no": self.incident_no,
            "threat_level": self.threat_level,
            "threat_score": self.threat_score,
            "camera_id": self.camera_id,
            "place": self.place,
            "timestamp": round(float(self.timestamp), 3),
            "message": self.message,
            "evidence_paths": self.evidence_paths,
            "details": _sanitize_for_json(self.details),
        }
