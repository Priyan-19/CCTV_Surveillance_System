"""Event normalizer and validator for the Threat Engine.

Validates incoming security events, ensures standard event types and schema,
and provides deterministic timestamp assignment.
Conforms to docs/04_DATA_FLOW.md Section 6 and docs/05_THREAT_ENGINE_SPEC.md.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Optional, Union

from src.common.types import Event, EventType, Severity

logger = logging.getLogger(__name__)

# Default severities matching docs/05 Section 2
DEFAULT_SEVERITY_MAP: Dict[EventType, Severity] = {
    EventType.RESTRICTED_ZONE_ENTRY: Severity.HIGH,
    EventType.LINE_CROSSING: Severity.MEDIUM,
    EventType.PROLONGED_STAY: Severity.MEDIUM,
    EventType.LOITERING: Severity.MEDIUM,
    EventType.ABANDONED_OBJECT: Severity.HIGH,
    EventType.OCCUPANCY_EXCEEDED: Severity.MEDIUM,
    EventType.FIGHTING: Severity.HIGH,
    EventType.RUNNING: Severity.LOW,
    EventType.UNKNOWN_PERSON: Severity.LOW,
}


class EventNormalizer:
    """Normalizes and validates raw incoming events into standard Event dataclasses."""

    def __init__(self, base_timestamp: Optional[float] = None) -> None:
        """Initialize the normalizer.

        Args:
            base_timestamp: Optional baseline wall-clock epoch timestamp. When provided,
                relative video timestamps can be translated deterministically.
        """
        self.base_timestamp = base_timestamp

    def normalize(self, raw_event: Union[Event, Dict[str, Any]]) -> Event:
        """Validate and normalize a raw dictionary or existing Event into a standard Event.

        Args:
            raw_event: Dictionary of event fields or an Event dataclass instance.

        Returns:
            Validated Event dataclass instance.

        Raises:
            ValueError: If mandatory fields are missing or invalid.
        """
        if isinstance(raw_event, Event):
            return raw_event

        if not isinstance(raw_event, dict):
            raise ValueError(f"Expected event dict or Event instance, got {type(raw_event)}")

        # 1. Event ID
        event_id = raw_event.get("event_id")
        if not event_id:
            event_id = f"EVT-{uuid.uuid4().hex[:8].upper()}"

        # 2. Camera ID
        camera_id = raw_event.get("camera_id")
        if not camera_id:
            raise ValueError("Event is missing required field: 'camera_id'")

        # 3. Event Type
        raw_type = raw_event.get("event_type")
        if not raw_type:
            raise ValueError("Event is missing required field: 'event_type'")

        if isinstance(raw_type, EventType):
            event_type = raw_type
        else:
            try:
                event_type = EventType(str(raw_type))
            except ValueError:
                # Attempt case-insensitive lookup
                type_str = str(raw_type).strip().upper()
                try:
                    event_type = EventType[type_str]
                except KeyError:
                    raise ValueError(
                        f"Unrecognized event_type: '{raw_type}'. Must be one of {[e.value for e in EventType]}"
                    )

        # 4. Timestamp
        raw_ts = raw_event.get("timestamp")
        if raw_ts is None:
            raise ValueError("Event is missing required field: 'timestamp'")
        try:
            timestamp = float(raw_ts)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid timestamp value: {raw_ts}")

        # 5. Severity
        raw_sev = raw_event.get("severity")
        if raw_sev:
            if isinstance(raw_sev, Severity):
                severity = raw_sev
            else:
                try:
                    severity = Severity(str(raw_sev).capitalize())
                except ValueError:
                    severity = DEFAULT_SEVERITY_MAP.get(event_type, Severity.LOW)
        else:
            severity = DEFAULT_SEVERITY_MAP.get(event_type, Severity.LOW)

        # 6. Confidence
        confidence = float(raw_event.get("confidence", 1.0))
        confidence = max(0.0, min(1.0, confidence))

        # 7. Location
        location = raw_event.get("location")
        if not isinstance(location, dict):
            location = {
                "zone_id": str(location) if location else "default",
                "zone_name": str(location) if location else "Default Zone",
                "coordinates": (0.0, 0.0),
            }
        else:
            location = {
                "zone_id": location.get("zone_id", "default"),
                "zone_name": location.get("zone_name", location.get("zone_id", "Default Zone")),
                "coordinates": location.get("coordinates", (0.0, 0.0)),
                "high_risk": bool(location.get("high_risk", False) or location.get("is_high_risk", False)),
                "authorized": bool(location.get("authorized", False) or location.get("is_authorized", False)),
            }

        # 8. Tracking & Identity Links
        track_id = raw_event.get("track_id")
        if track_id is not None:
            try:
                track_id = int(track_id)
            except (ValueError, TypeError):
                track_id = None

        person_id = raw_event.get("person_id")
        object_id = raw_event.get("object_id")
        details = raw_event.get("details", {})
        if not isinstance(details, dict):
            details = {"raw": details}

        return Event(
            event_id=str(event_id),
            camera_id=str(camera_id),
            track_id=track_id,
            event_type=event_type,
            timestamp=timestamp,
            severity=severity,
            confidence=confidence,
            location=location,
            details=details,
            person_id=person_id,
            object_id=object_id,
        )
