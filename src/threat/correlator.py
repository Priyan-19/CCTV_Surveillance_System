"""Single-camera event correlator for the Threat Engine.

Correlates incoming security events into active open incidents based on
track_id, person_id, object_id (bag-owner linkage), and spatial zone
within a configurable sliding time window.
Conforms strictly to docs/05_THREAT_ENGINE_SPEC.md Section 4.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.common.types import Event, EventType, Incident

logger = logging.getLogger(__name__)


class SingleCameraCorrelator:
    """Correlates security events for a single camera into unified incidents."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize correlator with configuration parameters.

        Args:
            config: Optional configuration dictionary. Uses docs/05 defaults if omitted.
        """
        cfg = config or {}
        corr_cfg = cfg.get("correlation", {})

        self.time_window: float = float(corr_cfg.get("time_window_seconds", 120.0))
        self.quiet_period: float = float(corr_cfg.get("quiet_period_seconds", 300.0))
        self.pending_unknown_buffer: float = float(
            corr_cfg.get("pending_unknown_buffer_seconds", 10.0)
        )
        self.correlation_keys: List[str] = corr_cfg.get(
            "correlation_keys", ["track_id", "person_id", "object_id", "location"]
        )

        # Buffer for UNKNOWN_PERSON context events held until subsequent rule events arrive
        # Key: (camera_id, track_id) -> Event
        self._pending_unknown_events: Dict[tuple[str, int], Event] = {}

    def correlate(
        self, event: Event, open_incidents: List[Incident]
    ) -> Optional[Incident]:
        """Correlate an incoming event against currently open incidents.

        Rules (docs/05 Section 4):
        1. Only consider open incidents for the same camera.
        2. Must fall within the correlation time window (abs(t_event - t_last) <= window).
        3. Matches if ANY correlation key holds:
           - Same Track ID or Person ID.
           - Same Object ID.
           - Owner linkage (e.g. ABANDONED_OBJECT owner_track_id matches person track_id).
           - Same non-trivial zone within time window.

        Args:
            event: The validated incoming Event.
            open_incidents: Active, unclosed incidents.

        Returns:
            The matched Incident if found, or None if a new incident must be started.
        """
        # Buffer UNKNOWN_PERSON as context rather than standalone incident trigger
        if event.event_type == EventType.UNKNOWN_PERSON and event.track_id is not None:
            self._pending_unknown_events[(event.camera_id, event.track_id)] = event
            # Also check if an open incident already exists for this track
            for inc in open_incidents:
                if inc.camera_id == event.camera_id and inc.is_open:
                    if self._matches_track_or_person(event, inc):
                        return inc
            return None

        # Filter candidate incidents by camera and open status
        candidates = [
            inc for inc in open_incidents
            if inc.camera_id == event.camera_id and inc.is_open
        ]

        if not candidates:
            return None

        # Find best matching incident within the time window
        best_match: Optional[Incident] = None
        min_time_delta: float = float("inf")

        for inc in candidates:
            time_delta = abs(event.timestamp - inc.last_event_time)
            if time_delta > self.time_window:
                continue

            if self._matches_incident(event, inc):
                if time_delta < min_time_delta:
                    min_time_delta = time_delta
                    best_match = inc

        return best_match

    def get_buffered_unknown(self, camera_id: str, track_id: Optional[int], timestamp: float) -> Optional[Event]:
        """Retrieve and consume buffered UNKNOWN_PERSON context if within buffer window."""
        if track_id is None:
            return None
        key = (camera_id, track_id)
        if key in self._pending_unknown_events:
            buf_evt = self._pending_unknown_events[key]
            if abs(timestamp - buf_evt.timestamp) <= self.pending_unknown_buffer:
                del self._pending_unknown_events[key]
                return buf_evt
            else:
                del self._pending_unknown_events[key]
        return None

    def find_expired_incidents(
        self, open_incidents: List[Incident], current_time: float
    ) -> List[Incident]:
        """Identify open incidents whose quiet period has expired.

        Args:
            open_incidents: List of currently open incidents.
            current_time: Current simulation or wall-clock timestamp.

        Returns:
            List of incidents that have exceeded the quiet period.
        """
        expired = []
        for inc in open_incidents:
            if inc.is_open and (current_time - inc.last_event_time) >= self.quiet_period:
                expired.append(inc)
        return expired

    def _matches_incident(self, event: Event, inc: Incident) -> bool:
        """Check if an event matches an open incident on any active correlation key."""
        # 1. Track ID or Person ID match
        if self._matches_track_or_person(event, inc):
            return True

        # 2. Object ID match (e.g. bag track ID)
        if event.object_id is not None:
            inc_obj = inc.subject.get("object_id")
            if inc_obj is not None and str(inc_obj) == str(event.object_id):
                return True
            for ev in inc.events:
                if ev.object_id is not None and str(ev.object_id) == str(event.object_id):
                    return True

        # 3. Bag / Owner linkage
        if self._matches_owner_linkage(event, inc):
            return True

        # 4. Same zone location within time window
        if "location" in self.correlation_keys:
            ev_zone = event.location.get("zone_id")
            if ev_zone and ev_zone not in ("default", "", "none", None):
                # Check incident subject zone or any event's zone
                for ev in inc.events:
                    if ev.location.get("zone_id") == ev_zone:
                        return True

        return False

    def _matches_track_or_person(self, event: Event, inc: Incident) -> bool:
        """Check direct track_id or person_id match."""
        # Track ID match
        if event.track_id is not None:
            if inc.subject.get("track_id") == event.track_id:
                return True
            for ev in inc.events:
                if ev.track_id == event.track_id:
                    return True

        # Person ID match
        if event.person_id is not None:
            if inc.subject.get("person_id") == event.person_id:
                return True
            for ev in inc.events:
                if ev.person_id == event.person_id:
                    return True

        return False

    def _matches_owner_linkage(self, event: Event, inc: Incident) -> bool:
        """Check owner-object linkage for abandoned objects or carried items."""
        # Case A: Incoming event is ABANDONED_OBJECT carrying owner_track_id in details
        if event.event_type == EventType.ABANDONED_OBJECT:
            owner_track_id = event.details.get("owner_track_id")
            if owner_track_id is not None:
                if inc.subject.get("track_id") == owner_track_id:
                    return True
                for ev in inc.events:
                    if ev.track_id == owner_track_id:
                        return True

        # Case B: Incident already has an ABANDONED_OBJECT whose owner_track_id matches incoming event
        for ev in inc.events:
            if ev.event_type == EventType.ABANDONED_OBJECT:
                owner_track_id = ev.details.get("owner_track_id")
                if owner_track_id is not None and event.track_id == owner_track_id:
                    return True

        return False
