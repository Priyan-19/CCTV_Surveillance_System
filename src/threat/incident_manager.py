"""Incident lifecycle manager coordinating correlation, scoring, and alerting.

Coordinates the end-to-end incident pipeline:
1. Normalizes incoming events.
2. Manages quiet-period incident closure (300s window).
3. Correlates events with open single-camera incidents (120s sliding window).
4. Re-scores incidents deterministically with ThreatScorer.
5. Records auditable score history and explanations.
6. Dispatches alerts on creation and level increases.
7. Persists incidents to SQLite via SurveillanceRepository when provided.

Conforms strictly to docs/05_THREAT_ENGINE_SPEC.md and docs/07_DATABASE_SCHEMA.md.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from src.common.types import Alert, Event, EventType, Incident, IncidentStatus, Severity
from src.threat.alert_dispatcher import AlertDispatcher
from src.threat.correlator import SingleCameraCorrelator
from src.threat.normalizer import EventNormalizer
from src.threat.scorer import ThreatScorer

logger = logging.getLogger(__name__)


class IncidentManager:
    """Manages active incidents, event correlation, threat scoring, and lifecycle."""

    def __init__(
        self,
        threat_config: Optional[Dict[str, Any]] = None,
        alerts_config: Optional[Dict[str, Any]] = None,
        repository: Optional[Any] = None,
    ) -> None:
        """Initialize IncidentManager with sub-components.

        Args:
            threat_config: Configuration for correlation and threat scoring.
            alerts_config: Configuration for alert sinks.
            repository: Optional SurveillanceRepository instance for database persistence.
        """
        self.threat_config = threat_config or {}
        self.alerts_config = alerts_config or {}
        self.repository = repository

        self.normalizer = EventNormalizer()
        self.correlator = SingleCameraCorrelator(self.threat_config)
        self.scorer = ThreatScorer(self.threat_config)
        self.alert_dispatcher = AlertDispatcher(self.alerts_config)

        # Active open incidents in memory: incident_no -> Incident
        self.open_incidents: Dict[str, Incident] = {}

        # Closed incidents archive (in memory): incident_no -> Incident
        self.closed_incidents: Dict[str, Incident] = {}

        # Sequential incident counter
        self._next_incident_seq: int = 1
        if self.repository is not None:
            try:
                next_no = self.repository.get_next_incident_no()
                # e.g. "INC-000001" -> 1
                self._next_incident_seq = int(next_no.split("-")[1])
            except Exception:
                self._next_incident_seq = 1

    def process_event(
        self, raw_event: Union[Event, Dict[str, Any]]
    ) -> Tuple[Optional[Incident], Optional[Alert]]:
        """Process an incoming raw or structured event through the Threat Engine.

        Args:
            raw_event: Incoming event dictionary or Event dataclass.

        Returns:
            Tuple of (Incident, Optional[Alert]). If the event is buffered context
            (e.g. UNKNOWN_PERSON awaiting rule events), incident may be None.
        """
        # 1. Normalize Event
        event = self.normalizer.normalize(raw_event)

        # 2. Check and close quiet-period expired incidents
        self.flush_quiet_period(event.timestamp)

        # 3. Correlate with active open incidents
        active_list = list(self.open_incidents.values())
        matched_incident = self.correlator.correlate(event, active_list)

        # Check for buffered UNKNOWN_PERSON context for this track
        buffered_unknown = self.correlator.get_buffered_unknown(
            event.camera_id, event.track_id, event.timestamp
        )

        if matched_incident is not None:
            incident = matched_incident
            # Append new event
            incident.events.append(event)
            if buffered_unknown:
                incident.events.append(buffered_unknown)
            incident.last_event_time = max(incident.last_event_time, event.timestamp)

            # Update unique activities
            ev_val = event.event_type.value
            if ev_val != EventType.UNKNOWN_PERSON.value and ev_val not in incident.activities:
                incident.activities.append(ev_val)

            # Update subject info if richer data arrived
            self._update_subject_info(incident, event, buffered_unknown)

        else:
            # If the event itself is an UNKNOWN_PERSON context event, it has been buffered
            if event.event_type == EventType.UNKNOWN_PERSON:
                return None, None

            # Create a brand new incident
            incident_no = f"INC-{self._next_incident_seq:06d}"
            self._next_incident_seq += 1

            initial_events = [event]
            if buffered_unknown:
                initial_events.append(buffered_unknown)

            activities = [event.event_type.value]
            place = (
                event.location.get("zone_name")
                or event.location.get("zone_id")
                or f"Camera {event.camera_id}"
            )

            incident = Incident(
                id=None,
                incident_no=incident_no,
                camera_id=event.camera_id,
                place=place,
                subject=self._build_subject_dict(event, buffered_unknown),
                status=IncidentStatus.NEW.value,
                activities=activities,
                threat_level=Severity.LOW.value,
                threat_score=0,
                start_time=event.timestamp,
                last_event_time=event.timestamp,
                closed_at=None,
                model_versions={"yolo": "yolo26s", "threat_engine": "v1.0"},
                keywords="",
                events=initial_events,
                evidence=[],
                score_history=[],
                explanation=[],
                details={"camera_id": event.camera_id, "place": place},
                is_open=True,
            )
            self.open_incidents[incident.incident_no] = incident

        # 4. Deterministic Re-scoring
        score_explanation = self.scorer.score_incident(incident)

        # Update incident threat score and level
        incident.threat_score = score_explanation.final_score
        incident.threat_level = score_explanation.threat_level
        incident.explanation = score_explanation.explanation_lines

        # Append to audit score history
        incident.score_history.append({
            "timestamp": event.timestamp,
            "score": score_explanation.final_score,
            "threat_level": score_explanation.threat_level,
            "unclamped_score": score_explanation.unclamped_score,
            "trigger_event": event.event_type.value,
            "modifiers": score_explanation.modifiers_applied,
        })

        # Update FTS keywords
        incident.keywords = self._build_keywords(incident)

        # 5. Alert Dispatch Evaluation
        alert = self.alert_dispatcher.evaluate_and_dispatch(incident)

        # 6. Database Persistence
        if self.repository is not None:
            try:
                if hasattr(self.repository, "register_camera"):
                    self.repository.register_camera(incident.camera_id, location=incident.place)
                if incident.id is None:
                    db_id = self.repository.create_incident(incident)
                    incident.id = db_id
                else:
                    self.repository.update_incident(incident)
                    db_id = incident.id
                # Also record individual event in DB
                self.repository.add_event(event, incident_id=db_id)
            except Exception as e:
                logger.error(f"Failed to persist incident {incident.incident_no} to repository: {e}")

        return incident, alert

    def flush_quiet_period(self, current_time: float) -> List[Incident]:
        """Close open incidents that have had no new events within quiet_period_seconds.

        Args:
            current_time: Current simulation or wall-clock timestamp.

        Returns:
            List of incidents that were closed during this flush.
        """
        active_list = list(self.open_incidents.values())
        expired = self.correlator.find_expired_incidents(active_list, current_time)

        closed_list = []
        for inc in expired:
            inc.is_open = False
            inc.closed_at = inc.last_event_time
            del self.open_incidents[inc.incident_no]
            self.closed_incidents[inc.incident_no] = inc
            closed_list.append(inc)

            if self.repository is not None and inc.id is not None:
                try:
                    self.repository.close_incident(inc.id, closed_at=inc.closed_at)
                except Exception as e:
                    logger.error(f"Failed to close incident {inc.incident_no} in repository: {e}")

        return closed_list

    def close_incident(self, incident_no: str, status: IncidentStatus = IncidentStatus.RESOLVED) -> Optional[Incident]:
        """Manually close an active incident."""
        if incident_no in self.open_incidents:
            inc = self.open_incidents.pop(incident_no)
            inc.is_open = False
            inc.status = status.value
            self.closed_incidents[inc.incident_no] = inc
            if self.repository is not None and inc.id is not None:
                self.repository.update_incident_status(inc.id, status.value)
            return inc
        return None

    def _build_subject_dict(self, event: Event, buffered_unknown: Optional[Event] = None) -> Dict[str, Any]:
        """Build subject tracking and identity dictionary."""
        subject: Dict[str, Any] = {
            "track_id": event.track_id,
            "person_id": event.person_id,
            "object_id": event.object_id,
            "identity_status": event.details.get("identity_status", "PENDING"),
            "person_name": event.details.get("person_name"),
            "reason": event.details.get("reason"),
        }
        if buffered_unknown:
            subject["identity_status"] = "UNKNOWN"
            if buffered_unknown.details.get("reason"):
                subject["reason"] = buffered_unknown.details.get("reason")

        if event.event_type == EventType.ABANDONED_OBJECT:
            subject["owner_track_id"] = event.details.get("owner_track_id")
            subject["object_class"] = event.details.get("object_class", "bag")

        return subject

    def _update_subject_info(self, incident: Incident, event: Event, buffered_unknown: Optional[Event]) -> None:
        """Update subject dictionary with newly acquired identity or tracking information."""
        subj = incident.subject
        if event.track_id is not None and subj.get("track_id") is None:
            subj["track_id"] = event.track_id
        if event.person_id is not None and subj.get("person_id") is None:
            subj["person_id"] = event.person_id
        if event.object_id is not None and subj.get("object_id") is None:
            subj["object_id"] = event.object_id

        # Update identity status if resolved
        ev_id = event.details.get("identity_status")
        if ev_id and ev_id != "PENDING":
            subj["identity_status"] = ev_id
            if event.details.get("person_name"):
                subj["person_name"] = event.details.get("person_name")
            if event.details.get("reason"):
                subj["reason"] = event.details.get("reason")

        if buffered_unknown:
            subj["identity_status"] = "UNKNOWN"
            if buffered_unknown.details.get("reason"):
                subj["reason"] = buffered_unknown.details.get("reason")

    def _build_keywords(self, incident: Incident) -> str:
        """Generate keywords string for FTS5 full-text indexing."""
        terms = [
            incident.incident_no,
            incident.camera_id,
            incident.place,
            incident.status,
            incident.threat_level,
            str(incident.threat_score),
            " ".join(incident.activities),
        ]
        if incident.subject:
            for k in ["person_name", "identity_status", "track_id", "object_id", "reason"]:
                val = incident.subject.get(k)
                if val:
                    terms.append(str(val))
        return " ".join(terms)
