"""Unit tests for Phase 4 Threat Engine.

Covers functional tests T-18 through T-22, T-24 from docs/10_TEST_PLAN.md:
- T-18: Single low event (logged only, no alert).
- T-19: Server room multi-event worked example (Critical, score 100).
- T-20: Unrelated events (> 120s apart) create separate incidents.
- T-21: Repeated event modifier (+25% once) and incident update without duplicate.
- T-22: Known authorized person deduction (-20) and restricted zone protection.
- Stratified unknown modifiers (+15 verified vs +5 overhead camera vs 0 pending).
- Bag-owner linkage correlation.
- Alert deduplication and escalation rules.
- Quiet-period incident closure (300s).
- Deterministic scoring & clamping boundaries [0, 100].
- SQLite repository persistence integration.
"""

import tempfile
from pathlib import Path
import pytest

from src.common.types import (
    Event,
    EventType,
    IdentityStatus,
    Incident,
    IncidentStatus,
    Severity,
)
from src.storage.repository import StorageRepository
from src.threat.alert_dispatcher import AlertDispatcher
from src.threat.correlator import SingleCameraCorrelator
from src.threat.incident_manager import IncidentManager
from src.threat.normalizer import EventNormalizer
from src.threat.scorer import ThreatScorer


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------

def make_event(
    event_type: EventType,
    timestamp: float = 10.0,
    camera_id: str = "CAM-01",
    track_id: int = 1,
    person_id: int = None,
    object_id: str = None,
    zone_id: str = "zone_main",
    high_risk: bool = False,
    authorized: bool = False,
    details: dict = None,
) -> Event:
    """Helper to construct validated Event instances for testing."""
    return Event(
        event_id=f"EVT-{int(timestamp*1000)}-{track_id or 0}",
        camera_id=camera_id,
        track_id=track_id,
        event_type=event_type,
        timestamp=timestamp,
        severity=Severity.LOW,
        confidence=0.95,
        location={
            "zone_id": zone_id,
            "zone_name": zone_id.replace("_", " ").title(),
            "coordinates": (100.0, 150.0),
            "high_risk": high_risk,
            "authorized": authorized,
        },
        details=details or {},
        person_id=person_id,
        object_id=object_id,
    )


# ---------------------------------------------------------------------------
# T-18: Single Low Event
# ---------------------------------------------------------------------------

def test_t18_single_low_event_logged_no_alert():
    """T-18: Single low-weight event (e.g. RUNNING = 10) logs incident with Low severity and suppresses alerts."""
    manager = IncidentManager()
    evt = make_event(EventType.RUNNING, timestamp=5.0, track_id=42)

    incident, alert = manager.process_event(evt)

    assert incident is not None
    assert incident.threat_score == 10
    assert incident.threat_level == Severity.LOW.value
    assert incident.is_open is True
    # Alerts are suppressed for Low level incidents
    assert alert is None


# ---------------------------------------------------------------------------
# T-19: Server Room Worked Example (docs/05 Section 5.5)
# ---------------------------------------------------------------------------

def test_t19_server_room_worked_example_critical_100():
    """T-19: Worked example from docs/05 Section 5.5:

    Unknown person enters server room (RESTRICTED_ZONE_ENTRY: 40),
    stays (PROLONGED_STAY: 15), leaves a bag (ABANDONED_OBJECT: 30),
    with verified unknown modifier (+15) and 3+ distinct events modifier (+10).
    Expected: Base 85 + 25 = 110 -> clamped to 100 (Critical), alert dispatched.
    """
    manager = IncidentManager()

    # 1. Unknown person context
    unknown_evt = make_event(
        EventType.UNKNOWN_PERSON,
        timestamp=10.0,
        track_id=7,
        details={"identity_status": "UNKNOWN", "reason": "no_gallery_match"},
    )
    manager.process_event(unknown_evt)

    # 2. Enters server room (RESTRICTED_ZONE_ENTRY: 40)
    e1 = make_event(
        EventType.RESTRICTED_ZONE_ENTRY,
        timestamp=12.0,
        track_id=7,
        zone_id="server_room",
        details={"identity_status": "UNKNOWN", "reason": "no_gallery_match"},
    )
    inc, alert1 = manager.process_event(e1)
    assert inc is not None
    # Base: 40 + Unknown: 15 = 55 (Medium)
    assert inc.threat_score == 55
    assert inc.threat_level == Severity.MEDIUM.value
    assert alert1 is not None

    # 3. Prolonged stay (PROLONGED_STAY: 15)
    e2 = make_event(
        EventType.PROLONGED_STAY,
        timestamp=35.0,
        track_id=7,
        zone_id="server_room",
        details={"identity_status": "UNKNOWN", "reason": "no_gallery_match"},
    )
    inc, alert2 = manager.process_event(e2)
    # Base: 40 + 15 = 55 + Unknown 15 = 70 (High)
    assert inc.threat_score == 70
    assert inc.threat_level == Severity.HIGH.value
    assert alert2 is not None  # Escalated from Medium -> High

    # 4. Abandoned object (ABANDONED_OBJECT: 30) linked to owner
    e3 = make_event(
        EventType.ABANDONED_OBJECT,
        timestamp=60.0,
        track_id=None,
        object_id="bag_101",
        zone_id="server_room",
        details={"owner_track_id": 7, "identity_status": "UNKNOWN", "reason": "no_gallery_match"},
    )
    inc, alert3 = manager.process_event(e3)

    # Total: Base (40+15+30=85) + Unknown (15) + Distinct 3+ types (10) = 110 -> 100
    assert inc.threat_score == 100
    assert inc.threat_level == Severity.CRITICAL.value
    assert len(inc.events) >= 3
    assert alert3 is not None  # Escalated from High -> Critical
    assert alert3.threat_level == Severity.CRITICAL.value
    assert alert3.threat_score == 100


# ---------------------------------------------------------------------------
# T-20: Unrelated Events Far Apart
# ---------------------------------------------------------------------------

def test_t20_unrelated_events_two_incidents():
    """T-20: Two unrelated events beyond correlation window (120s) create separate incidents."""
    manager = IncidentManager()

    # Event 1 at t=10s
    e1 = make_event(EventType.LINE_CROSSING, timestamp=10.0, track_id=1, camera_id="CAM-01")
    inc1, _ = manager.process_event(e1)

    # Event 2 at t=200s (> 120s from t=10s)
    e2 = make_event(EventType.LINE_CROSSING, timestamp=200.0, track_id=2, camera_id="CAM-01")
    inc2, _ = manager.process_event(e2)

    assert inc1.incident_no != inc2.incident_no
    assert inc1.incident_no == "INC-000001"
    assert inc2.incident_no == "INC-000002"
    assert len(manager.open_incidents) == 2


# ---------------------------------------------------------------------------
# T-21: Repeated Event Modifier (Clarification a)
# ---------------------------------------------------------------------------

def test_t21_repeated_event_modifier_applied_once():
    """T-21: Same person repeats an event within correlation window:

    Incident is updated (not duplicated).
    Clarification (a): +25% of highest-weight repeated event is applied ONCE,
    not compounding on subsequent repetitions.
    """
    manager = IncidentManager()

    # Event 1: RESTRICTED_ZONE_ENTRY (weight 40) at t=10s
    e1 = make_event(EventType.RESTRICTED_ZONE_ENTRY, timestamp=10.0, track_id=5)
    inc, _ = manager.process_event(e1)
    assert inc.threat_score == 40  # Base 40, no repeat yet

    # Event 2: Same person repeats RESTRICTED_ZONE_ENTRY at t=25s
    e2 = make_event(EventType.RESTRICTED_ZONE_ENTRY, timestamp=25.0, track_id=5)
    inc, _ = manager.process_event(e2)
    # Base: 40 (unique) + Repeated: 25% of 40 = +10 -> Total: 50
    assert inc.threat_score == 50
    assert inc.incident_no == "INC-000001"
    assert len(inc.events) == 2

    # Event 3: Third repeat at t=40s
    e3 = make_event(EventType.RESTRICTED_ZONE_ENTRY, timestamp=40.0, track_id=5)
    inc, _ = manager.process_event(e3)
    # Repeated modifier is applied ONCE per incident, so score remains 50
    assert inc.threat_score == 50
    assert len(inc.events) == 3


# ---------------------------------------------------------------------------
# T-22: Known Authorized Person Deduction (Clarification c)
# ---------------------------------------------------------------------------

def test_t22_known_authorized_bonus_and_restricted_exclusion():
    """T-22: Known authorized person receives -20 bonus when all events are in authorized zone.

    If a restricted zone entry occurs, the authorized bonus is voided.
    """
    manager = IncidentManager()

    # Case A: Known person loitering in authorized staff lobby (weight 20)
    e1 = make_event(
        EventType.LOITERING,
        timestamp=10.0,
        track_id=8,
        person_id=12,
        zone_id="staff_lobby",
        authorized=True,
        details={"identity_status": "KNOWN", "person_name": "Alice"},
    )
    inc, _ = manager.process_event(e1)
    # Base: 20 - Authorized bonus: 20 = 0 (Low)
    assert inc.threat_score == 0
    assert inc.threat_level == Severity.LOW.value

    # Case B: If known person intrudes into RESTRICTED_ZONE_ENTRY, authorized bonus does NOT apply
    manager2 = IncidentManager()
    e2 = make_event(
        EventType.RESTRICTED_ZONE_ENTRY,
        timestamp=15.0,
        track_id=8,
        person_id=12,
        zone_id="server_vault",
        authorized=False,
        details={"identity_status": "KNOWN", "person_name": "Alice"},
    )
    inc2, _ = manager2.process_event(e2)
    # Base 40, no -20 bonus -> 40
    assert inc2.threat_score == 40
    assert inc2.threat_level == Severity.MEDIUM.value


# ---------------------------------------------------------------------------
# Stratified Unknown Modifiers (Clarification b)
# ---------------------------------------------------------------------------

def test_stratified_unknown_modifiers():
    """Clarification (b):

    - Verified unknown (usable face compared against gallery, no match): +15
    - Unusable / blur / overhead wide camera face: +5
    - Pending: 0
    """
    scorer = ThreatScorer()

    # 1. Verified unknown (+15)
    inc_verified = Incident(
        events=[make_event(EventType.LINE_CROSSING, timestamp=10.0)],
        subject={"identity_status": "UNKNOWN", "reason": "no_gallery_match"},
    )
    expl_verified = scorer.score_incident(inc_verified)
    # Base: 20 + 15 = 35
    assert expl_verified.final_score == 35

    # 2. Unusable face overhead wide camera (+5)
    inc_overhead = Incident(
        events=[make_event(EventType.LINE_CROSSING, timestamp=10.0)],
        subject={"identity_status": "UNKNOWN", "reason": "overhead_low_quality_blur"},
    )
    expl_overhead = scorer.score_incident(inc_overhead)
    # Base: 20 + 5 = 25
    assert expl_overhead.final_score == 25

    # 3. Pending (0)
    inc_pending = Incident(
        events=[make_event(EventType.LINE_CROSSING, timestamp=10.0)],
        subject={"identity_status": "PENDING"},
    )
    expl_pending = scorer.score_incident(inc_pending)
    # Base: 20 + 0 = 20
    assert expl_pending.final_score == 20


# ---------------------------------------------------------------------------
# Bag-Owner Linkage Correlation
# ---------------------------------------------------------------------------

def test_bag_owner_linkage_correlation():
    """An abandoned bag links to an incident involving the person who carried/owned it."""
    manager = IncidentManager()

    # Person track 9 loiters in waiting hall at t=10s
    e1 = make_event(EventType.LOITERING, timestamp=10.0, track_id=9, zone_id="waiting_hall")
    inc, _ = manager.process_event(e1)
    assert inc.incident_no == "INC-000001"

    # Bag left behind at t=30s with owner_track_id = 9
    e2 = make_event(
        EventType.ABANDONED_OBJECT,
        timestamp=30.0,
        track_id=None,
        object_id="bag_55",
        zone_id="waiting_hall",
        details={"owner_track_id": 9},
    )
    inc2, _ = manager.process_event(e2)

    # Must be linked to the same incident!
    assert inc2.incident_no == "INC-000001"
    assert len(inc2.events) == 2


# ---------------------------------------------------------------------------
# Alert Deduplication and Escalation
# ---------------------------------------------------------------------------

def test_alert_deduplication_and_escalation():
    """Alerts are dispatched once on creation (Medium+) and again ONLY when threat level increases."""
    manager = IncidentManager()

    # Event 1: Score 30 (Medium) -> Alert dispatched
    e1 = make_event(EventType.ABANDONED_OBJECT, timestamp=10.0, track_id=3)
    inc, alert1 = manager.process_event(e1)
    assert inc.threat_level == Severity.MEDIUM.value
    assert alert1 is not None

    # Event 2: Adds RUNNING (weight 10) -> Score becomes 40 (still Medium)
    e2 = make_event(EventType.RUNNING, timestamp=20.0, track_id=3)
    inc, alert2 = manager.process_event(e2)
    assert inc.threat_score == 40
    assert inc.threat_level == Severity.MEDIUM.value
    assert alert2 is None  # Suppressed! No level escalation

    # Event 3: Adds RESTRICTED_ZONE_ENTRY (weight 40) -> Score reaches 80+ (Critical)
    e3 = make_event(EventType.RESTRICTED_ZONE_ENTRY, timestamp=30.0, track_id=3)
    inc, alert3 = manager.process_event(e3)
    assert inc.threat_level == Severity.CRITICAL.value
    assert alert3 is not None  # Dispatched on escalation!


# ---------------------------------------------------------------------------
# Quiet Period Incident Closure (300s)
# ---------------------------------------------------------------------------

def test_quiet_period_closure():
    """Incidents with no new events for >= 300s are closed; subsequent events start a new incident."""
    manager = IncidentManager()

    # Event 1 at t=10s
    e1 = make_event(EventType.LINE_CROSSING, timestamp=10.0, track_id=15)
    inc1, _ = manager.process_event(e1)
    assert inc1.is_open is True

    # Event 2 at t=350s (340s after event 1 > 300s quiet period)
    e2 = make_event(EventType.LINE_CROSSING, timestamp=350.0, track_id=15)
    inc2, _ = manager.process_event(e2)

    # First incident should be closed, and a new incident opened
    assert inc1.is_open is False
    assert inc1.closed_at == 10.0
    assert inc2.incident_no == "INC-000002"
    assert inc2.is_open is True


# ---------------------------------------------------------------------------
# Clamping & Determinism
# ---------------------------------------------------------------------------

def test_score_clamping_and_determinism():
    """Scores clamp within [0, 100] and identical inputs produce deterministic output."""
    scorer = ThreatScorer()

    # Overflow clamp to 100
    overflow_inc = Incident(
        events=[
            make_event(EventType.FIGHTING, timestamp=1.0),            # 50
            make_event(EventType.RESTRICTED_ZONE_ENTRY, timestamp=2.0), # 40
            make_event(EventType.ABANDONED_OBJECT, timestamp=3.0),     # 30
        ],
        subject={"identity_status": "UNKNOWN", "reason": "verified_unknown"}, # +15
    )
    expl = scorer.score_incident(overflow_inc)
    assert expl.unclamped_score > 100
    assert expl.final_score == 100
    assert expl.threat_level == Severity.CRITICAL.value

    # Repeat scoring on identical incident: must match exactly
    expl2 = scorer.score_incident(overflow_inc)
    assert expl.final_score == expl2.final_score
    assert expl.explanation_lines == expl2.explanation_lines


# ---------------------------------------------------------------------------
# Database Persistence Integration
# ---------------------------------------------------------------------------

def test_database_persistence_integration():
    """StorageRepository persists incidents, events, and score history atomically."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        repo = StorageRepository(db_path)
        repo.register_camera("CAM-01", location="Zone Main")
        manager = IncidentManager(repository=repo)

        e1 = make_event(EventType.RESTRICTED_ZONE_ENTRY, timestamp=10.0, track_id=77)
        inc, _ = manager.process_event(e1)

        # Query database via repository
        saved_inc = repo.get_incident(inc.id)
        assert saved_inc is not None
        assert saved_inc.incident_no == "INC-000001"
        assert saved_inc.threat_score == 40
        assert len(saved_inc.score_history) == 1

        # Search via FTS5
        search_results = repo.search_incidents("RESTRICTED_ZONE_ENTRY")
        assert len(search_results) == 1
        assert search_results[0].incident_no == "INC-000001"

    finally:
        Path(db_path).unlink(missing_ok=True)
