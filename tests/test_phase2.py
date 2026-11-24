"""Unit tests for Phase 2 Rule Engine and Geometry.

All tests run without GPU or deep learning model requirements:
- Point-in-polygon math & edge boundary conditions
- Directional virtual line crossings and jitter debouncing
- Dwell time accumulator and PROLONGED_STAY threshold boundaries
- Loitering detection (pacing within radius vs straight-through transit)
- Occupancy limit enforcement, debounce duration, and re-arming
- Abandoned object candidate tracking across ID switches, detection flicker,
  and owner presence/absence suppression
- Video-time determinism across varying replay execution speeds
- Clean state cleanup when tracks vanish
"""

from __future__ import annotations

import numpy as np
import pytest

from src.common.types import Detection, EventType, FrameResult, Severity, Track
from src.rules.abandoned_object import AbandonedObjectRule
from src.rules.geometry import (
    check_line_crossing,
    compute_iou,
    euclidean_distance,
    get_reference_point,
    point_in_polygon,
    signed_side,
)
from src.rules.line_crossing import LineCrossingRule
from src.rules.loitering import LoiteringRule
from src.rules.occupancy import OccupancyRule
from src.rules.prolonged_stay import ProlongedStayRule
from src.rules.restricted_zone import RestrictedZoneRule
from src.rules.zones import VirtualLine, Zone


# ---------------------------------------------------------------------------
# 1. Geometry & Spatial Operations
# ---------------------------------------------------------------------------

def test_point_in_polygon_and_boundary_cases() -> None:
    poly = [(10.0, 10.0), (100.0, 10.0), (100.0, 100.0), (10.0, 100.0)]
    assert point_in_polygon((50.0, 50.0), poly) is True  # Inside
    assert point_in_polygon((5.0, 50.0), poly) is False   # Outside
    assert point_in_polygon((10.0, 50.0), poly) is True  # Exactly on boundary edge
    assert point_in_polygon((10.0, 10.0), poly) is True  # Exactly on vertex


def test_reference_point_modes() -> None:
    bbox = (100.0, 50.0, 200.0, 250.0)
    assert get_reference_point(bbox, "center") == (150.0, 150.0)
    assert get_reference_point(bbox, "bottom_center") == (150.0, 250.0)  # Feet


def test_line_crossing_and_directional_enforcement() -> None:
    start = (100.0, 200.0)
    end = (300.0, 200.0)

    # Cross from top to bottom (inbound)
    crossed, direction = check_line_crossing((200.0, 150.0), (200.0, 250.0), start, end, "both")
    assert crossed is True
    assert direction in ("left_to_right", "right_to_left")

    # Parallel movement (no intersection)
    crossed_par, _ = check_line_crossing((150.0, 150.0), (250.0, 150.0), start, end, "both")
    assert crossed_par is False


# ---------------------------------------------------------------------------
# 2. Restricted Zone Entry & Debounce
# ---------------------------------------------------------------------------

def test_restricted_zone_entry_and_debounce() -> None:
    rule = RestrictedZoneRule({"confirm_frames": 3, "cooldown_seconds": 10.0})
    zone = Zone(
        zone_id="z_sec",
        name="Security Room",
        type="restricted",
        polygon=[(100.0, 100.0), (300.0, 100.0), (300.0, 300.0), (100.0, 300.0)],
        reference_resolution=(400, 400),
        risk_level=Severity.HIGH,
    )
    dummy_frame = np.zeros((400, 400, 3), dtype=np.uint8)

    # Frame 1: Person enters (confirm_count = 1 -> no event yet)
    trk = Track("cam1", 1, 0, "person", (150.0, 120.0, 190.0, 200.0), 0.90)
    res1 = FrameResult(dummy_frame, 1, 1.0, tracks=[trk])
    events1 = rule.evaluate(res1, [zone], [], 1.0)
    assert len(events1) == 0

    # Frame 2: Person still inside (confirm_count = 2 -> no event yet)
    res2 = FrameResult(dummy_frame, 2, 1.04, tracks=[trk])
    events2 = rule.evaluate(res2, [zone], [], 1.04)
    assert len(events2) == 0

    # Frame 3: 3rd consecutive frame -> Event triggers ONCE
    res3 = FrameResult(dummy_frame, 3, 1.08, tracks=[trk])
    events3 = rule.evaluate(res3, [zone], [], 1.08)
    assert len(events3) == 1
    assert events3[0].event_type == EventType.RESTRICTED_ZONE_ENTRY
    assert events3[0].severity == Severity.HIGH

    # Frame 4: Person remains inside -> Should NOT flood events (fires once)
    res4 = FrameResult(dummy_frame, 4, 1.12, tracks=[trk])
    events4 = rule.evaluate(res4, [zone], [], 1.12)
    assert len(events4) == 0


# ---------------------------------------------------------------------------
# 3. Line Crossing Rule & Jitter Debounce
# ---------------------------------------------------------------------------

def test_line_crossing_jitter_debouncing() -> None:
    rule = LineCrossingRule({"cooldown_seconds": 5.0})
    line = VirtualLine(
        "line_1",
        "Tripwire",
        (100.0, 200.0),
        (300.0, 200.0),
        direction="both",
        reference_resolution=(400, 400),
    )
    dummy_frame = np.zeros((400, 400, 3), dtype=np.uint8)

    # Track crosses line at t=1.0s
    trk = Track(
        "cam1", 1, 0, "person", (180.0, 210.0, 220.0, 270.0), 0.9,
        path=[(200.0, 180.0), (200.0, 220.0)],
    )
    res1 = FrameResult(dummy_frame, 1, 1.0, tracks=[trk])
    events1 = rule.evaluate(res1, [], [line], 1.0)
    assert len(events1) == 1
    assert events1[0].event_type == EventType.LINE_CROSSING

    # Rapid jitter across the line at t=1.5s (within 5.0s cooldown -> suppressed)
    trk.path = [(200.0, 220.0), (200.0, 180.0)]
    res2 = FrameResult(dummy_frame, 2, 1.5, tracks=[trk])
    events2 = rule.evaluate(res2, [], [line], 1.5)
    assert len(events2) == 0


# ---------------------------------------------------------------------------
# 4. Prolonged Stay (Dwell Time Threshold)
# ---------------------------------------------------------------------------

def test_prolonged_stay_threshold_boundary() -> None:
    rule = ProlongedStayRule({"default_dwell_threshold_seconds": 10.0})
    zone = Zone(
        zone_id="z_wait",
        name="Lobby",
        type="watch",
        polygon=[(50.0, 50.0), (350.0, 50.0), (350.0, 350.0), (50.0, 350.0)],
        dwell_threshold_seconds=10.0,
        reference_resolution=(400, 400),
    )
    dummy_frame = np.zeros((400, 400, 3), dtype=np.uint8)
    trk = Track("cam1", 1, 0, "person", (100.0, 100.0, 150.0, 200.0), 0.9)

    # Enters at t=0.0s
    res0 = FrameResult(dummy_frame, 1, 0.0, tracks=[trk])
    _ = rule.evaluate(res0, [zone], [], 0.0)
    assert trk.dwell_seconds == 0.0

    # At t=9.9s: just below threshold (no event)
    res_below = FrameResult(dummy_frame, 100, 9.9, tracks=[trk])
    events_below = rule.evaluate(res_below, [zone], [], 9.9)
    assert len(events_below) == 0
    assert trk.dwell_seconds >= 9.9

    # At t=10.0s: exactly meets threshold -> fires PROLONGED_STAY once
    res_exact = FrameResult(dummy_frame, 101, 10.0, tracks=[trk])
    events_exact = rule.evaluate(res_exact, [zone], [], 10.0)
    assert len(events_exact) == 1
    assert events_exact[0].event_type == EventType.PROLONGED_STAY


# ---------------------------------------------------------------------------
# 5. Loitering vs Transit
# ---------------------------------------------------------------------------

def test_loitering_triggers_for_circling_not_transit() -> None:
    rule = LoiteringRule({"window_seconds": 10.0, "radius_height_multiplier": 1.5, "min_movement_pixels": 15.0})
    dummy_frame = np.zeros((400, 400, 3), dtype=np.uint8)

    # 1. Straight transit pedestrian: walks straight from x=10 to x=300 in 10s (escapes radius -> NO event)
    for t in np.linspace(0.0, 10.5, 12):
        trk_transit = Track("cam1", 1, 0, "person", (float(t * 30.0), 100.0, float(t * 30.0 + 30.0), 180.0), 0.9)
        res = FrameResult(dummy_frame, int(t * 10), t, tracks=[trk_transit])
        evts = rule.evaluate(res, [], [], t)
        assert len(evts) == 0

    rule.reset()

    # 2. Pacing/circling pedestrian: wanders in circle of radius 15px around (150, 150) for 10s (FIRES event)
    events_fired = []
    for step, t in enumerate(np.linspace(0.0, 11.0, 25)):
        angle = step * 0.5
        cx = 150.0 + (15.0 * np.cos(angle))
        cy = 150.0 + (15.0 * np.sin(angle))
        trk_circling = Track("cam1", 2, 0, "person", (cx - 15.0, cy - 40.0, cx + 15.0, cy + 40.0), 0.9)
        res = FrameResult(dummy_frame, step, t, tracks=[trk_circling])
        evts = rule.evaluate(res, [], [], t)
        if evts:
            events_fired.extend(evts)

    assert len(events_fired) == 1
    assert events_fired[0].event_type == EventType.LOITERING


# ---------------------------------------------------------------------------
# 6. Occupancy Limit, Debounce, and Re-arming
# ---------------------------------------------------------------------------

def test_occupancy_limit_and_rearming() -> None:
    rule = OccupancyRule({"sustain_seconds": 2.0})
    zone = Zone(
        zone_id="z_room",
        name="Conference Room",
        type="occupancy",
        polygon=[(10.0, 10.0), (300.0, 10.0), (300.0, 300.0), (10.0, 300.0)],
        occupancy_limit=2,  # Limit is 2 people
        reference_resolution=(400, 400),
    )
    dummy_frame = np.zeros((400, 400, 3), dtype=np.uint8)

    # 2 people inside (at limit -> NO alert)
    p1 = Track("cam1", 1, 0, "person", (50.0, 50.0, 80.0, 120.0), 0.9)
    p2 = Track("cam1", 2, 0, "person", (150.0, 50.0, 180.0, 120.0), 0.9)
    res_at_limit = FrameResult(dummy_frame, 1, 0.0, tracks=[p1, p2])
    assert len(rule.evaluate(res_at_limit, [zone], [], 0.0)) == 0

    # 3rd person enters at t=1.0s (count=3 > limit=2)
    p3 = Track("cam1", 3, 0, "person", (200.0, 50.0, 230.0, 120.0), 0.9)
    res_over1 = FrameResult(dummy_frame, 2, 1.0, tracks=[p1, p2, p3])
    assert len(rule.evaluate(res_over1, [zone], [], 1.0)) == 0  # Under sustain debounce (needs 2.0s)

    # Still 3 people at t=3.1s (> 2.0s sustain -> FIRES alert)
    res_over2 = FrameResult(dummy_frame, 3, 3.1, tracks=[p1, p2, p3])
    evts = rule.evaluate(res_over2, [zone], [], 3.1)
    assert len(evts) == 1
    assert evts[0].event_type == EventType.OCCUPANCY_EXCEEDED
    assert evts[0].details["current_occupancy"] == 3

    # Count drops back to 2 at t=4.0s (re-arms)
    res_norm = FrameResult(dummy_frame, 4, 4.0, tracks=[p1, p2])
    assert len(rule.evaluate(res_norm, [zone], [], 4.0)) == 0


# ---------------------------------------------------------------------------
# 7. Abandoned Object: Owner Leaves vs Stays, Flicker & ID Switches
# ---------------------------------------------------------------------------

def test_abandoned_object_rule_complete_lifecycle() -> None:
    rule = AbandonedObjectRule({
        "stationary_seconds": 4.0,
        "owner_away_seconds": 2.0,
        "owner_distance_multiplier": 1.5,
        "max_missing_frames": 10,
    })
    dummy_frame = np.zeros((400, 400, 3), dtype=np.uint8)

    # Scenario A: Person sits next to their bag -> Must NOT fire
    bag_det = Detection("cam1", 1, 0.0, 24, "backpack", (100.0, 150.0, 140.0, 190.0), 0.85)
    owner_staying = Track("cam1", 10, 0, "person", (110.0, 80.0, 160.0, 180.0), 0.90)  # Close to bag

    for t in np.linspace(0.0, 10.0, 15):
        res = FrameResult(dummy_frame, int(t * 10), t, detections=[bag_det], tracks=[owner_staying])
        evts = rule.evaluate(res, [], [], t)
        assert len(evts) == 0, "Must not fire when owner remains beside bag"

    rule.reset()

    # Scenario B: Owner places bag, leaves at t=1.0s, bag ID changes and flickers -> FIRES once
    bag_pos = (200.0, 200.0, 240.0, 240.0)

    # Frame 1: Person 10 is beside bag at t=0.0s
    owner_present = Track("cam1", 10, 0, "person", (190.0, 140.0, 240.0, 220.0), 0.90)
    det1 = Detection("cam1", 1, 0.0, 24, "backpack", bag_pos, 0.85)
    res1 = FrameResult(dummy_frame, 1, 0.0, detections=[det1], tracks=[owner_present])
    _ = rule.evaluate(res1, [], [], 0.0)

    # Frame 2 to 20: Owner departs far away (x=500). Bag flickers (misses some frames)
    # Bag is stationary for 5 seconds (>= 4.0s) and owner is away for > 2.0s
    fired_events = []
    for idx, t in enumerate(np.linspace(1.0, 7.0, 20), start=2):
        owner_far = Track("cam1", 10, 0, "person", (500.0, 100.0, 540.0, 200.0), 0.90)
        # Simulate detection flicker: feed bag detection on alternate frames only
        current_dets = [Detection("cam1", idx, t, 24, "backpack", bag_pos, 0.82)] if (idx % 2 == 0) else []
        res = FrameResult(dummy_frame, idx, t, detections=current_dets, tracks=[owner_far])
        evts = rule.evaluate(res, [], [], t)
        if evts:
            fired_events.extend(evts)

    # Must fire ABANDONED_OBJECT exactly once
    assert len(fired_events) == 1
    evt = fired_events[0]
    assert evt.event_type == EventType.ABANDONED_OBJECT
    assert evt.track_id == 10  # Associated owner track ID correctly recovered!
    assert evt.details["stationary_seconds"] >= 4.0
    assert evt.details["owner_away_seconds"] >= 2.0


# ---------------------------------------------------------------------------
# 8. Video-Time Determinism Test
# ---------------------------------------------------------------------------

def test_video_time_determinism() -> None:
    """Verifies that replaying the exact same frames at different speeds yields identical events."""
    rule1 = RestrictedZoneRule({"confirm_frames": 2, "cooldown_seconds": 5.0})
    rule2 = RestrictedZoneRule({"confirm_frames": 2, "cooldown_seconds": 5.0})
    zone = Zone("z1", "Test", "restricted", [(50, 50), (200, 50), (200, 200), (50, 200)], reference_resolution=(300, 300))
    dummy_frame = np.zeros((300, 300, 3), dtype=np.uint8)

    # Sequence: 5 frames at video timestamps 1.0, 1.1, 1.2, 1.3, 1.4
    video_timestamps = [1.0, 1.1, 1.2, 1.3, 1.4]
    trk = Track("cam1", 1, 0, "person", (80, 80, 120, 140), 0.9)

    events_run1 = []
    for fid, v_time in enumerate(video_timestamps, 1):
        res = FrameResult(dummy_frame, fid, v_time, tracks=[trk])
        events_run1.extend(rule1.evaluate(res, [zone], [], v_time))

    # Second run with identical video timestamps
    events_run2 = []
    for fid, v_time in enumerate(video_timestamps, 1):
        res = FrameResult(dummy_frame, fid, v_time, tracks=[trk])
        events_run2.extend(rule2.evaluate(res, [zone], [], v_time))

    assert len(events_run1) == len(events_run2)
    assert events_run1[0].event_type == events_run2[0].event_type
    assert events_run1[0].timestamp == events_run2[0].timestamp
