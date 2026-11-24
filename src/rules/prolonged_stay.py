"""Prolonged Stay (Dwell Time) Rule.

Fires PROLONGED_STAY when an individual's accumulated dwell duration inside
a designated spatial zone exceeds the zone's dwell_threshold_seconds.
Populates Track.dwell_seconds and Track.zone_ids for downstream pipeline stages.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from src.common.types import Event, EventType, FrameResult, Severity, Track
from src.rules.base_rule import BaseRule
from src.rules.geometry import get_reference_point, point_in_polygon
from src.rules.zones import VirtualLine, Zone


class ProlongedStayRule(BaseRule):
    """Monitors dwell durations per zone and fires PROLONGED_STAY once per occurrence."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        super().__init__("prolonged_stay", cfg)
        self.ref_point_mode: str = cfg.get("reference_point", "bottom_center")
        self.default_dwell_threshold: float = float(cfg.get("default_dwell_threshold_seconds", 30.0))

        # (track_id, zone_id) -> entry timestamp
        self._entry_times: Dict[Tuple[int, str], float] = {}
        # (track_id, zone_id) -> last observed timestamp
        self._last_seen: Dict[Tuple[int, str], float] = {}
        # (track_id, zone_id) -> whether event has already fired for this stay
        self._fired_stays: Dict[Tuple[int, str], bool] = {}

    def evaluate(
        self,
        frame_result: FrameResult,
        zones: List[Zone],
        lines: List[VirtualLine],
        current_time: float,
    ) -> List[Event]:
        events: List[Event] = []
        if not zones:
            return events

        h, w = frame_result.frame.shape[:2]
        current_active_keys: Set[Tuple[int, str]] = set()

        for trk in frame_result.tracks:
            if trk.class_name.lower() != "person":
                continue

            ref_pt = get_reference_point(trk.bbox, mode=self.ref_point_mode)
            active_zones_for_track: List[str] = []
            max_dwell_for_track: float = 0.0

            for zone in zones:
                scaled_poly = zone.get_scaled_polygon(w, h)
                is_inside = point_in_polygon(ref_pt, scaled_poly)
                key = (trk.track_id, zone.zone_id)

                if is_inside:
                    current_active_keys.add(key)
                    active_zones_for_track.append(zone.zone_id)

                    # Initialize entry time if first observation inside zone
                    if key not in self._entry_times:
                        self._entry_times[key] = current_time

                    self._last_seen[key] = current_time
                    dwell_duration = max(0.0, current_time - self._entry_times[key])
                    max_dwell_for_track = max(max_dwell_for_track, dwell_duration)

                    dwell_limit = zone.dwell_threshold_seconds or self.default_dwell_threshold

                    # Check if dwell threshold is exceeded
                    if dwell_duration >= dwell_limit:
                        if not self._fired_stays.get(key, False):
                            evt = Event(
                                event_id=self.generate_event_id(EventType.PROLONGED_STAY, current_time),
                                camera_id=trk.camera_id,
                                track_id=trk.track_id,
                                event_type=EventType.PROLONGED_STAY,
                                timestamp=current_time,
                                severity=zone.risk_level or Severity.MEDIUM,
                                confidence=trk.confidence,
                                location={
                                    "zone_id": zone.zone_id,
                                    "zone_name": zone.name,
                                    "coordinates": (round(ref_pt[0], 1), round(ref_pt[1], 1)),
                                },
                                details={
                                    "dwell_seconds": round(dwell_duration, 1),
                                    "threshold_seconds": round(dwell_limit, 1),
                                    "zone_type": zone.type,
                                },
                            )
                            events.append(evt)
                            self._fired_stays[key] = True
                else:
                    # Clean up if person exited this specific zone
                    if key in self._entry_times:
                        self._entry_times.pop(key, None)
                        self._last_seen.pop(key, None)
                        self._fired_stays.pop(key, None)

            # Update Track object fields for downstream consumers
            trk.zone_ids = active_zones_for_track
            trk.dwell_seconds = max_dwell_for_track

        # Clean up tracks that vanished from the scene
        stale_keys = [k for k in self._entry_times.keys() if k not in current_active_keys]
        for k in stale_keys:
            self._entry_times.pop(k, None)
            self._last_seen.pop(k, None)
            self._fired_stays.pop(k, None)

        return events

    def reset(self) -> None:
        self._entry_times.clear()
        self._last_seen.clear()
        self._fired_stays.clear()
        self._event_counter = 0
