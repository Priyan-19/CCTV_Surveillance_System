"""Occupancy Limit Monitoring Rule.

Fires OCCUPANCY_EXCEEDED when the number of individuals within an
occupancy zone exceeds the zone's occupancy_limit for a sustained duration.
Re-arms only after the occupancy returns to or below the allowed threshold.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from src.common.types import Event, EventType, FrameResult, Severity, Track
from src.rules.base_rule import BaseRule
from src.rules.geometry import get_reference_point, point_in_polygon
from src.rules.zones import VirtualLine, Zone


class OccupancyRule(BaseRule):
    """Enforces maximum capacity limits within defined spatial zones."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        super().__init__("occupancy", cfg)
        self.ref_point_mode: str = cfg.get("reference_point", "bottom_center")
        self.sustain_seconds: float = float(cfg.get("sustain_seconds", 1.5))
        self.drop_tolerance_seconds: float = float(cfg.get("drop_tolerance_seconds", 0.5))

        # zone_id -> timestamp when excess occupancy started
        self._excess_start_time: Dict[str, float] = {}
        # zone_id -> timestamp when count dropped to/below limit (for debounce reset)
        self._drop_start_time: Dict[str, float] = {}
        # zone_id -> whether alert has been fired for current excess period
        self._fired_state: Dict[str, bool] = {}

    def evaluate(
        self,
        frame_result: FrameResult,
        zones: List[Zone],
        lines: List[VirtualLine],
        current_time: float,
    ) -> List[Event]:
        events: List[Event] = []
        occupancy_zones = [z for z in zones if z.type.lower() == "occupancy" or z.occupancy_limit > 0]
        if not occupancy_zones:
            return events

        h, w = frame_result.frame.shape[:2]
        person_tracks = [t for t in frame_result.tracks if t.class_name.lower() == "person"]

        for zone in occupancy_zones:
            scaled_poly = zone.get_scaled_polygon(w, h)
            # Count people inside this zone
            count = 0
            for p in person_tracks:
                ref_pt = get_reference_point(p.bbox, mode=self.ref_point_mode)
                if point_in_polygon(ref_pt, scaled_poly):
                    count += 1

            limit = zone.occupancy_limit

            if count > limit:
                # Excess observed -> clear any pending drop reset
                self._drop_start_time.pop(zone.zone_id, None)

                if zone.zone_id not in self._excess_start_time:
                    self._excess_start_time[zone.zone_id] = current_time

                excess_duration = max(0.0, current_time - self._excess_start_time[zone.zone_id])

                # Check if sustained beyond debounce duration
                if excess_duration >= self.sustain_seconds:
                    if not self._fired_state.get(zone.zone_id, False):
                        evt = Event(
                            event_id=self.generate_event_id(EventType.OCCUPANCY_EXCEEDED, current_time),
                            camera_id=frame_result.tracks[0].camera_id if frame_result.tracks else "cam",
                            track_id=None,
                            event_type=EventType.OCCUPANCY_EXCEEDED,
                            timestamp=current_time,
                            severity=zone.risk_level or Severity.MEDIUM,
                            confidence=0.95,
                            location={
                                "zone_id": zone.zone_id,
                                "zone_name": zone.name,
                                "coordinates": (0.0, 0.0),
                            },
                            details={
                                "current_occupancy": count,
                                "occupancy_limit": limit,
                                "excess_count": count - limit,
                                "sustain_seconds": round(excess_duration, 1),
                            },
                        )
                        events.append(evt)
                        self._fired_state[zone.zone_id] = True
            else:
                # Occupancy returned to normal or dropped due to temporary occlusion
                if zone.zone_id in self._excess_start_time:
                    if zone.zone_id not in self._drop_start_time:
                        self._drop_start_time[zone.zone_id] = current_time

                    drop_duration = current_time - self._drop_start_time[zone.zone_id]
                    if drop_duration >= self.drop_tolerance_seconds:
                        self._excess_start_time.pop(zone.zone_id, None)
                        self._drop_start_time.pop(zone.zone_id, None)
                        self._fired_state.pop(zone.zone_id, None)

        return events

    def reset(self) -> None:
        self._excess_start_time.clear()
        self._drop_start_time.clear()
        self._fired_state.clear()
        self._event_counter = 0
