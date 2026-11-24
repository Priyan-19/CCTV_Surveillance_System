"""Restricted Zone Entry Rule.

Fires RESTRICTED_ZONE_ENTRY when a person's reference point (feet)
enters a restricted zone polygon for at least the confirm duration.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from src.common.types import Event, EventType, FrameResult, Severity, Track
from src.rules.base_rule import BaseRule
from src.rules.geometry import get_reference_point, point_in_polygon
from src.rules.zones import VirtualLine, Zone


class RestrictedZoneRule(BaseRule):
    """Detects unauthorized entry into designated restricted perimeter polygons."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        super().__init__("restricted_zone", cfg)
        self.ref_point_mode: str = cfg.get("reference_point", "bottom_center")
        self.confirm_frames: int = int(cfg.get("confirm_frames", 5))
        self.cooldown_seconds: float = float(cfg.get("cooldown_seconds", 10.0))

        # Track State Tracking:
        # (track_id, zone_id) -> consecutive frames inside
        self._inside_frames: Dict[Tuple[int, str], int] = {}
        # (track_id, zone_id) -> timestamp of last fired event
        self._last_event_time: Dict[Tuple[int, str], float] = {}
        # (track_id, zone_id) -> whether currently in triggered state
        self._triggered: Dict[Tuple[int, str], bool] = {}

    def evaluate(
        self,
        frame_result: FrameResult,
        zones: List[Zone],
        lines: List[VirtualLine],
        current_time: float,
    ) -> List[Event]:
        events: List[Event] = []
        restricted_zones = [z for z in zones if z.type.lower() == "restricted"]
        if not restricted_zones:
            return events

        h, w = frame_result.frame.shape[:2]
        current_active_keys: Set[Tuple[int, str]] = set()

        for trk in frame_result.tracks:
            if trk.class_name.lower() != "person":
                continue

            ref_pt = get_reference_point(trk.bbox, mode=self.ref_point_mode)

            for zone in restricted_zones:
                scaled_poly = zone.get_scaled_polygon(w, h)
                is_inside = point_in_polygon(ref_pt, scaled_poly)
                key = (trk.track_id, zone.zone_id)

                if is_inside:
                    current_active_keys.add(key)
                    self._inside_frames[key] = self._inside_frames.get(key, 0) + 1

                    # Phase 3 Authorization Hook (default: empty / unauthorized)
                    is_authorized = False

                    # Check debouncing: must persist inside for confirm_frames
                    if self._inside_frames[key] >= self.confirm_frames and not is_authorized:
                        last_fired = self._last_event_time.get(key)
                        is_cooldown_expired = (last_fired is None) or ((current_time - last_fired) >= self.cooldown_seconds)

                        if not self._triggered.get(key, False) and is_cooldown_expired:
                            evt = Event(
                                event_id=self.generate_event_id(EventType.RESTRICTED_ZONE_ENTRY, current_time),
                                camera_id=frame_result.tracks[0].camera_id if frame_result.tracks else "cam",
                                track_id=trk.track_id,
                                event_type=EventType.RESTRICTED_ZONE_ENTRY,
                                timestamp=current_time,
                                severity=zone.risk_level or Severity.HIGH,
                                confidence=trk.confidence,
                                location={
                                    "zone_id": zone.zone_id,
                                    "zone_name": zone.name,
                                    "coordinates": (round(ref_pt[0], 1), round(ref_pt[1], 1)),
                                },
                                details={
                                    "person_bbox": [round(c, 1) for c in trk.bbox],
                                    "confirm_frames": self._inside_frames[key],
                                    "reference_point": self.ref_point_mode,
                                },
                            )
                            events.append(evt)
                            self._triggered[key] = True
                            self._last_event_time[key] = current_time
                else:
                    # Reset if outside zone
                    if key in self._inside_frames:
                        self._inside_frames[key] = 0
                    self._triggered[key] = False

        # Clean up stale tracks no longer present in frame
        stale_keys = [k for k in self._inside_frames.keys() if k not in current_active_keys]
        for k in stale_keys:
            self._inside_frames.pop(k, None)
            self._triggered.pop(k, None)

        return events

    def reset(self) -> None:
        self._inside_frames.clear()
        self._last_event_time.clear()
        self._triggered.clear()
        self._event_counter = 0
