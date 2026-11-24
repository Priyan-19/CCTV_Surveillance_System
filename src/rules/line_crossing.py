"""Line Crossing Virtual Tripwire Rule.

Fires LINE_CROSSING when an object's trajectory intersects a virtual tripwire
in a configured forbidden direction (both, a_to_b, b_to_a, inbound, outbound).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from src.common.types import Event, EventType, FrameResult, Severity, Track
from src.rules.base_rule import BaseRule
from src.rules.geometry import check_line_crossing, get_reference_point
from src.rules.zones import VirtualLine, Zone


class LineCrossingRule(BaseRule):
    """Detects movement across virtual lines and enforces directionality and jitter debouncing."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        super().__init__("line_crossing", cfg)
        self.ref_point_mode: str = cfg.get("reference_point", "bottom_center")
        self.cooldown_seconds: float = float(cfg.get("cooldown_seconds", 10.0))

        # (track_id, line_id) -> timestamp of last crossing
        self._last_crossed: Dict[Tuple[int, str], float] = {}

    def evaluate(
        self,
        frame_result: FrameResult,
        zones: List[Zone],
        lines: List[VirtualLine],
        current_time: float,
    ) -> List[Event]:
        events: List[Event] = []
        if not lines:
            return events

        h, w = frame_result.frame.shape[:2]

        for trk in frame_result.tracks:
            # Need at least 2 historical positions to compute motion vector
            if len(trk.path) < 2:
                continue

            prev_pt = trk.path[-2]
            curr_pt = trk.path[-1]

            for line in lines:
                scaled_start, scaled_end = line.get_scaled_line(w, h)
                crossed, direction_taken = check_line_crossing(
                    prev_pt=prev_pt,
                    curr_pt=curr_pt,
                    line_start=scaled_start,
                    line_end=scaled_end,
                    direction=line.direction,
                )

                if crossed:
                    key = (trk.track_id, line.line_id)
                    last_time = self._last_crossed.get(key)

                    # Debounce: suppress rapid back-and-forth jitter within cooldown
                    if (last_time is None) or ((current_time - last_time) >= self.cooldown_seconds):
                        evt = Event(
                            event_id=self.generate_event_id(EventType.LINE_CROSSING, current_time),
                            camera_id=trk.camera_id,
                            track_id=trk.track_id,
                            event_type=EventType.LINE_CROSSING,
                            timestamp=current_time,
                            severity=line.risk_level or Severity.MEDIUM,
                            confidence=trk.confidence,
                            location={
                                "line_id": line.line_id,
                                "line_name": line.name,
                                "coordinates": (round(curr_pt[0], 1), round(curr_pt[1], 1)),
                            },
                            details={
                                "class_name": trk.class_name,
                                "direction_taken": direction_taken,
                                "configured_direction": line.direction,
                                "from_point": (round(prev_pt[0], 1), round(prev_pt[1], 1)),
                                "to_point": (round(curr_pt[0], 1), round(curr_pt[1], 1)),
                            },
                        )
                        events.append(evt)
                        self._last_crossed[key] = current_time

        return events

    def reset(self) -> None:
        self._last_crossed.clear()
        self._event_counter = 0
