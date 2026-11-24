"""Rule-based Loitering Detection Rule.

Fires LOITERING when an individual stays within a small localized radius
(relative to their bounding box height) for at least T seconds over a sliding
window, while maintaining active movement (pacing/circling).

Distinct from PROLONGED_STAY:
- PROLONGED_STAY is strictly zone-anchored: it measures duration inside a fixed polygon.
- LOITERING is track-centric: it detects wandering/hovering anywhere in the scene,
  even where no administrative zone was configured.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

from src.common.types import Event, EventType, FrameResult, Severity, Track
from src.rules.base_rule import BaseRule
from src.rules.geometry import euclidean_distance, get_reference_point
from src.rules.zones import VirtualLine, Zone


class LoiteringRule(BaseRule):
    """Detects localized lingering or pacing within a floating spatial radius."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        super().__init__("loitering", cfg)
        self.ref_point_mode: str = cfg.get("reference_point", "bottom_center")
        self.window_seconds: float = float(cfg.get("window_seconds", 20.0))
        # Radius multiplier relative to person height (e.g. 1.5 * person_height)
        self.radius_height_multiplier: float = float(cfg.get("radius_height_multiplier", 1.5))
        # Minimum total movement distance in pixels to distinguish loitering from sitting still
        self.min_movement_pixels: float = float(cfg.get("min_movement_pixels", 20.0))
        self.cooldown_seconds: float = float(cfg.get("cooldown_seconds", 30.0))

        # track_id -> deque of (timestamp, centroid, bbox_height)
        self._history: Dict[int, Deque[Tuple[float, Tuple[float, float], float]]] = {}
        # track_id -> timestamp when loitering event was fired
        self._last_event_time: Dict[int, float] = {}

    def evaluate(
        self,
        frame_result: FrameResult,
        zones: List[Zone],
        lines: List[VirtualLine],
        current_time: float,
    ) -> List[Event]:
        events: List[Event] = []
        active_track_ids: Set[int] = set()

        for trk in frame_result.tracks:
            if trk.class_name.lower() != "person":
                continue

            active_track_ids.add(trk.track_id)
            ref_pt = get_reference_point(trk.bbox, mode=self.ref_point_mode)
            h = trk.height

            if trk.track_id not in self._history:
                self._history[trk.track_id] = deque()

            hist = self._history[trk.track_id]
            hist.append((current_time, ref_pt, h))

            # Prune points older than sliding window
            cutoff = current_time - self.window_seconds
            while hist and hist[0][0] < cutoff:
                hist.popleft()

            # Must span at least window_seconds to confirm loitering
            if not hist:
                continue

            earliest_time = hist[0][0]
            observed_window = current_time - earliest_time
            if observed_window < (self.window_seconds * 0.85):
                continue

            # Anchor point: centroid at the start of the window
            anchor_pt = hist[0][1]
            max_allowed_radius = max(40.0, h * self.radius_height_multiplier)

            # Check if all points within window remain within localized radius
            within_radius = True
            total_path_length = 0.0

            for i in range(len(hist)):
                dist_from_anchor = euclidean_distance(anchor_pt, hist[i][1])
                if dist_from_anchor > max_allowed_radius:
                    within_radius = False
                    break
                if i > 0:
                    total_path_length += euclidean_distance(hist[i - 1][1], hist[i][1])

            # Trigger condition: stayed within radius for window AND was actively moving
            if within_radius and total_path_length >= self.min_movement_pixels:
                last_fired = self._last_event_time.get(trk.track_id)
                if (last_fired is None) or ((current_time - last_fired) >= self.cooldown_seconds):
                    evt = Event(
                        event_id=self.generate_event_id(EventType.LOITERING, current_time),
                        camera_id=trk.camera_id,
                        track_id=trk.track_id,
                        event_type=EventType.LOITERING,
                        timestamp=current_time,
                        severity=Severity.MEDIUM,
                        confidence=trk.confidence,
                        location={
                            "zone_id": None,
                            "zone_name": "Open Area",
                            "coordinates": (round(ref_pt[0], 1), round(ref_pt[1], 1)),
                        },
                        details={
                            "loitering_duration_sec": round(observed_window, 1),
                            "max_displacement_px": round(max_allowed_radius, 1),
                            "total_pacing_distance_px": round(total_path_length, 1),
                        },
                    )
                    events.append(evt)
                    self._last_event_time[trk.track_id] = current_time

        # Cleanup disappeared tracks
        stale_ids = [tid for tid in self._history if tid not in active_track_ids]
        for tid in stale_ids:
            self._history.pop(tid, None)
            self._last_event_time.pop(tid, None)

        return events

    def reset(self) -> None:
        self._history.clear()
        self._last_event_time.clear()
        self._event_counter = 0
