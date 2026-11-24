"""Visual annotation module for Phase 2.

Renders:
- Anti-aliased bounding boxes, Track IDs, and motion history trails.
- Translucent spatial zone polygons with zone labels.
- Virtual tripwires with directional indicators.
- Live dwell timers on tracked individuals in zones.
- Static object candidate markers with stationary timers.
- Event banners displaying severity text and alert details (without threat scores).
- Real-time status HUD with stage latencies and occupancy stats.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np

from src.common.types import Event, Severity, Track
from src.rules.abandoned_object import StaticBagCandidate
from src.rules.zones import VirtualLine, Zone


class Phase2Annotator:
    """Renders comprehensive spatial zones, tripwires, tracks, and security events."""

    CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
        "person": (0, 255, 128),
        "backpack": (0, 215, 255),
        "handbag": (0, 215, 255),
        "suitcase": (0, 165, 255),
        "car": (255, 144, 30),
        "motorcycle": (255, 100, 0),
        "bus": (255, 191, 0),
        "truck": (255, 128, 0),
        "bicycle": (200, 255, 0),
    }

    SEVERITY_COLORS: Dict[Severity, Tuple[int, int, int]] = {
        Severity.LOW: (200, 200, 50),        # Cyan/Blue-Green
        Severity.MEDIUM: (0, 215, 255),     # Amber Gold
        Severity.HIGH: (0, 100, 255),       # Vibrant Orange
        Severity.CRITICAL: (0, 0, 255),     # Crimson Red
    }

    def __init__(
        self,
        camera_id: str,
        model_name: str,
        tracker_name: str = "ByteTrack",
        draw_trail: bool = True,
        max_trail_points: int = 30,
        banner_display_seconds: float = 3.5,
    ) -> None:
        self.camera_id = camera_id
        self.model_name = model_name
        self.tracker_name = tracker_name
        self.draw_trail = draw_trail
        self.max_trail_points = max_trail_points
        self.banner_display_seconds = banner_display_seconds

        # Active event banners to display: (timestamp, Event)
        self._active_banners: List[Tuple[float, Event]] = []

    def add_event(self, event: Event, current_time: float) -> None:
        """Register an event to trigger a visual alert banner."""
        self._active_banners.append((current_time, event))

    def _render_zones_and_lines(
        self,
        frame: np.ndarray,
        zones: List[Zone],
        lines: List[VirtualLine],
    ) -> np.ndarray:
        """Render translucent polygons and virtual tripwires."""
        h, w = frame.shape[:2]
        poly_overlay = frame.copy()

        # 1. Render Polygons
        for zone in zones:
            scaled_poly = zone.get_scaled_polygon(w, h)
            pts = np.array(scaled_poly, dtype=np.int32).reshape((-1, 1, 2))

            color = (0, 0, 255) if zone.type == "restricted" else (255, 200, 0)
            if zone.type == "occupancy":
                color = (255, 128, 0)

            # Fill polygon translucently
            cv2.fillPoly(poly_overlay, [pts], color)
            # Outline boundary
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

            # Label near first vertex
            lx, ly = int(scaled_poly[0][0]), max(15, int(scaled_poly[0][1]) - 6)
            tag = f"{zone.name} ({zone.type.upper()})"
            cv2.putText(frame, tag, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        cv2.addWeighted(poly_overlay, 0.20, frame, 0.80, 0, frame)

        # 2. Render Virtual Lines
        for line in lines:
            s_pt, e_pt = line.get_scaled_line(w, h)
            p1 = (int(s_pt[0]), int(s_pt[1]))
            p2 = (int(e_pt[0]), int(e_pt[1]))

            # Tripwire line
            cv2.line(frame, p1, p2, (0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
            # Directional mid-point indicator
            mid_x = (p1[0] + p2[0]) // 2
            mid_y = (p1[1] + p2[1]) // 2
            cv2.circle(frame, (mid_x, mid_y), 4, (0, 255, 255), -1)

            cv2.putText(
                frame,
                f"{line.name} [{line.direction}]",
                (mid_x + 5, mid_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return frame

    def annotate(
        self,
        frame: np.ndarray,
        tracks: Sequence[Track],
        zones: List[Zone],
        lines: List[VirtualLine],
        fps: float,
        latency_ms: float,
        rules_ms: float,
        frame_id: int,
        current_time: float,
        static_candidates: Optional[List[StaticBagCandidate]] = None,
    ) -> np.ndarray:
        """Annotate frame with zones, tripwires, tracks, timers, and alert banners."""
        out_frame = frame.copy()

        # 1. Spatial Zones & Tripwires
        out_frame = self._render_zones_and_lines(out_frame, zones, lines)

        people_count = 0
        bag_count = 0

        # 2. Render Tracks
        for trk in tracks:
            cls_lower = trk.class_name.lower()
            if cls_lower == "person":
                people_count += 1
            elif cls_lower in ("backpack", "handbag", "suitcase"):
                bag_count += 1

            color = self.CLASS_COLORS.get(cls_lower, (200, 200, 200))
            x1, y1, x2, y2 = [int(v) for v in trk.bbox]

            # Motion Trail
            if self.draw_trail and len(trk.path) > 1:
                trail_pts = np.array(trk.path, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(out_frame, [trail_pts], False, color, 2, cv2.LINE_AA)
                cx, cy = int(trk.centre[0]), int(trk.centre[1])
                cv2.circle(out_frame, (cx, cy), 3, color, -1)

            # Bounding Box
            cv2.rectangle(out_frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # Label: ID, class, confidence, dwell
            dwell_str = f" | dwell:{trk.dwell_seconds:.1f}s" if trk.dwell_seconds > 0 else ""
            label = f"ID:{trk.track_id} {trk.class_name}{dwell_str}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

            cv2.rectangle(out_frame, (x1, max(0, y1 - lh - 6)), (x1 + lw + 6, y1), color, -1)
            cv2.putText(out_frame, label, (x1 + 3, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        # 3. Render Static Object Candidates (Abandoned bag tracking)
        if static_candidates:
            for cand in static_candidates:
                cx, cy = int(cand.centroid[0]), int(cand.centroid[1])
                stat_dur = current_time - cand.stationary_start_time
                stat_str = f"STATIC BAG: {stat_dur:.1f}s"
                cv2.circle(out_frame, (cx, cy), 8, (0, 0, 255), 2)
                cv2.putText(out_frame, stat_str, (cx - 40, cy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1, cv2.LINE_AA)

        # 4. Render Event Alert Banners (Top of screen)
        # Purge expired banners
        self._active_banners = [
            (t, evt) for t, evt in self._active_banners
            if (current_time - t) <= self.banner_display_seconds
        ]

        banner_y = 10
        for t_fired, evt in self._active_banners[-3:]:  # Show up to 3 recent banners
            sev_color = self.SEVERITY_COLORS.get(evt.severity, (0, 255, 255))
            banner_text = f"ALERT [{evt.severity.value.upper()}] - {evt.event_type.value} @ {evt.location.get('zone_name', 'Scene')}"

            (bw, bh), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(out_frame, (10, banner_y), (20 + bw, banner_y + bh + 10), (20, 20, 20), -1)
            cv2.rectangle(out_frame, (10, banner_y), (20 + bw, banner_y + bh + 10), sev_color, 2)
            cv2.putText(out_frame, banner_text, (15, banner_y + bh + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, sev_color, 2, cv2.LINE_AA)
            banner_y += bh + 16

        # 5. Status HUD (Top-Left or Bottom-Left)
        hud_lines = [
            f"CAM: {self.camera_id} | FRAME: {frame_id} | TIME: {current_time:.2f}s",
            f"MODEL: {self.model_name} | FPS: {fps:.1f} | TOTAL: {latency_ms:.1f}ms (RULES: {rules_ms:.2f}ms)",
            f"PEOPLE: {people_count} | BAGS: {bag_count} | ACTIVE ZONES: {len(zones)}",
        ]

        hud_w = 440
        hud_h = (len(hud_lines) * 20) + 16
        hud_y_start = out_frame.shape[0] - hud_h - 10

        overlay = out_frame.copy()
        cv2.rectangle(overlay, (10, hud_y_start), (10 + hud_w, hud_y_start + hud_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, out_frame, 0.25, 0, out_frame)

        for idx, line in enumerate(hud_lines):
            y_pos = hud_y_start + 14 + (idx * 20)
            cv2.putText(out_frame, line, (18, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 200) if idx == 0 else (240, 240, 240), 1, cv2.LINE_AA)

        return out_frame
