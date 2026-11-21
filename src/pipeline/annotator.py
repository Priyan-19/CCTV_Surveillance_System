"""Visual annotation module.

Renders high-quality bounding boxes, class labels, detection confidence,
persistent Track IDs, motion history trails, and a real-time status HUD.
Note: Does not render threat levels (reserved for Phase 4).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np

from src.common.types import Track


class FrameAnnotator:
    """Draws tracking overlays, centroid trails, and diagnostic HUD onto video frames."""

    # High-contrast color palette (BGR format) by class category:
    CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
        "person": (0, 255, 128),       # Vibrant Spring Green
        "backpack": (0, 215, 255),     # Amber Gold
        "handbag": (0, 215, 255),      # Amber Gold
        "suitcase": (0, 165, 255),     # Deep Orange
        "car": (255, 144, 30),         # Azure Blue
        "motorcycle": (255, 100, 0),   # Deep Blue
        "bus": (255, 191, 0),          # Sky Blue
        "truck": (255, 128, 0),        # Cerulean
        "bicycle": (200, 255, 0),      # Chartreuse
    }
    DEFAULT_COLOR: Tuple[int, int, int] = (200, 200, 200)

    def __init__(
        self,
        camera_id: str,
        model_name: str,
        tracker_name: str = "ByteTrack",
        draw_trail: bool = True,
        max_trail_points: int = 30,
    ) -> None:
        self.camera_id = camera_id
        self.model_name = model_name
        self.tracker_name = tracker_name
        self.draw_trail = draw_trail
        self.max_trail_points = max_trail_points

    def _get_color(self, class_name: str) -> Tuple[int, int, int]:
        return self.CLASS_COLORS.get(class_name.lower(), self.DEFAULT_COLOR)

    def annotate(
        self,
        frame: np.ndarray,
        tracks: Sequence[Track],
        fps: float,
        latency_ms: float,
        frame_id: int,
    ) -> np.ndarray:
        """Annotate a copy of the frame with track boxes, motion trails, and status HUD."""
        out_frame = frame.copy()
        people_count = 0
        bag_count = 0

        # 1. Render Tracks (Boxes, Labels, Motion Trails)
        for trk in tracks:
            cls_lower = trk.class_name.lower()
            if cls_lower == "person":
                people_count += 1
            elif cls_lower in ("backpack", "handbag", "suitcase"):
                bag_count += 1

            color = self._get_color(trk.class_name)
            x1, y1, x2, y2 = [int(v) for v in trk.bbox]

            # Motion Trail
            if self.draw_trail and len(trk.path) > 1:
                trail_pts = np.array(trk.path, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(
                    out_frame,
                    [trail_pts],
                    isClosed=False,
                    color=color,
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )
                # Small circle at current centroid
                cx, cy = int(trk.centre[0]), int(trk.centre[1])
                cv2.circle(out_frame, (cx, cy), 3, color, -1)

            # Bounding Box
            cv2.rectangle(out_frame, (x1, y1), (x2, y2), color, thickness=2, lineType=cv2.LINE_AA)

            # Label text: e.g. "ID:12 person 0.88"
            label = f"ID:{trk.track_id} {trk.class_name} {trk.confidence:.2f}"
            (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            label_bg_y1 = max(0, y1 - lh - 6)
            label_bg_y2 = y1
            cv2.rectangle(
                out_frame,
                (x1, label_bg_y1),
                (x1 + lw + 6, label_bg_y2),
                color,
                thickness=-1,
            )
            cv2.putText(
                out_frame,
                label,
                (x1 + 3, label_bg_y2 - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),  # Black text for high readability over bright fill
                thickness=1,
                lineType=cv2.LINE_AA,
            )

        # 2. Render Status HUD Panel (Top-Left)
        hud_lines = [
            f"CAM: {self.camera_id} | FRAME: {frame_id}",
            f"MODEL: {self.model_name} | TRACKER: {self.tracker_name}",
            f"FPS: {fps:.1f} | LATENCY: {latency_ms:.1f} ms",
            f"PEOPLE: {people_count} | BAGS: {bag_count} | TRACKS: {len(tracks)}",
        ]

        hud_font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.52
        line_height = 20
        padding = 8
        hud_w = 420
        hud_h = (len(hud_lines) * line_height) + (padding * 2)

        # Semi-transparent dark background for HUD
        overlay = out_frame.copy()
        cv2.rectangle(
            overlay,
            (10, 10),
            (10 + hud_w, 10 + hud_h),
            (20, 20, 20),
            thickness=-1,
        )
        cv2.addWeighted(overlay, 0.75, out_frame, 0.25, 0, out_frame)

        # Draw HUD text
        for idx, line in enumerate(hud_lines):
            y_pos = 10 + padding + ((idx + 1) * line_height) - 4
            cv2.putText(
                out_frame,
                line,
                (18, y_pos),
                hud_font,
                font_scale,
                (0, 255, 200) if idx == 0 else (240, 240, 240),
                thickness=1,
                lineType=cv2.LINE_AA,
            )

        return out_frame
