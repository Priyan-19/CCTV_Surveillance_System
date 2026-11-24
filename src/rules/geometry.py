"""Geometric operations, spatial tests, and tripwire intersection mathematics.

Provides:
- Point-in-polygon verification via OpenCV pointPolygonTest.
- Virtual tripwire line crossing detection and directional sign determination.
- Bounding box reference point calculation (bottom-centre for floor planes vs centroid).
- Euclidean distances and IoU overlap calculations.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple
import cv2
import numpy as np


def get_reference_point(
    bbox: Tuple[float, float, float, float],
    mode: str = "bottom_center",
) -> Tuple[float, float]:
    """Compute the spatial reference point for a bounding box.

    Args:
        bbox: (x1, y1, x2, y2)
        mode: 'bottom_center' (feet contact point on floor plane) or 'center' (mid-torso).

    Note:
        In surveillance camera perspectives looking down at an angle, the bottom-centre
        accurately represents floor contact terrain, preventing premature triggers
        when tall people enter or lean across zones.
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    if mode == "center":
        cy = (y1 + y2) / 2.0
    else:
        # Default to bottom-centre (feet contact)
        cy = y2
    return (cx, cy)


def point_in_polygon(
    point: Tuple[float, float],
    polygon: Sequence[Tuple[float, float]],
) -> bool:
    """Check if a 2D point lies strictly inside or on the boundary of a polygon.

    Args:
        point: (x, y) coordinates
        polygon: List of (x, y) vertices defining a closed polygon.
    """
    if len(polygon) < 3:
        return False

    pts = np.array(polygon, dtype=np.float32).reshape((-1, 1, 2))
    # cv2.pointPolygonTest returns positive if inside, 0 if on edge, negative if outside
    dist = cv2.pointPolygonTest(pts, (float(point[0]), float(point[1])), measureDist=False)
    return dist >= 0.0


def signed_side(
    line_start: Tuple[float, float],
    line_end: Tuple[float, float],
    point: Tuple[float, float],
) -> float:
    """Determine which side of a directed line segment (A -> B) a point lies on.

    Returns:
        Positive float: Point is on the left side of vector AB.
        Negative float: Point is on the right side of vector AB.
        Zero: Point is colinear with vector AB.
    """
    x1, y1 = line_start
    x2, y2 = line_end
    px, py = point
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def segments_intersect(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    p4: Tuple[float, float],
) -> bool:
    """Check if line segment p1-p2 intersects line segment p3-p4."""
    def ccw(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> bool:
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))


def check_line_crossing(
    prev_pt: Tuple[float, float],
    curr_pt: Tuple[float, float],
    line_start: Tuple[float, float],
    line_end: Tuple[float, float],
    direction: str = "both",
) -> Tuple[bool, str]:
    """Determine whether motion from prev_pt to curr_pt crossed the virtual line.

    Args:
        prev_pt: Previous centroid (t-1).
        curr_pt: Current centroid (t).
        line_start: Point A of the line segment.
        line_end: Point B of the line segment.
        direction: Allowed crossing check: 'both', 'a_to_b', or 'b_to_a' (left-to-right vs right-to-left).

    Returns:
        (crossed: bool, crossing_direction: str)
    """
    if not segments_intersect(prev_pt, curr_pt, line_start, line_end):
        return False, ""

    side_prev = signed_side(line_start, line_end, prev_pt)
    side_curr = signed_side(line_start, line_end, curr_pt)

    # Must change signs across the line to be a genuine traversal
    if side_prev * side_curr > 0:
        return False, ""

    if side_prev > 0 and side_curr <= 0:
        crossing_dir = "left_to_right"  # from left of AB to right of AB
    else:
        crossing_dir = "right_to_left"

    if direction == "both":
        return True, crossing_dir
    elif direction in ("a_to_b", "left_to_right", "inbound") and crossing_dir == "left_to_right":
        return True, crossing_dir
    elif direction in ("b_to_a", "right_to_left", "outbound") and crossing_dir == "right_to_left":
        return True, crossing_dir

    return False, crossing_dir


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Compute 2D Euclidean distance between two points."""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def compute_iou(
    box1: Tuple[float, float, float, float],
    box2: Tuple[float, float, float, float],
) -> float:
    """Compute Intersection-over-Union (IoU) between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0
