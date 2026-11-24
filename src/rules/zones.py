"""Zone and Virtual Line configuration parser and spatial scaler.

Conforms to docs/07_DATABASE_SCHEMA.md (zones table).
Stores the reference resolution at which coordinates were drawn, and scales
polygons and tripwires dynamically to match incoming camera frame dimensions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import yaml

from src.common.types import Severity

logger = logging.getLogger("zones")


@dataclass
class Zone:
    """Represents a 2D spatial polygon defined for a camera stream.

    Attributes:
        zone_id: Unique string identifier (e.g. 'zone_restricted_corridor').
        name: Human-readable display label.
        type: 'restricted', 'occupancy', 'watch', or 'normal'.
        polygon: List of (x, y) vertices in reference resolution coordinates.
        reference_resolution: (width, height) at which coordinates were drawn.
        dwell_threshold_seconds: Dwell time threshold in seconds for PROLONGED_STAY.
        occupancy_limit: Maximum allowed persons for OCCUPANCY_EXCEEDED.
        risk_level: Associated threat severity (Low, Medium, High, Critical).
        color: Optional BGR color tuple for visual rendering.
    """

    zone_id: str
    name: str
    type: str  # 'restricted', 'occupancy', 'watch', 'normal'
    polygon: List[Tuple[float, float]]
    reference_resolution: Tuple[int, int] = (640, 480)
    dwell_threshold_seconds: float = 30.0
    occupancy_limit: int = 4
    risk_level: Severity = Severity.MEDIUM
    color: Optional[Tuple[int, int, int]] = None

    def __post_init__(self) -> None:
        if self.type.lower() == "restricted" and self.risk_level == Severity.MEDIUM:
            self.risk_level = Severity.HIGH

    def get_scaled_polygon(self, frame_w: int, frame_h: int) -> List[Tuple[float, float]]:
        """Scale polygon coordinates to match current video frame resolution."""
        ref_w, ref_h = self.reference_resolution
        if ref_w <= 0 or ref_h <= 0 or (ref_w == frame_w and ref_h == frame_h):
            return self.polygon

        scale_x = frame_w / float(ref_w)
        scale_y = frame_h / float(ref_h)
        return [(round(x * scale_x, 1), round(y * scale_y, 1)) for x, y in self.polygon]


@dataclass
class VirtualLine:
    """Represents a virtual tripwire defined for a camera stream.

    Attributes:
        line_id: Unique string identifier (e.g. 'line_entrance').
        name: Human-readable display label.
        start: (x1, y1) endpoint A.
        end: (x2, y2) endpoint B.
        direction: 'both', 'a_to_b', 'b_to_a', 'inbound', 'outbound'.
        reference_resolution: (width, height) at which line was drawn.
        risk_level: Associated threat severity.
    """

    line_id: str
    name: str
    start: Tuple[float, float]
    end: Tuple[float, float]
    direction: str = "both"
    reference_resolution: Tuple[int, int] = (640, 480)
    risk_level: Severity = Severity.MEDIUM

    def get_scaled_line(
        self, frame_w: int, frame_h: int
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Scale line endpoints to match current video frame resolution."""
        ref_w, ref_h = self.reference_resolution
        if ref_w <= 0 or ref_h <= 0 or (ref_w == frame_w and ref_h == frame_h):
            return (self.start, self.end)

        scale_x = frame_w / float(ref_w)
        scale_y = frame_h / float(ref_h)
        scaled_start = (round(self.start[0] * scale_x, 1), round(self.start[1] * scale_y, 1))
        scaled_end = (round(self.end[0] * scale_x, 1), round(self.end[1] * scale_y, 1))
        return (scaled_start, scaled_end)


class ZoneManager:
    """Loads, validates, and queries spatial zones and lines across cameras."""

    def __init__(self, config_path: Union[str, Path] = "configs/zones.yaml") -> None:
        self.config_path = Path(config_path)
        self.zones_by_camera: Dict[str, List[Zone]] = {}
        self.lines_by_camera: Dict[str, List[VirtualLine]] = {}
        self.load_config()

    def load_config(self) -> None:
        """Parse and validate configs/zones.yaml."""
        if not self.config_path.exists():
            logger.warning("Zone config file '%s' not found.", self.config_path)
            return

        with open(self.config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict) or "cameras" not in data:
            logger.warning("Zone config missing 'cameras' root dictionary.")
            return

        cams_data = data["cameras"]
        for cam_id, cam_cfg in cams_data.items():
            # 1. Parse Polygons (Zones)
            zone_list: List[Zone] = []
            for z_raw in cam_cfg.get("zones", []):
                ref_res = tuple(z_raw.get("reference_resolution", [640, 480]))
                poly_pts = [tuple(p) for p in z_raw.get("polygon", [])]
                if len(poly_pts) < 3:
                    logger.warning("Zone '%s' in camera '%s' has < 3 vertices. Skipped.", z_raw.get("id"), cam_id)
                    continue

                risk_str = z_raw.get("risk_level", "Medium").capitalize()
                try:
                    severity = Severity(risk_str)
                except ValueError:
                    severity = Severity.MEDIUM

                zone = Zone(
                    zone_id=z_raw["id"],
                    name=z_raw.get("name", z_raw["id"]),
                    type=z_raw.get("type", "restricted"),
                    polygon=poly_pts,
                    reference_resolution=(int(ref_res[0]), int(ref_res[1])),
                    dwell_threshold_seconds=float(z_raw.get("dwell_threshold_seconds", 30.0)),
                    occupancy_limit=int(z_raw.get("occupancy_limit", 4)),
                    risk_level=severity,
                )
                zone_list.append(zone)

            self.zones_by_camera[cam_id] = zone_list

            # 2. Parse Lines (Tripwires)
            line_list: List[VirtualLine] = []
            for l_raw in cam_cfg.get("lines", []):
                ref_res = tuple(l_raw.get("reference_resolution", [640, 480]))
                start_pt = tuple(l_raw.get("start", [0, 0]))
                end_pt = tuple(l_raw.get("end", [100, 100]))

                risk_str = l_raw.get("risk_level", "Medium").capitalize()
                try:
                    severity = Severity(risk_str)
                except ValueError:
                    severity = Severity.MEDIUM

                line = VirtualLine(
                    line_id=l_raw["id"],
                    name=l_raw.get("name", l_raw["id"]),
                    start=(float(start_pt[0]), float(start_pt[1])),
                    end=(float(end_pt[0]), float(end_pt[1])),
                    direction=l_raw.get("direction", "both"),
                    reference_resolution=(int(ref_res[0]), int(ref_res[1])),
                    risk_level=severity,
                )
                line_list.append(line)

            self.lines_by_camera[cam_id] = line_list

        logger.info(
            "Loaded zone configuration: %d cameras, %d total zones, %d total tripwires.",
            len(cams_data),
            sum(len(z) for z in self.zones_by_camera.values()),
            sum(len(l) for l in self.lines_by_camera.values()),
        )

    def get_zones(self, camera_id: str) -> List[Zone]:
        return self.zones_by_camera.get(camera_id, [])

    def get_lines(self, camera_id: str) -> List[VirtualLine]:
        return self.lines_by_camera.get(camera_id, [])
