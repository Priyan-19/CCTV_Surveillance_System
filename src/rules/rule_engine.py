"""Rule Engine Orchestrator.

Coordinates evaluation of all 6 spatial and temporal security rules:
1. RESTRICTED_ZONE_ENTRY
2. LINE_CROSSING
3. PROLONGED_STAY
4. LOITERING
5. ABANDONED_OBJECT
6. OCCUPANCY_EXCEEDED

Measures execution latency (rules_ms) and attaches triggered events to FrameResult.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml

from src.common.types import Event, FrameResult
from src.rules.abandoned_object import AbandonedObjectRule
from src.rules.base_rule import BaseRule
from src.rules.line_crossing import LineCrossingRule
from src.rules.loitering import LoiteringRule
from src.rules.occupancy import OccupancyRule
from src.rules.prolonged_stay import ProlongedStayRule
from src.rules.restricted_zone import RestrictedZoneRule
from src.rules.zones import ZoneManager

logger = logging.getLogger("rule_engine")


class RuleEngine:
    """Central engine orchestrating all rule evaluators for a video stream."""

    def __init__(
        self,
        rules_config_path: Union[str, Path] = "configs/rules.yaml",
        zone_manager: Optional[ZoneManager] = None,
    ) -> None:
        self.rules_config_path = Path(rules_config_path)
        self.zone_manager = zone_manager or ZoneManager()
        self.rules: List[BaseRule] = []

        self._load_rules()

    def _load_rules(self) -> None:
        """Instantiate rules from configs/rules.yaml."""
        cfg_data: Dict[str, Any] = {}
        if self.rules_config_path.exists():
            with open(self.rules_config_path, "r", encoding="utf-8") as f:
                cfg_data = yaml.safe_load(f) or {}

        # Instantiate each rule with its specific config section
        self.restricted_rule = RestrictedZoneRule(cfg_data.get("restricted_zone", {}))
        self.line_crossing_rule = LineCrossingRule(cfg_data.get("line_crossing", {}))
        self.prolonged_stay_rule = ProlongedStayRule(cfg_data.get("prolonged_stay", {}))
        self.loitering_rule = LoiteringRule(cfg_data.get("loitering", {}))
        self.abandoned_rule = AbandonedObjectRule(cfg_data.get("abandoned_object", {}))
        self.occupancy_rule = OccupancyRule(cfg_data.get("occupancy", {}))

        self.rules = [
            self.restricted_rule,
            self.line_crossing_rule,
            self.prolonged_stay_rule,
            self.loitering_rule,
            self.abandoned_rule,
            self.occupancy_rule,
        ]

        logger.info("RuleEngine initialized with %d active security rules.", len(self.rules))

    def evaluate(self, frame_result: FrameResult, camera_id: str) -> List[Event]:
        """Evaluate all security rules on the current frame result.

        Args:
            frame_result: Output from detector and tracker containing current tracks.
            camera_id: Camera stream identifier.

        Returns:
            List of newly triggered Event instances.
        """
        t0 = time.perf_counter()

        # Deterministic video time
        current_time = frame_result.timestamp
        zones = self.zone_manager.get_zones(camera_id)
        lines = self.zone_manager.get_lines(camera_id)

        all_events: List[Event] = []

        for rule in self.rules:
            rule_events = rule.evaluate(
                frame_result=frame_result,
                zones=zones,
                lines=lines,
                current_time=current_time,
            )
            if rule_events:
                all_events.extend(rule_events)

        t1 = time.perf_counter()
        rules_latency_ms = (t1 - t0) * 1000.0

        # Update timings and attach events to FrameResult
        frame_result.timings["rules_ms"] = rules_latency_ms
        frame_result.events.extend(all_events)

        return all_events

    def reset(self) -> None:
        """Reset all child rule states."""
        for rule in self.rules:
            rule.reset()
