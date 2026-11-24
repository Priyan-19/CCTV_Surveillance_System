"""Base class for all Phase 2 security rule evaluators.

Enforces:
- Common interface: evaluate(frame_result, zones, lines, current_time) -> List[Event]
- Video-time determinism (evaluates against frame video time, not system clock).
- Debouncing (requires N consecutive frames or minimum duration before trigger).
- Cooldown and re-arming (fires once per occurrence, prevents event flooding).
- Clean state cleanup when tracks disappear or exit scene.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple

from src.common.types import Event, EventType, FrameResult, Severity, Track
from src.rules.zones import VirtualLine, Zone


class BaseRule(ABC):
    """Abstract Base Class for spatial, temporal, and behavioural rules."""

    def __init__(self, rule_name: str, config: Dict[str, Any]) -> None:
        self.rule_name = rule_name
        self.config = config
        self._event_counter: int = 0

    def generate_event_id(self, event_type: EventType, timestamp: float) -> str:
        """Create a deterministic, monotonically increasing event ID."""
        self._event_counter += 1
        return f"evt_{event_type.value.lower()}_{int(timestamp)}_{self._event_counter:04d}"

    @abstractmethod
    def evaluate(
        self,
        frame_result: FrameResult,
        zones: List[Zone],
        lines: List[VirtualLine],
        current_time: float,
    ) -> List[Event]:
        """Evaluate rule conditions against the current frame result.

        Args:
            frame_result: Current processed frame containing tracks and detections.
            zones: Configured spatial polygons for this camera.
            lines: Configured virtual tripwires for this camera.
            current_time: Deterministic video time in seconds (frame_id / fps or timestamp).

        Returns:
            List of newly triggered Event instances (if any).
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal rule states, timers, and cooldown tracking."""
        pass
