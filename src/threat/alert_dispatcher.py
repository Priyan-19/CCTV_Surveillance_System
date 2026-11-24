"""Alert dispatcher for the Threat Engine.

Enforces alerting rules from docs/05 Section 7 and docs/11:
- Alert once on incident creation (Medium, High, Critical).
- Alert again only when the threat level increases (escalation).
- Single low-weight events do not create notifications.
- Privacy gate: face images are excluded from text/outbox notifications unless explicitly enabled.
- Dispatches to console and JSONL file outbox sinks.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.common.types import Alert, Incident, Severity

logger = logging.getLogger(__name__)

SEVERITY_RANK: Dict[str, int] = {
    Severity.LOW.value: 1,
    Severity.MEDIUM.value: 2,
    Severity.HIGH.value: 3,
    Severity.CRITICAL.value: 4,
}


class AlertDispatcher:
    """Dispatches validated alerts to console and file outbox sinks."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize AlertDispatcher with sink configuration.

        Args:
            config: Optional dict containing alerts configuration.
        """
        cfg = config or {}
        disp_cfg = cfg.get("dispatch", {})
        rules_cfg = cfg.get("rules", {})
        content_cfg = cfg.get("content", {})

        self.console_enabled: bool = bool(disp_cfg.get("console_enabled", True))
        self.file_outbox_enabled: bool = bool(disp_cfg.get("file_outbox_enabled", True))
        self.outbox_path: Path = Path(disp_cfg.get("outbox_path", "data/output/alerts/alerts_outbox.jsonl"))

        self.min_level_console: str = rules_cfg.get("min_level_console", "Medium")
        self.min_level_outbox: str = rules_cfg.get("min_level_outbox", "Medium")
        self.min_level_notification: str = rules_cfg.get("min_level_notification", "High")

        self.include_evidence_links: bool = bool(content_cfg.get("include_evidence_links", True))
        self.include_face_image: bool = bool(content_cfg.get("include_face_image_in_notification", False))

        # Alert sequence counter
        self._alert_seq: int = 1

        # Track highest threat level dispatched per incident_no
        # incident_no -> highest dispatched level name
        self._dispatched_levels: Dict[str, str] = {}

        if self.file_outbox_enabled:
            self.outbox_path.parent.mkdir(parents=True, exist_ok=True)

    def evaluate_and_dispatch(self, incident: Incident) -> Optional[Alert]:
        """Evaluate an incident against alerting rules and dispatch if conditions are met.

        Rules:
        - Low severity: Log only, visible in dashboard, no alert dispatched.
        - First time incident reaches Medium, High, or Critical: Dispatch alert.
        - Subsequent events: Dispatch ONLY if threat level increases (escalation).

        Args:
            incident: The active Incident to evaluate.

        Returns:
            The created Alert if dispatched, or None if suppressed.
        """
        current_level = incident.threat_level
        current_rank = SEVERITY_RANK.get(current_level, 1)

        # Rule 1: Single low-weight events (Low level) do not create alerts
        if current_rank <= SEVERITY_RANK[Severity.LOW.value]:
            return None

        # Check previous dispatch history
        prev_level = self._dispatched_levels.get(incident.incident_no)
        if prev_level is not None:
            prev_rank = SEVERITY_RANK.get(prev_level, 1)
            # Rule 2: Alert again ONLY when the threat level increases
            if current_rank <= prev_rank:
                return None

        # Record new dispatched level
        self._dispatched_levels[incident.incident_no] = current_level

        # Build Alert
        alert_id = f"ALT-{self._alert_seq:06d}"
        self._alert_seq += 1

        # Format privacy-compliant evidence paths
        evidence_paths: List[str] = []
        if self.include_evidence_links:
            for ev in incident.evidence:
                # Privacy gate (docs/11): exclude face images unless explicitly permitted
                if ev.type == "face_image" and not self.include_face_image:
                    continue
                if ev.file_path:
                    evidence_paths.append(ev.file_path)

        # Build human-readable notification message
        act_summary = ", ".join(incident.activities) if incident.activities else "Unknown activity"
        subj_desc = "Unknown Person"
        if incident.subject:
            if incident.subject.get("person_name"):
                subj_desc = str(incident.subject.get("person_name"))
            elif incident.subject.get("track_id"):
                subj_desc = f"Track #{incident.subject.get('track_id')}"

        message = (
            f"[{current_level.upper()}] Security Incident {incident.incident_no} at "
            f"{incident.camera_id} ({incident.place}): Threat Score {incident.threat_score}/100. "
            f"Subject: {subj_desc}. Activities: {act_summary}."
        )

        alert = Alert(
            alert_id=alert_id,
            incident_no=incident.incident_no,
            threat_level=current_level,
            threat_score=incident.threat_score,
            camera_id=incident.camera_id,
            place=incident.place,
            timestamp=incident.last_event_time,
            message=message,
            evidence_paths=evidence_paths,
            details={
                "activities": incident.activities,
                "start_time": incident.start_time,
                "score_history_count": len(incident.score_history),
                "explanation": incident.explanation,
            },
        )

        # Dispatch to configured sinks
        self._dispatch_to_sinks(alert, current_rank)

        return alert

    def _dispatch_to_sinks(self, alert: Alert, rank: int) -> None:
        """Route alert to enabled destinations."""
        min_console_rank = SEVERITY_RANK.get(self.min_level_console, 2)
        min_outbox_rank = SEVERITY_RANK.get(self.min_level_outbox, 2)

        # 1. Console Sink
        if self.console_enabled and rank >= min_console_rank:
            color_prefix = {
                "Medium": "\033[93m",    # Yellow
                "High": "\033[91m",      # Light Red
                "Critical": "\033[1;31m", # Bold Red
            }.get(alert.threat_level, "\033[0m")
            reset_color = "\033[0m"

            banner = (
                f"\n{color_prefix}"
                f"================================================================================\n"
                f" [ALERT DISPATCHED] {alert.alert_id} | {alert.threat_level.upper()} (Score: {alert.threat_score})\n"
                f" Incident : {alert.incident_no} | Camera: {alert.camera_id} ({alert.place})\n"
                f" Message  : {alert.message}\n"
                f" Evidence : {len(alert.evidence_paths)} items attached\n"
                f"================================================================================"
                f"{reset_color}"
            )
            print(banner)

        # 2. File Outbox Sink (append line)
        if self.file_outbox_enabled and rank >= min_outbox_rank:
            try:
                with open(self.outbox_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(alert.to_dict()) + "\n")
            except Exception as e:
                logger.error(f"Failed to write alert to outbox file {self.outbox_path}: {e}")
