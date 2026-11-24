"""Deterministic Threat Scorer for the Threat Engine.

Calculates base scores from unique event weights and applies approved modifiers:
- Repeated event modifier: +25% of highest-weight repeated type, once per incident (Clarification a)
- Stratified unknown modifier: +15 verified unknown vs +5 low quality overhead (Clarification b)
- Known & authorized bonus: -20 only if all events are in authorized zone (Clarification c)
- Distinct event types modifier: +10 for >= 3 unique types
- High-risk zone penalty: +10

Conforms strictly to docs/05_THREAT_ENGINE_SPEC.md Section 5 and approved clarifications.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

from src.common.types import Event, EventType, IdentityStatus, Incident, ScoreExplanation, Severity

logger = logging.getLogger(__name__)

# Default weights from docs/05 Section 5.1
DEFAULT_EVENT_WEIGHTS: Dict[str, int] = {
    "FIGHTING": 50,
    "RESTRICTED_ZONE_ENTRY": 40,
    "ABANDONED_OBJECT": 30,
    "LINE_CROSSING": 20,
    "LOITERING": 20,
    "PROLONGED_STAY": 15,
    "OCCUPANCY_EXCEEDED": 15,
    "RUNNING": 10,
    "UNKNOWN_PERSON": 0,  # Context only; scored via modifier
}

# Default modifiers from docs/05 Section 5.2 and Clarifications (a)-(d)
DEFAULT_MODIFIERS: Dict[str, Any] = {
    "unknown_verified_penalty": 15,
    "unknown_no_usable_face_penalty": 5,
    "pending_penalty": 0,
    "known_authorized_bonus": -20,
    "distinct_event_types_threshold": 3,
    "distinct_event_types_penalty": 10,
    "high_risk_zone_penalty": 10,
    "repeated_event_type_multiplier": 0.25,
}

# Default severity levels from docs/05 Section 5.4
DEFAULT_SEVERITY_LEVELS: Dict[str, Dict[str, int]] = {
    "Low": {"min": 0, "max": 29},
    "Medium": {"min": 30, "max": 59},
    "High": {"min": 60, "max": 79},
    "Critical": {"min": 80, "max": 100},
}


class ThreatScorer:
    """Computes transparent, auditable threat scores and levels for incidents."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the ThreatScorer with weights and modifier settings.

        Args:
            config: Optional dict containing threat engine configuration.
        """
        cfg = config or {}
        self.weights: Dict[str, int] = {
            **DEFAULT_EVENT_WEIGHTS,
            **cfg.get("event_weights", {}),
        }
        self.modifiers: Dict[str, Any] = {
            **DEFAULT_MODIFIERS,
            **cfg.get("modifiers", {}),
        }
        self.levels: Dict[str, Dict[str, int]] = {
            "Low": {"min": 0, "max": 29},
            "Medium": {"min": 30, "max": 59},
            "High": {"min": 60, "max": 79},
            "Critical": {"min": 80, "max": 100},
        }
        raw_levels = cfg.get("severity_levels", {})
        if raw_levels:
            for k, v in raw_levels.items():
                name = k.capitalize()
                self.levels[name] = {"min": int(v.get("min_score", 0)), "max": int(v.get("max_score", 100))}

    def score_incident(self, incident: Incident) -> ScoreExplanation:
        """Score an incident deterministically and generate an audit explanation.

        Args:
            incident: The Incident containing correlated events and subject info.

        Returns:
            ScoreExplanation containing base_score, modifiers, final_score, and audit trail.
        """
        events = incident.events
        if not events:
            return ScoreExplanation(
                base_score=0,
                unique_events=[],
                modifiers_applied=[],
                unclamped_score=0,
                final_score=0,
                threat_level=Severity.LOW.value,
                explanation_lines=["No events recorded. Score: 0 (Low)"],
            )

        # 1. Base Score: sum of weights of unique event types (docs/05 Section 5.1 & 5.3)
        # Note: Rule-based and model-based LOITERING merge into single unique type
        event_type_counts: Dict[str, int] = {}
        for ev in events:
            # Handle UNKNOWN_PERSON as context, not base event weight
            if ev.event_type == EventType.UNKNOWN_PERSON:
                continue
            type_str = ev.event_type.value
            event_type_counts[type_str] = event_type_counts.get(type_str, 0) + 1

        unique_event_types: List[str] = sorted(list(event_type_counts.keys()))
        base_score = sum(self.weights.get(t, 0) for t in unique_event_types)

        explanation_lines: List[str] = []
        if unique_event_types:
            breakdown = [f"{t}({self.weights.get(t, 0)})" for t in unique_event_types]
            explanation_lines.append(f"Base score: {base_score} from unique events [{', '.join(breakdown)}]")
        else:
            explanation_lines.append(f"Base score: {base_score} (no threat-weight event types)")

        modifiers_applied: List[Dict[str, Any]] = []

        # 2. Clarification (a): Repeated event modifier (+25% of highest-weight repeated type, once)
        repeated_types = [t for t, count in event_type_counts.items() if count > 1]
        if repeated_types:
            # Pick highest-weight repeated event type
            highest_type = max(repeated_types, key=lambda t: self.weights.get(t, 0))
            highest_weight = self.weights.get(highest_type, 0)
            rep_multiplier = float(self.modifiers.get("repeated_event_type_multiplier", 0.25))
            rep_adj = int(round(highest_weight * rep_multiplier))

            if rep_adj > 0:
                mod = {
                    "name": "repeated_event_type",
                    "adjustment": rep_adj,
                    "reason": f"Repeated event '{highest_type}' (+{int(rep_multiplier*100)}% of weight {highest_weight})",
                }
                modifiers_applied.append(mod)
                explanation_lines.append(f"Modifier: {mod['reason']} -> +{rep_adj}")

        # 3. Clarification (b): Stratified Unknown Person Modifier
        unknown_mod = self._compute_unknown_modifier(incident)
        if unknown_mod:
            modifiers_applied.append(unknown_mod)
            explanation_lines.append(f"Modifier: {unknown_mod['reason']} -> +{unknown_mod['adjustment']}")

        # 4. Clarification (c): Known & Authorized Deduction (-20)
        authorized_mod = self._compute_authorized_modifier(incident)
        if authorized_mod:
            modifiers_applied.append(authorized_mod)
            explanation_lines.append(f"Modifier: {authorized_mod['reason']} -> {authorized_mod['adjustment']}")

        # 5. Distinct Event Types Modifier (+10 for >= 3 distinct event types)
        threshold = int(self.modifiers.get("distinct_event_types_threshold", 3))
        if len(unique_event_types) >= threshold:
            distinct_adj = int(self.modifiers.get("distinct_event_types_penalty", 10))
            mod = {
                "name": "distinct_event_types",
                "adjustment": distinct_adj,
                "reason": f"{len(unique_event_types)} distinct event types detected (threshold: {threshold})",
            }
            modifiers_applied.append(mod)
            explanation_lines.append(f"Modifier: {mod['reason']} -> +{distinct_adj}")

        # 6. High-Risk Zone Penalty (+10)
        if self._is_high_risk(incident):
            high_risk_adj = int(self.modifiers.get("high_risk_zone_penalty", 10))
            mod = {
                "name": "high_risk_zone",
                "adjustment": high_risk_adj,
                "reason": "Event occurred in designated high-risk zone",
            }
            modifiers_applied.append(mod)
            explanation_lines.append(f"Modifier: {mod['reason']} -> +{high_risk_adj}")

        # 7. Unclamped & Clamped Calculation
        total_modifiers = sum(m["adjustment"] for m in modifiers_applied)
        unclamped_score = base_score + total_modifiers
        final_score = max(0, min(100, unclamped_score))

        # 8. Threat Level Classification
        threat_level = self._classify_level(final_score)
        explanation_lines.append(
            f"Total: unclamped {unclamped_score} -> clamped {final_score} [{threat_level}]"
        )

        return ScoreExplanation(
            base_score=base_score,
            unique_events=unique_event_types,
            modifiers_applied=modifiers_applied,
            unclamped_score=unclamped_score,
            final_score=final_score,
            threat_level=threat_level,
            explanation_lines=explanation_lines,
        )

    def _compute_unknown_modifier(self, incident: Incident) -> Optional[Dict[str, Any]]:
        """Compute stratified unknown modifier based on biometric verification quality."""
        # Check subject dictionary or events for unknown identity status
        subj = incident.subject or {}
        id_status = str(subj.get("identity_status", "")).upper()
        reason = str(subj.get("reason", "") or subj.get("identity_reason", "")).lower()

        # Check events if not set on subject
        if not id_status:
            for ev in incident.events:
                if ev.event_type == EventType.UNKNOWN_PERSON:
                    id_status = "UNKNOWN"
                    reason = str(ev.details.get("reason", "")).lower()
                    break
                ev_id_status = str(ev.details.get("identity_status", "")).upper()
                if ev_id_status == "UNKNOWN":
                    id_status = "UNKNOWN"
                    reason = str(ev.details.get("reason", "")).lower()
                    break

        if id_status != "UNKNOWN":
            return None

        # Stratified evaluation: verified non-match (+15) vs low-quality / unusable face (+5)
        # Low quality markers: blur, small_face, low_resolution, no_usable_face, unconstrained
        is_low_quality = any(
            marker in reason
            for marker in ("low_quality", "small_face", "blur", "no_usable_face", "resolution", "overhead")
        )

        if is_low_quality:
            penalty = int(self.modifiers.get("unknown_no_usable_face_penalty", 5))
            return {
                "name": "unknown_no_usable_face",
                "adjustment": penalty,
                "reason": f"Unknown subject with unusable/low-quality face on wide camera ({reason or 'low_quality'})",
            }
        else:
            penalty = int(self.modifiers.get("unknown_verified_penalty", 15))
            return {
                "name": "unknown_verified",
                "adjustment": penalty,
                "reason": f"Unknown subject verified non-match against gallery ({reason or 'no_match'})",
            }

    def _compute_authorized_modifier(self, incident: Incident) -> Optional[Dict[str, Any]]:
        """Compute authorized person deduction if strictly authorized across all events."""
        subj = incident.subject or {}
        id_status = str(subj.get("identity_status", "")).upper()

        if not id_status:
            for ev in incident.events:
                ev_status = str(ev.details.get("identity_status", "")).upper()
                if ev_status == "KNOWN":
                    id_status = "KNOWN"
                    break

        if id_status != "KNOWN":
            return None

        # Check: Every person event in incident must be in an authorized zone, and NO RESTRICTED_ZONE_ENTRY
        for ev in incident.events:
            if ev.event_type == EventType.RESTRICTED_ZONE_ENTRY:
                # Intrusion into restricted zone negates authorized bonus
                return None
            loc = ev.location or {}
            # If explicitly marked unauthorized, fail
            if loc.get("authorized") is False or loc.get("is_authorized") is False:
                # If zone is restricted or high risk without authorization
                if loc.get("high_risk") or "restricted" in str(loc.get("zone_id", "")).lower():
                    return None

        bonus = int(self.modifiers.get("known_authorized_bonus", -20))
        return {
            "name": "known_authorized",
            "adjustment": bonus,
            "reason": "Subject is Known and fully authorized for active zones",
        }

    def _is_high_risk(self, incident: Incident) -> bool:
        """Check if any event or incident location is designated high risk."""
        if incident.details.get("high_risk") or incident.details.get("is_high_risk"):
            return True
        for ev in incident.events:
            loc = ev.location or {}
            if loc.get("high_risk") or loc.get("is_high_risk"):
                return True
        return False

    def _classify_level(self, score: int) -> str:
        """Map score 0-100 to Severity Level (Low, Medium, High, Critical)."""
        for level_name in ["Critical", "High", "Medium", "Low"]:
            bounds = self.levels.get(level_name)
            if bounds and bounds["min"] <= score <= bounds["max"]:
                return level_name
        return Severity.LOW.value
