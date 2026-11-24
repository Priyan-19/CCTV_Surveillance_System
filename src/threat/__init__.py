"""Threat engine module for event normalization, correlation, scoring, and alerting."""

from src.threat.alert_dispatcher import AlertDispatcher
from src.threat.correlator import SingleCameraCorrelator
from src.threat.incident_manager import IncidentManager
from src.threat.normalizer import EventNormalizer
from src.threat.scorer import ThreatScorer

__all__ = [
    "EventNormalizer",
    "SingleCameraCorrelator",
    "ThreatScorer",
    "AlertDispatcher",
    "IncidentManager",
]
