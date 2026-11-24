"""Per-track identity state machine and temporal consensus aggregator.

Manages identity transitions across video frames:
    PENDING  -->  KNOWN   (upon K consistent positive matches)
    PENDING  -->  UNKNOWN (upon K consistent stranger votes OR timeout with no usable face)
    KNOWN    -->  holds state with hysteresis against momentary bad frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.common.types import Identity, IdentityStatus, Track
from src.face.matcher import MatchResult


@dataclass
class FaceObservation:
    """A single face recognition observation for a track."""

    timestamp: float
    status: IdentityStatus
    person_id: Optional[int]
    name: Optional[str]
    category: Optional[str]
    authorised_zone_ids: List[Union[int, str]]
    score: float
    margin: float
    reason: str
    quality_passed: bool
    face_crop: Optional[np.ndarray] = None


@dataclass
class TrackIdentityRecord:
    """Maintains state and evidence history for a single tracked individual."""

    track_id: int
    first_seen: float
    last_seen: float
    status: IdentityStatus = IdentityStatus.PENDING
    person_id: Optional[int] = None
    name: Optional[str] = None
    category: Optional[str] = None
    authorised_zone_ids: List[Union[int, str]] = field(default_factory=list)
    best_score: float = 0.0
    best_margin: float = 0.0
    decision_reason: str = "initial_pending"
    decision_timestamp: Optional[float] = None
    observations: List[FaceObservation] = field(default_factory=list)
    best_face_crop: Optional[np.ndarray] = None
    face_attempts: int = 0
    quality_passes: int = 0


class IdentityManager:
    """Manages identity states for all active tracks in a camera stream."""

    def __init__(
        self,
        min_consistent_matches: int = 2,
        min_consistent_stranger_votes: int = 3,
        timeout_seconds: float = 3.5,
        recheck_interval_seconds: float = 1.0,
        no_usable_face_backoff_multiplier: float = 3.0,
    ) -> None:
        """Initialize IdentityManager.

        Args:
            min_consistent_matches: Number of matching observations needed to confirm KNOWN.
            min_consistent_stranger_votes: Number of non-matching observations needed to label UNKNOWN.
            timeout_seconds: Lifespan threshold after which PENDING tracks without usable faces become UNKNOWN.
            recheck_interval_seconds: Minimum interval between face recognition checks on the same track.
            no_usable_face_backoff_multiplier: When a track is UNKNOWN with reason no_usable_face,
                multiply recheck_interval by this factor to slow down futile GPU submissions.
        """
        self.min_matches = int(min_consistent_matches)
        self.min_stranger_votes = int(min_consistent_stranger_votes)
        self.timeout_seconds = float(timeout_seconds)
        self.recheck_interval = float(recheck_interval_seconds)
        self.no_usable_face_backoff_multiplier = float(no_usable_face_backoff_multiplier)
        self._records: Dict[int, TrackIdentityRecord] = {}

    def get_or_create(self, track_id: int, timestamp: float) -> TrackIdentityRecord:
        """Retrieve existing track identity record or initialize new one."""
        if track_id not in self._records:
            self._records[track_id] = TrackIdentityRecord(
                track_id=track_id,
                first_seen=timestamp,
                last_seen=timestamp,
            )
        else:
            self._records[track_id].last_seen = timestamp
        return self._records[track_id]

    def should_process_face(self, track_id: int, timestamp: float) -> bool:
        """Determine whether this track is eligible for a new face inference job."""
        record = self.get_or_create(track_id, timestamp)

        # Fix 3: Once KNOWN, stop all re-checks — identity is confirmed and stable.
        if record.status == IdentityStatus.KNOWN:
            return False

        # Fix 2: UNKNOWN with no_usable_face — slow down re-checks with backoff.
        # Once a track is marked UNKNOWN because it never produced a usable face crop
        # (e.g. the person is facing away or is very small), subsequent re-submissions
        # only waste GPU SM time. Apply a configurable backoff multiplier to the interval.
        if record.status == IdentityStatus.UNKNOWN and record.decision_reason == "no_usable_face":
            if not record.observations:
                return True
            last_obs_time = record.observations[-1].timestamp
            backoff_interval = self.recheck_interval * self.no_usable_face_backoff_multiplier
            return (timestamp - last_obs_time) >= backoff_interval

        if not record.observations:
            return True

        last_obs_time = record.observations[-1].timestamp
        return (timestamp - last_obs_time) >= self.recheck_interval

    def record_attempt_without_usable_face(
        self,
        track_id: int,
        timestamp: float,
        reason: str,
    ) -> None:
        """Record that a frame crop was examined but no quality face was found."""
        record = self.get_or_create(track_id, timestamp)
        record.face_attempts += 1

    def update_observation(
        self,
        track_id: int,
        timestamp: float,
        match_result: MatchResult,
        face_crop: Optional[np.ndarray] = None,
        quality_passed: bool = True,
    ) -> Identity:
        """Incorporate a new face recognition observation into the track's state machine."""
        record = self.get_or_create(track_id, timestamp)
        record.face_attempts += 1
        if quality_passed:
            record.quality_passes += 1

        obs = FaceObservation(
            timestamp=timestamp,
            status=match_result.status,
            person_id=match_result.person_id,
            name=match_result.name,
            category=match_result.category,
            authorised_zone_ids=match_result.authorised_zone_ids,
            score=match_result.score,
            margin=match_result.margin,
            reason=match_result.reason,
            quality_passed=quality_passed,
            face_crop=face_crop,
        )
        record.observations.append(obs)

        # Update best face crop for evidence
        if face_crop is not None and match_result.score >= record.best_score:
            record.best_face_crop = face_crop
            record.best_score = match_result.score
            record.best_margin = match_result.margin

        # State transition logic
        if record.status == IdentityStatus.PENDING:
            # Check for KNOWN transition
            if match_result.status == IdentityStatus.KNOWN:
                # Count matching observations for this specific person_id
                person_id = match_result.person_id
                matching_count = sum(
                    1
                    for o in record.observations
                    if o.status == IdentityStatus.KNOWN and o.person_id == person_id
                )
                if matching_count >= self.min_matches:
                    record.status = IdentityStatus.KNOWN
                    record.person_id = person_id
                    record.name = match_result.name
                    record.category = match_result.category
                    record.authorised_zone_ids = match_result.authorised_zone_ids
                    record.decision_reason = f"consensus_matched_{matching_count}_times"
                    record.decision_timestamp = timestamp

            elif match_result.status == IdentityStatus.UNKNOWN:
                # Count unknown votes
                unknown_count = sum(
                    1 for o in record.observations if o.status == IdentityStatus.UNKNOWN
                )
                if unknown_count >= self.min_stranger_votes:
                    record.status = IdentityStatus.UNKNOWN
                    record.decision_reason = f"consensus_unknown_{unknown_count}_votes"
                    record.decision_timestamp = timestamp

        elif record.status == IdentityStatus.KNOWN:
            # Hysteresis: Keep identity unless overwhelming contradictory evidence appears
            if match_result.status == IdentityStatus.KNOWN and match_result.person_id == record.person_id:
                if match_result.score > record.best_score:
                    record.best_score = match_result.score
                    record.best_margin = match_result.margin

        return self.get_identity(track_id, timestamp)

    def evaluate_timeouts(self, active_track_ids: Sequence[int], current_timestamp: float) -> None:
        """Check all active PENDING tracks for timeout (e.g., person facing away or CCTV low-res)."""
        for tid in active_track_ids:
            record = self._records.get(tid)
            if record is None or record.status != IdentityStatus.PENDING:
                continue

            track_age = current_timestamp - record.first_seen
            if track_age >= self.timeout_seconds:
                # Track has lived beyond timeout without achieving KNOWN
                if record.quality_passes == 0:
                    record.status = IdentityStatus.UNKNOWN
                    record.decision_reason = "no_usable_face"
                    record.decision_timestamp = current_timestamp
                else:
                    record.status = IdentityStatus.UNKNOWN
                    record.decision_reason = "timeout_without_identity_match"
                    record.decision_timestamp = current_timestamp

    def finalize_track(self, track_id: int, current_timestamp: Optional[float] = None) -> Identity:
        """Finalize identity for a track that is terminating or leaving the field of view.

        If still PENDING, cleanly resolves to UNKNOWN so no terminated tracks linger in PENDING.
        """
        record = self._records.get(track_id)
        if record is not None and record.status == IdentityStatus.PENDING:
            t = current_timestamp if current_timestamp is not None else record.last_seen
            record.status = IdentityStatus.UNKNOWN
            if record.quality_passes == 0:
                record.decision_reason = "no_usable_face"
            else:
                record.decision_reason = "terminated_without_identity_match"
            record.decision_timestamp = t
        return self.get_identity(track_id, current_timestamp)

    def get_identity(self, track_id: int, timestamp: Optional[float] = None) -> Identity:
        """Return the current consolidated Identity object for a given track."""
        if track_id not in self._records:
            return Identity(
                status=IdentityStatus.PENDING,
                reason="no_history",
            )

        r = self._records[track_id]
        if timestamp is not None:
            r.last_seen = timestamp

        return Identity(
            status=r.status,
            person_id=r.person_id,
            name=r.name,
            confidence=r.best_score,
            margin=r.best_margin,
            observation_count=len(r.observations),
            reason=r.decision_reason,
        )

    def attach_to_track(self, track: Track, timestamp: float) -> None:
        """Attach identity status, name, and person_id directly onto a Track instance."""
        ident = self.get_identity(track.track_id, timestamp)
        track.identity_status = ident.status
        track.person_id = ident.person_id
        track.person_name = ident.name
        track.identity_confidence = ident.confidence
        track.identity_margin = ident.margin

    def is_authorised_for_zone(self, track_id: int, zone_id: Union[int, str]) -> bool:
        """Check whether a person is KNOWN and explicitly authorized for the specified zone."""
        record = self._records.get(track_id)
        if record is None or record.status != IdentityStatus.KNOWN:
            return False

        # If zone_id is string or int, compare flexibly
        zid_str = str(zone_id)
        auth_zones = [str(z) for z in record.authorised_zone_ids]
        return zid_str in auth_zones

    def prune_old_tracks(self, active_track_ids: Sequence[int], max_age_seconds: float = 60.0) -> None:
        """Clean up records for tracks that have disappeared from the scene."""
        active_set = set(active_track_ids)
        for tid in list(self._records.keys()):
            if tid not in active_set:
                rec = self._records[tid]
                if (rec.last_seen - rec.first_seen) > max_age_seconds:
                    del self._records[tid]
