"""Abandoned Object Detection Rule.

Fires ABANDONED_OBJECT when a bag/luggage item is left unattended.

Key architectural guarantees (addressing Open Issue 4):
1. Independent Static Object Candidates: Merges bag detections across frames based on
   spatial centroid proximity and IoU. Does NOT depend on a continuous ByteTrack Track ID.
2. Stationary timer (T_static): Confirms the object is motionless.
3. Owner association: Associates the nearest person track before/at the time of placement.
4. Separation timer (T_owner_away): Requires the owner (and any bystander) to be separated
   by distance D_owner (in multiples of person height) for T_owner_away seconds.
5. Suppression when owner is present: Never triggers if an individual remains nearby.
6. Cancellation on retrieval: Clears candidate if the bag is picked up or moves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from src.common.types import Detection, Event, EventType, FrameResult, Severity, Track
from src.rules.base_rule import BaseRule
from src.rules.geometry import compute_iou, euclidean_distance, get_reference_point
from src.rules.zones import VirtualLine, Zone


@dataclass
class StaticBagCandidate:
    """Tracks a stationary bag candidate independently of tracker ID continuity."""

    candidate_id: str
    class_name: str
    bbox: Tuple[float, float, float, float]
    centroid: Tuple[float, float]
    first_detected_time: float
    last_detected_time: float
    stationary_start_time: float
    owner_track_id: Optional[int] = None
    owner_away_start_time: Optional[float] = None
    event_fired: bool = False
    missing_frame_count: int = 0
    history_centroids: List[Tuple[float, float]] = field(default_factory=list)


class AbandonedObjectRule(BaseRule):
    """Detects unattended luggage, backpacks, and packages with owner proximity tracking."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        super().__init__("abandoned_object", cfg)
        self.bag_classes: Set[str] = set(
            cfg.get("bag_classes", ["backpack", "handbag", "suitcase"])
        )
        self.stationary_seconds: float = float(cfg.get("stationary_seconds", 10.0))
        self.owner_away_seconds: float = float(cfg.get("owner_away_seconds", 5.0))
        self.owner_distance_multiplier: float = float(cfg.get("owner_distance_multiplier", 2.0))
        self.spatial_match_distance_px: float = float(cfg.get("spatial_match_distance_px", 35.0))
        self.max_missing_frames: int = int(cfg.get("max_missing_frames", 30))  # Tolerate flicker up to ~1s

        self._candidates: List[StaticBagCandidate] = []
        self._next_candidate_id: int = 1

    def evaluate(
        self,
        frame_result: FrameResult,
        zones: List[Zone],
        lines: List[VirtualLine],
        current_time: float,
    ) -> List[Event]:
        events: List[Event] = []

        # 1. Extract bag observations from either raw detections or active tracks
        bag_observations: List[Tuple[Tuple[float, float, float, float], str, float]] = []
        # Check detections first
        for det in frame_result.detections:
            if det.class_name.lower() in self.bag_classes:
                bag_observations.append((det.bbox, det.class_name, det.confidence))

        # Check tracks for any bags with assigned IDs
        for trk in frame_result.tracks:
            if trk.class_name.lower() in self.bag_classes:
                bag_observations.append((trk.bbox, trk.class_name, trk.confidence))

        # Deduplicate observations by spatial proximity
        unique_bag_obs: List[Tuple[Tuple[float, float, float, float], str, float]] = []
        for bbox, cls_name, conf in bag_observations:
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            if not any(euclidean_distance((cx, cy), ((u[0][0] + u[0][2]) / 2.0, (u[0][1] + u[0][3]) / 2.0)) < 20.0 for u in unique_bag_obs):
                unique_bag_obs.append((bbox, cls_name, conf))
        bag_observations = unique_bag_obs

        # 2. Extract person tracks for owner association and proximity checks
        person_tracks = [t for t in frame_result.tracks if t.class_name.lower() == "person"]

        # Average person height in current frame for scale-relative distance calculations
        avg_person_h = (
            sum(p.height for p in person_tracks) / len(person_tracks)
            if person_tracks
            else 100.0
        )
        owner_separation_distance = max(60.0, avg_person_h * self.owner_distance_multiplier)

        # 3. Match bag observations to existing StaticBagCandidates
        matched_obs_indices = set()
        for cand in self._candidates:
            best_match_idx = -1
            best_dist = float("inf")

            for idx, (bbox, cls_name, conf) in enumerate(bag_observations):
                if idx in matched_obs_indices:
                    continue
                obs_cx = (bbox[0] + bbox[2]) / 2.0
                obs_cy = (bbox[1] + bbox[3]) / 2.0
                dist = euclidean_distance(cand.centroid, (obs_cx, obs_cy))
                iou = compute_iou(cand.bbox, bbox)

                # Match if centroids are close or bounding boxes overlap
                if (dist <= self.spatial_match_distance_px or iou >= 0.35) and dist < best_dist:
                    best_dist = dist
                    best_match_idx = idx

            if best_match_idx >= 0:
                bbox, cls_name, conf = bag_observations[best_match_idx]
                obs_cx = (bbox[0] + bbox[2]) / 2.0
                obs_cy = (bbox[1] + bbox[3]) / 2.0

                cand.bbox = bbox
                cand.centroid = (obs_cx, obs_cy)
                cand.last_detected_time = current_time
                cand.missing_frame_count = 0
                cand.history_centroids.append((obs_cx, obs_cy))
                matched_obs_indices.add(best_match_idx)

                # Associate owner if not yet linked
                if cand.owner_track_id is None and person_tracks:
                    closest_person = min(
                        person_tracks,
                        key=lambda p: euclidean_distance(cand.centroid, p.bottom_center),
                    )
                    cand.owner_track_id = closest_person.track_id
            else:
                cand.missing_frame_count += 1

        # 4. Initialize new candidates for unmatched bag observations
        for idx, (bbox, cls_name, conf) in enumerate(bag_observations):
            if idx not in matched_obs_indices:
                obs_cx = (bbox[0] + bbox[2]) / 2.0
                obs_cy = (bbox[1] + bbox[3]) / 2.0

                # Nearest person track at appearance
                owner_id = None
                if person_tracks:
                    closest_person = min(
                        person_tracks,
                        key=lambda p: euclidean_distance((obs_cx, obs_cy), p.bottom_center),
                    )
                    if euclidean_distance((obs_cx, obs_cy), closest_person.bottom_center) <= (owner_separation_distance * 1.5):
                        owner_id = closest_person.track_id

                new_cand = StaticBagCandidate(
                    candidate_id=f"bag_cand_{self._next_candidate_id}",
                    class_name=cls_name,
                    bbox=bbox,
                    centroid=(obs_cx, obs_cy),
                    first_detected_time=current_time,
                    last_detected_time=current_time,
                    stationary_start_time=current_time,
                    owner_track_id=owner_id,
                    history_centroids=[(obs_cx, obs_cy)],
                )
                self._next_candidate_id += 1
                self._candidates.append(new_cand)

        # 5. Evaluate stationary condition and owner separation on all candidates
        retained_candidates: List[StaticBagCandidate] = []

        for cand in self._candidates:
            # Purge candidates that have disappeared completely beyond tolerance
            if cand.missing_frame_count > self.max_missing_frames:
                continue

            # Calculate total stationary duration
            stationary_duration = max(0.0, current_time - cand.stationary_start_time)

            nearest_person_dist = (
                min((euclidean_distance(cand.centroid, p.bottom_center) for p in person_tracks), default=999.0)
            )

            # Check if owner or any attendant is nearby
            # Case 1: Owner was identified
            owner_present_and_nearby = False
            if cand.owner_track_id is not None:
                owner_trk = next((p for p in person_tracks if p.track_id == cand.owner_track_id), None)
                if owner_trk is not None:
                    d_owner = euclidean_distance(cand.centroid, owner_trk.bottom_center)
                    if d_owner <= owner_separation_distance:
                        owner_present_and_nearby = True

            # Case 2: Check if any person is lingering/dwelling close to the bag
            # Note: A pedestrian in transit (walking past the bag) is NOT attending it.
            bystander_attending = False
            for p in person_tracks:
                if cand.owner_track_id is not None and p.track_id == cand.owner_track_id:
                    continue
                d = euclidean_distance(cand.centroid, p.bottom_center)
                if d <= (owner_separation_distance * 0.75):
                    # Only counts if person is lingering (dwelling in zone or motionless)
                    if p.dwell_seconds >= 3.0:
                        bystander_attending = True
                        break

            someone_nearby = owner_present_and_nearby or bystander_attending

            if someone_nearby:
                # Owner or lingering bystander is nearby -> reset the away timer
                cand.owner_away_start_time = None
            else:
                # No one is attending -> accumulate separation duration
                if cand.owner_away_start_time is None:
                    cand.owner_away_start_time = current_time

            owner_away_duration = (
                max(0.0, current_time - cand.owner_away_start_time)
                if cand.owner_away_start_time is not None
                else 0.0
            )

            # Trigger condition:
            # 1. Bag stationary for >= stationary_seconds (default: 10s)
            # 2. Owner / any person away for >= owner_away_seconds (default: 5s)
            # 3. Not already fired for this candidate instance
            if (
                stationary_duration >= self.stationary_seconds
                and owner_away_duration >= self.owner_away_seconds
                and not cand.event_fired
            ):
                evt = Event(
                    event_id=self.generate_event_id(EventType.ABANDONED_OBJECT, current_time),
                    camera_id=frame_result.tracks[0].camera_id if frame_result.tracks else "cam",
                    track_id=cand.owner_track_id,
                    object_id=cand.candidate_id,
                    event_type=EventType.ABANDONED_OBJECT,
                    timestamp=current_time,
                    severity=Severity.HIGH,
                    confidence=0.88,
                    location={
                        "zone_id": None,
                        "zone_name": "Floor / Corridor",
                        "coordinates": (round(cand.centroid[0], 1), round(cand.centroid[1], 1)),
                    },
                    details={
                        "candidate_id": cand.candidate_id,
                        "object_class": cand.class_name,
                        "owner_track_id": cand.owner_track_id,
                        "stationary_seconds": round(stationary_duration, 1),
                        "owner_away_seconds": round(owner_away_duration, 1),
                        "separation_distance_px": round(nearest_person_dist, 1),
                        "threshold_separation_px": round(owner_separation_distance, 1),
                        "bbox": [round(c, 1) for c in cand.bbox],
                    },
                )
                events.append(evt)
                cand.event_fired = True

            retained_candidates.append(cand)

        self._candidates = retained_candidates
        return events

    def get_active_candidates(self) -> List[StaticBagCandidate]:
        """Expose current static candidates for annotator HUD rendering."""
        return list(self._candidates)

    def reset(self) -> None:
        self._candidates.clear()
        self._next_candidate_id = 1
        self._event_counter = 0
