"""Central Evidence Collector for the Intelligent CCTV Surveillance System.

Orchestrates rolling frame buffers per camera, extracts person/object/face crops,
assembles video clips with pre-roll and post-roll, creates Evidence records,
and persists them to SQLite via StorageRepository.

Conforms strictly to docs/05_THREAT_ENGINE_SPEC.md, docs/07_DATABASE_SCHEMA.md,
and docs/10_TEST_PLAN.md (T-23, T-24, T-25).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.common.types import (
    Evidence,
    EvidenceType,
    Incident,
    Severity,
)
from src.evidence.clip_writer import ClipWriter
from src.evidence.crop_extractor import CropExtractor
from src.evidence.ring_buffer import RollingFrameBuffer

logger = logging.getLogger(__name__)

SEVERITY_RANK: Dict[str, int] = {
    Severity.LOW.value: 1,
    Severity.MEDIUM.value: 2,
    Severity.HIGH.value: 3,
    Severity.CRITICAL.value: 4,
}


class EvidenceCollector:
    """Collects snapshots and video clips for correlated security incidents."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        repository: Optional[Any] = None,
    ) -> None:
        """Initialize the EvidenceCollector.

        Args:
            config: Optional evidence configuration dictionary.
            repository: Optional StorageRepository instance for persistence.
        """
        cfg = config or {}
        ring_cfg = cfg.get("ring_buffer", {})
        clip_cfg = cfg.get("video_clip", {})
        policy_cfg = cfg.get("policy", {})

        self.buffer_duration: float = float(ring_cfg.get("buffer_duration_seconds", 30.0))
        self.max_buffer_memory_mb: float = float(ring_cfg.get("max_buffer_memory_mb", 250.0))
        self.jpeg_quality: int = int(ring_cfg.get("jpeg_quality", 85))

        self.pre_roll: float = float(clip_cfg.get("pre_roll_seconds", 10.0))
        self.post_roll: float = float(clip_cfg.get("post_roll_seconds", 10.0))
        self.max_clip_length: float = float(clip_cfg.get("max_clip_length_seconds", 60.0))
        self.fps: float = float(clip_cfg.get("fps", 25.0))
        self.fourcc: str = str(clip_cfg.get("fourcc", "mp4v"))

        self.min_level_for_clip: str = policy_cfg.get("min_level_for_clip", "Medium")
        self.min_free_disk_mb: float = float(policy_cfg.get("min_free_disk_mb", 500.0))
        self.evidence_root: Path = Path(policy_cfg.get("evidence_dir", "evidence"))

        self.repository = repository
        self.crop_extractor = CropExtractor()
        self.clip_writer = ClipWriter(
            fps=self.fps,
            fourcc=self.fourcc,
            min_free_disk_mb=self.min_free_disk_mb,
        )

        # Camera buffers: camera_id -> RollingFrameBuffer
        self._camera_buffers: Dict[str, RollingFrameBuffer] = {}

    def get_or_create_buffer(self, camera_id: str) -> RollingFrameBuffer:
        """Get or initialize the RollingFrameBuffer for a camera."""
        if camera_id not in self._camera_buffers:
            self._camera_buffers[camera_id] = RollingFrameBuffer(
                buffer_duration_seconds=self.buffer_duration,
                max_buffer_memory_mb=self.max_buffer_memory_mb,
                jpeg_quality=self.jpeg_quality,
            )
        return self._camera_buffers[camera_id]

    def add_frame(
        self, camera_id: str, frame: np.ndarray, timestamp: float, frame_id: int
    ) -> None:
        """Push a processed frame into the camera's rolling buffer."""
        buf = self.get_or_create_buffer(camera_id)
        buf.add_frame(frame, timestamp, frame_id)

    def collect_for_incident(
        self,
        incident: Incident,
        current_frame: Optional[np.ndarray] = None,
        person_bbox: Optional[Tuple[float, float, float, float]] = None,
        object_bbox: Optional[Tuple[float, float, float, float]] = None,
        best_face_crop: Optional[np.ndarray] = None,
        generate_clip: bool = True,
    ) -> List[Evidence]:
        """Collect visual evidence for an active or closed incident without redundant re-encoding."""
        incident_dir = self.evidence_root / "incidents" / incident.incident_no
        incident_dir.mkdir(parents=True, exist_ok=True)

        collected: List[Evidence] = []
        buf = self.get_or_create_buffer(incident.camera_id)

        # Existing evidence types with valid files
        existing_types = {e.type: e for e in incident.evidence if e.file_path}

        # Use current_frame if provided, else retrieve latest from buffer
        frame_for_crops = current_frame
        if frame_for_crops is None:
            latest = buf.get_latest_frame()
            if latest:
                frame_for_crops = latest[2]

        subj = incident.subject or {}
        track_id = subj.get("track_id")

        # 1. Person Snapshot (T-23) - Collect once per incident
        if EvidenceType.PERSON_IMAGE.value not in existing_types:
            if frame_for_crops is not None and person_bbox is not None:
                person_path = incident_dir / f"person_track_{track_id or 'unknown'}.jpg"
                p_res = self.crop_extractor.extract_crop(
                    frame_for_crops, person_bbox, person_path
                )
                if p_res:
                    ev_person = Evidence(
                        incident_id=incident.id,
                        type=EvidenceType.PERSON_IMAGE.value,
                        file_path=p_res,
                        timestamp=incident.last_event_time,
                        location={"place": incident.place, "bbox": person_bbox},
                        details={"track_id": track_id},
                    )
                    collected.append(ev_person)

        # 2. Object Snapshot (e.g. abandoned bag) - Collect once per incident
        if EvidenceType.OBJECT_IMAGE.value not in existing_types:
            if frame_for_crops is not None and object_bbox is not None:
                obj_id = subj.get("object_id") or "bag"
                obj_path = incident_dir / f"object_{obj_id}.jpg"
                o_res = self.crop_extractor.extract_crop(
                    frame_for_crops, object_bbox, obj_path
                )
                if o_res:
                    ev_obj = Evidence(
                        incident_id=incident.id,
                        type=EvidenceType.OBJECT_IMAGE.value,
                        file_path=o_res,
                        timestamp=incident.last_event_time,
                        location={"place": incident.place, "bbox": object_bbox},
                        details={"object_id": obj_id},
                    )
                    collected.append(ev_obj)

        # 3. Face Snapshot (T-23 & T-25) - Collect once per incident
        if EvidenceType.FACE_IMAGE.value not in existing_types:
            if best_face_crop is not None and best_face_crop.size > 0:
                face_path = incident_dir / f"face_track_{track_id or 'unknown'}.jpg"
                f_res = self.crop_extractor.save_face_crop(best_face_crop, face_path)
                if f_res:
                    ev_face = Evidence(
                        incident_id=incident.id,
                        type=EvidenceType.FACE_IMAGE.value,
                        file_path=f_res,
                        timestamp=incident.last_event_time,
                        location={"place": incident.place},
                        details={
                            "track_id": track_id,
                            "identity_status": subj.get("identity_status"),
                            "person_name": subj.get("person_name"),
                        },
                    )
                    collected.append(ev_face)
            elif not any(e.type == EvidenceType.FACE_IMAGE.value for e in incident.evidence):
                # T-25: Face not visible -> Incident still saved, marked unavailable
                ev_face_unavail = Evidence(
                    incident_id=incident.id,
                    type=EvidenceType.FACE_IMAGE.value,
                    file_path="",
                    timestamp=incident.last_event_time,
                    location={"place": incident.place},
                    details={
                        "status": "unavailable",
                        "reason": "Face not visible or usable in camera view",
                    },
                )
                collected.append(ev_face_unavail)

        # 4. Video Clip (T-23, T-24) - Encode only when generate_clip=True (e.g. on alert or close)
        inc_rank = SEVERITY_RANK.get(incident.threat_level, 1)
        min_rank = SEVERITY_RANK.get(self.min_level_for_clip, 2)

        if generate_clip and inc_rank >= min_rank:
            # Calculate clip window: [start_time - pre_roll, last_event_time + post_roll]
            clip_start = max(0.0, incident.start_time - self.pre_roll)
            clip_end = incident.last_event_time + self.post_roll
            if (clip_end - clip_start) > self.max_clip_length:
                clip_start = max(0.0, clip_end - self.max_clip_length)

            clip_frames = buf.get_frames_in_range(clip_start, clip_end)
            if clip_frames:
                clip_path = incident_dir / f"clip_{incident.incident_no}.mp4"
                c_res = self.clip_writer.write_clip(clip_frames, clip_path)
                if c_res:
                    # Update existing clip record if already present
                    existing_clip = next((e for e in incident.evidence if e.type == EvidenceType.VIDEO_CLIP.value), None)
                    if existing_clip is not None:
                        existing_clip.file_path = c_res
                        existing_clip.timestamp = incident.last_event_time
                        existing_clip.details = {
                            "start_time": clip_start,
                            "end_time": clip_end,
                            "frame_count": len(clip_frames),
                            "fps": self.fps,
                        }
                    else:
                        ev_clip = Evidence(
                            incident_id=incident.id,
                            type=EvidenceType.VIDEO_CLIP.value,
                            file_path=c_res,
                            timestamp=incident.last_event_time,
                            location={"place": incident.place},
                            details={
                                "start_time": clip_start,
                                "end_time": clip_end,
                                "frame_count": len(clip_frames),
                                "fps": self.fps,
                            },
                        )
                        collected.append(ev_clip)

        # Attach to incident and persist to repository if available
        incident.evidence.extend(collected)

        if self.repository is not None and incident.id is not None:
            for ev in collected:
                if ev.file_path:  # Persist actual evidence artifacts
                    try:
                        ev.incident_id = incident.id
                        self.repository.add_evidence(ev)
                    except Exception as e:
                        logger.error(
                            f"Failed to persist evidence for incident {incident.incident_no}: {e}"
                        )

        return collected
