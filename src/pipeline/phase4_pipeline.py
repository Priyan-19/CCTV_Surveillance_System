"""Phase 4 Pipeline and Visual Annotator module.

Integrates:
- YOLOv8/26 object detection + ByteTrack multi-object tracking.
- Spatial Zone and Virtual Line Rule Engine.
- Asynchronous InsightFace face recognition worker.
- Threat Engine (Incident correlation, deterministic scoring, lifecycle management).
- Evidence Collector (Rolling RAM frame buffer, person/object/face crops, MP4 video clips).
- SQLite storage persistence and alert dispatching.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import cv2
import numpy as np
import torch
import yaml

from src.capture.video_source import CapturedFrame, VideoSource
from src.common.metrics import PerformanceMetrics
from src.common.types import (
    Alert,
    Event,
    Evidence,
    EvidenceType,
    FrameResult,
    IdentityStatus,
    Incident,
    Severity,
    Track,
)
from src.evidence.collector import EvidenceCollector
from src.face.face_worker import FaceWorker
from src.face.identity import IdentityManager
from src.face.matcher import FaceMatcher
from src.face.quality import FaceQualityFilter
from src.face.store import FaceStore
from src.pipeline.phase3_pipeline import Phase3Annotator
from src.rules.abandoned_object import StaticBagCandidate
from src.rules.rule_engine import RuleEngine
from src.rules.zones import VirtualLine, Zone, ZoneManager
from src.storage.repository import StorageRepository
from src.threat.incident_manager import IncidentManager
from src.tracking.tracker import BaseTracker, ByteTrackTracker

logger = logging.getLogger("phase4_pipeline")


class Phase4Annotator(Phase3Annotator):
    """Visual annotator extending Phase 3 with active Incident and Threat Score HUD."""

    THREAT_COLORS: Dict[str, Tuple[int, int, int]] = {
        Severity.LOW.value: (180, 180, 180),        # Slate Gray
        Severity.MEDIUM.value: (0, 215, 255),      # Amber Yellow
        Severity.HIGH.value: (0, 120, 255),        # Deep Orange
        Severity.CRITICAL.value: (0, 0, 240),      # Bright Red
    }

    def annotate_phase4(
        self,
        frame: np.ndarray,
        tracks: Sequence[Track],
        zones: List[Zone],
        lines: List[VirtualLine],
        fps: float,
        latency_ms: float,
        rules_ms: float,
        threat_ms: float,
        frame_id: int,
        current_time: float,
        active_incidents: List[Incident],
        dispatched_alerts_count: int,
        face_stats: Optional[Dict[str, Any]] = None,
        static_candidates: Optional[List[StaticBagCandidate]] = None,
    ) -> np.ndarray:
        """Render Phase 3 overlays plus Phase 4 Threat Incident banners and metrics."""
        # Base Phase 3 annotation (tracks, zones, identities, motion trails)
        out_frame = self.annotate_phase3(
            frame=frame,
            tracks=tracks,
            zones=zones,
            lines=lines,
            fps=fps,
            latency_ms=latency_ms,
            rules_ms=rules_ms,
            frame_id=frame_id,
            current_time=current_time,
            face_stats=face_stats,
            static_candidates=static_candidates,
        )

        h, w = out_frame.shape[:2]

        # Top-Right: Active Incident & Threat Score Card
        if active_incidents:
            # Pick highest severity active incident
            top_incident = max(active_incidents, key=lambda x: x.threat_score)
            color = self.THREAT_COLORS.get(top_incident.threat_level, (0, 0, 240))

            card_w = 340
            card_h = 86
            card_x = w - card_w - 12
            card_y = 12

            # Semi-transparent background
            overlay = out_frame.copy()
            cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.8, out_frame, 0.2, 0, out_frame)

            # Left color accent strip
            cv2.rectangle(out_frame, (card_x, card_y), (card_x + 8, card_y + card_h), color, -1)

            # Card Header
            header_text = f"THREAT: {top_incident.threat_level.upper()} ({top_incident.threat_score}/100)"
            cv2.putText(out_frame, header_text, (card_x + 16, card_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)

            inc_sub = f"Inc #{top_incident.incident_no} | Place: {top_incident.place[:16]}"
            cv2.putText(out_frame, inc_sub, (card_x + 16, card_y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

            act_str = f"Acts: {', '.join(top_incident.activities[:3])}"
            cv2.putText(out_frame, act_str, (card_x + 16, card_y + 64), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (190, 210, 230), 1, cv2.LINE_AA)

            # Score bar meter
            bar_w = 120
            bar_h = 8
            bx = card_x + card_w - bar_w - 14
            by = card_y + 14
            cv2.rectangle(out_frame, (bx, by), (bx + bar_w, by + bar_h), (60, 60, 60), -1)
            fill_w = int(bar_w * (top_incident.threat_score / 100.0))
            if fill_w > 0:
                cv2.rectangle(out_frame, (bx, by), (bx + fill_w, by + bar_h), color, -1)

        # Extended Telemetry in Bottom-Left HUD
        hud_extra_y = h - 35
        cv2.putText(
            out_frame,
            f"Threat Engine: {threat_ms:.2f}ms | Active Incs: {len(active_incidents)} | Alerts Sent: {dispatched_alerts_count}",
            (18, hud_extra_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 230, 255),
            1,
            cv2.LINE_AA,
        )

        return out_frame


class Phase4Pipeline:
    """Full Phase 4 Surveillance Pipeline integrating Threat Engine & Evidence Collector."""

    def __init__(
        self,
        camera_id: str,
        source: Union[str, int],
        tracker: Optional[BaseTracker] = None,
        model_name: str = "yolo26s.pt",
        tracker_config: str = "configs/bytetrack.yaml",
        rules_config: str = "configs/rules.yaml",
        zones_config: str = "configs/zones.yaml",
        face_config: str = "configs/face.yaml",
        threat_config: str = "configs/threat.yaml",
        alerts_config: str = "configs/alerts.yaml",
        evidence_config: str = "configs/evidence.yaml",
        db_path: str = "data/cctv.db",
        device: str = "cuda:0",
        half_precision: bool = True,
        imgsz: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.70,
        tracked_classes: Optional[List[int]] = None,
        output_dir: Union[str, Path] = "data/output",
        events_dir: Union[str, Path] = "data/output/events",
        save_video: bool = True,
        show_window: bool = False,
    ) -> None:
        self.camera_id = camera_id
        self.source_path = source
        self.model_name = model_name
        self.tracker_config = tracker_config
        self.rules_config = rules_config
        self.zones_config = zones_config
        self.face_config_path = face_config
        self.threat_config_path = threat_config
        self.alerts_config_path = alerts_config
        self.evidence_config_path = evidence_config
        self.db_path = db_path
        self.device = device
        self.half_precision = half_precision
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.tracked_classes = tracked_classes
        self.output_dir = Path(output_dir)
        self.events_dir = Path(events_dir)
        self.save_video = save_video
        self.show_window = show_window

        # 1. Video Source
        self.video_source = VideoSource(
            camera_id=self.camera_id,
            source=self.source_path,
        )

        # 2. Tracker
        if tracker is not None:
            self.tracker = tracker
        else:
            self.tracker = ByteTrackTracker(
                model_name=self.model_name,
                tracker_config=self.tracker_config,
                device=self.device,
                half_precision=self.half_precision,
                imgsz=self.imgsz,
                conf_threshold=self.conf_threshold,
                iou_threshold=self.iou_threshold,
                tracked_classes=self.tracked_classes,
            )

        # 3. Storage Repository
        self.repository = StorageRepository(db_path=self.db_path)
        self.repository.register_camera(self.camera_id, location=f"Camera {self.camera_id}")

        # 4. Face Recognition Subsystem
        with open(self.face_config_path, "r", encoding="utf-8") as f:
            self.face_cfg = yaml.safe_load(f)

        self.store = FaceStore(db_path=self.db_path)
        m_cfg = self.face_cfg.get("matcher", {})
        self.matcher = FaceMatcher(
            similarity_threshold=m_cfg.get("similarity_threshold", 0.45),
            required_margin=m_cfg.get("required_margin", 0.08),
        )
        self.matcher.set_gallery(self.store.get_gallery())

        q_cfg = self.face_cfg.get("quality", {})
        self.quality_filter = FaceQualityFilter(
            min_face_size=q_cfg.get("min_face_size", 36),
            min_blur_score=q_cfg.get("min_blur_score", 25.0),
            min_det_score=q_cfg.get("min_det_score", 0.55),
        )

        sm_cfg = self.face_cfg.get("state_machine", {})
        self.identity_manager = IdentityManager(
            min_consistent_matches=sm_cfg.get("min_consistent_matches", 2),
            min_consistent_stranger_votes=sm_cfg.get("min_consistent_stranger_votes", 2),
            timeout_seconds=sm_cfg.get("timeout_seconds", 2.0),
            recheck_interval_seconds=sm_cfg.get("recheck_interval_seconds", 1.0),
            no_usable_face_backoff_multiplier=sm_cfg.get("no_usable_face_backoff_multiplier", 3.0),
        )

        w_cfg = self.face_cfg.get("worker", {})
        model_cfg = self.face_cfg.get("model", {})
        self.face_worker = FaceWorker(
            matcher=self.matcher,
            identity_manager=self.identity_manager,
            quality_filter=self.quality_filter,
            model_name=model_cfg.get("name", "buffalo_s"),
            det_size=tuple(model_cfg.get("det_size", [320, 320])),
            queue_size=w_cfg.get("queue_size", 16),
            drop_policy=w_cfg.get("drop_policy", "drop_oldest"),
            ctx_id=model_cfg.get("ctx_id", 0),
            providers=model_cfg.get("providers", ["CUDAExecutionProvider", "CPUExecutionProvider"]),
        )

        # 5. Zone Manager & Rule Engine
        self.zone_manager = ZoneManager(config_path=self.zones_config)
        self.rule_engine = RuleEngine(
            rules_config_path=self.rules_config,
            zone_manager=self.zone_manager,
        )

        # 6. Threat Engine & Incident Manager
        with open(self.threat_config_path, "r", encoding="utf-8") as f:
            self.threat_cfg = yaml.safe_load(f)
        with open(self.alerts_config_path, "r", encoding="utf-8") as f:
            self.alerts_cfg = yaml.safe_load(f)

        self.incident_manager = IncidentManager(
            threat_config=self.threat_cfg,
            alerts_config=self.alerts_cfg,
            repository=self.repository,
        )

        # 7. Evidence Collector
        with open(self.evidence_config_path, "r", encoding="utf-8") as f:
            self.evidence_cfg = yaml.safe_load(f)

        self.evidence_collector = EvidenceCollector(
            config=self.evidence_cfg,
            repository=self.repository,
        )

        # 8. Annotator & Profiler
        self.annotator = Phase4Annotator(
            camera_id=self.camera_id,
            model_name=self.model_name,
            tracker_name="ByteTrack",
        )
        self.metrics = PerformanceMetrics()
        self._writer: Optional[cv2.VideoWriter] = None
        self._output_video_path: Optional[Path] = None
        self._events_file_path: Optional[Path] = None
        self._logged_events: List[Event] = []
        self._dispatched_alerts: List[Alert] = []

        # Face throttle state (token-bucket rate limiter)
        q_cfg2 = self.face_cfg.get("quality", {})
        w_cfg2 = self.face_cfg.get("worker", {})
        self._min_person_bbox_px: int = int(q_cfg2.get("min_person_bbox_px", 72))
        self._max_face_jobs_per_second: float = float(w_cfg2.get("max_face_jobs_per_second", 4.0))
        self._face_token_bucket: float = self._max_face_jobs_per_second  # start full
        self._face_token_last_refill: float = time.perf_counter()

        # Disable flags (set by run() caller for toggle benchmarks)
        self.disable_face: bool = False
        self.disable_evidence: bool = False
        self.disable_threat: bool = False
        self.frame_stride: int = 1  # 1 = process every frame

    @property
    def logged_events(self) -> List[Event]:
        return list(self._logged_events)

    @property
    def dispatched_alerts(self) -> List[Alert]:
        return list(self._dispatched_alerts)

    @property
    def output_video_path(self) -> Optional[Path]:
        return self._output_video_path

    @property
    def events_file_path(self) -> Optional[Path]:
        return self._events_file_path

    def _setup_outputs(self, width: int, height: int, fps: float) -> None:
        """Initialize VideoWriter and JSONL event log file."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events_dir.mkdir(parents=True, exist_ok=True)

        clean_model = Path(self.model_name).stem
        timestamp_tag = int(time.time())

        if self.save_video:
            filename = f"{self.camera_id}_{clean_model}_phase4.mp4"
            self._output_video_path = self.output_dir / filename
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                str(self._output_video_path),
                fourcc,
                fps if fps > 0 else 25.0,
                (width, height),
            )
            if not self._writer.isOpened():
                self._output_video_path = self._output_video_path.with_suffix(".avi")
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                self._writer = cv2.VideoWriter(
                    str(self._output_video_path),
                    fourcc,
                    fps if fps > 0 else 25.0,
                    (width, height),
                )

        self._events_file_path = self.events_dir / f"{self.camera_id}_{timestamp_tag}.jsonl"
        logger.info("Phase 4 events will be logged to: %s", self._events_file_path.resolve())

    def run(
        self,
        max_frames: Optional[int] = None,
        callback: Optional[Callable[[FrameResult], None]] = None,
        disable_face: bool = False,
        disable_evidence: bool = False,
        disable_threat: bool = False,
        frame_stride: int = 1,
    ) -> Dict[str, Any]:
        """Execute Phase 4 pipeline end-to-end.

        Args:
            max_frames: Stop after this many frames (None = run to end of source).
            callback: Optional per-frame callback receiving FrameResult.
            disable_face: Skip all face recognition jobs (benchmark toggle).
            disable_evidence: Skip evidence collection and frame buffering (benchmark toggle).
            disable_threat: Skip threat engine ingestion (benchmark toggle).
            frame_stride: Run YOLO+ByteTrack every Nth frame; interpolate tracks for skipped frames.
                          1 = process every frame (default). Values 2-3 improve FPS on long clips.
        """
        self.disable_face = disable_face
        self.disable_evidence = disable_evidence
        self.disable_threat = disable_threat
        self.frame_stride = max(1, int(frame_stride))
        self.metrics.start_pipeline()
        self.video_source.start()
        self.face_worker.start()

        outputs_initialized = False
        processed_count = 0
        src_fps = self.video_source.fps or 25.0

        zones = self.zone_manager.get_zones(self.camera_id)
        lines = self.zone_manager.get_lines(self.camera_id)
        # Last known detections/tracks — used on frame-strided frames
        detections: List[Any] = []
        tracks: List[Track] = []

        try:
            while True:
                # 1. Capture Frame
                t0 = time.perf_counter()
                captured: Optional[CapturedFrame] = self.video_source.read(timeout=2.0)
                t1 = time.perf_counter()
                read_ms = (t1 - t0) * 1000.0

                if captured is None:
                    break
                if max_frames and processed_count >= max_frames:
                    break

                h, w = captured.frame.shape[:2]
                if not outputs_initialized:
                    self._setup_outputs(w, h, src_fps)
                    outputs_initialized = True

                video_time = captured.frame_id / src_fps

                # 2. Push frame to Evidence Rolling RAM Buffer (JPEG compressed)
                t_buf_0 = time.perf_counter()
                if not self.disable_evidence:
                    self.evidence_collector.add_frame(
                        camera_id=self.camera_id,
                        frame=captured.frame,
                        timestamp=video_time,
                        frame_id=captured.frame_id,
                    )
                t_buf_1 = time.perf_counter()
                buffer_ms = (t_buf_1 - t_buf_0) * 1000.0

                # 3. Detection & Tracking (honour frame_stride — skip YOLO on strided frames)
                t2 = time.perf_counter()
                if processed_count % self.frame_stride == 0:
                    detections, tracks = self.tracker.update(
                        frame=captured.frame,
                        camera_id=captured.camera_id,
                        frame_id=captured.frame_id,
                        timestamp=video_time,
                    )
                    if torch.cuda.is_available() and "cuda" in str(self.device):
                        torch.cuda.synchronize()
                # On strided frames keep last detections/tracks — ByteTrack handles own interpolation
                t3 = time.perf_counter()
                track_total_ms = (t3 - t2) * 1000.0
                detect_ms = track_total_ms * 0.8
                track_ms = track_total_ms * 0.2

                # 4. Asynchronous Biometric Face Recognition
                t_face_0 = time.perf_counter()
                active_person_tids: List[int] = []
                tracks_by_id: Dict[int, Track] = {}

                if not self.disable_face:
                    # Refill token bucket (tokens replenish at max_face_jobs_per_second)
                    _now = time.perf_counter()
                    _elapsed = _now - self._face_token_last_refill
                    self._face_token_bucket = min(
                        self._max_face_jobs_per_second,
                        self._face_token_bucket + _elapsed * self._max_face_jobs_per_second,
                    )
                    self._face_token_last_refill = _now

                    for trk in tracks:
                        tracks_by_id[trk.track_id] = trk
                        if trk.class_name.lower() == "person":
                            active_person_tids.append(trk.track_id)
                            if self._face_token_bucket >= 1.0 and self.identity_manager.should_process_face(trk.track_id, video_time):
                                x1, y1, x2, y2 = [int(v) for v in trk.bbox]
                                cx1, cy1 = max(0, x1), max(0, y1)
                                cx2, cy2 = min(w, x2), min(h, y2)
                                # Fix 1: skip tiny bboxes — person narrower than min_person_bbox_px
                                # can never contain a usable min_face_size px face crop.
                                if (cx2 - cx1) >= self._min_person_bbox_px and (cy2 - cy1) >= self._min_person_bbox_px:
                                    person_crop = captured.frame[cy1:cy2, cx1:cx2]
                                    self.face_worker.submit_job(
                                        frame_id=captured.frame_id,
                                        timestamp=video_time,
                                        track_id=trk.track_id,
                                        person_crop=person_crop,
                                        track_bbox=trk.bbox,
                                    )
                                    self._face_token_bucket -= 1.0  # consume one token
                else:
                    for trk in tracks:
                        tracks_by_id[trk.track_id] = trk
                        if trk.class_name.lower() == "person":
                            active_person_tids.append(trk.track_id)

                self.identity_manager.evaluate_timeouts(active_person_tids, video_time)
                for trk in tracks:
                    if trk.class_name.lower() == "person":
                        self.identity_manager.attach_to_track(trk, video_time)

                t_face_1 = time.perf_counter()
                face_dispatch_ms = (t_face_1 - t_face_0) * 1000.0

                # 5. Rule Engine Evaluation
                frame_res = FrameResult(
                    frame=captured.frame,
                    frame_id=captured.frame_id,
                    timestamp=video_time,
                    detections=detections,
                    tracks=tracks,
                    timings={},
                )
                new_events = self.rule_engine.evaluate(frame_res, self.camera_id)
                rules_ms = frame_res.timings.get("rules_ms", 0.0)

                # 6. Threat Engine Processing & Correlation
                t_threat_0 = time.perf_counter()
                if not self.disable_threat:
                    for evt in new_events:
                        # Enrich event with person/identity context
                        if evt.track_id is not None and evt.track_id in tracks_by_id:
                            trk = tracks_by_id[evt.track_id]
                            if trk.class_name.lower() == "person":
                                evt.person_id = trk.person_id
                                evt.details["identity_status"] = trk.identity_status.value
                                evt.details["person_name"] = trk.person_name

                                zname = evt.location.get("zone_name")
                                if zname:
                                    is_auth = self.identity_manager.is_authorised_for_zone(trk.track_id, zname)
                                    evt.details["authorised"] = is_auth
                                else:
                                    evt.details["authorised"] = False

                        self._logged_events.append(evt)
                        if self._events_file_path:
                            with open(self._events_file_path, "a", encoding="utf-8") as f:
                                f.write(json.dumps(evt.to_dict(), default=str) + "\n")

                        # Ingest into Threat Engine
                        incident, alert = self.incident_manager.process_event(evt)
                        if alert:
                            self._dispatched_alerts.append(alert)

                        # Evidence Collection for Incident
                        if incident is not None and not self.disable_evidence:
                            # Find person bbox or object bbox
                            p_bbox = None
                            o_bbox = None
                            best_face = None

                            if evt.track_id is not None and evt.track_id in tracks_by_id:
                                p_bbox = tracks_by_id[evt.track_id].bbox

                            # If bag event, find bag bbox
                            if evt.event_type.value == "ABANDONED_OBJECT":
                                for d in detections:
                                    if d.class_name.lower() in ("backpack", "handbag", "suitcase"):
                                        o_bbox = d.bbox
                                        break

                            # Collect evidence snapshots once on creation, and re-clip only on alert escalation
                            should_collect = (alert is not None) or (len(incident.evidence) == 0)
                            if should_collect:
                                self.evidence_collector.collect_for_incident(
                                    incident=incident,
                                    current_frame=captured.frame,
                                    person_bbox=p_bbox,
                                    object_bbox=o_bbox,
                                    best_face_crop=best_face,
                                    generate_clip=(alert is not None),
                                )

                t_threat_1 = time.perf_counter()
                threat_ms = (t_threat_1 - t_threat_0) * 1000.0

                # 7. Visual Annotation
                t4 = time.perf_counter()
                current_fps = self.metrics.rolling_fps or src_fps
                total_latency_ms = track_total_ms + rules_ms + face_dispatch_ms + threat_ms

                static_cands = self.rule_engine.abandoned_rule.get_active_candidates()
                face_stats = self.face_worker.get_stats()
                active_incidents = list(self.incident_manager.open_incidents.values())

                annotated = self.annotator.annotate_phase4(
                    frame=captured.frame,
                    tracks=tracks,
                    zones=zones,
                    lines=lines,
                    fps=current_fps,
                    latency_ms=total_latency_ms,
                    rules_ms=rules_ms,
                    threat_ms=threat_ms,
                    frame_id=captured.frame_id,
                    current_time=video_time,
                    active_incidents=active_incidents,
                    dispatched_alerts_count=len(self._dispatched_alerts),
                    face_stats=face_stats,
                    static_candidates=static_cands,
                )
                t5 = time.perf_counter()
                draw_ms = (t5 - t4) * 1000.0

                # 8. Write Video
                write_ms = 0.0
                if self._writer and self._writer.isOpened():
                    t6 = time.perf_counter()
                    self._writer.write(annotated)
                    t7 = time.perf_counter()
                    write_ms = (t7 - t6) * 1000.0

                # Record Profiler Metrics
                timings = {
                    "read_ms": read_ms,
                    "buffer_ms": buffer_ms,
                    "detect_ms": detect_ms,
                    "track_ms": track_ms,
                    "face_dispatch_ms": face_dispatch_ms,
                    "rules_ms": rules_ms,
                    "threat_ms": threat_ms,
                    "draw_ms": draw_ms,
                    "write_ms": write_ms,
                }
                self.metrics.record_frame(
                    timings_ms=timings,
                    people_count=sum(1 for t in tracks if t.class_name.lower() == "person"),
                    bag_count=sum(1 for t in tracks if t.class_name.lower() in ("backpack", "handbag", "suitcase")),
                )
                processed_count += 1

                if self.show_window:
                    cv2.imshow(f"Phase 4 Integrated Stream - {self.camera_id}", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if callback:
                    frame_res.frame = annotated
                    frame_res.timings = timings
                    callback(frame_res)

        finally:
            self.metrics.stop_pipeline()
            self.video_source.stop()
            self.face_worker.stop()
            if self._writer:
                self._writer.release()
                self._writer = None
            if self.show_window:
                cv2.destroyAllWindows()

        # Ensure all incidents have their final evidence clip generated
        all_incidents = list(self.incident_manager.open_incidents.values()) + list(self.incident_manager.closed_incidents.values())
        for inc in all_incidents:
            if inc.threat_level in ("Medium", "High", "Critical"):
                self.evidence_collector.collect_for_incident(incident=inc, generate_clip=True)

        summary = self.metrics.generate_summary()
        summary["face_worker"] = self.face_worker.get_stats()
        summary["incidents"] = {
            "open_count": len(self.incident_manager.open_incidents),
            "closed_count": len(self.incident_manager.closed_incidents),
            "dispatched_alerts": len(self._dispatched_alerts),
        }
        return summary
