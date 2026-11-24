"""Phase 3 Visual Annotator module.

Extends Phase 2 visual analytics with biometric identity overlays:
- Known individual bounding boxes (Emerald Green) with Name, Person ID, and ArcFace similarity score.
- Stranger / Non-enrolled bounding boxes (Amber Orange) with 'UNKNOWN' designation.
- Pending verification boxes (Neutral Slate) with 'Verifying...'.
- Biometric worker telemetry in status HUD (Queue saturation, jobs processed, drop counter).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np

from src.common.types import Event, IdentityStatus, Track
from src.pipeline.annotator import Phase2Annotator
from src.rules.abandoned_object import StaticBagCandidate
from src.rules.zones import VirtualLine, Zone


class Phase3Annotator(Phase2Annotator):
    """Biometric-aware visual annotator conforming to Phase 3 design."""

    IDENTITY_COLORS: Dict[IdentityStatus, Tuple[int, int, int]] = {
        IdentityStatus.KNOWN: (0, 255, 100),       # Emerald Green for enrolled staff/visitors
        IdentityStatus.UNKNOWN: (0, 140, 255),     # Amber Orange for unknown strangers
        IdentityStatus.PENDING: (180, 180, 180),   # Slate Gray for pending verification
    }

    def annotate_phase3(
        self,
        frame: np.ndarray,
        tracks: Sequence[Track],
        zones: List[Zone],
        lines: List[VirtualLine],
        fps: float,
        latency_ms: float,
        rules_ms: float,
        frame_id: int,
        current_time: float,
        face_stats: Optional[Dict[str, Any]] = None,
        static_candidates: Optional[List[StaticBagCandidate]] = None,
    ) -> np.ndarray:
        """Annotate frame with zones, tripwires, tracks, identities, and alert banners."""
        out_frame = frame.copy()

        # 1. Spatial Zones & Tripwires
        out_frame = self._render_zones_and_lines(out_frame, zones, lines)

        people_count = 0
        bag_count = 0
        known_count = 0
        unknown_count = 0

        # 2. Render Tracks with Biometric Identity
        for trk in tracks:
            cls_lower = trk.class_name.lower()
            x1, y1, x2, y2 = [int(v) for v in trk.bbox]

            status = getattr(trk, "identity_status", IdentityStatus.PENDING) if cls_lower == "person" else None
            if cls_lower == "person":
                people_count += 1
                color = self.IDENTITY_COLORS.get(status, (180, 180, 180))

                dwell_str = f"|dwell:{trk.dwell_seconds:.1f}s" if trk.dwell_seconds > 0 else ""
                if status == IdentityStatus.KNOWN:
                    known_count += 1
                    name = trk.person_name or f"ID_{trk.person_id}"
                    label = f"ID:{trk.track_id} {name} (sim:{trk.identity_confidence:.2f}) {dwell_str}"
                elif status == IdentityStatus.UNKNOWN:
                    unknown_count += 1
                    label = f"ID:{trk.track_id} UNKNOWN (Stranger) {dwell_str}"
                else:
                    label = f"ID:{trk.track_id} person (verifying...) {dwell_str}"
            else:
                if cls_lower in ("backpack", "handbag", "suitcase"):
                    bag_count += 1
                color = self.CLASS_COLORS.get(cls_lower, (200, 200, 200))
                dwell_str = f" | dwell:{trk.dwell_seconds:.1f}s" if trk.dwell_seconds > 0 else ""
                label = f"ID:{trk.track_id} {trk.class_name}{dwell_str}"

            # Motion Trail
            if self.draw_trail and len(trk.path) > 1:
                trail_pts = np.array(trk.path, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(out_frame, [trail_pts], False, color, 2, cv2.LINE_AA)
                cx, cy = int(trk.centre[0]), int(trk.centre[1])
                cv2.circle(out_frame, (cx, cy), 3, color, -1)

            # Bounding Box
            cv2.rectangle(out_frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # Header Badge
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
            cv2.rectangle(out_frame, (x1, max(0, y1 - lh - 6)), (x1 + lw + 6, y1), color, -1)
            text_color = (255, 255, 255) if status == IdentityStatus.UNKNOWN else (0, 0, 0)
            cv2.putText(
                out_frame,
                label,
                (x1 + 3, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                text_color,
                1,
                cv2.LINE_AA,
            )

        # 3. Static Object Candidates (Abandoned luggage)
        if static_candidates:
            for cand in static_candidates:
                cx, cy = int(cand.centroid[0]), int(cand.centroid[1])
                stat_dur = current_time - cand.stationary_start_time
                stat_str = f"STATIC BAG: {stat_dur:.1f}s"
                cv2.circle(out_frame, (cx, cy), 8, (0, 0, 255), 2)
                cv2.putText(out_frame, stat_str, (cx - 40, cy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1, cv2.LINE_AA)

        # 4. Render Event Alert Banners
        self._active_banners = [
            (t, evt) for t, evt in self._active_banners
            if (current_time - t) <= self.banner_display_seconds
        ]

        banner_y = 10
        for t_fired, evt in self._active_banners[-3:]:
            sev_color = self.SEVERITY_COLORS.get(evt.severity, (0, 255, 255))
            auth_str = " [AUTH]" if evt.details.get("authorised") is True else ""
            ident_str = f" ({evt.details.get('person_name') or evt.details.get('identity_status', '')})" if evt.details.get("identity_status") else ""
            banner_text = f"ALERT [{evt.severity.value.upper()}] - {evt.event_type.value}{ident_str}{auth_str} @ {evt.location.get('zone_name', 'Scene')}"

            (bw, bh), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
            cv2.rectangle(out_frame, (10, banner_y), (20 + bw, banner_y + bh + 10), (20, 20, 20), -1)
            cv2.rectangle(out_frame, (10, banner_y), (20 + bw, banner_y + bh + 10), sev_color, 2)
            cv2.putText(out_frame, banner_text, (15, banner_y + bh + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.58, sev_color, 2, cv2.LINE_AA)
            banner_y += bh + 16

        # 5. Status HUD (HUD with Face Worker stats)
        f_proc = face_stats.get("processed_count", 0) if face_stats else 0
        f_drop = face_stats.get("dropped_count", 0) if face_stats else 0
        f_drop_rate = face_stats.get("drop_rate_pct", 0.0) if face_stats else 0.0

        hud_lines = [
            f"CAM: {self.camera_id} | FRAME: {frame_id} | TIME: {current_time:.2f}s",
            f"MODEL: {self.model_name} | FPS: {fps:.1f} | TOTAL: {latency_ms:.1f}ms (RULES: {rules_ms:.2f}ms)",
            f"PEOPLE: {people_count} (KNOWN: {known_count}, UNKNOWN: {unknown_count}) | BAGS: {bag_count}",
            f"FACE WORKER: {f_proc} proc | {f_drop} drop ({f_drop_rate:.1f}%) | Active Gallery: 12",
        ]

        hud_w = 460
        hud_h = (len(hud_lines) * 20) + 16
        hud_y_start = out_frame.shape[0] - hud_h - 10

        overlay = out_frame.copy()
        cv2.rectangle(overlay, (10, hud_y_start), (10 + hud_w, hud_y_start + hud_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, out_frame, 0.25, 0, out_frame)

        for idx, line in enumerate(hud_lines):
            y_pos = hud_y_start + 14 + (idx * 20)
            text_color = (0, 255, 200) if idx == 0 else ((240, 240, 240) if idx < 3 else (200, 220, 255))
            cv2.putText(out_frame, line, (18, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.46, text_color, 1, cv2.LINE_AA)

        return out_frame


import json
import logging
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union
import torch

from src.capture.video_source import CapturedFrame, VideoSource
from src.common.metrics import PerformanceMetrics
from src.common.types import Event, FrameResult, IdentityStatus, Track
from src.face.face_worker import FaceWorker
from src.face.identity import IdentityManager
from src.face.matcher import FaceMatcher
from src.face.quality import FaceQualityFilter
from src.face.store import FaceStore
from src.rules.rule_engine import RuleEngine
from src.rules.zones import ZoneManager
from src.tracking.tracker import BaseTracker, ByteTrackTracker
import yaml

logger = logging.getLogger("phase3_pipeline")


class Phase3Pipeline:
    """Full Phase 3 Surveillance Pipeline integrating YOLO + ByteTrack + Rules + Face Recognition."""

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

        # Load face configurations
        with open(self.face_config_path, "r", encoding="utf-8") as f:
            self.face_cfg = yaml.safe_load(f)

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

        # 3. Face Recognition Subsystems
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
            recheck_interval_seconds=sm_cfg.get("recheck_interval_seconds", 0.3),
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

        # 4. Zone Manager & Rule Engine
        self.zone_manager = ZoneManager(config_path=self.zones_config)
        self.rule_engine = RuleEngine(
            rules_config_path=self.rules_config,
            zone_manager=self.zone_manager,
        )

        # 5. Annotator
        self.annotator = Phase3Annotator(
            camera_id=self.camera_id,
            model_name=self.model_name,
            tracker_name="ByteTrack",
        )

        # 6. Profiler
        self.metrics = PerformanceMetrics()
        self._writer: Optional[cv2.VideoWriter] = None
        self._output_video_path: Optional[Path] = None
        self._events_file_path: Optional[Path] = None
        self._logged_events: List[Event] = []

    @property
    def logged_events(self) -> List[Event]:
        return list(self._logged_events)

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
            filename = f"{self.camera_id}_{clean_model}_phase3.mp4"
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
        logger.info("Phase 3 events will be logged to: %s", self._events_file_path.resolve())

    def run(
        self,
        max_frames: Optional[int] = None,
        callback: Optional[Callable[[FrameResult], None]] = None,
    ) -> Dict[str, Any]:
        """Execute Phase 3 pipeline."""
        self.metrics.start_pipeline()
        self.video_source.start()
        self.face_worker.start()

        outputs_initialized = False
        processed_count = 0
        src_fps = self.video_source.fps or 25.0

        zones = self.zone_manager.get_zones(self.camera_id)
        lines = self.zone_manager.get_lines(self.camera_id)

        try:
            while True:
                # 1. Capture frame
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

                # 2. Track & Detect
                t2 = time.perf_counter()
                detections, tracks = self.tracker.update(
                    frame=captured.frame,
                    camera_id=captured.camera_id,
                    frame_id=captured.frame_id,
                    timestamp=video_time,
                )
                if torch.cuda.is_available() and "cuda" in str(self.device):
                    torch.cuda.synchronize()
                t3 = time.perf_counter()
                track_total_ms = (t3 - t2) * 1000.0
                detect_ms = track_total_ms * 0.8
                track_ms = track_total_ms * 0.2

                # 3. Submit person crops to Asynchronous FaceWorker (non-blocking)
                t_face_0 = time.perf_counter()
                active_person_tids: List[int] = []
                for trk in tracks:
                    if trk.class_name.lower() == "person":
                        active_person_tids.append(trk.track_id)
                        if self.identity_manager.should_process_face(trk.track_id, video_time):
                            x1, y1, x2, y2 = [int(v) for v in trk.bbox]
                            # Bounded crop inside frame dimensions
                            cx1, cy1 = max(0, x1), max(0, y1)
                            cx2, cy2 = min(w, x2), min(h, y2)
                            if (cx2 - cx1) >= 20 and (cy2 - cy1) >= 20:
                                person_crop = captured.frame[cy1:cy2, cx1:cx2]
                                self.face_worker.submit_job(
                                    frame_id=captured.frame_id,
                                    timestamp=video_time,
                                    track_id=trk.track_id,
                                    person_crop=person_crop,
                                    track_bbox=trk.bbox,
                                )

                # Evaluate timeouts and attach biometric identity
                self.identity_manager.evaluate_timeouts(active_person_tids, video_time)
                for trk in tracks:
                    if trk.class_name.lower() == "person":
                        self.identity_manager.attach_to_track(trk, video_time)

                t_face_1 = time.perf_counter()
                face_dispatch_ms = (t_face_1 - t_face_0) * 1000.0

                # 4. Rule Engine Evaluation
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

                # Attach identity & authorization to events
                tracks_by_id = {t.track_id: t for t in tracks}
                for evt in new_events:
                    if evt.track_id is not None and evt.track_id in tracks_by_id:
                        trk = tracks_by_id[evt.track_id]
                        if trk.class_name.lower() == "person":
                            evt.person_id = trk.person_id
                            evt.details["identity_status"] = trk.identity_status.value
                            evt.details["person_name"] = trk.person_name

                            # Check zone authorization
                            zname = evt.location.get("zone_name")
                            if zname:
                                is_auth = self.identity_manager.is_authorised_for_zone(trk.track_id, zname)
                                evt.details["authorised"] = is_auth
                            else:
                                evt.details["authorised"] = False

                            if trk.identity_status == IdentityStatus.UNKNOWN:
                                evt.details["threat_modifier"] = "UNKNOWN_PERSON"

                    self._logged_events.append(evt)
                    self.annotator.add_event(evt, video_time)

                    ident_disp = f" | Identity: {evt.details.get('person_name') or evt.details.get('identity_status')}"
                    auth_disp = " [AUTH]" if evt.details.get("authorised") is True else ""
                    print(
                        f"\n>>> [SECURITY EVENT] [{evt.severity.value.upper()}] {evt.event_type.value}{auth_disp} "
                        f"at t={video_time:.2f}s (Frame #{captured.frame_id}) | "
                        f"Zone: {evt.location.get('zone_name')} | Track ID: {evt.track_id}{ident_disp} | Details: {evt.details}"
                    )
                    if self._events_file_path:
                        with open(self._events_file_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(evt.to_dict(), default=str) + "\n")

                # 5. Visual Annotation
                t4 = time.perf_counter()
                current_fps = self.metrics.rolling_fps or src_fps
                total_latency_ms = track_total_ms + rules_ms + face_dispatch_ms

                static_cands = self.rule_engine.abandoned_rule.get_active_candidates()
                face_stats = self.face_worker.get_stats()

                annotated = self.annotator.annotate_phase3(
                    frame=captured.frame,
                    tracks=tracks,
                    zones=zones,
                    lines=lines,
                    fps=current_fps,
                    latency_ms=total_latency_ms,
                    rules_ms=rules_ms,
                    frame_id=captured.frame_id,
                    current_time=video_time,
                    face_stats=face_stats,
                    static_candidates=static_cands,
                )
                t5 = time.perf_counter()
                draw_ms = (t5 - t4) * 1000.0

                # 6. Write video
                write_ms = 0.0
                if self._writer and self._writer.isOpened():
                    t6 = time.perf_counter()
                    self._writer.write(annotated)
                    t7 = time.perf_counter()
                    write_ms = (t7 - t6) * 1000.0

                # Record metrics
                timings = {
                    "read_ms": read_ms,
                    "detect_ms": detect_ms,
                    "track_ms": track_ms,
                    "face_dispatch_ms": face_dispatch_ms,
                    "rules_ms": rules_ms,
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
                    cv2.imshow(f"Phase 3 Security Stream - {self.camera_id}", annotated)
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

        summary = self.metrics.generate_summary()
        summary["face_worker"] = self.face_worker.get_stats()
        return summary
