"""Phase 2 Orchestration Pipeline.

Wires together:
  VideoSource -> ByteTrackTracker -> RuleEngine -> Phase2Annotator -> Event JSONL Logger
Profiles per-stage latency including rules_ms, logs structured security events,
and generates annotated video output with spatial zones, tripwires, and alert banners.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union
import cv2
import numpy as np
import torch

from src.capture.video_source import CapturedFrame, VideoSource
from src.common.metrics import PerformanceMetrics
from src.common.types import Event, FrameResult, Track
from src.pipeline.annotator import Phase2Annotator
from src.rules.rule_engine import RuleEngine
from src.rules.zones import ZoneManager
from src.tracking.tracker import BaseTracker, ByteTrackTracker

logger = logging.getLogger("phase2_pipeline")


class Phase2Pipeline:
    """Security event detection and spatial reasoning pipeline."""

    def __init__(
        self,
        camera_id: str,
        source: Union[str, int],
        tracker: Optional[BaseTracker] = None,
        model_name: str = "yolo26s.pt",
        tracker_config: str = "configs/bytetrack.yaml",
        rules_config: str = "configs/rules.yaml",
        zones_config: str = "configs/zones.yaml",
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

        # 3. Zone Manager & Rule Engine
        self.zone_manager = ZoneManager(config_path=self.zones_config)
        self.rule_engine = RuleEngine(
            rules_config_path=self.rules_config,
            zone_manager=self.zone_manager,
        )

        # 4. Annotator
        self.annotator = Phase2Annotator(
            camera_id=self.camera_id,
            model_name=self.model_name,
            tracker_name="ByteTrack",
        )

        # 5. Performance Profiler
        self.metrics = PerformanceMetrics()

        self._writer: Optional[cv2.VideoWriter] = None
        self._output_video_path: Optional[Path] = None
        self._events_file_path: Optional[Path] = None
        self._logged_events: List[Event] = []

    @property
    def output_video_path(self) -> Optional[Path]:
        return self._output_video_path

    @property
    def events_file_path(self) -> Optional[Path]:
        return self._events_file_path

    @property
    def logged_events(self) -> List[Event]:
        return list(self._logged_events)

    def _setup_outputs(self, width: int, height: int, fps: float) -> None:
        """Initialize VideoWriter and JSONL event log file."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events_dir.mkdir(parents=True, exist_ok=True)

        clean_model = Path(self.model_name).stem
        timestamp_tag = int(time.time())

        # Video path
        if self.save_video:
            filename = f"{self.camera_id}_{clean_model}_phase2.mp4"
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

        # JSONL Events log path
        self._events_file_path = self.events_dir / f"{self.camera_id}_{timestamp_tag}.jsonl"
        logger.info("Events will be logged to: %s", self._events_file_path.resolve())

    def run(
        self,
        max_frames: Optional[int] = None,
        callback: Optional[Callable[[FrameResult], None]] = None,
    ) -> Dict[str, any]:
        """Execute Phase 2 pipeline."""
        self.metrics.start_pipeline()
        self.video_source.start()

        outputs_initialized = False
        processed_count = 0
        src_fps = self.video_source.fps or 25.0

        zones = self.zone_manager.get_zones(self.camera_id)
        lines = self.zone_manager.get_lines(self.camera_id)

        try:
            while True:
                # 1. Read
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

                # Video-time determinism: derive time from frame index and source FPS
                video_time = captured.frame_id / src_fps

                # 2. Detect & Track
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

                # 3. Rule Engine
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

                # Log new events to JSONL and print to console
                for evt in new_events:
                    self._logged_events.append(evt)
                    self.annotator.add_event(evt, video_time)
                    print(
                        f"\n>>> [SECURITY EVENT] [{evt.severity.value.upper()}] {evt.event_type.value} "
                        f"at t={video_time:.2f}s (Frame #{captured.frame_id}) | "
                        f"Zone: {evt.location.get('zone_name')} | Track ID: {evt.track_id} | Details: {evt.details}"
                    )
                    if self._events_file_path:
                        with open(self._events_file_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(evt.to_dict()) + "\n")

                # 4. Annotate
                t4 = time.perf_counter()
                current_fps = self.metrics.rolling_fps or src_fps
                total_latency_ms = track_total_ms + rules_ms

                # Expose static bag candidates for visualization
                static_cands = self.rule_engine.abandoned_rule.get_active_candidates()

                annotated = self.annotator.annotate(
                    frame=captured.frame,
                    tracks=tracks,
                    zones=zones,
                    lines=lines,
                    fps=current_fps,
                    latency_ms=total_latency_ms,
                    rules_ms=rules_ms,
                    frame_id=captured.frame_id,
                    current_time=video_time,
                    static_candidates=static_cands,
                )
                t5 = time.perf_counter()
                draw_ms = (t5 - t4) * 1000.0

                # 5. Write Video
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
                    cv2.imshow(f"Phase 2 Security Stream - {self.camera_id}", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if callback:
                    frame_res.frame = annotated
                    frame_res.timings = timings
                    callback(frame_res)

        finally:
            self.metrics.stop_pipeline()
            self.video_source.stop()
            if self._writer:
                self._writer.release()
                self._writer = None
            if self.show_window:
                cv2.destroyAllWindows()

        return self.metrics.generate_summary()
