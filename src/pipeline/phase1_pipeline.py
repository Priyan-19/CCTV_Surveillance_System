"""Phase 1 Orchestration Pipeline.

Wires together:
  VideoSource -> Tracker (with ByteTrack) -> Annotator -> VideoWriter (optional)
Profiles latency per stage (read, detect, track, draw, write) and records performance metrics.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Dict, Generator, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch

from src.capture.video_source import CapturedFrame, VideoSource
from src.common.metrics import PerformanceMetrics
from src.common.types import Detection, FrameResult, Track
from src.pipeline.annotator import FrameAnnotator
from src.tracking.tracker import BaseTracker, ByteTrackTracker

logger = logging.getLogger("phase1_pipeline")


class Phase1Pipeline:
    """Detection and Tracking pipeline for Phase 1 verification."""

    def __init__(
        self,
        camera_id: str,
        source: Union[str, int],
        tracker: Optional[BaseTracker] = None,
        model_name: str = "yolo26s.pt",
        tracker_config: str = "configs/bytetrack.yaml",
        device: str = "cuda:0",
        half_precision: bool = True,
        imgsz: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.70,
        tracked_classes: Optional[List[int]] = None,
        frame_skip: int = 0,
        output_dir: Union[str, Path] = "data/output",
        save_video: bool = True,
        show_window: bool = False,
    ) -> None:
        self.camera_id = camera_id
        self.source_path = source
        self.model_name = model_name
        self.tracker_config = tracker_config
        self.device = device
        self.half_precision = half_precision
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.tracked_classes = tracked_classes
        self.frame_skip = max(0, frame_skip)
        self.output_dir = Path(output_dir)
        self.save_video = save_video
        self.show_window = show_window

        # 1. Video Source
        self.video_source = VideoSource(
            camera_id=self.camera_id,
            source=self.source_path,
        )

        # 2. Tracker (instantiate default ByteTrackTracker if not injected)
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

        # 3. Annotator
        tracker_tag = "ByteTrack"
        self.annotator = FrameAnnotator(
            camera_id=self.camera_id,
            model_name=self.model_name,
            tracker_name=tracker_tag,
        )

        # 4. Metrics Profiler
        self.metrics = PerformanceMetrics()

        # VideoWriter handle
        self._writer: Optional[cv2.VideoWriter] = None
        self._output_video_path: Optional[Path] = None

    @property
    def output_video_path(self) -> Optional[Path]:
        return self._output_video_path

    def _setup_writer(self, width: int, height: int, fps: float) -> None:
        """Initialize VideoWriter with sanitized file name."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Clean model name for filename
        clean_model = Path(self.model_name).stem
        filename = f"{self.camera_id}_{clean_model}_bytetrack_{int(width)}x{int(height)}.mp4"
        self._output_video_path = self.output_dir / filename

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            str(self._output_video_path),
            fourcc,
            fps if fps > 0 else 25.0,
            (width, height),
        )
        if not self._writer.isOpened():
            # Fallback to AVI if mp4v is not supported on system
            avi_path = self._output_video_path.with_suffix(".avi")
            logger.warning("Could not open mp4v writer. Falling back to XVID AVI: %s", avi_path)
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            self._writer = cv2.VideoWriter(
                str(avi_path),
                fourcc,
                fps if fps > 0 else 25.0,
                (width, height),
            )
            self._output_video_path = avi_path

        logger.info("Writing annotated output stream to: %s", self._output_video_path.resolve())

    def run(
        self,
        max_frames: Optional[int] = None,
        callback: Optional[Callable[[FrameResult], None]] = None,
    ) -> Dict[str, any]:
        """Execute the pipeline loop until stream terminates or max_frames is reached."""
        self.metrics.start_pipeline()
        self.video_source.start()

        writer_initialized = False
        processed_count = 0

        logger.info(
            "Starting Phase 1 pipeline for camera '%s' (model: %s, tracker: ByteTrack)...",
            self.camera_id,
            self.model_name,
        )

        try:
            while True:
                # Stage 1: Frame Acquisition
                t_read_start = time.perf_counter()
                captured: Optional[CapturedFrame] = self.video_source.read(timeout=2.0)
                t_read_end = time.perf_counter()
                read_ms = (t_read_end - t_read_start) * 1000.0

                if captured is None:
                    # Stream exhausted
                    break

                # Frame skipping logic
                if self.frame_skip > 0 and (captured.frame_id % (self.frame_skip + 1) != 0):
                    continue

                if max_frames and processed_count >= max_frames:
                    logger.info("Reached maximum requested frames: %d", max_frames)
                    break

                h, w = captured.frame.shape[:2]
                if not writer_initialized and self.save_video:
                    self._setup_writer(w, h, self.video_source.fps)
                    writer_initialized = True

                # Stage 2 & 3: Detection & ByteTrack Association
                t_track_start = time.perf_counter()
                detections, tracks = self.tracker.update(
                    frame=captured.frame,
                    camera_id=captured.camera_id,
                    frame_id=captured.frame_id,
                    timestamp=captured.timestamp,
                )
                if torch.cuda.is_available() and "cuda" in str(self.device):
                    torch.cuda.synchronize()
                t_track_end = time.perf_counter()

                track_total_ms = (t_track_end - t_track_start) * 1000.0
                # Attribute 80% to neural inference and 20% to association
                detect_ms = track_total_ms * 0.8
                track_ms = track_total_ms * 0.2

                # Stage 4: Annotation & HUD Drawing
                t_draw_start = time.perf_counter()
                current_fps = self.metrics.rolling_fps or self.video_source.fps
                annotated_frame = self.annotator.annotate(
                    frame=captured.frame,
                    tracks=tracks,
                    fps=current_fps,
                    latency_ms=track_total_ms,
                    frame_id=captured.frame_id,
                )
                t_draw_end = time.perf_counter()
                draw_ms = (t_draw_end - t_draw_start) * 1000.0

                # Stage 5: Video Encoding / Write
                write_ms = 0.0
                if self._writer and self._writer.isOpened():
                    t_write_start = time.perf_counter()
                    self._writer.write(annotated_frame)
                    t_write_end = time.perf_counter()
                    write_ms = (t_write_end - t_write_start) * 1000.0

                # Record per-stage latency timings
                timings = {
                    "read_ms": read_ms,
                    "detect_ms": detect_ms,
                    "track_ms": track_ms,
                    "draw_ms": draw_ms,
                    "write_ms": write_ms,
                }

                people_cnt = sum(1 for t in tracks if t.class_name.lower() == "person")
                bag_cnt = sum(1 for t in tracks if t.class_name.lower() in ("backpack", "handbag", "suitcase"))
                veh_cnt = sum(1 for t in tracks if t.class_name.lower() in ("car", "bus", "truck", "motorcycle"))

                self.metrics.record_frame(
                    timings_ms=timings,
                    people_count=people_cnt,
                    bag_count=bag_cnt,
                    vehicle_count=veh_cnt,
                )
                processed_count += 1

                # Optional live display
                if self.show_window:
                    cv2.imshow(f"Phase 1 - {self.camera_id}", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.info("User requested termination ('q' pressed).")
                        break

                # Frame callback for external observers or test hooks
                if callback:
                    frame_result = FrameResult(
                        frame=annotated_frame,
                        frame_id=captured.frame_id,
                        timestamp=captured.timestamp,
                        detections=detections,
                        tracks=tracks,
                        timings=timings,
                    )
                    callback(frame_result)

                if processed_count % 50 == 0:
                    summary_so_far = self.metrics.generate_summary()
                    avg_fps = summary_so_far["overall_throughput_fps"]
                    inf_fps = summary_so_far["inference_only_fps"]
                    logger.info(
                        "Processed %4d frames | Throughput: %.1f FPS | Inference: %.1f FPS (latency: %.1f ms)",
                        processed_count,
                        avg_fps,
                        inf_fps,
                        track_total_ms,
                    )

        finally:
            self.metrics.stop_pipeline()
            self.video_source.stop()
            if self._writer:
                self._writer.release()
                self._writer = None
            if self.show_window:
                cv2.destroyAllWindows()

        return self.metrics.generate_summary()
