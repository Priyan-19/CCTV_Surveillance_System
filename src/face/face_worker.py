"""Asynchronous Face Recognition Worker thread.

Decouples InsightFace / ArcFace neural network inference from the main 24+ FPS
capture/tracking loop using a bounded producer-consumer queue with drop metrics.
Ensures pipeline latency never spikes above real-time limits.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from src.common.types import IdentityStatus
from src.face.identity import IdentityManager
from src.face.matcher import FaceMatcher, MatchResult
from src.face.quality import FaceQualityFilter

logger = logging.getLogger("face_worker")


@dataclass
class FaceJob:
    """A unit of work submitted from the video pipeline to the FaceWorker."""

    frame_id: int
    timestamp: float
    track_id: int
    person_crop: np.ndarray
    track_bbox: Tuple[float, float, float, float]


class FaceWorker:
    """Background worker thread processing face crops asynchronously."""

    def __init__(
        self,
        matcher: FaceMatcher,
        identity_manager: IdentityManager,
        quality_filter: Optional[FaceQualityFilter] = None,
        model_name: str = "buffalo_s",
        det_size: Tuple[int, int] = (320, 320),
        queue_size: int = 16,
        drop_policy: str = "drop_oldest",
        providers: Optional[List[str]] = None,
        ctx_id: int = 0,
    ) -> None:
        self.matcher = matcher
        self.identity_manager = identity_manager
        self.quality_filter = quality_filter or FaceQualityFilter()
        self.model_name = model_name
        self.det_size = det_size
        self.queue_size = queue_size
        self.drop_policy = drop_policy
        self.ctx_id = ctx_id
        self.providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]

        # Producer-consumer queue
        self._queue: queue.Queue[Optional[FaceJob]] = queue.Queue(maxsize=queue_size)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Telemetry / performance metrics
        self.submitted_count: int = 0
        self.processed_count: int = 0
        self.dropped_count: int = 0
        self.faces_detected_count: int = 0
        self.faces_matched_count: int = 0
        self.total_inference_ms: float = 0.0

        # Lazy initialized InsightFace app
        self._app: Any = None
        self._init_lock = threading.Lock()

    def _ensure_app_initialized(self) -> None:
        """Initialize InsightFace FaceAnalysis instance."""
        if self._app is not None:
            return

        with self._init_lock:
            if self._app is not None:
                return

            import os
            try:
                import torch
                torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
                if os.path.exists(torch_lib):
                    if hasattr(os, "add_dll_directory"):
                        os.add_dll_directory(torch_lib)
                    os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
            except Exception:
                pass

            from insightface.app import FaceAnalysis

            logger.info(
                "Initializing FaceAnalysis(name='%s', ctx_id=%d) with providers=%s...",
                self.model_name,
                self.ctx_id,
                self.providers,
            )
            app = FaceAnalysis(name=self.model_name, providers=self.providers)
            app.prepare(ctx_id=self.ctx_id, det_size=self.det_size)
            self._app = app
            logger.info("FaceAnalysis initialization complete.")

    def start(self) -> None:
        """Start the background worker thread."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker_loop, name="FaceWorkerThread", daemon=True)
        self._thread.start()
        logger.info("FaceWorker thread started (queue size=%d).", self.queue_size)

    def stop(self, timeout: float = 2.0) -> None:
        """Signal the worker to stop and wait for completion."""
        self._stop_event.set()
        try:
            # Wake worker if blocked on empty queue
            self._queue.put_nowait(None)
        except queue.Full:
            pass

        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        logger.info("FaceWorker thread stopped.")

    def submit_job(
        self,
        frame_id: int,
        timestamp: float,
        track_id: int,
        person_crop: np.ndarray,
        track_bbox: Tuple[float, float, float, float],
    ) -> bool:
        """Submit a person crop for face analysis without blocking.

        Returns:
            True if accepted into queue, False if dropped due to queue congestion.
        """
        self.submitted_count += 1
        job = FaceJob(
            frame_id=frame_id,
            timestamp=timestamp,
            track_id=track_id,
            person_crop=person_crop.copy(),
            track_bbox=track_bbox,
        )

        try:
            self._queue.put_nowait(job)
            return True
        except queue.Full:
            self.dropped_count += 1
            if self.drop_policy == "drop_oldest":
                try:
                    _ = self._queue.get_nowait()
                    self._queue.task_done()
                except (queue.Empty, ValueError):
                    pass
                try:
                    self._queue.put_nowait(job)
                except queue.Full:
                    pass
            return False

    def _worker_loop(self) -> None:
        """Main loop executed inside background thread."""
        self._ensure_app_initialized()

        while not self._stop_event.is_set():
            try:
                job = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if job is None:
                # Sentinel wake-up signal
                self._queue.task_done()
                break

            t0 = time.perf_counter()
            self._process_single_job(job)
            dt_ms = (time.perf_counter() - t0) * 1000.0

            self.total_inference_ms += dt_ms
            self.processed_count += 1
            self._queue.task_done()

    def _process_single_job(self, job: FaceJob) -> None:
        """Execute face detection, quality gating, embedding and state update."""
        crop = job.person_crop
        if crop is None or crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
            self.identity_manager.record_attempt_without_usable_face(
                job.track_id, job.timestamp, reason="invalid_person_crop"
            )
            return

        # 1. Run InsightFace detection and feature extraction
        try:
            faces = self._app.get(crop)
        except Exception as exc:
            logger.warning("InsightFace inference failed on frame %d: %s", job.frame_id, exc)
            return

        if not faces:
            self.identity_manager.record_attempt_without_usable_face(
                job.track_id, job.timestamp, reason="no_face_detected"
            )
            return

        self.faces_detected_count += 1

        # 2. Select the most prominent face (largest bounding box area)
        faces_sorted = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True,
        )
        best_face = faces_sorted[0]
        fx1, fy1, fx2, fy2 = [max(0, int(v)) for v in best_face.bbox]
        det_score = float(getattr(best_face, "det_score", 1.0))

        # Crop face region for quality check and evidence persistence
        ch, cw = crop.shape[:2]
        fx1 = min(fx1, cw - 1)
        fy1 = min(fy1, ch - 1)
        fx2 = min(max(fx1 + 1, fx2), cw)
        fy2 = min(max(fy1 + 1, fy2), ch)
        face_crop = crop[fy1:fy2, fx1:fx2]

        # 3. Quality evaluation
        quality_res = self.quality_filter.evaluate(
            face_img=face_crop,
            det_score=det_score,
            face_bbox=(float(fx1), float(fy1), float(fx2), float(fy2)),
        )

        if not quality_res.is_valid:
            self.identity_manager.record_attempt_without_usable_face(
                job.track_id, job.timestamp, reason=quality_res.reason
            )
            return

        # 4. Extract ArcFace embedding
        raw_emb = getattr(best_face, "normed_embedding", None)
        if raw_emb is None:
            raw_emb = getattr(best_face, "embedding", None)
        if raw_emb is None:
            self.identity_manager.record_attempt_without_usable_face(
                job.track_id, job.timestamp, reason="no_embedding_found"
            )
            return

        # 5. Cosine similarity matching against enrolled gallery
        match_res = self.matcher.match(raw_emb)
        if match_res.status == IdentityStatus.KNOWN:
            self.faces_matched_count += 1

        # 6. Update track identity state machine
        self.identity_manager.update_observation(
            track_id=job.track_id,
            timestamp=job.timestamp,
            match_result=match_res,
            face_crop=face_crop,
            quality_passed=True,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Return diagnostic metrics for reporting."""
        mean_ms = (
            (self.total_inference_ms / self.processed_count) if self.processed_count > 0 else 0.0
        )
        drop_rate = (
            (self.dropped_count / self.submitted_count * 100.0) if self.submitted_count > 0 else 0.0
        )
        return {
            "submitted_count": self.submitted_count,
            "processed_count": self.processed_count,
            "dropped_count": self.dropped_count,
            "drop_rate_pct": round(drop_rate, 2),
            "faces_detected": self.faces_detected_count,
            "faces_matched": self.faces_matched_count,
            "mean_inference_ms": round(mean_ms, 2),
        }
