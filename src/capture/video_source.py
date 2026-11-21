"""Video capture module providing thread-safe stream acquisition.

Supports:
- Video files (no-drop mode, ensures 100% of frames are processed).
- Webcams (drop-oldest mode, bounds latency to real-time).
- RTSP streams with exponential reconnect backoff (marked UNTESTED in local lab).
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from enum import Enum
from pathlib import Path
from typing import NamedTuple, Optional, Tuple, Union
import cv2
import numpy as np

logger = logging.getLogger("video_source")


class SourceKind(str, Enum):
    FILE = "file"
    WEBCAM = "webcam"
    RTSP = "rtsp"


class CapturedFrame(NamedTuple):
    camera_id: str
    frame_id: int
    timestamp: float
    frame: np.ndarray


def detect_source_kind(source: Union[str, int]) -> Tuple[SourceKind, Union[str, int]]:
    """Determine whether a source is a local webcam index, an RTSP stream, or a file."""
    if isinstance(source, int):
        return SourceKind.WEBCAM, source

    source_str = str(source).strip()
    if source_str.isdigit():
        return SourceKind.WEBCAM, int(source_str)

    lower = source_str.lower()
    if lower.startswith("rtsp://") or lower.startswith("rtsps://"):
        return SourceKind.RTSP, source_str

    if lower.startswith("http://") or lower.startswith("https://"):
        if "mjpg" in lower or "mjpeg" in lower:
            return SourceKind.RTSP, source_str

    return SourceKind.FILE, source_str


class VideoSource:
    """Threaded video stream reader with bounded queuing and clean lifecycle management."""

    def __init__(
        self,
        camera_id: str,
        source: Union[str, int],
        queue_size: int = 60,
        drop_frames: Optional[bool] = None,
        target_fps: Optional[float] = None,
    ) -> None:
        self.camera_id = camera_id
        self.raw_source = source
        self.kind, self.parsed_source = detect_source_kind(source)
        self.queue_size = queue_size
        self.target_fps = target_fps

        # By default: files never drop frames; live streams (webcam/rtsp) drop oldest when full
        if drop_frames is None:
            self.drop_frames = self.kind != SourceKind.FILE
        else:
            self.drop_frames = drop_frames

        self._frame_queue: queue.Queue[Optional[CapturedFrame]] = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_counter: int = 0
        self._fps: float = 25.0
        self._width: int = 0
        self._height: int = 0
        self._total_file_frames: int = 0

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def total_file_frames(self) -> int:
        return self._total_file_frames

    def _open_capture(self) -> bool:
        """Initialize OpenCV VideoCapture handle."""
        if self.kind == SourceKind.WEBCAM:
            self._cap = cv2.VideoCapture(int(self.parsed_source), cv2.CAP_DSHOW)
            if not self._cap.isOpened():
                self._cap = cv2.VideoCapture(int(self.parsed_source))
        elif self.kind == SourceKind.RTSP:
            # Note: RTSP handling path marked UNTESTED in local development
            logger.info("[%s] Connecting to RTSP endpoint (UNTESTED path): %s", self.camera_id, self.parsed_source)
            self._cap = cv2.VideoCapture(str(self.parsed_source), cv2.CAP_FFMPEG)
        else:
            file_path = Path(str(self.parsed_source))
            if not file_path.exists():
                logger.error("[%s] Video file does not exist: %s", self.camera_id, file_path)
                return False
            self._cap = cv2.VideoCapture(str(file_path))

        if not self._cap or not self._cap.isOpened():
            logger.error("[%s] Failed to open capture source: %s", self.camera_id, self.parsed_source)
            return False

        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 360
        self._fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or 25.0
        if self.kind == SourceKind.FILE:
            self._total_file_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(
            "[%s] Opened %s source (%dx%d @ %.2f FPS, total=%d frames)",
            self.camera_id,
            self.kind.value,
            self._width,
            self._height,
            self._fps,
            self._total_file_frames,
        )
        return True

    def _reader_loop(self) -> None:
        """Continuous background thread fetching frames from VideoCapture."""
        reconnect_attempts = 0
        backoff_sec = 1.0

        while not self._stop_event.is_set():
            if self._cap is None or not self._cap.isOpened():
                if self.kind == SourceKind.RTSP:
                    logger.warning(
                        "[%s] [RTSP - UNTESTED] Connection dropped. Reconnecting in %.1fs (attempt #%d)...",
                        self.camera_id,
                        backoff_sec,
                        reconnect_attempts + 1,
                    )
                    time.sleep(backoff_sec)
                    backoff_sec = min(backoff_sec * 2.0, 30.0)
                    reconnect_attempts += 1
                    if not self._open_capture():
                        continue
                    backoff_sec = 1.0
                    reconnect_attempts = 0
                else:
                    break

            ret, frame = self._cap.read()
            now = time.time()

            if not ret or frame is None:
                if self.kind == SourceKind.FILE:
                    logger.info("[%s] End of video file reached (frames: %d)", self.camera_id, self._frame_counter)
                    break
                elif self.kind == SourceKind.RTSP:
                    logger.warning("[%s] [RTSP - UNTESTED] Empty frame received from stream.", self.camera_id)
                    if self._cap:
                        self._cap.release()
                    continue
                else:
                    # Webcam read glitch
                    time.sleep(0.01)
                    continue

            self._frame_counter += 1
            captured = CapturedFrame(
                camera_id=self.camera_id,
                frame_id=self._frame_counter,
                timestamp=now,
                frame=frame,
            )

            if self.drop_frames:
                # Live stream mode: if queue is full, discard the oldest frame
                while not self._stop_event.is_set():
                    try:
                        self._frame_queue.put_nowait(captured)
                        break
                    except queue.Full:
                        try:
                            _ = self._frame_queue.get_nowait()
                        except queue.Empty:
                            pass
            else:
                # File mode: block until reader has consumer capacity (zero dropped frames)
                while not self._stop_event.is_set():
                    try:
                        self._frame_queue.put(captured, timeout=0.1)
                        break
                    except queue.Full:
                        continue

        # Signal completion with sentinel None
        try:
            self._frame_queue.put(None, timeout=1.0)
        except queue.Full:
            pass

        if self._cap:
            self._cap.release()
            self._cap = None

    def start(self) -> None:
        """Start background reader worker."""
        if not self._open_capture():
            raise RuntimeError(f"Could not open video source '{self.raw_source}' for camera '{self.camera_id}'")

        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._reader_loop,
            name=f"ReaderThread-{self.camera_id}",
            daemon=True,
        )
        self._worker_thread.start()

    def read(self, timeout: float = 2.0) -> Optional[CapturedFrame]:
        """Fetch the next captured frame from the bounded queue.

        Returns None when the stream finishes or has been stopped.
        """
        if self._stop_event.is_set() and self._frame_queue.empty():
            return None

        try:
            item = self._frame_queue.get(timeout=timeout)
            if item is None:
                # Sentinel reached
                return None
            return item
        except queue.Empty:
            return None

    def stop(self) -> None:
        """Terminate background thread and release capture resources."""
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
            self._cap = None

        # Drain remaining items
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break

    def __enter__(self) -> VideoSource:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
