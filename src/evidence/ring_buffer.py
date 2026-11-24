"""Rolling RAM Frame Buffer for the Evidence Collector.

Maintains a bounded duration (e.g. 30 seconds) of compressed JPEG frames in RAM
per camera stream, enforcing strict memory limits (e.g. max 250 MB) and
evicting oldest frames first (FIFO).

Conforms to docs/04_DATA_FLOW.md, docs/05_THREAT_ENGINE_SPEC.md, and docs/11_SECURITY_PRIVACY.md.
"""

from __future__ import annotations

from collections import deque
import logging
from typing import Any, Deque, Dict, List, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class RollingFrameBuffer:
    """Bounded circular RAM buffer storing compressed video frames."""

    def __init__(
        self,
        buffer_duration_seconds: float = 30.0,
        max_buffer_memory_mb: float = 250.0,
        jpeg_quality: int = 85,
    ) -> None:
        """Initialize the rolling buffer.

        Args:
            buffer_duration_seconds: Maximum time span of frames retained.
            max_buffer_memory_mb: Strict RAM memory cap in megabytes.
            jpeg_quality: JPEG compression quality (1-100) to minimize memory footprint.
        """
        self.duration: float = float(buffer_duration_seconds)
        self.max_bytes: int = int(max_buffer_memory_mb * 1024 * 1024)
        self.jpeg_quality: int = int(jpeg_quality)

        # Deque of tuples: (timestamp: float, frame_id: int, compressed_bytes: bytes)
        self._buffer: Deque[Tuple[float, int, bytes]] = deque()
        self._current_bytes: int = 0

    @property
    def current_memory_mb(self) -> float:
        """Current buffer memory consumption in Megabytes."""
        return self._current_bytes / (1024.0 * 1024.0)

    @property
    def frame_count(self) -> int:
        """Number of frames currently in the buffer."""
        return len(self._buffer)

    def add_frame(self, frame: np.ndarray, timestamp: float, frame_id: int) -> None:
        """Compress and enqueue a video frame, enforcing duration and memory limits.

        Args:
            frame: Raw BGR numpy image frame.
            timestamp: Video-time or capture timestamp in seconds.
            frame_id: Monotonically increasing frame index.
        """
        if frame is None or frame.size == 0:
            return

        # Compress to JPEG in memory
        success, enc = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if not success:
            logger.warning(f"Failed to compress frame {frame_id} for rolling buffer.")
            return

        frame_bytes = enc.tobytes()
        item_size = len(frame_bytes)

        # Evict oldest frames if memory limit would be exceeded
        while self._buffer and (self._current_bytes + item_size > self.max_bytes):
            _, _, old_bytes = self._buffer.popleft()
            self._current_bytes -= len(old_bytes)

        # Evict frames older than buffer_duration_seconds
        while self._buffer and (timestamp - self._buffer[0][0] > self.duration):
            _, _, old_bytes = self._buffer.popleft()
            self._current_bytes -= len(old_bytes)

        # Append new frame
        self._buffer.append((timestamp, frame_id, frame_bytes))
        self._current_bytes += item_size

    def get_frames_in_range(
        self, start_time: float, end_time: float
    ) -> List[Tuple[float, int, np.ndarray]]:
        """Retrieve and decode all frames within the given time range [start_time, end_time].

        Args:
            start_time: Start timestamp in seconds (inclusive).
            end_time: End timestamp in seconds (inclusive).

        Returns:
            List of (timestamp, frame_id, decoded_frame_bgr).
        """
        results: List[Tuple[float, int, np.ndarray]] = []
        for ts, fid, fbytes in self._buffer:
            if start_time <= ts <= end_time:
                nparr = np.frombuffer(fbytes, np.uint8)
                decoded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if decoded is not None:
                    results.append((ts, fid, decoded))
        return results

    def get_latest_frame(self) -> Optional[Tuple[float, int, np.ndarray]]:
        """Retrieve and decode the most recent frame in the buffer."""
        if not self._buffer:
            return None
        ts, fid, fbytes = self._buffer[-1]
        nparr = np.frombuffer(fbytes, np.uint8)
        decoded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return (ts, fid, decoded) if decoded is not None else None

    def clear(self) -> None:
        """Empty the buffer and release RAM."""
        self._buffer.clear()
        self._current_bytes = 0
