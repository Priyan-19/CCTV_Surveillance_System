"""Browser-compatible video clip writer for Incident Evidence.

Extracts frames from the rolling frame buffer covering pre-roll, event duration,
and post-roll, encoding an H.264/yuv420p MP4 video clip that plays natively in
all modern browsers without transcoding plugins.

Encoding strategy:
  1. Write frames to a temporary MPEG-4 Part-2 file via OpenCV (always available).
  2. Re-encode the temp file to H.264/yuv420p/faststart using ffmpeg subprocess
     running in the same thread.  If ffmpeg is not on PATH, fall back to the raw
     mp4v file and log a WARNING — the file still plays in VLC/desktop players but
     may not play in a browser.

Enforces free disk space checks (docs/11).
Conforms to docs/05 Section 6, docs/07, and docs/10 (T-23, T-24).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Detect ffmpeg availability once at module import time.
_FFMPEG_PATH: Optional[str] = shutil.which("ffmpeg")
if _FFMPEG_PATH:
    logger.info("ffmpeg found at %s — evidence clips will be H.264/yuv420p/faststart.", _FFMPEG_PATH)
else:
    logger.warning(
        "ffmpeg not found on PATH. Evidence clips will be encoded with OpenCV mp4v (MPEG-4 Part 2). "
        "These files play in desktop players but are NOT browser-compatible without transcoding. "
        "Install ffmpeg and ensure it is on PATH to enable H.264 output."
    )


class ClipWriter:
    """Encodes a sequence of video frames into a browser-compatible H.264 MP4 clip."""

    def __init__(
        self,
        fps: float = 25.0,
        fourcc: str = "mp4v",
        min_free_disk_mb: float = 500.0,
    ) -> None:
        """Initialize the clip writer.

        Args:
            fps: Playback frame rate.
            fourcc: Fallback video codec FourCC code (used if ffmpeg is unavailable).
            min_free_disk_mb: Minimum required free disk space in megabytes.
        """
        self.fps: float = float(fps)
        self.fourcc: str = str(fourcc)
        self.min_free_disk_mb: float = float(min_free_disk_mb)

    def write_clip(
        self,
        frames: List[Tuple[float, int, np.ndarray]],
        output_path: Path,
    ) -> Optional[str]:
        """Encode a list of (timestamp, frame_id, bgr_frame) into an H.264 MP4 file.

        If ffmpeg is available the output will be:
            - Codec:      H.264 (libx264)
            - Pixel format: yuv420p  (required for browser playback)
            - Flags:      +faststart (moov atom at the start for progressive download)
            - Preset:     fast

        If ffmpeg is NOT available the output will be MPEG-4 Part-2 (mp4v via OpenCV)
        with a WARNING emitted. The caller may still serve the file; browsers will
        require a transcoding proxy.

        Args:
            frames: Chronologically ordered (timestamp, frame_id, bgr_frame) tuples.
            output_path: Desired final MP4 file path.

        Returns:
            Normalised string path of written clip, or None if skipped/failed.
        """
        if not frames:
            logger.warning("No frames provided to write_clip; skipping clip generation.")
            return None

        # 1. Disk Space Safety Check (docs/11)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            usage = shutil.disk_usage(str(output_path.parent))
            free_mb = usage.free / (1024.0 * 1024.0)
            if free_mb < self.min_free_disk_mb:
                logger.error(
                    "Insufficient free disk space: %.1f MB available, minimum %.1f MB required. "
                    "Aborting clip write.",
                    free_mb,
                    self.min_free_disk_mb,
                )
                return None
        except Exception as exc:
            logger.warning("Could not inspect disk usage: %s", exc)

        # 2. Extract dimensions from first frame
        first_frame = frames[0][2]
        h, w = first_frame.shape[:2]

        if _FFMPEG_PATH:
            return self._write_via_ffmpeg(frames, output_path, h, w)
        else:
            return self._write_via_opencv(frames, output_path, h, w)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _write_via_ffmpeg(
        self,
        frames: List[Tuple[float, int, np.ndarray]],
        output_path: Path,
        h: int,
        w: int,
    ) -> Optional[str]:
        """Write frames to a temp mp4v file then re-encode to H.264 with ffmpeg."""
        # Step A: Write raw frames to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_f:
            tmp_path = Path(tmp_f.name)

        try:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(tmp_path), fourcc, self.fps, (w, h), isColor=True)
            if not writer.isOpened():
                logger.error("OpenCV VideoWriter failed to open temp path %s", tmp_path)
                return None

            try:
                for _, _, frame_bgr in frames:
                    if frame_bgr.shape[:2] != (h, w):
                        frame_bgr = cv2.resize(frame_bgr, (w, h))
                    writer.write(frame_bgr)
            finally:
                writer.release()

            if not tmp_path.exists() or tmp_path.stat().st_size == 0:
                logger.error("Temp clip is empty after writing frames: %s", tmp_path)
                return None

            # Step B: Re-encode to H.264/yuv420p/faststart with ffmpeg
            cmd = [
                _FFMPEG_PATH,
                "-y",                        # overwrite output
                "-i", str(tmp_path),         # input: temp mp4v file
                "-c:v", "libx264",           # H.264 codec
                "-pix_fmt", "yuv420p",       # required for browser playback
                "-movflags", "+faststart",   # moov atom first (progressive download)
                "-preset", "fast",           # encode speed vs quality trade-off
                "-an",                       # no audio
                str(output_path),
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=120)
            if result.returncode != 0:
                logger.error(
                    "ffmpeg re-encode failed (exit %d).\nSTDERR: %s",
                    result.returncode,
                    result.stderr.decode("utf-8", errors="replace")[-2000:],
                )
                # Fallback: copy the raw mp4v file so caller gets something
                shutil.copy2(str(tmp_path), str(output_path))
                logger.warning(
                    "Copied raw mp4v clip as fallback: %s (NOT browser-compatible)", output_path
                )
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

        if output_path.exists() and output_path.stat().st_size > 0:
            logger.info(
                "H.264 evidence clip written (%d frames, %.1f s): %s",
                len(frames),
                len(frames) / max(self.fps, 1.0),
                output_path,
            )
            return str(output_path).replace("\\", "/")
        else:
            logger.error("ffmpeg produced no output file at %s", output_path)
            return None

    def _write_via_opencv(
        self,
        frames: List[Tuple[float, int, np.ndarray]],
        output_path: Path,
        h: int,
        w: int,
    ) -> Optional[str]:
        """Fallback: write directly with OpenCV mp4v when ffmpeg is unavailable."""
        logger.warning(
            "Writing MPEG-4 Part-2 clip (NOT H.264) because ffmpeg is unavailable: %s", output_path
        )
        fourcc_code = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc_code, self.fps, (w, h), isColor=True)

        if not writer.isOpened():
            fallback_code = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(str(output_path), fallback_code, self.fps, (w, h), isColor=True)

        if not writer.isOpened():
            logger.error("Failed to open VideoWriter for %s", output_path)
            return None

        try:
            for _, _, frame_bgr in frames:
                if frame_bgr.shape[:2] != (h, w):
                    frame_bgr = cv2.resize(frame_bgr, (w, h))
                writer.write(frame_bgr)
        finally:
            writer.release()

        if output_path.exists() and output_path.stat().st_size > 0:
            return str(output_path).replace("\\", "/")
        else:
            logger.error("Clip writer finished but output file is empty: %s", output_path)
            return None
