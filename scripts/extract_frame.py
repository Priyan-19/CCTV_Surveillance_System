"""Video Frame Extraction Tool.

Extracts a single frame (by index or timestamp) from a video clip and saves it as
an image file (JPEG/PNG) for visual inspection, scene validation, or zone drawing.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional
import cv2

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("extract_frame")


def extract_frame(
    video_path: Path,
    output_path: Path,
    frame_idx: int = 0,
    time_sec: Optional[float] = None,
    dry_run: bool = False,
) -> bool:
    """Extract specified frame from video and save to disk."""
    if not video_path.exists():
        logger.error("Video file does not exist: %s", video_path)
        return False

    if dry_run:
        logger.info(
            "[DRY-RUN] Would extract frame %d from %s -> %s",
            frame_idx,
            video_path.name,
            output_path,
        )
        return True

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Failed to open video: %s", video_path)
        return False

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        target_idx = frame_idx
        if time_sec is not None and fps > 0:
            target_idx = int(time_sec * fps)

        if target_idx < 0 or (total_frames > 0 and target_idx >= total_frames):
            logger.warning(
                "Target frame %d out of bounds (0 to %d). Clamping to 0.",
                target_idx,
                total_frames - 1,
            )
            target_idx = 0

        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        success, frame = cap.read()
        if not success or frame is None:
            logger.error("Failed to read frame %d from %s", target_idx, video_path.name)
            return False

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), frame)
        logger.info(
            "Extracted frame %d (%dx%d) -> %s",
            target_idx,
            width,
            height,
            output_path,
        )
        return True
    finally:
        cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a frame from a surveillance video clip as an image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Path to input video file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination path for saved frame (default: evidence/frames/<clip_stem>_frame<N>.jpg).",
    )
    parser.add_argument(
        "--frame-idx",
        type=int,
        default=0,
        help="Zero-based frame index to extract.",
    )
    parser.add_argument(
        "--time-sec",
        type=float,
        default=None,
        help="Timestamp in seconds to extract (overrides --frame-idx).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate frame extraction without saving image.",
    )

    args = parser.parse_args()

    video_path: Path = args.video
    if args.output is None:
        idx_str = f"time{int(args.time_sec)}s" if args.time_sec is not None else f"f{args.frame_idx}"
        output_path = Path("evidence/frames") / f"{video_path.stem}_{idx_str}.jpg"
    else:
        output_path = args.output

    ok = extract_frame(
        video_path=video_path,
        output_path=output_path,
        frame_idx=args.frame_idx,
        time_sec=args.time_sec,
        dry_run=args.dry_run,
    )

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
