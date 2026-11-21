"""Track Stability & Trajectory Analysis Script.

Processes a video clip with ByteTrack and evaluates tracking stability:
1. Counts unique Track IDs generated over the video duration.
2. Computes trajectory length distributions (min, mean, median, max frames).
3. Identifies short/fragmented tracks (< threshold frames).
4. Generates an ID Timeline plot and a visual Contact Sheet of sampled tracked frames.

Usage:
    python scripts/track_report.py --source data/samples/sample_walking_01.mp4 --model yolo26s.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.phase1_pipeline import Phase1Pipeline
from src.common.types import FrameResult, Track

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("track_report")


def analyze_tracks(
    video_path: Path,
    model_name: str = "yolo26s.pt",
    imgsz: int = 640,
    conf: float = 0.25,
    short_track_thresh: int = 15,
    output_dir: Path = Path("data/output/track_reports"),
    max_frames: int = 600,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    camera_id = f"eval_{video_path.stem}"

    # Track data collectors
    # track_id -> list of frame_ids
    track_frames: Dict[int, List[int]] = defaultdict(list)
    track_classes: Dict[int, str] = {}
    sampled_frames: List[Tuple[int, np.ndarray]] = []

    def frame_callback(result: FrameResult) -> None:
        # Sample every 30th frame for the visual contact sheet
        if result.frame_id % 30 == 0 and len(sampled_frames) < 12:
            sampled_frames.append((result.frame_id, result.frame.copy()))

        for trk in result.tracks:
            track_frames[trk.track_id].append(result.frame_id)
            track_classes[trk.track_id] = trk.class_name

    pipeline = Phase1Pipeline(
        camera_id=camera_id,
        source=str(video_path),
        model_name=model_name,
        tracker_config="configs/bytetrack.yaml",
        device="cuda:0",
        half_precision=True,
        imgsz=imgsz,
        conf_threshold=conf,
        frame_skip=0,
        output_dir=output_dir,
        save_video=False,
        show_window=False,
    )

    print("=" * 80)
    print(f"             TRACK ID STABILITY ANALYSIS: {video_path.name}")
    print("=" * 80)
    print(f"Model: {model_name} | ImgSz: {imgsz}px | Conf: {conf} | Max Frames: {max_frames}")

    _ = pipeline.run(max_frames=max_frames, callback=frame_callback)

    unique_ids = list(track_frames.keys())
    total_unique_ids = len(unique_ids)

    if total_unique_ids == 0:
        print("[WARNING] Zero tracks were formed during this evaluation.")
        print("=" * 80)
        return

    lengths = [len(frames) for frames in track_frames.values()]
    short_tracks = [tid for tid, frames in track_frames.items() if len(frames) < short_track_thresh]

    min_len = min(lengths)
    max_len = max(lengths)
    mean_len = sum(lengths) / len(lengths)
    median_len = sorted(lengths)[len(lengths) // 2]

    print("\n[QUANTITATIVE TRACKING METRICS]")
    print(f"Total Unique Track IDs : {total_unique_ids}")
    print(f"Short Track Fragments  : {len(short_tracks)} tracks (< {short_track_thresh} frames duration)")
    print(f"Track Lifespan (Min)   : {min_len} frames")
    print(f"Track Lifespan (Mean)  : {mean_len:.1f} frames")
    print(f"Track Lifespan (Median): {median_len} frames")
    print(f"Track Lifespan (Max)   : {max_len} frames")
    print("-" * 80)

    print("Track ID Breakdown:")
    for tid in sorted(unique_ids):
        f_list = track_frames[tid]
        cls = track_classes.get(tid, "unknown")
        is_frag = "[FRAGMENT?]" if len(f_list) < short_track_thresh else "[STABLE]"
        print(f"  Track ID {tid:2d} ({cls:<10}): {len(f_list):4d} frames (from frame {f_list[0]} to {f_list[-1]}) {is_frag}")

    # 1. Generate ID Timeline Chart using Matplotlib
    fig, ax = plt.subplots(figsize=(12, max(4, total_unique_ids * 0.4)))
    y_ticks = []
    y_labels = []

    for idx, tid in enumerate(sorted(unique_ids)):
        f_list = track_frames[tid]
        cls = track_classes.get(tid, "unknown")
        ax.hlines(y=idx, xmin=min(f_list), xmax=max(f_list), linewidth=6, alpha=0.8)
        y_ticks.append(idx)
        y_labels.append(f"ID {tid} ({cls})")

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Active Track")
    ax.set_title(f"Track Lifespan Timeline: {video_path.name} ({model_name} + ByteTrack)")
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()

    timeline_path = output_dir / f"{video_path.stem}_id_timeline.png"
    plt.savefig(timeline_path, dpi=150)
    plt.close()
    print(f"\n[SAVED] ID Timeline chart saved to: {timeline_path.resolve()}")

    # 2. Generate Visual Contact Sheet of Sampled Annotated Frames
    if sampled_frames:
        num_samples = len(sampled_frames)
        cols = 3
        rows = (num_samples + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3.5))
        axes = np.array(axes).reshape(-1)

        for i in range(len(axes)):
            if i < num_samples:
                fid, bgr_img = sampled_frames[i]
                rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
                axes[i].imshow(rgb_img)
                axes[i].set_title(f"Frame #{fid}")
                axes[i].axis("off")
            else:
                axes[i].axis("off")

        plt.suptitle(f"Sampled Frames Contact Sheet: {video_path.name}", fontsize=14)
        plt.tight_layout()
        contact_path = output_dir / f"{video_path.stem}_contact_sheet.png"
        plt.savefig(contact_path, dpi=130)
        plt.close()
        print(f"[SAVED] Contact sheet saved to: {contact_path.resolve()}")

    print("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Track ID Stability Report and Timeline.")
    parser.add_argument(
        "--source",
        type=str,
        default="data/samples/sample_walking_01.mp4",
        help="Path to video file to evaluate.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo26s.pt",
        help="YOLO model weights (default: yolo26s.pt).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference frame resolution dimension (default: 640).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold.",
    )
    parser.add_argument(
        "--short-thresh",
        type=int,
        default=15,
        help="Frame threshold below which a track is flagged as potential fragment.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=600,
        help="Maximum frames to process.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/output/track_reports",
        help="Directory where timeline plots and contact sheets will be saved.",
    )

    args = parser.parse_args()
    v_path = Path(args.source)
    if not v_path.exists():
        logger.error("Target video '%s' not found.", v_path)
        sys.exit(1)

    analyze_tracks(
        video_path=v_path,
        model_name=args.model,
        imgsz=args.imgsz,
        conf=args.conf,
        short_track_thresh=args.short_thresh,
        output_dir=Path(args.output_dir),
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
