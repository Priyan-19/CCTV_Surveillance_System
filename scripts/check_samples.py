"""Sample Video Inspection & Validation Script.

Scans the data/samples/ directory for test clips, extracts video properties
(resolution, FPS, total frame count, duration), cross-references
data/samples_manifest.yaml for scenarios and expected events, and validates:
1. At least 5 sample clips are present for Phase 0 completion.
2. Flags warnings when video width is below 640px (YOLO detection sensitivity).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional
import cv2
import yaml


class VideoMetadata(NamedTuple):
    filename: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float
    codec: str


def inspect_video(video_path: Path) -> Optional[VideoMetadata]:
    """Read video properties using OpenCV VideoCapture."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]).strip()
        duration = (frame_count / fps) if fps > 0 else 0.0

        return VideoMetadata(
            filename=video_path.name,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration_sec=duration,
            codec=codec,
        )
    finally:
        cap.release()


def load_manifest(manifest_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load sample manifest yaml if present."""
    if not manifest_path.exists():
        return {}
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if isinstance(data, dict) and "clips" in data and isinstance(data["clips"], list):
                return {item["file"]: item for item in data["clips"] if "file" in item}
    except Exception:
        pass
    return {}


def scan_samples_directory(
    samples_dir: Path = Path("data/samples"),
    manifest_path: Path = Path("data/samples_manifest.yaml"),
) -> int:
    """Scan, tabulate, and validate all video clips in the target directory."""
    print("=" * 95)
    print(f"               SAMPLE VIDEO RECONNAISSANCE & VALIDATION ({samples_dir.resolve()})")
    print("=" * 95)

    if not samples_dir.exists():
        print(f"[ERROR] Directory '{samples_dir}' does not exist.")
        return 1

    video_extensions = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".webm", ".m4v", ".mpg"}
    found_files: List[Path] = [
        f for f in sorted(samples_dir.iterdir())
        if f.is_file() and f.suffix.lower() in video_extensions and not f.name.startswith("output_")
    ]

    count = len(found_files)
    print(f"Total Video Clips Discovered: {count}\n")

    manifest = load_manifest(manifest_path)

    # Print table header
    print(f"{'#':<3} | {'Filename':<28} | {'Resolution':<12} | {'FPS':<6} | {'Frames':<8} | {'Duration (s)':<12} | {'Scenario'}")
    print("-" * 95)

    low_res_warnings: List[str] = []
    valid_count = 0

    for idx, file_path in enumerate(found_files, start=1):
        meta = inspect_video(file_path)
        if meta is None:
            print(f"{idx:<3} | {file_path.name:<28} | [CORRUPT / UNABLE TO READ]")
            continue

        valid_count += 1
        res_str = f"{meta.width}x{meta.height}"
        scenario = manifest.get(file_path.name, {}).get("scenario", "n/a")

        print(
            f"{idx:<3} | {meta.filename:<28} | {res_str:<12} | {meta.fps:<6.2f} | "
            f"{meta.frame_count:<8} | {meta.duration_sec:<12.2f} | {scenario}"
        )

        if meta.width < 640:
            low_res_warnings.append(
                f"  * {meta.filename}: {res_str} is below 640 px wide. Small people and distant bags will be hard for YOLO to detect."
            )

    print("-" * 95)

    # Validation Checks
    has_errors = False

    if count < 5:
        print(f"\n[WARNING] Found {count} clip(s). At least 5 clips representing distinct scenarios are required.")
        print("  Required scenarios: walking, restricted entry, bag left behind, crowd/occupancy, running/fighting.")
        has_errors = True
    else:
        print(f"\n[PASSED] Quantity check satisfied: {count} sample clips available (>= 5 required).")

    if low_res_warnings:
        print("\n[RESOLUTION ADVISORY - LOW RESOLUTION DETECTED]:")
        for warn in low_res_warnings:
            print(warn)
        print("  Recommendation: Supplement with higher-resolution benchmarks (e.g. ABODA 1080p) for small-object validation.")

    print("=" * 95)
    return 1 if (count == 0 or has_errors) else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect video properties and validate sample clips in data/samples/.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=Path("data/samples"),
        help="Path to directory containing sample video files.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("data/samples_manifest.yaml"),
        help="Path to sample manifest YAML file.",
    )

    args = parser.parse_args()
    exit_code = scan_samples_directory(args.samples_dir, args.manifest_path)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
