"""Sample Set Assembler & Manifest Generator.

Selects, standardises, and copies representative test clips into data/samples/
for Phases 0 to 3 testing.
Inspects video dimensions, FPS, and durations using OpenCV and generates
data/samples_manifest.yaml with ground-truth expected events.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("make_sample_set")

SAMPLE_MAPPINGS: List[Dict[str, Any]] = [
    {
        "target_name": "sample_walking_01.mp4",
        "source_candidates": [
            Path("data/datasets/caviar/mp4/Walk1.mp4"),
            Path("data/datasets/caviar/raw/Walk1.mpg"),
        ],
        "source_dataset": "CAVIAR",
        "scenario": "walking_single",
        "expected_events": "One person walking in a corridor and returning. Validates YOLO person detection and single-track ID persistence.",
    },
    {
        "target_name": "sample_walking_02.mp4",
        "source_candidates": [
            Path("data/datasets/caviar/mp4/Walk2.mp4"),
            Path("data/datasets/caviar/raw/Walk2.mpg"),
        ],
        "source_dataset": "CAVIAR",
        "scenario": "walking_pair",
        "expected_events": "Two pedestrians walking in lobby. Validates multi-target tracking without ID swap.",
    },
    {
        "target_name": "sample_abandoned_bag_01.mp4",
        "source_candidates": [
            Path("data/datasets/caviar/mp4/LeftBag.mp4"),
            Path("data/datasets/caviar/raw/LeftBag.mpg"),
        ],
        "source_dataset": "CAVIAR",
        "scenario": "abandoned_object_wall",
        "expected_events": "Person enters, drops backpack on floor near wall, and departs. Expected: ABANDONED_OBJECT trigger after dwell threshold.",
    },
    {
        "target_name": "sample_abandoned_bag_02.mp4",
        "source_candidates": [
            Path("data/datasets/caviar/mp4/LeftBag_AtChair.mp4"),
            Path("data/datasets/caviar/raw/LeftBag_AtChair.mpg"),
        ],
        "source_dataset": "CAVIAR",
        "scenario": "abandoned_object_chair",
        "expected_events": "Person drops bag beside waiting chairs and leaves scene. Expected: ABANDONED_OBJECT trigger.",
    },
    {
        "target_name": "sample_abandoned_bag_hd.mp4",
        "source_candidates": [
            Path("data/datasets/aboda/video1.mp4"),
            Path("data/datasets/aboda/video1.avi"),
        ],
        "source_dataset": "ABODA",
        "scenario": "abandoned_object_hd",
        "expected_events": "Unattended bag scenario at 640x480 resolution (meets 640px YOLO baseline). Expected: ABANDONED_OBJECT trigger.",
    },
    {
        "target_name": "sample_fight_01.mp4",
        "source_candidates": [
            Path("data/datasets/caviar/mp4/Fight_RunAway1.mp4"),
            Path("data/datasets/caviar/raw/Fight_RunAway1.mpg"),
        ],
        "source_dataset": "CAVIAR",
        "scenario": "fighting_and_running",
        "expected_events": "Two individuals meet, fight briefly, then separate and run away. Expected: Altercation motion and rapid running.",
    },
    {
        "target_name": "sample_normal_01.mp4",
        "source_candidates": [
            Path("data/datasets/caviar/mp4/Walk3.mp4"),
            Path("data/datasets/caviar/raw/Walk3.mpg"),
        ],
        "source_dataset": "CAVIAR",
        "scenario": "normal_pedestrians",
        "expected_events": "Multiple individuals walking and crossing paths peacefully. Validates baseline false-positive prevention.",
    },
]


def inspect_video_file(video_path: Path) -> Dict[str, Any]:
    """Inspect video file properties using OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV failed to open video file: {video_path}")

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = (frame_count / fps) if fps > 0 else 0.0

        return {
            "width": width,
            "height": height,
            "resolution": f"{width}x{height}",
            "fps": round(fps, 2),
            "frame_count": frame_count,
            "duration_seconds": round(duration, 2),
        }
    finally:
        cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble sample test clips into data/samples/ and generate samples_manifest.yaml.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=Path("data/samples"),
        help="Destination directory for sample clips.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("data/samples_manifest.yaml"),
        help="Path where samples_manifest.yaml will be written.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing clips in data/samples/.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate file copies and print manifest content without writing.",
    )

    args = parser.parse_args()

    args.samples_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("                    SAMPLE SET ASSEMBLER & MANIFEST GENERATOR")
    print("=" * 80)
    print(f"Target Samples Directory : {args.samples_dir.resolve()}")
    print(f"Manifest Output Path     : {args.manifest_path.resolve()}")
    print(f"Mode                     : {'DRY RUN (simulation)' if args.dry_run else 'LIVE COPY & MANIFEST'}")
    print("-" * 80)

    manifest_entries: List[Dict[str, Any]] = []
    copied_count = 0

    for mapping in SAMPLE_MAPPINGS:
        target_name = mapping["target_name"]
        target_path = args.samples_dir / target_name
        source_found: Optional[Path] = None

        for cand in mapping["source_candidates"]:
            if cand.exists():
                source_found = cand
                break

        if not source_found:
            logger.warning(
                "Source clip not found for %s. (Checked: %s)",
                target_name,
                [str(c) for c in mapping["source_candidates"]],
            )
            continue

        # Check if already copied
        if not target_path.exists() or args.force:
            if args.dry_run:
                logger.info("[DRY-RUN] Would copy: %s -> %s", source_found, target_path)
            else:
                shutil.copy2(source_found, target_path)
                logger.info("Copied: %s -> %s", source_found.name, target_name)
                copied_count += 1
        else:
            logger.info("Sample clip already exists: %s", target_name)

        # Inspect video
        inspect_file = target_path if target_path.exists() else source_found
        try:
            v_meta = inspect_video_file(inspect_file)
        except Exception as exc:
            logger.error("Could not inspect %s: %s", inspect_file.name, exc)
            v_meta = {
                "width": 0,
                "height": 0,
                "resolution": "unknown",
                "fps": 0.0,
                "frame_count": 0,
                "duration_seconds": 0.0,
            }

        entry = {
            "file": target_name,
            "source_dataset": mapping["source_dataset"],
            "original_name": source_found.name,
            "resolution": v_meta["resolution"],
            "fps": v_meta["fps"],
            "frame_count": v_meta["frame_count"],
            "duration_seconds": v_meta["duration_seconds"],
            "scenario": mapping["scenario"],
            "expected_events": mapping["expected_events"],
        }
        manifest_entries.append(entry)

    print("-" * 80)
    manifest_data = {
        "version": "1.0",
        "description": "Sample test clips inventory for Phases 0 to 3. Quarantined from training data.",
        "clips": manifest_entries,
    }

    if args.dry_run:
        print("[DRY-RUN] Generated Manifest YAML:")
        print(yaml.safe_dump(manifest_data, sort_keys=False))
    else:
        with open(args.manifest_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(manifest_data, f, sort_keys=False)
        logger.info("Saved manifest with %d clips to: %s", len(manifest_entries), args.manifest_path)

    print("=" * 80)
    print(f"Summary: {len(manifest_entries)} clips catalogued in manifest.")
    print("=" * 80)


if __name__ == "__main__":
    main()
