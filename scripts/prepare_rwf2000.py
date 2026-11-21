"""RWF-2000 Dataset Preparation Script (Phase 7 Placeholder).

Prepares and verifies the RWF-2000 surveillance violence dataset:
- 2,000 video clips (5s @ 30 FPS)
- Official 80/20 train/validation split:
    data/datasets/rwf2000/train/Fight (800 clips)
    data/datasets/rwf2000/train/NonFight (800 clips)
    data/datasets/rwf2000/val/Fight (200 clips)
    data/datasets/rwf2000/val/NonFight (200 clips)

Source:
https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection
Mirror: https://huggingface.co/datasets/DanJoshua/RWF-2000

SAFETY GUARD:
This script is prepared for Phase 7 (CNN-LSTM Activity Recognition).
Per project plan, it must NOT be executed for full download during Phase 0.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("prepare_rwf2000")

EXPECTED_SUBDIRS = [
    Path("train/Fight"),
    Path("train/NonFight"),
    Path("val/Fight"),
    Path("val/NonFight"),
]


def verify_split_structure(base_dir: Path) -> Dict[str, int]:
    """Inspect and count existing video clips in RWF-2000 directory structure."""
    counts: Dict[str, int] = {}
    for subdir in EXPECTED_SUBDIRS:
        folder = base_dir / subdir
        if not folder.exists():
            counts[str(subdir)] = 0
            continue
        valid_exts = {".mp4", ".avi", ".mkv"}
        clips = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
        counts[str(subdir)] = len(clips)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare and verify RWF-2000 dataset for CNN-LSTM violence model (Phase 7).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/datasets/rwf2000"),
        help="Base directory for RWF-2000 dataset.",
    )
    parser.add_argument(
        "--check-status",
        action="store_true",
        help="Check existing dataset folders and clip counts without modifying anything.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate directory creation and print Phase 7 download instructions.",
    )
    parser.add_argument(
        "--confirm-phase7",
        action="store_true",
        help="Safety flag required to authorize live dataset download in Phase 7.",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("               RWF-2000 VIOLENCE DATASET PREPARATION (PHASE 7)")
    print("=" * 80)
    print(f"Base Directory : {args.dataset_dir.resolve()}")
    print("Official Split : 80% Train (1600 clips) / 20% Val (400 clips)")
    print("Target Classes : Fight (1000 clips) and NonFight (1000 clips)")
    print("-" * 80)

    # Always ensure skeleton directories exist with .gitkeep
    for subdir in EXPECTED_SUBDIRS:
        folder = args.dataset_dir / subdir
        if not args.dry_run:
            folder.mkdir(parents=True, exist_ok=True)
            gitkeep = folder / ".gitkeep"
            if not gitkeep.exists():
                gitkeep.touch()

    counts = verify_split_structure(args.dataset_dir)
    print("Current Clip Inventory:")
    for sub, cnt in counts.items():
        print(f"  * {sub:<20}: {cnt:>4} clips")
    print("-" * 80)

    if args.check_status:
        total = sum(counts.values())
        print(f"Total clips present: {total}/2000")
        print("=" * 80)
        return

    if not args.confirm_phase7:
        print("[SAFETY NOTICE]")
        print("  RWF-2000 download is scheduled for Phase 7 (CNN-LSTM fine-tuning).")
        print("  Phase 0 only requires the folder skeleton and verification script.")
        print("  To simulate actions, run with '--dry-run'.")
        print("  To execute live download in Phase 7, run with '--confirm-phase7'.")
        print("=" * 80)
        return

    if args.dry_run:
        print("[DRY-RUN] Directory skeleton verified.")
        print("[DRY-RUN] In Phase 7, clips will be downloaded from:")
        print("          https://huggingface.co/datasets/DanJoshua/RWF-2000")
        print("          or cloned via official git repo.")
        print("=" * 80)
        return

    logger.info("Phase 7 confirmed. Downloading dataset via Hugging Face Hub snapshot...")
    try:
        from huggingface_hub import snapshot_download  # type: ignore

        snapshot_download(
            repo_id="DanJoshua/RWF-2000",
            repo_type="dataset",
            local_dir=str(args.dataset_dir),
        )
        logger.info("Download completed.")
    except ImportError:
        logger.error(
            "huggingface_hub package is not installed. In Phase 7, install it via: uv add huggingface_hub"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
