"""Dataset Leakage & Quarantine Verification Script.

Enforces strict data hygiene across project phases:
1. Verifies that NO video file or hash in data/samples/ exists in any training or validation
   dataset directory (e.g. data/datasets/rwf2000/train, data/datasets/rwf2000/val).
2. Verifies that the RWF-2000 train/val split is strictly disjoint (no duplicate clips or hashes).

Exits with code 0 if completely clean.
Exits with code 1 if any data leakage or split contamination is detected.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("check_leakage")

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".mpg", ".mpeg", ".wmv", ".webm"}


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def collect_video_files(directory: Path) -> List[Path]:
    """Recursively collect all video files under a directory."""
    if not directory.exists():
        return []
    return [
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS and not p.name.startswith("output_")
    ]


def verify_sample_quarantine(
    samples_dir: Path,
    training_dirs: List[Path],
) -> List[str]:
    """Check if any sample file matches any training file by hash or name."""
    violations: List[str] = []
    sample_files = collect_video_files(samples_dir)

    if not sample_files:
        logger.info("No sample files found in %s to check against training sets.", samples_dir)
        return violations

    logger.info("Indexing %d sample clips for quarantine verification...", len(sample_files))
    sample_hashes: Dict[str, Path] = {}
    sample_stems: Dict[str, Path] = {}

    for s in sample_files:
        h = compute_sha256(s)
        sample_hashes[h] = s
        sample_stems[s.stem.lower()] = s

    for train_dir in training_dirs:
        if not train_dir.exists():
            continue

        train_files = collect_video_files(train_dir)
        for t in train_files:
            h = compute_sha256(t)
            if h in sample_hashes:
                violations.append(
                    f"EXACT HASH MATCH: Sample '{sample_hashes[h]}' is duplicated in training set '{t}' (SHA256: {h})"
                )
            if t.stem.lower() in sample_stems:
                violations.append(
                    f"POTENTIAL STEM OVERLAP: Sample '{sample_stems[t.stem.lower()]}' shares basename with training clip '{t}'"
                )

    return violations


def verify_split_disjoint(
    split_a_dir: Path,
    split_b_dir: Path,
    label: str,
) -> List[str]:
    """Verify that two dataset folders have zero overlapping files or hashes."""
    violations: List[str] = []
    files_a = collect_video_files(split_a_dir)
    files_b = collect_video_files(split_b_dir)

    if not files_a or not files_b:
        return violations

    hashes_a: Dict[str, Path] = {compute_sha256(f): f for f in files_a}
    names_a: Dict[str, Path] = {f.name.lower(): f for f in files_a}

    for fb in files_b:
        hb = compute_sha256(fb)
        if hb in hashes_a:
            violations.append(
                f"[{label}] Hash collision between '{hashes_a[hb]}' and '{fb}' (SHA256: {hb})"
            )
        if fb.name.lower() in names_a:
            violations.append(
                f"[{label}] Duplicate filename between '{names_a[fb.name.lower()]}' and '{fb}'"
            )

    return violations


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify zero data leakage between data/samples/ and dataset training folders.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=Path("data/samples"),
        help="Path to quarantined sample test clips.",
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=Path("data/datasets"),
        help="Path to datasets root directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform index reconnaissance without reporting failure.",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("                    DATA LEAKAGE & QUARANTINE AUDITOR")
    print("=" * 80)
    print(f"Sample Quarantine Directory : {args.samples_dir.resolve()}")
    print(f"Datasets Root Directory     : {args.datasets_dir.resolve()}")
    print("-" * 80)

    # 1. Training directories to quarantine against samples
    training_target_dirs = [
        args.datasets_dir / "rwf2000" / "train",
        args.datasets_dir / "rwf2000" / "val",
    ]

    quarantine_violations = verify_sample_quarantine(args.samples_dir, training_target_dirs)

    # 2. RWF-2000 Train vs Val split overlap check
    rwf_train_fight = args.datasets_dir / "rwf2000" / "train" / "Fight"
    rwf_val_fight = args.datasets_dir / "rwf2000" / "val" / "Fight"
    rwf_train_nonfight = args.datasets_dir / "rwf2000" / "train" / "NonFight"
    rwf_val_nonfight = args.datasets_dir / "rwf2000" / "val" / "NonFight"

    split_violations: List[str] = []
    split_violations.extend(verify_split_disjoint(rwf_train_fight, rwf_val_fight, "RWF2000-Fight"))
    split_violations.extend(verify_split_disjoint(rwf_train_nonfight, rwf_val_nonfight, "RWF2000-NonFight"))

    all_violations = quarantine_violations + split_violations

    print("-" * 80)
    if all_violations:
        print(f"[FAIL] DATA LEAKAGE DETECTED: {len(all_violations)} violation(s) found!")
        for idx, v in enumerate(all_violations, start=1):
            print(f"  {idx}. {v}")
        print("=" * 80)
        if not args.dry_run:
            sys.exit(1)
    else:
        print("[PASS] ZERO LEAKAGE DETECTED.")
        print("  - Samples in data/samples/ are strictly quarantined from all training sets.")
        print("  - No split overlap detected between training and validation partitions.")
        print("=" * 80)
        sys.exit(0)


if __name__ == "__main__":
    main()
