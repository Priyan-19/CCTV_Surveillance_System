"""ChokePoint Dataset Fetcher (Phase 3 Face Recognition).

Downloads selected ChokePoint sequences from Zenodo (record 815657):
https://zenodo.org/records/815657

Features:
- Enforces strict disk space safety check (aborts if free space < 2x download size).
- Downloads only selected sequences (default: 3 Portal 1 sequences + ground truth).
- Verifies official MD5 checksums before extraction.
- Skips already downloaded/extracted sequences.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, List, NamedTuple

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fetch_chokepoint")

ZENODO_RECORD_ID = "815657"
ZENODO_BASE_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files"


class ChokePointArchive(NamedTuple):
    filename: str
    md5: str
    size_bytes: int
    description: str


# Catalog of Portal 1 archives with official Zenodo checksums
CHOKEPOINT_FILES: Dict[str, ChokePointArchive] = {
    "groundtruth.tar.xz": ChokePointArchive(
        filename="groundtruth.tar.xz",
        md5="5c9d4d38a1c614905fe48da48ecdf06c",
        size_bytes=467352,  # ~456 KB
        description="Portal 1 & 2 XML bounding boxes and eye/nose landmark coordinates",
    ),
    "P1E.tar.xz": ChokePointArchive(
        filename="P1E.tar.xz",
        md5="07203fe0de5ebb1270742bd711f11dd8",
        size_bytes=69625440,  # ~66.4 MB
        description="Portal 1 Enter evaluation set (compact)",
    ),
    "P1E_S1.tar.xz": ChokePointArchive(
        filename="P1E_S1.tar.xz",
        md5="195191772856c1a62650a1d47e897b2a",
        size_bytes=404016332,  # ~385.3 MB
        description="Portal 1 Enter, Session 1 (25 subjects entering doorway)",
    ),
    "P1E_S2.tar.xz": ChokePointArchive(
        filename="P1E_S2.tar.xz",
        md5="bc58ad9ef36ceb92da592cb092353d55",
        size_bytes=421731532,  # ~402.2 MB
        description="Portal 1 Enter, Session 2 (25 subjects entering doorway)",
    ),
    "P1E_S3.tar.xz": ChokePointArchive(
        filename="P1E_S3.tar.xz",
        md5="8f0c55f28905cb922f4f6513d55c2646",
        size_bytes=444700876,  # ~424.1 MB
        description="Portal 1 Enter, Session 3 (25 subjects entering doorway)",
    ),
}

DEFAULT_SELECTION: List[str] = [
    "groundtruth.tar.xz",
    "P1E_S1.tar.xz",
    "P1E_S2.tar.xz",
    "P1E_S3.tar.xz",
]


def compute_md5(file_path: Path) -> str:
    """Compute MD5 checksum of a file."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def download_with_progress(url: str, dest_path: Path, expected_bytes: int) -> bool:
    """Download file from URL showing progress."""
    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    logger.info("Downloading from: %s", url)
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (CCTV-Research-Dataset-Fetcher/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp, open(temp_path, "wb") as out_f:
            downloaded = 0
            block_size = 128 * 1024
            while True:
                buf = resp.read(block_size)
                if not buf:
                    break
                downloaded += len(buf)
                out_f.write(buf)
                if expected_bytes > 0:
                    pct = downloaded * 100 / expected_bytes
                    print(
                        f"\r  Downloaded: {downloaded / (1024 * 1024):.1f}/{expected_bytes / (1024 * 1024):.1f} MB ({pct:.1f}%)",
                        end="",
                        flush=True,
                    )
            print()
        temp_path.replace(dest_path)
        return True
    except Exception as exc:
        if temp_path.exists():
            temp_path.unlink()
        logger.error("Download failed for %s: %s", dest_path.name, exc)
        return False


def extract_archive(archive_path: Path, extract_dir: Path) -> bool:
    """Extract .tar.xz archive."""
    logger.info("Extracting %s into %s...", archive_path.name, extract_dir)
    try:
        with tarfile.open(archive_path, "r:xz") as tar:
            tar.extractall(path=extract_dir)
        logger.info("Extraction complete: %s", archive_path.name)
        return True
    except Exception as exc:
        logger.error("Failed to extract %s: %s", archive_path.name, exc)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch and verify selected ChokePoint face recognition dataset sequences.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--archives",
        nargs="+",
        default=DEFAULT_SELECTION,
        choices=list(CHOKEPOINT_FILES.keys()),
        help="Archives to download.",
    )
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=Path("data/datasets/chokepoint"),
        help="Directory to store downloaded and extracted archives.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Display plan, sizes, and disk check without downloading.",
    )

    args = parser.parse_args()

    dest_dir = args.dest_dir
    dest_dir.mkdir(parents=True, exist_ok=True)

    # 1. Calculate required download size
    total_download_bytes = sum(CHOKEPOINT_FILES[name].size_bytes for name in args.archives)
    total_download_mb = total_download_bytes / (1024 * 1024)
    required_free_bytes = total_download_bytes * 2
    required_free_mb = required_free_bytes / (1024 * 1024)

    # 2. Check disk space
    total_disk, used_disk, free_disk = shutil.disk_usage(dest_dir)
    free_mb = free_disk / (1024 * 1024)

    print("=" * 80)
    print("                     CHOKEPOINT DATASET ACQUISITION")
    print("=" * 80)
    print(f"Target Directory     : {dest_dir.resolve()}")
    print(f"Selected Archives    : {len(args.archives)}")
    print(f"Total Download Size  : {total_download_mb:.1f} MB")
    print(f"Required Free Space  : {required_free_mb:.1f} MB (2x safety margin)")
    print(f"Available Free Space : {free_mb:.1f} MB")
    print(f"Mode                 : {'DRY RUN (simulation)' if args.dry_run else 'LIVE ACQUISITION'}")
    print("-" * 80)

    for name in args.archives:
        info = CHOKEPOINT_FILES[name]
        print(f"  * {name:<18} ({info.size_bytes / (1024*1024):>6.1f} MB) - {info.description}")
    print("-" * 80)

    if free_disk < required_free_bytes:
        logger.error(
            "INSUFFICIENT DISK SPACE: Free space (%.1f MB) is below safety requirement 2x download size (%.1f MB).",
            free_mb,
            required_free_mb,
        )
        sys.exit(1)

    if args.dry_run:
        print("[DRY-RUN] Disk space safety check passed. No files downloaded.")
        print("=" * 80)
        return

    # 3. Process each archive
    for name in args.archives:
        info = CHOKEPOINT_FILES[name]
        target_file = dest_dir / name
        url = f"{ZENODO_BASE_URL}/{name}?download=1"

        if target_file.exists():
            logger.info("Found existing archive: %s. Verifying MD5...", name)
            local_md5 = compute_md5(target_file)
            if local_md5 == info.md5:
                logger.info("MD5 matches (%s). Skipping download.", local_md5)
            else:
                logger.warning("MD5 mismatch for %s! Redownloading.", name)
                target_file.unlink()

        if not target_file.exists():
            ok = download_with_progress(url, target_file, info.size_bytes)
            if not ok:
                logger.error("Download failed for %s. Aborting.", name)
                sys.exit(1)

            logger.info("Verifying MD5 checksum for %s...", name)
            downloaded_md5 = compute_md5(target_file)
            if downloaded_md5 != info.md5:
                logger.error("Checksum verification FAILED for %s (got %s, expected %s)", name, downloaded_md5, info.md5)
                target_file.unlink()
                sys.exit(1)
            logger.info("Checksum verified OK: %s", downloaded_md5)

        # Extract
        extract_archive(target_file, dest_dir)

    print("=" * 80)
    print("[SUCCESS] ChokePoint sequences downloaded, verified, and extracted.")
    print("=" * 80)


if __name__ == "__main__":
    main()
