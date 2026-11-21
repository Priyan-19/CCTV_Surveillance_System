"""ABODA Dataset Fetcher.

Clones or downloads the Abandoned Objects Dataset (ABODA) repository:
https://github.com/kevinlin311tw/ABODA

ABODA comprises 11 video sequences (.avi) labeled for abandoned object detection
across crowded, night, indoor, and outdoor surveillance environments.
"""

from __future__ import annotations

import argparse
import io
import logging
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fetch_aboda")

REPO_URL = "https://github.com/kevinlin311tw/ABODA.git"
ZIP_URL = "https://github.com/kevinlin311tw/ABODA/archive/refs/heads/master.zip"


def is_already_downloaded(dest_dir: Path) -> bool:
    """Check if ABODA sequences are already present."""
    if not dest_dir.exists():
        return False
    video_files = list(dest_dir.glob("*.avi")) + list(dest_dir.glob("*/*.avi"))
    return len(video_files) >= 1


def clone_repo(repo_url: str, dest_dir: Path, dry_run: bool = False) -> bool:
    """Clone repository via git with shallow depth."""
    cmd = ["git", "clone", "--depth", "1", repo_url, str(dest_dir)]
    cmd_str = " ".join(cmd)
    if dry_run:
        logger.info("[DRY-RUN] Command: %s", cmd_str)
        return True

    # If destination exists and only contains .gitkeep, clean it up so git clone can proceed
    if dest_dir.exists():
        existing_items = list(dest_dir.iterdir())
        if len(existing_items) == 1 and existing_items[0].name == ".gitkeep":
            existing_items[0].unlink()
            dest_dir.rmdir()
        elif len(existing_items) == 0:
            dest_dir.rmdir()

    logger.info("Executing git clone: %s", cmd_str)
    try:
        proc = subprocess.run(cmd, check=False)
        return proc.returncode == 0
    except Exception as exc:
        logger.error("Git execution failed: %s", exc)
        return False


def download_zip(zip_url: str, dest_dir: Path, dry_run: bool = False) -> bool:
    """Download repository archive as zip and extract."""
    if dry_run:
        logger.info("[DRY-RUN] Would download zip: %s and extract to %s", zip_url, dest_dir)
        return True

    logger.info("Downloading ZIP from: %s", zip_url)
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(
            zip_url,
            headers={"User-Agent": "Mozilla/5.0 (CCTV-Research-Dataset-Fetcher/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            content = resp.read()
            logger.info("Downloaded %d bytes. Extracting...", len(content))
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                # Strip root directory prefix if present (e.g. ABODA-master/)
                for member in zf.infolist():
                    parts = Path(member.filename).parts
                    if len(parts) > 1:
                        target_path = dest_dir / Path(*parts[1:])
                    else:
                        target_path = dest_dir / parts[0]

                    if member.is_dir():
                        target_path.mkdir(parents=True, exist_ok=True)
                    else:
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        with zf.open(member) as src, open(target_path, "wb") as dst:
                            shutil.copyfileobj(src, dst)
        logger.info("ZIP extracted successfully to: %s", dest_dir)
        return True
    except Exception as exc:
        logger.error("Failed to download or extract zip: %s", exc)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch ABODA (Abandoned Objects Dataset) via git clone or direct zip download.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=Path("data/datasets/aboda"),
        help="Target destination directory for ABODA dataset.",
    )
    parser.add_argument(
        "--use-zip",
        action="store_true",
        help="Force direct HTTP zip download instead of git clone.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without modifying disk.",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("                      ABODA DATASET ACQUISITION")
    print("=" * 80)
    print(f"Target Directory: {args.dest_dir.resolve()}")
    print(f"Repository URL  : {REPO_URL}")
    print(f"ZIP Fallback URL: {ZIP_URL}")
    print(f"Mode            : {'DRY RUN (simulation)' if args.dry_run else 'LIVE ACQUISITION'}")
    print("-" * 80)

    if is_already_downloaded(args.dest_dir):
        logger.info("ABODA sequences already found in %s. Skipping download.", args.dest_dir)
        print("=" * 80)
        return

    git_available = shutil.which("git") is not None
    if git_available and not args.use_zip:
        logger.info("Using git clone (shallow clone)...")
        ok = clone_repo(REPO_URL, args.dest_dir, dry_run=args.dry_run)
    else:
        logger.info("Using direct HTTP ZIP download...")
        ok = download_zip(ZIP_URL, args.dest_dir, dry_run=args.dry_run)

    print("-" * 80)
    if ok:
        print("[SUCCESS] ABODA acquisition plan complete.")
    else:
        print("[FAILED] Could not acquire ABODA dataset. Check network or permissions.")
    print("=" * 80)


if __name__ == "__main__":
    main()
