"""CAVIAR Dataset Fetcher.

Downloads selected test sequences and ground-truth XML files from the CAVIAR
benchmark (Context Aware Vision using Image-based Active Recognition):
https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/
"""

from __future__ import annotations

import argparse
import logging
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fetch_caviar")

BASE_URLS = [
    "https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1",
    "http://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1",
]

CAVIAR_CATALOG: Dict[str, Dict[str, str | float]] = {
    "Walk1": {
        "video": "Walk1.mpg",
        "gt": "wk1gt.xml",
        "path_video": "Walk1/Walk1.mpg",
        "path_gt": "Walk1/wk1gt.xml",
        "size_mb": 7.0,
        "scenario": "Walking alone along corridor",
        "expected_events": "person walking in straight line and returning",
    },
    "Walk2": {
        "video": "Walk2.mpg",
        "gt": "wk2gt.xml",
        "path_video": "Walk2/Walk2.mpg",
        "path_gt": "Walk2/wk2gt.xml",
        "size_mb": 8.0,
        "scenario": "Two people walking in lobby",
        "expected_events": "two persons walking along lobby path",
    },
    "Walk3": {
        "video": "Walk3.mpg",
        "gt": "wk3gt.xml",
        "path_video": "Walk3/Walk3.mpg",
        "path_gt": "Walk3/wk3gt.xml",
        "size_mb": 10.0,
        "scenario": "Multiple people walking and crossing paths",
        "expected_events": "pedestrians crossing paths with occlusions",
    },
    "LeftBag": {
        "video": "LeftBag.mpg",
        "gt": "lb1gt.xml",
        "path_video": "LeftBag/LeftBag.mpg",
        "path_gt": "LeftBag/lb1gt.xml",
        "size_mb": 17.0,
        "scenario": "Abandoned backpack near wall",
        "expected_events": "abandoned_object after bag put down and owner departs",
    },
    "LeftBag_AtChair": {
        "video": "LeftBag_AtChair.mpg",
        "gt": "lb2gt.xml",
        "path_video": "LeftBag_AtChair/LeftBag_AtChair.mpg",
        "path_gt": "LeftBag_AtChair/lb2gt.xml",
        "size_mb": 11.0,
        "scenario": "Abandoned bag beside waiting chairs",
        "expected_events": "abandoned_object beside waiting chairs",
    },
    "LeftBox": {
        "video": "LeftBox.mpg",
        "gt": "lbgt.xml",
        "path_video": "LeftBox/LeftBox.mpg",
        "path_gt": "LeftBox/lbgt.xml",
        "size_mb": 15.0,
        "scenario": "Abandoned cardboard box",
        "expected_events": "abandoned_object package left unattended",
    },
    "Fight_RunAway1": {
        "video": "Fight_RunAway1.mpg",
        "gt": "fra1gt.xml",
        "path_video": "Fight_RunAway1/Fight_RunAway1.mpg",
        "path_gt": "Fight_RunAway1/fra1gt.xml",
        "size_mb": 6.0,
        "scenario": "Two people fighting and running away",
        "expected_events": "altercation and fighting followed by rapid run_away",
    },
}

DEFAULT_CLIPS: List[str] = [
    "Walk1",
    "Walk2",
    "Walk3",
    "LeftBag",
    "LeftBag_AtChair",
    "LeftBox",
    "Fight_RunAway1",
]


def candidate_urls(relative_path: str, filename: str) -> List[str]:
    """Generate candidate URLs supporting subfolder, root, https, and http."""
    urls = []
    for base in BASE_URLS:
        urls.append(f"{base}/{relative_path}")
        urls.append(f"{base}/{filename}")
    return urls


def download_file(candidate_list: List[str], dest_path: Path, dry_run: bool = False) -> bool:
    """Download a remote file trying candidates until one succeeds."""
    if dest_path.exists():
        logger.info("File already exists, skipping: %s", dest_path.name)
        return True

    if dry_run:
        logger.info("[DRY-RUN] Would download: %s -> %s", candidate_list[0], dest_path)
        return True

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"}

    last_error: Exception | None = None
    for url in candidate_list:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as response, open(temp_path, "wb") as out_f:
                total_size = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                block_size = 64 * 1024

                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    out_f.write(buffer)
                    if total_size > 0:
                        percent = downloaded * 100 / total_size
                        print(
                            f"\r  Downloading {dest_path.name}: {downloaded / (1024 * 1024):.2f}/{total_size / (1024 * 1024):.2f} MB ({percent:.1f}%)",
                            end="",
                            flush=True,
                        )
                print()

            temp_path.replace(dest_path)
            logger.info("Saved: %s (%.2f MB from %s)", dest_path.name, dest_path.stat().st_size / (1024 * 1024), url)
            return True
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if temp_path.exists():
                temp_path.unlink()
            continue

    logger.error("Failed to download %s from any candidate URL. Last error: %s", dest_path.name, last_error)
    return False


def list_catalog() -> None:
    """Print tabular inventory of all catalogued clips."""
    print("=" * 80)
    print(f"{'Clip Key':<18} | {'Video File':<20} | {'GT XML':<12} | {'Est. Size':<10} | {'Scenario'}")
    print("-" * 80)
    total_mb = 0.0
    for key, info in CAVIAR_CATALOG.items():
        mb = float(info["size_mb"])
        total_mb += mb
        print(f"{key:<18} | {str(info['video']):<20} | {str(info['gt']):<12} | {mb:>6.1f} MB | {info['scenario']}")
    print("-" * 80)
    print(f"Total catalogued clips: {len(CAVIAR_CATALOG)} (~{total_mb:.1f} MB)")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download CAVIAR surveillance clips and ground-truth XML files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--clips",
        nargs="+",
        default=DEFAULT_CLIPS,
        help="Space-separated list of clip keys to download (or 'all').",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available clips with sizes and exit without downloading.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/datasets/caviar/raw"),
        help="Target folder for raw MPEG video clips.",
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=Path("data/datasets/caviar/ground_truth"),
        help="Target folder for XML ground truth files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print download targets and size estimates without downloading.",
    )

    args = parser.parse_args()

    if args.list:
        list_catalog()
        return

    chosen_keys = args.clips
    if "all" in chosen_keys:
        chosen_keys = list(CAVIAR_CATALOG.keys())

    unknown = [k for k in chosen_keys if k not in CAVIAR_CATALOG]
    if unknown:
        logger.error("Unknown clip key(s): %s. Use --list to see available keys.", unknown)
        sys.exit(1)

    print("=" * 80)
    print("                      CAVIAR DATASET DOWNLOAD PLAN")
    print("=" * 80)
    total_est_mb = sum(float(CAVIAR_CATALOG[k]["size_mb"]) for k in chosen_keys)
    print(f"Target Raw Directory : {args.raw_dir.resolve()}")
    print(f"Target GT Directory  : {args.gt_dir.resolve()}")
    print(f"Selected Clips Count : {len(chosen_keys)}")
    print(f"Estimated Total Size : ~{total_est_mb:.1f} MB")
    print(f"Mode                 : {'DRY RUN (simulation)' if args.dry_run else 'LIVE DOWNLOAD'}")
    print("-" * 80)

    for idx, key in enumerate(chosen_keys, start=1):
        info = CAVIAR_CATALOG[key]
        video_cands = candidate_urls(str(info["path_video"]), str(info["video"]))
        gt_cands = candidate_urls(str(info["path_gt"]), str(info["gt"]))
        video_dest = args.raw_dir / str(info["video"])
        gt_dest = args.gt_dir / str(info["gt"])

        print(f"[{idx}/{len(chosen_keys)}] {key}:")
        print(f"  Video: {video_dest.name} (~{info['size_mb']} MB) <- {video_cands[0]}")
        print(f"  GT   : {gt_dest.name} <- {gt_cands[0]}")

        download_file(video_cands, video_dest, dry_run=args.dry_run)
        download_file(gt_cands, gt_dest, dry_run=args.dry_run)
        print()

    print("=" * 80)
    if args.dry_run:
        print("[DRY-RUN] Simulation finished. No files were written to disk.")
    else:
        print("[SUCCESS] All selected CAVIAR files processed.")
    print("=" * 80)


if __name__ == "__main__":
    main()
