"""Video Converter to Standard MP4 (H.264).

Converts raw surveillance video clips (.mpg, .avi) to standard web/OpenCV compatible
MP4 files using ffmpeg with H.264 video codec.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("convert_to_mp4")


def check_ffmpeg() -> Optional[str]:
    """Check if ffmpeg executable exists in system PATH."""
    return shutil.which("ffmpeg")


def convert_video(
    ffmpeg_bin: str,
    input_path: Path,
    output_path: Path,
    overwrite: bool = False,
    dry_run: bool = False,
) -> bool:
    """Convert a single video file to MP4 format using ffmpeg."""
    if output_path.exists() and not overwrite:
        logger.info("Target already exists, skipping: %s", output_path.name)
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd: List[str] = [
        ffmpeg_bin,
        "-y" if overwrite else "-n",
        "-i",
        str(input_path),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "22",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",  # Strip audio track for clean silent surveillance evaluation
        str(output_path),
    ]

    cmd_str = " ".join(f'"{c}"' if " " in c else c for c in cmd)
    if dry_run:
        logger.info("[DRY-RUN] Command: %s", cmd_str)
        return True

    logger.info("Converting: %s -> %s", input_path.name, output_path.name)
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            logger.error("ffmpeg failed on %s (code %d):\n%s", input_path.name, proc.returncode, proc.stderr[-500:])
            return False

        in_size_mb = input_path.stat().st_size / (1024 * 1024)
        out_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info("Converted successfully: %.2f MB -> %.2f MB", in_size_mb, out_size_mb)
        return True
    except Exception as exc:
        logger.error("Exception executing ffmpeg: %s", exc)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert raw surveillance clips (.mpg, .avi) to MP4 (H.264) using ffmpeg.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/datasets/caviar/raw"),
        help="Directory containing source videos to convert.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/datasets/caviar/mp4"),
        help="Directory where converted MP4 videos will be saved.",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Convert a specific single file instead of whole directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing converted MP4 files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print conversion commands without executing.",
    )

    args = parser.parse_args()

    ffmpeg_bin = check_ffmpeg()
    if not ffmpeg_bin:
        print("=" * 80)
        print("[ERROR] 'ffmpeg' is not found in your system PATH.")
        print("=" * 80)
        print("To install ffmpeg on Windows 11 via PowerShell, run:")
        print("    winget install Gyan.FFmpeg")
        print("Or download the release build directly from:")
        print("    https://www.gyan.dev/ffmpeg/builds/")
        print("Then restart your PowerShell terminal session.")
        print("=" * 80)
        sys.exit(1)

    logger.info("Found ffmpeg binary at: %s", ffmpeg_bin)

    files_to_convert: List[Path] = []
    if args.file:
        if not args.file.exists():
            logger.error("File not found: %s", args.file)
            sys.exit(1)
        files_to_convert.append(args.file)
    else:
        if not args.input_dir.exists():
            logger.error("Input directory does not exist: %s", args.input_dir)
            sys.exit(1)
        valid_exts = {".mpg", ".mpeg", ".avi", ".mov", ".mkv"}
        files_to_convert = [f for f in sorted(args.input_dir.iterdir()) if f.is_file() and f.suffix.lower() in valid_exts]

    if not files_to_convert:
        logger.warning("No convertible video files found in %s.", args.input_dir)
        return

    print("=" * 80)
    print("                      VIDEO MP4 CONVERSION PLAN")
    print("=" * 80)
    print(f"Total videos to convert: {len(files_to_convert)}")
    print(f"Input directory        : {args.input_dir.resolve()}")
    print(f"Output directory       : {args.output_dir.resolve()}")
    print(f"Mode                   : {'DRY RUN (simulation)' if args.dry_run else 'LIVE CONVERSION'}")
    print("-" * 80)

    success_count = 0
    for idx, in_file in enumerate(files_to_convert, start=1):
        out_file = args.output_dir / f"{in_file.stem}.mp4"
        print(f"[{idx}/{len(files_to_convert)}] {in_file.name} -> {out_file.name}")
        ok = convert_video(
            ffmpeg_bin=ffmpeg_bin,
            input_path=in_file,
            output_path=out_file,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        if ok:
            success_count += 1

    print("=" * 80)
    print(f"Completed: {success_count}/{len(files_to_convert)} files processed.")
    print("=" * 80)


if __name__ == "__main__":
    main()
