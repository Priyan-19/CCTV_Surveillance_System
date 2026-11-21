"""Interactive Zone & Tripwire Drawing Utility.

Opens a video frame or image with OpenCV so you can click vertices to define
spatial polygons (restricted zones, occupancy areas) or virtual tripwire lines,
and automatically appends or updates configs/zones.yaml.

Interactive Controls:
  - Left Mouse Button : Add vertex / tripwire point
  - Right Mouse Button: Undo last point
  - 'c' Key           : Close polygon
  - 'r' Key           : Reset / clear all points
  - 's' Key           : Confirm and save to configs/zones.yaml
  - 'q' or ESC Key    : Cancel and exit without saving
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("draw_zone")


def parse_points_arg(points_str: str) -> List[List[int]]:
    """Parse string formatted points: 'x1,y1;x2,y2;x3,y3'."""
    pts: List[List[int]] = []
    pairs = points_str.strip().split(";")
    for pair in pairs:
        if not pair.strip():
            continue
        parts = pair.strip().split(",")
        if len(parts) != 2:
            raise ValueError(f"Invalid point format '{pair}', expected 'x,y'")
        pts.append([int(parts[0].strip()), int(parts[1].strip())])
    return pts


def interactive_point_collector(
    image: cv2.Mat,
    title: str,
    is_line: bool = False,
) -> List[List[int]]:
    """Display image and collect clicked points interactively via OpenCV mouse callback."""
    points: List[List[int]] = []
    closed = False

    def mouse_callback(event: int, x: int, y: int, flags: int, param: Any) -> None:
        nonlocal points, closed
        if event == cv2.EVENT_LBUTTONDOWN:
            if is_line and len(points) >= 2:
                logger.info("Tripwire line already has 2 points. Press 'r' to reset or 's' to save.")
                return
            points.append([x, y])
            closed = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            if points:
                points.pop()
                closed = False

    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(title, mouse_callback)

    print("\n--- DRAWING INSTRUCTIONS ---")
    print("  Left Click : Add point")
    print("  Right Click: Undo last point")
    print("  'c' key    : Close polygon")
    print("  'r' key    : Reset points")
    print("  's' key    : Save zone and exit")
    print("  'q' / ESC  : Cancel and exit")
    print("----------------------------\n")

    while True:
        display = image.copy()

        # Draw existing points and lines
        if len(points) > 0:
            for idx, pt in enumerate(points):
                cv2.circle(display, (pt[0], pt[1]), 5, (0, 0, 255), -1)
                cv2.putText(
                    display,
                    f"P{idx+1} ({pt[0]},{pt[1]})",
                    (pt[0] + 8, pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

            for i in range(len(points) - 1):
                cv2.line(
                    display,
                    (points[i][0], points[i][1]),
                    (points[i + 1][0], points[i + 1][1]),
                    (0, 255, 0),
                    2,
                )

            if closed and len(points) >= 3:
                cv2.line(
                    display,
                    (points[-1][0], points[-1][1]),
                    (points[0][0], points[0][1]),
                    (0, 255, 0),
                    2,
                )

        # Instructions overlay
        status_text = f"Points: {len(points)} | {'LINE' if is_line else 'POLYGON'} | [s] Save  [c] Close  [r] Reset  [q] Quit"
        cv2.putText(
            display,
            status_text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(title, display)
        key = cv2.waitKey(20) & 0xFF

        if key in (ord("q"), 27):  # 'q' or ESC
            logger.info("Cancelled by user.")
            cv2.destroyAllWindows()
            sys.exit(0)
        elif key == ord("r"):
            points.clear()
            closed = False
            logger.info("Points reset.")
        elif key == ord("c"):
            if len(points) >= 3:
                closed = True
                logger.info("Polygon closed.")
            else:
                logger.warning("Need at least 3 points to close polygon.")
        elif key == ord("s"):
            if is_line:
                if len(points) != 2:
                    logger.warning("Tripwire line requires exactly 2 points (Start and End). Current: %d", len(points))
                    continue
                cv2.destroyAllWindows()
                return points
            else:
                if len(points) < 3:
                    logger.warning("Polygon requires at least 3 points. Current: %d", len(points))
                    continue
                cv2.destroyAllWindows()
                return points


def update_zones_config(
    config_path: Path,
    camera_id: str,
    item_type: str,
    entry: Dict[str, Any],
    dry_run: bool = False,
) -> None:
    """Read zones.yaml, insert or update the specified camera entry, and save."""
    data: Dict[str, Any] = {"cameras": {}}
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict) and "cameras" in loaded:
                    data = loaded
        except Exception as exc:
            logger.warning("Could not read %s (%s). Creating new config structure.", config_path, exc)

    if "cameras" not in data or data["cameras"] is None:
        data["cameras"] = {}

    if camera_id not in data["cameras"]:
        data["cameras"][camera_id] = {"zones": [], "lines": []}
    else:
        cam_dict = data["cameras"][camera_id]
        if "zones" not in cam_dict or cam_dict["zones"] is None:
            cam_dict["zones"] = []
        if "lines" not in cam_dict or cam_dict["lines"] is None:
            cam_dict["lines"] = []

    target_list_key = "lines" if item_type == "line" else "zones"
    target_list: List[Dict[str, Any]] = data["cameras"][camera_id][target_list_key]

    # Check if entry ID already exists -> update it
    existing_idx = -1
    for idx, existing in enumerate(target_list):
        if existing.get("id") == entry["id"]:
            existing_idx = idx
            break

    if existing_idx >= 0:
        logger.info("Updating existing %s '%s' for camera '%s'", item_type, entry["id"], camera_id)
        target_list[existing_idx] = entry
    else:
        logger.info("Adding new %s '%s' for camera '%s'", item_type, entry["id"], camera_id)
        target_list.append(entry)

    if dry_run:
        print("\n[DRY-RUN] Resulting Config for Camera '%s':" % camera_id)
        print(yaml.safe_dump({camera_id: data["cameras"][camera_id]}, sort_keys=False))
        return

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    logger.info("Saved updated zones config to: %s", config_path.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive tool to draw zones or tripwires and update configs/zones.yaml.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=Path, help="Video clip to read frame from.")
    group.add_argument("--image", type=Path, help="Image frame path.")

    parser.add_argument("--camera-id", type=str, required=True, help="Camera identifier (e.g. cam_caviar_walk1).")
    parser.add_argument("--id", type=str, required=True, help="Unique zone or line ID.")
    parser.add_argument("--name", type=str, default="Virtual Zone", help="Human-readable zone label.")
    parser.add_argument(
        "--type",
        choices=["restricted", "occupancy", "line"],
        default="restricted",
        help="Type of geometry.",
    )
    parser.add_argument(
        "--direction",
        choices=["inbound", "outbound", "bidirectional"],
        default="inbound",
        help="Crossing direction (for lines only).",
    )
    parser.add_argument(
        "--dwell-seconds",
        type=float,
        default=10.0,
        help="Dwell threshold in seconds for zone alerts.",
    )
    parser.add_argument(
        "--occupancy-limit",
        type=int,
        default=4,
        help="Maximum allowed person count (for occupancy type).",
    )
    parser.add_argument(
        "--risk-level",
        choices=["low", "medium", "high", "critical"],
        default=None,
        help="Risk priority level (defaults: high for restricted, medium for occupancy).",
    )
    parser.add_argument(
        "--points",
        type=str,
        default=None,
        help="Optional non-interactive points: 'x1,y1;x2,y2;x3,y3...'",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/zones.yaml"),
        help="Path to zones configuration YAML.",
    )
    parser.add_argument(
        "--frame-idx",
        type=int,
        default=0,
        help="Frame index if using --video.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview YAML output without saving to file.",
    )

    args = parser.parse_args()

    # 1. Load image
    frame: Optional[cv2.Mat] = None
    if args.image:
        if not args.image.exists():
            logger.error("Image file not found: %s", args.image)
            sys.exit(1)
        frame = cv2.imread(str(args.image))
    elif args.video:
        if not args.video.exists():
            logger.error("Video file not found: %s", args.video)
            sys.exit(1)
        cap = cv2.VideoCapture(str(args.video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            logger.error("Failed to read frame %d from %s", args.frame_idx, args.video)
            sys.exit(1)

    if frame is None:
        logger.error("Could not obtain frame for drawing.")
        sys.exit(1)

    h, w = frame.shape[:2]
    logger.info("Frame loaded: %dx%d px", w, h)

    is_line = args.type == "line"

    # 2. Collect points
    if args.points:
        points = parse_points_arg(args.points)
        logger.info("Using points provided via argument: %s", points)
    else:
        window_title = f"Draw Zone: {args.name} ({args.type}) - {w}x{h}"
        points = interactive_point_collector(frame, window_title, is_line=is_line)

    # 3. Assemble entry
    if is_line:
        if len(points) != 2:
            logger.error("Tripwire line requires exactly 2 points: [start, end]. Got: %d", len(points))
            sys.exit(1)
        entry: Dict[str, Any] = {
            "id": args.id,
            "name": args.name,
            "start": points[0],
            "end": points[1],
            "direction": args.direction,
        }
    else:
        risk = args.risk_level or ("high" if args.type == "restricted" else "medium")
        entry = {
            "id": args.id,
            "name": args.name,
            "type": args.type,
            "polygon": points,
            "dwell_threshold_seconds": args.dwell_seconds,
            "risk_level": risk,
        }
        if args.type == "occupancy":
            entry["occupancy_limit"] = args.occupancy_limit

    # 4. Save to config
    update_zones_config(
        config_path=args.config,
        camera_id=args.camera_id,
        item_type=args.type,
        entry=entry,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
