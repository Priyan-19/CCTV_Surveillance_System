"""Interactive Zone and Tripwire Authoring Tool.

Allows visual calibration of zones (polygons) and tripwires (lines) directly
on a video frame using OpenCV mouse clicks.
Saves definitions to configs/zones.yaml with the reference_resolution automatically attached.

Usage:
    python scripts/draw_zone.py --source data/samples/sample_walking_01.mp4 --camera-id cam_walk_01
    python scripts/draw_zone.py --source data/samples/sample_abandoned_bag_hd.mp4 --camera-id cam_abandoned_hd
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import cv2
import yaml

# Global state for mouse callbacks
current_points: List[Tuple[int, int]] = []
drawn_zones: List[Dict[str, Any]] = []
drawn_lines: List[Dict[str, Any]] = []
mode = "polygon"  # 'polygon' or 'line'


def mouse_callback(event: int, x: int, y: int, flags: int, param: Any) -> None:
    global current_points
    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append((x, y))
        print(f"  + Added point #{len(current_points)} at ({x}, {y})")
    elif event == cv2.EVENT_RBUTTONDOWN:
        if current_points:
            removed = current_points.pop()
            print(f"  - Removed point at {removed}")


def main() -> None:
    global current_points, mode
    parser = argparse.ArgumentParser(description="Interactively calibrate zones and tripwires.")
    parser.add_argument("--source", type=str, default="data/samples/sample_walking_01.mp4", help="Video clip path.")
    parser.add_argument("--camera-id", type=str, default="cam_walk_01", help="Target camera ID.")
    parser.add_argument("--frame-idx", type=int, default=30, help="Frame index to draw upon.")
    parser.add_argument("--output-config", type=str, default="configs/zones.yaml", help="Path to zones YAML.")

    args = parser.parse_args()
    video_path = Path(args.source)
    if not video_path.exists():
        print(f"[ERROR] Video '{video_path}' not found.")
        sys.exit(1)

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print(f"[ERROR] Failed to extract frame #{args.frame_idx} from {video_path}")
        sys.exit(1)

    h, w = frame.shape[:2]
    window_name = f"Zone Calibrator - {args.camera_id} ({w}x{h})"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("=" * 75)
    print("                    ZONE & TRIPWIRE INTERACTIVE CALIBRATOR")
    print("=" * 75)
    print(f"Camera ID: {args.camera_id} | Reference Resolution: {w}x{h}")
    print("Controls:")
    print("  [Left Click]   : Add vertex point")
    print("  [Right Click]  : Undo last vertex point")
    print("  [Key 'c']      : Close Polygon & save zone properties")
    print("  [Key 'l']      : Switch to Tripwire Line mode (2 points: start, end)")
    print("  [Key 'p']      : Switch to Polygon mode")
    print("  [Key 's']      : Save all drawn zones/lines to configs/zones.yaml")
    print("  [Key 'q']      : Quit without saving")
    print("=" * 75)

    zones_to_save: List[Dict[str, Any]] = []
    lines_to_save: List[Dict[str, Any]] = []

    while True:
        display_frame = frame.copy()

        # Draw already saved zones
        for z in zones_to_save:
            pts = [(int(p[0]), int(p[1])) for p in z["polygon"]]
            poly_np = [tuple(p) for p in pts]
            cv2.polylines(display_frame, [poly_np], True, (0, 255, 0), 2)
            cv2.putText(display_frame, z["name"], pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw already saved lines
        for l in lines_to_save:
            p1 = (int(l["start"][0]), int(l["start"][1]))
            p2 = (int(l["end"][0]), int(l["end"][1]))
            cv2.line(display_frame, p1, p2, (0, 255, 255), 2)
            cv2.putText(display_frame, l["name"], p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Draw in-progress points
        for idx, pt in enumerate(current_points):
            cv2.circle(display_frame, pt, 4, (0, 0, 255), -1)
            if idx > 0:
                cv2.line(display_frame, current_points[idx - 1], pt, (0, 165, 255), 2)

        # HUD Instruction
        mode_text = f"MODE: {mode.upper()} | Points: {len(current_points)} | Press 'c' to close poly | 's' to save"
        cv2.putText(display_frame, mode_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            print("[INFO] Exited without saving.")
            break

        elif key == ord("p"):
            mode = "polygon"
            current_points.clear()
            print("Switched to POLYGON mode.")

        elif key == ord("l"):
            mode = "line"
            current_points.clear()
            print("Switched to LINE (Tripwire) mode.")

        elif key == ord("c"):
            if mode == "polygon" and len(current_points) >= 3:
                print("\n--- Enter Zone Properties ---")
                zid = input(f"Zone ID [zone_{len(zones_to_save)+1}]: ").strip() or f"zone_{len(zones_to_save)+1}"
                name = input("Zone Display Name: ").strip() or zid
                ztype = input("Zone Type (restricted / occupancy / watch / normal) [restricted]: ").strip() or "restricted"
                dwell = float(input("Dwell Threshold Seconds [30.0]: ").strip() or "30.0")
                limit = int(input("Occupancy Limit [4]: ").strip() or "4")
                risk = input("Risk Level (Low / Medium / High / Critical) [High]: ").strip() or "High"

                zone_dict = {
                    "id": zid,
                    "name": name,
                    "type": ztype,
                    "polygon": [[float(p[0]), float(p[1])] for p in current_points],
                    "reference_resolution": [w, h],
                    "dwell_threshold_seconds": dwell,
                    "occupancy_limit": limit,
                    "risk_level": risk,
                }
                zones_to_save.append(zone_dict)
                print(f"[OK] Zone '{name}' recorded.\n")
                current_points.clear()

            elif mode == "line" and len(current_points) >= 2:
                print("\n--- Enter Tripwire Line Properties ---")
                lid = input(f"Line ID [line_{len(lines_to_save)+1}]: ").strip() or f"line_{len(lines_to_save)+1}"
                name = input("Line Display Name: ").strip() or lid
                direction = input("Direction (both / a_to_b / b_to_a / inbound / outbound) [both]: ").strip() or "both"
                risk = input("Risk Level (Low / Medium / High / Critical) [Medium]: ").strip() or "Medium"

                line_dict = {
                    "id": lid,
                    "name": name,
                    "start": [float(current_points[0][0]), float(current_points[0][1])],
                    "end": [float(current_points[1][0]), float(current_points[1][1])],
                    "direction": direction,
                    "reference_resolution": [w, h],
                    "risk_level": risk,
                }
                lines_to_save.append(line_dict)
                print(f"[OK] Tripwire '{name}' recorded.\n")
                current_points.clear()

        elif key == ord("s"):
            cfg_file = Path(args.output_config)
            existing_data: Dict[str, Any] = {"cameras": {}}
            if cfg_file.exists():
                with open(cfg_file, "r", encoding="utf-8") as f:
                    existing_data = yaml.safe_load(f) or {"cameras": {}}

            if "cameras" not in existing_data:
                existing_data["cameras"] = {}

            if args.camera_id not in existing_data["cameras"]:
                existing_data["cameras"][args.camera_id] = {}

            existing_data["cameras"][args.camera_id]["zones"] = zones_to_save
            existing_data["cameras"][args.camera_id]["lines"] = lines_to_save

            with open(cfg_file, "w", encoding="utf-8") as f:
                yaml.dump(existing_data, f, default_flow_style=False, sort_keys=False)

            print(f"\n[SUCCESS] Saved {len(zones_to_save)} zones and {len(lines_to_save)} lines to {cfg_file.resolve()}!")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
