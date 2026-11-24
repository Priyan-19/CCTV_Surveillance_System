"""Phase 2 CLI Runner.

Executes detection, tracking, spatial & temporal rules, and event logging over video feeds.
Outputs annotated video, JSONL events, and performance summaries.

Usage:
    python scripts/run_phase2.py --source data/samples/sample_walking_01.mp4 --camera-id cam_walk_01 --show
    python scripts/run_phase2.py --source data/samples/sample_abandoned_bag_hd.mp4 --camera-id cam_abandoned_hd --imgsz 960 --show
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.phase2_pipeline import Phase2Pipeline

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_phase2")


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute Phase 2 Rules & Event Pipeline.")
    parser.add_argument(
        "--source",
        type=str,
        default="data/samples/sample_walking_01.mp4",
        help="Video file path, webcam index, or RTSP URL.",
    )
    parser.add_argument(
        "--camera-id",
        type=str,
        default="cam_walk_01",
        help="Camera identifier matching configs/zones.yaml.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo26s.pt",
        help="YOLO model weights (default: yolo26s.pt).",
    )
    parser.add_argument(
        "--tracker-config",
        type=str,
        default="configs/bytetrack.yaml",
        help="ByteTrack config YAML.",
    )
    parser.add_argument(
        "--rules-config",
        type=str,
        default="configs/rules.yaml",
        help="Rule thresholds config YAML.",
    )
    parser.add_argument(
        "--zones-config",
        type=str,
        default="configs/zones.yaml",
        help="Zones and tripwires config YAML.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference frame resolution dimension (default: 640; use 960 for small bags).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open live OpenCV preview window.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable saving annotated video.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit frames to process.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/output",
        help="Destination directory for video and metrics.",
    )
    parser.add_argument(
        "--events-dir",
        type=str,
        default="data/output/events",
        help="Directory to write JSONL event stream.",
    )

    args = parser.parse_args()

    pipeline = Phase2Pipeline(
        camera_id=args.camera_id,
        source=args.source,
        model_name=args.model,
        tracker_config=args.tracker_config,
        rules_config=args.rules_config,
        zones_config=args.zones_config,
        device="cuda:0",
        half_precision=True,
        imgsz=args.imgsz,
        conf_threshold=args.conf,
        output_dir=args.output_dir,
        events_dir=args.events_dir,
        save_video=not args.no_save,
        show_window=args.show,
    )

    summary = pipeline.run(max_frames=args.max_frames)

    # Print Report
    print("\n" + "=" * 80)
    print(f"             PHASE 2 RULES EXECUTION SUMMARY: {args.camera_id}")
    print("=" * 80)
    print(f"Frames Processed       : {summary['total_frames']} frames in {summary['elapsed_seconds']} sec")
    print(f"Throughput Frame Rate  : {summary['overall_throughput_fps']} FPS (all stages)")
    print(f"Inference Frame Rate   : {summary['inference_only_fps']} FPS")
    print("-" * 80)
    print("Stage Latencies (Mean):")
    for stg, lat in summary["stage_latencies"].items():
        print(f"  {stg:<12} -> Mean: {lat['mean_ms']:>6.2f} ms | p50: {lat['p50_ms']:>6.2f} ms | p95: {lat['p95_ms']:>6.2f} ms")
    print("-" * 80)
    print(f"Security Events Fired  : {len(pipeline.logged_events)}")
    for idx, evt in enumerate(pipeline.logged_events, 1):
        print(f"  {idx}. [{evt.severity.value.upper()}] {evt.event_type.value} at t={evt.timestamp:.2f}s (Track ID: {evt.track_id}) -> {evt.details}")
    print("-" * 80)
    if pipeline.output_video_path:
        print(f"Annotated Video Output : {pipeline.output_video_path.resolve()}")
    if pipeline.events_file_path:
        print(f"JSONL Event Log Saved  : {pipeline.events_file_path.resolve()}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
