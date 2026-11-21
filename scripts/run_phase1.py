"""Phase 1 Pipeline CLI Runner.

Executes the detection and tracking pipeline over a video file, webcam, or RTSP stream,
saves annotated video to data/output/, and dumps detailed performance metrics to JSON.

Usage:
    python scripts/run_phase1.py --source data/samples/sample_walking_01.mp4 --camera-id cam_walk_01 --model yolo26s.pt --show
    python scripts/run_phase1.py --source 0 --camera-id cam_webcam --model yolo26s.pt --show
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.phase1_pipeline import Phase1Pipeline

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_phase1")


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute Phase 1 Detection & ByteTrack Pipeline.")
    parser.add_argument(
        "--source",
        type=str,
        default="data/samples/sample_walking_01.mp4",
        help="Input video file path, webcam index (e.g. '0'), or rtsp:// URL.",
    )
    parser.add_argument(
        "--camera-id",
        type=str,
        default="cam_walk_01",
        help="Unique camera identifier.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo26s.pt",
        help="YOLO model weights (default: yolo26s.pt; fallback: yolo11s.pt).",
    )
    parser.add_argument(
        "--tracker-config",
        type=str,
        default="configs/bytetrack.yaml",
        help="ByteTrack configuration YAML path (default: configs/bytetrack.yaml).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference frame resolution dimension (default: 640).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold (default: 0.25).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.70,
        help="NMS IoU threshold (default: 0.70).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Target PyTorch device (default: 'cuda:0').",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open live OpenCV display window.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable video recording to disk.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to process (useful for quick evaluation).",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=0,
        help="Process every (N+1)-th frame (default: 0 = process all frames).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/output",
        help="Destination directory for output video and JSON metrics.",
    )

    args = parser.parse_args()

    pipeline = Phase1Pipeline(
        camera_id=args.camera_id,
        source=args.source,
        model_name=args.model,
        tracker_config=args.tracker_config,
        device=args.device,
        half_precision=True,
        imgsz=args.imgsz,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        frame_skip=args.frame_skip,
        output_dir=args.output_dir,
        save_video=not args.no_save,
        show_window=args.show,
    )

    summary = pipeline.run(max_frames=args.max_frames)

    # Save summary JSON
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_model = Path(args.model).stem
    json_path = out_dir / f"{args.camera_id}_{clean_model}_bytetrack_metrics.json"

    # Enrich summary with run parameters
    summary["camera_id"] = args.camera_id
    summary["source"] = str(args.source)
    summary["model_name"] = args.model
    summary["tracker_config"] = args.tracker_config
    summary["imgsz"] = args.imgsz
    summary["conf_threshold"] = args.conf
    summary["output_video"] = str(pipeline.output_video_path) if pipeline.output_video_path else None

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Print formatted benchmark table to console
    print("\n" + "=" * 78)
    print(f"             PHASE 1 EXECUTION REPORT ({args.camera_id})")
    print("=" * 78)
    print(f"Model Architecture    : {args.model}")
    print(f"Tracker Algorithm     : ByteTrack ({args.tracker_config})")
    print(f"Inference Image Size  : {args.imgsz} px (FP16)")
    print(f"Frames Processed      : {summary['total_frames']} frames in {summary['elapsed_seconds']} sec")
    print(f"Overall Throughput    : {summary['overall_throughput_fps']} FPS (I/O, tracking, drawing, write)")
    print(f"Model + Track Rate    : {summary['inference_only_fps']} FPS (pure inference latency)")
    print("-" * 78)
    print("Latency Breakdown (ms):")
    for stg, lat in summary["stage_latencies"].items():
        print(f"  {stg:<12} -> Mean: {lat['mean_ms']:>6.2f} ms | p50: {lat['p50_ms']:>6.2f} ms | p95: {lat['p95_ms']:>6.2f} ms")
    print("-" * 78)
    vram = summary["vram_metrics"]
    print(f"VRAM Allocated        : {vram['allocated_mb']} MB")
    print(f"VRAM Reserved         : {vram['reserved_mb']} MB (Total Hardware: {vram['total_device_gb']} GB)")
    print(f"Average People Count  : {summary['average_people_count']}")
    print(f"Max People Detected   : {summary['max_people_count']}")
    print(f"Max Bags Detected     : {summary['max_bag_count']}")
    if pipeline.output_video_path:
        print(f"Annotated Video Saved : {pipeline.output_video_path.resolve()}")
    print(f"Metrics JSON Saved    : {json_path.resolve()}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
