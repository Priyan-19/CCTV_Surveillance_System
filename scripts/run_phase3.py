"""Phase 3 Full Surveillance Pipeline Runner with Face Recognition.

Usage:
    # Run on pedestrian sample clip
    python scripts/run_phase3.py --source data/samples/sample_walking_01.mp4 --camera-id cam_walk_01 --show

    # Headless benchmark mode
    python scripts/run_phase3.py --source data/samples/sample_walking_01.mp4 --camera-id cam_walk_01
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.phase3_pipeline import Phase3Pipeline

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("run_phase3")


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute Phase 3 surveillance pipeline with face recognition.")
    parser.add_argument("--source", type=str, default="data/samples/sample_walking_01.mp4", help="Video source path.")
    parser.add_argument("--camera-id", type=str, default="cam_walk_01", help="Camera identifier.")
    parser.add_argument("--model", type=str, default="yolo26s.pt", help="YOLO model weights path.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution.")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Inference device.")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to process.")
    parser.add_argument("--show", action="store_true", default=False, help="Display video stream window.")
    parser.add_argument("--no-save-video", action="store_true", default=False, help="Skip saving annotated video.")
    parser.add_argument("--db-path", type=str, default="data/cctv.db", help="Path to SQLite database.")
    parser.add_argument("--face-config", type=str, default="configs/face.yaml", help="Path to face.yaml.")
    args = parser.parse_args()

    pipeline = Phase3Pipeline(
        camera_id=args.camera_id,
        source=args.source,
        model_name=args.model,
        imgsz=args.imgsz,
        conf_threshold=args.conf,
        device=args.device,
        save_video=not args.no_save_video,
        show_window=args.show,
        db_path=args.db_path,
        face_config=args.face_config,
    )

    print("=" * 80)
    print("                 PHASE 3 SURVEILLANCE PIPELINE EXECUTION")
    print("=" * 80)
    print(f"Source Feed       : {args.source}")
    print(f"Camera ID         : {args.camera_id}")
    print(f"Detector Model    : {args.model} ({args.imgsz}px, FP16)")
    print(f"Face Engine       : InsightFace (buffalo_s) on RTX 4050 GPU")
    print(f"Face Database     : {args.db_path}")
    print(f"Display Window    : {'ENABLED' if args.show else 'DISABLED (Headless)'}")
    print("-" * 80)

    summary = pipeline.run(max_frames=args.max_frames)

    print("\n" + "=" * 80)
    print("                      PHASE 3 PERFORMANCE SCORECARD")
    print("=" * 80)
    fps_all = summary.get("overall_throughput_fps", summary.get("fps_overall", 0.0))
    fps_inf = summary.get("inference_only_fps", summary.get("fps_inference_only", 0.0))
    stage_latencies = summary.get("stage_latencies", {})
    total_ms = stage_latencies.get("total_ms", {}).get("mean_ms", 0.0) if isinstance(stage_latencies.get("total_ms"), dict) else 0.0
    face_stats = summary.get("face_worker", {})

    print(f"Overall Pipeline Throughput : {fps_all:5.2f} FPS (Target: 18.0 - 22.0 FPS) -> [{'PASS' if fps_all >= 18.0 else 'BELOW TARGET'}]")
    print(f"Inference-Only Throughput   : {fps_inf:5.2f} FPS")
    print("-" * 80)
    print("Per-Stage Mean Latencies:")
    for stage, sstats in stage_latencies.items():
        if stage == "total_ms":
            continue
        mean_val = sstats.get("mean_ms", 0.0) if isinstance(sstats, dict) else sstats
        print(f"  - {stage:<18} : {mean_val:5.2f} ms")
    print(f"  * Total Mean Frame Latency : {total_ms:5.2f} ms (Target: 20.0 - 45.0 ms) -> [{'PASS' if 20.0 <= total_ms <= 45.0 else 'BELOW TARGET'}]")
    print("-" * 80)
    print("Asynchronous Face Worker Diagnostics:")
    print(f"  - Jobs Submitted           : {face_stats.get('submitted_count', 0)}")
    print(f"  - Jobs Processed           : {face_stats.get('processed_count', 0)}")
    print(f"  - Jobs Dropped (Queue Sat) : {face_stats.get('dropped_count', 0)} ({face_stats.get('drop_rate_pct', 0.0):.1f}%)")
    print(f"  - Faces Detected           : {face_stats.get('faces_detected', 0)}")
    print(f"  - Faces Matched to Known   : {face_stats.get('faces_matched', 0)}")
    print(f"  - Mean Neural Inference ms : {face_stats.get('mean_inference_ms', 0.0):.2f} ms/face")
    print("-" * 80)
    print(f"Security Events Logged      : {len(pipeline.logged_events)}")
    if pipeline.events_file_path:
        print(f"Events File Location        : {pipeline.events_file_path}")
    if pipeline.output_video_path:
        print(f"Annotated Video Location    : {pipeline.output_video_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
