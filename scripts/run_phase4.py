"""Phase 4 Full Surveillance Pipeline Runner with Threat Engine & Evidence Collector.

Integrates detection, tracking, zone rules, face recognition, threat engine correlation,
evidence collection, and alert dispatching.

Usage:
    # Run on ABODA HD abandoned bag sample clip
    python scripts/run_phase4.py --source data/samples/sample_abandoned_bag_hd.mp4 --camera-id cam_bag_hd --imgsz 640

    # Run on walking pedestrian sample clip
    python scripts/run_phase4.py --source data/samples/sample_walking_01.mp4 --camera-id cam_walk_01 --imgsz 640

    # Display video stream window
    python scripts/run_phase4.py --source data/samples/sample_abandoned_bag_hd.mp4 --camera-id cam_bag_hd --show
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Dict

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.phase4_pipeline import Phase4Pipeline

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("run_phase4")


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute Phase 4 surveillance pipeline with threat engine.")
    parser.add_argument("--source", type=str, default="data/samples/sample_abandoned_bag_hd.mp4", help="Video source path.")
    parser.add_argument("--camera-id", type=str, default="cam_abandoned_hd", help="Camera identifier.")
    parser.add_argument("--model", type=str, default="yolo26s.pt", help="YOLO model weights path.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution.")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Inference device.")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to process.")
    parser.add_argument("--show", action="store_true", default=False, help="Display video stream window.")
    parser.add_argument("--no-save-video", action="store_true", default=False, help="Skip saving annotated video.")
    parser.add_argument("--db-path", type=str, default="data/cctv.db", help="Path to SQLite database.")
    parser.add_argument("--threat-config", type=str, default="configs/threat.yaml", help="Path to threat.yaml.")
    parser.add_argument("--alerts-config", type=str, default="configs/alerts.yaml", help="Path to alerts.yaml.")
    parser.add_argument("--evidence-config", type=str, default="configs/evidence.yaml", help="Path to evidence.yaml.")
    # Benchmark toggle flags (Problem 1b isolation benchmarks)
    parser.add_argument("--disable-face", action="store_true", default=False,
                        help="[Benchmark] Disable face recognition worker.")
    parser.add_argument("--disable-evidence", action="store_true", default=False,
                        help="[Benchmark] Disable evidence collection and frame buffering.")
    parser.add_argument("--disable-threat", action="store_true", default=False,
                        help="[Benchmark] Disable threat engine correlation and scoring.")
    parser.add_argument("--frame-stride", type=int, default=1,
                        help="[Benchmark] Run YOLO+ByteTrack every Nth frame (1=all frames, 2=every other, etc.).")
    args = parser.parse_args()

    pipeline = Phase4Pipeline(
        camera_id=args.camera_id,
        source=args.source,
        model_name=args.model,
        imgsz=args.imgsz,
        conf_threshold=args.conf,
        device=args.device,
        save_video=not args.no_save_video,
        show_window=args.show,
        db_path=args.db_path,
        threat_config=args.threat_config,
        alerts_config=args.alerts_config,
        evidence_config=args.evidence_config,
    )

    print("=" * 80)
    print("                 PHASE 4 SURVEILLANCE PIPELINE EXECUTION")
    print("=" * 80)
    print(f"Source Feed       : {args.source}")
    print(f"Camera ID         : {args.camera_id}")
    print(f"Detector Model    : {args.model} ({args.imgsz}px, FP16)")
    face_mode = "DISABLED" if args.disable_face else f"InsightFace (buffalo_s, det_size {pipeline.face_cfg.get('model', {}).get('det_size', [160, 160])}) on {pipeline.face_cfg.get('model', {}).get('providers', ['CPU'])[0]}"
    evidence_mode = "DISABLED" if args.disable_evidence else "Active (RAM Ring Buffer 30s / Max 250MB)"
    threat_mode = "DISABLED" if args.disable_threat else "Active (120s Correlation Window, 300s Quiet Period)"

    print(f"Face Engine       : {face_mode}")
    print(f"Threat Engine     : {threat_mode}")
    print(f"Evidence Capture  : {evidence_mode}")
    print(f"Frame Stride      : {args.frame_stride} (YOLO runs every {args.frame_stride} frame(s))")
    print(f"Surveillance DB   : {args.db_path}")
    print(f"Display Window    : {'ENABLED' if args.show else 'DISABLED (Headless)'}")
    print("-" * 80)

    summary = pipeline.run(
        max_frames=args.max_frames,
        disable_face=args.disable_face,
        disable_evidence=args.disable_evidence,
        disable_threat=args.disable_threat,
        frame_stride=args.frame_stride,
    )

    print("\n" + "=" * 80)
    print("                      PHASE 4 PERFORMANCE SCORECARD")
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

    # Threat Engine & Incident Summary
    events_count = len(pipeline.logged_events)
    incidents = list(pipeline.incident_manager.open_incidents.values()) + list(pipeline.incident_manager.closed_incidents.values())
    alerts_count = len(pipeline.dispatched_alerts)
    reduction_pct = ((events_count - alerts_count) / events_count * 100.0) if events_count > 0 else 0.0

    print("Threat Engine & Incident Summary:")
    print(f"  - Raw Security Events      : {events_count}")
    print(f"  - Correlated Incidents     : {len(incidents)}")
    print(f"  - Dispatched Alerts        : {alerts_count}")
    print(f"  - Alert Reduction Rate     : {reduction_pct:5.1f}% (vs 1-alert-per-event baseline)")
    print("-" * 80)

    if incidents:
        print("Correlated Incidents Detail:")
        for inc in incidents:
            print(f"  * {inc.incident_no} | {inc.threat_level.upper():<8} (Score: {inc.threat_score}/100) | Place: {inc.place}")
            print(f"    Activities    : {', '.join(inc.activities)}")
            print(f"    Evidence Items: {len(inc.evidence)}")
            for ev in inc.evidence:
                if ev.file_path:
                    print(f"      - [{ev.type}] {ev.file_path}")
            print(f"    Audit Trace   :")
            for line in inc.explanation:
                print(f"      - {line}")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
