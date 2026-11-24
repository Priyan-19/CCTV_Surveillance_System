"""Phase 2 Performance Re-Baseline Script.

Executes Phase 2 detection, tracking, and rules pipeline across all sample clips
in both headless and display modes to establish an unambiguous, verified baseline
before adding Phase 3 Face Recognition workloads.

Reports Throughput FPS, Inference FPS, Rules Latency, and Per-Frame Latency,
strictly marking metrics outside TRD targets (18-22 FPS, 20-45 ms) as BELOW TARGET.

Usage:
    python scripts/rebaseline_phase2.py
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Any, Dict, List
import sys
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.phase2_pipeline import Phase2Pipeline

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("rebaseline_phase2")

BENCHMARK_CONDITIONS = [
    {
        "clip": "data/samples/sample_walking_01.mp4",
        "camera_id": "cam_walk_01",
        "imgsz": 640,
        "max_frames": 400,
        "desc": "Single walker (CAVIAR 384x288)",
    },
    {
        "clip": "data/samples/sample_abandoned_bag_hd.mp4",
        "camera_id": "cam_abandoned_hd",
        "imgsz": 960,
        "max_frames": 500,
        "desc": "Unattended bag HD (ABODA 640x480, 960px inference)",
    },
    {
        "clip": "data/samples/sample_fight_01.mp4",
        "camera_id": "cam_fight_01",
        "imgsz": 640,
        "max_frames": 400,
        "desc": "Two-person altercation (CAVIAR 384x288)",
    },
    {
        "clip": "data/samples/sample_normal_01.mp4",
        "camera_id": "cam_walk_01",
        "imgsz": 640,
        "max_frames": 400,
        "desc": "Normal pedestrians (CAVIAR 384x288)",
    },
]


def run_rebaseline(output_dir: Path = Path("data/output/benchmarks")) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "phase2_rebaseline.md"
    csv_path = output_dir / "phase2_rebaseline.csv"

    results: List[Dict[str, Any]] = []

    print("=" * 105)
    print("                    PHASE 2 PRE-PHASE 3 PERFORMANCE RE-BASELINE")
    print("=" * 105)
    print(f"{'Clip / Condition':<36} | {'FPS (All)':<10} | {'FPS (Inf)':<10} | {'Total ms':<10} | {'Rules ms':<10} | {'Status':<12}")
    print("-" * 105)

    for cond in BENCHMARK_CONDITIONS:
        clip_path = Path(cond["clip"])
        if not clip_path.exists():
            logger.warning("Clip %s does not exist, skipping.", clip_path)
            continue

        pipeline = Phase2Pipeline(
            camera_id=cond["camera_id"],
            source=str(clip_path),
            model_name="yolo26s.pt",
            tracker_config="configs/bytetrack.yaml",
            rules_config="configs/rules.yaml",
            zones_config="configs/zones.yaml",
            device="cuda:0",
            half_precision=True,
            imgsz=cond["imgsz"],
            conf_threshold=0.25,
            save_video=False,
            show_window=False,
        )

        summary = pipeline.run(max_frames=cond["max_frames"])
        fps_thru = summary["overall_throughput_fps"]
        fps_inf = summary["inference_only_fps"]
        total_lat = summary["stage_latencies"]["total_ms"]["mean_ms"]
        rules_lat = summary["stage_latencies"].get("rules_ms", {}).get("mean_ms", 0.0)

        # TRD criteria: Throughput >= 18.0 FPS, Total Latency <= 45.0 ms
        is_fps_pass = fps_thru >= 18.0
        is_lat_pass = total_lat <= 45.0

        if is_fps_pass and is_lat_pass:
            status = "PASS"
        elif not is_fps_pass and not is_lat_pass:
            status = "BELOW TARGET (FPS & Latency)"
        elif not is_fps_pass:
            status = "BELOW TARGET (FPS)"
        else:
            status = "BELOW TARGET (Latency)"

        row = {
            "condition": cond["desc"],
            "clip_file": clip_path.name,
            "imgsz": cond["imgsz"],
            "frames_evaluated": summary["total_frames"],
            "throughput_fps": fps_thru,
            "inference_fps": fps_inf,
            "mean_total_latency_ms": total_lat,
            "mean_rules_latency_ms": rules_lat,
            "status": status,
        }
        results.append(row)

        print(
            f"{cond['desc'][:36]:<36} | {fps_thru:>7.2f} FPS | {fps_inf:>7.2f} FPS | {total_lat:>7.2f} ms | {rules_lat:>7.2f} ms | {status:<12}"
        )

    print("-" * 105)

    # Write CSV
    if results:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    # Write Markdown
    if results:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Phase 2 Pre-Phase 3 Performance Re-Baseline Report\n\n")
            f.write("| Condition | ImgSz | Frames | Throughput FPS | Inference FPS | Total Latency (ms) | Rules Latency (ms) | Status |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
            for r in results:
                f.write(
                    f"| {r['condition']} | {r['imgsz']}px | {r['frames_evaluated']} | {r['throughput_fps']:.2f} | "
                    f"{r['inference_fps']:.2f} | {r['mean_total_latency_ms']:.2f} | {r['mean_rules_latency_ms']:.2f} | **{r['status']}** |\n"
                )

    print(f"[SAVED] Re-baseline Markdown report : {md_path.resolve()}")
    print(f"[SAVED] Re-baseline CSV report      : {csv_path.resolve()}")
    print("=" * 105)


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-baseline Phase 2 performance across clips.")
    parser.add_argument("--output-dir", type=str, default="data/output/benchmarks")
    args = parser.parse_args()

    run_rebaseline(output_dir=Path(args.output_dir))


if __name__ == "__main__":
    main()
