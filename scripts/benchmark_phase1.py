"""Phase 1 Benchmark Matrix Script.

Automates benchmarking across model variants (yolo26s, yolo26n, yolo11s),
input dimensions (640 vs 960 px), and confidence thresholds (0.25 vs 0.40).
Generates structured Markdown and CSV comparison tables in data/output/benchmarks/.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.phase1_pipeline import Phase1Pipeline

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark_phase1")


def run_benchmark_matrix(
    clips: List[Path],
    models: List[str],
    img_sizes: List[int],
    confs: List[float],
    max_frames_per_run: int = 300,
    output_dir: Path = Path("data/output/benchmarks"),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "phase1_benchmark_results.csv"
    md_path = output_dir / "phase1_benchmark_results.md"

    results: List[Dict[str, Any]] = []

    print("=" * 85)
    print("                 PHASE 1 DETECTION & TRACKING BENCHMARK MATRIX")
    print("=" * 85)
    print(f"Clips to benchmark: {[c.name for c in clips]}")
    print(f"Models: {models}")
    print(f"Resolutions: {img_sizes}")
    print(f"Confidence Thresholds: {confs}")
    print(f"Max evaluation frames per condition: {max_frames_per_run}")
    print("-" * 85)

    total_runs = len(clips) * len(models) * len(img_sizes) * len(confs)
    run_idx = 1

    for clip in clips:
        camera_id = f"cam_{clip.stem.replace('sample_', '')}"
        for model_name in models:
            for imgsz in img_sizes:
                for conf in confs:
                    print(f"\n[{run_idx}/{total_runs}] Running: {clip.name} | Model: {model_name} | {imgsz}px | Conf: {conf}...")
                    run_idx += 1

                    pipeline = Phase1Pipeline(
                        camera_id=camera_id,
                        source=str(clip),
                        model_name=model_name,
                        tracker_config="configs/bytetrack.yaml",
                        device="cuda:0",
                        half_precision=True,
                        imgsz=imgsz,
                        conf_threshold=conf,
                        iou_threshold=0.70,
                        frame_skip=0,
                        output_dir=output_dir / "videos",
                        save_video=False,  # Don't spend disk I/O during heavy matrix sweep
                        show_window=False,
                    )

                    try:
                        summary = pipeline.run(max_frames=max_frames_per_run)
                        row = {
                            "clip": clip.name,
                            "model": model_name,
                            "imgsz": imgsz,
                            "conf": conf,
                            "throughput_fps": summary["overall_throughput_fps"],
                            "inference_fps": summary["inference_only_fps"],
                            "detect_ms": summary["stage_latencies"]["detect_ms"]["mean_ms"],
                            "track_ms": summary["stage_latencies"]["track_ms"]["mean_ms"],
                            "total_latency_ms": summary["stage_latencies"]["total_ms"]["mean_ms"],
                            "vram_alloc_mb": summary["vram_metrics"]["allocated_mb"],
                            "vram_res_mb": summary["vram_metrics"]["reserved_mb"],
                            "max_people": summary["max_people_count"],
                            "max_bags": summary["max_bag_count"],
                        }
                        results.append(row)
                    except Exception as exc:
                        logger.error("Failed benchmark run for %s (%s): %s", clip.name, model_name, exc)

    # 1. Write CSV
    if results:
        fieldnames = list(results[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n[SAVED] Benchmark CSV written to: {csv_path.resolve()}")

    # 2. Write Markdown Table
    if results:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Phase 1 Performance Benchmark Results\n\n")
            f.write("| Clip | Model | ImgSz | Conf | Inference FPS | Throughput FPS | Latency (ms) | VRAM Alloc | VRAM Res | Max People | Max Bags |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
            for r in results:
                f.write(
                    f"| {r['clip']} | {r['model']} | {r['imgsz']} | {r['conf']} | "
                    f"**{r['inference_fps']}** | {r['throughput_fps']} | {r['total_latency_ms']} ms | "
                    f"{r['vram_alloc_mb']} MB | {r['vram_res_mb']} MB | {r['max_people']} | {r['max_bags']} |\n"
                )
        print(f"[SAVED] Benchmark Markdown report written to: {md_path.resolve()}")

    print("=" * 85)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 1 Benchmark Matrix.")
    parser.add_argument(
        "--clips",
        nargs="+",
        default=[
            "data/samples/sample_walking_01.mp4",
            "data/samples/sample_abandoned_bag_01.mp4",
        ],
        help="List of clip paths to benchmark.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["yolo26s.pt", "yolo11s.pt"],
        help="List of model weights to compare.",
    )
    parser.add_argument(
        "--img-sizes",
        nargs="+",
        type=int,
        default=[640, 960],
        help="Inference resolutions to test.",
    )
    parser.add_argument(
        "--confs",
        nargs="+",
        type=float,
        default=[0.25, 0.40],
        help="Confidence thresholds to test.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=250,
        help="Frames per condition to evaluate.",
    )

    args = parser.parse_args()
    clip_paths = [Path(p) for p in args.clips if Path(p).exists()]
    if not clip_paths:
        logger.error("No valid clip paths found to benchmark.")
        sys.exit(1)

    run_benchmark_matrix(
        clips=clip_paths,
        models=args.models,
        img_sizes=args.img_sizes,
        confs=args.confs,
        max_frames_per_run=args.max_frames,
    )


if __name__ == "__main__":
    main()
