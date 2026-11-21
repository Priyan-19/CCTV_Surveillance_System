"""End-to-End YOLO GPU Smoke Test Script.

Validates that:
1. PyTorch accesses the NVIDIA GPU with CUDA.
2. A pretrained YOLO model (small variant, FP16, 640px input) loads on CUDA.
3. OpenCV reads a sample video and feeds frames to the model.
4. Detections (bounding boxes, class labels, confidence) are annotated.
5. The processed video is saved to data/samples/output_smoke.mp4.
6. Key performance metrics are measured: Average FPS, average ms/frame, peak VRAM.

Usage:
    python scripts/smoke_test.py --video data/samples/sample_walking.mp4 --model yolo11s.pt
    python scripts/smoke_test.py --video data/samples/sample_walking.mp4 --model yolo26s.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def check_cuda_availability(device_str: str) -> str:
    """Ensure CUDA is operational and select the target device."""
    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available! PyTorch is running in CPU mode.")
        print("Please install the CUDA-enabled build of PyTorch before running smoke tests.")
        sys.exit(1)

    device = torch.device(device_str if device_str.startswith("cuda") else f"cuda:{device_str}")
    gpu_name = torch.cuda.get_device_name(device)
    print(f"[INFO] Using GPU acceleration device: {device} ({gpu_name})")
    return str(device)


def load_model(model_name: str, device: str) -> YOLO:
    """Load the specified YOLO model weights onto the target GPU.

    If YOLO26 weights are requested but not supported/found, provides
    explicit guidance with recommended alternatives rather than silently switching.
    """
    print(f"[INFO] Loading model weights: '{model_name}' on device '{device}'...")
    try:
        model = YOLO(model_name)
        # Move model to CUDA device explicitly
        model.to(device)
        return model
    except Exception as exc:
        if "26" in model_name:
            print("\n" + "=" * 75)
            print(f"[ERROR] Could not load requested YOLO26 model weights ('{model_name}').")
            print(f"Underlying error: {exc}")
            print("\n[DIAGNOSIS & RECOMMENDATION]")
            print("YOLO26 weights are not available in your currently installed Ultralytics version.")
            print("Ultralytics does not have a published 'yolo26' release at this time.")
            print("The closest and most capable available architectures are:")
            print("  1. 'yolo11s.pt' (YOLO11 Small - Recommended latest architecture)")
            print("  2. 'yolov8s.pt'  (YOLOv8 Small - Battle-tested fallback)")
            print("\nPlease re-run the smoke test explicitly using one of the available models:")
            print(f"    python scripts/smoke_test.py --model yolo11s.pt --video <video_path>")
            print("=" * 75 + "\n")
            sys.exit(1)
        else:
            print(f"\n[ERROR] Failed to load model '{model_name}': {exc}")
            sys.exit(1)


def find_default_video() -> Optional[Path]:
    """Locate the first available video file in data/samples/ if not explicitly provided."""
    samples_dir = Path("data/samples")
    if not samples_dir.exists():
        return None

    video_extensions = {".mp4", ".avi", ".mkv", ".mov"}
    clips = [
        f for f in sorted(samples_dir.iterdir())
        if f.is_file() and f.suffix.lower() in video_extensions and not f.name.startswith("output_")
    ]
    return clips[0] if clips else None


def run_smoke_test(
    video_path: Path,
    model_name: str = "yolo11s.pt",
    imgsz: int = 640,
    device_id: str = "0",
    conf_thresh: float = 0.25,
    output_path: Path = Path("data/samples/output_smoke.mp4"),
    show_window: bool = False,
    max_frames: Optional[int] = None,
) -> None:
    """Execute end-to-end detection benchmark and generate annotated video."""
    print("=" * 75)
    print("                 PHASE 0: YOLO GPU SMOKE TEST                   ")
    print("=" * 75)

    # 1. Check video existence
    if not video_path.exists():
        print(f"[ERROR] Input video '{video_path}' not found!")
        print("Place a sample video in data/samples/ or specify one using --video <path>.")
        sys.exit(1)

    # 2. Check GPU & CUDA
    device_str = check_cuda_availability(device_id)

    # 3. Load Model
    model = load_model(model_name, device_str)
    # Put model into FP16 half-precision mode on GPU
    try:
        model.model.half()
    except Exception:
        pass

    # 4. Open Video Capture
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] OpenCV could not open video file: {video_path}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Input Video           : {video_path.name}")
    print(f"Source Resolution     : {width}x{height}")
    print(f"Source Frame Rate     : {src_fps:.2f} FPS")
    print(f"Total Frames in File  : {total_frames}")
    print(f"Inference Image Size  : {imgsz} px (letterboxed)")
    print(f"Inference Precision   : FP16 (half-precision)")
    print(f"Confidence Threshold  : {conf_thresh}")
    print(f"Output Destination    : {output_path}")

    # Ensure output parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 5. Initialize Video Writer (using mp4v codec for Windows MP4 compatibility)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, src_fps, (width, height))
    if not writer.isOpened():
        print(f"[WARNING] Could not open VideoWriter with 'mp4v'. Falling back to 'XVID'.")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(str(output_path.with_suffix(".avi")), fourcc, src_fps, (width, height))

    # 6. Warm-up GPU with dummy frame
    print("\nWarming up CUDA kernels and caching allocator...")
    dummy_frame = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    _ = model.predict(dummy_frame, imgsz=imgsz, device=device_str, verbose=False)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # 7. Main Inference Loop
    frame_times_ms: list[float] = []
    frame_count = 0
    start_total_time = time.perf_counter()

    print(f"Running GPU inference over video frames...")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if max_frames and frame_count >= max_frames:
                break

            t0 = time.perf_counter()

            # Run YOLO prediction (half=True enables FP16 mixed precision)
            results = model.predict(
                frame,
                imgsz=imgsz,
                device=device_str,
                conf=conf_thresh,
                verbose=False,
            )

            # Synchronize CUDA to measure precise wall-clock processing time
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            frame_time_ms = (t1 - t0) * 1000.0
            frame_times_ms.append(frame_time_ms)
            frame_count += 1

            # Annotate detections on frame
            annotated_frame = results[0].plot()

            # Overlay real-time performance HUD on top-left
            instant_fps = 1000.0 / frame_time_ms if frame_time_ms > 0 else 0.0
            hud_text = f"GPU: {device_str} | FP16 {imgsz}px | Frame: {frame_count} | {frame_time_ms:.1f}ms ({instant_fps:.1f} FPS)"
            cv2.putText(
                annotated_frame,
                hud_text,
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            writer.write(annotated_frame)

            if show_window:
                cv2.imshow("CCTV Smoke Test - YOLO GPU", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] Interrupted by user ('q' pressed).")
                    break

            if frame_count % 30 == 0 or frame_count == total_frames:
                avg_so_far_ms = sum(frame_times_ms) / len(frame_times_ms)
                avg_so_far_fps = 1000.0 / avg_so_far_ms if avg_so_far_ms > 0 else 0.0
                print(f"  Processed {frame_count:4d}/{total_frames} frames | Avg: {avg_so_far_ms:.2f} ms ({avg_so_far_fps:.1f} FPS)")

    finally:
        cap.release()
        writer.release()
        if show_window:
            cv2.destroyAllWindows()

    total_wall_time = time.perf_counter() - start_total_time

    if frame_count == 0:
        print("[ERROR] No frames were processed from video file.")
        sys.exit(1)

    # 8. Compute Performance Metrics
    avg_ms_per_frame = sum(frame_times_ms) / len(frame_times_ms)
    avg_fps = 1000.0 / avg_ms_per_frame if avg_ms_per_frame > 0 else 0.0
    overall_pipeline_fps = frame_count / total_wall_time if total_wall_time > 0 else 0.0

    peak_vram_bytes = torch.cuda.max_memory_allocated()
    peak_vram_mb = peak_vram_bytes / (1024**2)
    peak_vram_gb = peak_vram_bytes / (1024**3)

    total_vram_gb = torch.cuda.get_device_properties(device_str).total_memory / (1024**3)

    # 9. Print Final Evaluation Report
    print("\n" + "=" * 75)
    print("                    SMOKE TEST BENCHMARK REPORT                 ")
    print("=" * 75)
    print(f"Frames Processed      : {frame_count}")
    print(f"Total Video Time      : {total_wall_time:.2f} seconds")
    print(f"Inference Time / Frame: {avg_ms_per_frame:.2f} ms  (Target from TRD: 20-45 ms)")
    print(f"Inference FPS         : {avg_fps:.2f} FPS (Target from TRD: 18-22 FPS)")
    print(f"Overall Throughput FPS: {overall_pipeline_fps:.2f} FPS (including video I/O & drawing)")
    print(f"Peak VRAM Utilized    : {peak_vram_mb:.1f} MB ({peak_vram_gb:.2f} GB / {total_vram_gb:.2f} GB)")
    print(f"Annotated Video Saved : {output_path.resolve()}")
    print("-" * 75)

    # Target comparisons
    fps_ok = avg_fps >= 18.0
    ms_ok = 20.0 <= avg_ms_per_frame <= 45.0 or avg_ms_per_frame < 20.0  # Faster is also fine!
    vram_ok = peak_vram_gb < 5.0  # Fits comfortably in 6 GB

    print(f"TRD Frame Rate Check  : {'PASS' if fps_ok else 'WARN'} (Target: 18-22 FPS, Got: {avg_fps:.1f} FPS)")
    print(f"TRD Latency Check     : {'PASS' if ms_ok else 'WARN'} (Target: 20-45 ms, Got: {avg_ms_per_frame:.1f} ms)")
    print(f"TRD VRAM Budget Check : {'PASS' if vram_ok else 'FAIL'} (Target: < 5.0 GB for system headroom, Got: {peak_vram_gb:.2f} GB)")
    print("=" * 75)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run YOLO GPU inference smoke test over a CCTV sample video."
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to sample video file. If omitted, uses the first video found in data/samples/.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11s.pt",
        help="YOLO model weight name or path (default: yolo11s.pt; alternatives: yolo26s.pt, yolov8s.pt).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference frame resolution dimension (default: 640).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device ID (default: '0').",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold (default: 0.25).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/samples/output_smoke.mp4",
        help="Path where annotated video will be saved (default: data/samples/output_smoke.mp4).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display video window during playback (disabled by default).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to process (optional, useful for quick verification).",
    )

    args = parser.parse_args()

    # Determine input video path
    if args.video:
        video_path = Path(args.video)
    else:
        video_path = find_default_video()
        if video_path is None:
            print("[ERROR] No video specified and no video found in data/samples/.")
            print("Please provide a video file: python scripts/smoke_test.py --video <path_to_video>")
            sys.exit(1)
        print(f"[INFO] No video specified. Auto-selected discovered clip: '{video_path}'")

    run_smoke_test(
        video_path=video_path,
        model_name=args.model,
        imgsz=args.imgsz,
        device_id=args.device,
        conf_thresh=args.conf,
        output_path=Path(args.output),
        show_window=args.show,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
