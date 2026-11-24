"""Diagnostic tool to evaluate bag class detection rates.

Specifically evaluates:
- Bag detections (backpack, handbag, suitcase) per second.
- Percentage of frames with bag detected.
- Compares imgsz=640 vs imgsz=960 with yolo26s.pt.

Usage:
    python scripts/check_bag_detection.py --source data/samples/sample_abandoned_bag_hd.mp4 --model yolo26s.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List
import cv2
import torch
from ultralytics import YOLO

# Surveillance bag class IDs in COCO: 24: backpack, 26: handbag, 28: suitcase
BAG_CLASS_IDS = {24, 26, 28}
BAG_NAMES = {24: "backpack", 26: "handbag", 28: "suitcase"}


def evaluate_bag_detections(
    video_path: Path,
    model_name: str = "yolo26s.pt",
    img_sizes: List[int] = [640, 960],
    conf_thresh: float = 0.25,
    max_frames: int = 600,
) -> None:
    print("=" * 80)
    print(f"             BAG DETECTION AUDIT: {video_path.name}")
    print("=" * 80)
    print(f"Model: {model_name} | Confidence Threshold: {conf_thresh} | Max Frames: {max_frames}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_name)
    model.to(device)
    if "cuda" in device:
        try:
            model.model.half()
        except Exception:
            pass

    for imgsz in img_sizes:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[ERROR] Could not open video: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_file_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        eval_frames = min(max_frames, total_file_frames) if total_file_frames > 0 else max_frames

        frame_count = 0
        frames_with_bag = 0
        total_bag_detections = 0
        bag_confidences: List[float] = []

        t0 = time.perf_counter()
        while cap.isOpened() and frame_count < eval_frames:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frame_count += 1

            results = model.predict(
                frame,
                imgsz=imgsz,
                device=device,
                conf=conf_thresh,
                classes=list(BAG_CLASS_IDS),
                verbose=False,
            )

            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                classes = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()

                has_bag_in_frame = False
                for c, score in zip(classes, confs):
                    if c in BAG_CLASS_IDS:
                        has_bag_in_frame = True
                        total_bag_detections += 1
                        bag_confidences.append(float(score))

                if has_bag_in_frame:
                    frames_with_bag += 1

        cap.release()
        elapsed_sec = time.perf_counter() - t0
        video_duration_sec = frame_count / fps if fps > 0 else 1.0

        detections_per_second = total_bag_detections / video_duration_sec
        bag_frame_pct = (frames_with_bag / frame_count * 100.0) if frame_count > 0 else 0.0
        avg_conf = (sum(bag_confidences) / len(bag_confidences)) if bag_confidences else 0.0

        print(f"\n--- Results for imgsz = {imgsz} px ---")
        print(f"Frames Evaluated        : {frame_count} frames ({video_duration_sec:.1f} video seconds)")
        print(f"Frames with Bag Detected: {frames_with_bag} ({bag_frame_pct:.1f}%)")
        print(f"Total Bag Detections    : {total_bag_detections}")
        print(f"Bag Detections / Second : {detections_per_second:.2f} det/sec")
        print(f"Average Bag Confidence  : {avg_conf:.3f}")
        print(f"Audit Throughput Speed  : {frame_count / elapsed_sec:.1f} FPS")

    print("\n" + "=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit bag detection frequency.")
    parser.add_argument(
        "--source",
        type=str,
        default="data/samples/sample_abandoned_bag_hd.mp4",
        help="Path to video clip.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo26s.pt",
        help="Model weights (default: yolo26s.pt).",
    )
    parser.add_argument(
        "--img-sizes",
        nargs="+",
        type=int,
        default=[640, 960],
        help="Image sizes to test.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=1200,
        help="Max frames to test.",
    )
    args = parser.parse_args()

    v_path = Path(args.source)
    if not v_path.exists():
        print(f"[ERROR] Video file '{v_path}' not found.")
        sys.exit(1)

    evaluate_bag_detections(
        video_path=v_path,
        model_name=args.model,
        img_sizes=args.img_sizes,
        conf_thresh=args.conf,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
