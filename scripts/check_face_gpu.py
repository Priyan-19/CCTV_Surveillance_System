"""Check ONNX Runtime GPU Execution Provider and Measure Raw InsightFace Speed.

Verifies:
1. onnxruntime.get_available_providers() contains 'CUDAExecutionProvider'.
2. InsightFace loads on CUDAExecutionProvider without falling back to CPU.
3. Measures raw detection and ArcFace embedding speed (ms/call) on synthetic frames.

Usage:
    python scripts/check_face_gpu.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("check_face_gpu")


def main() -> None:
    print("=" * 80)
    print("         ONNX RUNTIME & INSIGHTFACE GPU PROVIDER DIAGNOSTIC")
    print("=" * 80)

    # 1. Check ONNX Runtime providers
    try:
        # On Windows, register PyTorch CUDA/cuDNN DLL directory so onnxruntime can load onnxruntime_providers_cuda.dll
        import os
        try:
            import torch
            torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
            if os.path.exists(torch_lib):
                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(torch_lib)
                os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
        except Exception:
            pass

        import onnxruntime as ort
        print(f"ONNX Runtime Version   : {ort.__version__}")
        providers = ort.get_available_providers()
        print(f"Available Providers    : {providers}")

        if "CUDAExecutionProvider" not in providers:
            print("\n[CRITICAL ERROR] 'CUDAExecutionProvider' NOT FOUND in onnxruntime providers!")
            print("onnxruntime is configured for CPU-only.")
            print("Please fix by installing matching onnxruntime-gpu for your CUDA version:")
            print("  uv pip install --upgrade onnxruntime-gpu")
            sys.exit(1)
        else:
            print("[CONFIRMED] 'CUDAExecutionProvider' is available in ONNX Runtime.")
    except ImportError as e:
        print(f"[ERROR] Failed to import onnxruntime: {e}")
        sys.exit(1)

    # 2. Check InsightFace initialization
    try:
        import insightface
        from insightface.app import FaceAnalysis

        print(f"\nInsightFace Version    : {insightface.__version__}")
        print("Initializing FaceAnalysis (name='buffalo_s') on CUDAExecutionProvider...")

        # Initialize with buffalo_s first (lightweight for CCTV face worker)
        app = FaceAnalysis(
            name="buffalo_s",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        app.prepare(ctx_id=0, det_size=(320, 320))
        print("[CONFIRMED] FaceAnalysis (buffalo_s) prepared successfully on ctx_id=0.")

        # Inspect models loaded
        print("\nLoaded InsightFace Sub-Models & Active Providers:")
        all_on_cuda = True
        for model_name, model in app.models.items():
            session = getattr(model, "session", None)
            if session:
                active_providers = session.get_providers()
                print(f"  - {model_name:<16}: Active Provider -> {active_providers[0]}")
                if "CUDAExecutionProvider" not in active_providers[0]:
                    all_on_cuda = False
            else:
                print(f"  - {model_name:<16}: (custom model)")

        if not all_on_cuda:
            print("\n[WARNING] Some models fell back to CPUExecutionProvider!")
        else:
            print("\n[CONFIRMED] ALL InsightFace sub-models are actively running on CUDAExecutionProvider.")

        # 3. Benchmark raw inference speed on synthetic frame
        print("\nBenchmarking raw detection + embedding speed (320x320 crop, 30 iterations)...")
        synthetic_crop = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)

        # Warm-up
        for _ in range(5):
            _ = app.get(synthetic_crop)

        latencies = []
        for _ in range(30):
            t0 = time.perf_counter()
            _ = app.get(synthetic_crop)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

        mean_ms = sum(latencies) / len(latencies)
        p50_ms = sorted(latencies)[len(latencies) // 2]
        p95_ms = sorted(latencies)[int(len(latencies) * 0.95)]

        print("-" * 80)
        print(f"Raw Face Processing Latency : Mean: {mean_ms:5.2f} ms | p50: {p50_ms:5.2f} ms | p95: {p95_ms:5.2f} ms")
        print(f"Max Asynchronous Throughput : {1000.0 / mean_ms:5.1f} face jobs/sec")
        print("=" * 80)

    except Exception as exc:
        print(f"\n[EXCEPTION] Failed during InsightFace GPU execution: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
