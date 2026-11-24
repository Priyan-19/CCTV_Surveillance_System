# Phase 3 Face Recognition & Surveillance Pipeline Benchmark

**Date:** 2026-11-22 12:59:53
**Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM), Windows 11
**Architecture:** YOLO26s (FP16) + ByteTrack + 6 Security Rules + Asynchronous InsightFace (`buffalo_s`)

## 1. Raw Neural Model Latency (InsightFace `buffalo_s` on GPU)

- **Mean Latency:** 9.06 ms per crop
- **p50 Latency:**  8.56 ms
- **p95 Latency:**  17.32 ms
- **Max Asynchronous Capacity:** 110.4 face jobs/sec

## 2. End-to-End Pipeline Performance vs Phase 2 Baseline

| Scenario | Resolution | Phase 2 FPS | Phase 3 FPS | FPS Delta | Phase 3 Latency | Queue Drops | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Single walker (CAVIAR 384x288) | 640px | 23.49 | **24.81** | +1.32 | 40.23 ms | 0 (0.0%) | PASS |
| Abandoned bag HD (ABODA 640x480) | 960px | 24.64 | **18.83** | -5.81 | 53.03 ms | 0 (0.0%) | BELOW TARGET |
| Crowd altercation (CAVIAR 384x288) | 640px | 27.92 | **27.32** | -0.60 | 36.46 ms | 0 (0.0%) | PASS |
| Normal pedestrians (CAVIAR 384x288) | 640px | 27.72 | **27.72** | +0.00 | 35.88 ms | 0 (0.0%) | PASS |

## 3. Real-Time Compliance Summary

- **TRD Target Window:** 18.0 - 22.0 FPS | 20.0 - 45.0 ms per frame
- **Outcome:** SOME CLIPS BELOW TARGET
- **Asynchronous Queue Impact:** Main capture and tracking thread incurred < 0.15 ms dispatch overhead. Face neural inference ran entirely in background worker thread without choking video FPS.
