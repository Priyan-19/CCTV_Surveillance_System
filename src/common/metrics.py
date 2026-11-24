"""Performance metrics accumulator and latency profiler.

Tracks rolling and final statistics for:
- Frame rates (end-to-end throughput FPS and inference-only FPS).
- Latency per pipeline stage (read, detect, track, draw, write) with mean, p50, and p95.
- Object counts (people, bags, vehicles).
- GPU VRAM consumption: torch.cuda allocated and reserved, plus system GPU memory.
"""

from __future__ import annotations

import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional
import torch


@dataclass
class StageStats:
    """Statistical summary for a single latency metric."""
    count: int = 0
    mean_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0


class PerformanceMetrics:
    """Thread-safe performance metrics collector across pipeline frames."""

    def __init__(self, rolling_window_size: int = 60) -> None:
        self.rolling_window_size = rolling_window_size
        self.stage_samples: Dict[str, List[float]] = {
            "read_ms": [],
            "detect_ms": [],
            "track_ms": [],
            "draw_ms": [],
            "write_ms": [],
            "total_ms": [],
        }
        self.recent_frame_times: Deque[float] = deque(maxlen=rolling_window_size)
        self.start_wall_time: Optional[float] = None
        self.end_wall_time: Optional[float] = None
        self.total_frames_processed: int = 0

        # Object tracking counts history
        self.people_counts: List[int] = []
        self.bag_counts: List[int] = []
        self.vehicle_counts: List[int] = []

    def start_pipeline(self) -> None:
        """Mark pipeline start time."""
        self.start_wall_time = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def record_frame(
        self,
        timings_ms: Dict[str, float],
        people_count: int = 0,
        bag_count: int = 0,
        vehicle_count: int = 0,
    ) -> None:
        """Record per-stage latency timings and object counts for one frame."""
        now = time.perf_counter()
        self.recent_frame_times.append(now)
        self.total_frames_processed += 1

        total_frame_ms = sum(timings_ms.values())
        timings_ms["total_ms"] = total_frame_ms

        for stage, duration in timings_ms.items():
            if stage not in self.stage_samples:
                self.stage_samples[stage] = []
            self.stage_samples[stage].append(duration)

        self.people_counts.append(people_count)
        self.bag_counts.append(bag_count)
        self.vehicle_counts.append(vehicle_count)

    def stop_pipeline(self) -> None:
        """Mark pipeline termination time."""
        self.end_wall_time = time.perf_counter()

    @property
    def rolling_fps(self) -> float:
        """Calculate FPS over the rolling window."""
        if len(self.recent_frame_times) < 2:
            return 0.0
        elapsed = self.recent_frame_times[-1] - self.recent_frame_times[0]
        return (len(self.recent_frame_times) - 1) / elapsed if elapsed > 0 else 0.0

    @property
    def overall_throughput_fps(self) -> float:
        """Overall end-to-end FPS including video reading, drawing, and encoding."""
        if self.start_wall_time is None:
            return 0.0
        end_time = self.end_wall_time or time.perf_counter()
        elapsed = end_time - self.start_wall_time
        return self.total_frames_processed / elapsed if elapsed > 0 else 0.0

    @property
    def inference_only_fps(self) -> float:
        """FPS considering only model detection + tracking latencies."""
        det_times = self.stage_samples.get("detect_ms", [])
        trk_times = self.stage_samples.get("track_ms", [])
        if not det_times:
            return 0.0

        n = len(det_times)
        total_model_ms = sum(det_times)
        if trk_times:
            total_model_ms += sum(trk_times)

        avg_model_ms = total_model_ms / n if n > 0 else 0.0
        return (1000.0 / avg_model_ms) if avg_model_ms > 0 else 0.0

    def compute_stage_stats(self, stage: str) -> StageStats:
        """Compute mean, p50 (median), p95, min, and max for a given stage."""
        samples = self.stage_samples.get(stage, [])
        if not samples:
            return StageStats()

        sorted_samples = sorted(samples)
        count = len(sorted_samples)
        mean_val = statistics.mean(sorted_samples)
        p50_val = statistics.median(sorted_samples)
        p95_idx = min(int(count * 0.95), count - 1)
        p95_val = sorted_samples[p95_idx]

        return StageStats(
            count=count,
            mean_ms=mean_val,
            p50_ms=p50_val,
            p95_ms=p95_val,
            min_ms=sorted_samples[0],
            max_ms=sorted_samples[-1],
        )

    def get_vram_usage(self) -> Dict[str, float]:
        """Query PyTorch CUDA VRAM allocations (allocated, reserved, and device capacity)."""
        if not torch.cuda.is_available():
            return {
                "allocated_mb": 0.0,
                "reserved_mb": 0.0,
                "total_device_gb": 0.0,
            }

        allocated_mb = torch.cuda.max_memory_allocated() / (1024**2)
        reserved_mb = torch.cuda.max_memory_reserved() / (1024**2)
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        return {
            "allocated_mb": round(allocated_mb, 2),
            "reserved_mb": round(reserved_mb, 2),
            "total_device_gb": round(total_gb, 2),
        }

    def generate_summary(self) -> Dict[str, Any]:
        """Produce a complete metrics dictionary for logging, JSON export, and reports."""
        stage_names = list(self.stage_samples.keys())
        if "total_ms" in stage_names:
            stage_names.remove("total_ms")
            stage_names.append("total_ms")
        stage_dict = {}
        for name in stage_names:
            stats = self.compute_stage_stats(name)
            stage_dict[name] = {
                "mean_ms": round(stats.mean_ms, 2),
                "p50_ms": round(stats.p50_ms, 2),
                "p95_ms": round(stats.p95_ms, 2),
                "min_ms": round(stats.min_ms, 2),
                "max_ms": round(stats.max_ms, 2),
            }

        vram = self.get_vram_usage()
        elapsed_sec = (self.end_wall_time or time.perf_counter()) - (self.start_wall_time or time.perf_counter())

        return {
            "total_frames": self.total_frames_processed,
            "elapsed_seconds": round(elapsed_sec, 2),
            "overall_throughput_fps": round(self.overall_throughput_fps, 2),
            "inference_only_fps": round(self.inference_only_fps, 2),
            "stage_latencies": stage_dict,
            "vram_metrics": vram,
            "average_people_count": round(statistics.mean(self.people_counts), 2) if self.people_counts else 0.0,
            "max_people_count": max(self.people_counts) if self.people_counts else 0,
            "max_bag_count": max(self.bag_counts) if self.bag_counts else 0,
        }
