"""Unit tests for Phase 1 components.

All tests run cleanly without GPU or neural network weight requirements:
- Dataclass invariants and properties
- Class filtering logic
- Bounding box centroid math and trail history trimming
- YAML configuration file validation
- VideoSource queuing and thread lifecycle
- Tracker interface ID persistence on synthetic trajectories
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List
import numpy as np
import pytest
import yaml

from src.capture.video_source import VideoSource
from src.common.types import Detection, FrameResult, Track
from src.tracking.tracker import SyntheticDetectionTracker


# ---------------------------------------------------------------------------
# 1. Dataclass and Helper Property Tests
# ---------------------------------------------------------------------------

def test_detection_properties() -> None:
    det = Detection(
        camera_id="cam_test",
        frame_id=1,
        timestamp=1000.0,
        class_id=0,
        class_name="person",
        bbox=(100.0, 50.0, 200.0, 250.0),
        confidence=0.92,
    )
    assert det.width == 100.0
    assert det.height == 200.0
    assert det.area == 20000.0
    assert det.centre == (150.0, 150.0)


def test_track_properties_and_lifespan() -> None:
    trk = Track(
        camera_id="cam_test",
        track_id=5,
        class_id=24,
        class_name="backpack",
        bbox=(50.0, 80.0, 110.0, 160.0),
        confidence=0.85,
        path=[(80.0, 120.0)],
        first_seen=100.0,
        last_seen=115.5,
    )
    assert trk.centre == (80.0, 120.0)
    assert trk.lifespan_seconds == 15.5
    assert trk.zone_ids == []
    assert trk.dwell_seconds == 0.0


def test_frame_result_structure() -> None:
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    res = FrameResult(
        frame=dummy_frame,
        frame_id=42,
        timestamp=500.0,
        detections=[],
        tracks=[],
        timings={"detect_ms": 15.2, "track_ms": 3.1},
    )
    assert res.frame_id == 42
    assert "detect_ms" in res.timings
    assert len(res.tracks) == 0


# ---------------------------------------------------------------------------
# 2. Centroid and Trail Trimming Tests
# ---------------------------------------------------------------------------

def test_path_trimming_limit() -> None:
    max_trail = 10
    tracker = SyntheticDetectionTracker(max_trail_length=max_trail)

    # Feed 25 consecutive detections for the same moving object
    for fid in range(1, 26):
        det = Detection(
            camera_id="cam_test",
            frame_id=fid,
            timestamp=float(fid),
            class_id=0,
            class_name="person",
            bbox=(float(fid * 2), 50.0, float(fid * 2 + 20), 100.0),
            confidence=0.90,
        )
        tracks = tracker.update_with_detections([det], "cam_test", fid, float(fid))
        assert len(tracks) == 1
        trk = tracks[0]
        assert trk.track_id == 1  # ID must persist

    # Path must be trimmed to exactly max_trail points
    assert len(tracks[0].path) == max_trail
    # Newest point must match the 25th frame's centroid
    assert tracks[0].path[-1] == (float(25 * 2 + 10), 75.0)


# ---------------------------------------------------------------------------
# 3. Class Filtering Logic
# ---------------------------------------------------------------------------

def test_class_filtering() -> None:
    allowed_classes = {0, 24, 26}  # person, backpack, handbag

    raw_detections = [
        Detection("cam1", 1, 1.0, 0, "person", (0, 0, 10, 10), 0.9),
        Detection("cam1", 1, 1.0, 15, "cat", (0, 0, 10, 10), 0.8),
        Detection("cam1", 1, 1.0, 24, "backpack", (0, 0, 10, 10), 0.7),
        Detection("cam1", 1, 1.0, 62, "tv", (0, 0, 10, 10), 0.85),
    ]

    filtered = [d for d in raw_detections if d.class_id in allowed_classes]
    assert len(filtered) == 2
    assert {d.class_name for d in filtered} == {"person", "backpack"}


# ---------------------------------------------------------------------------
# 4. Configuration File Validation Tests
# ---------------------------------------------------------------------------

def test_pipeline_config_exists_and_valid() -> None:
    cfg_path = Path("configs/pipeline.yaml")
    assert cfg_path.exists(), "configs/pipeline.yaml is missing"
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert "model" in data
    assert "tracked_classes" in data
    assert "tracker" in data
    assert data["model"]["imgsz"] in (640, 960)
    assert 0 in data["tracked_classes"]  # person class must be present


def test_bytetrack_config_exists_and_valid() -> None:
    cfg_path = Path("configs/bytetrack.yaml")
    assert cfg_path.exists(), "configs/bytetrack.yaml is missing"
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data.get("tracker_type") == "bytetrack"
    assert "track_high_thresh" in data
    assert "track_low_thresh" in data
    assert "track_buffer" in data
    assert data["track_high_thresh"] > data["track_low_thresh"]


def test_cameras_config_exists_and_valid() -> None:
    cfg_path = Path("configs/cameras.yaml")
    assert cfg_path.exists(), "configs/cameras.yaml is missing"
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert "cameras" in data
    assert len(data["cameras"]) >= 5


# ---------------------------------------------------------------------------
# 5. Tracker Interface & Smooth Trajectory ID Persistence Test
# ---------------------------------------------------------------------------

def test_tracker_id_persistence_on_smooth_trajectory() -> None:
    tracker = SyntheticDetectionTracker(iou_thresh=0.3)

    # Object 1 walks smoothly across 12 frames (x shifts by 5px per frame)
    # Object 2 appears at frame 5 at an isolated position
    for fid in range(1, 13):
        dets = []
        # Object 1
        x1_obj1 = 100.0 + (fid * 5.0)
        dets.append(
            Detection("cam1", fid, float(fid), 0, "person", (x1_obj1, 100.0, x1_obj1 + 40.0, 180.0), 0.95)
        )
        # Object 2 (enters later)
        if fid >= 5:
            dets.append(
                Detection("cam1", fid, float(fid), 24, "backpack", (400.0, 300.0, 440.0, 340.0), 0.88)
            )

        tracks = tracker.update_with_detections(dets, "cam1", fid, float(fid))

        # Check Object 1 ID persistence
        obj1_tracks = [t for t in tracks if t.class_id == 0]
        assert len(obj1_tracks) == 1
        assert obj1_tracks[0].track_id == 1  # Track ID 1 must NEVER change across all 12 frames

        if fid >= 5:
            # Check Object 2 has distinct ID
            obj2_tracks = [t for t in tracks if t.class_id == 24]
            assert len(obj2_tracks) == 1
            assert obj2_tracks[0].track_id == 2  # Track ID 2


# ---------------------------------------------------------------------------
# 6. VideoSource File Reading and Lifecycle Test
# ---------------------------------------------------------------------------

def test_video_source_on_sample_file() -> None:
    sample_file = Path("data/samples/sample_walking_01.mp4")
    if not sample_file.exists():
        pytest.skip(f"Test clip {sample_file} not found")

    vs = VideoSource("test_cam", str(sample_file), queue_size=20, drop_frames=False)
    vs.start()

    frames_read = 0
    prev_frame_id = 0

    try:
        for _ in range(15):
            cap_frame = vs.read(timeout=2.0)
            if cap_frame is None:
                break
            frames_read += 1
            # Frame IDs must start at 1 and increment monotonically
            assert cap_frame.frame_id == prev_frame_id + 1
            prev_frame_id = cap_frame.frame_id
            assert cap_frame.frame is not None
            assert cap_frame.camera_id == "test_cam"
    finally:
        vs.stop()

    assert frames_read == 15
    # Ensure background worker thread terminated cleanly
    assert vs._worker_thread is None or not vs._worker_thread.is_alive()
