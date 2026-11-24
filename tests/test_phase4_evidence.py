"""Unit tests for Phase 4 Evidence Collector.

Covers functional tests T-23, T-24, T-25 from docs/10_TEST_PLAN.md:
- T-23: Incident created -> Person, face, object images and video clip collected.
- T-24: Open evidence -> Files load, exist with non-zero size, and video clip is playable.
- T-25: Face not visible -> Incident still valid, face image marked unavailable.
- Rolling RAM frame buffer bounded memory limit & duration eviction.
- StorageRepository SQLite persistence of evidence records.
"""

from pathlib import Path
import tempfile
import cv2
import numpy as np
import pytest

from src.common.types import (
    Evidence,
    EvidenceType,
    Incident,
    Severity,
)
from src.evidence.clip_writer import ClipWriter
from src.evidence.collector import EvidenceCollector
from src.evidence.crop_extractor import CropExtractor
from src.evidence.ring_buffer import RollingFrameBuffer
from src.storage.repository import StorageRepository


def make_dummy_frame(width: int = 320, height: int = 240, color: tuple = (100, 150, 200)) -> np.ndarray:
    """Create a synthetic solid color image frame."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = color
    # Add a rectangle pattern
    cv2.rectangle(frame, (50, 50), (120, 180), (255, 255, 255), -1)
    return frame


def test_rolling_frame_buffer_duration_and_memory():
    """Verify rolling frame buffer enforces duration eviction and memory caps."""
    # 5.0 second duration buffer with 1.0 MB cap
    buffer = RollingFrameBuffer(
        buffer_duration_seconds=5.0,
        max_buffer_memory_mb=1.0,
        jpeg_quality=75,
    )

    frame = make_dummy_frame(640, 480)

    # Add frames across 10 seconds
    for fid, ts in enumerate(np.linspace(0.0, 10.0, 30)):
        buffer.add_frame(frame, timestamp=float(ts), frame_id=fid)

    # 1. Total memory should be strictly within 1.0 MB
    assert buffer.current_memory_mb <= 1.0

    # 2. Oldest frame should not be older than (latest_ts - duration)
    latest = buffer.get_latest_frame()
    assert latest is not None
    latest_ts = latest[0]
    frames = buffer.get_frames_in_range(0.0, 10.0)
    for ts, fid, _ in frames:
        assert (latest_ts - ts) <= 5.5  # duration + margin


def test_frame_range_retrieval_and_decoding():
    """Verify exact decoding of frames within specified range."""
    buffer = RollingFrameBuffer(buffer_duration_seconds=10.0)
    frame = make_dummy_frame(320, 240, color=(0, 255, 0))

    for fid in range(10):
        buffer.add_frame(frame, timestamp=float(fid), frame_id=fid)

    retrieved = buffer.get_frames_in_range(3.0, 6.0)
    assert len(retrieved) == 4
    timestamps = [r[0] for r in retrieved]
    assert timestamps == [3.0, 4.0, 5.0, 6.0]
    # Check decoded shape
    assert retrieved[0][2].shape == (240, 320, 3)


def test_t23_t24_evidence_collection_and_playable_clip(tmp_path):
    """T-23 & T-24: Person, face, object crops and playable MP4 clip are produced."""
    cfg = {
        "ring_buffer": {"buffer_duration_seconds": 30.0, "max_buffer_memory_mb": 50.0},
        "video_clip": {"pre_roll_seconds": 2.0, "post_roll_seconds": 2.0, "fps": 10.0},
        "policy": {"min_level_for_clip": "Medium", "evidence_dir": str(tmp_path / "evidence")},
    }

    collector = EvidenceCollector(config=cfg)
    cam_id = "CAM-TEST"
    frame = make_dummy_frame(640, 480)

    # Buffer 20 frames from t=0.0 to t=10.0
    for fid, ts in enumerate(np.linspace(0.0, 10.0, 20)):
        collector.add_frame(cam_id, frame, timestamp=float(ts), frame_id=fid)

    incident = Incident(
        id=1,
        incident_no="INC-000099",
        camera_id=cam_id,
        place="Test Gate",
        subject={"track_id": 42, "object_id": "bag_7", "identity_status": "KNOWN", "person_name": "Bob"},
        threat_level=Severity.HIGH.value,
        threat_score=65,
        start_time=2.0,
        last_event_time=8.0,
    )

    face_crop = make_dummy_frame(80, 80, color=(200, 100, 50))
    person_bbox = (50.0, 50.0, 150.0, 300.0)
    object_bbox = (160.0, 200.0, 220.0, 280.0)

    evidence_items = collector.collect_for_incident(
        incident=incident,
        current_frame=frame,
        person_bbox=person_bbox,
        object_bbox=object_bbox,
        best_face_crop=face_crop,
    )

    # Should collect 4 items: person_image, object_image, face_image, video_clip
    types = [e.type for e in evidence_items]
    assert EvidenceType.PERSON_IMAGE.value in types
    assert EvidenceType.OBJECT_IMAGE.value in types
    assert EvidenceType.FACE_IMAGE.value in types
    assert EvidenceType.VIDEO_CLIP.value in types

    # T-24: Verify each physical file exists, is non-zero, and video is playable
    for ev in evidence_items:
        fpath = Path(ev.file_path)
        assert fpath.exists()
        assert fpath.stat().st_size > 0

        if ev.type == EvidenceType.VIDEO_CLIP.value:
            cap = cv2.VideoCapture(str(fpath))
            assert cap.isOpened()
            read_ok, read_frame = cap.read()
            assert read_ok is True
            assert read_frame.shape == (480, 640, 3)
            cap.release()


def test_clip_writer_h264_browser_compatibility(tmp_path):
    """Problem 2: Verify ClipWriter produces H.264 / yuv420p video clips compatible with HTML5 browsers."""
    import json
    import shutil
    import subprocess

    ffprobe_bin = shutil.which("ffprobe")
    if not ffprobe_bin:
        pytest.skip("ffprobe not available in environment")

    writer = ClipWriter(fps=25.0)
    out_file = tmp_path / "browser_test_clip.mp4"
    frames = [
        (float(i) * 0.04, i, make_dummy_frame(320, 240, color=(10 * i % 255, 100, 150)))
        for i in range(25)
    ]

    result_path = writer.write_clip(frames, out_file)
    assert result_path is not None
    assert out_file.exists()
    assert out_file.stat().st_size > 0

    cmd = [
        ffprobe_bin,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,pix_fmt",
        "-of", "json",
        str(out_file),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    probe_data = json.loads(res.stdout)
    stream = probe_data["streams"][0]

    assert stream["codec_name"] == "h264", f"Expected h264 codec, got {stream['codec_name']}"
    assert stream["pix_fmt"] == "yuv420p", f"Expected yuv420p pixel format, got {stream['pix_fmt']}"


def test_t25_missing_face_handled_gracefully(tmp_path):
    """T-25: Face not visible -> Incident still saved; face image marked unavailable."""
    cfg = {
        "ring_buffer": {"buffer_duration_seconds": 15.0},
        "policy": {"evidence_dir": str(tmp_path / "evidence")},
    }
    collector = EvidenceCollector(config=cfg)

    incident = Incident(
        id=2,
        incident_no="INC-000100",
        camera_id="CAM-OVERHEAD",
        place="Warehouse High Bay",
        subject={"track_id": 99, "identity_status": "UNKNOWN", "reason": "no_usable_face"},
        threat_level=Severity.MEDIUM.value,
        threat_score=35,
        start_time=1.0,
        last_event_time=2.0,
    )

    evidence_items = collector.collect_for_incident(
        incident=incident,
        best_face_crop=None,  # Face not visible
    )

    face_ev = [e for e in evidence_items if e.type == EvidenceType.FACE_IMAGE.value]
    assert len(face_ev) == 1
    assert face_ev[0].file_path == ""
    assert face_ev[0].details.get("status") == "unavailable"


def test_evidence_persistence_in_storage_repository(tmp_path):
    """Verify Evidence artifacts are persisted to SQLite via StorageRepository."""
    db_file = tmp_path / "test_surveillance.db"
    repo = StorageRepository(db_path=db_file)
    repo.register_camera("CAM-STORE")

    cfg = {"policy": {"evidence_dir": str(tmp_path / "evidence")}}
    collector = EvidenceCollector(config=cfg, repository=repo)

    incident = Incident(
        incident_no="INC-000101",
        camera_id="CAM-STORE",
        place="Storefront",
        threat_level=Severity.HIGH.value,
        threat_score=60,
        start_time=5.0,
        last_event_time=10.0,
    )
    inc_id = repo.create_incident(incident)
    incident.id = inc_id

    frame = make_dummy_frame(320, 240)
    collector.add_frame("CAM-STORE", frame, timestamp=5.0, frame_id=1)

    collector.collect_for_incident(
        incident=incident,
        current_frame=frame,
        person_bbox=(20.0, 20.0, 80.0, 180.0),
    )

    # Query back from DB
    loaded = repo.get_incident(inc_id)
    assert loaded is not None
    assert len(loaded.evidence) >= 1
    p_ev = loaded.evidence[0]
    assert p_ev.type == EvidenceType.PERSON_IMAGE.value
    assert Path(p_ev.file_path).exists()
