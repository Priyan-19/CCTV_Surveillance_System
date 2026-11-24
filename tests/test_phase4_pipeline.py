"""Integration tests for Phase 4 Pipeline.

Verifies end-to-end pipeline execution:
Video Source -> Detector -> ByteTrack -> FaceWorker -> RuleEngine -> ThreatEngine -> EvidenceCollector -> SQLite.
"""

from pathlib import Path
import tempfile
import pytest

from src.pipeline.phase4_pipeline import Phase4Pipeline


def test_phase4_pipeline_end_to_end_smoke(tmp_path):
    """Smoke test: execute Phase 4 pipeline on sample clip for 25 frames."""
    sample_clip = Path("data/samples/sample_walking_01.mp4")
    if not sample_clip.exists():
        pytest.skip("Sample walking clip not found.")

    db_path = tmp_path / "smoke_cctv.db"
    out_dir = tmp_path / "output"
    ev_dir = tmp_path / "evidence"

    pipeline = Phase4Pipeline(
        camera_id="cam_walk_01",
        source=str(sample_clip),
        model_name="yolo26s.pt",
        imgsz=640,
        device="cuda:0",
        save_video=False,
        show_window=False,
        db_path=str(db_path),
        output_dir=out_dir,
    )

    summary = pipeline.run(max_frames=25)

    assert summary is not None
    assert summary.get("total_frames", 0) >= 20
    assert "stage_latencies" in summary
    assert "threat_ms" in summary["stage_latencies"]
    assert "incidents" in summary
