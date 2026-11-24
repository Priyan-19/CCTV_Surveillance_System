"""Comprehensive Unit Test Suite for Phase 3 (Face Recognition).

Conforms to:
- docs/10_TEST_PLAN.md (T-05, T-06, T-07)
- docs/06_ML_MODEL_PLAN.md (Section 4)
- docs/07_DATABASE_SCHEMA.md (persons, face_embeddings cascade delete)
- docs/11_SECURITY_PRIVACY.md (Right to remove, consent)
- docs/05_THREAT_ENGINE_SPEC.md (UNKNOWN_PERSON, authorized zones)

Features:
- 100% CPU-safe (no GPU, camera, or model download required).
- Uses synthetic 512-d L2-normalized vectors and synthetic image matrices.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import cv2
import numpy as np
import pytest

from src.common.types import IdentityStatus, Track
from src.face.identity import IdentityManager
from src.face.matcher import FaceMatcher, MatchResult
from src.face.quality import FaceQualityFilter
from src.face.store import FaceStore


# ==============================================================================
# 1. Face Quality Filter Tests (docs/10_TEST_PLAN.md T-07)
# ==============================================================================

def test_quality_filter_size_and_confidence():
    """Verify quality filter rejects undersized faces and low detection confidence."""
    qfilter = FaceQualityFilter(min_face_size=40, min_blur_score=30.0, min_det_score=0.60)

    # Undersized crop (24x24 px)
    small_crop = np.full((24, 24, 3), 128, dtype=np.uint8)
    res_small = qfilter.evaluate(small_crop, det_score=0.95)
    assert not res_small.is_valid
    assert "low_resolution" in res_small.reason

    # Low detection score (0.45 < 0.60)
    normal_crop = np.full((64, 64, 3), 128, dtype=np.uint8)
    res_low_conf = qfilter.evaluate(normal_crop, det_score=0.45)
    assert not res_low_conf.is_valid
    assert "low_detection_confidence" in res_low_conf.reason


def test_quality_filter_blur_detection():
    """Verify Laplacian variance correctly rejects blurry crops and passes sharp crops."""
    qfilter = FaceQualityFilter(min_face_size=40, min_blur_score=35.0, min_det_score=0.60)

    # Completely flat image (variance = 0) -> blurry
    flat_crop = np.full((80, 80, 3), 128, dtype=np.uint8)
    res_blur = qfilter.evaluate(flat_crop, det_score=0.90)
    assert not res_blur.is_valid
    assert "blurred_face" in res_blur.reason

    # High frequency / high contrast pattern -> sharp
    sharp_crop = np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8)
    res_sharp = qfilter.evaluate(sharp_crop, det_score=0.90)
    assert res_sharp.is_valid
    assert res_sharp.reason == "quality_passed"
    assert res_sharp.blur_score > 35.0


# ==============================================================================
# 2. Cosine Similarity Matcher & Margin Tests (T-05, T-06)
# ==============================================================================

def test_matcher_known_and_unknown():
    """Verify matcher correctly declares KNOWN when similarity exceeds threshold, UNKNOWN otherwise."""
    matcher = FaceMatcher(similarity_threshold=0.50, required_margin=0.08)

    # Create synthetic orthogonal 512-d base vectors
    vec_alice = np.zeros(512, dtype=np.float32)
    vec_alice[0] = 1.0

    vec_bob = np.zeros(512, dtype=np.float32)
    vec_bob[1] = 1.0

    gallery = [
        {"person_id": 1, "name": "Alice", "category": "staff", "authorised_zone_ids": ["zone_1"], "embedding": vec_alice},
        {"person_id": 2, "name": "Bob", "category": "visitor", "authorised_zone_ids": [], "embedding": vec_bob},
    ]
    matcher.set_gallery(gallery)

    # Query identical to Alice
    res_alice = matcher.match(vec_alice)
    assert res_alice.status == IdentityStatus.KNOWN
    assert res_alice.person_id == 1
    assert res_alice.name == "Alice"
    assert pytest.approx(res_alice.score, 0.001) == 1.0

    # Query orthogonal to all enrolled subjects (Charlie / stranger)
    vec_charlie = np.zeros(512, dtype=np.float32)
    vec_charlie[2] = 1.0
    res_stranger = matcher.match(vec_charlie)
    assert res_stranger.status == IdentityStatus.UNKNOWN
    assert res_stranger.person_id is None
    assert res_stranger.reason == "below_threshold"
    assert pytest.approx(res_stranger.score, 0.001) == 0.0


def test_matcher_margin_rejection_for_ambiguity():
    """Verify that when 1st and 2nd rank candidates have close scores (< margin), match is rejected."""
    matcher = FaceMatcher(similarity_threshold=0.50, required_margin=0.10)

    # Construct two similar gallery vectors
    vec_person_a = np.zeros(512, dtype=np.float32)
    vec_person_a[0] = 0.8
    vec_person_a[1] = 0.6

    vec_person_b = np.zeros(512, dtype=np.float32)
    vec_person_b[0] = 0.78
    vec_person_b[1] = 0.625

    gallery = [
        {"person_id": 10, "name": "Person A", "category": "staff", "authorised_zone_ids": [], "embedding": vec_person_a},
        {"person_id": 11, "name": "Person B", "category": "staff", "authorised_zone_ids": [], "embedding": vec_person_b},
    ]
    matcher.set_gallery(gallery)

    # Query matches Person A at 0.75 and Person B at 0.72 (gap = 0.03 < 0.10 margin)
    query = np.zeros(512, dtype=np.float32)
    query[0] = 0.79
    query[1] = 0.61
    query = query / np.linalg.norm(query)

    res = matcher.match(query)
    # Both are above 0.50, but margin is too narrow -> ambiguity protection
    assert res.status == IdentityStatus.UNKNOWN
    assert res.reason == "insufficient_margin"
    assert res.margin < 0.10


# ==============================================================================
# 3. Identity State Machine & Timeout Tests (T-05, T-06, T-07)
# ==============================================================================

def test_identity_state_machine_consensus_and_hysteresis():
    """Verify track state requires K matches for KNOWN, and exhibits hysteresis."""
    mgr = IdentityManager(min_consistent_matches=2, min_consistent_stranger_votes=3, timeout_seconds=4.0)

    track_id = 42
    t = 1.0

    # Observation 1: KNOWN vote (1 match) -> stays PENDING
    match_1 = MatchResult(
        status=IdentityStatus.KNOWN,
        person_id=1,
        name="Alice",
        category="staff",
        authorised_zone_ids=["zone_secure"],
        score=0.75,
        margin=0.25,
        reason="matched",
    )
    ident_1 = mgr.update_observation(track_id, t, match_1, quality_passed=True)
    assert ident_1.status == IdentityStatus.PENDING

    # Observation 2: 2nd KNOWN vote -> flips to KNOWN
    t += 0.5
    ident_2 = mgr.update_observation(track_id, t, match_1, quality_passed=True)
    assert ident_2.status == IdentityStatus.KNOWN
    assert ident_2.person_id == 1
    assert ident_2.name == "Alice"

    # Observation 3 (Hysteresis): momentary bad frame (below threshold) does NOT flip back to UNKNOWN
    t += 0.5
    bad_match = MatchResult(
        status=IdentityStatus.UNKNOWN,
        person_id=None,
        name=None,
        category=None,
        authorised_zone_ids=[],
        score=0.35,
        margin=0.05,
        reason="below_threshold",
    )
    ident_3 = mgr.update_observation(track_id, t, bad_match, quality_passed=True)
    assert ident_3.status == IdentityStatus.KNOWN
    assert ident_3.person_id == 1


def test_identity_state_machine_no_usable_face_timeout():
    """Verify track times out to UNKNOWN with 'no_usable_face' when no quality face appears."""
    mgr = IdentityManager(timeout_seconds=3.0)

    track_id = 99
    # Track appears at t=1.0s
    mgr.get_or_create(track_id, timestamp=1.0)
    assert mgr.get_identity(track_id).status == IdentityStatus.PENDING

    # Track lives until t=5.0s (age = 4.0s > 3.0s timeout), but all attempts failed quality
    mgr.record_attempt_without_usable_face(track_id, 2.0, reason="low_resolution")
    mgr.record_attempt_without_usable_face(track_id, 3.5, reason="blurred_face")

    mgr.evaluate_timeouts(active_track_ids=[track_id], current_timestamp=5.0)

    ident = mgr.get_identity(track_id)
    assert ident.status == IdentityStatus.UNKNOWN
    assert ident.reason == "no_usable_face"


# ==============================================================================
# 4. SQLite Face Store & Cascade Deletion (docs/07_DATABASE_SCHEMA.md)
# ==============================================================================

def test_face_store_crud_and_cascade_delete():
    """Verify SQLite storage, vector normalization, and ON DELETE CASCADE purging."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_cctv.db"
        store = FaceStore(db_path=db_path)

        # 1. Add person
        pid = store.add_person(name="Diana Prince", category="staff", authorised_zone_ids=["zone_vault"])
        assert pid == 1
        assert store.count_persons() == 1

        # 2. Add embeddings
        fake_emb = np.random.randn(512).astype(np.float32)
        fake_emb = fake_emb / np.linalg.norm(fake_emb)

        # Create dummy image file
        dummy_img = Path(tmp_dir) / "diana_01.jpg"
        dummy_img.write_text("dummy image data")

        eid = store.add_embedding(person_id=pid, embedding=fake_emb, image_path=str(dummy_img))
        assert eid == 1
        assert store.count_embeddings() == 1

        # Retrieve gallery
        gallery = store.get_gallery()
        assert len(gallery) == 1
        assert gallery[0]["name"] == "Diana Prince"
        assert gallery[0]["authorised_zone_ids"] == ["zone_vault"]
        assert pytest.approx(np.dot(gallery[0]["embedding"], fake_emb), 0.001) == 1.0

        # 3. Test CASCADE DELETE (docs/11_SECURITY_PRIVACY.md Right to remove)
        deleted = store.delete_person(person_id=pid, remove_images=True)
        assert deleted is True

        # Person must be gone
        assert store.count_persons() == 0
        # Embeddings must be cascade-deleted by SQLite foreign key constraint
        assert store.count_embeddings() == 0
        # Image file on disk must be removed
        assert not dummy_img.exists()


# ==============================================================================
# 5. Zone Authorization Check (docs/05_THREAT_ENGINE_SPEC.md)
# ==============================================================================

def test_zone_authorization_logic():
    """Verify KNOWN persons check against authorized zone lists."""
    mgr = IdentityManager(min_consistent_matches=1)

    track_id = 7
    # Enrol person who is authorized for 'server_room' but NOT 'vault'
    match = MatchResult(
        status=IdentityStatus.KNOWN,
        person_id=5,
        name="Security Admin",
        category="staff",
        authorised_zone_ids=["server_room", 102],
        score=0.90,
        margin=0.40,
        reason="matched",
    )
    mgr.update_observation(track_id, timestamp=1.0, match_result=match, quality_passed=True)

    # Allowed zone check
    assert mgr.is_authorised_for_zone(track_id, "server_room") is True
    assert mgr.is_authorised_for_zone(track_id, 102) is True
    assert mgr.is_authorised_for_zone(track_id, "102") is True

    # Unauthorized zone check
    assert mgr.is_authorised_for_zone(track_id, "vault") is False

    # Unenrolled / stranger track
    unknown_track = 8
    assert mgr.is_authorised_for_zone(unknown_track, "server_room") is False


# ==============================================================================
# 6. FaceWorker Bounded Queue Drop Behavior
# ==============================================================================

def test_face_worker_queue_saturation_drop():
    """Verify that FaceWorker drops jobs non-blockingly when queue is full."""
    from src.face.face_worker import FaceWorker

    matcher = FaceMatcher()
    mgr = IdentityManager()
    worker = FaceWorker(matcher=matcher, identity_manager=mgr, queue_size=2)

    dummy_crop = np.zeros((100, 100, 3), dtype=np.uint8)

    # Fill queue of size 2
    assert worker.submit_job(1, 0.1, 1, dummy_crop, (0, 0, 100, 100)) is True
    assert worker.submit_job(2, 0.2, 1, dummy_crop, (0, 0, 100, 100)) is True

    # 3rd job must be dropped non-blockingly without hanging
    assert worker.submit_job(3, 0.3, 1, dummy_crop, (0, 0, 100, 100)) is False

    stats = worker.get_stats()
    assert stats["submitted_count"] == 3
    assert stats["dropped_count"] == 1
    assert stats["drop_rate_pct"] == pytest.approx(33.33, 0.1)
