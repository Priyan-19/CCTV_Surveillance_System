"""ChokePoint Automated Enrolment and Open-Set Partition Script.

Partitions ChokePoint subjects into:
- 50% KNOWN (enrolled in database)
- 50% STRANGER (non-enrolled, used for open-set FAR evaluation)

Enrols known subjects using select frames from Portal 1 Session 1 Camera 1.
Generates configs/chokepoint_split.yaml recording:
- Fixed random seed
- Known vs stranger subject lists
- Enrolled frame numbers (strictly quarantined from test evaluation)
- Tuning sequence (P1E_S1_C2)
- Validation sequence (P1E_S2_C1)

Usage:
    python scripts/enroll_from_chokepoint.py --seed 42 --consent-confirmed
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
import cv2
import numpy as np
import yaml

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.face.quality import FaceQualityFilter
from src.face.store import FaceStore

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("enroll_chokepoint")


def load_insightface_app(model_name: str = "buffalo_s"):
    """Initialize InsightFace with CUDA/CPU fallback."""
    try:
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.exists(torch_lib):
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(torch_lib)
            os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass

    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name=model_name, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(320, 320))
    return app


def parse_chokepoint_gt(xml_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Parse ChokePoint XML ground truth into per-subject frame appearances."""
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    subjects: Dict[str, List[Dict[str, Any]]] = {}

    for frame in root.findall("frame"):
        frame_num = frame.attrib["number"]
        for person in frame.findall("person"):
            pid = person.attrib["id"]
            left_eye = person.find("leftEye")
            right_eye = person.find("rightEye")
            if left_eye is not None and right_eye is not None:
                lx, ly = float(left_eye.attrib["x"]), float(left_eye.attrib["y"])
                rx, ry = float(right_eye.attrib["x"]), float(right_eye.attrib["y"])
                if pid not in subjects:
                    subjects[pid] = []
                subjects[pid].append({
                    "frame_num": frame_num,
                    "left_eye": (lx, ly),
                    "right_eye": (rx, ry),
                })
    return subjects


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrol 50% of ChokePoint subjects and configure open-set split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible subject split.")
    parser.add_argument("--known-ratio", type=float, default=0.48, help="Fraction of subjects to enrol (default: ~50%%).")
    parser.add_argument("--images-per-person", type=int, default=4, help="Number of enrolment images per known person.")
    parser.add_argument("--db-path", type=str, default="data/cctv.db", help="Path to SQLite database.")
    parser.add_argument("--consent-confirmed", action="store_true", default=False, help="Consent verification flag.")
    parser.add_argument(
        "--config-out",
        type=str,
        default="configs/chokepoint_split.yaml",
        help="Path to output split configuration file.",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("           CHOKEPOINT 50/50 OPEN-SET ENROLMENT & SPLIT GENERATOR")
    print("=" * 80)

    if not args.consent_confirmed:
        print("\n[CRITICAL PRIVACY VIOLATION] Enrolment rejected!")
        print("Please pass --consent-confirmed to certify compliance with docs/11_SECURITY_PRIVACY.md.")
        sys.exit(1)

    s1_gt_path = Path("data/datasets/chokepoint/groundtruth/P1E_S1_C1.xml")
    s1_img_dir = Path("data/datasets/chokepoint/P1E_S1_C1")

    if not s1_gt_path.exists():
        print(f"[ERROR] Ground truth XML not found: {s1_gt_path}")
        sys.exit(1)
    if not s1_img_dir.exists():
        print(f"[ERROR] Image directory not found: {s1_img_dir}")
        sys.exit(1)

    # 1. Parse ground truth
    print(f"Parsing XML ground truth: {s1_gt_path.name}...")
    subjects_dict = parse_chokepoint_gt(s1_gt_path)
    all_subject_ids = sorted(list(subjects_dict.keys()))
    total_subjects = len(all_subject_ids)
    print(f"Total unique subjects detected in P1E_S1_C1: {total_subjects}")

    # 2. Split into 50% Known and 50% Stranger
    rng = random.Random(args.seed)
    shuffled_ids = list(all_subject_ids)
    rng.shuffle(shuffled_ids)

    n_known = int(round(total_subjects * args.known_ratio))
    known_ids = sorted(shuffled_ids[:n_known])
    stranger_ids = sorted(shuffled_ids[n_known:])

    print(f"\nSplit Configuration (seed={args.seed}):")
    print(f"  KNOWN / Enrolled Subjects ({len(known_ids)})  : {known_ids}")
    print(f"  STRANGER / Non-Enrolled ({len(stranger_ids)}): {stranger_ids}")

    # 3. Load InsightFace and Quality filter
    print("\nInitializing InsightFace (buffalo_s) & Quality Filter...")
    app = load_insightface_app()
    qfilter = FaceQualityFilter(min_face_size=36, min_blur_score=25.0, min_det_score=0.55)
    store = FaceStore(db_path=args.db_path)

    # Purge any previously enrolled ChokePoint benchmark subjects to prevent cross-ID duplicate ambiguity
    for p in store.list_persons():
        if p["name"].startswith("Subject_"):
            store.delete_person(p["id"], remove_images=True)

    # 4. Enrol each Known Subject
    enrolled_manifest: Dict[str, List[str]] = {}
    storage_base = Path("data/known_persons")
    storage_base.mkdir(parents=True, exist_ok=True)

    print("\nEnrolling KNOWN subjects into SQLite FaceStore:")
    for subj_id in known_ids:
        appearances = subjects_dict[subj_id]
        # Sort appearances by eye distance (larger eye distance = subject is closer to camera)
        appearances_sorted = sorted(
            appearances,
            key=lambda a: (a["right_eye"][0] - a["left_eye"][0]) ** 2 + (a["right_eye"][1] - a["left_eye"][1]) ** 2,
            reverse=True,
        )

        valid_for_subj: List[Tuple[np.ndarray, str, np.ndarray]] = []
        # Find args.images_per_person good frames
        for app_info in appearances_sorted:
            fnum = app_info["frame_num"]
            img_file = s1_img_dir / f"{fnum}.jpg"
            if not img_file.exists():
                continue

            frame_bgr = cv2.imread(str(img_file))
            if frame_bgr is None:
                continue

            # Crop around head region based on eye coordinates
            lx, ly = app_info["left_eye"]
            rx, ry = app_info["right_eye"]
            cx = (lx + rx) / 2.0
            cy = (ly + ry) / 2.0
            eye_dist = max(20.0, np.sqrt((rx - lx) ** 2 + (ry - ly) ** 2))
            crop_half = int(eye_dist * 2.2)

            h, w = frame_bgr.shape[:2]
            x1 = max(0, int(cx - crop_half))
            y1 = max(0, int(cy - crop_half * 1.2))
            x2 = min(w, int(cx + crop_half))
            y2 = min(h, int(cy + crop_half * 1.5))
            head_crop = frame_bgr[y1:y2, x1:x2]

            # Detect face with InsightFace
            faces = app.get(head_crop)
            if not faces:
                continue

            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            det_score = float(getattr(face, "det_score", 1.0))
            fx1, fy1, fx2, fy2 = [max(0, int(v)) for v in face.bbox]
            f_crop = head_crop[fy1:fy2, fx1:fx2]

            qres = qfilter.evaluate(f_crop, det_score=det_score)
            if not qres.is_valid:
                continue

            emb = getattr(face, "normed_embedding", None)
            if emb is None:
                emb = getattr(face, "embedding", None)
            if emb is None:
                continue

            valid_for_subj.append((emb, fnum, f_crop))
            if len(valid_for_subj) >= args.images_per_person:
                break

        if len(valid_for_subj) < 3:
            logger.warning("Could not find 3 valid enrolment images for subject %s!", subj_id)
            continue

        # Insert into FaceStore
        person_name = f"Subject_{subj_id}"
        person_id = store.add_person(
            name=person_name,
            category="staff",
            authorised_zone_ids=["portal_1"],
        )

        subj_folder = storage_base / f"chokepoint_{subj_id}"
        subj_folder.mkdir(parents=True, exist_ok=True)
        enrolled_fnums: List[str] = []

        for idx, (emb, fnum, f_crop) in enumerate(valid_for_subj, start=1):
            crop_path = subj_folder / f"face_{idx:02d}_{fnum}.jpg"
            cv2.imwrite(str(crop_path), f_crop)
            store.add_embedding(person_id=person_id, embedding=emb, image_path=str(crop_path))
            enrolled_fnums.append(fnum)

        enrolled_manifest[subj_id] = enrolled_fnums
        print(f"  - Enrolled {person_name:<16} (ID: {person_id:2d}) | {len(enrolled_fnums)} embeddings | Frames: {enrolled_fnums}")

    # 5. Output split configuration
    split_config = {
        "dataset": "ChokePoint",
        "random_seed": args.seed,
        "known_ratio": args.known_ratio,
        "total_subjects": total_subjects,
        "known_subjects": known_ids,
        "stranger_subjects": stranger_ids,
        "enrolled_frames": enrolled_manifest,
        "enrolment_sequence": "P1E_S1_C1",
        "tuning": {
            "sequence": "P1E_S2_C1",
            "camera_id": "P1E_S2_C1",
            "gt_xml": "data/datasets/chokepoint/groundtruth/P1E_S2_C1.xml",
            "images_dir": "data/datasets/chokepoint/P1E_S2_C1",
        },
        "validation": {
            "sequence": "P1E_S3_C1",
            "camera_id": "P1E_S3_C1",
            "gt_xml": "data/datasets/chokepoint/groundtruth/P1E_S3_C1.xml",
            "images_dir": "data/datasets/chokepoint/P1E_S3_C1",
        },
    }

    out_path = Path(args.config_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(split_config, f, default_flow_style=False, sort_keys=False)

    print("-" * 80)
    print(f"[SUCCESS] Split manifest saved to: {out_path}")
    print(f"Total enrolled persons in DB: {store.count_persons()} (Active embeddings: {store.count_embeddings()})")
    print("=" * 80)


if __name__ == "__main__":
    main()
