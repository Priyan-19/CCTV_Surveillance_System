"""CLI script to enrol a known person conforming to docs/11_SECURITY_PRIVACY.md.

STRICT PRIVACY POLICY:
- Enrolment is strictly voluntary and requires explicit consent (--consent-confirmed).
- At least 3 quality face images are required per enrolled person (docs/06_ML_MODEL_PLAN.md).
- Images are stored in data/known_persons/<person_id>_<name>/ and embeddings in SQLite.

Usage:
    python scripts/enroll_person.py --name "Alice Smith" --category staff --zones "Deep Corridor,Zone_A" --images-dir path/to/images --consent-confirmed
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.face.quality import FaceQualityFilter
from src.face.store import FaceStore

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("enroll_person")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrol a known individual with ArcFace embeddings and explicit consent check."
    )
    parser.add_argument("--name", type=str, required=True, help="Full name of person to enrol.")
    parser.add_argument(
        "--category",
        type=str,
        default="staff",
        choices=["staff", "visitor", "contractor", "vip", "watchlist"],
        help="Access category (default: 'staff').",
    )
    parser.add_argument(
        "--zones",
        type=str,
        default="",
        help="Comma-separated authorized zone names or IDs (e.g. 'Deep Corridor,Emergency Exit Passage').",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        required=True,
        help="Directory containing enrolment face images (JPG/PNG).",
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=3,
        help="Minimum number of quality face images required (default: 3).",
    )
    parser.add_argument(
        "--consent-confirmed",
        action="store_true",
        default=False,
        help="Mandatory flag confirming the individual provided explicit consent for CCTV biometric enrolment.",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/cctv.db",
        help="Path to SQLite database (default: 'data/cctv.db').",
    )
    parser.add_argument(
        "--storage-dir",
        type=str,
        default="data/known_persons",
        help="Base folder to store enrolment images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 80)
    print("                     BIOMETRIC FACE ENROLMENT TOOL")
    print("=" * 80)

    # 1. Strict Privacy & Consent Validation (docs/11_SECURITY_PRIVACY.md Section 4)
    if not args.consent_confirmed:
        print("\n[CRITICAL PRIVACY VIOLATION] Enrolment rejected!")
        print("Under docs/11_SECURITY_PRIVACY.md (Principle 4: 'Consent for enrolment'),")
        print("individuals may only be enrolled into the biometric database with informed consent.")
        print("To proceed, you must pass the --consent-confirmed flag.")
        sys.exit(1)

    # 2. Collect input images
    img_dir = Path(args.images_dir)
    if not img_dir.is_dir():
        print(f"[ERROR] Images directory not found: {img_dir}")
        sys.exit(1)

    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    img_files = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in extensions])
    if len(img_files) < args.min_images:
        print(
            f"[ERROR] Found only {len(img_files)} images in {img_dir}. At least {args.min_images} required."
        )
        sys.exit(1)

    print(f"Candidate images found : {len(img_files)}")
    print(f"Enrollee Name          : {args.name}")
    print(f"Category               : {args.category}")
    auth_zones = [z.strip() for z in args.zones.split(",") if z.strip()]
    print(f"Authorized Zones       : {auth_zones if auth_zones else 'None'}")
    print(f"Database Path          : {args.db_path}")

    # 3. Load model and quality filter
    print("\nLoading InsightFace detector & ArcFace embedding model...")
    app = load_insightface_app()
    qfilter = FaceQualityFilter(min_face_size=40, min_blur_score=25.0, min_det_score=0.60)

    # 4. Process images and extract embeddings
    valid_embeddings: List[Tuple[np.ndarray, Path]] = []
    print("\nEvaluating face image quality and extracting embeddings:")
    for img_path in img_files:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  - {img_path.name:<25}: FAILED (could not read image)")
            continue

        faces = app.get(img_bgr)
        if not faces:
            print(f"  - {img_path.name:<25}: FAILED (no face detected)")
            continue

        # Take largest face
        faces_sorted = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True,
        )
        face = faces_sorted[0]
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        h, w = img_bgr.shape[:2]
        crop = img_bgr[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

        qres = qfilter.evaluate(crop, det_score=float(getattr(face, "det_score", 1.0)))
        if not qres.is_valid:
            print(f"  - {img_path.name:<25}: REJECTED ({qres.reason})")
            continue

        emb = getattr(face, "normed_embedding", None)
        if emb is None:
            emb = getattr(face, "embedding", None)
        if emb is None:
            print(f"  - {img_path.name:<25}: FAILED (no embedding vector)")
            continue

        valid_embeddings.append((emb, img_path))
        print(f"  - {img_path.name:<25}: ACCEPTED ({qres.dimensions}, blur={qres.blur_score:.1f})")

    if len(valid_embeddings) < args.min_images:
        print(
            f"\n[ERROR] Only {len(valid_embeddings)} images passed quality filtering. "
            f"At least {args.min_images} valid face images are required for robust enrolment."
        )
        sys.exit(1)

    # 5. Persist person and embeddings to SQLite
    store = FaceStore(db_path=args.db_path)
    person_id = store.add_person(
        name=args.name,
        category=args.category,
        authorised_zone_ids=auth_zones,
    )

    # Copy accepted images to data/known_persons/<id>_<sanitized_name>/
    clean_name = "".join(c if c.isalnum() else "_" for c in args.name).strip("_")
    target_person_dir = Path(args.storage_dir) / f"{person_id}_{clean_name}"
    target_person_dir.mkdir(parents=True, exist_ok=True)

    stored_count = 0
    for idx, (emb, src_path) in enumerate(valid_embeddings, start=1):
        target_img_name = f"face_{idx:02d}{src_path.suffix.lower()}"
        dest_path = target_person_dir / target_img_name
        shutil.copy2(src_path, dest_path)

        store.add_embedding(
            person_id=person_id,
            embedding=emb,
            image_path=str(dest_path),
        )
        stored_count += 1

    print("-" * 80)
    print(f"[SUCCESS] Enrolled {args.name} successfully!")
    print(f"  Person ID            : {person_id}")
    print(f"  Category             : {args.category}")
    print(f"  Authorized Zones     : {auth_zones}")
    print(f"  Stored Embeddings    : {stored_count}")
    print(f"  Image Archive        : {target_person_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
