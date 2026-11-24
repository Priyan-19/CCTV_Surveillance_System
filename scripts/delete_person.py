"""CLI tool to delete an enrolled person and purge associated biometrics.

Enforces:
1. SQLite ON DELETE CASCADE removing all associated rows in face_embeddings.
2. Purges biometric face crop images from disk (docs/11_SECURITY_PRIVACY.md Principle 6: 'Right to remove').

Usage:
    python scripts/delete_person.py --person-id 1 [--yes]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.face.store import FaceStore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Delete an enrolled person and cascade purge their biometric embeddings."
    )
    parser.add_argument(
        "--person-id",
        type=int,
        required=True,
        help="Numeric ID of the enrolled person to delete.",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/cctv.db",
        help="Path to SQLite database (default: 'data/cctv.db').",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        default=False,
        help="Confirm deletion without interactive prompt.",
    )
    args = parser.parse_args()

    store = FaceStore(db_path=args.db_path)
    person = store.get_person(args.person_id)

    if person is None:
        print(f"[ERROR] No enrolled person found with ID {args.person_id} in {args.db_path}.")
        sys.exit(1)

    print("=" * 80)
    print("                     BIOMETRIC DATA PURGE CONFIRMATION")
    print("=" * 80)
    print(f"Target Person ID : {person['id']}")
    print(f"Name             : {person['name']}")
    print(f"Category         : {person['category']}")
    print(f"Authorized Zones : {person['authorised_zone_ids']}")
    print("-" * 80)
    print("WARNING: This will permanently delete the person record, cascade-delete all")
    print("associated face embeddings in the database, and purge all saved enrolment images.")

    if not args.yes:
        confirm = input("\nType 'yes' or 'y' to confirm permanent deletion: ").strip().lower()
        if confirm not in ("yes", "y"):
            print("[CANCELLED] Deletion aborted by user.")
            sys.exit(0)

    # Perform cascade delete
    success = store.delete_person(args.person_id, remove_images=True)
    if success:
        print(f"\n[SUCCESS] Person ID {args.person_id} ('{person['name']}') and all biometric records deleted.")
    else:
        print(f"\n[ERROR] Failed to delete person ID {args.person_id}.")
        sys.exit(1)


if __name__ == "__main__":
    main()
