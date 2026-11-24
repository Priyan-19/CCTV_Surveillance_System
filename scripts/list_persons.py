"""CLI tool to list all enrolled individuals in the Face Store.

Usage:
    python scripts/list_persons.py [--db-path data/cctv.db]
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
    parser = argparse.ArgumentParser(description="List enrolled persons in the CCTV face store.")
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/cctv.db",
        help="Path to SQLite database (default: 'data/cctv.db').",
    )
    args = parser.parse_args()

    store = FaceStore(db_path=args.db_path)
    persons = store.list_persons()

    print("=" * 95)
    print("                           ENROLLED CCTV PERSONS ROSTER")
    print("=" * 95)
    print(f"Database File: {args.db_path}\n")

    if not persons:
        print("[INFO] No enrolled persons found in database.")
        print("To enrol a person, use: python scripts/enroll_person.py --name ... --consent-confirmed")
        print("=" * 95)
        return

    # Header
    print(
        f"{'ID':<6} | {'Name':<24} | {'Category':<12} | {'Embeddings':<10} | {'Authorized Zones':<22} | {'Created At':<12}"
    )
    print("-" * 95)

    for p in persons:
        zid_str = ", ".join(str(z) for z in p["authorised_zone_ids"]) if p["authorised_zone_ids"] else "(None)"
        if len(zid_str) > 22:
            zid_str = zid_str[:19] + "..."
        created = str(p["created_at"])[:16]
        print(
            f"{p['id']:<6} | {p['name']:<24} | {p['category']:<12} | {p['embedding_count']:<10} | {zid_str:<22} | {created:<12}"
        )

    print("-" * 95)
    total_embeddings = store.count_embeddings()
    print(f"Total Enrolled Persons: {len(persons)} | Total Active Embeddings: {total_embeddings}")
    print("=" * 95)


if __name__ == "__main__":
    main()
