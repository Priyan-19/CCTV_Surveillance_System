"""Database Initialization and Migration Utility.

Conforms to docs/07_DATABASE_SCHEMA.md.

Initializes all tables, indexes, FTS5 virtual search, seeds the system user,
and applies schema migrations.

Usage:
    python scripts/init_db.py
    python scripts/init_db.py --db-path data/cctv.db
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.storage.migrations import apply_migrations
from src.storage.repository import StorageRepository
from src.storage.schema import check_fts5_supported

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("init_db")


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize and migrate surveillance SQLite database.")
    parser.add_argument("--db-path", type=str, default="data/cctv.db", help="Path to SQLite database file.")
    args = parser.parse_args()

    db_path = Path(args.db_path)
    print("=" * 80)
    print("             INTELLIGENT CCTV SURVEILLANCE DATABASE INITIALIZER")
    print("=" * 80)
    print(f"Target Database File : {db_path.resolve()}")
    print("-" * 80)

    # 1. Apply Migrations & Initialize Schema
    version, fts5_supported = apply_migrations(db_path)

    # 2. Inspect created tables
    repo = StorageRepository(db_path=db_path)
    with repo._connection() as conn:
        cursor = conn.cursor()

        # Check foreign keys
        cursor.execute("PRAGMA foreign_keys;")
        fk_enabled = bool(cursor.fetchone()[0])

        # Check journal mode
        cursor.execute("PRAGMA journal_mode;")
        journal_mode = cursor.fetchone()[0].upper()

        # List tables
        cursor.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name;
            """
        )
        tables = [row[0] for row in cursor.fetchall()]

        # Count system user
        cursor.execute("SELECT id, username, role FROM users WHERE username = 'system';")
        sys_user = cursor.fetchone()

        # Query existing incident count
        cursor.execute("SELECT COUNT(*) FROM incidents;")
        inc_count = cursor.fetchone()[0]

    print("[SUCCESS] Database schema initialized and migrated successfully.")
    print(f"  * Schema Version       : v{version}")
    print(f"  * Foreign Keys Active  : {'YES (PRAGMA foreign_keys = ON)' if fk_enabled else 'NO'}")
    print(f"  * Journal Mode         : {journal_mode}")
    print(f"  * FTS5 Search Engine   : {'ACTIVE (Virtual Table & Triggers)' if fts5_supported else 'UNAVAILABLE (Falling back to LIKE)'}")
    print(f"  * System User Seeded   : User ID {sys_user['id']} ('{sys_user['username']}', role='{sys_user['role']}')")
    print(f"  * Next Incident ID     : {repo.get_next_incident_no()}")
    print(f"  * Current Incidents    : {inc_count}")
    print("-" * 80)
    print(f"Total Tables Created ({len(tables)}):")
    for idx, tbl in enumerate(tables, start=1):
        print(f"  {idx:2d}. {tbl}")
    print("=" * 80)


if __name__ == "__main__":
    main()
