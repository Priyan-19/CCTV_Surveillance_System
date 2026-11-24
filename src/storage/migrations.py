"""Schema versioning and migration manager conforming to docs/07_DATABASE_SCHEMA.md.

Applies incremental schema updates and verifies structural integrity.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

from src.storage.schema import init_schema

logger = logging.getLogger("storage.migrations")

# Catalog of versioned migrations
MIGRATIONS: Dict[int, str] = {
    1: "Phase 4 Initial Unified Schema (users, cameras, zones, persons, embeddings, events, incidents, evidence, actions, metrics, fts5)",
}


def get_current_version(conn: sqlite3.Connection) -> int:
    """Query highest applied schema version, or 0 if uninitialized."""
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(version) FROM schema_version;")
        row = cursor.fetchone()
        if row and row[0] is not None:
            return int(row[0])
        return 0
    except sqlite3.OperationalError:
        return 0


def apply_migrations(db_path: Path) -> Tuple[int, bool]:
    """Execute pending schema migrations in an atomic transaction.

    Returns:
        Tuple of (new_schema_version: int, fts5_supported: bool).
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path), timeout=15.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")

    try:
        current_v = get_current_version(conn)
        fts5_supported = False

        if current_v < 1:
            logger.info("Applying Migration 1: Initial unified surveillance schema...")
            _, fts5_supported = init_schema(conn)

            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO schema_version (version, description)
                VALUES (1, ?);
                """,
                (MIGRATIONS[1],),
            )
            conn.commit()
            logger.info("Migration 1 successfully applied.")
            current_v = 1
        else:
            from src.storage.schema import check_fts5_supported
            fts5_supported = check_fts5_supported(conn)

        return current_v, fts5_supported
    finally:
        conn.close()
