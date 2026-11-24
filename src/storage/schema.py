"""Database schema definitions and DDL conforming to docs/07_DATABASE_SCHEMA.md.

Defines all relational tables, foreign key constraints (PRAGMA foreign_keys = ON),
performance indexes, full-text search (FTS5 / LIKE fallback), and system user seeding.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Tuple

logger = logging.getLogger("storage.schema")

# Core table definitions
SCHEMA_TABLES_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'officer',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS cameras (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    source_url TEXT NOT NULL,
    location TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'online',
    occupancy_limit INTEGER NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS zones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id TEXT NOT NULL,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    geometry TEXT NOT NULL,
    direction TEXT NULL,
    dwell_threshold_s INTEGER NULL,
    high_risk INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (camera_id) REFERENCES cameras(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS persons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'staff',
    authorised_zone_ids TEXT NOT NULL DEFAULT '[]',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS face_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER NOT NULL,
    embedding BLOB NOT NULL,
    image_path TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS incidents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    incident_no TEXT UNIQUE NOT NULL,
    camera_id TEXT NOT NULL,
    place TEXT NOT NULL,
    subject TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'new',
    activities TEXT NOT NULL DEFAULT '[]',
    threat_level TEXT NOT NULL,
    threat_score INTEGER NOT NULL,
    start_time DATETIME NOT NULL,
    last_event_time DATETIME NOT NULL,
    closed_at DATETIME NULL,
    model_versions TEXT NOT NULL DEFAULT '{}',
    keywords TEXT NOT NULL DEFAULT '',
    details TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (camera_id) REFERENCES cameras(id)
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT UNIQUE NOT NULL,
    incident_id INTEGER NULL,
    camera_id TEXT NOT NULL,
    zone_id INTEGER NULL,
    track_id INTEGER NULL,
    person_id INTEGER NULL,
    object_id INTEGER NULL,
    event_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    confidence REAL NOT NULL,
    timestamp DATETIME NOT NULL,
    location TEXT NOT NULL DEFAULT '{}',
    details TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (incident_id) REFERENCES incidents(id) ON DELETE SET NULL,
    FOREIGN KEY (camera_id) REFERENCES cameras(id),
    FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS evidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    incident_id INTEGER NOT NULL,
    type TEXT NOT NULL,
    file_path TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    location TEXT NOT NULL DEFAULT '{}',
    details TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (incident_id) REFERENCES incidents(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS incident_actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    incident_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    action TEXT NOT NULL,
    note TEXT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (incident_id) REFERENCES incidents(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS camera_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    people_count INTEGER NOT NULL DEFAULT 0,
    fps REAL NOT NULL DEFAULT 0.0,
    processing_ms REAL NOT NULL DEFAULT 0.0,
    alert_state TEXT NOT NULL DEFAULT 'None',
    FOREIGN KEY (camera_id) REFERENCES cameras(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    description TEXT NOT NULL
);
"""

# Indexes specified in docs/07_DATABASE_SCHEMA.md Section 3
SCHEMA_INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_incidents_no ON incidents(incident_no);
CREATE INDEX IF NOT EXISTS idx_incidents_level_start ON incidents(threat_level, start_time);
CREATE INDEX IF NOT EXISTS idx_incidents_cam_start ON incidents(camera_id, start_time);
CREATE INDEX IF NOT EXISTS idx_events_incident ON events(incident_id);
CREATE INDEX IF NOT EXISTS idx_events_cam_time ON events(camera_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_events_track ON events(track_id);
CREATE INDEX IF NOT EXISTS idx_evidence_incident ON evidence(incident_id);
CREATE INDEX IF NOT EXISTS idx_incident_actions_inc ON incident_actions(incident_id);
CREATE INDEX IF NOT EXISTS idx_camera_metrics_cam_time ON camera_metrics(camera_id, timestamp);
"""

# FTS5 Full-Text Search Virtual Table & Triggers
FTS5_SETUP_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS incidents_fts USING fts5(
    incident_no,
    place,
    keywords,
    content='incidents',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS trg_incidents_ai AFTER INSERT ON incidents BEGIN
    INSERT INTO incidents_fts(rowid, incident_no, place, keywords)
    VALUES (new.id, new.incident_no, new.place, new.keywords);
END;

CREATE TRIGGER IF NOT EXISTS trg_incidents_ad AFTER DELETE ON incidents BEGIN
    INSERT INTO incidents_fts(incidents_fts, rowid, incident_no, place, keywords)
    VALUES ('delete', old.id, old.incident_no, old.place, old.keywords);
END;

CREATE TRIGGER IF NOT EXISTS trg_incidents_au AFTER UPDATE ON incidents BEGIN
    INSERT INTO incidents_fts(incidents_fts, rowid, incident_no, place, keywords)
    VALUES ('delete', old.id, old.incident_no, old.place, old.keywords);
    INSERT INTO incidents_fts(rowid, incident_no, place, keywords)
    VALUES (new.id, new.incident_no, new.place, new.keywords);
END;
"""


def check_fts5_supported(conn: sqlite3.Connection) -> bool:
    """Verify whether FTS5 virtual table extension is enabled in this SQLite build."""
    try:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS _test_fts5 USING fts5(x);")
        conn.execute("DROP TABLE IF EXISTS _test_fts5;")
        return True
    except sqlite3.OperationalError:
        return False


def seed_system_user(conn: sqlite3.Connection) -> None:
    """Seed one non-login system user for automated threat engine and retention actions."""
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR IGNORE INTO users (id, username, password_hash, role)
        VALUES (1, 'system', '$SYSTEM_NO_LOGIN$', 'admin');
        """
    )
    conn.commit()


def init_schema(conn: sqlite3.Connection) -> Tuple[bool, bool]:
    """Initialize all tables, indexes, FTS5 virtual tables, and seed data.

    Returns:
        Tuple of (schema_initialized: bool, fts5_enabled: bool).
    """
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")

    conn.executescript(SCHEMA_TABLES_SQL)
    conn.executescript(SCHEMA_INDEXES_SQL)
    seed_system_user(conn)

    fts5_active = check_fts5_supported(conn)
    if fts5_active:
        conn.executescript(FTS5_SETUP_SQL)
        logger.info("SQLite FTS5 full-text search successfully enabled for incidents.")
    else:
        logger.warning("SQLite FTS5 not available in this Python build; falling back to LIKE keyword search.")

    conn.commit()
    return True, fts5_active
