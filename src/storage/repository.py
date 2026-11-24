"""Storage Repository layer providing high-level CRUD operations.

Conforms to docs/05_THREAT_ENGINE_SPEC.md, docs/07_DATABASE_SCHEMA.md,
and docs/11_SECURITY_PRIVACY.md.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

from src.common.types import (
    Event,
    EventType,
    Evidence,
    EvidenceType,
    Incident,
    IncidentStatus,
    Severity,
    _sanitize_for_json,
)
from src.storage.schema import check_fts5_supported, init_schema

logger = logging.getLogger("storage.repository")


class StorageRepository:
    """Repository managing surveillance data persistence in SQLite."""

    def __init__(self, db_path: Union[str, Path] = "data/cctv.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_initialized()

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager guaranteeing connection closing, WAL mode, and foreign keys."""
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        try:
            yield conn
        finally:
            conn.close()

    def _ensure_initialized(self) -> None:
        """Initialize database schema if not present."""
        with self._connection() as conn:
            init_schema(conn)

    # =========================================================================
    # Incident Management
    # =========================================================================

    def get_next_incident_no(self) -> str:
        """Generate the next sequential incident number (e.g. 'INC-000001')."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT incident_no FROM incidents ORDER BY id DESC LIMIT 1;")
            row = cursor.fetchone()
            if row and row[0] and "-" in row[0]:
                try:
                    last_num = int(row[0].split("-")[1])
                    return f"INC-{last_num + 1:06d}"
                except (IndexError, ValueError):
                    pass
            cursor.execute("SELECT MAX(id) FROM incidents;")
            row = cursor.fetchone()
            next_id = (row[0] or 0) + 1
            return f"INC-{next_id:06d}"

    def create_incident(self, incident: Incident) -> int:
        """Insert a new incident record and return its integer primary key."""
        if not incident.incident_no:
            incident.incident_no = self.get_next_incident_no()

        keywords = incident.keywords
        if not keywords:
            keywords = f"{incident.incident_no} {incident.camera_id} {incident.place} {incident.threat_level} " + " ".join(incident.activities)

        details_json = json.dumps({
            "score_history": incident.score_history,
            "explanation": incident.explanation,
            "details": incident.details,
            "is_open": incident.is_open,
        }, default=str)

        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO incidents (
                    incident_no, camera_id, place, subject, status,
                    activities, threat_level, threat_score, start_time,
                    last_event_time, closed_at, model_versions, keywords, details
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    incident.incident_no,
                    incident.camera_id,
                    incident.place,
                    json.dumps(_sanitize_for_json(incident.subject), default=str),
                    incident.status,
                    json.dumps(incident.activities, default=str),
                    incident.threat_level,
                    int(incident.threat_score),
                    incident.start_time,
                    incident.last_event_time,
                    incident.closed_at,
                    json.dumps(incident.model_versions, default=str),
                    keywords.strip(),
                    details_json,
                ),
            )
            incident_id = cursor.lastrowid
            conn.commit()
            incident.id = incident_id

            # Write audit action
            cursor.execute(
                """
                INSERT INTO incident_actions (incident_id, user_id, action, note)
                VALUES (?, 1, 'created', 'Incident automatically opened by Threat Engine');
                """,
                (incident_id,),
            )
            conn.commit()
            return incident_id

    def update_incident(self, incident: Incident) -> bool:
        """Update an existing incident with new score, activities, and events."""
        if incident.id is None:
            raise ValueError("Cannot update incident without an id")

        keywords = incident.keywords
        if not keywords:
            keywords = f"{incident.incident_no} {incident.camera_id} {incident.place} {incident.threat_level} " + " ".join(incident.activities)

        details_json = json.dumps({
            "score_history": incident.score_history,
            "explanation": incident.explanation,
            "details": incident.details,
            "is_open": incident.is_open,
        }, default=str)

        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE incidents SET
                    place = ?,
                    subject = ?,
                    status = ?,
                    activities = ?,
                    threat_level = ?,
                    threat_score = ?,
                    last_event_time = ?,
                    closed_at = ?,
                    model_versions = ?,
                    keywords = ?,
                    details = ?
                WHERE id = ?;
                """,
                (
                    incident.place,
                    json.dumps(_sanitize_for_json(incident.subject), default=str),
                    incident.status,
                    json.dumps(incident.activities, default=str),
                    incident.threat_level,
                    int(incident.threat_score),
                    incident.last_event_time,
                    incident.closed_at,
                    json.dumps(incident.model_versions, default=str),
                    keywords.strip(),
                    details_json,
                    incident.id,
                ),
            )
            conn.commit()
            return cursor.rowcount > 0

    def close_incident(self, incident_id: int, closed_at: float) -> bool:
        """Mark an incident as closed due to quiet period expiration."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT details FROM incidents WHERE id = ?;", (incident_id,))
            row = cursor.fetchone()
            if not row:
                return False

            details = json.loads(row["details"] or "{}")
            details["is_open"] = False

            cursor.execute(
                """
                UPDATE incidents SET
                    closed_at = ?,
                    details = ?
                WHERE id = ?;
                """,
                (closed_at, json.dumps(details, default=str), incident_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def get_incident(self, incident_identifier: Union[int, str]) -> Optional[Incident]:
        """Retrieve full incident record by numeric ID or incident_no, including events and evidence."""
        with self._connection() as conn:
            cursor = conn.cursor()
            if isinstance(incident_identifier, int) or str(incident_identifier).isdigit():
                cursor.execute("SELECT * FROM incidents WHERE id = ?;", (int(incident_identifier),))
            else:
                cursor.execute("SELECT * FROM incidents WHERE incident_no = ?;", (str(incident_identifier),))

            row = cursor.fetchone()
            if not row:
                return None

            inc_id = row["id"]
            details_dict = json.loads(row["details"] or "{}")

            incident = Incident(
                id=inc_id,
                incident_no=row["incident_no"],
                camera_id=row["camera_id"],
                place=row["place"],
                subject=json.loads(row["subject"] or "{}"),
                status=row["status"],
                activities=json.loads(row["activities"] or "[]"),
                threat_level=row["threat_level"],
                threat_score=row["threat_score"],
                start_time=float(row["start_time"]),
                last_event_time=float(row["last_event_time"]),
                closed_at=float(row["closed_at"]) if row["closed_at"] is not None else None,
                model_versions=json.loads(row["model_versions"] or "{}"),
                keywords=row["keywords"] or "",
                score_history=details_dict.get("score_history", []),
                explanation=details_dict.get("explanation", []),
                details=details_dict.get("details", {}),
                is_open=details_dict.get("is_open", True),
            )

            # Fetch associated events
            cursor.execute("SELECT * FROM events WHERE incident_id = ? ORDER BY timestamp ASC;", (inc_id,))
            for erow in cursor.fetchall():
                ev = Event(
                    event_id=erow["event_id"],
                    camera_id=erow["camera_id"],
                    track_id=erow["track_id"],
                    event_type=EventType(erow["event_type"]),
                    timestamp=float(erow["timestamp"]),
                    severity=Severity(erow["severity"]),
                    confidence=float(erow["confidence"]),
                    location=json.loads(erow["location"] or "{}"),
                    details=json.loads(erow["details"] or "{}"),
                    person_id=erow["person_id"],
                    object_id=erow["object_id"],
                )
                incident.events.append(ev)

            # Fetch associated evidence
            cursor.execute("SELECT * FROM evidence WHERE incident_id = ? ORDER BY timestamp ASC;", (inc_id,))
            for evrow in cursor.fetchall():
                evi = Evidence(
                    id=evrow["id"],
                    incident_id=evrow["incident_id"],
                    type=evrow["type"],
                    file_path=evrow["file_path"],
                    timestamp=float(evrow["timestamp"]),
                    location=json.loads(evrow["location"] or "{}"),
                    details=json.loads(evrow["details"] or "{}"),
                )
                incident.evidence.append(evi)

            return incident

    def list_incidents(
        self,
        camera_id: Optional[str] = None,
        threat_level: Optional[str] = None,
        status: Optional[str] = None,
        start_time_gte: Optional[float] = None,
        start_time_lte: Optional[float] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Incident]:
        """Query incidents with filtering, sorting, and pagination."""
        query = "SELECT id FROM incidents WHERE 1=1"
        params: List[Any] = []

        if camera_id is not None:
            query += " AND camera_id = ?"
            params.append(camera_id)
        if threat_level is not None:
            query += " AND threat_level = ?"
            params.append(threat_level)
        if status is not None:
            query += " AND status = ?"
            params.append(status)
        if start_time_gte is not None:
            query += " AND start_time >= ?"
            params.append(start_time_gte)
        if start_time_lte is not None:
            query += " AND start_time <= ?"
            params.append(start_time_lte)

        query += " ORDER BY start_time DESC LIMIT ? OFFSET ?;"
        params.extend([limit, offset])

        results: List[Incident] = []
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()

        for r in rows:
            inc = self.get_incident(r["id"])
            if inc is not None:
                results.append(inc)

        return results

    def search_incidents(self, query_text: str, limit: int = 50) -> List[Incident]:
        """Perform full-text search on incidents using FTS5 (or LIKE fallback)."""
        clean_q = query_text.strip()
        if not clean_q:
            return self.list_incidents(limit=limit)

        results: List[Incident] = []
        with self._connection() as conn:
            cursor = conn.cursor()
            fts5_active = check_fts5_supported(conn)

            if fts5_active:
                try:
                    # Sanitize FTS5 query token
                    safe_q = ' '.join(f'"{t}"' for t in clean_q.split() if t)
                    cursor.execute(
                        """
                        SELECT rowid FROM incidents_fts
                        WHERE incidents_fts MATCH ?
                        LIMIT ?;
                        """,
                        (safe_q, limit),
                    )
                    rows = cursor.fetchall()
                    ids = [r[0] for r in rows]
                except sqlite3.OperationalError:
                    ids = []
            else:
                ids = []

            if not ids:
                # LIKE fallback
                like_pat = f"%{clean_q}%"
                cursor.execute(
                    """
                    SELECT id FROM incidents
                    WHERE incident_no LIKE ? OR place LIKE ? OR keywords LIKE ?
                    LIMIT ?;
                    """,
                    (like_pat, like_pat, like_pat, limit),
                )
                ids = [r[0] for r in cursor.fetchall()]

        for inc_id in ids:
            inc = self.get_incident(inc_id)
            if inc is not None:
                results.append(inc)

        return results

    # =========================================================================
    # Camera Registration
    # =========================================================================

    def register_camera(
        self,
        camera_id: str,
        name: str = "",
        source_url: str = "",
        location: str = "",
        status: str = "online",
        occupancy_limit: Optional[int] = None,
    ) -> None:
        """Ensure camera record exists in cameras table."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR IGNORE INTO cameras (id, name, source_url, location, status, occupancy_limit)
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (
                    camera_id,
                    name or f"Camera {camera_id}",
                    source_url or f"rtsp://local/{camera_id}",
                    location or "Default",
                    status,
                    occupancy_limit,
                ),
            )
            conn.commit()

    # =========================================================================
    # Events & Evidence
    # =========================================================================

    def add_event(self, event: Event, incident_id: Optional[int] = None) -> int:
        """Insert a security event into the events table."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO events (
                    event_id, incident_id, camera_id, zone_id, track_id,
                    person_id, object_id, event_type, severity, confidence,
                    timestamp, location, details
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    event.event_id,
                    incident_id,
                    event.camera_id,
                    event.location.get("zone_id") if isinstance(event.location, dict) else None,
                    event.track_id,
                    event.person_id if isinstance(event.person_id, int) else None,
                    event.object_id if isinstance(event.object_id, int) else None,
                    event.event_type.value,
                    event.severity.value,
                    float(event.confidence),
                    float(event.timestamp),
                    json.dumps(_sanitize_for_json(event.location), default=str),
                    json.dumps(_sanitize_for_json(event.details), default=str),
                ),
            )
            event_pk = cursor.lastrowid
            conn.commit()
            return event_pk

    def add_evidence(self, evidence: Evidence) -> int:
        """Insert evidence record referencing an incident."""
        if evidence.incident_id is None:
            raise ValueError("Evidence must be associated with an incident_id")

        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO evidence (
                    incident_id, type, file_path, timestamp, location, details
                ) VALUES (?, ?, ?, ?, ?, ?);
                """,
                (
                    evidence.incident_id,
                    evidence.type,
                    evidence.file_path,
                    evidence.timestamp,
                    json.dumps(_sanitize_for_json(evidence.location), default=str),
                    json.dumps(_sanitize_for_json(evidence.details), default=str),
                ),
            )
            evi_id = cursor.lastrowid
            conn.commit()
            evidence.id = evi_id
            return evi_id

    # =========================================================================
    # Workflow Actions & Status
    # =========================================================================

    def set_status(
        self,
        incident_id: int,
        status: str,
        user_id: int = 1,
        note: Optional[str] = None,
    ) -> bool:
        """Update incident workflow status and append an audit record to incident_actions."""
        valid_statuses = {s.value for s in IncidentStatus}
        if status not in valid_statuses:
            raise ValueError(f"Invalid status '{status}'. Must be one of {valid_statuses}")

        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE incidents SET status = ? WHERE id = ?;", (status, incident_id))
            if cursor.rowcount == 0:
                return False

            cursor.execute(
                """
                INSERT INTO incident_actions (incident_id, user_id, action, note)
                VALUES (?, ?, ?, ?);
                """,
                (incident_id, user_id, status, note),
            )
            conn.commit()
            return True

    # =========================================================================
    # Camera Metrics
    # =========================================================================

    def record_camera_metric(
        self,
        camera_id: str,
        timestamp: float,
        people_count: int,
        fps: float,
        processing_ms: float,
        alert_state: str = "None",
    ) -> int:
        """Insert periodic camera runtime metric."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO camera_metrics (
                    camera_id, timestamp, people_count, fps, processing_ms, alert_state
                ) VALUES (?, ?, ?, ?, ?, ?);
                """,
                (camera_id, timestamp, people_count, fps, processing_ms, alert_state),
            )
            m_id = cursor.lastrowid
            conn.commit()
            return m_id

    # =========================================================================
    # Retention & Pruning (docs/07 Section 5 & docs/11)
    # =========================================================================

    def cleanup_old(
        self,
        retention_days: int = 90,
        metrics_days: int = 30,
        evidence_root: Union[str, Path] = "evidence",
    ) -> Dict[str, int]:
        """Purge incidents older than retention_days, delete physical evidence files, and prune old metrics."""
        now_ts = time.time()
        incident_cutoff = now_ts - (retention_days * 86400.0)
        metrics_cutoff = now_ts - (metrics_days * 86400.0)

        purged_incidents = 0
        purged_evidence_files = 0
        purged_metrics = 0

        evidence_base = Path(evidence_root)

        with self._connection() as conn:
            cursor = conn.cursor()

            # 1. Collect files to delete on disk for expired incidents
            cursor.execute(
                """
                SELECT e.file_path, i.incident_no FROM evidence e
                JOIN incidents i ON e.incident_id = i.id
                WHERE i.start_time < ?;
                """,
                (incident_cutoff,),
            )
            for row in cursor.fetchall():
                fpath = row["file_path"]
                full_path = evidence_base.parent / fpath if not Path(fpath).is_absolute() else Path(fpath)
                if full_path.exists() and full_path.is_file():
                    try:
                        full_path.unlink()
                        purged_evidence_files += 1
                    except Exception as e:
                        logger.warning("Could not delete evidence file %s: %e", full_path, e)

            # 2. Delete expired incidents (cascades to evidence and actions)
            cursor.execute("DELETE FROM incidents WHERE start_time < ?;", (incident_cutoff,))
            purged_incidents = cursor.rowcount

            # 3. Delete expired camera metrics
            cursor.execute("DELETE FROM camera_metrics WHERE timestamp < ?;", (metrics_cutoff,))
            purged_metrics = cursor.rowcount

            conn.commit()

        return {
            "purged_incidents": purged_incidents,
            "purged_evidence_files": purged_evidence_files,
            "purged_metrics": purged_metrics,
        }
