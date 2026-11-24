"""SQLite embeddings and person store conforming to docs/07_DATABASE_SCHEMA.md.

Implements tables:
1. persons: (id, name, category, authorised_zone_ids, created_at)
2. face_embeddings: (id, person_id, embedding, image_path, created_at)
   with FOREIGN KEY(person_id) REFERENCES persons(id) ON DELETE CASCADE.

All embeddings are stored as 512-dimensional float32 L2-normalized BLOBs.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence, Union
import numpy as np

logger = logging.getLogger("face_store")


class FaceStore:
    """Manages SQLite storage for enrolled persons and their 512-d ArcFace embeddings."""

    def __init__(self, db_path: Union[str, Path] = "data/cctv.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Create a new SQLite connection with foreign keys and WAL mode enabled, ensuring clean closure."""
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initialize tables according to docs/07_DATABASE_SCHEMA.md."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS persons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT 'staff',
                    authorised_zone_ids TEXT NOT NULL DEFAULT '[]',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS face_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER NOT NULL,
                    embedding BLOB NOT NULL,
                    image_path TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE
                );
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_face_embeddings_person ON face_embeddings(person_id);"
            )
            conn.commit()

    def add_person(
        self,
        name: str,
        category: str = "staff",
        authorised_zone_ids: Optional[Sequence[Union[int, str]]] = None,
    ) -> int:
        """Insert a new person and return the generated person_id."""
        zones_json = json.dumps(list(authorised_zone_ids) if authorised_zone_ids is not None else [])
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO persons (name, category, authorised_zone_ids)
                VALUES (?, ?, ?)
                """,
                (name.strip(), category.strip(), zones_json),
            )
            conn.commit()
            return cursor.lastrowid

    def add_embedding(
        self,
        person_id: int,
        embedding: np.ndarray,
        image_path: Union[str, Path],
    ) -> int:
        """Store an L2-normalized 512-d float32 embedding vector for an enrolled person."""
        emb = np.asarray(embedding, dtype=np.float32).flatten()
        if emb.shape[0] != 512:
            raise ValueError(f"Expected 512-dimensional embedding, got {emb.shape[0]}")

        # Ensure L2 normalization
        norm = np.linalg.norm(emb)
        if norm > 1e-6:
            emb = emb / norm
        else:
            raise ValueError("Cannot store zero-norm embedding vector.")

        blob = emb.tobytes()
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO face_embeddings (person_id, embedding, image_path)
                VALUES (?, ?, ?)
                """,
                (person_id, blob, str(image_path)),
            )
            conn.commit()
            return cursor.lastrowid

    def get_person(self, person_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve person record by ID."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, name, category, authorised_zone_ids, created_at FROM persons WHERE id = ?",
                (person_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return {
                "id": row["id"],
                "name": row["name"],
                "category": row["category"],
                "authorised_zone_ids": json.loads(row["authorised_zone_ids"]),
                "created_at": row["created_at"],
            }

    def list_persons(self) -> List[Dict[str, Any]]:
        """List all enrolled persons with their embedding counts."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT p.id, p.name, p.category, p.authorised_zone_ids, p.created_at,
                       COUNT(f.id) as embedding_count
                FROM persons p
                LEFT JOIN face_embeddings f ON p.id = f.person_id
                GROUP BY p.id
                ORDER BY p.id ASC
                """
            )
            rows = cursor.fetchall()
            return [
                {
                    "id": row["id"],
                    "name": row["name"],
                    "category": row["category"],
                    "authorised_zone_ids": json.loads(row["authorised_zone_ids"]),
                    "created_at": row["created_at"],
                    "embedding_count": row["embedding_count"],
                }
                for row in rows
            ]

    def get_gallery(self) -> List[Dict[str, Any]]:
        """Retrieve all enrolled embeddings along with person details for fast cosine matching.

        Returns list of dicts:
            {
                'embedding_id': int,
                'person_id': int,
                'name': str,
                'category': str,
                'authorised_zone_ids': list,
                'embedding': np.ndarray (shape=(512,), dtype=np.float32),
                'image_path': str
            }
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT f.id as embedding_id, f.person_id, f.embedding, f.image_path,
                       p.name, p.category, p.authorised_zone_ids
                FROM face_embeddings f
                JOIN persons p ON f.person_id = p.id
                ORDER BY f.person_id ASC
                """
            )
            rows = cursor.fetchall()
            gallery: List[Dict[str, Any]] = []
            for r in rows:
                emb = np.frombuffer(r["embedding"], dtype=np.float32)
                gallery.append(
                    {
                        "embedding_id": r["embedding_id"],
                        "person_id": r["person_id"],
                        "name": r["name"],
                        "category": r["category"],
                        "authorised_zone_ids": json.loads(r["authorised_zone_ids"]),
                        "embedding": emb,
                        "image_path": r["image_path"],
                    }
                )
            return gallery

    def delete_person(self, person_id: int, remove_images: bool = True) -> bool:
        """Delete an enrolled person.

        Enforces SQLite ON DELETE CASCADE to automatically delete associated face_embeddings rows.
        Optionally deletes image files on disk per privacy principle (docs/11_SECURITY_PRIVACY.md).
        """
        with self._connection() as conn:
            cursor = conn.cursor()

            # Retrieve image paths first if file removal requested
            image_paths: List[str] = []
            if remove_images:
                cursor.execute(
                    "SELECT image_path FROM face_embeddings WHERE person_id = ?",
                    (person_id,),
                )
                image_paths = [r["image_path"] for r in cursor.fetchall()]

            cursor.execute("DELETE FROM persons WHERE id = ?", (person_id,))
            deleted = cursor.rowcount > 0
            conn.commit()

        if deleted and remove_images:
            for p in image_paths:
                try:
                    fpath = Path(p)
                    if fpath.is_file():
                        fpath.unlink(missing_ok=True)
                        # Also attempt to remove parent folder if empty
                        parent = fpath.parent
                        if parent.is_dir() and not any(parent.iterdir()):
                            parent.rmdir()
                except OSError as err:
                    logger.warning("Could not delete enrolment file %s: %s", p, err)

        return deleted

    def count_persons(self) -> int:
        """Return total number of enrolled persons."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM persons")
            return int(cursor.fetchone()[0])

    def count_embeddings(self) -> int:
        """Return total number of enrolled face embeddings."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM face_embeddings")
            return int(cursor.fetchone()[0])
