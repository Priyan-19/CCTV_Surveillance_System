"""Cosine similarity face matcher with threshold and margin enforcement.

Performs open-set 1:N face identification against enrolled ArcFace embeddings.
Implements:
1. Cosine similarity calculation over L2-normalized 512-d feature vectors.
2. Per-person aggregation (max similarity across multiple enrolment shots).
3. Ambiguity rejection via margin check (similarity(1st) - similarity(2nd) >= margin).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
import numpy as np

from src.common.types import IdentityStatus


@dataclass
class MatchResult:
    """Outcome of 1:N face identification."""

    status: IdentityStatus
    person_id: Optional[int]
    name: Optional[str]
    category: Optional[str]
    authorised_zone_ids: List[Union[int, str]]
    score: float
    margin: float
    reason: str
    second_name: Optional[str] = None
    second_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "person_id": self.person_id,
            "name": self.name,
            "category": self.category,
            "authorised_zone_ids": self.authorised_zone_ids,
            "score": round(self.score, 4),
            "margin": round(self.margin, 4),
            "reason": self.reason,
            "second_name": self.second_name,
            "second_score": round(self.second_score, 4),
        }


class FaceMatcher:
    """Matches query face embeddings against an in-memory or database gallery."""

    def __init__(
        self,
        similarity_threshold: float = 0.45,
        required_margin: float = 0.08,
    ) -> None:
        """Initialize matcher.

        Args:
            similarity_threshold: Minimum cosine similarity required to label as KNOWN.
            required_margin: Minimum score separation between rank-1 and rank-2 identities.
        """
        self.similarity_threshold = float(similarity_threshold)
        self.required_margin = float(required_margin)
        self._gallery: List[Dict[str, Any]] = []
        self._gallery_matrix: Optional[np.ndarray] = None  # Shape (N, 512)

    def set_gallery(self, gallery: Sequence[Dict[str, Any]]) -> None:
        """Load or refresh gallery embeddings.

        Each gallery item must contain:
            'person_id': int,
            'name': str,
            'category': str,
            'authorised_zone_ids': list,
            'embedding': np.ndarray (512,)
        """
        self._gallery = list(gallery)
        if self._gallery:
            embs = [item["embedding"] for item in self._gallery]
            matrix = np.vstack(embs).astype(np.float32)
            # Ensure row-wise L2 normalization
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms < 1e-6] = 1.0
            self._gallery_matrix = matrix / norms
        else:
            self._gallery_matrix = None

    def match(self, query_embedding: np.ndarray) -> MatchResult:
        """Perform 1:N cosine similarity match for a single query embedding.

        Args:
            query_embedding: 512-d float32 vector.

        Returns:
            MatchResult with status KNOWN or UNKNOWN, score, margin, and rationale.
        """
        if self._gallery_matrix is None or len(self._gallery) == 0:
            return MatchResult(
                status=IdentityStatus.UNKNOWN,
                person_id=None,
                name=None,
                category=None,
                authorised_zone_ids=[],
                score=0.0,
                margin=0.0,
                reason="empty_gallery",
            )

        q = np.asarray(query_embedding, dtype=np.float32).flatten()
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-6:
            return MatchResult(
                status=IdentityStatus.UNKNOWN,
                person_id=None,
                name=None,
                category=None,
                authorised_zone_ids=[],
                score=0.0,
                margin=0.0,
                reason="zero_norm_query",
            )
        q = q / q_norm

        # Vectorized dot products across all gallery embeddings
        similarities = np.dot(self._gallery_matrix, q)  # Shape (N,)

        # Group by person_id, taking maximum similarity per person
        person_best: Dict[int, Dict[str, Any]] = {}
        for idx, sim in enumerate(similarities):
            item = self._gallery[idx]
            pid = item["person_id"]
            if pid not in person_best or sim > person_best[pid]["score"]:
                person_best[pid] = {
                    "score": float(sim),
                    "name": item["name"],
                    "category": item.get("category", "staff"),
                    "authorised_zone_ids": item.get("authorised_zone_ids", []),
                }

        # Rank candidates by descending score
        ranked = sorted(person_best.items(), key=lambda kv: kv[1]["score"], reverse=True)

        top_pid, top_info = ranked[0]
        top_score = top_info["score"]

        # Check second rank for margin calculation
        if len(ranked) > 1:
            second_pid, second_info = ranked[1]
            second_score = second_info["score"]
            second_name = second_info["name"]
            margin = top_score - second_score
        else:
            second_score = 0.0
            second_name = None
            margin = top_score

        # Evaluation against threshold and margin
        if top_score < self.similarity_threshold:
            return MatchResult(
                status=IdentityStatus.UNKNOWN,
                person_id=None,
                name=None,
                category=None,
                authorised_zone_ids=[],
                score=top_score,
                margin=margin,
                reason="below_threshold",
                second_name=second_name,
                second_score=second_score,
            )

        if len(ranked) > 1 and margin < self.required_margin:
            return MatchResult(
                status=IdentityStatus.UNKNOWN,
                person_id=None,
                name=None,
                category=None,
                authorised_zone_ids=[],
                score=top_score,
                margin=margin,
                reason="insufficient_margin",
                second_name=second_name,
                second_score=second_score,
            )

        return MatchResult(
            status=IdentityStatus.KNOWN,
            person_id=top_pid,
            name=top_info["name"],
            category=top_info["category"],
            authorised_zone_ids=top_info["authorised_zone_ids"],
            score=top_score,
            margin=margin,
            reason="matched",
            second_name=second_name,
            second_score=second_score,
        )
