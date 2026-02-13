"""Conversational fact store — discrete facts with categories and dedup."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from enum import Enum
from pathlib import Path
from uuid import uuid4

import numpy as np

from .embeddings import BedrockEmbeddings


class FactCategory(str, Enum):
    PREFERENCE = "preference"
    DECISION = "decision"
    ENTITY = "entity"
    FACT = "fact"
    OTHER = "other"


class Fact:
    """A single stored fact."""

    __slots__ = ("id", "text", "category", "importance", "created_at")

    def __init__(
        self,
        id: str,
        text: str,
        category: FactCategory,
        importance: float,
        created_at: datetime,
    ) -> None:
        self.id = id
        self.text = text
        self.category = category
        self.importance = importance
        self.created_at = created_at


class FactRecallResult:
    __slots__ = ("fact", "score")

    def __init__(self, fact: Fact, score: float) -> None:
        self.fact = fact
        self.score = score


_DEDUP_THRESHOLD = 0.95


class FactStore:
    """SQLite-backed store for conversational facts with vector dedup."""

    def __init__(self, db_path: str | Path, embeddings: BedrockEmbeddings) -> None:
        path = Path(db_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._emb = embeddings
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS facts (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                category TEXT NOT NULL,
                importance REAL NOT NULL DEFAULT 0.7,
                embedding BLOB NOT NULL,
                created_at TEXT NOT NULL
            )"""
        )
        self._conn.commit()

    def store(
        self,
        text: str,
        *,
        category: FactCategory = FactCategory.OTHER,
        importance: float = 0.7,
    ) -> Fact | None:
        """Store a fact. Returns None if a duplicate exists (>= 0.95 similarity)."""
        vec = self._emb.embed(text)
        existing = self._find_similar(vec, threshold=_DEDUP_THRESHOLD)
        if existing:
            return None
        fact_id = uuid4().hex[:12]
        now = datetime.utcnow()
        blob = np.array(vec, dtype=np.float32).tobytes()
        self._conn.execute(
            "INSERT INTO facts (id, text, category, importance, embedding, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (fact_id, text, category.value, importance, blob, now.isoformat()),
        )
        self._conn.commit()
        return Fact(id=fact_id, text=text, category=category, importance=importance, created_at=now)

    def recall(
        self,
        query: str,
        *,
        limit: int = 5,
        min_score: float = 0.3,
    ) -> list[FactRecallResult]:
        """Search facts by semantic similarity."""
        vec = self._emb.embed(query)
        return self._search(vec, limit=limit, min_score=min_score)

    def forget(self, fact_id: str) -> bool:
        """Delete a fact by ID."""
        cur = self._conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def forget_by_query(self, query: str, *, min_score: float = 0.9) -> list[str]:
        """Find and delete facts matching a query. Returns deleted IDs."""
        results = self.recall(query, limit=5, min_score=min_score)
        deleted: list[str] = []
        for r in results:
            if self.forget(r.fact.id):
                deleted.append(r.fact.id)
        return deleted

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM facts").fetchone()
        return row[0] if row else 0

    def close(self) -> None:
        self._conn.close()

    # -- Internal --

    def _find_similar(self, vec: list[float], threshold: float) -> Fact | None:
        results = self._search(vec, limit=1, min_score=threshold)
        return results[0].fact if results else None

    def _search(
        self, vec: list[float], limit: int, min_score: float
    ) -> list[FactRecallResult]:
        rows = self._conn.execute(
            "SELECT id, text, category, importance, embedding, created_at FROM facts"
        ).fetchall()
        if not rows:
            return []
        q = np.array(vec, dtype=np.float64)
        scored: list[FactRecallResult] = []
        for row in rows:
            stored = np.frombuffer(row[4], dtype=np.float32).astype(np.float64)
            score = float(np.dot(q, stored))
            if score >= min_score:
                fact = Fact(
                    id=row[0],
                    text=row[1],
                    category=FactCategory(row[2]),
                    importance=row[3],
                    created_at=datetime.fromisoformat(row[5]),
                )
                scored.append(FactRecallResult(fact=fact, score=score))
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:limit]
