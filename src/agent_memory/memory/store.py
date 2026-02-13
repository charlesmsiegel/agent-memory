"""SQLite-backed storage with vector search and FTS5 for hybrid retrieval."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from .types import MemoryChunk, MemorySearchResult, MemorySource


class MemoryStore:
    """SQLite store with vector table + FTS5 for hybrid search."""

    def __init__(self, db_path: str | Path) -> None:
        path = Path(db_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._has_sqlite_vec = self._try_load_sqlite_vec()
        self._vec_dims: int | None = None
        self._init_schema()

    # -- Schema --

    def _init_schema(self) -> None:
        c = self._conn
        c.execute(
            """CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                text TEXT NOT NULL,
                source TEXT NOT NULL,
                embedding BLOB,
                hash TEXT
            )"""
        )
        c.execute("CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(hash)")
        # FTS5 virtual table for BM25 keyword search
        c.execute(
            """CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
               USING fts5(text, content=chunks, content_rowid=id)"""
        )
        # Triggers to keep FTS in sync
        c.executescript(
            """
            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
            END;
            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.id, old.text);
            END;
            CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.id, old.text);
                INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
            END;
            """
        )
        # Embedding cache — cross-file deduplication by content hash
        c.execute(
            """CREATE TABLE IF NOT EXISTS embedding_cache (
                hash TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                dims INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            )"""
        )
        # Files table — track path/hash/mtime for delta sync
        c.execute(
            """CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY,
                source TEXT NOT NULL DEFAULT 'memory',
                hash TEXT NOT NULL,
                mtime REAL NOT NULL,
                size INTEGER NOT NULL DEFAULT 0
            )"""
        )
        c.commit()

    # -- Write --

    def upsert_chunks(self, chunks: list[MemoryChunk]) -> None:
        """Insert or replace chunks for a given path. Removes stale chunks first."""
        if not chunks:
            return
        paths = {ch.path for ch in chunks}
        c = self._conn
        for p in paths:
            c.execute("DELETE FROM chunks WHERE path = ?", (p,))
        for ch in chunks:
            blob = _vec_to_blob(ch.embedding) if ch.embedding else None
            c.execute(
                "INSERT INTO chunks (path, start_line, end_line, text, source, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (ch.path, ch.start_line, ch.end_line, ch.text, ch.source.value, blob),
            )
        c.commit()

    def remove_path(self, path: str) -> None:
        """Remove all chunks for a file path."""
        self._conn.execute("DELETE FROM chunks WHERE path = ?", (path,))
        self._conn.commit()

    # -- Vector search --

    def _try_load_sqlite_vec(self) -> bool:
        """Try to load sqlite-vec extension. Returns True if available."""
        try:
            import sqlite_vec  # type: ignore
            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)
            return True
        except Exception:
            return False

    def _ensure_vec_table(self, dims: int) -> None:
        """Create sqlite-vec virtual table if extension is loaded and dims match."""
        if not self._has_sqlite_vec or self._vec_dims == dims:
            return
        self._conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0("
            f"  id TEXT PRIMARY KEY, embedding FLOAT[{dims}])"
        )
        self._vec_dims = dims
        self._conn.commit()

    def vector_search(
        self,
        query_vec: list[float],
        limit: int = 6,
        min_score: float = 0.35,
    ) -> list[MemorySearchResult]:
        """Vector similarity search. Uses sqlite-vec if available, else numpy brute-force."""
        if self._has_sqlite_vec and self._vec_dims:
            return self._vector_search_sqlite_vec(query_vec, limit, min_score)
        return self._vector_search_numpy(query_vec, limit, min_score)

    def _vector_search_sqlite_vec(
        self, query_vec: list[float], limit: int, min_score: float
    ) -> list[MemorySearchResult]:
        """Vector search via sqlite-vec extension."""
        blob = _vec_to_blob(query_vec)
        rows = self._conn.execute(
            "SELECT v.id, v.distance, c.path, c.start_line, c.end_line, c.text, c.source "
            "FROM chunks_vec v "
            "JOIN chunks c ON c.id = CAST(v.id AS INTEGER) "
            "WHERE v.embedding MATCH ? "
            "ORDER BY v.distance LIMIT ?",
            (blob, limit * 2),
        ).fetchall()
        results = []
        for row in rows:
            score = 1.0 / (1.0 + row[1])  # distance to similarity
            if score >= min_score:
                results.append(MemorySearchResult(
                    path=row[2], start_line=row[3], end_line=row[4],
                    score=score, snippet=row[5][:500], source=MemorySource(row[6]),
                ))
        return results[:limit]

    def _vector_search_numpy(
        self, query_vec: list[float], limit: int, min_score: float
    ) -> list[MemorySearchResult]:
        """Brute-force cosine similarity search (fallback)."""
        rows = self._conn.execute(
            "SELECT id, path, start_line, end_line, text, source, embedding FROM chunks "
            "WHERE embedding IS NOT NULL"
        ).fetchall()
        if not rows:
            return []
        q = np.array(query_vec, dtype=np.float64)
        scored: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            stored = _blob_to_vec(row[6])
            score = float(np.dot(q, stored))
            if score >= min_score:
                scored.append((score, row))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            MemorySearchResult(
                path=r[1],
                start_line=r[2],
                end_line=r[3],
                score=s,
                snippet=r[4][:500],
                source=MemorySource(r[5]),
            )
            for s, r in scored[:limit]
        ]

    # -- BM25 search --

    def bm25_search(
        self,
        query: str,
        limit: int = 6,
    ) -> list[MemorySearchResult]:
        """FTS5 BM25 keyword search."""
        rows = self._conn.execute(
            "SELECT c.id, c.path, c.start_line, c.end_line, c.text, c.source, "
            "       rank "
            "FROM chunks_fts f "
            "JOIN chunks c ON c.id = f.rowid "
            "WHERE chunks_fts MATCH ? "
            "ORDER BY rank "
            "LIMIT ?",
            (_fts_escape(query), limit),
        ).fetchall()
        return [
            MemorySearchResult(
                path=r[1],
                start_line=r[2],
                end_line=r[3],
                score=_bm25_rank_to_score(r[6]),
                snippet=r[4][:500],
                source=MemorySource(r[5]),
            )
            for r in rows
        ]

    # -- Metadata --

    def list_indexed_paths(self) -> list[str]:
        """Return all distinct file paths currently indexed."""
        rows = self._conn.execute("SELECT DISTINCT path FROM chunks").fetchall()
        return [r[0] for r in rows]

    def chunk_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return row[0] if row else 0

    def close(self) -> None:
        self._conn.close()

    # -- Embedding cache --

    def get_cached_embeddings(self, hashes: list[str]) -> dict[str, list[float]]:
        """Look up cached embeddings by content hash. Returns {hash: vector}."""
        if not hashes:
            return {}
        result: dict[str, list[float]] = {}
        for h in hashes:
            row = self._conn.execute(
                "SELECT embedding FROM embedding_cache WHERE hash = ?", (h,)
            ).fetchone()
            if row:
                result[h] = _blob_to_vec(row[0]).tolist()
        return result

    def store_cached_embeddings(self, entries: dict[str, list[float]], dims: int) -> None:
        """Store embeddings in cache by content hash."""
        import time
        now = int(time.time())
        for h, vec in entries.items():
            blob = _vec_to_blob(vec)
            self._conn.execute(
                "INSERT OR REPLACE INTO embedding_cache (hash, embedding, dims, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (h, blob, dims, now),
            )
        self._conn.commit()

    def cache_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM embedding_cache").fetchone()
        return row[0] if row else 0

    # -- File tracking for delta sync --

    def get_file_record(self, path: str) -> tuple[str, float, int] | None:
        """Get (hash, mtime, size) for a tracked file, or None."""
        row = self._conn.execute(
            "SELECT hash, mtime, size FROM files WHERE path = ?", (path,)
        ).fetchone()
        return (row[0], row[1], row[2]) if row else None

    def upsert_file_record(self, path: str, source: str, file_hash: str, mtime: float, size: int) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO files (path, source, hash, mtime, size) VALUES (?, ?, ?, ?, ?)",
            (path, source, file_hash, mtime, size),
        )
        self._conn.commit()

    def remove_file_record(self, path: str) -> None:
        self._conn.execute("DELETE FROM files WHERE path = ?", (path,))
        self._conn.commit()

    def list_tracked_files(self) -> list[str]:
        rows = self._conn.execute("SELECT path FROM files").fetchall()
        return [r[0] for r in rows]


# -- Helpers --


def _vec_to_blob(vec: list[float]) -> bytes:
    return np.array(vec, dtype=np.float32).tobytes()


def _blob_to_vec(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).astype(np.float64)


def _bm25_rank_to_score(rank: float) -> float:
    """Convert FTS5 rank (negative, lower = better) to 0-1 score."""
    return 1.0 / (1.0 + abs(rank))


def _fts_escape(query: str) -> str:
    """Escape FTS5 special characters and join as OR terms."""
    tokens = query.split()
    escaped = [t.replace('"', '""') for t in tokens if t.strip()]
    if not escaped:
        return '""'
    return " OR ".join(f'"{t}"' for t in escaped)
