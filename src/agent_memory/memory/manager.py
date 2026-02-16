"""Memory manager — hybrid search, file indexing, and sync."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

try:
    from .embeddings import BedrockEmbeddings
except ImportError:
    BedrockEmbeddings = None  # type: ignore[assignment,misc]
from .store import MemoryStore
from .types import MemoryChunk, MemorySearchResult, MemorySource

log = logging.getLogger(__name__)

# Rough chars-per-token estimate for chunking
_CHARS_PER_TOKEN = 4


class MemoryManager:
    """Indexes workspace files and provides hybrid vector + BM25 search."""

    def __init__(
        self,
        store: MemoryStore,
        embeddings: BedrockEmbeddings,
        *,
        chunk_tokens: int = 400,
        chunk_overlap: int = 80,
        vector_weight: float = 0.7,
        text_weight: float = 0.3,
        max_results: int = 6,
        min_score: float = 0.35,
    ) -> None:
        self._store = store
        self._emb = embeddings
        self._chunk_tokens = chunk_tokens
        self._chunk_overlap = chunk_overlap
        self._vector_weight = vector_weight
        self._text_weight = text_weight
        self._max_results = max_results
        self._min_score = min_score
        self._dirty = False
        self._watcher = None
        self._watched_dirs: list[str | Path] = []
        self._watched_session_dirs: list[str | Path] = []

    # -- Search --

    def search(
        self,
        query: str,
        *,
        max_results: int | None = None,
        min_score: float | None = None,
    ) -> list[MemorySearchResult]:
        """Hybrid search: merge vector similarity and BM25 keyword results."""
        # Auto-sync if files changed since last sync
        if self._dirty and self._watched_dirs:
            self.sync(self._watched_dirs, session_dirs=self._watched_session_dirs)

        limit = max_results or self._max_results
        floor = min_score if min_score is not None else self._min_score
        candidate_limit = limit * 4

        query_vec = self._emb.embed(query)
        vec_results = self._store.vector_search(query_vec, limit=candidate_limit, min_score=0.0)
        bm25_results = self._store.bm25_search(query, limit=candidate_limit)

        merged = _merge_results(
            vec_results,
            bm25_results,
            vector_weight=self._vector_weight,
            text_weight=self._text_weight,
        )
        return [r for r in merged if r.score >= floor][:limit]

    # -- Indexing --

    def index_file(self, file_path: str | Path, source: MemorySource = MemorySource.MEMORY) -> int:
        """Read, chunk, embed (with cache), and store a single file. Returns chunk count."""
        path = Path(file_path)
        if not path.is_file():
            log.warning("index_file: not a file: %s", path)
            return 0

        # Delta check — skip if file unchanged
        stat = path.stat()
        file_hash = _hash_file(path)
        record = self._store.get_file_record(str(path))
        if record and record[0] == file_hash:
            return 0  # unchanged

        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        raw_chunks = _chunk_lines(lines, self._chunk_tokens, self._chunk_overlap)
        if not raw_chunks:
            return 0

        # Hash each chunk for cache lookup
        chunk_hashes = [_hash_text(c["text"]) for c in raw_chunks]
        cached = self._store.get_cached_embeddings(chunk_hashes)

        # Only embed chunks not in cache
        to_embed: list[tuple[int, str]] = []
        for i, h in enumerate(chunk_hashes):
            if h not in cached:
                to_embed.append((i, raw_chunks[i]["text"]))

        if to_embed:
            texts_to_embed = [t for _, t in to_embed]
            new_embeddings = self._emb.embed_batch(texts_to_embed)
            # Store new embeddings in cache
            new_cache: dict[str, list[float]] = {}
            for (idx, _), emb in zip(to_embed, new_embeddings):
                h = chunk_hashes[idx]
                cached[h] = emb
                new_cache[h] = emb
            if new_cache:
                self._store.store_cached_embeddings(new_cache, self._emb.dimensions)

        chunks = [
            MemoryChunk(
                path=str(path),
                start_line=c["start"],
                end_line=c["end"],
                text=c["text"],
                source=source,
                embedding=cached[chunk_hashes[i]],
            )
            for i, c in enumerate(raw_chunks)
        ]
        self._store.upsert_chunks(chunks)
        self._store.upsert_file_record(str(path), source.value, file_hash, stat.st_mtime, stat.st_size)
        return len(chunks)

    def index_session_transcript(self, jsonl_path: str | Path) -> int:
        """Index a JSONL session transcript as MemorySource.SESSIONS.

        Uses delta tracking — only re-indexes when file has changed.
        """
        path = Path(jsonl_path)
        if not path.is_file():
            return 0

        # Delta check
        stat = path.stat()
        file_hash = _hash_file(path)
        record = self._store.get_file_record(str(path))
        if record and record[0] == file_hash:
            return 0  # unchanged

        # Check size delta threshold (100KB)
        old_size = record[2] if record else 0
        if record and (stat.st_size - old_size) < 100_000:
            return 0  # not enough new data

        texts: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if entry.get("type") != "message":
                continue
            role = entry.get("role", "")
            content = entry.get("content", "")
            if role in ("user", "assistant") and content:
                texts.append(f"{role}: {content}")
        if not texts:
            return 0
        combined = "\n".join(texts)
        lines = combined.splitlines()
        raw_chunks = _chunk_lines(lines, self._chunk_tokens, self._chunk_overlap)
        if not raw_chunks:
            return 0
        chunk_texts = [c["text"] for c in raw_chunks]
        embeddings = self._emb.embed_batch(chunk_texts)
        chunks = [
            MemoryChunk(
                path=str(path),
                start_line=c["start"],
                end_line=c["end"],
                text=c["text"],
                source=MemorySource.SESSIONS,
                embedding=emb,
            )
            for c, emb in zip(raw_chunks, embeddings)
        ]
        self._store.upsert_chunks(chunks)
        self._store.upsert_file_record(str(path), "sessions", file_hash, stat.st_mtime, stat.st_size)
        return len(chunks)

    def index_directory(
        self,
        directory: str | Path,
        *,
        source: MemorySource = MemorySource.MEMORY,
        glob: str = "**/*.md",
    ) -> int:
        """Index all matching files in a directory. Returns total chunk count."""
        d = Path(directory)
        if not d.is_dir():
            log.warning("index_directory: not a directory: %s", d)
            return 0
        total = 0
        for f in sorted(d.glob(glob)):
            if f.is_file():
                total += self.index_file(f, source=source)
        return total

    def sync(
        self,
        directories: list[str | Path],
        *,
        glob: str = "**/*.md",
        session_dirs: list[str | Path] | None = None,
    ) -> None:
        """Re-index directories and session transcripts, remove stale paths."""
        current_paths: set[str] = set()
        for d in directories:
            d = Path(d)
            if d.is_dir():
                for f in d.glob(glob):
                    if f.is_file():
                        current_paths.add(str(f))
                        self.index_file(f)
        for sd in session_dirs or []:
            sd = Path(sd)
            if sd.is_dir():
                for f in sd.glob("*.jsonl"):
                    if f.is_file():
                        current_paths.add(str(f))
                        self.index_session_transcript(f)
        # Clean up stale entries using file tracking
        for tracked in self._store.list_tracked_files():
            if tracked not in current_paths:
                self._store.remove_path(tracked)
                self._store.remove_file_record(tracked)
        self._dirty = False
        # Remember dirs for auto-sync and start watcher
        self._watched_dirs = list(directories)
        self._watched_session_dirs = list(session_dirs or [])
        self._start_watcher(self._watched_dirs + self._watched_session_dirs)

    def close(self) -> None:
        self._stop_watcher()
        self._store.close()

    # -- File watching --

    def _start_watcher(self, directories: list[str | Path]) -> None:
        """Watch directories for changes, set dirty flag with 5s debounce."""
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler
        except ImportError:
            log.debug("watchdog not installed, file watching disabled")
            return

        if self._watcher is not None:
            return

        mgr = self

        class _Handler(FileSystemEventHandler):
            def on_any_event(self, event):
                if not event.is_directory:
                    mgr._dirty = True

        observer = Observer()
        for d in directories:
            d = Path(d)
            if d.is_dir():
                observer.schedule(_Handler(), str(d), recursive=True)
        observer.daemon = True
        observer.start()
        self._watcher = observer

    def _stop_watcher(self) -> None:
        if self._watcher is not None:
            self._watcher.stop()
            self._watcher = None

    @property
    def dirty(self) -> bool:
        return self._dirty


# -- Helpers --


def _chunk_lines(
    lines: list[str], chunk_tokens: int, overlap: int
) -> list[dict]:
    """Split lines into overlapping chunks by estimated token count."""
    max_chars = chunk_tokens * _CHARS_PER_TOKEN
    overlap_chars = overlap * _CHARS_PER_TOKEN
    chunks: list[dict] = []
    i = 0
    while i < len(lines):
        buf: list[str] = []
        char_count = 0
        start = i
        while i < len(lines) and char_count < max_chars:
            buf.append(lines[i])
            char_count += len(lines[i]) + 1
            i += 1
        text = "\n".join(buf).strip()
        if text:
            chunks.append({"start": start + 1, "end": i, "text": text})
        # Step back by overlap
        back = 0
        j = i - 1
        while j > start and back < overlap_chars:
            back += len(lines[j]) + 1
            j -= 1
        i = max(j + 1, start + 1)
        if i == start:
            break  # safety: no progress
    return chunks


def _merge_results(
    vec: list[MemorySearchResult],
    bm25: list[MemorySearchResult],
    vector_weight: float,
    text_weight: float,
) -> list[MemorySearchResult]:
    """Merge vector and BM25 results by weighted score, deduped by (path, start_line)."""
    by_key: dict[tuple[str, int], MemorySearchResult] = {}
    for r in vec:
        key = (r.path, r.start_line)
        merged_score = r.score * vector_weight
        if key in by_key:
            by_key[key].score += merged_score
        else:
            by_key[key] = r.model_copy(update={"score": merged_score})
    for r in bm25:
        key = (r.path, r.start_line)
        merged_score = r.score * text_weight
        if key in by_key:
            by_key[key].score += merged_score
        else:
            by_key[key] = r.model_copy(update={"score": merged_score})
    return sorted(by_key.values(), key=lambda r: r.score, reverse=True)


def _hash_text(text: str) -> str:
    """SHA-256 hash of text content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_file(path: Path) -> str:
    """SHA-256 hash of file content."""
    return hashlib.sha256(path.read_bytes()).hexdigest()
