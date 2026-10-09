"""Bounded, repeatable local ingestion and SQLite FTS5 retrieval."""

import hashlib
import json
import re
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from ingest.local.coverage import BOUNDED_METADATA_SQL, book_coverage
from ingest.local.extract import extract
from ingest.local.query import contains_literals, plan_query
from ingest.local.schema import initialize, inspect_schema
from ingest.models import ExtractedContent, SearchResult, TextChunk

PIPELINE_VERSION = "local-text-v2-words400-overlap80"
MAX_SOURCE_BYTES = 64 * 1024 * 1024


def chunks_for(content: ExtractedContent, book_id: str, source: Path, size=400, overlap=80):
    """Word windows stay within source sections; never discard short sections."""
    if not 0 <= overlap < size:
        raise ValueError("Require 0 <= overlap < chunk size")
    chunks = []
    for section in content.chapters:
        words = list(re.finditer(r"\S+", section.content))
        for start in range(0, len(words), size - overlap):
            end = min(start + size, len(words))
            char_start, char_end = words[start].start(), words[end - 1].end()
            location = content.metadata.get("section_sources", {}).get(str(section.number))
            metadata = {
                "title": content.title,
                "authors": content.metadata.get("authors", []),
                "source_path": str(source),
                "source_sha256": book_id,
                "epub_member": location,
                "word_start": start,
                "word_end": end,
                "char_start": char_start,
                "char_end": char_end,
                "token_count_method": "ceil_characters_div_4_estimate",
            }
            text = section.content[char_start:char_end]
            index = len(chunks)
            chunks.append(
                TextChunk(
                    id=f"{book_id}:{index}",
                    book_id=book_id,
                    content=text,
                    chapter=section.title,
                    chapter_number=section.number,
                    page_start=section.start_page,
                    page_end=section.end_page,
                    chunk_index=index,
                    token_count=(len(text) + 3) // 4,
                    metadata=metadata,
                )
            )
            if end == len(words):
                break
    for chunk in chunks:
        chunk.total_chunks = len(chunks)
    return chunks


class LocalIndex:
    """One writer at a time; writes are atomic per book, not per invocation."""

    def __init__(self, path: Path, *, create=False):
        path = path.resolve()
        if create:
            path.parent.mkdir(parents=True, exist_ok=True)
        elif not path.is_file():
            raise ValueError(f"Index does not exist: {path}; run ingest first")
        self.db = sqlite3.connect(
            str(path) if create else path.as_uri() + "?mode=ro", uri=not create
        )
        self.db.row_factory = sqlite3.Row
        try:
            if create:
                initialize(self.db)
            elif inspect_schema(self.db) is None:
                raise ValueError("Empty file is not an index")
        except BaseException:
            self.db.close()
            raise

    def close(self):
        self.db.close()

    def _record(self, source, digest, status, error=None):
        self.db.execute(
            """INSERT INTO ingest_state VALUES (?, ?, ?, ?, 1, ?, ?)
            ON CONFLICT(source_path) DO UPDATE SET
                source_sha256=excluded.source_sha256,
                pipeline_version=excluded.pipeline_version,
                status=excluded.status, attempts=ingest_state.attempts+1,
                error=excluded.error, updated_at=excluded.updated_at""",
            (str(source), digest, PIPELINE_VERSION, status, error,
             datetime.now(timezone.utc).isoformat()),
        )

    def ingest(self, source: Path, *, resume=False):
        source = source.resolve()
        digest = None
        try:
            # Parse the exact byte snapshot whose digest is cited, even if the original changes.
            if source.suffix.lower() not in {".epub", ".pdf"} or not source.is_file():
                raise ValueError("Expected an explicit EPUB or PDF file")
            with tempfile.TemporaryDirectory(prefix="librarian-source-") as directory:
                snapshot = Path(directory) / source.name
                hasher = hashlib.sha256()
                size = 0
                with source.open("rb") as original, snapshot.open("xb") as copied:
                    for block in iter(lambda: original.read(1024 * 1024), b""):
                        size += len(block)
                        if size > MAX_SOURCE_BYTES:
                            raise ValueError("Source exceeds the 64 MiB pilot file limit")
                        hasher.update(block)
                        copied.write(block)
                if not size:
                    raise ValueError("Expected a non-empty file")
                digest = hasher.hexdigest()
                if resume:
                    prior = self.db.execute(
                        f"""SELECT b.title, b.chunk_count, {BOUNDED_METADATA_SQL} FROM ingest_state s
                        JOIN books b ON b.book_id=s.source_sha256
                        WHERE s.source_path=? AND b.source_path=? AND s.source_sha256=?
                          AND s.pipeline_version=? AND s.status='completed'
                          AND b.chunk_count=(SELECT count(*) FROM chunks c WHERE c.book_id=b.book_id)""",
                        (str(source), str(source), digest, PIPELINE_VERSION),
                    ).fetchone()
                    if prior:
                        report = book_coverage(source, prior[2])
                        return {"status": "skipped", "book_id": digest, "title": prior[0],
                                "chunks": prior[1], "source_path": str(source),
                                "pages_without_text": report["pages_without_text"],
                                "extraction_coverage": report}
                content = extract(snapshot)
            chunks = chunks_for(content, digest, source)
            if not chunks:
                raise ValueError("No chunks extracted; index unchanged")
            for chunk in chunks:
                chunk.metadata["pipeline_version"] = PIPELINE_VERSION
            metadata = json.dumps(content.metadata)
            report = book_coverage(source, metadata)
            with self.db:
                self.db.execute("DELETE FROM chunks WHERE book_id = ?", (digest,))
                self.db.executemany(
                    "INSERT INTO chunks(chunk_id, book_id, content, payload) VALUES (?, ?, ?, ?)",
                    [(ch.id, digest, ch.content, ch.model_dump_json()) for ch in chunks],
                )
                self.db.execute(
                    "INSERT OR REPLACE INTO books VALUES (?, ?, ?, ?, ?)",
                    (digest, content.title, str(source), metadata, len(chunks)),
                )
                self._record(source, digest, "completed")
            return {
                "status": "completed", "book_id": digest, "title": content.title,
                "chunks": len(chunks), "words": content.word_count, "pages": content.page_count,
                "pages_without_text": report["pages_without_text"], "extraction_coverage": report,
                "source_path": str(source),
            }
        except Exception as error:
            try:
                with self.db:
                    self._record(source, digest, "failed", f"{type(error).__name__}: {error}"[:500])
            except sqlite3.Error:
                pass  # Preserve the original error if the database itself is unavailable.
            raise

    def ingest_many(self, sources, *, resume=True):
        """Isolate file errors; durable receipts make a restarted invocation resumable."""
        if not 1 <= len(sources) <= 5:
            raise ValueError("Supply 1-5 explicit files; directory/bulk processing is not enabled")
        results = []
        for source in sources:
            try:
                results.append(self.ingest(source, resume=resume))
            except Exception as error:
                results.append({"source_path": str(source.resolve()), "status": "failed",
                                "error": f"{type(error).__name__}: {error}"[:500]})
        return results

    def receipts(self):
        if inspect_schema(self.db) == 0:
            raise ValueError("Legacy index has no receipts; migrate first")
        return [dict(row) for row in self.db.execute("SELECT * FROM ingest_state ORDER BY source_path")]

    def books(self):
        rows = self.db.execute(
            f"SELECT book_id, title, source_path, chunk_count, {BOUNDED_METADATA_SQL} AS coverage_metadata "
            "FROM books ORDER BY title"
        )
        items = []
        for row in rows:
            item = dict(row)
            item["extraction_coverage"] = book_coverage(item["source_path"], item.pop("coverage_metadata"))
            items.append(item)
        return items

    def search(self, query: str, limit=5, *, book_id: str | None = None, drop_question_words=False):
        if not 1 <= limit <= 100:
            raise ValueError("Search limit must be 1-100")
        # Treat user input as literal words, never raw FTS operators or SQL.
        plan = plan_query(query, drop_question_words=drop_question_words)
        if not plan.candidate_terms:
            return []
        if plan.literals:
            self.db.create_function("librarian_literal_match", 1,
                                    lambda content: int(contains_literals(content, plan.literals)),
                                    deterministic=True)
        rows = self.db.execute(
            "SELECT payload, bm25(chunks) AS rank FROM chunks WHERE chunks MATCH ? "
            + ("AND book_id=? " if book_id is not None else "")
            + ("AND librarian_literal_match(content)=1 " if plan.literals else "")
            + "ORDER BY rank, chunk_id LIMIT ?",
            (plan.expression, book_id, limit) if book_id is not None else (plan.expression, limit),
        )
        results = []
        for row in rows:
            chunk = TextChunk.model_validate_json(row["payload"])
            location = (
                f"PDF page {chunk.page_start}"
                if chunk.page_start
                else f"EPUB {chunk.metadata['epub_member']} (spine section {chunk.chapter_number})"
            )
            citation = f"{chunk.metadata['title']} | {location} | chunk {chunk.chunk_index} | sha256:{chunk.book_id}"
            results.append(
                SearchResult(
                    chunk_id=chunk.id,
                    book_id=chunk.book_id,
                    score=-row["rank"],
                    content=chunk.content,
                    metadata={
                        **chunk.metadata,
                        "citation": citation,
                        "page_start": chunk.page_start,
                        "page_end": chunk.page_end,
                        "chapter": chunk.chapter,
                        "retrieval": "sqlite-fts5-bm25",
                        "query_normalization": plan.provenance(),
                    },
                )
            )
        return results
