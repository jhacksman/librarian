"""Read-only evidence interface shared by human and agent entry points."""

import re
from inspect import getattr_static
from pathlib import Path
from typing import Protocol

from ingest.local.coverage import (
    BOUNDED_METADATA_SQL,
    book_coverage,
    coverage_note,
    summarize_coverage,
    unknown_summary,
)
from ingest.local.index import LocalIndex
from ingest.local.query import first_literal_match, plan_query
from ingest.models import TextChunk

MAX_QUOTE_CHARS = 8000
MAX_LABEL_CHARS = 500
MAX_PASSAGE_OFFSET = 2_147_483_647
BOOK_ID_PATTERN = r"[0-9a-f]{64}"
CHUNK_ID_PATTERN = r"[0-9a-f]{64}:[0-9]{1,10}"


class LibraryRequestError(ValueError):
    """Safe user-correctable error; never includes source paths or raw DB errors."""


class Retriever(Protocol):
    def search(self, query: str, limit: int, book_id: str | None) -> list[dict]: ...
    def passage(self, chunk_id: str, offset: int = 0) -> dict | None: ...
    def books(self, limit: int, offset: int, book_id: str | None = None) -> list[dict]: ...
    def coverage(self) -> dict: ...


def public_passage(chunk: TextChunk, offset=0):
    if type(offset) is not int or not 0 <= offset < len(chunk.content):
        raise LibraryRequestError("Excerpt offset outside passage")
    quote = chunk.content[offset:offset + MAX_QUOTE_CHARS]
    start = chunk.metadata.get("char_start")
    end = chunk.metadata.get("char_end")
    if type(start) is not int or type(end) is not int or start < 0 or end - start != len(chunk.content):
        raise ValueError("Invalid indexed source span")
    member = chunk.metadata.get("epub_member")
    if member is not None and (not isinstance(member, str) or len(member) > 2048):
        raise ValueError("Invalid indexed EPUB locator")
    return {
        "chunk_id": chunk.id, "book_id": chunk.book_id,
        "title": str(chunk.metadata.get("title", "Untitled"))[:MAX_LABEL_CHARS],
        "quote": quote, "chapter": chunk.chapter[:MAX_LABEL_CHARS] if chunk.chapter else None,
        "page_start": chunk.page_start, "page_end": chunk.page_end,
        "epub_member": member,
        "source_sha256": chunk.metadata.get("source_sha256", chunk.book_id),
        "char_start": start + offset, "char_end": start + offset + len(quote),
        "original_char_start": start, "original_char_end": end,
        "excerpt_offset": offset, "excerpt_truncated": len(quote) < len(chunk.content),
        "next_offset": offset + len(quote) if offset + len(quote) < len(chunk.content) else None,
        "offset_basis": "extracted_section_unicode_codepoints",
        "pipeline_version": str(chunk.metadata.get("pipeline_version", "legacy-unspecified"))[:100],
        "source_uri": f"librarian://books/{chunk.book_id}/chunks/{chunk.chunk_index}",
        "content_kind": "source_excerpt",
    }


class SQLiteRetriever:
    """Open a read-only connection per request, without model/network initialization."""

    def __init__(self, path: Path):
        self.path = path

    def search(self, query, limit, book_id=None):
        index = LocalIndex(self.path)
        try:
            plan = plan_query(query, drop_question_words=True)
            hits = index.search(query, limit, book_id=book_id, drop_question_words=True)
            return [self._passage(index, hit.chunk_id, query=plan.preserved_query, literals=plan.literals) for hit in hits]
        finally:
            index.close()

    @staticmethod
    def _passage(index, chunk_id, offset=0, query=None, literals=()):
        row = index.db.execute(
            f"SELECT payload, books.source_path, {BOUNDED_METADATA_SQL} FROM chunks "
            "LEFT JOIN books ON books.book_id=chunks.book_id WHERE chunk_id=?", (chunk_id,),
        ).fetchone()
        if row is None:
            return None
        chunk = TextChunk.model_validate_json(row[0])
        if query and len(chunk.content) > MAX_QUOTE_CHARS:
            if literals:
                literal = first_literal_match(chunk.content, literals)
                if literal is None:
                    raise ValueError("Indexed payload lacks its matched literal")
                match_start = literal.start
            else:
                terms = re.findall(r"\w+", query)
                pattern = r"(?<!\w)(?:" + "|".join(re.escape(term) for term in terms) + r")(?!\w)"
                match = re.search(pattern, chunk.content, re.IGNORECASE)
                match_start = match.start() if match else None
            if match_start is not None:
                offset = max(0, min(match_start - MAX_QUOTE_CHARS // 2, len(chunk.content) - MAX_QUOTE_CHARS))
        return {**public_passage(chunk, offset), "extraction_coverage": book_coverage(row[1], row[2])}

    def passage(self, chunk_id, offset=0):
        index = LocalIndex(self.path)
        try:
            return self._passage(index, chunk_id, offset)
        finally:
            index.close()

    def books(self, limit, offset, book_id=None):
        index = LocalIndex(self.path)
        try:
            rows = index.db.execute(
                f"SELECT book_id, substr(title, 1, ?), chunk_count, source_path, {BOUNDED_METADATA_SQL} "
                "FROM books " + ("WHERE book_id=? " if book_id is not None else "")
                + "ORDER BY book_id LIMIT ? OFFSET ?",
                (MAX_LABEL_CHARS, book_id, limit, offset) if book_id is not None
                else (MAX_LABEL_CHARS, limit, offset),
            )
            return [{"book_id": row[0], "title": row[1], "chunk_count": row[2],
                     "extraction_coverage": book_coverage(row[3], row[4])} for row in rows]
        finally:
            index.close()

    def coverage(self):
        summary = self.extraction_coverage()
        return {"indexed_books": summary["indexed_books"], "indexed_chunks": summary["indexed_chunks"],
                "extraction_coverage": summary}

    def extraction_coverage(self, book_id=None):
        index = LocalIndex(self.path)
        try:
            rows = index.db.execute(
                f"SELECT source_path, {BOUNDED_METADATA_SQL}, chunk_count FROM books"
                + (" WHERE book_id=?" if book_id is not None else ""),
                (book_id,) if book_id is not None else (),
            )
            return summarize_coverage(rows, book_id)
        finally:
            index.close()


class LibraryService:
    """Evidence-only now; a future reviewed semantic adapter uses this same contract."""

    def __init__(self, retriever: Retriever, backend="sqlite-fts5-bm25"):
        self.retriever = retriever
        self.backend = backend

    def _extraction_summary(self, book_id=None):
        try:
            capability = self.retriever.extraction_coverage
        except AttributeError:
            missing = object()
            if getattr_static(self.retriever, "extraction_coverage", missing) is not missing:
                raise  # A declared capability failed while being bound.
            return unknown_summary(book_id)
        return capability(book_id)

    def ask(self, question: str, limit: int = 5, book_id: str | None = None):
        if not isinstance(question, str) or not question.strip() or len(question) > 2000:
            raise LibraryRequestError("Question must contain 1-2000 characters")
        if type(limit) is not int or not 1 <= limit <= 10:
            raise LibraryRequestError("Limit must be an integer from 1 to 10")
        if book_id is not None and (not isinstance(book_id, str) or not re.fullmatch(BOOK_ID_PATTERN, book_id)):
            raise LibraryRequestError("Invalid book ID")
        query = question.strip()
        normalization = None
        if self.backend == "sqlite-fts5-bm25":
            plan = plan_query(question, drop_question_words=True)
            query = plan.preserved_query
            normalization = plan.provenance()
            term_count = len(plan.candidate_terms)
        else:
            term_count = len(re.findall(r"\w+", query))
        if term_count > 128:
            raise LibraryRequestError("Question exceeds 128 search terms")
        # The lexical adapter plans from original spelling too. Reparsing a
        # normalized display string could turn unsupported syntax into a literal.
        request_query = question if self.backend == "sqlite-fts5-bm25" else query
        passages = self.retriever.search(request_query, limit, book_id) if query else []
        coverage = self._extraction_summary(book_id)
        for number, passage in enumerate(passages, 1):
            passage["citation_id"] = f"[{number}]"
        return {
            "question": question, "retrieval_query": query, "backend": self.backend,
            "query_normalization": normalization,
            "answer_mode": "evidence_only", "abstained": not passages,
            "message": ("Source excerpts found. No generated answer has been produced."
                        if passages else "No matching evidence was found in the indexed books. I cannot answer from this library search."),
            "passages": passages,
            "extraction_coverage": coverage,
            "limitations": "Matches are retrieval evidence, not proof that the question is answered. Only indexed books are searched.",
        }

    def get_passage(self, chunk_id: str, offset: int = 0):
        if not isinstance(chunk_id, str) or not re.fullmatch(CHUNK_ID_PATTERN, chunk_id):
            raise LibraryRequestError("Invalid chunk ID")
        if type(offset) is not int or not 0 <= offset <= MAX_PASSAGE_OFFSET:
            raise LibraryRequestError("Invalid excerpt offset")
        passage = self.retriever.passage(chunk_id, offset)
        if passage is None:
            raise LibraryRequestError("Passage not found")
        return passage

    def status(self):
        coverage = dict(self.retriever.coverage())
        if "extraction_coverage" not in coverage:
            coverage["extraction_coverage"] = self._extraction_summary()
        return {"backend": self.backend, "answer_mode": "evidence_only", **coverage,
                "semantic_model_loaded": False, "read_only": True}

    def books(self, limit=50, offset=0, book_id=None):
        if type(limit) is not int or not 1 <= limit <= 50:
            raise LibraryRequestError("Book limit must be an integer from 1 to 50")
        if type(offset) is not int or not 0 <= offset <= 1_000_000:
            raise LibraryRequestError("Book offset must be an integer from 0 to 1000000")
        if book_id is not None and (not isinstance(book_id, str) or not re.fullmatch(BOOK_ID_PATTERN, book_id)):
            raise LibraryRequestError("Invalid book ID")
        items = (self.retriever.books(limit + 1, offset, book_id=book_id) if book_id is not None
                 else self.retriever.books(limit + 1, offset))
        return {"books": items[:limit], "next_offset": offset + limit if len(items) > limit else None}


def render_answer(answer):
    lines = [answer["message"], f"Search: {answer['retrieval_query']}", ""]
    if answer.get("limitations"):
        lines.append(answer["limitations"])
    lines.extend(answer.get("extraction_coverage", {}).get("warnings", []))
    for passage in answer["passages"]:
        location = f"PDF page {passage['page_start']}" if passage["page_start"] else f"EPUB {passage['epub_member']}"
        lines.extend([f"{passage['citation_id']} {passage['title']} — {location}",
                      coverage_note(passage.get("extraction_coverage")),
                      f"Source excerpt: {passage['quote']}",
                      f"Source span: {passage['char_start']}–{passage['char_end']}" + (" (excerpt shortened)" if passage["excerpt_truncated"] else ""),
                      f"SHA-256: {passage['source_sha256']}", f"Passage: {passage['chunk_id']}", ""])
    return "\n".join(lines)
