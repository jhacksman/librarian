"""Bounded reports of stored PDF extraction omissions, never completeness claims."""

import json
from pathlib import Path

MAX_METADATA_CHARS = 65_536
MAX_PAGE_PREVIEW = 20
MAX_PAGE_NUMBER = 2_147_483_647
# Do not use substr: SQLite length/substr stop at NUL and can accept a JSON prefix.
# Rejected values cross the SQLite boundary as NULL, not unbounded metadata.
BOUNDED_METADATA_SQL = (
    f"CASE WHEN typeof(metadata)='text' AND length(metadata)<={MAX_METADATA_CHARS} "
    "AND instr(metadata, char(0))=0 THEN metadata ELSE NULL END"
)
COUNT_FIELDS = (
    "indexed_books", "indexed_chunks", "reported_page_coverage_books",
    "unknown_page_coverage_books", "books_with_reported_omissions", "reported_pages_without_text",
)
COMPLETENESS_NOTE = "Extraction completeness has not been assessed."


def book_coverage(source_path, raw_metadata):
    report = {"page_omissions": "unknown", "pages_without_text_count": None,
              "pages_without_text": None, "pages_without_text_truncated": False,
              "completeness": "not_assessed"}
    if (not isinstance(source_path, (str, Path)) or Path(source_path).suffix.lower() != ".pdf"
            or not isinstance(raw_metadata, str) or len(raw_metadata) > MAX_METADATA_CHARS
            or "\x00" in raw_metadata):
        return report
    try:
        metadata = json.loads(raw_metadata)
    except (ValueError, RecursionError):
        return report
    pages = metadata.get("pages_without_text") if isinstance(metadata, dict) else None
    if not isinstance(pages, list):
        return report
    seen = set()
    for page in pages:
        if type(page) is not int or not 1 <= page <= MAX_PAGE_NUMBER or page in seen:
            return report
        seen.add(page)
    return {"page_omissions": "reported", "pages_without_text_count": len(pages),
            "pages_without_text": sorted(pages)[:MAX_PAGE_PREVIEW],
            "pages_without_text_truncated": len(pages) > MAX_PAGE_PREVIEW,
            "completeness": "not_assessed"}


def unknown_summary(book_id=None):
    return {"scope": "book" if book_id is not None else "all_indexed", "book_id": book_id,
            **dict.fromkeys(COUNT_FIELDS), "completeness": "not_assessed",
            "warnings": ["Recorded extraction coverage is unavailable for this retriever. " + COMPLETENESS_NOTE]}


def summarize_coverage(rows, book_id=None):
    """Consume scoped (source_path, bounded metadata, chunk_count) rows lazily."""
    summary = {"scope": "book" if book_id is not None else "all_indexed", "book_id": book_id,
               **dict.fromkeys(COUNT_FIELDS, 0), "completeness": "not_assessed"}
    for source_path, raw_metadata, chunk_count in rows:
        summary["indexed_books"] += 1
        summary["indexed_chunks"] += chunk_count
        report = book_coverage(source_path, raw_metadata)
        if report["page_omissions"] == "unknown":
            summary["unknown_page_coverage_books"] += 1
        else:
            summary["reported_page_coverage_books"] += 1
            count = report["pages_without_text_count"]
            summary["reported_pages_without_text"] += count
            summary["books_with_reported_omissions"] += int(count > 0)
    warnings = []
    if summary["reported_pages_without_text"]:
        warnings.append(
            f"When indexed, {summary['reported_pages_without_text']} PDF pages produced no extractable text "
            "and are not searchable. They may be blank or image-only."
        )
    if summary["unknown_page_coverage_books"]:
        warnings.append(
            f"Recorded PDF page omission information is unavailable for {summary['unknown_page_coverage_books']} "
            "indexed books in this scope."
        )
    warnings.append(COMPLETENESS_NOTE)
    summary["warnings"] = warnings
    return summary


def coverage_note(report):
    """Render one validated per-book report without paths or raw metadata."""
    if not report or report["page_omissions"] == "unknown":
        return "Recorded PDF page omission information is unavailable for this book. " + COMPLETENESS_NOTE
    count = report["pages_without_text_count"]
    if not count:
        return "No PDF pages without extractable text were recorded for this book. " + COMPLETENESS_NOTE
    pages = ", ".join(str(page) for page in report["pages_without_text"])
    preview = " (first 20 shown)" if report["pages_without_text_truncated"] else ""
    return (f"When indexed, {count} PDF pages produced no extractable text and are not searchable: "
            f"{pages}{preview}. They may be blank or image-only. " + COMPLETENESS_NOTE)
