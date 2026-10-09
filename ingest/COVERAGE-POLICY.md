# C1: recorded PDF omission visibility

This change reports metadata already stored by the local ingestion pilot. It
does not recover omitted text, classify a page as meaningful, or assess complete
extraction. The original C1 P3 finding was independently confirmed by two
skeptics before being deferred behind the now-closed literal-search work.

PDF extraction records one-based physical page numbers in `pages_without_text`
when `extract_text()` returns no non-whitespace text. Existing book metadata
retains that list, but public search/catalog/status responses omit it and resumed
ingestion omits it from the returned result. EPUB metadata has no corresponding
page report. Total PDF page count is not stored in existing book metadata, so
coverage percentages cannot be derived honestly.

## Per-book contract

Add `extraction_coverage` to completed/skipped ingestion results, local/public
catalog items and SQLite-backed returned/direct passages:

```text
page_omissions: "reported" | "unknown"
pages_without_text_count: integer | null
pages_without_text: [physical page numbers, at most 20] | null
pages_without_text_truncated: boolean
completeness: "not_assessed"
```

A report is known only for a stored `.pdf` source path (case-insensitive suffix)
and an object containing
a valid `pages_without_text` list. No source file is opened to determine this.
Entries must be distinct integers from 1 through 2,147,483,647; booleans are not
integers for this contract. Return sorted page previews. Missing metadata/keys,
invalid JSON, non-object metadata, invalid lists and duplicates are unknown.
EPUB and other source formats remain unknown for this page-specific report.

Fetch at most 65,536 metadata characters per book, or a bounded overflow indicator;
oversized metadata is unknown. Reject embedded NUL instead of accepting a JSON
prefix truncated by SQLite text functions. This bounds metadata decoding and the
returned preview. Counts retain the full validated list length. A valid empty
PDF list means no such pages were recorded, never complete extraction. Unknown
uses null counts/lists and a false truncation flag, not invented zeros. The
completeness field is always `not_assessed`.

Completed and skipped ingestion results share this same validated report. Their
top-level `pages_without_text` field is its bounded preview (or null), with the
full count and truncation indicator in `extraction_coverage`. The full original
metadata stays persisted. Thus an EPUB no longer emits a fabricated empty PDF
page report, and resume neither reparses the book nor loses its recorded warning.

## Search scope and human visibility

Add an `extraction_coverage` summary to library status and each ask response:

```text
scope: "all_indexed" | "book"
book_id: null | requested book hash
indexed_books, indexed_chunks
reported_page_coverage_books, unknown_page_coverage_books
books_with_reported_omissions, reported_pages_without_text
completeness: "not_assessed"
warnings: at most 3 safe explanatory strings, each at most 500 characters
```

For SQLite, counts describe the requested indexed-book scope, independently of
matching passages. Reported and unknown book counts partition that scope; sum
omissions only from validated metadata. A filtered no-match or an empty normalized
question still returns that book's report. A nonexistent book filter yields zero
books in scope, without substituting global counts. No list of every affected
book is added to an ask response; the existing paginated catalog supplies details.

Preserve the existing retriever seam: adapters that do not implement the optional
extraction-summary capability preserve scope/book ID and report null for all six
numeric count fields above, rather than
inferring coverage from hits or failing solely for the absent capability. Errors
from an implemented capability remain actual service errors. SQLite's catalog
envelope and pagination remain unchanged; its items gain only the per-book report.

CLI text and the loopback HTML preview display scope warnings and per-passage
page notes, including no-match responses. MCP exposes the same structured data.
Use wording such as: "When indexed, 2 PDF pages produced no extractable text and
are not searchable. They may be blank or image-only. Extraction completeness has
not been assessed." Missing records are explicitly unavailable. Zero recorded
omissions also retain the completeness caveat. Output must not expose source
paths, raw metadata, parser diagnostics or unbounded page lists through public
service/MCP/web responses.

## Failure behavior and verification

Parser exceptions and entirely textless books already fail ingestion explicitly.
Keep per-file errors, durable failed receipts, nonzero CLI exit status and
transaction behavior unchanged. Do not add page-level exception suppression,
partial parser recovery, OCR, metadata backfills, schema changes or reindexing.
Do not alter query grammar, the accepted literal guard, ranking, excerpt contents,
citation IDs, source hashes or offsets. Reading summaries never writes the index.

Synthetic checks cover mixed text/blank PDF pages, recorded-zero PDF, unknown
EPUB/legacy/malformed/oversized metadata, page-preview bounds, resume, all-empty
and page-extraction failures, per-book and aggregate counts, filtered no-match,
source removal, unchanged read-only DB bytes/citations, CLI/HTML output and real
MCP transport parity. Run tests and required lint only through Spark. Keep the
closed PERF1 measurements attributed to their original source; this reporting
work makes no new end-to-end latency claim and does not rerun that benchmark.
