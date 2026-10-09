"""Independent C1 synthetic coverage checks; execute only on Spark."""

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from html import escape
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
from pypdf import PageObject, PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from ingest.local.coverage import book_coverage
from ingest.local.index import LocalIndex
from ingest.local.mcp_adapter import register_tools
from ingest.local.schema import BASE_SQL
from ingest.local.service import LibraryService, SQLiteRetriever, render_answer
from ingest.local.web import render_page

UNKNOWN = {"page_omissions": "unknown", "pages_without_text_count": None,
           "pages_without_text": None, "pages_without_text_truncated": False,
           "completeness": "not_assessed"}
MIXED = {"page_omissions": "reported", "pages_without_text_count": 2,
         "pages_without_text": [2, 4], "pages_without_text_truncated": False,
         "completeness": "not_assessed"}
ZERO = {"page_omissions": "reported", "pages_without_text_count": 0,
        "pages_without_text": [], "pages_without_text_truncated": False,
        "completeness": "not_assessed"}
COUNT_FIELDS = ("indexed_books", "indexed_chunks", "reported_page_coverage_books",
                "unknown_page_coverage_books", "books_with_reported_omissions", "reported_pages_without_text")


def write_pdf(path, pages, *, author=None):
    writer = PdfWriter()
    for text in pages:
        page = writer.add_blank_page(612, 792)
        if text is None:
            continue
        font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                                 NameObject("/Subtype"): NameObject("/Type1"),
                                 NameObject("/BaseFont"): NameObject("/Helvetica")})
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 40 700 Td ({text}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    metadata = {"/Title": "Synthetic coverage PDF"}
    if author is not None:
        metadata["/Author"] = author
    writer.add_metadata(metadata)
    writer.write(path)
    return path


def write_epub(path):
    with ZipFile(path, "w") as archive:
        archive.writestr("META-INF/container.xml",
                         '<container><rootfiles><rootfile full-path="book.opf"/></rootfiles></container>')
        archive.writestr("book.opf", '<package xmlns="http://www.idpf.org/2007/opf">'
                         '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Coverage EPUB</dc:title></metadata>'
                         '<manifest><item id="one" href="one.xhtml" media-type="application/xhtml+xml"/></manifest>'
                         '<spine><itemref idref="one"/></spine></package>')
        archive.writestr("one.xhtml", "<p>Amber lantern evidence.</p>")
    return path


@pytest.fixture
def library(tmp_path):
    sources = {"mixed": write_pdf(tmp_path / "mixed.PDF", ["Alpha needle evidence.", None, "Omega anchor evidence.", None]),
               "zero": write_pdf(tmp_path / "zero.pdf", ["Clear copper evidence."]),
               "epub": write_epub(tmp_path / "unknown.epub")}
    path = tmp_path / "library.sqlite"
    index = LocalIndex(path, create=True)
    try:
        receipts = {label: index.ingest(source) for label, source in sources.items()}
    finally:
        index.close()
    return SimpleNamespace(path=path, sources=sources, receipts=receipts,
                           ids={label: receipt["book_id"] for label, receipt in receipts.items()},
                           service=LibraryService(SQLiteRetriever(path)))


def assert_summary(summary, counts, *, book_id=None):
    assert summary["scope"] == ("book" if book_id is not None else "all_indexed")
    assert summary["book_id"] == book_id
    assert tuple(summary[field] for field in COUNT_FIELDS) == counts
    assert summary["completeness"] == "not_assessed"
    warnings = summary["warnings"]
    assert 1 <= len(warnings) <= 3
    assert all(isinstance(warning, str) and 0 < len(warning) <= 500 for warning in warnings)
    assert "completeness" in " ".join(warnings).lower()


def save_evidence(name, report):
    destination = os.environ.get("LIBRARIAN_EVIDENCE_DIR")
    if destination:
        root = Path(destination)
        root.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def replace_metadata(library, value):
    with sqlite3.connect(library.path) as db:
        db.execute("UPDATE books SET metadata=? WHERE book_id=?", (value, library.ids["mixed"]))


def indexed_rows(path):
    with sqlite3.connect(path) as db:
        return (list(db.execute("SELECT * FROM books ORDER BY book_id")),
                list(db.execute("SELECT chunk_id, book_id, content, payload FROM chunks ORDER BY chunk_id")))


def test_mixed_pdf_coverage_survives_resume_without_reextracting(library, monkeypatch):
    import ingest.local.index as index_module

    receipt = library.receipts["mixed"]
    assert receipt["extraction_coverage"] == MIXED
    assert receipt["pages_without_text"] == [2, 4]
    assert library.receipts["zero"]["extraction_coverage"] == ZERO
    assert library.receipts["epub"]["extraction_coverage"] == UNKNOWN
    assert library.receipts["epub"]["pages_without_text"] is None
    monkeypatch.setattr(index_module, "extract", lambda _path: pytest.fail("Resume reparsed the source"))
    before = library.path.read_bytes()
    index = LocalIndex(library.path, create=True)
    try:
        resumed = index.ingest(library.sources["mixed"], resume=True)
        assert resumed["status"] == "skipped"
        assert resumed["extraction_coverage"] == MIXED and resumed["pages_without_text"] == [2, 4]
        catalog = {book["book_id"]: book for book in index.books()}
        assert catalog[library.ids["mixed"]]["extraction_coverage"] == MIXED
        raw = index.db.execute("SELECT metadata FROM books WHERE book_id=?", (library.ids["mixed"],)).fetchone()[0]
        assert json.loads(raw)["pages_without_text"] == [2, 4]
    finally:
        index.close()
    assert library.path.read_bytes() == before


@pytest.mark.parametrize("raw", [
    None, "{}", "null", "[]", "{broken", '{"pages_without_text": null}',
    '{"pages_without_text": "2"}', '{"pages_without_text": [true]}',
    '{"pages_without_text": [2.0]}', '{"pages_without_text": ["2"]}',
    '{"pages_without_text": [0]}', '{"pages_without_text": [-1]}',
    '{"pages_without_text": [2147483648]}', '{"pages_without_text": [2, 2]}',
    '{"pages_without_text": [2]}\x00ignored suffix',
    json.dumps({"pages_without_text": [*range(1, 22), False]}),
    json.dumps({"pages_without_text": [*range(1, 22), 1]}),
    pytest.param('{"pages_without_text": ' + "[" * 5000 + "1" + "]" * 5000 + "}", id="deeply-nested"),
    pytest.param('{"pages_without_text": [' + "9" * 5000 + "]}", id="oversized-integer"),
])
def test_missing_and_malformed_page_reports_are_unknown(raw):
    assert book_coverage("unopened.pdf", raw) == UNKNOWN


def test_validated_preview_is_sorted_bounded_and_checks_the_whole_list():
    pages = [2147483647, *range(27, 0, -1)]
    report = book_coverage("unopened.PDF", json.dumps({"pages_without_text": pages}))
    assert report == {"page_omissions": "reported", "pages_without_text_count": 28,
                      "pages_without_text": list(range(1, 21)), "pages_without_text_truncated": True,
                      "completeness": "not_assessed"}
    assert book_coverage("unopened.epub", '{"pages_without_text": []}') == UNKNOWN
    assert book_coverage("unopened.txt", '{"pages_without_text": [2]}') == UNKNOWN
    assert book_coverage("unopened.pdf", '{"pages_without_text": []}') == ZERO


@pytest.mark.parametrize("size,expected", [(65536, MIXED), (65537, UNKNOWN)])
def test_metadata_character_limit_applies_before_json_prefix_acceptance(library, size, expected):
    prefix = '{"pages_without_text": [2, 4]}'
    raw = prefix + " " * (size - len(prefix))
    assert book_coverage("unopened.pdf", raw) == expected
    replace_metadata(library, raw)
    before = library.path.read_bytes()
    passage = library.service.ask("needle")["passages"][0]
    assert passage["extraction_coverage"] == expected
    assert library.path.read_bytes() == before


def test_metadata_limit_counts_unicode_characters_not_utf8_bytes(library):
    raw = json.dumps({"pages_without_text": [2, 4], "note": "界" * 30000}, ensure_ascii=False)
    assert len(raw) < 65536 < len(raw.encode("utf-8"))
    replace_metadata(library, raw)
    assert library.service.ask("needle")["passages"][0]["extraction_coverage"] == MIXED


@pytest.mark.parametrize("raw", ["{}", "{broken", '{"pages_without_text": [2]}\x00private parser suffix'])
def test_stored_unknown_metadata_does_not_leak_or_become_zero(library, raw):
    original_passage = library.service.ask("needle")["passages"][0]
    replace_metadata(library, raw)
    before = library.path.read_bytes()
    answer = library.service.ask("needle", book_id=library.ids["mixed"])
    assert answer["passages"][0]["extraction_coverage"] == UNKNOWN
    assert {key: value for key, value in answer["passages"][0].items() if key != "extraction_coverage"} == {
        key: value for key, value in original_passage.items() if key != "extraction_coverage"}
    assert_summary(answer["extraction_coverage"], (1, 2, 0, 1, 0, 0), book_id=library.ids["mixed"])
    assert "private parser suffix" not in json.dumps(answer) and "source_path" not in json.dumps(answer)
    catalog = {book["book_id"]: book for book in library.service.books()["books"]}
    assert catalog[library.ids["mixed"]]["extraction_coverage"] == UNKNOWN
    assert library.service.get_passage(answer["passages"][0]["chunk_id"])["extraction_coverage"] == UNKNOWN
    index = LocalIndex(library.path, create=True)
    try:
        local = {book["book_id"]: book for book in index.books()}
        assert local[library.ids["mixed"]]["extraction_coverage"] == UNKNOWN
        resumed = index.ingest(library.sources["mixed"], resume=True)
        assert resumed["status"] == "skipped" and resumed["extraction_coverage"] == UNKNOWN
        assert resumed["pages_without_text"] is None
    finally:
        index.close()
    assert library.path.read_bytes() == before


def test_page_preview_is_bounded_without_discarding_persisted_omissions(tmp_path):
    source = write_pdf(tmp_path / "many.pdf", ["Text remains searchable.", *[None] * 27])
    path = tmp_path / "many.sqlite"
    index = LocalIndex(path, create=True)
    try:
        receipt = index.ingest(source)
        report = receipt["extraction_coverage"]
        assert report["pages_without_text_count"] == 27
        assert receipt["pages_without_text"] == report["pages_without_text"] == list(range(2, 22))
        assert report["pages_without_text_truncated"]
        raw = index.db.execute("SELECT metadata FROM books").fetchone()[0]
        assert json.loads(raw)["pages_without_text"] == list(range(2, 29))
        assert index.ingest(source, resume=True)["extraction_coverage"] == report
    finally:
        index.close()
    service = LibraryService(SQLiteRetriever(path))
    assert service.books()["books"][0]["extraction_coverage"] == report
    answer = service.ask("searchable")
    assert answer["passages"][0]["extraction_coverage"] == report
    assert_summary(service.status()["extraction_coverage"], (1, 1, 1, 0, 1, 27))
    visible_preview = ", ".join(str(page) for page in range(2, 22))
    for rendered in (render_answer(answer), render_page(service, "searchable", answer)):
        assert visible_preview in rendered and "(first 20 shown)" in rendered
        assert "Extraction completeness has not been assessed." in rendered


def test_oversized_author_metadata_is_unknown_on_completed_and_resumed_ingestion(tmp_path):
    source = write_pdf(tmp_path / "large-author.pdf", ["Large author needle.", None], author="A" * 65536)
    path = tmp_path / "large-author.sqlite"
    index = LocalIndex(path, create=True)
    try:
        completed = index.ingest(source)
        resumed = index.ingest(source, resume=True)
        assert completed["status"] == "completed" and resumed["status"] == "skipped"
        for result in (completed, resumed):
            assert result["extraction_coverage"] == UNKNOWN and result["pages_without_text"] is None
        raw = index.db.execute("SELECT metadata FROM books").fetchone()[0]
        assert len(raw) > 65536 and json.loads(raw)["pages_without_text"] == [2]
    finally:
        index.close()
    assert LibraryService(SQLiteRetriever(path)).ask("needle")["passages"][0]["extraction_coverage"] == UNKNOWN


def test_summaries_cover_search_scope_even_when_no_passages_match(library):
    service = library.service
    status = service.status()
    assert_summary(status["extraction_coverage"], (3, 4, 2, 1, 1, 2))
    reports = {}
    for label, counts in (("mixed", (1, 2, 1, 0, 1, 2)), ("zero", (1, 1, 1, 0, 0, 0)),
                          ("epub", (1, 1, 0, 1, 0, 0))):
        for question in ("unfindable quasar", "How can I?"):
            answer = service.ask(question, book_id=library.ids[label])
            assert answer["abstained"] and answer["passages"] == []
            assert_summary(answer["extraction_coverage"], counts, book_id=library.ids[label])
            reports[f"{label}:{question}"] = answer["extraction_coverage"]
    absent = service.ask("needle", book_id="0" * 64)
    assert absent["abstained"]
    assert_summary(absent["extraction_coverage"], (0, 0, 0, 0, 0, 0), book_id="0" * 64)
    unfiltered = service.ask("unfindable quasar")
    assert_summary(unfiltered["extraction_coverage"], (3, 4, 2, 1, 1, 2))
    save_evidence("coverage-service.json", {"fixture_provenance": "Original synthetic mixed/zero PDF and EPUB",
                  "status": status["extraction_coverage"], "filtered_no_matches": reports,
                  "nonexistent_filter": absent["extraction_coverage"], "passed": True})


def test_catalog_pagination_retains_reports_without_changing_the_envelope(library):
    expected = {library.ids["mixed"]: MIXED, library.ids["zero"]: ZERO, library.ids["epub"]: UNKNOWN}
    items, offset = [], 0
    for _ in range(3):
        page = library.service.books(limit=1, offset=offset)
        assert set(page) == {"books", "next_offset"} and len(page["books"]) == 1
        book = page["books"][0]
        assert book["extraction_coverage"] == expected[book["book_id"]]
        items.append(book["book_id"])
        offset = page["next_offset"]
    assert set(items) == set(expected) and offset is None
    assert library.service.books(offset=3) == {"books": [], "next_offset": None}


def test_empty_library_reports_an_empty_scope_without_assessing_completeness(tmp_path):
    path = tmp_path / "empty.sqlite"
    index = LocalIndex(path, create=True)
    index.close()
    service = LibraryService(SQLiteRetriever(path))
    assert_summary(service.status()["extraction_coverage"], (0, 0, 0, 0, 0, 0))
    answer = service.ask("needle")
    assert answer["abstained"] and answer["passages"] == []
    assert_summary(answer["extraction_coverage"], (0, 0, 0, 0, 0, 0))
    assert service.books() == {"books": [], "next_offset": None}


def test_removed_sources_are_not_needed_for_reports_or_exact_citations(library, monkeypatch):
    import ingest.local.index as index_module

    service = library.service
    answer, status, catalog = service.ask("needle"), service.status(), service.books()
    passage = answer["passages"][0]
    assert passage["quote"] == "Alpha needle evidence."
    assert passage["page_start"] == passage["page_end"] == 1
    assert passage["char_start"] == 0 and passage["char_end"] == len(passage["quote"])
    assert passage["source_sha256"] == hashlib.sha256(library.sources["mixed"].read_bytes()).hexdigest() == library.ids["mixed"]
    assert passage["citation_id"] == "[1]"
    assert passage["source_uri"] == f'librarian://books/{library.ids["mixed"]}/chunks/0'
    assert passage["extraction_coverage"] == MIXED
    assert service.get_passage(passage["chunk_id"]) == {key: value for key, value in passage.items() if key != "citation_id"}
    for source in library.sources.values():
        source.unlink()
    monkeypatch.setattr(index_module, "extract", lambda _path: pytest.fail("Read path attempted extraction"))
    before = library.path.read_bytes()
    assert service.ask("needle") == answer
    assert service.status() == status and service.books() == catalog
    assert service.get_passage(passage["chunk_id"])["quote"] == passage["quote"]
    assert library.path.read_bytes() == before
    serialized = json.dumps([answer, status, catalog])
    assert str(library.path.parent) not in serialized and "source_path" not in serialized


def test_legacy_metadata_is_reported_unknown_without_migration(library, tmp_path):
    books, chunks = indexed_rows(library.path)
    legacy = tmp_path / "legacy.sqlite"
    with sqlite3.connect(legacy) as db:
        for statement in BASE_SQL:
            db.execute(statement)
        for book in books:
            db.execute("INSERT INTO books VALUES (?, ?, ?, ?, ?)", (*book[:3], "{}", book[4]))
        db.executemany("INSERT INTO chunks(chunk_id, book_id, content, payload) VALUES (?, ?, ?, ?)", chunks)
    before = legacy.read_bytes()
    service = LibraryService(SQLiteRetriever(legacy))
    assert_summary(service.status()["extraction_coverage"], (3, 4, 0, 3, 0, 0))
    assert service.ask("needle")["passages"][0]["extraction_coverage"] == UNKNOWN
    assert legacy.read_bytes() == before


@pytest.mark.parametrize("failure", ["all-empty", "page-error"])
def test_failed_pdf_does_not_install_partial_book_and_retains_failed_receipt(library, monkeypatch, failure):
    source = write_pdf(library.path.parent / "failed.pdf", [None, None] if failure == "all-empty" else ["Partial first page.", None])
    before_rows = indexed_rows(library.path)
    original_answer = library.service.ask("needle")
    calls = []
    if failure == "page-error":
        def fail_second_page(_page, *_args, **_kwargs):
            calls.append(True)
            if len(calls) == 2:
                raise ValueError("injected page extraction failure")
            return "Partial first page."
        monkeypatch.setattr(PageObject, "extract_text", fail_second_page)
    index = LocalIndex(library.path, create=True)
    try:
        result = index.ingest_many([source])[0]
        assert result["status"] == "failed"
        assert ("OCR" if failure == "all-empty" else "injected page extraction failure") in result["error"]
        receipt = next(row for row in index.receipts() if row["source_path"] == str(source))
        assert receipt["status"] == "failed" and receipt["attempts"] == 1
    finally:
        index.close()
    assert indexed_rows(library.path) == before_rows
    assert library.service.ask("needle") == original_answer
    if failure == "page-error":
        assert len(calls) == 2


def test_all_empty_pdf_cli_still_exits_nonzero(library):
    source = write_pdf(library.path.parent / "empty-cli.pdf", [None])
    before = indexed_rows(library.path)
    result = subprocess.run([sys.executable, "-m", "ingest.local", "--index", str(library.path), "ingest", str(source)],
                            env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")},
                            capture_output=True, text=True, timeout=15, check=False)
    assert result.returncode == 1
    assert json.loads(result.stdout)[0]["status"] == "failed"
    assert "Pilot error" in result.stderr and "Traceback" not in result.stderr
    assert indexed_rows(library.path) == before


def test_cli_json_and_text_preserve_filtered_no_match_warnings(library):
    command = [sys.executable, "-m", "ingest.local", "--index", str(library.path), "ask",
               "missing evidence", "--book-id", library.ids["mixed"]]
    environment = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")}
    before = library.path.read_bytes()
    structured = subprocess.run([*command, "--json"], env=environment, capture_output=True, text=True, timeout=15, check=True)
    answer = json.loads(structured.stdout)
    assert answer == library.service.ask("missing evidence", book_id=library.ids["mixed"])
    assert answer["abstained"] and answer["extraction_coverage"]["reported_pages_without_text"] == 2
    plain = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=15, check=True)
    assert all(warning in plain.stdout for warning in answer["extraction_coverage"]["warnings"])
    assert str(library.path.parent) not in plain.stdout and "source_path" not in plain.stdout
    assert library.path.read_bytes() == before


@pytest.mark.parametrize("book_id", [None, "a" * 64])
def test_search_only_adapter_reports_unknown_scope_without_inventing_counts(book_id):
    service = LibraryService(SimpleNamespace(search=lambda *_args: []), backend="mock-semantic")
    answer = service.ask("missing evidence", book_id=book_id)
    assert answer["abstained"]
    assert_summary(answer["extraction_coverage"], (None,) * 6, book_id=book_id)
    assert "unavailable" in " ".join(answer["extraction_coverage"]["warnings"]).lower()


@pytest.mark.parametrize("error_type", [RuntimeError, AttributeError])
def test_implemented_coverage_errors_remain_service_errors(error_type):
    def unavailable(book_id=None):
        raise error_type("coverage backend failed")

    service = LibraryService(SimpleNamespace(search=lambda *_args: [], extraction_coverage=unavailable))
    with pytest.raises(error_type, match="coverage backend failed"):
        service.ask("missing evidence")


def test_legacy_status_preserves_outer_counts_and_reports_unknown_extraction():
    legacy_counts = {"indexed_books": 3, "indexed_chunks": 7}
    service = LibraryService(SimpleNamespace(coverage=lambda: legacy_counts), backend="mock-semantic")
    status = service.status()
    assert status["indexed_books"] == 3 and status["indexed_chunks"] == 7
    assert_summary(status["extraction_coverage"], (None,) * 6)
    assert "unavailable" in " ".join(status["extraction_coverage"]["warnings"]).lower()
    assert legacy_counts == {"indexed_books": 3, "indexed_chunks": 7}


def test_status_uses_optional_extraction_capability_when_legacy_counts_lack_it():
    legacy_counts = {"indexed_books": 3, "indexed_chunks": 7}
    expected = {
        "scope": "all_indexed", "book_id": None, "indexed_books": 3, "indexed_chunks": 7,
        "reported_page_coverage_books": 2, "unknown_page_coverage_books": 1,
        "books_with_reported_omissions": 1, "reported_pages_without_text": 2,
        "completeness": "not_assessed",
        "warnings": ["Two reported pages have no extractable text.",
                     "Omission information is unavailable for one book.",
                     "Extraction completeness has not been assessed."],
    }
    calls = []

    def extraction_coverage(book_id=None):
        calls.append(book_id)
        return expected

    service = LibraryService(SimpleNamespace(coverage=lambda: legacy_counts, extraction_coverage=extraction_coverage))
    status = service.status()
    assert status["extraction_coverage"] == expected
    assert status["indexed_books"] == 3 and status["indexed_chunks"] == 7
    assert calls == [None]
    assert legacy_counts == {"indexed_books": 3, "indexed_chunks": 7}


@pytest.mark.parametrize("error_type", [RuntimeError, AttributeError])
def test_status_does_not_hide_implemented_extraction_capability_errors(error_type):
    def unavailable(book_id=None):
        raise error_type("status extraction failed")

    retriever = SimpleNamespace(
        coverage=lambda: {"indexed_books": 3, "indexed_chunks": 7}, extraction_coverage=unavailable,
    )
    with pytest.raises(error_type, match="status extraction failed"):
        LibraryService(retriever).status()


@pytest.mark.parametrize("operation", ["ask", "status"])
def test_declared_extraction_property_binding_errors_are_not_treated_as_absence(operation):
    class BrokenPropertyRetriever:
        def search(self, _query, _limit, _book_id):
            return []

        def coverage(self):
            return {"indexed_books": 3, "indexed_chunks": 7}

        @property
        def extraction_coverage(self):
            raise AttributeError("declared extraction property failed")

    service = LibraryService(BrokenPropertyRetriever())
    with pytest.raises(AttributeError, match="declared extraction property failed"):
        if operation == "ask":
            service.ask("missing evidence", book_id="a" * 64)
        else:
            service.status()


def test_sqlite_status_reads_its_extraction_summary_once(library, monkeypatch):
    retriever = library.service.retriever
    original = retriever.extraction_coverage
    calls = []

    def observed(book_id=None):
        calls.append(book_id)
        return original(book_id)

    monkeypatch.setattr(retriever, "extraction_coverage", observed)
    before = library.path.read_bytes()
    status = library.service.status()
    assert calls == [None]
    assert_summary(status["extraction_coverage"], (3, 4, 2, 1, 1, 2))
    assert status["indexed_books"] == 3 and status["indexed_chunks"] == 4
    assert library.path.read_bytes() == before


def test_scope_warnings_are_visible_in_cli_text_and_html_even_on_no_match(library):
    for question in ("needle", "missing evidence"):
        answer = library.service.ask(question, book_id=library.ids["mixed"])
        text = render_answer(answer)
        html = render_page(library.service, question, answer)
        for warning in answer["extraction_coverage"]["warnings"]:
            assert warning in text and escape(warning) in html
        assert "source_path" not in text + html and str(library.path.parent) not in text + html
    answer["extraction_coverage"]["warnings"] = ['<script>untrusted coverage & "note"</script>']
    html = render_page(library.service, question, answer)
    assert "<script>" not in html and "&lt;script&gt;" in html


def test_returned_and_direct_passage_notes_show_page_numbers_and_completeness(library):
    answer = library.service.ask("needle")
    passage = library.service.get_passage(answer["passages"][0]["chunk_id"])
    direct = {"message": "Indexed source excerpt", "retrieval_query": "Direct passage lookup",
              "passages": [{**passage, "citation_id": "[1]"}]}
    for response in (answer, direct):
        for rendered in (render_answer(response), render_page(library.service, answer=response)):
            assert "2, 4" in rendered and "2 PDF pages" in rendered
            assert "Extraction completeness has not been assessed." in rendered
            assert "Alpha needle evidence." in rendered


class RegistrationDouble:
    def __init__(self):
        self.tools = {}

    def tool(self, *, annotations):
        def register(function):
            self.tools[function.__name__] = function
            return function
        return register


def test_registered_tools_forward_coverage_for_filtered_abstention(library):
    tools = register_tools(RegistrationDouble(), library.service, dict).tools
    assert tools["ask_library"]("missing evidence", book_id=library.ids["mixed"]) == library.service.ask(
        "missing evidence", book_id=library.ids["mixed"])
    assert tools["library_status"]() == library.service.status()
    assert tools["list_books"]() == library.service.books()


@pytest.fixture(scope="module")
def coverage_sdk():
    pytest.importorskip("mcp", reason="Real coverage wire checks require the approved optional MCP runtime")
    assert version("mcp") == "2.3.0"
    import anyio
    from mcp import Client, StdioServerParameters

    return anyio, Client, StdioServerParameters


@pytest.mark.parametrize("mode,protocol", [("auto", "2026-07-28"), ("legacy", "2025-11-25")])
def test_real_stdio_returns_the_same_coverage_and_preserves_database(library, coverage_sdk, mode, protocol):
    anyio, client_type, parameters_type = coverage_sdk
    before = hashlib.sha256(library.path.read_bytes()).hexdigest()
    parameters = parameters_type(command=sys.executable,
                                 args=["-m", "ingest.local.mcp_server", "--index", str(library.path)],
                                 env={"PYTHONPATH": str(Path(__file__).parents[1] / "src"), "PYTHONDONTWRITEBYTECODE": "1"})
    records = []

    async def round_trip():
        with anyio.fail_after(30):
            async with client_type(parameters, mode=mode, read_timeout_seconds=5, cache=None) as client:
                assert client.protocol_version == protocol

                async def call(name, arguments, expected):
                    result = await client.call_tool(name, arguments)
                    assert not result.is_error
                    assert result.structured_content == expected
                    assert len(result.content) == 1 and json.loads(result.content[0].text) == expected
                    serialized = result.model_dump_json(by_alias=True)
                    assert str(library.path.parent) not in serialized and "source_path" not in serialized
                    assert len(serialized.encode("utf-8")) < 128 * 1024
                    records.append({"tool": name, "arguments": arguments, "result": expected})

                await call("library_status", {}, library.service.status())
                await call("list_books", {}, library.service.books())
                for question, book_id in (("needle", library.ids["mixed"]), ("missing evidence", library.ids["mixed"]),
                                          ("How can I?", library.ids["mixed"]), ("needle", "0" * 64)):
                    await call("ask_library", {"question": question, "book_id": book_id},
                               library.service.ask(question, book_id=book_id))
                passage = library.service.ask("needle")["passages"][0]
                await call("get_passage", {"chunk_id": passage["chunk_id"]}, library.service.get_passage(passage["chunk_id"]))

    anyio.run(round_trip)
    assert hashlib.sha256(library.path.read_bytes()).hexdigest() == before
    save_evidence(f"coverage-mcp-{mode}.json", {"fixture_provenance": "Original synthetic mixed/zero PDF and EPUB",
                  "mode": mode, "protocol": protocol, "calls": records, "database_unchanged": True, "passed": True})
