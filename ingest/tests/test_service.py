import hashlib
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pytest
from test_local_pilot import epub, pdf

from ingest.local.index import LocalIndex
from ingest.local.mcp_adapter import READ_ONLY, register_tools
from ingest.local.service import (
    MAX_QUOTE_CHARS,
    LibraryService,
    SQLiteRetriever,
    public_passage,
    render_answer,
)
from ingest.local.web import make_server, render_page


@pytest.fixture
def library(tmp_path):
    path = tmp_path / "index.sqlite"
    index = LocalIndex(path, create=True)
    index.ingest(epub(tmp_path / "book.epub"))
    index.ingest(pdf(tmp_path / "book.pdf"))
    index.close()
    return path, LibraryService(SQLiteRetriever(path))


def test_human_question_citations_and_abstention(library):
    path, service = library
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    result = service.ask("What are regular expressions?")
    assert result["retrieval_query"] == "regular expressions"
    assert not result["abstained"] and result["answer_mode"] == "evidence_only"
    hit = result["passages"][0]
    assert hit["epub_member"] == "OPS/one.xhtml"
    assert hit["content_kind"] == "source_excerpt"
    assert "source_path" not in hit
    assert service.get_passage(hit["chunk_id"])["quote"] == hit["quote"]
    assert service.ask("quasar spectroscopy")["abstained"]
    assert service.ask("How can I?")["abstained"]
    pdf_result = service.ask("cybersecurity")
    assert pdf_result["passages"][0]["page_start"] == 2
    assert "PDF page 2" in render_answer(pdf_result)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_book_filter_and_bounded_inputs(library):
    _, service = library
    pdf_book = service.ask("cybersecurity")["passages"][0]["book_id"]
    assert service.ask("regular expressions", book_id=pdf_book)["abstained"]
    for question, limit in [("", 5), ("x" * 2001, 5), ("regular", 11), ("regular", True)]:
        with pytest.raises(ValueError):
            service.ask(question, limit)
    with pytest.raises(ValueError):
        service.get_passage("/etc/passwd")
    with pytest.raises(ValueError, match="not found"):
        service.get_passage("a" * 64 + ":0")
    assert service.status()["indexed_books"] == 2
    assert not service.status()["semantic_model_loaded"]


class FakeMCPServer:
    """Registration contract double only; this is not an MCP wire-protocol test."""

    def __init__(self):
        self.tools = {}
        self.annotations = {}

    def tool(self, *, annotations):
        def register(function):
            self.tools[function.__name__] = function
            self.annotations[function.__name__] = annotations
            return function
        return register


def test_mcp_adapter_and_human_share_read_only_backend(library):
    _, service = library
    server = register_tools(FakeMCPServer(), service, dict)
    assert set(server.tools) == {"ask_library", "get_passage", "list_books", "library_status"}
    assert all(value == READ_ONLY for value in server.annotations.values())
    expected = service.ask("What are regular expressions?")
    assert server.tools["ask_library"]("What are regular expressions?") == expected
    assert server.tools["get_passage"](expected["passages"][0]["chunk_id"])["quote"]
    assert server.tools["list_books"]() == service.books()
    assert server.tools["library_status"]()["read_only"]


def test_semantic_adapter_seam_preserves_original_question():
    class FakeSemanticRetriever:
        def search(self, query, limit, book_id):
            assert query == "What are regular expressions?"
            return []
    service = LibraryService(FakeSemanticRetriever(), backend="mock-semantic")
    assert service.ask("What are regular expressions?")["abstained"]


def test_human_cli_returns_structured_evidence(library):
    path, service = library
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")}
    result = subprocess.run([sys.executable, "-m", "ingest.local", "--index", str(path), "ask", "What are regular expressions?", "--json"], env=env, check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) == service.ask("What are regular expressions?")


def test_ui_escapes_source_text(library):
    _, service = library
    result = service.ask("regular")
    result["passages"][0]["quote"] = '<script>alert("source")</script>'
    html = render_page(service, '<img src=x onerror="bad">', result)
    assert "<script>" not in html and "<img src=x" not in html
    assert "&lt;script&gt;" in html and "Source details" in html


def test_loopback_http_question_round_trip(library):
    _, service = library
    server = make_server(service)
    assert server.server_address[0] == "127.0.0.1"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(url, timeout=3) as response:
            assert "Ask Librarian" in response.read().decode()
            assert response.headers["Cache-Control"] == "no-store"
        body = urlencode({"question": "What are regular expressions?"}).encode()
        with urlopen(Request(url + "/ask", data=body), timeout=3) as response:
            text = response.read().decode()
            assert "OPS/one.xhtml" in text and "Source excerpts found" in text
            if os.environ.get("LIBRARIAN_EVIDENCE_DIR"):
                evidence = Path(os.environ["LIBRARIAN_EVIDENCE_DIR"])
                evidence.mkdir(parents=True, exist_ok=True)
                (evidence / "human-preview-synthetic.html").write_text(text, encoding="utf-8")
                (evidence / "human-preview.json").write_text(json.dumps({
                    "fixture_provenance": "Original synthetic EPUB/PDF fixtures",
                    "transport": "loopback HTTP", "status": response.status,
                    "epub_location_present": True, "source_excerpt_present": True,
                    "visual_browser_review": False,
                }, indent=2) + "\n")
        with pytest.raises(HTTPError) as error:
            urlopen(Request(url, headers={"Host": "untrusted.example"}), timeout=3)
        assert error.value.code == 403
        with pytest.raises(HTTPError) as error:
            urlopen(Request(url + "/ask", data=body, headers={"Origin": "https://untrusted.example"}), timeout=3)
        assert error.value.code == 403
    finally:
        server.shutdown()
        thread.join(timeout=3)
        server.server_close()


def test_catalog_pagination_and_strict_service_inputs(library):
    _, service = library
    first = service.books(limit=1)
    second = service.books(limit=1, offset=first["next_offset"])
    assert len(first["books"]) == len(second["books"]) == 1
    assert first["books"][0]["book_id"] != second["books"][0]["book_id"]
    assert second["next_offset"] is None
    for limit, offset in [(True, 0), (51, 0), (1, -1), (1, "0")]:
        with pytest.raises(ValueError):
            service.books(limit, offset)
    for chunk_id in [None, 42, "a" * 64 + ":" + "1" * 100]:
        with pytest.raises(ValueError):
            service.get_passage(chunk_id)
    with pytest.raises(ValueError):
        service.ask("regular", book_id=7)
    with pytest.raises(ValueError, match="128"):
        service.ask("term " * 129)


def test_large_excerpt_retains_exact_bounded_source_span(library):
    from ingest.models import TextChunk

    path, service = library
    hit = service.ask("regular")["passages"][0]
    index = LocalIndex(path)
    try:
        row = index.db.execute("SELECT payload FROM chunks WHERE chunk_id=?", (hit["chunk_id"],)).fetchone()
    finally:
        index.close()
    chunk = TextChunk.model_validate_json(row[0])
    source = "prefix " + "界" * (MAX_QUOTE_CHARS + 2000)
    chunk.content = source[7:]
    chunk.metadata.update(char_start=7, char_end=len(source))
    chunk.metadata["title"] = "T" * 20000
    passage = public_passage(chunk)
    assert len(passage["quote"]) == MAX_QUOTE_CHARS
    assert passage["excerpt_truncated"]
    assert passage["original_char_end"] == len(source)
    assert source[passage["char_start"]:passage["char_end"]] == passage["quote"]
    assert len(passage["title"]) == 500


def test_adapter_masks_backend_failure_without_abstaining(library):
    _, service = library
    class BrokenRetriever:
        def search(self, *args):
            raise ValueError("private path /sensitive/library.sqlite SQL text")
    service.retriever = BrokenRetriever()
    server = register_tools(FakeMCPServer(), service, dict)
    with pytest.raises(ValueError, match="Library unavailable") as error:
        server.tools["ask_library"]("regular")
    assert "/sensitive" not in str(error.value)


def test_human_preview_reports_safe_unavailability(library):
    path, service = library
    server = make_server(service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    path.unlink()
    try:
        with pytest.raises(HTTPError) as error:
            urlopen(f"http://127.0.0.1:{server.server_port}", timeout=3)
        assert error.value.code == 503
        text = error.value.read().decode()
        assert "Library unavailable" in text and str(path) not in text
        assert not path.exists()
    finally:
        server.shutdown()
        thread.join(timeout=3)
        server.server_close()


def test_human_preview_late_match_and_continuation(tmp_path):
    source_text = ("a" * 30 + " ") * 399 + "needle"
    path = tmp_path / "index.sqlite"
    index = LocalIndex(path, create=True)
    try:
        assert index.ingest(epub(tmp_path / "long.epub", text=source_text))["chunks"] == 1
    finally:
        index.close()
    service = LibraryService(SQLiteRetriever(path))
    passage = service.ask("needle")["passages"][0]
    assert "needle" in passage["quote"]
    server = make_server(service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(Request(base + "/ask", data=urlencode({"question": "needle"}).encode()), timeout=3) as response:
            html = response.read().decode()
            assert "needle" in html and "Read from start" in html
        first = service.get_passage(passage["chunk_id"])
        route = "/passage?" + urlencode({"chunk_id": passage["chunk_id"], "offset": first["next_offset"]})
        with urlopen(base + route, timeout=3) as response:
            html = response.read().decode()
            assert "needle" in html and "Indexed source excerpt" in html
        with pytest.raises(HTTPError) as error:
            urlopen(base + "/passage?chunk_id=/etc/passwd", timeout=3)
        assert error.value.code == 400
    finally:
        server.shutdown()
        thread.join(timeout=3)
        server.server_close()


def test_corrupt_passage_is_unavailable_not_bad_request(library):
    path, service = library
    chunk_id = service.ask("regular")["passages"][0]["chunk_id"]
    index = LocalIndex(path, create=True)
    try:
        row = index.db.execute("SELECT payload FROM chunks WHERE chunk_id=?", (chunk_id,)).fetchone()
        payload = json.loads(row[0])
        payload["metadata"]["char_end"] += 1
        with index.db:
            index.db.execute("UPDATE chunks SET payload=? WHERE chunk_id=?", (json.dumps(payload), chunk_id))
    finally:
        index.close()
    assert service.status()["indexed_books"] == 2
    server = make_server(service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        requests = [base + "/passage?" + urlencode({"chunk_id": chunk_id}),
                    Request(base + "/ask", data=urlencode({"question": "regular"}).encode())]
        for request in requests:
            with pytest.raises(HTTPError) as error:
                urlopen(request, timeout=3)
            assert error.value.code == 503
            body = error.value.read().decode()
            assert "Library unavailable" in body and str(path) not in body
        with pytest.raises(HTTPError) as error:
            urlopen(base + "/passage?" + urlencode({"chunk_id": chunk_id, "offset": "invalid"}), timeout=3)
        assert error.value.code == 400
    finally:
        server.shutdown()
        thread.join(timeout=3)
        server.server_close()
