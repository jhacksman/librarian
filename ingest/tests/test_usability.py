"""U1 independent source oracle and real interface checks; execute only on Spark.

The unchanged oracle predates the policy freeze. USABILITY-POLICY.md governs the
interface; its historical draft notes do not weaken the frozen expectations.
"""

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import threading
from contextlib import closing, contextmanager
from html.parser import HTMLParser
from http.client import HTTPConnection
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pytest

from ingest.local.extract import extract
from ingest.local.index import LocalIndex
from ingest.local.service import LibraryRequestError, LibraryService, SQLiteRetriever
from ingest.local.web import make_server

ORACLE_SHA256 = "e750904e58a8a180e07dae4c84e2cfdd0cf1fc6a26d5a082472ab862652aff52"
UNKNOWN_ID = "0" * 64


def save_evidence(name, report):
    destination = os.environ.get("LIBRARIAN_EVIDENCE_DIR")
    if destination:
        root = Path(destination)
        root.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@pytest.fixture
def cases():
    raw = (Path(__file__).parents[1] / "evaluation" / "usability-cases.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == ORACLE_SHA256, "Do not rewrite the independent oracle"
    return json.loads(raw)


@pytest.fixture
def authoring(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "evaluation"))
    import quality_fixtures
    import quality_scoring

    return SimpleNamespace(pdf=quality_fixtures.write_pdf, epub=quality_fixtures.write_epub,
                           verify=quality_scoring.verify_citation)


def build_library(root, documents, authoring):
    """Generate files, then verify extraction against the separately authored text."""
    path = root / "usability.sqlite"
    oracle, ids, manifest, sources = {}, {}, [], {}
    index = LocalIndex(path, create=True)
    try:
        for document in documents:
            authored = document.get("pages", document.get("sections"))
            texts = [part["text"] for part in authored]
            source = root / f'{document["id"]}.{document["format"]}'
            writer = authoring.pdf if document["format"] == "pdf" else authoring.epub
            writer(source, {"title": document["title"], "sections": texts})
            content = extract(source)
            expected = {number: text for number, text in enumerate(texts, 1) if text}
            assert {chapter.number: chapter.content for chapter in content.chapters} == expected
            if document["format"] == "pdf":
                assert content.metadata["pages_without_text"] == [i for i, text in enumerate(texts, 1) if not text]
            receipt = index.ingest(source)
            book_id = receipt["book_id"]
            assert book_id == hashlib.sha256(source.read_bytes()).hexdigest()
            chunks = {}
            for row in index.db.execute("SELECT chunk_id, content, payload FROM chunks WHERE book_id=?", (book_id,)):
                payload = json.loads(row["payload"])
                section = payload["chapter_number"]
                start, end = payload["metadata"]["char_start"], payload["metadata"]["char_end"]
                assert row["content"] == expected[section][start:end]
                assert payload["id"] == row["chunk_id"] and payload["book_id"] == book_id
                chunks[row["chunk_id"]] = {"section": section, "char_start": start, "char_end": end}
            oracle[book_id] = {"label": document["id"], "title": document["title"],
                               "format": document["format"], "sections": expected, "chunks": chunks}
            ids[document["id"]] = book_id
            sources[document["id"]] = source
            manifest.append({"document": document["id"], "source_sha256": book_id,
                             "sections": len(expected), "chunks": len(chunks),
                             "extraction_coverage": receipt["extraction_coverage"]})
    finally:
        index.close()
    return SimpleNamespace(path=path, oracle=oracle, ids=ids, manifest=manifest, sources=sources,
                           service=LibraryService(SQLiteRetriever(path)), authoring=authoring)


@pytest.fixture
def library(tmp_path, cases, authoring):
    return build_library(tmp_path, cases["documents"], authoring)


def assert_coverage(library, answer, label):
    summary = answer["extraction_coverage"]
    selected = library.ids[label] if label else None
    assert summary["scope"] == ("book" if label else "all_indexed")
    assert summary["book_id"] == selected
    assert summary["completeness"] == "not_assessed"
    assert "completeness" in " ".join(summary["warnings"]).lower()
    fields = ("indexed_books", "indexed_chunks", "reported_page_coverage_books",
              "unknown_page_coverage_books", "books_with_reported_omissions", "reported_pages_without_text")
    expected = {None: (3, 9, 2, 1, 1, 1), "station-2022": (1, 3, 1, 0, 0, 0),
                "station-2026": (1, 4, 1, 0, 1, 1), "bench-notes": (1, 2, 0, 1, 0, 0)}
    assert tuple(summary[field] for field in fields) == expected[label]


def assess_case(library, case, answer):
    assert answer["question"] == case["question"] and answer["answer_mode"] == "evidence_only"
    assert answer["abstained"] == (not answer["passages"])
    assert "answer" not in answer and "generated_answer" not in answer
    assert "not proof" in answer["limitations"]
    assert len(answer["passages"]) <= 5
    if "expected_literals" in case:
        assert answer["query_normalization"]["literal_constraints"] == case["expected_literals"]
    assert_coverage(library, answer, case["book_filter"])
    observed = []
    for rank, passage in enumerate(answer["passages"], 1):
        section = library.authoring.verify(passage, library.oracle)
        assert passage["citation_id"] == f"[{rank}]"
        label = library.oracle[passage["book_id"]]["label"]
        assert case["book_filter"] is None or label == case["book_filter"]
        page_report = passage["extraction_coverage"]
        assert page_report["completeness"] == "not_assessed"
        if label == "station-2026":
            assert page_report["pages_without_text"] == [2]
        elif label == "station-2022":
            assert page_report["pages_without_text"] == []
        else:
            assert page_report["page_omissions"] == "unknown" and page_report["pages_without_text"] is None
        observed.append((label, section, passage["quote"]))
    relevant = [(item["document"], item.get("physical_page") or item["section"], item["quote"])
                for item in case["relevant"]]
    missing = [quote for label, section, quote in relevant
               if not any(label == found_label and section == found_section and quote in found_quote
                          for found_label, found_section, found_quote in observed)]
    unexpected = [(label, section) for label, section, _ in observed
                  if (label, section) not in {(item[0], item[1]) for item in relevant}]
    passed = not missing and not unexpected and answer["abstained"] == case["expected_no_match"]
    return {"id": case["id"], "gate": not case["proposed_role"].startswith("diagnostic only"),
            "question": case["question"], "book_filter": case["book_filter"],
            "answerability": case["answerability"], "expected_no_match": case["expected_no_match"],
            "observed_no_match": answer["abstained"], "missing_authored_quotes": missing,
            "unexpected_locations": unexpected, "citation_checks_passed": True,
            "passed": passed, "answer": answer}


def test_twelve_independent_scenarios_keep_retrieval_and_answerability_separate(library, cases):
    before = library.path.read_bytes()
    report = {"oracle_sha256": ORACLE_SHA256, "fixture_manifest": library.manifest,
              "scope": "Small disclosed synthetic evidence exercise; no semantic or performance claim",
              "completed": False, "rows": []}
    try:
        assert len(cases["scenarios"]) == 12
        for case in cases["scenarios"]:
            row = {"id": case["id"], "completed": False}
            report["rows"].append(row)
            answer = library.service.ask(case["question"], book_id=library.ids.get(case["book_filter"]))
            row["answer"] = answer  # Retain the observed failure before any oracle assertion.
            row.update(assess_case(library, case, answer))
            row["completed"] = True
        report["gate_failures"] = [row["id"] for row in report["rows"] if row["gate"] and not row["passed"]]
        report["diagnostic_failures"] = [row["id"] for row in report["rows"] if not row["gate"] and not row["passed"]]
        report["completed"] = True
        assert sum(row["gate"] for row in report["rows"]) == 11
        assert not report["gate_failures"], report["gate_failures"]
    finally:
        report["database_unchanged"] = library.path.read_bytes() == before
        save_evidence("usability-evaluation.json", report)
        save_evidence("usability-diagnostics.json", {
            "oracle_sha256": ORACLE_SHA256,
            "rows": [row for row in report["rows"] if row.get("gate") is False],
            "note": "An authored useful source remains relevant even if lexical retrieval misses it.",
        })
    assert report["database_unchanged"]


class Page(HTMLParser):
    """Read successful form controls and quotations from the actual served HTML."""

    def __init__(self, html):
        super().__init__(convert_charrefs=True)
        self.html = html
        self.forms, self.buttons, self.quotes, self.text, self.urls = [], [], [], [], []
        self.articles, self.article = [], None
        self.tags = []
        self.scope_parts, self.heading, self.capture_scope = [], None, False
        self.current_form = self.textarea = self.button = self.quote = None
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.tags.append(tag)
        for key in ("href", "action", "formaction"):
            if key in attrs:
                self.urls.append(attrs[key])
        if tag == "form":
            assert self.current_form is None, "Nested forms do not preserve browser submission state"
            self.current_form = {"attrs": attrs, "fields": {}}
            self.forms.append(self.current_form)
        elif tag == "input" and self.current_form is not None and "name" in attrs and "disabled" not in attrs:
            assert attrs["name"] not in self.current_form["fields"], "Duplicate successful form control"
            self.current_form["fields"][attrs["name"]] = attrs.get("value", "")
        elif tag == "textarea":
            assert self.current_form is not None, "Question input must belong to the submitted form"
            assert "disabled" not in attrs and attrs["name"] not in self.current_form["fields"]
            self.capture_scope = False
            self.textarea = attrs["name"]
            self.current_form["fields"][self.textarea] = ""
        elif tag == "button":
            self.button = {"attrs": attrs, "form": self.current_form, "text": ""}
            self.buttons.append(self.button)
        elif tag == "blockquote":
            self.quote = ""
        elif tag == "h2":
            self.heading, self.capture_scope = "", False
        elif tag == "article":
            self.article = {"text": [], "quotes": []}

    def handle_endtag(self, tag):
        if tag == "form":
            self.current_form = None
        elif tag == "textarea":
            self.textarea = None
        elif tag == "button":
            self.button = None
        elif tag == "blockquote":
            self.quotes.append(self.quote)
            if self.article is not None:
                self.article["quotes"].append(self.quote)
            self.quote = None
        elif tag == "h2":
            if self.heading.startswith(("Search this book:", "Search scope:", "Unavailable book selection")):
                self.scope_parts = [self.heading]
                self.capture_scope = True
            self.heading = None
        elif tag == "article":
            self.articles.append(self.article)
            self.article = None

    def handle_data(self, data):
        self.text.append(data)
        if self.article is not None:
            self.article["text"].append(data)
        if self.heading is not None:
            self.heading += data
        elif self.capture_scope:
            self.scope_parts.append(data)
        if self.textarea is not None:
            self.current_form["fields"][self.textarea] += data
        if self.button is not None:
            self.button["text"] += data
        if self.quote is not None:
            self.quote += data

    @property
    def visible(self):
        return " ".join(self.text)

    @property
    def scope(self):
        return " ".join(self.scope_parts)

    @property
    def state(self):
        forms = [form for form in self.forms if "question" in form["fields"]]
        assert len(forms) == 1
        return forms[0]["fields"]

    def find_button(self, *, name=None, value=None, text=None):
        matches = [button for button in self.buttons
                   if (name is None or button["attrs"].get("name") == name)
                   and (value is None or button["attrs"].get("value") == value)
                   and (text is None or button["text"].strip() == text)]
        assert len(matches) == 1, (name, value, text, len(matches))
        return matches[0]

    def submission(self, button, **edits):
        form = button["form"]
        attrs = button["attrs"]
        assert form is not None and "question" in form["fields"]
        assert attrs.get("type", "submit").lower() == "submit", "Navigation requires a submit button"
        method = attrs.get("formmethod", form["attrs"].get("method", "get"))
        assert method.lower() == "post", "Navigation requires an effective POST method"
        assert "disabled" not in attrs, "Disabled buttons are not successful submit controls"
        assert edits.keys() <= form["fields"].keys()
        fields = {**form["fields"], **edits}
        action = attrs.get("formaction", form["attrs"]["action"])
        if action in {"/browse", "/passage"}:
            assert "formnovalidate" in attrs, "Navigation must work with an empty required question"
        if "name" in attrs:
            assert attrs["name"] not in fields, "Clicked control duplicates a successful form field"
            fields[attrs["name"]] = attrs.get("value", "")
        assert "?" not in action and all("question=" not in url for url in self.urls)
        return action, list(fields.items())


@pytest.mark.parametrize("method,button_attrs", [("post", ""), ("POST", 'type="submit"'),
                                               ("get", 'formmethod="POST"')])
def test_form_helper_accepts_default_submit_and_effective_post(method, button_attrs):
    page = Page(f'<form method="{method}" action="/ask"><input name="book_id" value="">'
                '<textarea name="question" required>draft</textarea>'
                f'<button {button_attrs} name="choose_book" value="" formaction="/browse" formnovalidate>All books</button></form>')
    action, fields = page.submission(page.find_button(text="All books"), question="edited & preserved")
    assert action == "/browse"
    assert fields == [("book_id", ""), ("question", "edited & preserved"), ("choose_book", "")]


@pytest.mark.parametrize("attrs,message", [('type="button"', "submit button"),
                                         ('type="reset"', "submit button"),
                                         ('formmethod="GET"', "effective POST"),
                                         ("disabled", "Disabled buttons")])
def test_form_helper_rejects_controls_that_cannot_submit_post(attrs, message):
    page = Page('<form method="post" action="/ask"><textarea name="question">draft</textarea>'
                f'<button {attrs} formaction="/browse" formnovalidate>Browse</button></form>')
    with pytest.raises(AssertionError, match=message):
        page.submission(page.find_button(text="Browse"))


@pytest.mark.parametrize("controls", [
    '<input name="book_id" value="first"><input name="book_id" value="second"><textarea name="question">draft</textarea>',
    '<textarea name="question">first</textarea><textarea name="question">second</textarea>',
    '<textarea name="question" disabled>not submitted</textarea>',
])
def test_form_helper_rejects_duplicate_controls_and_disabled_question(controls):
    with pytest.raises(AssertionError):
        Page(f'<form method="post" action="/ask">{controls}<button>Find</button></form>')


def test_form_helper_excludes_disabled_hidden_controls():
    page = Page('<form method="post" action="/ask"><input name="book_id" value="discarded" disabled>'
                '<input name="book_id" value="retained"><textarea name="question">draft</textarea>'
                '<button>Find</button></form>')
    action, fields = page.submission(page.find_button(text="Find"))
    assert action == "/ask" and fields == [("book_id", "retained"), ("question", "draft")]


def test_form_helper_rejects_clicked_button_collision_with_hidden_field():
    page = Page('<form method="post" action="/ask"><input name="choose_book" value="hidden">'
                '<textarea name="question">draft</textarea><button name="choose_book" value="clicked" '
                'formaction="/browse" formnovalidate>Choose</button></form>')
    with pytest.raises(AssertionError, match="Clicked control duplicates"):
        page.submission(page.find_button(text="Choose"))


@contextmanager
def browser(service):
    server = make_server(service, port=0)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"

    def request(path="/", fields=None, headers=None):
        data = urlencode(fields).encode("utf-8") if fields is not None else None
        req = Request(base + path, data=data, headers=headers or {})
        try:
            response = urlopen(req, timeout=5)
        except HTTPError as error:
            response = error
        with response:
            status, html = response.status, response.read().decode("utf-8")
        return status, Page(html)

    request.port = server.server_port
    try:
        yield request
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
        assert not thread.is_alive()


def click(request, page, *, name=None, value=None, text=None, **edits):
    return request(*page.submission(page.find_button(name=name, value=value, text=text), **edits))


def assert_state(page, question, book_id, offset=0):
    assert page.state == {"question": question, "book_id": book_id or "", "catalog_offset": str(offset)}
    if book_id:
        assert f"Book ID: {book_id}" in page.scope
        assert "Search this book:" in page.scope
    else:
        assert "Search scope: All books" in page.scope


def assert_rendered_passages(page, passages, library):
    assert page.quotes == [passage["quote"] for passage in passages]
    assert len(page.articles) == len(passages)
    for article, passage in zip(page.articles, passages, strict=True):
        visible = " ".join(article["text"])
        library.authoring.verify(passage, library.oracle)
        location = f'PDF page {passage["page_start"]}' if passage["page_start"] else f'EPUB {passage["epub_member"]}'
        assert article["quotes"] == [passage["quote"]]
        assert location in visible
        assert f'{passage["citation_id"]} {passage["title"]}' in visible
        assert f'Passage: {passage["chunk_id"]}' in visible
        assert f'SHA-256: {passage["source_sha256"]}' in visible
        assert f'Source characters: {passage["char_start"]}–{passage["char_end"]}' in visible
        assert "Extraction completeness has not been assessed." in visible
        if passage["extraction_coverage"]["page_omissions"] == "unknown":
            assert "omission information is unavailable for this book" in visible
        elif passage["extraction_coverage"]["pages_without_text_count"]:
            assert "not searchable: 2." in visible


def test_http_choose_same_title_editions_miss_then_return_to_all(library, monkeypatch):
    before = library.path.read_bytes()
    for source in library.sources.values():
        source.unlink()  # Browsing and citations must work from the read-only stored index.
    original_ask, calls = library.service.ask, []
    evidence_dir = os.environ.get("LIBRARIAN_EVIDENCE_DIR")
    demo_root = Path(evidence_dir) / "usability-demo" if evidence_dir else None
    demo = None
    if demo_root is not None:
        project = Path(__file__).parents[1]
        application_files = sorted([*(project / "src" / "ingest" / "local").glob("*.py"),
                                    project / "src" / "ingest" / "models.py"])
        demo = {
            "schema_version": 1,
            "capture_kind": "Actual loopback HTTP response bodies; static captures, not a live demo",
            "completed": False,
            "frozen_oracle_sha256": ORACLE_SHA256,
            "test_file_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "application_file_sha256": {str(path.relative_to(project)): hashlib.sha256(path.read_bytes()).hexdigest()
                                        for path in application_files},
            "fixture_manifest": library.manifest,
            "book_ids": dict(library.ids),
            "source_files_removed_before_requests": True,
            "database_sha256_before": hashlib.sha256(before).hexdigest(),
            "database_sha256_after": None,
            "database_unchanged": False,
            "captures": [],
        }
        demo_root.mkdir(parents=True, exist_ok=True)

    def persist_demo():
        if demo_root is not None:
            (demo_root / "manifest.json").write_text(json.dumps(demo, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def capture_demo(filename, status, page):
        if demo_root is not None:
            # browser() strictly decodes UTF-8, so this round trip preserves the
            # response body bytes, including whitespace; no HTML is rewritten.
            body = page.html.encode("utf-8")
            (demo_root / filename).write_bytes(body)
            demo["captures"].append({"filename": filename, "html_bytes": len(body),
                                     "html_sha256": hashlib.sha256(body).hexdigest(),
                                     "status": status, "current_form_state": dict(page.state)})
            demo["database_sha256_after"] = hashlib.sha256(library.path.read_bytes()).hexdigest()
            persist_demo()

    persist_demo()  # A partial or failed run must not leave an old completed manifest.

    def observed_ask(*args, **kwargs):
        calls.append((args, kwargs))
        return original_ask(*args, **kwargs)

    monkeypatch.setattr(library.service, "ask", observed_ask)
    transcript = []
    with browser(library.service) as request:
        status, page = request()
        assert status == 200 and not calls
        for label in ("station-2022", "station-2026"):
            assert f"Book ID: {library.ids[label]}" in page.visible
        assert page.visible.count("Station Logbook") == 2
        capture_demo("01-catalog.html", status, page)
        for label, question, quote in [
            ("station-2026", "What is the observation interval?", "42 minutes"),
            ("station-2026", "blue ledger", None),
            ("station-2026", "--trace-log", None),
            ("station-2022", "blue ledger", "blue ledger"),
            ("bench-notes", "What is the bench serial label?", "SB-73"),
        ]:
            count = len(calls)
            status, page = click(request, page, name="choose_book", value=library.ids[label], question=question)
            assert status == 200 and len(calls) == count
            assert_state(page, question, library.ids[label])
            if label == "station-2026" and question == "What is the observation interval?":
                capture_demo("02-selected-2026.html", status, page)
            status, page = click(request, page, text="Find source passages")
            expected = original_ask(question, book_id=library.ids[label])
            assert status == 200 and len(calls) == count + 1
            assert_rendered_passages(page, expected["passages"], library)
            assert bool(page.quotes) == (quote is not None)
            if quote:
                assert quote in page.quotes[0]
            else:
                assert "No matching evidence" in page.visible
            assert_state(page, question, library.ids[label])
            assert "completeness" in page.scope.lower()
            if label == "station-2026":
                assert "not searchable: 2." in page.scope
            elif label == "bench-notes":
                assert "omission information is unavailable for this book" in page.scope
            transcript.append({"question": question, "book_id": library.ids[label], "answer": expected})
            if label == "station-2026" and question == "What is the observation interval?":
                capture_demo("03-selected-42-minute-answer.html", status, page)
            elif label == "station-2026" and question == "blue ledger":
                capture_demo("04-scoped-blue-ledger-no-match.html", status, page)
        count = len(calls)
        question = "What is the observation interval?"
        status, page = click(request, page, name="choose_book", value="", question=question)
        assert status == 200 and len(calls) == count
        assert_state(page, question, None)
        status, page = click(request, page, text="Find source passages")
        expected = original_ask(question)
        assert status == 200 and len(calls) == count + 1
        assert len(page.quotes) == 2 and page.quotes == [item["quote"] for item in expected["passages"]]
        assert_rendered_passages(page, expected["passages"], library)
        assert "[1] Station Logbook" in page.visible and "[2] Station Logbook" in page.visible
        assert str(library.path.parent) not in page.html and "source_path" not in page.html
        capture_demo("05-all-books-interval-result.html", status, page)
    assert library.path.read_bytes() == before
    save_evidence("usability-http-flow.json", {"calls": transcript, "database_unchanged": True, "source_files_removed": True,
                                              "scope": "Actual HTTP and parsed HTML controls; not visual browser verification"})
    if demo is not None:
        assert len(demo["captures"]) == 5
        demo["database_sha256_after"] = hashlib.sha256(library.path.read_bytes()).hexdigest()
        demo["database_unchanged"] = True
        demo["completed"] = True
        persist_demo()


def test_http_catalog_pages_preserve_unsent_question_and_off_page_selection(tmp_path, cases, authoring, monkeypatch):
    extras = [{"id": f"shelf-{i}", "format": "epub", "title": f"Shelf volume {i}",
               "sections": [{"text": f"Shelfmarker{i} identifies this original catalog fixture."}]} for i in range(22)]
    library = build_library(tmp_path, [*cases["documents"], *extras], authoring)
    before = library.path.read_bytes()
    monkeypatch.setattr(library.service, "ask", lambda *args, **kwargs: pytest.fail("Catalog navigation ran a search"))
    original_books, lookups = library.service.books, []

    def books(*args, **kwargs):
        lookups.append((args, kwargs))
        return original_books(*args, **kwargs)

    monkeypatch.setattr(library.service, "books", books)
    with browser(library.service) as request:
        status, page = request()
        first_ids = [b["attrs"]["value"] for b in page.buttons if b["attrs"].get("name") == "choose_book" and b["attrs"].get("value")]
        assert status == 200 and len(first_ids) == 20
        selected = first_ids[0]
        status, page = click(request, page, name="choose_book", value=selected, question="")
        assert status == 200
        question = '<script>unsent & "quoted" --trace-log</script>'
        status, page = click(request, page, text="Next books", question=question)
        assert status == 200
        assert_state(page, question, selected, 20)
        assert "Extraction completeness has not been assessed." in page.scope
        second_ids = [b["attrs"]["value"] for b in page.buttons if b["attrs"].get("name") == "choose_book" and b["attrs"].get("value")]
        assert len(second_ids) == 5 and not set(first_ids) & set(second_ids)
        assert "script" not in page.tags
        assert set(first_ids + second_ids) == set(library.ids.values())
        status, page = click(request, page, name="choose_book", value=second_ids[0])
        assert status == 200
        status, page = click(request, page, text="First books")
        assert status == 200
        assert_state(page, question, second_ids[0])
        assert "Extraction completeness has not been assessed." in page.scope
        assert all("question=" not in url for url in page.urls)
    assert any(kwargs == {"limit": 1, "offset": 0, "book_id": selected} for _, kwargs in lookups)
    assert all(not args and kwargs["limit"] == (1 if kwargs.get("book_id") else 20) for args, kwargs in lookups)
    assert library.path.read_bytes() == before


def test_http_full_id_labels_distinguish_common_prefix_and_escape_titles():
    ids = ["a" * 63 + "1", "a" * 63 + "2"]
    title = '<img src=x onerror="alert(1)"> & same title'
    coverage = {"page_omissions": "unknown", "pages_without_text_count": None,
                "pages_without_text": None, "pages_without_text_truncated": False, "completeness": "not_assessed"}

    class CatalogOnly:
        def books(self, limit, offset, book_id=None):
            rows = [{"book_id": identifier, "title": title, "chunk_count": 1,
                     "extraction_coverage": coverage} for identifier in ids if book_id in {None, identifier}]
            return rows[offset:offset + limit]

        def coverage(self):
            return {"indexed_books": 2, "indexed_chunks": 2}

    # Catalog identities are deliberately synthetic; no source/citation claim uses them.
    with browser(LibraryService(CatalogOnly())) as request:
        status, page = request()
        assert status == 200 and "img" not in page.tags
        for identifier in ids:
            assert f"Book ID: {identifier}" in page.visible
            status, page = click(request, page, name="choose_book", value=identifier, question="untouched")
            assert status == 200
            assert_state(page, "untouched", identifier)
            assert f"Search this book: {title}" in page.visible and "img" not in page.tags


def test_http_long_unicode_excerpt_navigation_preserves_edited_state(tmp_path, authoring, monkeypatch):
    words = ["openingmarker", *["界" * 49 + str(i) for i in range(180)], "middlemarker",
             *["🧭" * 49 + str(i) for i in range(180)], "closingmarker"]
    source_text = " ".join(words)
    documents = [{"id": "long", "format": "epub", "title": "Unicode excerpt <&>",
                  "sections": [{"text": source_text}]},
                 {"id": "other", "format": "epub", "title": "Other book", "sections": [{"text": "Other source."}]}]
    library = build_library(tmp_path, documents, authoring)
    before = library.path.read_bytes()
    original_ask, searches = library.service.ask, []

    def ask(*args, **kwargs):
        searches.append(args)
        return original_ask(*args, **kwargs)

    monkeypatch.setattr(library.service, "ask", ask)
    selected = library.ids["long"]
    expected = original_ask("middlemarker", book_id=selected)["passages"][0]
    assert expected["excerpt_offset"] > 0 and expected["next_offset"] is not None
    assert len(library.oracle[selected]["chunks"]) == 1
    with browser(library.service) as request:
        _, page = request()
        _, page = click(request, page, name="choose_book", value=selected, question="middlemarker")
        status, page = click(request, page, text="Find source passages")
        assert status == 200 and page.quotes == [expected["quote"]]
        assert_rendered_passages(page, [expected], library)
        assert len(searches) == 1
        question = 'Unsent <&> edits --trace-log "preserved"'
        status, page = click(request, page, text="Read from start", question=question, catalog_offset="20")
        assert status == 200
        pieces, offset = [], 0
        for _ in range(5):
            passage = library.service.get_passage(expected["chunk_id"], offset)
            authoring.verify(passage, library.oracle)
            assert page.quotes == [source_text[passage["char_start"]:passage["char_end"]]] == [passage["quote"]]
            assert_rendered_passages(page, [{**passage, "citation_id": "[1]"}], library)
            assert_state(page, question, selected, 20)
            assert "[1] Unicode excerpt <&>" in page.visible
            assert expected["chunk_id"] in page.visible and "completeness" in page.visible.lower()
            pieces.append(passage["quote"])
            if passage["next_offset"] is None:
                assert not any(b["text"].strip() == "Continue excerpt" for b in page.buttons)
                break
            offset = passage["next_offset"]
            status, page = click(request, page, text="Continue excerpt")
            assert status == 200
        else:
            pytest.fail("Continuation failed to reach the source end")
        assert "".join(pieces) == source_text and len(searches) == 1
        status, direct = request("/passage?" + urlencode({"chunk_id": expected["chunk_id"], "offset": 0}))
        assert status == 200 and direct.quotes == [source_text[:8000]]
        status, rejected = request("/passage", [("question", question), ("book_id", library.ids["other"]),
                                                ("catalog_offset", "20"), ("passage_location", expected["chunk_id"] + "@0")])
        assert status == 400 and not rejected.quotes and "outside the selected book" in rejected.visible
        assert_state(rejected, question, library.ids["other"], 20)
        assert len(searches) == 1
    assert library.path.read_bytes() == before


@pytest.mark.parametrize("path,extra", [
    ("/ask", [("unknown", "value")]),
    ("/ask", [("question", "ambiguous")]),
    ("/browse", [("choose_book", ""), ("page_offset", "20")]),
    ("/browse", [("page_offset", "1000001")]),
    ("/browse", []),
    ("/passage", [("passage_location", "not-a-location")]),
])
def test_http_parseable_errors_keep_question_and_scope_without_search(library, monkeypatch, path, extra):
    monkeypatch.setattr(library.service, "ask", lambda *args, **kwargs: pytest.fail("Invalid request ran a search"))
    before = library.path.read_bytes()
    question, selected = "Unsent <&> --trace-log", library.ids["station-2026"]
    with browser(library.service) as request:
        status, page = request(path, [("question", question), ("book_id", selected), ("catalog_offset", "0"), *extra])
        assert status == 400 and not page.quotes
        assert_state(page, question, selected)
    assert library.path.read_bytes() == before


@pytest.mark.parametrize("selection", [UNKNOWN_ID, "A" * 64, "bad", "repeated"])
def test_http_invalid_or_ambiguous_selection_never_becomes_all_books(library, monkeypatch, selection):
    monkeypatch.setattr(library.service, "ask", lambda *args, **kwargs: pytest.fail("Unavailable scope ran a search"))
    selected = library.ids["station-2026"] if selection == "repeated" else selection
    fields = [("question", "blue ledger"), ("book_id", selected), ("catalog_offset", "0")]
    if selection == "repeated":
        fields.append(("book_id", library.ids["station-2022"]))
    with browser(library.service) as request:
        status, page = request("/ask", fields)
        assert status == 400 and not page.quotes
        assert page.state["question"] == "blue ledger"
        assert page.state["book_id"] and "Unavailable book selection" in page.visible
        assert "Search scope: All books" not in page.visible
        assert "disabled" in page.find_button(text="Find source passages")["attrs"]
        status, page = click(request, page, name="choose_book", value="")
        assert status == 200
        assert_state(page, "blue ledger", None)


def test_http_removed_selection_is_explicit_and_does_not_search(library, monkeypatch):
    selected = library.ids["station-2026"]
    with browser(library.service) as request:
        _, page = request()
        _, page = click(request, page, name="choose_book", value=selected, question="blue ledger")
        with sqlite3.connect(library.path) as db:
            db.execute("DELETE FROM chunks WHERE book_id=?", (selected,))
            db.execute("DELETE FROM books WHERE book_id=?", (selected,))
        before = library.path.read_bytes()
        monkeypatch.setattr(library.service, "ask", lambda *args, **kwargs: pytest.fail("Removed scope ran a search"))
        status, page = click(request, page, text="Find source passages")
        assert status == 400 and "no longer indexed" in page.visible and not page.quotes
        assert page.state["book_id"] == selected and page.state["question"] == "blue ledger"
        assert library.path.read_bytes() == before


@pytest.mark.parametrize("path,method,extra", [
    ("/browse", "books", [("choose_book", "")]),
    ("/ask", "ask", []),
    ("/passage", "get_passage", [("passage_location", UNKNOWN_ID + ":0@0")]),
])
def test_http_internal_failure_is_safe_503_not_abstention(library, monkeypatch, path, method, extra):
    def fail(*args, **kwargs):
        raise RuntimeError(str(library.path) + " private failure detail")

    monkeypatch.setattr(library.service, method, fail)
    with browser(library.service) as request:
        status, page = request(path, [("question", "blue ledger"), ("book_id", ""), ("catalog_offset", "0"), *extra])
        assert status == 503 and "Library unavailable" in page.visible
        assert "No matching evidence" not in page.visible and "private failure" not in page.visible
        assert str(library.path) not in page.html


@pytest.mark.parametrize("path,extra", [("/browse", [("choose_book", "")]),
                                       ("/passage", [("passage_location", UNKNOWN_ID + ":0@0")])])
@pytest.mark.parametrize("headers", [{"Origin": "https://outside.invalid"}, {"Host": "outside.invalid"}])
def test_new_post_routes_keep_origin_and_host_guards(library, monkeypatch, path, extra, headers):
    monkeypatch.setattr(library.service, "books", lambda *args, **kwargs: pytest.fail("Rejected origin read catalog"))
    with browser(library.service) as request:
        status, page = request(path, [("question", "private question"), *extra], headers=headers)
        assert status == 403 and "private question" not in page.html


@pytest.mark.parametrize("path", ["/browse", "/passage"])
@pytest.mark.parametrize("headers", [[("Transfer-Encoding", "chunked")],
                                     [("Content-Length", "0"), ("Content-Length", "0")],
                                     [("Content-Length", "16001")]])
def test_new_post_routes_reject_ambiguous_or_oversized_framing(library, monkeypatch, path, headers):
    # An error page may read the catalog to render; it must not execute an action.
    monkeypatch.setattr(library.service, "ask", lambda *args, **kwargs: pytest.fail("Invalid framing ran a search"))
    monkeypatch.setattr(library.service, "get_passage", lambda *args, **kwargs: pytest.fail("Invalid framing read a passage"))
    with browser(library.service) as request, closing(HTTPConnection("127.0.0.1", request.port, timeout=5)) as connection:
        connection.putrequest("POST", path)
        for name, value in headers:
            connection.putheader(name, value)
        connection.endheaders()
        with connection.getresponse() as response:
            assert response.status == 400
            assert b"No matching evidence" not in response.read()


def test_empty_catalog_can_browse_and_ask_without_inventing_selection(tmp_path):
    path = tmp_path / "empty.sqlite"
    index = LocalIndex(path, create=True)
    index.close()
    before = path.read_bytes()
    with browser(LibraryService(SQLiteRetriever(path))) as request:
        status, page = request()
        assert status == 200 and "0 books" in page.visible
        assert not [b for b in page.buttons if b["attrs"].get("name") == "choose_book" and b["attrs"].get("value")]
        status, page = click(request, page, name="choose_book", value="", question="absent observation")
        assert status == 200
        status, page = click(request, page, text="Find source passages")
        assert status == 200 and not page.quotes and "No matching evidence" in page.visible
        assert_state(page, "absent observation", None)
    assert path.read_bytes() == before


def test_filtered_catalog_applies_identity_before_offset_and_keeps_legacy_calls(library):
    before = library.path.read_bytes()
    chosen = sorted(library.ids.values())[-1]
    result = library.service.books(limit=1, offset=0, book_id=chosen)
    assert [book["book_id"] for book in result["books"]] == [chosen] and result["next_offset"] is None
    assert library.service.books(limit=1, offset=1, book_id=chosen) == {"books": [], "next_offset": None}
    assert library.service.books(book_id=UNKNOWN_ID) == {"books": [], "next_offset": None}
    assert library.path.read_bytes() == before
    calls = []

    class Legacy:
        def books(self, limit, offset):
            calls.append((limit, offset))
            return []

    service = LibraryService(Legacy())
    assert service.books(limit=2, offset=7) == {"books": [], "next_offset": None}
    assert calls == [(3, 7)]
    with pytest.raises(TypeError):
        service.books(book_id=chosen)
    assert calls == [(3, 7)]


@pytest.mark.parametrize("book_id", [True, 123, "", "A" * 64, "g" * 64, "a" * 63, "a" * 65])
def test_service_catalog_rejects_invalid_filter_before_adapter(library, monkeypatch, book_id):
    monkeypatch.setattr(library.service.retriever, "books", lambda *args, **kwargs: pytest.fail("Invalid filter reached adapter"))
    with pytest.raises(LibraryRequestError, match="book ID"):
        library.service.books(book_id=book_id)


def child_environment():
    return {"PYTHONPATH": os.environ.get("PYTHONPATH", str(Path(__file__).parents[1] / "src")),
            "PYTHONDONTWRITEBYTECODE": "1"}


def test_cli_existing_selected_evidence_and_miss_match_independent_oracle(library, cases):
    before = library.path.read_bytes()
    reports = []
    for case in cases["scenarios"][2:4]:
        command = [sys.executable, "-m", "ingest.local", "--index", str(library.path), "ask",
                   case["question"], "--book-id", library.ids[case["book_filter"]]]
        result = subprocess.run([*command, "--json"], env={**os.environ, **child_environment()},
                                capture_output=True, text=True, timeout=15, check=False)
        assert result.returncode == 0 and not result.stderr
        answer = json.loads(result.stdout)
        assert answer == library.service.ask(case["question"], book_id=library.ids[case["book_filter"]])
        row = assess_case(library, case, answer)
        assert row["passed"]
        reports.append(row)
        rendered = subprocess.run(command, env={**os.environ, **child_environment()},
                                  capture_output=True, text=True, timeout=15, check=False)
        assert rendered.returncode == 0 and not rendered.stderr
        assert answer["message"] in rendered.stdout
        assert all(warning in rendered.stdout for warning in answer["extraction_coverage"]["warnings"])
        assert all(passage["quote"] in rendered.stdout for passage in answer["passages"])
    assert library.path.read_bytes() == before
    save_evidence("usability-cli.json", {"rows": reports, "database_unchanged": True})


@pytest.fixture(scope="module")
def sdk():
    pytest.importorskip("mcp", reason="The real stdio checks require the approved Spark MCP profile")
    assert version("mcp") == "2.3.0"
    import anyio
    from mcp import Client, StdioServerParameters

    return anyio, Client, StdioServerParameters


@pytest.mark.parametrize("mode,protocol", [("auto", "2026-07-28"), ("legacy", "2025-11-25")])
def test_real_mcp_filtered_catalog_and_selected_evidence(library, cases, sdk, mode, protocol):
    anyio, client_type, parameters_type = sdk
    before = library.path.read_bytes()
    parameters = parameters_type(command=sys.executable,
                                 args=["-m", "ingest.local.mcp_server", "--index", str(library.path)],
                                 env=child_environment(), cwd=Path.cwd())
    transcript = {"mode": mode, "protocol": protocol, "completed": False, "calls": []}

    async def round_trip():
        with anyio.fail_after(45):
            async with client_type(parameters, mode=mode, read_timeout_seconds=5, cache=None) as client:
                assert client.protocol_version == protocol
                listing = await client.list_tools()
                tools = {tool.name: tool for tool in listing.tools}
                assert set(tools) == {"list_books", "ask_library", "get_passage", "library_status"}
                field = tools["list_books"].input_schema["properties"]["book_id"]
                assert field["type"] == "string" and field["default"] == "" and "pattern" in field

                async def call(name, arguments=None):
                    result = await client.call_tool(name, arguments or {})
                    assert not result.is_error
                    assert len(result.content) == 1 and result.content[0].type == "text"
                    assert json.loads(result.content[0].text) == result.structured_content
                    serialized = result.model_dump_json(by_alias=True)
                    assert str(library.path.parent) not in serialized and "source_path" not in serialized
                    assert len(serialized.encode("utf-8")) < 128 * 1024
                    transcript["calls"].append({"tool": name, "arguments": arguments or {}, "answer": result.structured_content})
                    return result.structured_content

                default = await call("list_books")
                assert default == library.service.books() == await call("list_books", {"book_id": ""})
                chosen = sorted(library.ids.values())[-1]
                selected = await call("list_books", {"limit": 1, "offset": 0, "book_id": chosen})
                assert selected == library.service.books(limit=1, book_id=chosen)
                assert [book["book_id"] for book in selected["books"]] == [chosen]
                assert await call("list_books", {"offset": 1, "book_id": chosen}) == {"books": [], "next_offset": None}
                assert await call("list_books", {"book_id": UNKNOWN_ID}) == {"books": [], "next_offset": None}
                for invalid in (None, True, 123, "A" * 64, "bad", "a" * 65):
                    result = await client.call_tool("list_books", {"book_id": invalid})
                    assert result.is_error
                    transcript["calls"].append({"tool": "list_books", "invalid_filter": invalid, "is_error": True})
                assert await call("library_status") == library.service.status()
                for number in (2, 3, 7, 11):
                    case = cases["scenarios"][number]
                    answer = await call("ask_library", {"question": case["question"], "book_id": library.ids[case["book_filter"]]})
                    assert answer == library.service.ask(case["question"], book_id=library.ids[case["book_filter"]])
                    assert assess_case(library, case, answer)["passed"]
                    for passage in answer["passages"]:
                        fetched = await call("get_passage", {"chunk_id": passage["chunk_id"]})
                        library.authoring.verify(fetched, library.oracle)
                        assert fetched == {key: value for key, value in passage.items() if key != "citation_id"}
        transcript["completed"] = True

    try:
        anyio.run(round_trip)
    finally:
        transcript["database_unchanged"] = library.path.read_bytes() == before
        save_evidence(f"usability-mcp-{mode}.json", transcript)
    assert transcript["database_unchanged"]
