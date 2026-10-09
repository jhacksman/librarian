"""Literal fidelity and candidate completeness; run only on Spark."""

import hashlib
import sys

import pytest
from test_local_pilot import epub
from test_mcp_transport import child_environment
from test_mcp_transport import sdk as sdk

from ingest.local import query as query_module
from ingest.local.index import LocalIndex
from ingest.local.query import contains_literals, literal_spans, plan_query
from ingest.local.service import LibraryService, SQLiteRetriever


def test_literal_components_survive_question_word_removal():
    plan = plan_query("What is pkg::for_each?", drop_question_words=True)
    assert plan.preserved_query == "pkg::for_each"
    assert plan.literals == ("pkg::for_each",)
    assert plan.candidate_terms == ("pkg", "for", "each")
    assert plan.expression == '"pkg" AND "for" AND "each"'
    assert plan.provenance()["literal_case_policy"] == "case_sensitive"


@pytest.mark.parametrize("literal, source, expected", [
    ("C++", "C++17", False), ("C++", "(C++).", True), ("C++", "c++", False),
    ("--force", "--forceful", False), ("--force", "--force-all", False),
    ("--force", "(--force=true)", True), ("--force", "---force", False),
    ("-v", "-verbose", False), ("-v", "'-v'", True),
    ("pkg::for_each", "outer::pkg::for_each", False),
    ("pkg::for_each", "pkg::for_each()", True),
    ("alpha_beta", "alpha_betas", False), ("alpha_beta", "alpha_beta.", True),
    ("alpha_beta", "éalpha_beta", False), ("alpha_beta", "alpha_beta\ue000", False),
    ("alpha_beta", "alpha_beta\u0301", False),
    ("obj.field", "obj.field.extra", False), ("obj.field", "obj.field;", True),
])
def test_maximal_literal_tokens(literal, source, expected):
    assert contains_literals(source, (literal,)) is expected


def test_all_literals_required_and_offsets_use_original_unicode():
    source = "İstanbul 🧭: (--force) then pkg::for_each()."
    assert contains_literals(source, ("--force", "pkg::for_each"))
    assert not contains_literals(source, ("--force", "pkg::missing"))
    span = next(span for span in literal_spans(source) if span.text == "--force")
    assert span.start == source.index("--force") and source[span.start:span.end] == "--force"


def test_missing_substring_and_empty_iterable_need_no_lexer(monkeypatch):
    def unexpected_lexer(text):
        raise AssertionError("An absent substring or empty constraint set needs no lexer")

    monkeypatch.setattr(query_module, "literal_spans", unexpected_lexer)
    assert not contains_literals("archive catalogue " * 400, iter(["archive++"]))
    assert contains_literals("archive catalogue", iter(()))


@pytest.mark.parametrize("source, literal, expected", [
    ("C++17", "C++", False),
    ("--forceful", "--force", False),
    ("outer::pkg::name", "pkg::name", False),
    ("alpha_beta\u0301", "alpha_beta", False),
    ("(--force=true)", "--force", True),
])
def test_substring_presence_still_uses_maximal_lexer(monkeypatch, source, literal, expected):
    original = query_module.literal_spans
    calls = []

    def observed_lexer(text):
        calls.append(text)
        yield from original(text)

    monkeypatch.setattr(query_module, "literal_spans", observed_lexer)
    assert contains_literals(source, iter([literal, literal])) is expected
    assert calls == [source]


def test_ordinary_query_behavior_and_literal_term_bound():
    plan = plan_query("What are regular expressions?", drop_question_words=True)
    assert plan.preserved_query == "regular expressions" and plan.literals == ()
    assert plan.expression == '"regular" AND "expressions"'
    query = "::".join(["part"] * 129)
    service = LibraryService(None)
    with pytest.raises(ValueError, match="128 search terms"):
        service.ask(query)


def test_literal_filter_precedes_limit_and_preserves_book_filter(tmp_path):
    path = tmp_path / "index.sqlite"
    index = LocalIndex(path, create=True)
    try:
        distractors = []
        for number in range(35):
            distractors.append(index.ingest(epub(tmp_path / f"d{number}.epub", text=f"C distractor{number}"))["book_id"])
        target = index.ingest(epub(tmp_path / "target.epub", text="padding " * 80 + "C++ target"))["book_id"]
        assert target not in {hit.book_id for hit in index.search("C", limit=3)}
        hits = index.search("C++", limit=1)
        assert len(hits) == 1 and hits[0].book_id == target
        assert index.search("C++", book_id=distractors[0]) == []
        assert index.search("C++", book_id=target)[0].book_id == target
    finally:
        index.close()
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    answer = LibraryService(SQLiteRetriever(path)).ask("C++", limit=1)
    assert answer["retrieval_query"] == "C++"
    assert answer["query_normalization"]["fts_candidate_expression"] == '"c"'
    assert answer["query_normalization"]["literal_constraints"] == ["C++"]
    assert answer["passages"][0]["book_id"] == target
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_unsupported_unicode_adjacency_is_not_reparsed_as_a_literal(tmp_path):
    path = tmp_path / "index.sqlite"
    index = LocalIndex(path, create=True)
    try:
        index.ingest(epub(tmp_path / "ordinary.epub", text="foo bar"))
    finally:
        index.close()
    answer = LibraryService(SQLiteRetriever(path)).ask("foo_bar\u0301")
    assert answer["query_normalization"]["literal_constraints"] == []
    assert not answer["abstained"]


@pytest.mark.parametrize("mode", ["auto", "legacy"])
def test_real_stdio_literal_and_long_excerpt(tmp_path, sdk, mode):
    anyio, client_type, parameters_type, _ = sdk
    text = "C İ " + ("a" * 30 + " ") * 397 + "C++"
    path = tmp_path / "index.sqlite"
    index = LocalIndex(path, create=True)
    try:
        book = index.ingest(epub(tmp_path / "long.epub", text=text))["book_id"]
    finally:
        index.close()
    service = LibraryService(SQLiteRetriever(path))
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    parameters = parameters_type(command=sys.executable,
                                 args=["-m", "ingest.local.mcp_server", "--index", str(path)],
                                 env=child_environment())

    async def check():
        with anyio.fail_after(30):
            async with client_type(parameters, mode=mode, read_timeout_seconds=5, cache=None) as client:
                response = await client.call_tool("ask_library", {"question": "C++", "book_id": book})
                assert not response.is_error
                answer = response.structured_content
                assert answer == service.ask("C++", book_id=book)
                passage = answer["passages"][0]
                assert "C++" in passage["quote"] and passage["char_start"] > 0
                assert text[passage["char_start"]:passage["char_end"]] == passage["quote"]
                resolved = await client.call_tool("get_passage", {"chunk_id": passage["chunk_id"], "offset": passage["excerpt_offset"]})
                assert not resolved.is_error and resolved.structured_content["quote"] == passage["quote"]
                mismatch = await client.call_tool("ask_library", {"question": "c++"})
                assert not mismatch.is_error and mismatch.structured_content["abstained"]

    anyio.run(check)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
