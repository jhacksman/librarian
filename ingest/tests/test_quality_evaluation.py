"""Exercise the new evaluator's failure detection; execution is Spark-only."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def quality(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "evaluation"))
    import evaluate_quality
    import quality_fixtures
    import quality_scoring

    return SimpleNamespace(fixtures=quality_fixtures, scoring=quality_scoring, runner=evaluate_quality)


def labeled_answer():
    book = "a" * 64
    text = "The needle is blue."
    oracle = {book: {"format": "epub", "label": "one", "title": "One", "sections": {1: text, 2: text},
                     "chunks": {book + ":0": {"section": 1, "char_start": 0, "char_end": len(text)},
                                book + ":1": {"section": 2, "char_start": 0, "char_end": len(text)}}}}
    passage = {
        "book_id": book, "source_sha256": book, "chunk_id": book + ":0", "title": "One",
        "source_uri": f"librarian://books/{book}/chunks/0", "quote": text,
        "page_start": None, "page_end": None, "epub_member": "s1.xhtml",
        "char_start": 0, "char_end": len(text), "content_kind": "source_excerpt",
        "original_char_start": 0, "original_char_end": len(text), "excerpt_offset": 0,
        "offset_basis": "extracted_section_unicode_codepoints", "pipeline_version": "local-text-v2-words400-overlap80",
        "citation_id": "[1]",
    }
    case = {"id": "needle", "category": "example", "gate": True, "question": "needle", "expected_no_match": False}
    answer = {"question": "needle", "retrieval_query": "needle", "backend": "sqlite-fts5-bm25",
              "answer_mode": "evidence_only", "abstained": False, "passages": [passage]}
    golden = {"document": "one", "book_id": book, "section": 1, "quote": text, "char_start": 0, "char_end": len(text)}
    return case, answer, [golden], oracle


def test_wrong_section_cannot_receive_relevance_credit(quality):
    case, answer, goldens, oracle = labeled_answer()
    answer["passages"][0]["epub_member"] = "s2.xhtml"
    book = answer["passages"][0]["book_id"]
    answer["passages"][0]["chunk_id"] = book + ":1"
    answer["passages"][0]["source_uri"] = f"librarian://books/{book}/chunks/1"
    score = quality.scoring.score_case(case, answer, goldens, oracle)
    assert score["source_span_recall_at_3"] == 0 and not score["passed"]


def test_partial_quote_cannot_receive_full_span_credit(quality):
    case, answer, goldens, oracle = labeled_answer()
    answer["passages"][0]["quote"] = "The needle"
    answer["passages"][0]["char_end"] = len("The needle")
    score = quality.scoring.score_case(case, answer, goldens, oracle)
    assert score["source_span_recall_at_3"] == 0 and not score["passed"]


def test_duplicate_hits_cannot_inflate_recall(quality):
    case, answer, goldens, oracle = labeled_answer()
    duplicate = copy.deepcopy(answer["passages"][0])
    duplicate["citation_id"] = "[2]"
    answer["passages"].append(duplicate)
    goldens.append({**goldens[0], "section": 2})
    score = quality.scoring.score_case(case, answer, goldens, oracle)
    assert score["matched_golden_count"] == 1 and score["source_span_recall_at_3"] == .5
    assert not score["passed"]


def test_corrupt_citation_is_a_hard_failure(quality):
    case, answer, goldens, oracle = labeled_answer()
    answer["passages"][0]["char_start"] = 1
    with pytest.raises(AssertionError):
        quality.scoring.score_case(case, answer, goldens, oracle)


@pytest.mark.parametrize("chunk_number, message", [(999999, "Unknown indexed chunk ID"), (1, "another source section")])
def test_plausible_wrong_chunk_id_is_rejected(quality, chunk_number, message):
    case, answer, goldens, oracle = labeled_answer()
    passage = answer["passages"][0]
    passage["chunk_id"] = f'{passage["book_id"]}:{chunk_number}'
    passage["source_uri"] = f'librarian://books/{passage["book_id"]}/chunks/{chunk_number}'
    with pytest.raises(AssertionError, match=message):
        quality.scoring.score_case(case, answer, goldens, oracle)


def test_diagnostic_failure_is_reported_separately(quality):
    case, answer, goldens, oracle = labeled_answer()
    case["gate"] = False
    answer.update(abstained=True, passages=[])
    row = quality.scoring.score_case(case, answer, goldens, oracle)
    report = quality.scoring.summarize([row])
    assert report["gate_failures"] == [] and report["diagnostic_failures"] == ["needle"]
    assert report["all"]["macro_source_span_recall_at_3"] == 0


def test_negative_with_a_hit_is_not_a_correct_abstention(quality):
    case, answer, _, oracle = labeled_answer()
    case["expected_no_match"] = True
    row = quality.scoring.score_case(case, answer, [], oracle)
    assert not row["passed"] and not row["no_match_label_correct"]
    assert quality.scoring.summarize([row])["all"]["correct_no_match_negatives"] == 0


def test_authored_goldens_and_fixture_bytes_are_reproducible(quality, tmp_path):
    path = Path(__file__).parents[1] / "evaluation/quality-cases.json"
    cases = quality.fixtures.load_cases(path)
    for kind, writer in (("epub", quality.fixtures.write_epub), ("pdf", quality.fixtures.write_pdf)):
        document = next(item for item in cases["documents"] if item["format"] == kind)
        first, second = tmp_path / f"first.{kind}", tmp_path / f"second.{kind}"
        writer(first, document)
        writer(second, document)
        assert first.read_bytes() == second.read_bytes()
    first = list(quality.fixtures.background_documents(cases["seed"], 1))
    second = list(quality.fixtures.background_documents(cases["seed"], 1))
    assert first == second and len(first[0]["sections"]) == 16


def test_nearest_rank_latency_measurement(quality):
    report = quality.scoring.latency_summary(list(range(1, 21)))
    assert report["samples"] == 20 and report["p50_ms"] == 10 and report["p95_ms"] == 19


def test_alternate_cases_require_explicit_comparison_profile(quality, tmp_path):
    with pytest.raises(ValueError, match="explicit query-comparison"):
        quality.runner.run(tmp_path / "report.json", 1, cases_path=tmp_path / "cases.json")


def test_comparison_profile_does_not_depend_on_default_fixture_terms(quality, tmp_path, monkeypatch):
    pytest.importorskip("mcp", reason="Comparison profile verifies both optional real MCP protocols")
    text = "Copper keys unlock the laboratory."
    cases = {"version": 1, "provenance": "Original alternate-profile test fixture", "seed": 20261004,
             "documents": [{"id": "copper", "format": "epub", "title": "Keys", "sections": [text]}],
             "queries": [{"id": "copper", "category": "filter", "gate": True, "question": "copper",
                          "book_filter": "copper", "expected_no_match": False,
                          "relevant": [{"document": "copper", "section": 1, "quote": text}]},
                         {"id": "absent", "category": "negative", "gate": True, "question": "velvet",
                          "expected_no_match": True, "relevant": []}]}
    case_path = tmp_path / "alternate.json"
    case_path.write_text(json.dumps(cases))
    monkeypatch.setattr(quality.runner, "BACKGROUND_BOOKS", 1)
    output = tmp_path / "comparison.json"
    assert quality.runner.run(output, 1, cases_path=case_path, profile="query-comparison", probe_literal_cost=True)
    report = json.loads(output.read_text())
    assert report["profile"] == "query-comparison" and report["contract_probes"]["status"] == "not_run"
    assert report["fixture_sha256"] == quality.fixtures.digest(case_path)
    assert report["database_unchanged"] and report["service"]["completed"]
    assert report["service"]["passage_boundaries"] == report["service"]["invalid_requests"] == []
    assert report["literal_cost"]["completed"] and report["literal_cost"]["fts_candidates"] > 0
    assert report["literal_cost"]["latency"]["samples"] == 15
    for phase in report["mcp"].values():
        assert phase["completed"] and phase["catalog_books"] == 2 and phase["passed"]
        assert phase["passage_boundaries"] == phase["invalid_requests"] == []
        assert len(phase["queries"]) == 2


def test_late_service_failure_preserves_rows_timings_and_context(quality, tmp_path, monkeypatch):
    pytest.importorskip("mcp", reason="Full harness fault injection requires the optional MCP runtime")
    original = quality.runner.LibraryService.ask
    calls = 0

    def fail_fourth(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise AssertionError
        return original(self, *args, **kwargs)

    monkeypatch.setattr(quality.runner, "BACKGROUND_BOOKS", 1)
    monkeypatch.setattr(quality.runner.LibraryService, "ask", fail_fourth)
    output = tmp_path / "partial.json"
    with pytest.raises(AssertionError):
        quality.runner.run(output, 1)
    report = json.loads(output.read_text())
    assert not report["passed"] and not report["service"]["completed"]
    assert len(report["service"]["queries"]) == len(report["service"]["latency"]["first_pass_ms"]) == 3
    assert report["error"] == {"type": "AssertionError", "message": "AssertionError",
                               "context": {"phase": "service", "check": "first_pass_query", "case": "wrong-edition-no-match"}}


def test_late_mcp_failure_preserves_rows_and_context(quality, tmp_path, monkeypatch):
    pytest.importorskip("mcp", reason="Real stdio fault injection requires the optional MCP runtime")
    from mcp import Client

    original = Client.call_tool
    calls = 0

    async def fail_fourth(self, name, arguments=None, **kwargs):
        nonlocal calls
        if name == "ask_library":
            calls += 1
            if calls == 4:
                raise AssertionError("injected MCP failure")
        return await original(self, name, arguments, **kwargs)

    monkeypatch.setattr(quality.runner, "BACKGROUND_BOOKS", 1)
    monkeypatch.setattr(Client, "call_tool", fail_fourth)
    output = tmp_path / "partial-mcp.json"
    with pytest.raises(Exception, match="injected MCP failure|TaskGroup|task group"):
        quality.runner.run(output, 1)
    report = json.loads(output.read_text())
    assert not report["passed"] and report["service"]["completed"]
    phase = report["mcp"]["auto"]
    assert not phase["completed"] and len(phase["queries"]) == len(phase["latency"]["first_pass_ms"]) == 3
    assert report["error"]["message"]
    assert report["error"]["context"] == {"phase": "mcp:auto", "check": "first_pass_query",
                                            "case": "wrong-edition-no-match", "tool": "ask_library"}
