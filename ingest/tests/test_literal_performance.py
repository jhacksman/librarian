"""Independent experiment-control checks; execute only in the reviewed Spark job."""

import copy
import json
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def performance(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "evaluation"))
    import evaluate_literal_performance

    return evaluate_literal_performance


def abba_samples(block_times):
    """Authored observations in the frozen three-block, five-call phase order."""
    samples = []
    for block, (a_ms, b_ms) in enumerate(block_times):
        for phase, version in enumerate(("A", "B", "B", "A")):
            for repetition in range(5):
                samples.append({"sequence": len(samples), "block": block, "phase": phase,
                                "version": version, "repetition": repetition,
                                "elapsed_ms": a_ms if version == "A" else b_ms,
                                "response_equal": True})
    return samples


def measured_row(performance, case_id, block_times):
    samples = abba_samples(block_times)
    return {"id": case_id, "completed": True, "correctness_passed": True,
            "samples": samples, "paired": performance.paired_summary(samples)}


def complete_experiment(performance):
    return [{"variant": variant, "background_books": size, "completed": True,
             "database_unchanged": True,
             "planned_query_ids": ["primary-archive-miss", "ordinary-control"],
             "queries": [measured_row(performance, "primary-archive-miss", [(100, 70)] * 3),
                         measured_row(performance, "ordinary-control", [(20, 20)] * 3)]}
            for size in (1, 8, 32, 128) for variant in ("plain", "collision")]


def corpus_for(corpora, variant="plain", size=128):
    return next(corpus for corpus in corpora
                if corpus["variant"] == variant and corpus["background_books"] == size)


def replace_row(performance, corpora, case_id, block_times, *, variant="plain", size=128):
    corpus = corpus_for(corpora, variant, size)
    position = next(index for index, row in enumerate(corpus["queries"]) if row["id"] == case_id)
    corpus["queries"][position] = measured_row(performance, case_id, block_times)


def test_paired_summary_preserves_raw_order_and_uses_arithmetic_median(performance):
    samples = abba_samples([(1, 2)] * 3)
    seen = {"A": 0, "B": 0}
    for sample in samples:
        seen[sample["version"]] += 1
        sample["elapsed_ms"] = seen[sample["version"]] * (1 if sample["version"] == "A" else 2)
    original = copy.deepcopy(samples)

    summary = performance.paired_summary(samples)

    assert samples == original
    assert summary["versions"]["A"]["samples"] == summary["versions"]["B"]["samples"] == 30
    assert summary["versions"]["A"]["median_ms"] == 15.5
    assert summary["versions"]["A"]["p95_ms"] == 29
    assert summary["versions"]["A"]["max_ms"] == 30
    assert summary["versions"]["B"]["median_ms"] == 31
    assert summary["versions"]["B"]["p95_ms"] == 58
    assert summary["versions"]["B"]["max_ms"] == 60
    assert [block["A_median_ms"] for block in summary["blocks"]] == [5.5, 15.5, 25.5]
    assert [block["B_median_ms"] for block in summary["blocks"]] == [11, 31, 51]
    assert all(block["improvement_fraction"] == -1 for block in summary["blocks"])


def test_exact_twenty_percent_in_every_primary_block_can_be_accepted(performance):
    corpora = complete_experiment(performance)
    replace_row(performance, corpora, "primary-archive-miss", [(100, 80)] * 3)
    result = performance.decision(corpora, quality_passed=True)
    assert result["accepted"] and result["primary_each_block_improves_20_percent"]
    assert result["counterexample_vetoes"] == result["incomplete_timing_cases"] == []


def test_pooled_speedup_cannot_hide_a_failing_primary_block(performance):
    corpora = complete_experiment(performance)
    replace_row(performance, corpora, "primary-archive-miss", [(100, 30), (100, 30), (100, 81)])
    primary = corpus_for(corpora)["queries"][0]
    assert primary["paired"]["versions"]["B"]["median_ms"] == 30
    result = performance.decision(corpora, quality_passed=True)
    assert not result["accepted"] and not result["primary_each_block_improves_20_percent"]


@pytest.mark.parametrize("variant,size,case_id", [
    ("collision", 128, "primary-archive-miss"),
    ("plain", 1, "ordinary-control"),
    ("collision", 8, "ordinary-control"),
])
def test_counterexample_veto_applies_across_variants_and_sizes(performance, variant, size, case_id):
    corpora = complete_experiment(performance)
    replace_row(performance, corpora, case_id, [(20, 22), (20, 20), (20, 22)], variant=variant, size=size)
    result = performance.decision(corpora, quality_passed=True)
    assert result["primary_each_block_improves_20_percent"]
    assert not result["accepted"]
    assert result["counterexample_vetoes"] == [
        {"variant": variant, "background_books": size, "query": case_id, "blocks": [0, 2]}]


@pytest.mark.parametrize("block_times,accepted", [
    ([(100, 105)] * 3, True),  # Exactly five percent does not exceed the relative threshold.
    ([(10, 11)] * 3, True),  # Exactly one millisecond does not exceed the absolute threshold.
    ([(20, 22), (20, 20), (20, 20)], True),  # One regressing block is insufficient.
    ([(100, 105.01), (100, 105.01), (100, 100)], False),
    ([(10, 11.01), (10, 11.01), (10, 10)], False),
])
def test_veto_needs_both_strict_thresholds_in_two_blocks(performance, block_times, accepted):
    corpora = complete_experiment(performance)
    replace_row(performance, corpora, "ordinary-control", block_times)
    result = performance.decision(corpora, quality_passed=True)
    assert result["accepted"] is accepted
    assert bool(result["counterexample_vetoes"]) is (not accepted)


@pytest.mark.parametrize("failure", ["quality", "corpus", "database", "query", "correctness"])
def test_correctness_or_completion_failure_blocks_adoption(performance, failure):
    corpora = complete_experiment(performance)
    corpus = corpus_for(corpora, "collision", 1)
    if failure == "corpus":
        corpus["completed"] = False
    elif failure == "database":
        corpus["database_unchanged"] = False
    elif failure == "query":
        corpus["queries"][1]["completed"] = False
    elif failure == "correctness":
        corpus["queries"][1]["correctness_passed"] = False
    result = performance.decision(corpora, quality_passed=failure != "quality")
    assert not result["accepted"] and not result["correctness_passed"]


@pytest.mark.parametrize("failure", ["empty", "missing-corpus", "duplicate-corpus", "missing-query", "duplicate-query"])
def test_incomplete_experiment_matrix_fails_closed(performance, failure):
    corpora = complete_experiment(performance)
    if failure == "empty":
        corpora.clear()
    elif failure == "missing-corpus":
        corpora.pop(0)
    elif failure == "duplicate-corpus":
        corpora[0] = copy.deepcopy(corpora[1])
    elif failure == "missing-query":
        corpora[0]["queries"].pop()
    else:
        corpora[0]["queries"][1] = copy.deepcopy(corpora[0]["queries"][0])
    assert not performance.decision(corpora, quality_passed=True)["accepted"]


@pytest.mark.parametrize("failure", ["missing-sample", "missing-phase", "missing-block", "duplicate-block"])
def test_incomplete_paired_measurements_cannot_pass(performance, failure):
    corpora = complete_experiment(performance)
    row = corpus_for(corpora, "collision", 32)["queries"][1]
    if failure == "duplicate-block":
        row["paired"]["blocks"][2] = copy.deepcopy(row["paired"]["blocks"][1])
    else:
        if failure == "missing-sample":
            row["samples"].pop()
        elif failure == "missing-phase":
            row["samples"] = [sample for sample in row["samples"]
                              if not (sample["block"] == 1 and sample["phase"] == 2)]
        else:
            row["samples"] = [sample for sample in row["samples"] if sample["block"] != 2]
        row["paired"] = performance.paired_summary(row["samples"])
    result = performance.decision(corpora, quality_passed=True)
    assert not result["accepted"]
    assert result["incomplete_timing_cases"] == [
        {"variant": "collision", "background_books": 32, "query": "ordinary-control"}]


def test_counters_observe_actual_predicate_lexing_and_exclude_other_lexing(performance):
    query, index = performance.query, performance.index_module
    original_predicate, original_lexer = index.contains_literals, query.literal_spans
    counters = {}
    texts = ["archive prose", "archive++17", "archive++", "plain"]
    with performance.predicate_binding(query.contains_literals, counters):
        assert query.plan_query("archive++").literals == ("archive++",)
        assert query.first_literal_match("archive++", ["archive++"]) is not None
        assert counters["lexer_invocations"] == 0
        assert not index.contains_literals(texts[0], ["archive++"])
        assert not index.contains_literals(texts[1], ["archive++"])
        assert index.contains_literals(texts[2], ["archive++"])
        assert index.contains_literals(texts[3], [])
        observed = copy.deepcopy(counters)
        query.plan_query("archive++")
        query.first_literal_match("archive++", ["archive++"])
        assert counters == observed
    assert index.contains_literals is original_predicate and query.literal_spans is original_lexer
    assert counters == {"predicate_calls": 4, "accepted": 2, "rejected": 2,
                        "guard_only_rejections": 1, "lexer_invocations": 2,
                        "predicate_input_characters": sum(map(len, texts)),
                        "lexer_input_characters": len(texts[1]) + len(texts[2])}


def test_baseline_miss_is_observed_as_a_lexer_rejection(performance):
    counters = {}
    with performance.predicate_binding(performance.baseline_contains_literals, counters):
        assert not performance.index_module.contains_literals("archive prose", ["archive++"])
    assert counters["predicate_calls"] == counters["rejected"] == counters["lexer_invocations"] == 1
    assert counters["guard_only_rejections"] == 0
    assert counters["predicate_input_characters"] == counters["lexer_input_characters"] == 13


@pytest.mark.parametrize("instrumented", [False, True])
def test_predicate_and_lexer_bindings_restore_after_candidate_exception(performance, instrumented):
    query, index = performance.query, performance.index_module
    original_predicate, original_lexer = index.contains_literals, query.literal_spans

    def interrupted_candidate(text, _literals):
        list(query.literal_spans(text))
        raise RuntimeError("injected predicate failure")

    counters = {} if instrumented else None
    with (
        pytest.raises(RuntimeError, match="injected predicate failure"),
        performance.predicate_binding(interrupted_candidate, counters),
    ):
        assert index.contains_literals is not original_predicate
        index.contains_literals("archive++", ["archive++"])
    assert index.contains_literals is original_predicate and query.literal_spans is original_lexer
    if instrumented:
        assert counters["predicate_calls"] == counters["lexer_invocations"] == 1
        assert counters["accepted"] == counters["rejected"] == 0


def authored_continuation():
    """Three exact Unicode source slices, authored without service passage helpers."""
    book = "a" * 64
    text = "archive++ " + "🧭é" * 8100 + "終"
    oracle = {book: {"format": "epub", "label": "probe", "title": "Unicode probe",
                     "sections": {1: text},
                     "chunks": {f"{book}:0": {"section": 1, "char_start": 0, "char_end": len(text)},
                                f"{book}:1": {"section": 1, "char_start": 0, "char_end": len(text)}}}}
    passages = []
    for start in range(0, len(text), 8000):
        end = min(start + 8000, len(text))
        passages.append({"book_id": book, "source_sha256": book, "chunk_id": f"{book}:0",
                         "title": "Unicode probe", "source_uri": f"librarian://books/{book}/chunks/0",
                         "quote": text[start:end], "page_start": None, "page_end": None,
                         "epub_member": "s1.xhtml", "char_start": start, "char_end": end,
                         "content_kind": "source_excerpt", "original_char_start": 0,
                         "original_char_end": len(text), "excerpt_offset": start,
                         "excerpt_truncated": True, "next_offset": end if end < len(text) else None,
                         "offset_basis": "extracted_section_unicode_codepoints",
                         "pipeline_version": "local-text-v2-words400-overlap80"})
    passages[0]["citation_id"] = "[1]"
    case = {"id": "unicode-probe", "question": "archive++", "expected_no_match": False,
            "book_filter": "probe", "expected_continuation": True, "required_quotes": ["archive++"],
            "required_continuation_quotes": ["終"]}
    answer = {"question": "archive++", "answer_mode": "evidence_only", "backend": "sqlite-fts5-bm25",
              "abstained": False, "passages": [passages[0]]}
    calls = []

    def get_passage(chunk_id, offset=0):
        calls.append((chunk_id, offset))
        return copy.deepcopy(passages[offset // 8000])

    return SimpleNamespace(case=case, answer=answer, service=SimpleNamespace(get_passage=get_passage),
                           oracle=oracle, by_label={"probe": book}, passages=passages, calls=calls)


def test_source_oracle_accepts_all_three_unicode_excerpt_slices(performance):
    fixture = authored_continuation()
    records = performance.verify_answer(fixture.case, fixture.answer, fixture.service, fixture.oracle, fixture.by_label)
    assert records == fixture.passages[1:]
    book = fixture.by_label["probe"]
    assert fixture.calls == [(f"{book}:0", 8000), (f"{book}:0", 16000)]
    assert "".join(passage["quote"] for passage in fixture.passages) == fixture.oracle[book]["sections"][1]


@pytest.mark.parametrize("failure", ["quote", "citation", "filter", "abstention", "required-quote", "continuation-quote"])
def test_source_and_label_faults_are_hard_failures(performance, failure):
    fixture = authored_continuation()
    if failure == "quote":
        fixture.answer["passages"][0]["quote"] = "invented source"
    elif failure == "citation":
        fixture.answer["passages"][0]["citation_id"] = "[2]"
    elif failure == "filter":
        fixture.by_label["probe"] = "b" * 64
    elif failure == "abstention":
        fixture.answer.update(abstained=True, passages=[])
    elif failure == "required-quote":
        fixture.case["required_quotes"] = ["not in the authored source"]
    else:
        fixture.case["required_continuation_quotes"] = ["not in any continuation"]
    with pytest.raises(AssertionError):
        performance.verify_answer(fixture.case, fixture.answer, fixture.service, fixture.oracle, fixture.by_label)


@pytest.mark.parametrize("failure", ["skipped-codepoint", "wrong-chunk", "early-end", "cycle"])
def test_plausible_continuation_faults_cannot_pass_source_validation(performance, failure):
    fixture = authored_continuation()
    following = fixture.passages[1]
    if failure == "skipped-codepoint":
        # This remains a valid citation; only the continuation adjacency is broken.
        following["char_start"] += 1
        following["excerpt_offset"] += 1
        following["quote"] = following["quote"][1:]
    elif failure == "wrong-chunk":
        # Both chunk IDs have valid oracle spans in the same section.
        book = fixture.by_label["probe"]
        following["chunk_id"] = f"{book}:1"
        following["source_uri"] = f"librarian://books/{book}/chunks/1"
    elif failure == "early-end":
        following["next_offset"] = None
    else:
        following["next_offset"] = 8000
    with pytest.raises(AssertionError):
        performance.verify_answer(fixture.case, fixture.answer, fixture.service, fixture.oracle, fixture.by_label)
    assert len(fixture.calls) <= 2


@pytest.mark.parametrize("error", [TimeoutError("injected timeout"), MemoryError("injected memory limit")])
def test_partial_report_preserves_raw_observations_and_failure_context(performance, tmp_path, error):
    output = tmp_path / "partial.json"
    samples = abba_samples([(100, 70)] * 3)[:7]
    report = {"passed": True, "completed": False,
              "corpora": [{"variant": "collision", "background_books": 32,
                           "queries": [{"id": "primary-archive-miss", "samples": samples}]}]}
    context = {"stage": "query", "query": "primary-archive-miss", "version": "B", "block": 0}
    with (
        pytest.raises(type(error), match="injected"),
        performance.preserved_report(output, report, context),
    ):
        context["step"] = "timed_service_call"
        raise error
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert not saved["passed"] and not saved["completed"]
    assert saved["corpora"][0]["queries"][0]["samples"] == samples
    assert saved["error"] == {"type": type(error).__name__, "message": str(error), "context": context}


def test_failed_atomic_replace_preserves_the_previous_complete_checkpoint(performance, tmp_path, monkeypatch):
    output = tmp_path / "checkpoint.json"
    report = {"passed": False, "completed": False, "samples": [12.5]}
    context = {"stage": "query", "step": "phase_complete", "phase": 0}
    memory = iter([{"driver_peak_rss_mib": 10}, {"driver_peak_rss_mib": 20}])
    monkeypatch.setattr(performance, "memory_snapshot", lambda: next(memory))
    performance.checkpoint(output, report, context)
    previous_bytes = output.read_bytes()
    previous = json.loads(previous_bytes)
    assert previous["checkpoint"]["context"] == context
    assert previous["checkpoint"]["memory"] == {"driver_peak_rss_mib": 10}
    assert previous["checkpoint"]["utc_timestamp"]

    report["samples"].append(13.5)
    context["phase"] = 1
    replacements = []

    def failed_replace(source, destination):
        replacements.append((Path(source), Path(destination)))
        candidate = json.loads(Path(source).read_text(encoding="utf-8"))
        assert candidate["samples"] == [12.5, 13.5]
        assert candidate["checkpoint"]["context"]["phase"] == 1
        assert candidate["checkpoint"]["memory"] == {"driver_peak_rss_mib": 20}
        raise OSError("injected replacement failure")

    monkeypatch.setattr(performance.os, "replace", failed_replace)
    with pytest.raises(OSError, match="injected replacement failure"):
        performance.checkpoint(output, report, context)
    assert output.read_bytes() == previous_bytes
    assert len(replacements) == 1
    assert replacements[0][0].parent == output.parent and replacements[0][1] == output
    assert not replacements[0][0].exists()


def test_owned_child_hard_exit_retains_checkpoint_without_running_finally(performance, tmp_path):
    output = tmp_path / "child-checkpoint.json"
    finally_marker = tmp_path / "finally-ran"
    module_paths = [str(Path(performance.__file__).parent), str(Path(__file__).parents[1] / "src")]
    child = """
import json
import os
import sys
from pathlib import Path

sys.path[:0] = json.loads(sys.argv[1])
import evaluate_literal_performance as performance

output, marker = Path(sys.argv[2]), Path(sys.argv[3])
report = {"passed": False, "completed": False, "samples": []}
context = {"stage": "query", "step": "initial"}
with performance.preserved_report(output, report, context):
    report["samples"].append({"phase": 0, "elapsed_ms": 12.5})
    context.update(step="phase_complete", phase=0)
    performance.checkpoint(output, report, context)
    report["samples"].append({"phase": 1, "elapsed_ms": 999})
    context.update(step="interrupted_phase", phase=1)
    try:
        os._exit(73)
    finally:
        marker.write_text("unexpected cleanup", encoding="utf-8")
"""
    result = subprocess.run([sys.executable, "-c", child, json.dumps(module_paths), str(output), str(finally_marker)],
                            capture_output=True, text=True, timeout=20, check=False)
    assert result.returncode == 73, result.stderr
    assert not finally_marker.exists()
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert not saved["passed"] and not saved["completed"] and "error" not in saved
    assert saved["samples"] == [{"phase": 0, "elapsed_ms": 12.5}]
    assert saved["checkpoint"]["context"] == {"stage": "query", "step": "phase_complete", "phase": 0}
    assert "memory" in saved["checkpoint"]


def test_budget_preserves_time_already_spent_on_setup(performance, tmp_path, monkeypatch):
    budget = tmp_path / "budget.json"
    budget.write_text(json.dumps({"command_started_monotonic": 800, "deadline_monotonic": 1310}), encoding="utf-8")
    monkeypatch.setattr(performance.time, "monotonic", lambda: 1000)
    result = performance.read_budget(budget)
    assert result["command_started_monotonic"] == 800
    assert result["deadline_monotonic"] == 1310
    assert result["remaining_seconds_at_read"] == 310
    assert result["command_budget_seconds"] == 510 and result["nominal_outer_reserve_seconds"] == 90
    assert result["file_sha256"] == performance.quality_fixtures.digest(budget)


@pytest.mark.parametrize("values", [
    {"command_started_monotonic": True, "deadline_monotonic": 511},
    {"command_started_monotonic": 800, "deadline_monotonic": float("nan")},
    {"command_started_monotonic": 1001, "deadline_monotonic": 1511},
    {"command_started_monotonic": 800, "deadline_monotonic": 1400},
    {"command_started_monotonic": -1, "deadline_monotonic": 509},
])
def test_budget_rejects_invalid_or_reset_deadlines(performance, tmp_path, monkeypatch, values):
    budget = tmp_path / "budget.json"
    budget.write_text(json.dumps(values), encoding="utf-8")
    monkeypatch.setattr(performance.time, "monotonic", lambda: 1000)
    with pytest.raises(ValueError):
        performance.read_budget(budget)


@pytest.mark.parametrize("failure", ["missing", "invalid", "expired"])
def test_run_requires_a_valid_unexpired_budget_before_setup(performance, tmp_path, monkeypatch, failure):
    budget, output = tmp_path / "budget.json", tmp_path / "report.json"
    if failure != "missing":
        values = {} if failure == "invalid" else {"command_started_monotonic": 100, "deadline_monotonic": 610}
        budget.write_text(json.dumps(values), encoding="utf-8")
    monkeypatch.setattr(performance.time, "monotonic", lambda: 1000)

    def unexpected_setup(_package):
        pytest.fail("Environment setup ran without a usable command budget")

    monkeypatch.setattr(performance, "version", unexpected_setup)
    with pytest.raises(FileNotFoundError if failure == "missing" else ValueError):
        performance.run(output, quality_reports=[], budget_file=budget)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert not saved["passed"] and not saved["completed"]
    assert saved["corpora"] == [] and "environment" not in saved
    assert saved["error"]["context"] == {"stage": "read_command_budget"}
    assert saved["checkpoint"]["context"] == saved["error"]["context"]


def test_internal_deadline_retains_context_and_restores_all_bindings(performance, tmp_path, monkeypatch):
    output = tmp_path / "deadline.json"
    report = {"passed": False, "completed": False, "samples": [12.5]}
    context = {"stage": "query", "query": "collision-probe", "version": "B", "block": 1}
    original_predicate = performance.index_module.contains_literals
    original_lexer = performance.query.literal_spans
    prior_handler = object()
    handlers, timers = [], []
    monkeypatch.setattr(performance.time, "monotonic", lambda: 1000)

    def install_handler(signum, handler):
        assert signum == performance.signal.SIGALRM
        handlers.append(handler)
        return prior_handler

    monkeypatch.setattr(performance.signal, "signal", install_handler)
    monkeypatch.setattr(performance.signal, "setitimer", lambda timer, seconds: timers.append((timer, seconds)))

    def interrupted_candidate(text, _literals):
        list(performance.query.literal_spans(text))
        context["step"] = "instrumented_service_call"
        # Invoke the installed timeout handler deterministically, with no real wait or OOM.
        handlers[0](performance.signal.SIGALRM, None)

    with (
        pytest.raises(TimeoutError, match="budget exhausted"),
        performance.preserved_report(output, report, context),
        performance.deadline(1007),
        performance.predicate_binding(interrupted_candidate, {}),
    ):
        performance.index_module.contains_literals("archive++17", ["archive++"])
    assert performance.index_module.contains_literals is original_predicate
    assert performance.query.literal_spans is original_lexer
    assert handlers[-1] is prior_handler
    assert timers == [(performance.signal.ITIMER_REAL, 7), (performance.signal.ITIMER_REAL, 0)]
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert not saved["passed"] and not saved["completed"]
    assert saved["samples"] == [12.5]
    assert saved["error"]["type"] == "TimeoutError"
    assert saved["error"]["context"] == context == saved["checkpoint"]["context"]


@pytest.mark.parametrize("end", [999, 1511, float("nan")])
def test_invalid_deadline_never_arms_timer_and_restores_predicate(performance, monkeypatch, end):
    original_predicate = performance.index_module.contains_literals
    original_lexer = performance.query.literal_spans
    monkeypatch.setattr(performance.time, "monotonic", lambda: 1000)

    def unexpected_timer(*_arguments):
        pytest.fail("Invalid deadline changed process signal state")

    monkeypatch.setattr(performance.signal, "signal", unexpected_timer)
    monkeypatch.setattr(performance.signal, "setitimer", unexpected_timer)
    with (
        pytest.raises(ValueError, match="expired or invalid"),
        performance.predicate_binding(performance.baseline_contains_literals, {}),
        performance.deadline(end),
    ):
        pytest.fail("Invalid deadline entered the protected operation")
    assert performance.index_module.contains_literals is original_predicate
    assert performance.query.literal_spans is original_lexer


def test_environment_metadata_needs_only_approved_minimal_dependencies(performance, tmp_path, monkeypatch):
    budget, output = tmp_path / "budget.json", tmp_path / "environment.json"
    budget.write_text(json.dumps({"command_started_monotonic": 800, "deadline_monotonic": 1310}), encoding="utf-8")
    monkeypatch.setattr(performance.time, "monotonic", lambda: 1000)
    monkeypatch.setattr(performance, "deadline", lambda _end: nullcontext())
    monkeypatch.setattr(performance.platform, "system", lambda: "Linux")
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    requested = []

    def approved_version(package):
        assert package in {"pydantic", "pypdf"}, "Metadata requested an unapproved dependency"
        requested.append(package)
        return "installed-" + package

    def stop_before_quality(_paths):
        raise RuntimeError("reached quality after environment setup")

    monkeypatch.setattr(performance, "version", approved_version)
    monkeypatch.setattr(performance, "quality_evidence", stop_before_quality)
    with pytest.raises(RuntimeError, match="reached quality after environment setup"):
        performance.run(output, quality_reports=[], budget_file=budget)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert requested == ["pydantic", "pypdf"]
    assert saved["environment"]["packages"] == {"pydantic": "installed-pydantic", "pypdf": "installed-pypdf"}
    assert saved["error"]["context"] == {"stage": "frozen_quality_reports"}
    assert not saved["passed"] and not saved["completed"] and saved["corpora"] == []
