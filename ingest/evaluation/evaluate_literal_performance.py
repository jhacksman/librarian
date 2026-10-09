"""Frozen PERF1 synthetic A/B experiment. Execute only in the reviewed Spark job.

No installation, subprocesses, network, models or private documents are used.
The coordinator supplies the 600-second / 2-CPU / 4-GiB job boundary and runs the
unchanged quality suites separately before passing their JSON reports here.
"""

import argparse
import cProfile
import hashlib
import inspect
import json
import math
import os
import platform
import pstats
import resource
import signal
import sqlite3
import statistics
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import quality_fixtures
from literal_performance_fixtures import broad_cases, fixtures
from quality_scoring import verify_citation

from ingest.local import index as index_module
from ingest.local import query
from ingest.local.service import LibraryService, SQLiteRetriever

BASELINE_COMMIT = "c527e480330242fdc81ac656c4900d44bccead51"
BASELINE_PATH = "ingest/src/ingest/local/query.py"
BASELINE_SOURCE_SHA256 = "e86abce173637bacf170f2bdf2eb9afacbb9e0e6f85cbedcbc01ef9b93f8f151"
POLICY_SHA256 = "7a674ed84a7d582417da32cc5f7f15775d97d2ce0a4e6df36ecf089448aa0fea"
SIZES = (1, 8, 32, 128)
BLOCKS = 3
CALLS_PER_PHASE = 5
PHASES = ("A", "B", "B", "A")
COUNTER_NAMES = (
    "predicate_calls", "accepted", "rejected", "guard_only_rejections",
    "lexer_invocations", "predicate_input_characters", "lexer_input_characters",
)
FROZEN_QUALITY = {
    "quality-cases.json": {
        "sha256": "452e5b5c92ed34474b48b02a79e23279929ecea615802123ee7638283e89d936",
        "queries": 22, "gates": 16, "profile": "full",
        "diagnostic_failures": ["recovery-keywords", "recovery-natural-question", "recovery-paraphrase"],
    },
    "lexical-heldout-cases.json": {
        "sha256": "4327aa1c0463f49c719402411e5b4ad4b3f653144bc55e9cae7304602569b721",
        "queries": 20, "gates": 6, "profile": "query-comparison", "diagnostic_failures": [],
    },
}


def baseline_contains_literals(text, literals):
    """One source scan per candidate; no ranking limit is applied here."""
    remaining = set(literals)
    if not remaining:
        return True
    for literal in query.literal_spans(text):
        remaining.discard(literal.text)
        if not remaining:
            return True
    return False


@contextmanager
def predicate_binding(candidate, counters=None):
    """Patch the binding actually called by SQLite; observe the real candidate.

    False without any predicate lexer invocation is an observed early rejection,
    called guard-only below. This never duplicates or predicts the guard. Query
    planning and excerpt lexing remain excluded by the predicate-depth boundary.
    """
    previous_predicate = index_module.contains_literals
    previous_lexer = query.literal_spans
    depth = 0
    if counters is not None:
        counters.update(dict.fromkeys(COUNTER_NAMES, 0))

    def observed_lexer(text):
        if depth:
            counters["lexer_invocations"] += 1
            counters["lexer_input_characters"] += len(text)
        return previous_lexer(text)

    def observed_predicate(text, literals):
        nonlocal depth
        counters["predicate_calls"] += 1
        counters["predicate_input_characters"] += len(text)
        prior_calls = counters["lexer_invocations"]
        depth += 1
        try:
            result = candidate(text, literals)
        finally:
            depth -= 1
        counters["accepted" if result else "rejected"] += 1
        if not result and counters["lexer_invocations"] == prior_calls:
            counters["guard_only_rejections"] += 1
        return result

    try:
        index_module.contains_literals = candidate if counters is None else observed_predicate
        if counters is not None:
            query.literal_spans = observed_lexer
        yield
    finally:
        index_module.contains_literals = previous_predicate
        query.literal_spans = previous_lexer


@contextmanager
def background_variant(variant):
    """Transform authored text before the writer, so the source oracle stays honest."""
    if variant not in {"plain", "collision"}:
        raise ValueError("Unknown corpus variant")
    original = quality_fixtures.background_documents

    def collision_documents(seed, count):
        for document in original(seed, count):
            yield {**document, "sections": [text.replace("archive", "archive++17") for text in document["sections"]]}

    try:
        if variant == "collision":
            quality_fixtures.background_documents = collision_documents
        yield
    finally:
        quality_fixtures.background_documents = original


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def checkpoint(output, report, context):
    """Atomically preserve the last completed step, outside all measured calls.

    A killed writer can leave a sibling temporary file, but cannot truncate the
    previous complete JSON. An interrupted phase's newest observations may be
    absent: this is evidence retention, not an OOM or hard-kill recovery claim.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report["checkpoint"] = {
        "utc_timestamp": datetime.now(timezone.utc).isoformat(),
        "monotonic": time.monotonic(), "context": dict(context), "memory": memory_snapshot(),
    }
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                         prefix=f".{output.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def preserved_report(output, report, context):
    """Keep already appended rows and the precise failing step even on timeout."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        checkpoint(output, report, context)
        yield
    except BaseException as error:
        report["passed"] = False
        report["completed"] = False
        report["error"] = {"type": type(error).__name__, "message": str(error) or type(error).__name__,
                           "context": dict(context)}
        raise
    finally:
        checkpoint(output, report, context)


def latency_summary(values):
    ordered = sorted(values)
    if not ordered:
        return {"samples": 0}
    return {"samples": len(ordered), "median_ms": statistics.median(ordered),
            "p95_ms": ordered[math.ceil(len(ordered) * .95) - 1], "max_ms": ordered[-1],
            "method": "arithmetic median; nearest-rank p95; elapsed wall clock"}


def paired_summary(samples):
    blocks = []
    for block in range(BLOCKS):
        values = {name: [row["elapsed_ms"] for row in samples if row["block"] == block and row["version"] == name]
                  for name in ("A", "B")}
        if any(len(value) != 2 * CALLS_PER_PHASE for value in values.values()):
            continue
        a, b = (statistics.median(values[name]) for name in ("A", "B"))
        blocks.append({"block": block, "A_samples": len(values["A"]), "B_samples": len(values["B"]),
                       "A_median_ms": a, "B_median_ms": b,
                       "improvement_fraction": (a - b) / a if a else 0.0,
                       "regression_fraction": (b - a) / a if a else None, "regression_ms": b - a})
    return {"versions": {name: latency_summary([row["elapsed_ms"] for row in samples if row["version"] == name])
                         for name in ("A", "B")}, "blocks": blocks}


def decision(corpora, quality_passed):
    """Apply frozen thresholds, retaining every counterexample that vetoes adoption."""
    correctness = bool(corpora) and quality_passed and all(
        corpus.get("completed") and corpus.get("database_unchanged") and corpus.get("queries")
        and len(set(corpus.get("planned_query_ids", []))) == len(corpus["queries"])
        and [row.get("id") for row in corpus["queries"]] == corpus.get("planned_query_ids")
        and all(row.get("completed") and row.get("correctness_passed") for row in corpus["queries"])
        for corpus in corpora
    )
    primary = [row for corpus in corpora if corpus.get("variant") == "plain" and corpus.get("background_books") == max(SIZES)
               for row in corpus.get("queries", []) if row.get("id") == "primary-archive-miss"]
    blocks = primary[0].get("paired", {}).get("blocks", []) if len(primary) == 1 else []
    primary_passed = len(blocks) == BLOCKS and {row["block"] for row in blocks} == set(range(BLOCKS)) and all(
        row["improvement_fraction"] >= .20 for row in blocks
    )
    vetoes, incomplete = [], []
    for corpus in corpora:
        for row in corpus.get("queries", []):
            pairs = row.get("paired", {}).get("blocks", [])
            versions = row.get("paired", {}).get("versions", {})
            if (len(pairs) != BLOCKS or {pair["block"] for pair in pairs} != set(range(BLOCKS))
                    or any(versions.get(name, {}).get("samples") != 30 for name in ("A", "B"))
                    or any(pair.get(name + "_samples") != 10 for pair in pairs for name in ("A", "B"))):
                incomplete.append({"variant": corpus["variant"], "background_books": corpus["background_books"], "query": row["id"]})
            if corpus.get("variant") == "plain" and corpus.get("background_books") == max(SIZES) and row["id"] == "primary-archive-miss":
                continue
            regressed = [pair["block"] for pair in pairs
                         if pair["regression_fraction"] is not None and pair["regression_fraction"] > .05 and pair["regression_ms"] > 1.0]
            if len(regressed) >= 2:
                vetoes.append({"variant": corpus["variant"], "background_books": corpus["background_books"],
                               "query": row["id"], "blocks": regressed})
    observed = {(row.get("variant"), row.get("background_books")) for row in corpora}
    expected = {(variant, size) for variant in ("plain", "collision") for size in SIZES}
    expansion_complete = len(corpora) == len(expected) and observed == expected
    return {"accepted": bool(correctness and primary_passed and expansion_complete and not vetoes and not incomplete),
            "correctness_passed": bool(correctness), "frozen_quality_passed": bool(quality_passed),
            "expansion_complete": expansion_complete, "primary_each_block_improves_20_percent": bool(primary_passed),
            "counterexample_vetoes": vetoes, "incomplete_timing_cases": incomplete,
            "rule": "Each largest plain primary median improves >=20%; any counterexample >5% AND >1ms regression in >=2 blocks vetoes."}


def arguments_for(case, by_label):
    result = {"question": case["question"], "limit": 3}
    if case.get("book_filter"):
        result["book_id"] = by_label[case["book_filter"]]
    return result


def verify_answer(case, answer, service, oracle, by_label):
    """Check independently authored labels and unchanged source coordinates."""
    assert answer["question"] == case["question"]
    assert answer["answer_mode"] == "evidence_only" and answer["backend"] == "sqlite-fts5-bm25"
    assert answer["abstained"] == (not answer["passages"])
    assert answer["abstained"] == case["expected_no_match"], "Independent no-match label mismatch"
    assert len(answer["passages"]) <= 3
    if "expected_literals" in case:
        assert answer["query_normalization"]["literal_constraints"] == case["expected_literals"]
    continuations, quotes = [], []
    for rank, passage in enumerate(answer["passages"], 1):
        section = verify_citation(passage, oracle)
        assert passage["citation_id"] == f"[{rank}]"
        for label in (case.get("book_filter"), case.get("expected_book")):
            assert label is None or passage["book_id"] == by_label[label]
        quotes.append(passage["quote"])
        if case.get("expected_excerpt_offset_positive"):
            assert passage["excerpt_offset"] > 0
        source = oracle[passage["book_id"]]["sections"][section]
        current = passage
        if case.get("expected_continuation"):
            assert current["next_offset"] is not None, "Expected the long excerpt to offer continuation"
        # A fixed bound prevents a broken continuation from becoming a hidden loop.
        count = math.ceil((passage["original_char_end"] - passage["original_char_start"]) / 8000)
        for _ in range(count):
            offset = current["next_offset"]
            if offset is None:
                break
            assert offset == current["char_end"] - current["original_char_start"]
            following = service.get_passage(current["chunk_id"], offset)
            assert verify_citation(following, oracle) == section
            assert following["book_id"] == passage["book_id"] and following["chunk_id"] == passage["chunk_id"]
            assert following["char_start"] == current["char_end"]
            assert following["quote"] == source[following["char_start"]:following["char_end"]]
            continuations.append(following)
            current = following
        assert current["next_offset"] is None
        assert current["char_end"] == passage["original_char_end"]
    for quote in case.get("required_quotes", []):
        assert any(quote in result for result in quotes), "Required authored quote missing from excerpts"
    for quote in case.get("required_continuation_quotes", []):
        assert any(quote in result["quote"] for result in continuations), "Required authored continuation quote missing"
    return continuations


def sql_anchors(path, case, by_label):
    plan = query.plan_query(case["question"], drop_question_words=True)
    parameters = [plan.expression]
    condition = "chunks MATCH ?"
    if case.get("book_filter"):
        condition += " AND book_id=?"
        parameters.append(by_label[case["book_filter"]])
    index = index_module.LocalIndex(path)
    try:
        count_sql = "SELECT count(*) FROM chunks WHERE " + condition
        count = index.db.execute(count_sql, parameters).fetchone()[0]
        result = {"expression": plan.expression, "literals": list(plan.literals), "sql": count_sql,
                  "parameters": parameters, "fts_candidates": count}
        if "expected_fts_candidates" in case:
            assert count == case["expected_fts_candidates"], "Probe lost its intended FTS candidate shape"
        if "expected_candidate_rank_minimum" in case:
            rank_sql = "SELECT chunk_id, content, bm25(chunks) AS rank FROM chunks WHERE " + condition + " ORDER BY rank, chunk_id"
            rows = list(index.db.execute(rank_sql, parameters))
            target = case["required_quotes"][0]
            ranks = [rank for rank, row in enumerate(rows, 1) if target in row["content"]]
            assert ranks and min(ranks) >= case["expected_candidate_rank_minimum"]
            result.update(unfiltered_ranking_sql=rank_sql, valid_target_anchor_ranks=ranks,
                          rejected_candidates_ahead=min(ranks) - 1)
        return result
    finally:
        index.close()


def profile_call(service, arguments, candidate):
    profiler = cProfile.Profile(timer=time.process_time)
    with predicate_binding(candidate):
        profiler.enable()
        try:
            response = service.ask(**arguments)
        finally:
            profiler.disable()
    stats = pstats.Stats(profiler)
    rows = []
    for (filename, line, name), (primitive, total, own, cumulative, _callers) in stats.stats.items():
        rows.append({"file": filename, "line": line, "function": name, "primitive_calls": primitive,
                     "calls": total, "own_cpu_seconds": own, "cumulative_cpu_seconds": cumulative})
    return response, {"timer": "time.process_time", "intrusive_excluded_from_latency": True,
                      "total_cpu_seconds": stats.total_tt,
                      "functions": sorted(rows, key=lambda item: item["cumulative_cpu_seconds"], reverse=True)}


def evaluate_case(path, service, case, oracle, by_label, row, context, *, profile=False, save=None):
    candidates = {"A": baseline_contains_literals, "B": query.contains_literals}
    arguments = arguments_for(case, by_label)
    row.update(id=case["id"], case=case, arguments=arguments, completed=False, correctness_passed=False,
               samples=[], counters={}, profiles={})
    context.update(query=case["id"], step="anchor_count")
    row["anchors"] = sql_anchors(path, case, by_label)
    expected = None
    expected_continuations = None
    for name, candidate in candidates.items():
        context.update(step="warm_and_validate", version=name)
        with predicate_binding(candidate):
            answer = service.ask(**arguments)
            continuations = verify_answer(case, answer, service, oracle, by_label)
        if expected is None:
            expected, expected_continuations = answer, continuations
            row["representative_response"] = answer
            row["continuations"] = continuations
        else:
            assert answer == expected and continuations == expected_continuations, "A/B warm response or continuation mismatch"
        context["step"] = "warm_and_validation_complete"
        if save is not None:
            save()
    row["response_sha256"] = object_digest(expected)
    for block in range(BLOCKS):
        for phase, name in enumerate(PHASES):
            with predicate_binding(candidates[name]):
                for repetition in range(CALLS_PER_PHASE):
                    context.update(step="timed_service_call", block=block, phase=phase, version=name, repetition=repetition)
                    started = time.perf_counter()
                    answer = service.ask(**arguments)
                    duration = (time.perf_counter() - started) * 1000
                    sample = {"sequence": len(row["samples"]), "block": block, "phase": phase,
                              "version": name, "repetition": repetition, "elapsed_ms": duration}
                    row["samples"].append(sample)
                    context["step"] = "untimed_response_validation"
                    assert answer == expected, "Timed service response differs from the validated reference"
                    sample["response_equal"] = True
            context["step"] = "measurement_phase_complete"
            row["paired"] = paired_summary(row["samples"])
            if save is not None:
                save()
    row["paired"] = paired_summary(row["samples"])
    for name, candidate in candidates.items():
        context.update(step="instrumented_service_call", version=name)
        counts = row["counters"][name] = {}
        with predicate_binding(candidate, counts):
            answer = service.ask(**arguments)
        assert answer == expected
        assert counts["predicate_calls"] == counts["accepted"] + counts["rejected"]
        assert counts["lexer_invocations"] <= counts["predicate_calls"]
        counts["scope"] = "One untimed service call; only lexing within predicate; character totals are input lengths, not scanned characters."
        context["step"] = "instrumented_call_complete"
        if save is not None:
            save()
        if profile:
            context.update(step="intrusive_cpu_profile", version=name)
            answer, row["profiles"][name] = profile_call(service, arguments, candidate)
            assert answer == expected
            context["step"] = "cpu_profile_complete"
            if save is not None:
                save()
    row.update(correctness_passed=True, completed=True)
    context["step"] = "query_complete"
    if save is not None:
        save()


def quality_evidence(paths):
    if len(paths) != len(FROZEN_QUALITY):
        raise ValueError("Supply exactly two separately executed frozen quality reports")
    evidence, seen = [], set()
    for path in paths:
        path = Path(path)
        report = json.loads(path.read_text(encoding="utf-8"))
        name = report["fixture_file"]
        assert name in FROZEN_QUALITY and name not in seen
        seen.add(name)
        frozen = FROZEN_QUALITY[name]
        fixture_path = Path(__file__).with_name(name)
        assert quality_fixtures.digest(fixture_path) == frozen["sha256"] == report["fixture_sha256"]
        cases = json.loads(fixture_path.read_text(encoding="utf-8"))
        expected_ids = [case["id"] for case in cases["queries"]]
        assert report["case_counts"]["queries"] == frozen["queries"] == len(expected_ids)
        assert report["case_counts"]["gates"] == frozen["gates"]
        assert report["profile"] == frozen["profile"]
        assert report["passed"] and report["database_unchanged"] and "error" not in report
        phases = {"service": report["service"], "mcp:auto": report["mcp"]["auto"], "mcp:legacy": report["mcp"]["legacy"]}
        outcomes = {}
        for phase, result in phases.items():
            summary = result["summary"]
            assert result["completed"] and not summary["gate_failures"]
            assert [row["id"] for row in result["queries"]] == expected_ids
            for row, case in zip(result["queries"], cases["queries"], strict=True):
                assert row["gate"] == case["gate"] and row["expected_no_match"] == case["expected_no_match"]
                assert row["question"] == case["question"]
            assert summary["gate_queries"] == frozen["gates"]
            assert summary["all"]["queries"] == frozen["queries"]
            assert sorted(summary["diagnostic_failures"]) == sorted(frozen["diagnostic_failures"])
            assert [row["id"] for row in result["queries"] if row["gate"] and not row["passed"]] == []
            assert sorted(row["id"] for row in result["queries"] if not row["gate"] and not row["passed"]) == sorted(frozen["diagnostic_failures"])
            outcomes[phase] = {"completed": True, "gate_failures": [], "diagnostic_failures": summary["diagnostic_failures"],
                               "queries": len(result["queries"]), "summary": summary}
        evidence.append({"report": str(path), "report_sha256": quality_fixtures.digest(path), "fixture": name,
                         "fixture_sha256": frozen["sha256"], "database_unchanged": True, "outcomes": outcomes})
    assert seen == set(FROZEN_QUALITY)
    return evidence


def memory_snapshot():
    result = {"driver_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    for name in ("memory.peak", "memory.max", "cpu.max"):
        path = Path("/sys/fs/cgroup") / name
        result[name] = path.read_text().strip() if path.exists() else None
    return result


def read_budget(path):
    """Validate the coordinator command's pre-existing monotonic deadline."""
    path = Path(path)
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("Budget file must contain a JSON object")
    start, end = (values.get(name) for name in ("command_started_monotonic", "deadline_monotonic"))
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in (start, end)):
        raise ValueError("Budget timestamps must be finite numeric monotonic seconds")
    now = time.monotonic()
    remaining = end - now
    if (start < 0 or start > now or not math.isclose(end - start, 510.0, rel_tol=0, abs_tol=0.000001)
            or not 0 < remaining <= 510):
        raise ValueError("Budget must have an unexpired command-relative 510-second deadline")
    return {"command_started_monotonic": start, "deadline_monotonic": end,
            "remaining_seconds_at_read": remaining, "file_sha256": quality_fixtures.digest(path),
            "command_budget_seconds": 510, "nominal_outer_reserve_seconds": 90,
            "scope": "Deadline measured from the coordinator command's first step, before installation/tests/quality. "
                     "The outer 600-second clock begins before archive/container setup and is unavailable here; "
                     "the 90-second startup/cleanup reserve does not guarantee the actual remaining outer time."}


@contextmanager
def deadline(deadline_monotonic):
    def expired(_number, _frame):
        raise TimeoutError("Coordinator command-relative 510-second budget exhausted")

    remaining = deadline_monotonic - time.monotonic()
    if not math.isfinite(remaining) or not 0 < remaining <= 510:
        raise ValueError("Cannot arm an expired or invalid coordinator deadline")
    old = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def run(output, *, quality_reports, budget_file):
    report = {"passed": False, "completed": False, "policy": "PERF1", "corpora": [], "active_context": {}}
    context = report["active_context"]
    started = time.monotonic()

    def save():
        checkpoint(output, report, context)

    with preserved_report(output, report, context):
        context.update(stage="read_command_budget")
        report["command_budget"] = read_budget(budget_file)
        with deadline(report["command_budget"]["deadline_monotonic"]):
            context.update(stage="environment")
            if platform.system() != "Linux" or os.environ.get("PYTHONHASHSEED") != "0":
                raise ValueError("Use the reviewed Linux Spark job with PYTHONHASHSEED=0")
            if sys.flags.optimize:
                raise ValueError("Assertions are correctness gates; optimized Python is forbidden")
            report["environment"] = {
                "python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
                "sqlite": sqlite3.sqlite_version, "PYTHONHASHSEED": os.environ["PYTHONHASHSEED"],
                "packages": {package: version(package) for package in ("pydantic", "pypdf")},
                "memory_before": memory_snapshot(), "coordinator_budget": {"seconds": 600, "cpus": 2, "memory_gib": 4},
            }
            policy = Path(__file__).with_name("PERFORMANCE-POLICY.md")
            assert quality_fixtures.digest(policy) == POLICY_SHA256
            source_paths = [Path(__file__), Path(__file__).with_name("literal_performance_fixtures.py"),
                            Path(quality_fixtures.__file__), Path(query.__file__), Path(index_module.__file__),
                            Path(inspect.getfile(LibraryService)), policy]
            report["source_hashes"] = {str(path): quality_fixtures.digest(path) for path in source_paths}
            report["baseline_copy"] = {
                "commit": BASELINE_COMMIT, "path": BASELINE_PATH, "origin_file_sha256": BASELINE_SOURCE_SHA256,
                "copied_function_sha256": hashlib.sha256(inspect.getsource(baseline_contains_literals).encode()).hexdigest(),
                "only_copy_edits": "Function renamed baseline_contains_literals; literal_spans qualified as query.literal_spans for instrumentation.",
            }
            report["measurement"] = {"sizes": list(SIZES), "variants": ["plain", "collision"], "blocks": BLOCKS,
                                     "phases": list(PHASES), "calls_per_phase": CALLS_PER_PHASE,
                                     "samples_per_version_per_query": 30, "warm_calls_per_version_per_query": 1,
                                     "timing_scope": "LibraryService.ask only; patches, validation, counters, profiles, checkpoints and corpus setup excluded",
                                     "set_order_note": "PYTHONHASHSEED=0 fixes string set order; input order is not guaranteed short-circuit order."}
            context.update(stage="frozen_quality_reports")
            report["frozen_quality"] = quality_evidence(quality_reports)
            cases = fixtures()
            report["authored_fixture_sha256"] = object_digest(cases)
            context.update(stage="setup_complete")
            save()
            for size in SIZES:
                for variant in ("plain", "collision"):
                    context.clear()
                    context.update(stage="build_corpus", background_books=size, variant=variant)
                    corpus = {"variant": variant, "background_books": size, "completed": False, "queries": []}
                    report["corpora"].append(corpus)
                    save()
                    with tempfile.TemporaryDirectory(prefix=f"literal-perf-{variant}-{size}-") as directory:
                        with background_variant(variant):
                            path, oracle, by_label, manifest = quality_fixtures.build_corpus(Path(directory), cases, size)
                        corpus["manifest"] = manifest
                        before = quality_fixtures.digest(path)
                        corpus["database_sha256_before"] = before
                        context.update(stage="corpus_setup_complete")
                        save()
                        try:
                            service = LibraryService(SQLiteRetriever(path))
                            selected = broad_cases(variant) + (cases["queries"] if size == max(SIZES) else [])
                            corpus["planned_query_ids"] = [case["id"] for case in selected]
                            for case in selected:
                                row = {}
                                corpus["queries"].append(row)
                                context.update(stage="query")
                                evaluate_case(path, service, case, oracle, by_label, row, context,
                                              profile=size == max(SIZES) and case["id"] == "primary-archive-miss", save=save)
                            corpus["completed"] = True
                        finally:
                            corpus["database_sha256_after"] = quality_fixtures.digest(path)
                            corpus["database_unchanged"] = corpus["database_sha256_after"] == before
                        assert corpus["database_unchanged"]
                        context.update(stage="corpus_complete")
                        save()
                    print(json.dumps({"stage": "corpus_complete", "variant": variant, "background_books": size,
                                      "queries": len(corpus["queries"]), "elapsed_seconds": time.monotonic() - started}), flush=True)
            context.clear()
            context.update(stage="decision")
            report["decision"] = decision(report["corpora"], True)
            report.update(passed=report["decision"]["accepted"], completed=True, elapsed_seconds=time.monotonic() - started,
                          memory_after=memory_snapshot())
            report["limitations"] = [
                "Synthetic sequential warmed service calls; no cold-cache, concurrency, private-library or capacity claim.",
                "Frozen disclosed regression cases are checked separately; their labels are never applied to the changed corpus.",
                "Profile CPU times are intrusive and excluded from latency; input-character totals do not measure characters scanned.",
                "The coordinator must separately enforce 2 CPUs/4 GiB and combine independent regression-test results with this decision.",
                "The 510-second deadline starts at the coordinator command; its 90-second nominal reserve does not expose or guarantee the actual outer deadline.",
                "Atomic checkpoints retain completed phases; hard kill or OOM can lose the newest interrupted phase or checkpoint.",
            ]
            context.update(stage="complete")
    return report["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quality-report", type=Path, action="append", required=True)
    parser.add_argument("--budget-file", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.output, quality_reports=args.quality_report, budget_file=args.budget_file) else 1)
