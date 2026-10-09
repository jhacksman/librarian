"""Bounded synthetic lexical/service/MCP quality exercise. Run only on Spark."""

import argparse
import json
import os
import platform
import resource
import sys
import tempfile
import time
from importlib.metadata import version
from pathlib import Path

from quality_fixtures import BACKGROUND_BOOKS, build_corpus, digest, load_cases, resolve_goldens
from quality_scoring import latency_summary, score_case, summarize, verify_citation

from ingest.local.index import LocalIndex
from ingest.local.service import LibraryRequestError, LibraryService, SQLiteRetriever


def elapsed_ms(started):
    return (time.perf_counter() - started) * 1000


def memory_snapshot():
    # Linux ru_maxrss is KiB. Report the two process scopes separately, never sum maxima.
    output = {
        "driver_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "largest_exited_child_peak_rss_mib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024,
        "scope": "Driver lifetime high-water mark; child value is the largest exited child, not aggregate live memory",
    }
    for name in ("memory.peak", "memory.max", "cpu.max", "pids.max"):
        path = Path("/sys/fs/cgroup") / name
        output[name] = path.read_text().strip() if path.exists() else None
    output["cgroup_scope"] = "Whole job container, including installation and any preceding checks; memory values are bytes"
    return output


def arguments_for(case, by_label):
    arguments = {"question": case["question"], "limit": 3}
    if case.get("book_filter"):
        arguments["book_id"] = by_label[case["book_filter"]]
    return arguments


def set_context(context, phase, check, case=None):
    context.clear()
    context.update(phase=phase, check=check, case=case)


def invalid_requests(chunk_id, chunk_length):
    return [
        ("object-question", "ask_library", {"question": {"text": "retry"}}),
        ("array-question", "ask_library", {"question": ["retry"]}),
        ("null-question", "ask_library", {"question": None}),
        ("oversized-question", "ask_library", {"question": "x" * 2001}),
        ("too-many-terms", "ask_library", {"question": "zeta " * 129}),
        ("fractional-limit", "ask_library", {"question": "retry", "limit": 1.5}),
        ("boolean-limit", "ask_library", {"question": "retry", "limit": True}),
        ("zero-limit", "ask_library", {"question": "retry", "limit": 0}),
        ("overflow-limit", "ask_library", {"question": "retry", "limit": 11}),
        ("string-null-filter", "ask_library", {"question": "retry", "book_id": "null"}),
        ("uppercase-filter", "ask_library", {"question": "retry", "book_id": "A" * 64}),
        ("one-past-passage-end", "get_passage", {"chunk_id": chunk_id, "offset": chunk_length}),
        ("oversized-passage-offset", "get_passage", {"chunk_id": chunk_id, "offset": 2_147_483_648}),
        ("negative-passage-offset", "get_passage", {"chunk_id": chunk_id, "offset": -1}),
        ("boolean-passage-offset", "get_passage", {"chunk_id": chunk_id, "offset": True}),
        ("oversized-book-limit", "list_books", {"limit": 51}),
        ("oversized-catalog-offset", "list_books", {"offset": 1_000_001}),
    ]


def boundary_answers(service, oracle, records, context):
    for question in ("LANTERN9", "PAGE2END", "PAGE3START", "naïve compass"):
        set_context(context, "service", "passage_boundary_lookup", question)
        passage = service.ask(question)["passages"][0]
        length = passage["original_char_end"] - passage["original_char_start"]
        for offset in (0, length - 1):
            set_context(context, "service", "passage_boundary_offset", f"{question}:{offset}")
            result = service.get_passage(passage["chunk_id"], offset)
            verify_citation(result, oracle)
            if offset == length - 1:
                assert len(result["quote"]) == 1 and result["next_offset"] is None
            records.append({"question": question, "offset": offset, "result": result})


def service_contract_checks(service, oracle, report, context):
    boundaries = report["passage_boundaries"]
    boundary_answers(service, oracle, boundaries, context)
    example = boundaries[0]["result"]
    length = example["original_char_end"] - example["original_char_start"]
    methods = {"ask_library": service.ask, "get_passage": service.get_passage, "list_books": service.books}
    for label, name, arguments in invalid_requests(example["chunk_id"], length):
        set_context(context, "service", "invalid_request", label)
        try:
            methods[name](**arguments)
        except LibraryRequestError:
            report["invalid_requests"].append({"case": label, "rejected": True})
        else:
            raise AssertionError(f"Invalid service request accepted: {label}")
    for label, arguments in (("question-2000-chars", {"question": "x" * 2000}),
                             ("question-128-terms", {"question": "zeta " * 128}),
                             ("unknown-book", {"question": "retry", "book_id": "0" * 64})):
        set_context(context, "service", "valid_boundary_request", label)
        answer = service.ask(**arguments)
        assert answer["abstained"]
        report["valid_boundary_requests"].append({"case": label, "accepted_no_match": True, "arguments": arguments})


def service_evaluation(service, cases, oracle, by_label, repeats, report, context, profile="full"):
    report.update(completed=False, queries=[], invalid_requests=[], valid_boundary_requests=[], passage_boundaries=[],
                  latency={"first_pass_ms": [], "warm_ms": [], "warm_per_query_ms": {}, "common_term_limit_10_ms": []})
    rows, answers = report["queries"], {}
    timings = report["latency"]
    first_pass, warm, per_query = timings["first_pass_ms"], timings["warm_ms"], timings["warm_per_query_ms"]
    for case in cases["queries"]:
        set_context(context, "service", "first_pass_query", case["id"])
        arguments = arguments_for(case, by_label)
        started = time.perf_counter()
        answer = service.ask(**arguments)
        first_pass.append(elapsed_ms(started))
        answers[case["id"]] = answer
        rows.append(score_case(case, answer, resolve_goldens(case, oracle, by_label), oracle, arguments.get("book_id")))
        per_query[case["id"]] = []
    for _ in range(repeats):
        for case in cases["queries"]:
            set_context(context, "service", "warm_query", case["id"])
            started = time.perf_counter()
            answer = service.ask(**arguments_for(case, by_label))
            duration = elapsed_ms(started)
            assert answer == answers[case["id"]]
            warm.append(duration)
            per_query[case["id"]].append(duration)
    common_term = timings["common_term_limit_10_ms"]
    for repetition in range(15):
        set_context(context, "service", "common_term_limit_10", str(repetition))
        started = time.perf_counter()
        answer = service.ask("archive", limit=10)
        common_term.append(elapsed_ms(started))
        assert len(answer["passages"]) == 10
        for passage in answer["passages"]:
            verify_citation(passage, oracle)
    if profile == "full":
        service_contract_checks(service, oracle, report, context)
    report.update(summary=summarize(rows), completed=True)
    timings.update(first_pass=latency_summary(first_pass), warm=latency_summary(warm),
                   common_term_limit_10=latency_summary(common_term), scope="Service method call; excludes score and citation validation")
    return answers


async def mcp_evaluation(path, service, cases, oracle, by_label, answers, service_report, mode, repeats, report, context):
    import anyio
    from mcp import Client, StdioServerParameters

    assert version("mcp") == "2.3.0"
    parameters = StdioServerParameters(
        command=sys.executable, args=["-m", "ingest.local.mcp_server", "--index", str(path)],
        env={"PYTHONPATH": os.environ["PYTHONPATH"], "PYTHONDONTWRITEBYTECODE": "1"},
    )
    phase = f"mcp:{mode}"
    set_context(context, phase, "startup")
    startup = time.perf_counter()
    report.update(mode=mode, completed=False, queries=[], invalid_requests=[], passage_boundaries=[],
                  latency={"first_pass_ms": [], "warm_ms": [], "warm_per_query_ms": {}})
    with anyio.fail_after(180):
        async with Client(parameters, mode=mode, read_timeout_seconds=10, cache=None) as client:
            report["startup_ms"] = elapsed_ms(startup)
            report["protocol_version"] = client.protocol_version
            set_context(context, phase, "discovery")
            assert client.protocol_version == {"auto": "2026-07-28", "legacy": "2025-11-25"}[mode]
            assert {tool.name for tool in (await client.list_tools()).tools} == {
                "ask_library", "get_passage", "list_books", "library_status"}

            async def call(name, arguments):
                context["tool"] = name
                result = await client.call_tool(name, arguments)
                assert not result.is_error and isinstance(result.structured_content, dict)
                assert len(result.content) == 1 and json.loads(result.content[0].text) == result.structured_content
                assert str(path.parent) not in result.model_dump_json()
                return result.structured_content

            expected_status = service.status()
            set_context(context, phase, "initial_status")
            assert await call("library_status", {}) == expected_status
            timings = report["latency"]
            first_pass, warm, per_query = timings["first_pass_ms"], timings["warm_ms"], timings["warm_per_query_ms"]
            for case in cases["queries"]:
                set_context(context, phase, "first_pass_query", case["id"])
                arguments = arguments_for(case, by_label)
                started = time.perf_counter()
                answer = await call("ask_library", arguments)
                first_pass.append(elapsed_ms(started))
                assert answer == answers[case["id"]]
                report["queries"].append(score_case(case, answer, resolve_goldens(case, oracle, by_label), oracle, arguments.get("book_id")))
                per_query[case["id"]] = []
            for _ in range(repeats):
                for case in cases["queries"]:
                    set_context(context, phase, "warm_query", case["id"])
                    started = time.perf_counter()
                    answer = await call("ask_library", arguments_for(case, by_label))
                    duration = elapsed_ms(started)
                    assert answer == answers[case["id"]]
                    warm.append(duration)
                    per_query[case["id"]].append(duration)
            timings.update(first_pass=latency_summary(first_pass), warm=latency_summary(warm),
                           scope="Client call plus structured/text parity and path-check validation; not pure transport time")
            for boundary in service_report["passage_boundaries"]:
                set_context(context, phase, "passage_boundary", f'{boundary["question"]}:{boundary["offset"]}')
                expected = boundary["result"]
                actual = await call("get_passage", {"chunk_id": expected["chunk_id"], "offset": boundary["offset"]})
                verify_citation(actual, oracle)
                assert actual == expected
                report["passage_boundaries"].append(boundary)
            if service_report["passage_boundaries"]:
                example = service_report["passage_boundaries"][0]["result"]
                length = example["original_char_end"] - example["original_char_start"]
                for label, name, arguments in invalid_requests(example["chunk_id"], length):
                    set_context(context, phase, "invalid_request", label)
                    result = await client.call_tool(name, arguments)
                    assert result.is_error, label
                    assert str(path.parent) not in result.model_dump_json()
                    assert await call("library_status", {}) == expected_status
                    report["invalid_requests"].append({"case": label, "rejected": True, "recovered": True})
                for request in service_report["valid_boundary_requests"]:
                    set_context(context, phase, "valid_boundary_request", request["case"])
                    assert (await call("ask_library", request["arguments"]))["abstained"]
            # Catalog identity, rather than ambiguous title, must survive pagination at scale.
            offset, books = 0, []
            while offset is not None:
                set_context(context, phase, "catalog_page", str(offset))
                page = await call("list_books", {"limit": 50, "offset": offset})
                assert len(page["books"]) <= 50
                books.extend(page["books"])
                offset = page["next_offset"]
                assert len(books) <= len(oracle)
            assert len(books) == len(oracle) and {book["book_id"] for book in books} == set(oracle)
            report["catalog_books"] = len(books)
    report["summary"] = summarize(report["queries"])
    report["passed"] = not report["summary"]["gate_failures"]
    report["completed"] = True


def literal_cost_probe(path, service, oracle, report, context):
    """Observe an intentionally broad absent literal; no hidden candidate cap."""
    index = LocalIndex(path)
    try:
        candidates = index.db.execute('SELECT count(*) FROM chunks WHERE chunks MATCH ?', ('"archive"',)).fetchone()[0]
    finally:
        index.close()
    report.update(question="archive++", fts_anchor='"archive"', fts_candidates=candidates,
                  samples_ms=[], completed=False,
                  scope="Service calls over every FTS anchor candidate; observational cost, not a latency gate")
    for repetition in range(15):
        set_context(context, "literal_cost", "broad_candidate_scan", str(repetition))
        started = time.perf_counter()
        answer = service.ask("archive++", limit=10)
        report["samples_ms"].append(elapsed_ms(started))
        for passage in answer["passages"]:
            verify_citation(passage, oracle)
        observation = {"abstained": answer["abstained"], "hits": len(answer["passages"]),
                       "retrieval_query": answer["retrieval_query"],
                       "query_normalization": answer.get("query_normalization")}
        if repetition:
            assert report["observation"] == observation
        report["observation"] = observation
    report.update(completed=True, latency=latency_summary(report["samples_ms"]))


def run(output, repeats, *, cases_path=None, profile="full", probe_literal_cost=False):
    if platform.system() != "Linux" or not 1 <= repeats <= 5:
        raise ValueError("Use the reviewed Linux Spark job and 1-5 warm repetitions")
    if profile not in {"full", "query-comparison"}:
        raise ValueError("Unknown evaluation profile")
    default_cases = Path(__file__).with_name("quality-cases.json")
    cases_path = default_cases if cases_path is None else Path(cases_path)
    if profile == "full" and cases_path.resolve() != default_cases.resolve():
        raise ValueError("Alternate cases require the explicit query-comparison profile")
    report = {
        "passed": False, "fixture_file": cases_path.name, "fixture_provenance": None, "fixture_sha256": None,
        "literal_cost_probe_enabled": probe_literal_cost,
        "python": platform.python_version(), "platform": platform.machine(), "sdk_version": version("mcp"),
        "backend": "sqlite-fts5-bm25", "profile": profile,
        "contract_probes": {"status": "enabled" if profile == "full" else "not_run",
                            "reason": "Default-fixture passage boundaries and invalid-request probes require the full profile"},
        "warm_repetitions": repeats, "memory_before": memory_snapshot(),
        "limits": [
            "Synthetic development corpus; repeated vocabulary measures volume, not production relevance or semantic retrieval.",
            "Only citation/filter/input/no-match gates determine success; diagnostic ranking misses remain visible.",
            "First pass follows ingestion and is not a cold disk-cache benchmark. Warm samples are sequential, not concurrent load.",
            "The driver retains fixture oracle text; RSS includes generation and verification overhead.",
            "Quoted instruction checks cover evidence-only transport, not model prompt-injection resistance.",
            "The full profile exercises malformed tool arguments; query-comparison omits these probes. Neither is raw framing fuzzing or a network service test.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    context = report["active_context"] = {}
    try:
        set_context(context, "setup", "load_cases")
        report["fixture_sha256"] = digest(cases_path)
        cases = load_cases(cases_path)
        report.update(fixture_provenance=cases["provenance"],
                      case_counts={"queries": len(cases["queries"]), "documents": len(cases["documents"]),
                                   "gates": sum(case["gate"] for case in cases["queries"]),
                                   "diagnostics": sum(not case["gate"] for case in cases["queries"])})
        set_context(context, "setup", "build_corpus")
        with tempfile.TemporaryDirectory(prefix="librarian-quality-") as directory:
            path, oracle, by_label, corpus = build_corpus(Path(directory), cases, BACKGROUND_BOOKS)
            report["corpus"] = corpus
            print(json.dumps({"stage": "corpus_ready", "books": corpus["books"], "words": corpus["words"], "chunks": corpus["chunks"]}), flush=True)
            before = digest(path)
            service = LibraryService(SQLiteRetriever(path))
            report["service"] = {}
            answers = service_evaluation(service, cases, oracle, by_label, repeats, report["service"], context, profile)
            if probe_literal_cost:
                report["literal_cost"] = {}
                literal_cost_probe(path, service, oracle, report["literal_cost"], context)
            report["memory_after_service"] = memory_snapshot()
            report["mcp"] = {}
            import anyio
            for mode in ("auto", "legacy"):
                report["mcp"][mode] = {}
                anyio.run(mcp_evaluation, path, service, cases, oracle, by_label, answers, report["service"], mode, repeats, report["mcp"][mode], context)
            set_context(context, "final", "database_unchanged")
            report["database_unchanged"] = digest(path) == before
            assert report["database_unchanged"]
            report["memory_after_mcp"] = memory_snapshot()
            report["passed"] = not report["service"]["summary"]["gate_failures"] and all(item["passed"] for item in report["mcp"].values())
            set_context(context, "complete", "finished")
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error) or type(error).__name__,
                           "context": dict(context)}
        raise
    finally:
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "summary": report["service"]["summary"],
                      "service_warm": report["service"]["latency"]["warm"], "memory": report["memory_after_mcp"]}, indent=2))
    return report["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warm-repetitions", type=int, default=3)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--profile", choices=("full", "query-comparison"), default="full")
    parser.add_argument("--literal-cost-probe", action="store_true")
    args = parser.parse_args()
    raise SystemExit(0 if run(args.output, args.warm_repetitions, cases_path=args.cases, profile=args.profile, probe_literal_cost=args.literal_cost_probe) else 1)
