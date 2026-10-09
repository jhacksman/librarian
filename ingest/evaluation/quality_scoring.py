"""Source-span quality scoring, independent of ranking and query rewriting."""

import math
import re
from collections import defaultdict

from ingest.local.index import PIPELINE_VERSION


def verify_citation(passage, oracle):
    source = oracle[passage["book_id"]]
    if source["format"] == "pdf":
        section = passage["page_start"]
        assert type(section) is int and passage["page_end"] == section
        assert passage["epub_member"] is None
    else:
        match = re.fullmatch(r"s([1-9][0-9]*)\.xhtml", passage["epub_member"])
        assert match and passage["page_start"] is None and passage["page_end"] is None
        section = int(match[1])
    text = source["sections"][section]
    start, end = passage["char_start"], passage["char_end"]
    assert type(start) is int and type(end) is int and 0 <= start < end <= len(text)
    assert text[start:end] == passage["quote"]
    assert passage["source_sha256"] == passage["book_id"]
    assert passage["title"] == source["title"]
    assert re.fullmatch(re.escape(passage["book_id"]) + r":[0-9]+", passage["chunk_id"])
    chunk = source["chunks"].get(passage["chunk_id"])
    assert chunk is not None, "Unknown indexed chunk ID"
    assert chunk["section"] == section, "Chunk ID belongs to another source section"
    assert chunk["char_start"] <= start < end <= chunk["char_end"], "Excerpt outside indexed chunk"
    assert passage["original_char_start"] == chunk["char_start"]
    assert passage["original_char_end"] == chunk["char_end"]
    assert passage["excerpt_offset"] == start - chunk["char_start"]
    assert passage["source_uri"] == f'librarian://books/{passage["book_id"]}/chunks/{passage["chunk_id"].split(":")[1]}'
    assert passage["content_kind"] == "source_excerpt"
    assert passage["offset_basis"] == "extracted_section_unicode_codepoints"
    assert passage["pipeline_version"] == PIPELINE_VERSION
    assert len(passage["quote"]) <= 8000
    assert "source_path" not in passage
    return section


def score_case(case, answer, goldens, oracle, book_filter=None):
    assert answer["question"] == case["question"] and answer["answer_mode"] == "evidence_only"
    assert answer["backend"] == "sqlite-fts5-bm25"
    passages = answer["passages"]
    assert answer["abstained"] == (not passages) and len(passages) <= 3
    matched, first_rank, relevant_hits, observed = set(), None, 0, []
    for rank, passage in enumerate(passages, 1):
        section = verify_citation(passage, oracle)
        assert passage["citation_id"] == f"[{rank}]"
        assert book_filter is None or passage["book_id"] == book_filter
        covered = [number for number, golden in enumerate(goldens)
                   if golden["book_id"] == passage["book_id"] and golden["section"] == section
                   and passage["char_start"] <= golden["char_start"]
                   and passage["char_end"] >= golden["char_end"]]
        matched.update(covered)
        if covered:
            relevant_hits += 1
            first_rank = first_rank or rank
        observed.append({"rank": rank, "document": oracle[passage["book_id"]]["label"],
                         "section": section, "book_id": passage["book_id"], "chunk_id": passage["chunk_id"],
                         "char_start": passage["char_start"], "char_end": passage["char_end"],
                         "quote": passage["quote"], "matched_goldens": covered, "citation_verified": True})
    no_match_correct = answer["abstained"] == case["expected_no_match"]
    recall = len(matched) / len(goldens) if goldens else None
    precision = relevant_hits / len(passages) if passages else None
    passed = no_match_correct and (recall is None or recall == 1) and (precision is None or precision == 1)
    return {
        "id": case["id"], "category": case["category"], "gate": case["gate"], "question": case["question"],
        "retrieval_query": answer["retrieval_query"], "query_normalization": answer.get("query_normalization"),
        "book_filter": case.get("book_filter"),
        "expected_no_match": case["expected_no_match"], "observed_no_match": answer["abstained"],
        "no_match_label_correct": no_match_correct, "goldens": goldens, "hits": observed,
        "source_span_recall_at_3": recall, "precision_at_returned_hits": precision,
        "reciprocal_rank_at_3": 1 / first_rank if first_rank else 0.0,
        "matched_golden_count": len(matched), "passed": passed,
    }


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["category"]].append(row)

    def metrics(items):
        positives = [row for row in items if row["goldens"]]
        negatives = [row for row in items if row["expected_no_match"]]
        return {
            "queries": len(items), "positive_queries": len(positives), "negative_queries": len(negatives),
            "macro_source_span_recall_at_3": sum(row["source_span_recall_at_3"] for row in positives) / len(positives) if positives else None,
            "mrr_at_3": sum(row["reciprocal_rank_at_3"] for row in positives) / len(positives) if positives else None,
            "correct_no_match_negatives": sum(row["observed_no_match"] for row in negatives),
            "missing_evidence_queries": [row["id"] for row in positives if row["source_span_recall_at_3"] < 1],
            "extraneous_hit_queries": [row["id"] for row in items if row["precision_at_returned_hits"] is not None and row["precision_at_returned_hits"] < 1],
        }
    return {"all": metrics(rows), "by_category": {key: metrics(value) for key, value in sorted(groups.items())},
            "gate_queries": sum(row["gate"] for row in rows),
            "gate_failures": [row["id"] for row in rows if row["gate"] and not row["passed"]],
            "diagnostic_failures": [row["id"] for row in rows if not row["gate"] and not row["passed"]],
            "verified_citations": sum(len(row["hits"]) for row in rows)}


def latency_summary(values):
    ordered = sorted(values)
    if not ordered:
        return {"samples": 0}
    return {"samples": len(ordered), "p50_ms": ordered[math.ceil(len(ordered) * .50) - 1],
            "p95_ms": ordered[math.ceil(len(ordered) * .95) - 1], "max_ms": ordered[-1],
            "method": "nearest-rank percentile; elapsed wall clock"}
