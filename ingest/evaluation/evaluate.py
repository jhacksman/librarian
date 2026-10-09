"""Small deterministic retrieval regression gate. Execute only on Spark CI."""

import argparse
import json
import tempfile
from html import escape
from pathlib import Path
from zipfile import ZipFile

from ingest.local.extract import extract
from ingest.local.index import LocalIndex


def write_fixture(path, document):
    sections = document["sections"]
    manifest = "".join(f'<item id="s{i}" href="s{i}.xhtml" media-type="application/xhtml+xml"/>' for i in range(1, len(sections) + 1))
    spine = "".join(f'<itemref idref="s{i}"/>' for i in range(1, len(sections) + 1))
    with ZipFile(path, "w") as archive:
        archive.writestr("META-INF/container.xml", '<container><rootfiles><rootfile full-path="book.opf"/></rootfiles></container>')
        archive.writestr("book.opf", f'<package xmlns="http://www.idpf.org/2007/opf"><metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>{escape(document["title"])}</dc:title></metadata><manifest>{manifest}</manifest><spine>{spine}</spine></package>')
        for number, text in enumerate(sections, 1):
            archive.writestr(f"s{number}.xhtml", f"<h1>Section {number}</h1><p>{escape(text)}</p>")


def evaluate(cases_path):
    cases = json.loads(cases_path.read_text())
    with tempfile.TemporaryDirectory(prefix="librarian-evaluation-") as directory:
        root = Path(directory)
        index = LocalIndex(root / "evaluation.sqlite", create=True)
        try:
            identities = {}
            for document in cases["documents"]:
                source = root / (document["id"] + ".epub")
                write_fixture(source, document)
                identities[document["id"]] = index.ingest(source)["book_id"]
            positive, retrieved, reciprocal, negatives, correct_negatives, citations = 0, 0, 0.0, 0, 0, 0
            details = []
            for case in cases["queries"]:
                results = index.search(case["query"], limit=3)
                rank = None
                if case["document"] is None:
                    negatives += 1
                    correct_negatives += not results
                else:
                    positive += 1
                    for position, hit in enumerate(results, 1):
                        if hit.book_id == identities[case["document"]] and hit.metadata["epub_member"] == f's{case["section"]}.xhtml':
                            rank = position
                            break
                    if rank:
                        retrieved += 1
                        reciprocal += 1 / rank
                for hit in results:
                    content = extract(Path(hit.metadata["source_path"]))
                    section = next(ch for ch in content.chapters if content.metadata["section_sources"][str(ch.number)] == hit.metadata["epub_member"])
                    assert section.content[hit.metadata["char_start"]:hit.metadata["char_end"]] == hit.content
                    citations += 1
                details.append({"query": case["query"], "rank": rank, "hits": len(results)})
            report = {"fixture_provenance": cases["provenance"], "positive_queries": positive,
                      "recall_at_3": retrieved / positive, "mrr_at_3": reciprocal / positive,
                      "negative_queries": negatives, "correct_abstentions": correct_negatives,
                      "verified_citations": citations, "queries": details}
            report["passed"] = retrieved == positive and correct_negatives == negatives and reciprocal == positive
            return report
        finally:
            index.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(Path(__file__).with_name("synthetic-cases.json"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
