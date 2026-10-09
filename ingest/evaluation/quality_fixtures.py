"""Original deterministic fixtures for the Spark-only lexical quality exercise."""

import hashlib
import json
import random
import re
import time
from html import escape
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile, ZipInfo

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from ingest.local.extract import extract
from ingest.local.index import PIPELINE_VERSION, LocalIndex

BACKGROUND_BOOKS = 128
BACKGROUND_SECTIONS = 16
WORDS_PER_SECTION = 480
VOCABULARY = (
    "archive", "catalogue", "inventory", "shelf", "ledger", "folio", "volume", "record",
    "collection", "notation", "appendix", "reference", "diagram", "summary", "specimen", "register",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_epub(path, document):
    sections = document["sections"]
    manifest = "".join(f'<item id="s{i}" href="s{i}.xhtml" media-type="application/xhtml+xml"/>'
                       for i in range(1, len(sections) + 1))
    spine = "".join(f'<itemref idref="s{i}"/>' for i in range(1, len(sections) + 1))
    files = {
        "mimetype": "application/epub+zip",
        "META-INF/container.xml": '<container><rootfiles><rootfile full-path="book.opf"/></rootfiles></container>',
        "book.opf": ('<package xmlns="http://www.idpf.org/2007/opf">'
                     '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
                     f'<dc:title>{escape(document["title"])}</dc:title></metadata>'
                     f'<manifest>{manifest}</manifest><spine>{spine}</spine></package>'),
    }
    files.update({f"s{i}.xhtml": f"<p>{escape(text)}</p>" for i, text in enumerate(sections, 1)})
    with ZipFile(path, "w", compression=ZIP_STORED) as archive:
        for name, text in files.items():
            entry = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.external_attr = 0o600 << 16
            archive.writestr(entry, text.encode("utf-8"))


def write_pdf(path, document):
    writer = PdfWriter()
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                             NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    font_reference = writer._add_object(font)
    for text in document["sections"]:
        # Fixtures deliberately use ASCII, one text line per physical page.
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        page = writer.add_blank_page(900, 792)
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_reference})})
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 10 Tf 30 700 Td ({escaped}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    writer.add_metadata({"/Title": document["title"], "/Author": "Librarian synthetic fixture"})
    writer.write(path)


def load_cases(path):
    cases = json.loads(path.read_text(encoding="utf-8"))
    documents = {document["id"]: document for document in cases["documents"]}
    if len(documents) != len(cases["documents"]):
        raise ValueError("Duplicate document labels")
    identifiers = [case["id"] for case in cases["queries"]]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate query labels")
    for document in documents.values():
        if not re.fullmatch(r"[a-z0-9-]+", document["id"]) or document["format"] not in {"epub", "pdf"}:
            raise ValueError("Invalid fixture document")
    for case in cases["queries"]:
        if type(case["gate"]) is not bool or type(case["expected_no_match"]) is not bool:
            raise ValueError("Expected explicit gate and no-match labels")
        if case["expected_no_match"] != (not case["relevant"]):
            raise ValueError("Inconsistent no-match label")
        if case.get("book_filter") not in {None, *documents}:
            raise ValueError("Unknown book filter")
        seen = set()
        for reference in case["relevant"]:
            document = documents[reference["document"]]
            section = reference["section"]
            if type(section) is not int or not 1 <= section <= len(document["sections"]):
                raise ValueError("Invalid golden section")
            quote = reference["quote"]
            if not quote or document["sections"][section - 1].count(quote) != 1:
                raise ValueError("Golden quote must occur exactly once in its authored section")
            key = (reference["document"], section, quote)
            if key in seen or case.get("book_filter", reference["document"]) != reference["document"]:
                raise ValueError("Duplicate or out-of-filter golden")
            seen.add(key)
    return cases


def background_documents(seed, count):
    generator = random.Random(seed)
    for number in range(count):
        label = f"background-{number:03}"
        sections = []
        for section in range(BACKGROUND_SECTIONS):
            words = [f"document{number:03}", f"section{section:02}"]
            words.extend(generator.choices(VOCABULARY, k=WORDS_PER_SECTION - len(words)))
            sections.append(" ".join(words))
        yield {"id": label, "format": "epub", "title": f"Synthetic Archive Volume {number:03}", "sections": sections}


def build_corpus(root: Path, cases, background_count=BACKGROUND_BOOKS):
    if not 0 <= background_count <= BACKGROUND_BOOKS:
        raise ValueError("Background size outside the reviewed bound")
    index_path = root / "quality.sqlite"
    index = LocalIndex(index_path, create=True)
    oracle, by_label, manifest = {}, {}, []
    started = time.perf_counter()
    try:
        documents = [*cases["documents"], *background_documents(cases["seed"], background_count)]
        for document in documents:
            source = root / f'{document["id"]}.{document["format"]}'
            (write_pdf if document["format"] == "pdf" else write_epub)(source, document)
            content = extract(source)
            if [chapter.content.strip() for chapter in content.chapters] != document["sections"]:
                raise ValueError("Extracted fixture differs from independently authored section text")
            receipt = index.ingest(source)
            if receipt["book_id"] != digest(source):
                raise ValueError("Source identity mismatch")
            book_id = receipt["book_id"]
            by_label[document["id"]] = book_id
            sections = {chapter.number: chapter.content for chapter in content.chapters}
            chunks = {}
            for row in index.db.execute("SELECT chunk_id, book_id, content, payload FROM chunks WHERE book_id=?", (book_id,)):
                payload = json.loads(row["payload"])
                section = payload["chapter_number"]
                start, end = payload["metadata"]["char_start"], payload["metadata"]["char_end"]
                if (payload["id"] != row["chunk_id"] or payload["book_id"] != row["book_id"]
                        or sections[section][start:end] != row["content"]):
                    raise ValueError("Indexed chunk does not resolve to its source span")
                chunks[row["chunk_id"]] = {"section": section, "char_start": start, "char_end": end}
            oracle[book_id] = {
                "label": document["id"], "format": document["format"], "title": document["title"],
                "sections": sections, "chunks": chunks,
            }
            manifest.append({"document": document["id"], "format": document["format"],
                             "source_sha256": book_id, "source_bytes": source.stat().st_size,
                             "sections": len(content.chapters), "words": content.word_count,
                             "chunks": receipt["chunks"]})
    finally:
        index.close()
    return index_path, oracle, by_label, {
        "seed": cases["seed"], "background_books": background_count,
        "background_sections_per_book": BACKGROUND_SECTIONS, "background_words_per_section": WORDS_PER_SECTION,
        "books": len(manifest), "sections": sum(item["sections"] for item in manifest),
        "words": sum(item["words"] for item in manifest), "chunks": sum(item["chunks"] for item in manifest),
        "source_bytes": sum(item["source_bytes"] for item in manifest), "index_bytes": index_path.stat().st_size,
        "generation_and_ingestion_seconds": time.perf_counter() - started,
        "pipeline_version": PIPELINE_VERSION, "documents": manifest,
        "corpus_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
    }


def resolve_goldens(case, oracle, by_label):
    goldens = []
    for reference in case["relevant"]:
        book_id = by_label[reference["document"]]
        text = oracle[book_id]["sections"][reference["section"]]
        start = text.index(reference["quote"])
        goldens.append({**reference, "book_id": book_id, "char_start": start, "char_end": start + len(reference["quote"])})
    return goldens
