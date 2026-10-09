import json
import os
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from ingest.local.extract import extract, member_path
from ingest.local.index import LocalIndex, chunks_for
from ingest.models import ChapterInfo, ExtractedContent


def epub(path, *, text=None):
    with ZipFile(path, "w") as archive:
        archive.writestr(
            "META-INF/container.xml",
            '<container><rootfiles><rootfile full-path="OPS/book.opf"/></rootfiles></container>',
        )
        archive.writestr(
            "OPS/book.opf",
            """<package xmlns="http://www.idpf.org/2007/opf"><metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Pilot Handbook</dc:title><dc:creator>Test Author</dc:creator></metadata><manifest><item id="two" href="two.xhtml" media-type="application/xhtml+xml"/><item id="one" href="one.xhtml" media-type="application/xhtml+xml"/></manifest><spine><itemref idref="two" linear="no"/><itemref idref="one"/><itemref idref="two"/></spine></package>""",
        )
        archive.writestr("OPS/two.xhtml", "<h1>Second</h1><p>Firewalls protect networks.</p>" if text is None else "")
        archive.writestr(
            "OPS/one.xhtml",
            "<head><title>Hidden title</title></head><h1>First</h1><p>Regular expressions match patterns.</p><script>secret script</script>" if text is None else f"<p>{text}</p>",
        )
    return path


def pdf(path):
    writer = PdfWriter()
    for text in ["First page text", "Cybersecurity protects networks"]:
        page = writer.add_blank_page(612, 792)
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})}
        )
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 40 700 Td ({text}) Tj ET".encode())
        page[NameObject("/Contents")] = writer._add_object(stream)
    writer.add_metadata({"/Title": "PDF Handbook", "/Author": "Test Author"})
    writer.write(path)
    return path


def test_epub_spine_order_and_short_sections(tmp_path):
    content = extract(epub(tmp_path / "book.epub"))
    assert [ch.title for ch in content.chapters] == ["First", "Second"]
    assert [ch.number for ch in content.chapters] == [2, 3]
    assert "secret" not in content.raw_text and "Hidden" not in content.raw_text
    assert content.metadata["authors"] == ["Test Author"]
    assert content.metadata["section_sources"]["2"] == "OPS/one.xhtml"


def test_pdf_pages_and_citations(tmp_path):
    source = pdf(tmp_path / "book.pdf")
    content = extract(source)
    assert content.page_count == 2
    index = LocalIndex(tmp_path / "index.sqlite", create=True)
    try:
        index.ingest(source)
        result = index.search("cybersecurity")[0]
        assert result.metadata["page_start"] == 2
        assert "PDF page 2" in result.metadata["citation"]
        assert result.metadata["source_path"] == str(source)
        assert result.content in content.chapters[1].content
    finally:
        index.close()


def test_chunk_coverage_overlap_and_offsets(tmp_path):
    text = " ".join(f"word{i}" for i in range(1001))
    content = ExtractedContent(
        title="Book", chapters=[ChapterInfo(number=1, title="Chapter", content=text)]
    )
    chunks = chunks_for(content, "hash", tmp_path / "book.epub", size=400, overlap=80)
    assert len(chunks) == 3
    assert set(" ".join(ch.content for ch in chunks).split()) == set(text.split())
    assert chunks[0].content.split()[-80:] == chunks[1].content.split()[:80]
    for ch in chunks:
        assert text[ch.metadata["char_start"] : ch.metadata["char_end"]] == ch.content
        assert ch.total_chunks == 3


def test_idempotent_persistent_search_and_no_match(tmp_path):
    source = epub(tmp_path / "book.epub")
    before = source.read_bytes()
    path = tmp_path / "index.sqlite"
    index = LocalIndex(path, create=True)
    first = index.ingest(source)
    index.ingest(source)
    assert len(index.books()) == 1
    assert index.db.execute("SELECT count(*) FROM chunks").fetchone()[0] == first["chunks"]
    index.close()
    index = LocalIndex(path)
    try:
        result = index.search("regular expressions")[0]
        assert result.metadata["epub_member"] == "OPS/one.xhtml"
        assert result.book_id == first["book_id"]
        assert index.search("unfindable") == []
        assert index.search('" OR * ; --') == []
        assert index.search("") == []
        with pytest.raises(ValueError):
            index.search("regular", 0)
    finally:
        index.close()
    assert source.read_bytes() == before


def test_failed_ingest_preserves_previous_book(tmp_path):
    source = epub(tmp_path / "book.epub")
    index = LocalIndex(tmp_path / "index.sqlite", create=True)
    try:
        index.ingest(source)
        empty = tmp_path / "empty.pdf"
        empty.touch()
        with pytest.raises(ValueError, match="non-empty"):
            index.ingest(empty)
        assert len(index.books()) == 1
        assert len(index.search("regular")) == 1
    finally:
        index.close()


def test_scanned_pdf_rejected(tmp_path):
    writer = PdfWriter()
    writer.add_blank_page(612, 792)
    path = tmp_path / "scan.pdf"
    writer.write(path)
    with pytest.raises(ValueError, match="OCR"):
        extract(path)


@pytest.mark.parametrize(
    "href",
    ["../../escape.xhtml", "https://example.com/book", "/absolute.xhtml", "..%2F..%2Fescape"],
)
def test_unsafe_member_path(href):
    with pytest.raises(ValueError):
        member_path("OPS", href)


def test_missing_index_does_not_create_file(tmp_path):
    path = tmp_path / "missing.sqlite"
    with pytest.raises(ValueError, match="does not exist"):
        LocalIndex(path)
    assert not path.exists()


def test_minimal_import_and_cli(tmp_path):
    source = epub(tmp_path / "book.epub")
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")}
    script = "import ingest; from ingest import TextChunk; import sys; assert 'ingest.pipeline' not in sys.modules; assert 'openai' not in sys.modules"
    subprocess.run([sys.executable, "-c", script], check=True, env=env)
    command = [sys.executable, "-m", "ingest.local", "--index", str(tmp_path / "index.sqlite")]
    result = subprocess.run(
        [*command, "ingest", str(source)], check=True, env=env, capture_output=True, text=True
    )
    assert json.loads(result.stdout)[0]["chunks"] == 2
    result = subprocess.run(
        [*command, "search", "regular expressions"],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )
    assert "EPUB OPS/one.xhtml" in json.loads(result.stdout)[0]["metadata"]["citation"]
    result = subprocess.run([*command, "ingest", *[str(source)] * 6], env=env, capture_output=True)
    assert result.returncode == 2


def test_database_failure_rolls_back_replacement(tmp_path):
    import sqlite3

    source = epub(tmp_path / "book.epub")
    index = LocalIndex(tmp_path / "index.sqlite", create=True)
    try:
        index.ingest(source)
        index.db.execute(
            "CREATE TRIGGER fail_write BEFORE INSERT ON books BEGIN SELECT RAISE(ABORT, 'test failure'); END"
        )
        with pytest.raises(sqlite3.IntegrityError, match="test failure"):
            index.ingest(source)
        assert len(index.books()) == 1
        assert len(index.search("regular")) == 1
        assert index.db.execute("SELECT count(*) FROM chunks").fetchone()[0] == 2
    finally:
        index.close()


def test_bad_file_cli_reports_failure(tmp_path):
    path = tmp_path / "bad.epub"
    path.write_text("not a zip")
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ingest.local",
            "--index",
            str(tmp_path / "index.sqlite"),
            "ingest",
            str(path),
        ],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1
    assert "Pilot error:" in result.stderr
    assert "Traceback" not in result.stderr


def test_persistent_path_precedence(tmp_path, monkeypatch):
    from ingest.local.paths import index_path

    monkeypatch.setenv("LIBRARIAN_DATA_DIR", str(tmp_path / "environment"))
    assert index_path() == tmp_path / "environment" / "library.sqlite"
    assert index_path(data_dir=tmp_path / "explicit") == tmp_path / "explicit" / "library.sqlite"
    assert index_path(index=tmp_path / "custom.sqlite") == tmp_path / "custom.sqlite"


def test_macos_default_is_user_application_data(tmp_path, monkeypatch):
    import ingest.local.paths as paths

    monkeypatch.delenv("LIBRARIAN_DATA_DIR", raising=False)
    monkeypatch.setattr(paths.sys, "platform", "darwin")
    monkeypatch.setattr(paths.Path, "home", lambda: tmp_path)
    assert paths.index_path() == tmp_path / "Library" / "Application Support" / "Librarian" / "library.sqlite"


def test_data_directory_survives_cli_processes_and_restores(tmp_path):
    import shutil

    source = epub(tmp_path / "book.epub")
    data_dir = tmp_path / "user-data" / "Librarian"
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src"), "LIBRARIAN_DATA_DIR": str(data_dir)}
    command = [sys.executable, "-m", "ingest.local"]
    for _ in range(2):
        subprocess.run([*command, "ingest", str(source)], check=True, env=env, capture_output=True)
    listed = subprocess.run([*command, "list"], check=True, env=env, text=True, capture_output=True)
    assert len(json.loads(listed.stdout)) == 1
    assert json.loads(listed.stdout)[0]["chunk_count"] == 2
    # All writer processes exited before copying the rollback-journal database.
    snapshot = tmp_path / "snapshot.sqlite"
    shutil.copyfile(data_dir / "library.sqlite", snapshot)
    restore_dir = tmp_path / "restored-data"
    restore_dir.mkdir()
    shutil.copyfile(snapshot, restore_dir / "library.sqlite")
    result = subprocess.run([*command, "--data-dir", str(restore_dir), "search", "regular"], check=True, env=env, capture_output=True, text=True)
    assert len(json.loads(result.stdout)) == 1
    index = LocalIndex(restore_dir / "library.sqlite")
    try:
        assert index.db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert index.db.execute("SELECT count(*) FROM chunks").fetchone()[0] == 2
    finally:
        index.close()
