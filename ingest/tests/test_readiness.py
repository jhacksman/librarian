"""Synthetic fixtures only; run on the Spark CI queue, never on the M6."""

import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from test_local_pilot import epub

from ingest.local.index import LocalIndex
from ingest.local.schema import BASE_SQL, backup, inspect_schema, migrate, restore, verify


def test_resume_survives_restart_and_isolates_malformed_file(tmp_path, monkeypatch):
    good = epub(tmp_path / "good.epub")
    bad = tmp_path / "bad.epub"
    bad.write_bytes(b"broken zip")
    later = tmp_path / "later.epub"
    shutil.copyfile(good, later)
    path = tmp_path / "index.sqlite"
    index = LocalIndex(path, create=True)
    results = index.ingest_many([good, bad, later])
    assert [item["status"] for item in results] == ["completed", "failed", "completed"]
    assert [item["status"] for item in index.receipts()] == ["failed", "completed", "completed"]
    index.close()
    index = LocalIndex(path, create=True)
    try:
        # Same-content files retain one book, with the most recently ingested source path.
        result = index.ingest(later, resume=True)
        assert result["status"] == "skipped"
        import ingest.local.index as module
        monkeypatch.setattr(module, "extract", lambda path: pytest.fail("Resume reparsed input"))
        assert index.ingest(later, resume=True)["status"] == "skipped"
        assert len(index.books()) == 1
    finally:
        index.close()


def test_failed_file_is_retried_after_repair(tmp_path):
    source = tmp_path / "book.epub"
    source.write_bytes(b"invalid")
    index = LocalIndex(tmp_path / "index.sqlite", create=True)
    try:
        assert index.ingest_many([source])[0]["status"] == "failed"
        epub(source)
        assert index.ingest_many([source])[0]["status"] == "completed"
        assert index.receipts()[0]["attempts"] == 2
        assert index.receipts()[0]["error"] is None
    finally:
        index.close()


def test_parser_reads_cited_snapshot_even_if_original_changes(tmp_path, monkeypatch):
    import ingest.local.index as module

    source = epub(tmp_path / "source.epub")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    extractor = module.extract
    def modify_original(snapshot):
        assert snapshot != source
        source.write_bytes(b"changed after snapshot")
        return extractor(snapshot)
    monkeypatch.setattr(module, "extract", modify_original)
    index = LocalIndex(tmp_path / "index.sqlite", create=True)
    try:
        assert index.ingest(source)["book_id"] == digest
        hit = index.search("regular")[0]
        assert hit.metadata["source_sha256"] == digest
        assert hit.metadata["source_path"] == str(source)
    finally:
        index.close()


def test_legacy_requires_explicit_migration_and_preserves_backup(tmp_path):
    path = tmp_path / "legacy.sqlite"
    db = sqlite3.connect(path)
    for statement in BASE_SQL:
        db.execute(statement)
    db.commit()
    db.close()
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Legacy index"):
        LocalIndex(path, create=True)
    assert path.read_bytes() == before
    saved = tmp_path / "legacy-backup.sqlite"
    assert migrate(path, saved)["status"] == "migrated"
    index = LocalIndex(path, create=True)
    try:
        assert inspect_schema(index.db) == 1
        index.ingest(epub(tmp_path / "book.epub"))
    finally:
        index.close()
    copy = LocalIndex(saved)
    try:
        assert inspect_schema(copy.db) == 0
        assert copy.books() == []
    finally:
        copy.close()
    assert migrate(path, tmp_path / "unneeded.sqlite")["status"] == "already_current"
    assert not (tmp_path / "unneeded.sqlite").exists()


def test_unknown_database_unchanged_and_future_version_rejected(tmp_path):
    path = tmp_path / "foreign.sqlite"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE important(data TEXT)")
    db.commit()
    db.close()
    before = path.read_bytes()
    with pytest.raises(ValueError, match="schema"):
        LocalIndex(path, create=True)
    assert before == path.read_bytes()
    future = tmp_path / "future.sqlite"
    index = LocalIndex(future, create=True)
    index.db.execute("PRAGMA user_version=42")
    index.close()
    with pytest.raises(ValueError, match="Unsupported"):
        LocalIndex(future)


def test_backup_restore_no_clobber_and_integrity(tmp_path):
    index = LocalIndex(tmp_path / "live.sqlite", create=True)
    try:
        index.ingest(epub(tmp_path / "book.epub"))
        snapshot = tmp_path / "backup.sqlite"
        backup(index.db, snapshot)
        before = snapshot.read_bytes()
        with pytest.raises(FileExistsError):
            backup(index.db, snapshot)
        assert snapshot.read_bytes() == before
        destination = tmp_path / "restored.sqlite"
        restore(snapshot, destination)
        with pytest.raises(FileExistsError):
            restore(snapshot, destination)
        restored = LocalIndex(destination)
        try:
            assert verify(restored.db) == 1
            assert restored.search("regular")[0].book_id == index.search("regular")[0].book_id
            assert restored.receipts() == index.receipts()
        finally:
            restored.close()
    finally:
        index.close()


def test_inconsistent_index_backup_refused(tmp_path):
    index = LocalIndex(tmp_path / "live.sqlite", create=True)
    try:
        index.ingest(epub(tmp_path / "book.epub"))
        with index.db:
            index.db.execute("DELETE FROM chunks")
        with pytest.raises(ValueError, match="inconsistent"):
            backup(index.db, tmp_path / "bad-snapshot.sqlite")
        assert not (tmp_path / "bad-snapshot.sqlite").exists()
        # A completed receipt must not hide missing chunks.
        assert index.ingest_many([tmp_path / "book.epub"])[0]["status"] == "completed"
        assert verify(index.db) == 1
    finally:
        index.close()


def test_cli_reports_each_file_then_resumes(tmp_path):
    good = epub(tmp_path / "good.epub")
    bad = tmp_path / "bad.pdf"
    bad.touch()
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")}
    command = [sys.executable, "-m", "ingest.local", "--index", str(tmp_path / "index.sqlite")]
    first = subprocess.run([*command, "ingest", str(bad), str(good)], env=env, capture_output=True, text=True)
    assert first.returncode == 1
    assert [row["status"] for row in json.loads(first.stdout)] == ["failed", "completed"]
    second = subprocess.run([*command, "ingest", str(good)], env=env, capture_output=True, text=True)
    assert second.returncode == 0
    assert json.loads(second.stdout)[0]["status"] == "skipped"


def test_legacy_migration_preserves_existing_citations(tmp_path):
    path = tmp_path / "legacy.sqlite"
    index = LocalIndex(path, create=True)
    source = epub(tmp_path / "book.epub")
    index.ingest(source)
    original_hit = index.search("regular")[0].model_dump()
    with index.db:
        index.db.execute("DROP TABLE ingest_state")
        index.db.execute("PRAGMA application_id=0")
        index.db.execute("PRAGMA user_version=0")
    index.close()
    migrate(path, tmp_path / "before-migration.sqlite")
    upgraded = LocalIndex(path, create=True)
    try:
        assert upgraded.search("regular")[0].model_dump() == original_hit
        assert upgraded.receipts() == []
        assert upgraded.ingest(source, resume=True)["status"] == "completed"
        assert upgraded.ingest(source, resume=True)["status"] == "skipped"
    finally:
        upgraded.close()


def test_pipeline_change_invalidates_resume(tmp_path, monkeypatch):
    import ingest.local.index as module

    source = epub(tmp_path / "book.epub")
    index = LocalIndex(tmp_path / "index.sqlite", create=True)
    try:
        index.ingest(source)
        monkeypatch.setattr(module, "PIPELINE_VERSION", "next-reviewed-pipeline")
        assert index.ingest(source, resume=True)["status"] == "completed"
        assert index.receipts()[0]["attempts"] == 2
        assert index.search("regular")[0].metadata["pipeline_version"] == "next-reviewed-pipeline"
    finally:
        index.close()


def test_input_size_limit_leaves_existing_index_usable(tmp_path, monkeypatch):
    import ingest.local.index as module

    source = epub(tmp_path / "book.epub")
    index = LocalIndex(tmp_path / "index.sqlite", create=True)
    try:
        index.ingest(source)
        monkeypatch.setattr(module, "MAX_SOURCE_BYTES", 1)
        with pytest.raises(ValueError, match="file limit"):
            index.ingest(source)
        assert index.search("regular")
        assert index.receipts()[0]["status"] == "failed"
    finally:
        index.close()


def test_backup_rejects_uncommitted_writes_without_hanging(tmp_path):
    index = LocalIndex(tmp_path / "index.sqlite", create=True)
    try:
        index.db.execute("INSERT INTO books VALUES ('pending', 'Pending', 'source', '{}', 0)")
        with pytest.raises(ValueError, match="pending writes"):
            backup(index.db, tmp_path / "snapshot.sqlite")
        assert not (tmp_path / "snapshot.sqlite").exists()
        index.db.rollback()
    finally:
        index.close()
