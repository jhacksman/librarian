"""Versioned local indexes and explicit, recoverable legacy migration."""

import os
import sqlite3
from pathlib import Path

SCHEMA_VERSION = 1
APPLICATION_ID = 0x4C494252  # LIBR
STATE_SQL = """CREATE TABLE ingest_state (
    source_path TEXT PRIMARY KEY, source_sha256 TEXT,
    pipeline_version TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('completed', 'failed')),
    attempts INTEGER NOT NULL, error TEXT, updated_at TEXT NOT NULL
)"""
BASE_SQL = [
    """CREATE TABLE books (
        book_id TEXT PRIMARY KEY, title TEXT NOT NULL, source_path TEXT NOT NULL,
        metadata TEXT NOT NULL, chunk_count INTEGER NOT NULL
    )""",
    """CREATE VIRTUAL TABLE chunks USING fts5(
        chunk_id UNINDEXED, book_id UNINDEXED, content, payload UNINDEXED,
        tokenize='unicode61'
    )""",
]


def inspect_schema(db):
    version = db.execute("PRAGMA user_version").fetchone()[0]
    application = db.execute("PRAGMA application_id").fetchone()[0]
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if not tables and version == 0 and application == 0:
        return None
    if version not in (0, SCHEMA_VERSION) or application not in (0, APPLICATION_ID):
        raise ValueError("Unsupported index version or application; index left unchanged")
    expected = {
        "books": ["book_id", "title", "source_path", "metadata", "chunk_count"],
        "chunks": ["chunk_id", "book_id", "content", "payload"],
    }
    if version == SCHEMA_VERSION:
        if application != APPLICATION_ID:
            raise ValueError("Index application ID is missing")
        expected["ingest_state"] = [
            "source_path", "source_sha256", "pipeline_version", "status", "attempts",
            "error", "updated_at",
        ]
    elif application != 0:
        raise ValueError("Invalid legacy index application ID")
    for table, columns in expected.items():
        info = list(db.execute(f"PRAGMA table_info({table})"))
        actual = [row[1] for row in info]
        if actual != columns:
            raise ValueError(f"Invalid index schema: {table}")
        if table != "chunks" and not info[0][5]:
            raise ValueError(f"Missing primary key: {table}")
    sql = db.execute("SELECT sql FROM sqlite_master WHERE name='chunks'").fetchone()[0]
    if "using fts5" not in sql.lower():
        raise ValueError("Index chunks table is not FTS5")
    allowed = set(expected) | {"chunks_data", "chunks_idx", "chunks_content", "chunks_docsize", "chunks_config"}
    if tables - allowed:
        raise ValueError("Unexpected tables in index; refusing to modify it")
    return version


def initialize(db):
    version = inspect_schema(db)
    if version == 0:
        raise ValueError("Legacy index: run migrate with an explicit backup destination first")
    if version is None:
        with db:
            db.execute("BEGIN IMMEDIATE")
            for statement in [*BASE_SQL, STATE_SQL]:
                db.execute(statement)
            db.execute(f"PRAGMA application_id={APPLICATION_ID}")
            db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")


def verify(db):
    version = inspect_schema(db)
    if version is None:
        raise ValueError("Empty file is not an index")
    if [row[0] for row in db.execute("PRAGMA integrity_check")] != ["ok"]:
        raise ValueError("Index integrity check failed")
    inconsistent = db.execute("""
        SELECT b.book_id FROM books b
        WHERE b.chunk_count != (SELECT count(*) FROM chunks c WHERE c.book_id=b.book_id)
        UNION ALL
        SELECT c.book_id FROM chunks c LEFT JOIN books b ON c.book_id=b.book_id
        WHERE b.book_id IS NULL LIMIT 1
    """).fetchone()
    if inconsistent:
        raise ValueError("Book/chunk counts are inconsistent")
    return version


def backup(db, destination: Path):
    """Create a verified SQLite snapshot; never overwrite any existing path."""
    if db.in_transaction:
        raise ValueError("Commit or roll back pending writes before backup")
    verify(db)
    destination = destination.expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    try:
        with sqlite3.connect(destination) as copy:
            db.backup(copy)
            verify(copy)
    except BaseException:
        if "copy" in locals():
            copy.close()
        destination.unlink(missing_ok=True)
        raise
    finally:
        if "copy" in locals():
            copy.close()
    return str(destination)


def migrate(path: Path, backup_path: Path):
    """Only upgrade the known version-zero pilot, after saving a valid snapshot."""
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=rw", uri=True)
    try:
        version = verify(db)
        if version == SCHEMA_VERSION:
            return {"schema_version": version, "status": "already_current"}
        saved = backup(db, backup_path)
        with db:
            db.execute("BEGIN IMMEDIATE")
            if inspect_schema(db) != 0:
                raise ValueError("Index changed during migration; retry after stopping writers")
            db.execute(STATE_SQL)
            db.execute(f"PRAGMA application_id={APPLICATION_ID}")
            db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        return {"schema_version": SCHEMA_VERSION, "status": "migrated", "backup": saved}
    finally:
        db.close()


def restore(snapshot: Path, destination: Path):
    db = sqlite3.connect(snapshot.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        return {"restored_to": backup(db, destination), "schema_version": inspect_schema(db)}
    finally:
        db.close()
