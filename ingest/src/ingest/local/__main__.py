"""Run with PYTHONPATH=ingest/src python -m ingest.local."""

import argparse
import json
import sys
from pathlib import Path

from ingest.local.index import LocalIndex
from ingest.local.paths import index_path
from ingest.local.schema import backup, migrate, restore
from ingest.local.service import LibraryService, SQLiteRetriever, render_answer


def main():
    parser = argparse.ArgumentParser(description="Offline EPUB/PDF pilot: cited lexical search")
    storage = parser.add_mutually_exclusive_group()
    storage.add_argument("--index", type=Path, help="Explicit private SQLite index path")
    storage.add_argument(
        "--data-dir",
        type=Path,
        help="Persistent local data directory (default: LIBRARIAN_DATA_DIR or user application data)",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest", help="Ingest 1-5 explicit files; no directory scanning")
    ingest.add_argument("files", type=Path, nargs="+")
    ingest.add_argument("--force", action="store_true", help="Reparse even when a current receipt exists")
    search = commands.add_parser("search", help="Literal word search; all words must match a chunk")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=5)
    commands.add_parser("list", help="List indexed books")
    ask = commands.add_parser("ask", help="Ask for cited source evidence; no generated answer")
    ask.add_argument("question")
    ask.add_argument("--limit", type=int, default=5)
    ask.add_argument("--book-id")
    ask.add_argument("--json", action="store_true")
    commands.add_parser("receipts", help="Show durable per-file completion/failure receipts")
    backup_command = commands.add_parser("backup", help="Create a verified snapshot without overwriting")
    backup_command.add_argument("destination", type=Path)
    restore_command = commands.add_parser("restore", help="Restore a verified snapshot to a new index")
    restore_command.add_argument("snapshot", type=Path)
    migrate_command = commands.add_parser("migrate", help="Explicitly upgrade the legacy pilot schema")
    migrate_command.add_argument("--backup", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "ingest" and not 1 <= len(args.files) <= 5:
        parser.error("The pilot accepts at most five explicit files per invocation")
    index = None
    try:
        path = index_path(args.index, args.data_dir)
        if args.command == "ask":
            answer = LibraryService(SQLiteRetriever(path)).ask(args.question, args.limit, args.book_id)
            print(json.dumps(answer, ensure_ascii=False, indent=2) if args.json else render_answer(answer))
            return 0
        if args.command == "restore":
            output = restore(args.snapshot, path)
        elif args.command == "migrate":
            output = migrate(path, args.backup)
        else:
            index = LocalIndex(path, create=args.command == "ingest")
            if args.command == "ingest":
                output = index.ingest_many(args.files, resume=not args.force)
            elif args.command == "list":
                output = index.books()
            elif args.command == "receipts":
                output = index.receipts()
            elif args.command == "backup":
                output = {"snapshot": backup(index.db, args.destination)}
            else:
                output = [item.model_dump() for item in index.search(args.query, args.limit)]
        print(json.dumps(output, ensure_ascii=False, indent=2))
        if args.command == "ingest" and any(item["status"] == "failed" for item in output):
            print("Pilot error: one or more files failed; inspect JSON results/receipts", file=sys.stderr)
            return 1
        return 0
    except Exception as error:  # CLI boundary: malformed books produce a nonzero exit.
        print(f"Pilot error: {error}", file=sys.stderr)
        return 1
    finally:
        if index is not None:
            index.close()


if __name__ == "__main__":
    sys.exit(main())
