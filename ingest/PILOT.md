# Offline ingestion and cited search pilot

This is a small, working text-only path through extraction, metadata, chunks,
and retrieval. It reuses `ExtractedContent`, `ChapterInfo`, `TextChunk`, and
`SearchResult` from `ingest.models`. It has no network client, LLM, model download,
OCR, or background service. Retrieval is SQLite FTS5/BM25 lexical search, not
semantic embeddings or generated answers.

The legacy `ingest.pipeline` / `librarian-ingest` commands are a separate,
unfinished path. Configuration fields and models disagree across the chunker,
embedding generator, Qdrant storage, and committee. This pilot does not certify
or repair that stack. Package imports are now lazy so using the local path does
not require the legacy stack's dependencies.

## Setup

From the repository root, use Python 3.10+ with SQLite FTS5. Python 3.12 was
verified. Keep the virtual environment on a local disk if the NAS refuses package
copies. The `/tmp` environment used below is disposable and may need recreation
after a reboot.

```sh
uv venv --python 3.12 /tmp/librarian-pilot-py312
uv pip install --offline --link-mode copy \
  --python /tmp/librarian-pilot-py312/bin/python \
  -r ingest/requirements-pilot-dev.txt
export PYTHONPATH="$PWD/ingest/src"
PILOT_PYTHON=/tmp/librarian-pilot-py312/bin/python
export LIBRARIAN_DATA_DIR="$HOME/Library/Application Support/Librarian/m6miniprojects"
PILOT_INDEX="$LIBRARIAN_DATA_DIR/library.sqlite"
```

`--offline` requires cached packages. If any are missing, installation needs
network access; do not silently remove that flag in a restricted environment.
Runtime requirements are `pydantic` and `pypdf`; pytest and Ruff are only for
development. The disposable Python runtime is separate from the durable index;
clearing `/tmp` does not remove the index or NAS snapshot. No full `pip install -e .` is necessary for the pilot.

## Reproduce the two-book pilot

Keep the original ebook files and index outside Git. Only explicit filenames
are accepted; one invocation is limited to five files, with no recursive batch
processing. These commands read the two originals in place:

```sh
"$PILOT_PYTHON" -m ingest.local --index "$PILOT_INDEX" ingest \
  '../ebooks/Humble Bundle 2025/Ultimate AI - Algorithms and LLMs by Manning/Regular_Expression_Puzzles_and_AI_Coding.epub' \
  '../ebooks/Humble Bundle 2025/Coding Challenges and Interview Prep Book Bundle - Mammoth Club/Learn-Cybersecurity-Fundamentals(3).pdf'

"$PILOT_PYTHON" -m ingest.local --index "$PILOT_INDEX" list
"$PILOT_PYTHON" -m ingest.local --index "$PILOT_INDEX" search 'regular expressions' --limit 2
"$PILOT_PYTHON" -m ingest.local --index "$PILOT_INDEX" search 'cybersecurity' --limit 2
```

Commands emit JSON; errors go to stderr and return a nonzero exit code. Search
opens an existing database read-only and never silently creates an empty index.
Ordinary query words use lexical matching. Supported technical tokens such as
`C++`, `--force`, and `pkg::name` must match as case-sensitive whole literals;
see [QUERY-POLICY.md](QUERY-POLICY.md) for supported forms and boundaries. All
query terms and literal constraints must match within one chunk. Raw FTS operators
are not interpreted, and no matches return `[]`. BM25 scores rank matches and
are not confidence values.

## Provenance and repeatability

- EPUBs follow the package spine, excluding navigation and non-linear entries.
  Each citation includes the archive member and spine section. This is not a
  printed page number. HTML is parsed as text, not executed. Archive members are
  read in memory without filesystem extraction. The pilot limits expanded EPUB
  size to 128 MiB and individual members to 16 MiB.
- PDFs are extracted page by page. Citations use physical, one-based PDF page
  numbers, which can differ from printed page labels. Pages producing no
  extractable text are recorded in the index. Completed and resumed ingestion
  return their count and a preview of up to 20 page numbers in
  `extraction_coverage`; top-level `pages_without_text` is the same preview.
  Unavailable reports use null, including EPUBs. These pages may be blank or
  image-only; zero recorded omissions never establishes complete extraction.
  Scanned books with no text and encrypted PDFs fail
  explicitly; no OCR or DRM removal is attempted.
- Chunks contain up to 400 whitespace-delimited words with 80-word overlap and
  never cross source sections/pages. Short sections are retained. Character and
  word offsets locate each exact excerpt within extracted section text. Token
  counts are character-based estimates, not model tokenizer counts.
- A book ID is its full SHA-256; chunk IDs combine that hash and chunk index.
  Reingesting identical content replaces its rows in one transaction, without
  duplicates. A failed replacement rolls back. Each book commits independently;
  if a later input fails, earlier successful books remain. Changed file bytes
  create a new book version. Identical bytes at another path update the stored
  citation path to the most recently ingested copy.
- The database stores embedded bibliography, full chunk text, source paths,
  hashes, offsets, and citation locations. It contains private corpus content.
  Do not publish it or search reports. The sibling `pilot-output/` is the
  intended directory for closed snapshots and reports; ignore rules also protect ebook and SQLite files.

## Persistent local data configuration

An example persistent macOS index location is:

`~/Library/Application Support/Librarian/library.sqlite`

This location is independent of the NAS ebook directory and survives `/tmp`
cleanup. Set `LIBRARIAN_DATA_DIR` as above, or pass `--data-dir` before the command.
`--index` selects an exact database file instead. The two flags are mutually
exclusive. Precedence is an explicit flag, `LIBRARIAN_DATA_DIR`, then the platform's
user application-data directory (`~/Library/Application Support/Librarian` on
macOS, `%LOCALAPPDATA%/Librarian` on Windows, and `$XDG_DATA_HOME/librarian` or
`~/.local/share/librarian` elsewhere). Search/list never create a missing index.

```sh
"$PILOT_PYTHON" -m ingest.local --data-dir "$LIBRARIAN_DATA_DIR" list
"$PILOT_PYTHON" -m ingest.local --data-dir "$LIBRARIAN_DATA_DIR" search 'cybersecurity'
```

This task's default sandbox write roots include the NAS project and temporary
folders, but not user Application Support. Migration and write verification used
approved execution scoped to this data directory. Read-only `list` was also
verified under the ordinary sandbox. Future ingestion needs write permission for
this directory or approved execution; do not work around a denial by moving the
live index back into `/tmp`.

## Preserve and restore the index

The original verified NAS snapshot remains untouched at
`../pilot-output/library.snapshot.sqlite`. Once all CLI writers have exited,
create a new named snapshot rather than overwriting that recovery point:

```sh
mkdir -p ../pilot-output
# Use a fresh name for each subsequent snapshot.
test ! -e ../pilot-output/library-next.snapshot.sqlite && \
  cp "$PILOT_INDEX" ../pilot-output/library-next.snapshot.sqlite
```

To restore a missing live index, with no writer running:

```sh
mkdir -p "$LIBRARIAN_DATA_DIR"
test ! -e "$PILOT_INDEX" && \
  cp ../pilot-output/library.snapshot.sqlite "$PILOT_INDEX"
"$PILOT_PYTHON" -m ingest.local --index "$PILOT_INDEX" list
"$PILOT_PYTHON" -m ingest.local --index "$PILOT_INDEX" search 'regular expressions'
```

An existing index is deliberately not overwritten by those commands. To inspect
an older snapshot alongside the live index, copy it into a separate persistent
local directory and use `--data-dir` for that directory. Do not copy an active
database while another process is writing. The pilot uses SQLite's default
rollback journal, not WAL. Snapshots and search reports contain private book text
and stay outside the repository.

## Observed NAS permission failure

The initial sandboxed connection to an empty database on this NAS failed with
`sqlite3.OperationalError: access permission denied`, SQLite error code **3**,
**SQLITE_PERM**. A disposable probe confirmed that ordinary file writes worked,
but opening/querying the empty SQLite database failed before schema creation.
The identical probe under approved execution succeeded on the NAS: DELETE
journal mode, table creation, insertion, commit, and `PRAGMA integrity_check = ok`.
The local temporary-directory control succeeded in both modes. SQLite version
was 3.53.1.

Evidence is in the private sibling reports
`pilot-output/sqlite-diagnostic-sandboxed.json` and
`pilot-output/sqlite-diagnostic-approved.json`; the reproducible disposable probe
is `scratch/probe_sqlite.py`. This isolates the observed difference to execution
permissions. It does **not** establish that this NAS cannot host SQLite, nor does
it test concurrent network writers, disconnect recovery, or crash durability.
The persistent user-local index is the selected deployment for this pilot.

## Tests

```sh
PYTHONPATH=ingest/src "$PILOT_PYTHON" -m pytest \
  -q -c /dev/null -p no:cacheprovider ingest/tests/test_local_pilot.py
/tmp/librarian-pilot-py312/bin/ruff check \
  ingest/src/ingest/local ingest/src/ingest/__init__.py ingest/tests/test_local_pilot.py
```

Fixtures are generated in temporary directories, without copying licensed books
into Git. Coverage includes reading order, short sections, page citations,
chunk coverage/overlap/offsets, persistence, idempotence, transaction rollback,
source preservation, unsupported input, no-match behavior, minimal imports,
the CLI's sample limit, persistent-path precedence, fresh-process reopening,
and snapshot restoration. The latest run passed 17 tests. Existing shared models emit Pydantic deprecation
warnings; those do not prevent the pilot from running.

## Next milestone and limits

Validate citation quality on a wider deliberately chosen sample before scaling
to the collection. Tables, mathematical layout, images, and multi-column PDF
reading order are not validated here. EPUB decoding currently assumes UTF-8.
The implementation is a bounded pilot, not a hardened hostile-document service.
Use one writer at a time. See the scoped permission diagnosis above; NAS
concurrency and disconnect behavior remain untested.

For semantic retrieval, first reconcile the legacy contracts, then provision
`sentence-transformers`, its compatible Torch dependencies, `qdrant-client`,
and approved local `BAAI/bge-base-en-v1.5` embedding weights. Start with
Qdrant's embedded local mode under a persistent data directory; this requires
no server or container. Choose compatible package versions and approve any
package/weight downloads before installation. Verify persistent local Qdrant, valid
point IDs, dimension checks, idempotent updates, and citation round trips before
running any batch. No embedding weights or Qdrant service were installed or
downloaded for this pilot.
