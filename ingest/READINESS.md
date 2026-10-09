# Local ingestion readiness (2026-10-03)

This branch preserves the existing offline pilot as a separate initial commit.
The original dirty `repo/` checkout, NAS corpus, and private durable index have
not been modified by readiness work. The changes below apply in this worktree.
No full-corpus ingestion or model execution is part of this milestone.

## Workflow

Use the existing minimal requirements, not the unfinished legacy LLM pipeline.
Select a persistent directory independently of the corpus:

```sh
export PYTHONPATH="$PWD/ingest/src"
export LIBRARIAN_DATA_DIR=/your/persistent/local/librarian-data
python -m ingest.local ingest /explicit/book.epub /explicit/other.pdf
python -m ingest.local receipts
python -m ingest.local search 'regular expressions'
```

Ingestion still accepts at most five explicit files. It returns a JSON outcome
for every input, continues past individual malformed documents, and exits 1 if
any file failed. Successful books commit independently. A second invocation
skips files with the same SHA-256, pipeline version, completed receipt, source
path, and expected chunk count. Failed files retry. `ingest --force` reparses.
A crash before a book transaction commits leaves no success receipt, so rerunning
retries it. Resume state lives in SQLite, not a separate checkpoint file.
The receipt counts parse attempts; skipping does not increment it.

Extraction reads a bounded private temporary snapshot (64 MiB maximum input),
and the cited SHA-256 covers those exact parsed bytes. Original filenames remain
in citations; temporary filenames never enter the index. Each chunk records the
pipeline version, source hash, EPUB member or PDF page, and character offsets.
If the source changes afterward, the cited hash identifies the indexed version;
a citation is not a claim that the current on-disk file still has those bytes.
Changed bytes create a separate book version; old versions are retained.
Identical bytes are one book with the last ingested source path, not two books.
No source file is rewritten or executed.

Failure isolation covers Python parser exceptions and transaction rollback. It
does not sandbox parser native code or enforce per-document CPU time. The sample
limit and input size bound are not a general hostile-document service guarantee.

## Schema and recovery

New indexes use application ID `LIBR` and user_version 1. Unknown versions,
foreign tables, and malformed table layouts fail before modification. Version-zero
pilot indexes remain readable, but writes require explicit migration and backup:

```sh
python -m ingest.local --index /local/legacy.sqlite migrate --backup /local/legacy-before-upgrade.sqlite
python -m ingest.local --index /local/live.sqlite backup /local/new-snapshot.sqlite
python -m ingest.local --index /local/restored.sqlite restore /local/new-snapshot.sqlite
```

Destinations must not exist. Migration never upgrades the real private index
implicitly. Backup uses SQLite's online backup API and verifies integrity,
schema, and book/chunk counts; restore verifies the source and new destination.
Use one writer and stop writers before migration. Keep snapshots on local disk
first, then copy the closed snapshot to NAS. Keep existing verified snapshots.
The earlier NAS permission diagnosis is unchanged; this work does not investigate
locks or change storage permissions.

## Spark-only validation

All tests, lint, dependency installation and evaluation for this readiness
milestone execute through the shared Spark coordinator. Nothing runs on M6 or
GitHub Actions. The coordinator pins an ARM64 Python 3.12 image; source is an exact
Git archive, with no private corpus/index, credentials or submodules. Dependencies
are the existing pinned `requirements-pilot-ci.txt` from PyPI, wheel-only. This locks the
transitive versions observed in the first Spark run as well as direct dependencies.
No repository setup/build scripts, editable installation, Torch or embeddings.

Reviewed container command (coordinator controls CPU, RAM, network and timeout):

```sh
python -m venv /tmp/librarian-ci-venv
/tmp/librarian-ci-venv/bin/python -m pip install --only-binary=:all: -r ingest/requirements-pilot-ci.txt
export PYTHONPATH="$PWD/ingest/src"
/tmp/librarian-ci-venv/bin/python -m pytest -q -c /dev/null -p no:cacheprovider --junitxml=ci-output/tests.xml ingest/tests
/tmp/librarian-ci-venv/bin/ruff check ingest/src/ingest/local ingest/src/ingest/__init__.py ingest/tests ingest/evaluation
/tmp/librarian-ci-venv/bin/python ingest/evaluation/evaluate.py --output ci-output/retrieval.json
```

The generated EPUB/PDF fixtures and evaluation sentences are original synthetic
material, not licensed ebook extracts or purported generated replacements for
original books. The retrieval gate reports recall@3, MRR@3, no-match abstention,
and verifies excerpt offsets against source sections. Its six positive queries
and one negative query are deterministic regression coverage, not a representative
quality claim for 491 books or semantic retrieval. No embedding model is needed.

Current limits: no OCR, no semantic retrieval, no multiwriter certification,
no destructive version cleanup, no automatically processed private corpus.
The legacy service pipeline remains unfinished. The real private schema migration
and wider quality assessment require a separate deliberate operation.

## Recorded PDF omissions (C1)

The bounded C1 change exposes already stored PDF page omissions through ingestion,
resume, catalog, status, search and direct passage responses. Search warnings cover
the requested indexed-book scope even on no-match responses. Unknown metadata is
never converted into a zero-omission claim. The contract is frozen in
[COVERAGE-POLICY.md](COVERAGE-POLICY.md); no parser, schema, query grammar or citation
offset changes are part of it.

The independent synthetic test module covers mixed text/blank PDFs, recorded-zero
and unknown reports, metadata and page-preview bounds, resume, source removal,
unchanged read-only database bytes, explicit parser failures, scoped abstention,
CLI/HTML warning visibility and real MCP auto/legacy transport parity. The test
command below belongs in the coordinator's reviewed Spark runtime:

```sh
PYTHONPATH=ingest/src /path/to/reviewed/python -m pytest \
  -q -c /dev/null -p no:cacheprovider ingest/tests/test_coverage.py
```

C1 is independently verified and closed within this scope. Spark job
`librarian-20261004-coverage-001` passed at
`b20a126c49b03a61896b419e47955237c633815b`: 204 full-profile tests, 192 minimal-profile
tests with 12 expected optional-SDK skips, scoped Ruff and the original seven-query
retrieval gate. The new C1 module contributes 54 full-profile passes and 52
minimal-profile passes plus two SDK skips. The existing 35-wheel and 12-wheel
hash-locked dependency sets are unchanged. Exact-source artifacts, runtime limits
and cleanup were independently verified and are recorded in the
[C1 results](evaluation/COVERAGE-RESULTS.md). No large PERF1 benchmark was rerun
and no private-library files were processed.

Human-to-Librarian gaps remain: lexical search can miss paraphrases; extraction
does not understand image-only pages, complex layout or completeness; generated
synthesis and semantic retrieval remain unimplemented in the working pilot.
Actual client registration, private-library quality checks, LAN access and
persistent hosting remain separate work. The C1 change makes missing indexed
text visible; it cannot recover or answer from that text.

The next decisions are which local model/runtime to evaluate, which small private
sample to validate deliberately, and which trusted client or LAN deployment to
support. Existing configuration names are historical baselines rather than an
approved model choice. None of these decisions is needed to use the verified
lexical evidence interface in an already reviewed runtime.

## Scoped browsing (U1)

Readers can browse the indexed catalog, choose a book by its full ID and preserve
their question and scope through searches, no-match responses and excerpt
continuation. Agents can use the optional exact `book_id` filter on `list_books`;
its result envelope and the four MCP tool names are unchanged.

Spark job `librarian-20261004-usability-001` passed at application/test commit
`3754a4f619641b942ee8104a257a138b7f855292`: 257 full-profile tests, 243 minimal-profile
tests with 14 expected SDK skips, scoped Ruff and the original seven-query gate.
Separate independent verification closed this bounded result with no blockers.
The twelve-scenario synthetic exercise has eleven passing gates and one retained
paraphrase miss. See [USABILITY-RESULTS.md](evaluation/USABILITY-RESULTS.md) for
source identity, HTTP/MCP evidence, reproducible checks and limits. A visual
walkthrough is separate from this actual-HTTP and parsed-HTML verification.

## Publishing

This repository currently has no tracked `.github/workflows` files in the reviewed
commit. Do not push yet: parent must confirm the CI-safe draft PR procedure and
whether remote workflows changed. No GitHub-hosted job was started by this work.
Exact tested commit, Spark job IDs, commands and results are recorded after the
coordinator returns its artifacts; a pending queue submission is not a test pass.
