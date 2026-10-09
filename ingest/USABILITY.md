# Human questions and agent access: first interface slice

## Historical baseline; future choice reopened for research

The committed configuration (`config/config.example.yaml` and
`src/ingest/config.py`) names `BAAI/bge-base-en-v1.5`: 768 dimensions, normalized
embeddings, cosine distance, Qdrant collection `librarian_books`, default endpoint
`http://localhost:6333`. The ingest README targets GB10/DGX Spark-class hardware.
The website pseudocode also explicitly loads that BGE model. There is no immutable
model revision pin or later local config override in this checkout.

These files establish the historical architecture, not the selected future
model or retrieval method. The user requested a fresh research round on
2026-10-03. Compare alternatives before recommending any change; the current
configuration is a baseline, not installation or execution permission. The live
two-book pilot remains lexical. No model was downloaded or executed during this
interface work. No alternative is selected or silently substituted.

Answer-model documents disagree: ingest configuration uses
`Qwen/Qwen2.5-32B-Instruct`; README hardware notes use a 72B AWQ example; website
pseudocode says Qwen3-72B in its heading but calls Qwen2.5-72B in examples. These
are not sufficient to select or activate a live answer-model endpoint.

## Implemented human entry points

The CLI and loopback web preview use the same read-only `LibraryService`:

```sh
export PYTHONPATH="$PWD/ingest/src"
export LIBRARIAN_DATA_DIR=/your/persistent/local/librarian-data
python -m ingest.local ask 'What are regular expressions?'
python -m ingest.local ask 'What are regular expressions?' --json
python -m ingest.local.web --port 8765
```

The web preview binds only `127.0.0.1`, runs in the foreground, and stops with
Ctrl-C. It is not started automatically, installed as a service, or exposed to the
LAN. During this workstream all execution/HTTP round-trip tests run in the Spark
container, never on M6. No live private index is migrated or reingested.

The interface displays indexed coverage, original quoted excerpts, title, physical
PDF page or EPUB member, chunk ID and source hash. It renders source text as text,
not HTML; source instructions are not executed. Host filesystem paths and file
contents outside the index are not exposed as tools or download routes.

### Choose a book before asking

Open **Browse indexed books**, page through the catalog and choose a book. Each
entry shows its full book ID so books with the same title remain distinct. The
selected title, ID and extraction warning appear above the question. **All books**
explicitly restores the wider scope.

Typing a question, then browsing or selecting a book, keeps the typed text. Search
results and no-match responses retain that scope. Read-from-start and continuation
buttons preserve the form state while displaying exact cited excerpts. These
actions submit the current form; questions are not put in navigation URLs. A
removed or invalid selection is an explicit error, never an automatic wider
search. Catalog browsing itself does not run a search.

An agent can use `list_books` to discover IDs, or pass `{"book_id":"<full hash>"}`
to refresh one selected book's public metadata. Use that same `book_id` with
`ask_library`. An unknown hash produces an empty filtered catalog; omitted or
empty filters retain existing pagination over all indexed books. The catalog
still returns `books` and `next_offset`, and the server still exposes four tools.

This interface change passed independent synthetic verification; see the
[U1 results](evaluation/USABILITY-RESULTS.md) and
[USABILITY-POLICY.md](USABILITY-POLICY.md) for the frozen scope. It makes existing
book filtering accessible to humans and exact catalog lookup accessible to
agents. It does not improve ranking, infer which edition is authoritative or
produce a generated answer.

This is **evidence-only**, not generated question answering. The lexical adapter
removes a small explicit set of question/filler words from ordinary prose and
reports the resulting query. Explicit supported technical tokens retain spelling
and must match as case-sensitive whole literals. The response also reports the
actual FTS candidate expression, literal constraints and normalization version;
see [QUERY-POLICY.md](QUERY-POLICY.md) for the conservative ASCII grammar and its
Unicode boundary limits. Ordinary terms and all literal constraints must match
within the same chunk, with filtering before the final result limit. This requires
no schema migration or reindex. It returns an explicit abstention when
nothing matches, and never falls back to the web or an external API. Matching
passages do not prove the question is answered. Negation, paraphrases, ambiguous
questions, comparisons, and synthesis require stronger retrieval/answer evaluation.
The mock semantic seam receives the original question without lexical rewriting.

### Technical literal search

The readiness branch automatically rejects candidates missing a required literal
substring before applying the existing whole-literal check. This requires no new
option, migration or reindex. The bounded synthetic results and remaining limits
are recorded in [PERFORMANCE-RESULTS.md](evaluation/PERFORMANCE-RESULTS.md).

For an already reviewed runtime and an existing index, from this checkout's root:

```sh
export PYTHONPATH="$PWD/ingest/src"
PILOT_PYTHON=/path/to/reviewed/python
PILOT_INDEX=/path/to/existing/library.sqlite

"$PILOT_PYTHON" -m ingest.local --index "$PILOT_INDEX" \
  ask --limit 3 -- 'C++'
"$PILOT_PYTHON" -m ingest.local --index "$PILOT_INDEX" \
  ask --json --limit 3 -- '--force'
```

Replace the two paths with the selected runtime and index. The `--` before the
question is needed for questions starting with a hyphen, such as `--force`;
shell quoting alone does not stop option parsing. Text output includes cited
excerpts; JSON also includes normalization provenance and continuation offsets.
An already configured trusted local MCP client can use the `ask_library` tool
with `{"question":"C++","limit":3}` or
`{"question":"--force","limit":3}` directly.

The [pilot guide](PILOT.md#persistent-local-data-configuration) records the
existing private index location and earlier disposable interpreter path. These
commands are usage instructions, not a new runtime installation or private-index
validation. This workstream's execution remains Spark-only; no M6 application,
private-library search, client registration or service was started for PERF1.

### Recorded extraction coverage

Search responses now carry `extraction_coverage` for the requested indexed-book
scope, even when no passages match. Status reports the whole indexed library;
catalog entries and returned/direct passages carry a per-book report. CLI text
and the loopback preview show the corresponding warnings, with omitted page
numbers alongside each passage. The same fields pass through existing MCP tools.

When stored PDF metadata records pages with no extractable text, those pages
are not searchable. They may be blank or image-only. Reports include the full
validated count and at most 20 sorted physical page numbers. Missing, malformed,
oversized or non-PDF reports are explicitly unknown. No report claims extraction
is complete, including a PDF with zero recorded omissions. A no-match response
therefore remains an abstention about the indexed text, with its extraction limits
visible. Entirely textless documents and parser exceptions still fail ingestion
explicitly; there is no OCR or partial-page recovery.

These fields use existing index metadata without opening source books, migrating
the schema or reindexing. The metadata limit is 65,536 characters per book; summary
reads stream through the selected book rows. That bounds per-book decoding, not
total library scan cost. Earlier PERF1 timings remain attributed to their original
source and do not measure this added reporting work. See
[COVERAGE-POLICY.md](COVERAGE-POLICY.md) for precise unknown, scope and adapter rules,
and [READINESS.md](READINESS.md#recorded-pdf-omissions-c1) for verification status.

## Shared agent interface and stdio transport

The optional `mcp_server` entry point uses the pinned official MCP Python SDK
2.3.0 over **stdio only**. It requires an explicit existing index, runs in the
foreground, and has no HTTP option, install command or listener. Model selection
is independent of this transport. After the coordinator approves the exact
wheel-only dependencies, a local client launches:

```sh
PYTHONPATH=ingest/src /path/to/reviewed/python -m ingest.local.mcp_server --index /path/to/index.sqlite
```

Stdout belongs exclusively to MCP protocol messages. Startup fails without
creating an index when the index or exact SDK version is unavailable. This is
for trusted local callers; it provides no user/book authorization boundary.
Connecting a cloud model would send retrieved excerpts outside the local system
and is not approved for the private library.

`register_tools` supplies four read-only tools backed by the same `LibraryService`
as the human CLI and loopback UI. They declare structured JSON results and strict
input schemas:

| Tool | Bounded input | Result |
| --- | --- | --- |
| `ask_library` | Question 1–2000 characters, at most 128 search terms, integer limit 1–10, 64-character book hash or omit/empty string | Evidence-only response, explicit no-match abstention, original question/query, cited excerpts, scoped extraction coverage |
| `get_passage` | Chunk ID: 64 lowercase hex characters, colon, 1–10 digits; integer character offset 0–2,147,483,647 | Bounded original excerpt with continuation and per-book extraction coverage, no filesystem path |
| `list_books` | Integer limit 1–50, offset 0–1,000,000; optional exact book hash using the same filter rules as `ask_library` | `books` with per-book extraction coverage plus `next_offset`; stable order by book ID for an unchanged index |
| `library_status` | None | Actual backend, indexed book/chunk counts, aggregate extraction coverage, read-only/evidence-only state |

Booleans and stringified numbers are not integer limits. MCP book filters use a
plain strict string: omit the parameter or use `""` for no filter. Both JSON null
and the string `"null"` are rejected, avoiding SDK pre-parsing of nullable fields. Excerpts contain at most
8,000 characters, labels 500 characters, and EPUB locators 2,048 characters.
Search excerpts prioritize a matching technical literal when one is required,
otherwise a matching ordinary term. `get_passage`
accepts the returned `excerpt_offset` to reproduce that excerpt, or starts at zero
and follows `next_offset` to read successive bounded slices. The human preview
provides equivalent read-from-start/continue buttons. Tokenizer normalization can
differ from literal matching; continuation preserves access to the full chunk.
Shortened excerpts explicitly report `excerpt_truncated`, original end offset,
and adjusted `char_end` so the displayed quote still equals the exact source
slice. `offset_basis` identifies Unicode codepoints in the extracted section,
and `pipeline_version` identifies its extraction/chunking policy. These bounds
are per-field character limits, not a claimed 128 KiB wire-message cap. Pages are physical PDF pages; EPUB members are reflowable locators.
There are no ingest, deletion, migration, file-open, SQL, shell or network-fetch
tools. Tool annotations describe these constraints; read-only SQLite connections
and service validation enforce them.

A no-match abstention is distinct from an unavailable/corrupt index. Expected
input/not-found errors remain actionable; internal exceptions become a short
unavailable error without source paths or SQL. The web preview likewise returns
503 for an unavailable index, checks Host/Origin, caps request bodies, and bounds
socket reads. It is still a temporary local preview, not a hardened LAN service.

The real transport test uses the official SDK client to launch a separate server
process against original synthetic EPUB/PDF fixtures. It checks discovery,
structured output, schemas, calls, citations, abstention, errors, unchanged DB
bytes and process shutdown. Record the actual negotiated protocol version in
`ci-output/mcp-stdio-auto.json` and `mcp-stdio-legacy.json`; a successful compatibility handshake does not
prove every protocol revision or client works. Test execution remains Spark-only.
A pending job or a fake registration test alone is not transport verification.
See [the dependency and verification commands](MCP.md).

LAN TLS/authentication, client/book authorization, host binding, credentials,
firewall rules and persistent service remain separate implementation and approval
work. [LAN-PLAN.md](LAN-PLAN.md) gives the concrete acceptance sequence. No LAN
listener or client registration is installed by the stdio entry point.

## Research comparison before model selection

The 2026-10-03 research recommends retaining BGE as a control, comparing
Qwen3-Embedding-0.6B and Voyage 4 Nano, then measuring BM25 fusion and reranking
independently. Parsing and visual retrieval are selective escalation paths.
See the [evaluation plan](evaluation/README.md#research-informed-comparison-plan-2026-10-03-not-executed)
for proposed question quotas, held-out evaluation, metrics and approval boundaries.
These are planned comparisons, not model installations or measured results.

## Next vertical slice

1. After the research recommendation and user decision, pin the selected model
   revision and approved local embedding dependencies on Spark; reconcile the
   existing embedding/config/storage field drift. Preserve
   original metadata/citation IDs and embed only a deliberately chosen sample.
2. Add the chosen retrieval/index adapter behind the shared retrieval contract.
   Record model
   revision, dimension, normalization, query instruction, chunking version, and
   distance in an index manifest. Reject mismatches instead of mixing vectors.
   Combine lexical and vector candidates, deduplicate, and evaluate ranking on
   useful human questions, source/page grounding, and unanswerable questions.
3. Select an already-approved local answer-model endpoint, if desired. Keep quoted
   evidence distinct from generated synthesis; validate citation IDs and abstain
   when evidence is insufficient. No private passages leave the local deployment.
4. Test the actual MCP SDK/client and human preview against that same sample on
   Spark. Only after review approve LAN access and persistent hosting.
5. Broaden the corpus deliberately after provenance, retrieval quality, resource
   usage, interruption/resume, and backup/restore acceptance checks pass. The
   current synthetic gate is regression evidence, not proof of library-wide quality.
