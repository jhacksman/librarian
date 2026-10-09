# Librarian on one machine

This is the JavaScript product implementation. The earlier Python pilot remains
as reference and regression evidence; this application uses its own SQLite schema
and refuses to modify that index. Current product implementation and dependency
qualification are in progress. A source commit is not evidence that the real
collection has been indexed or that local AI has passed acceptance.

The browser, HTTP API, catalog, import coordinator and retrieval code are
JavaScript ES modules. Native PDF rendering and a local model engine run on the
same machine. There is no hosted inference API, CDN, remote font or separate
vector service. SQLite stores the catalog, full text, source locators and model-
identified vectors. Exact vector search is a first implementation whose complete-
collection latency still needs measurement.

## Run on Spark

Use Node 24.20 or a qualified later Node 24 patch. During development, all installs,
tests, import, embedding and browser execution go through the Spark coordinator.
Do not run them on the M6. The initial dependency qualification produces and
returns a package lock; subsequent installs must use the reviewed lock with
`npm ci --ignore-scripts`.

```sh
cd app
npm ci --ignore-scripts --no-audit --no-fund
export LIBRARIAN_DATA_DIR=/home/devin/librarian-data
export LIBRARIAN_IMPORT_ROOTS=/home/devin/librarian-input
npm start
```

The default address is `http://127.0.0.1:3474`. This is a local development binding,
not a claim of LAN deployment. An admitted Spark browser-access route must set
`LIBRARIAN_HOST`, `LIBRARIAN_PORT` and `LIBRARIAN_ORIGIN` to its exact address.
The API checks Host and Origin and does not trust forwarded headers. No accounts,
firewall changes or public deployment are created by these commands.

For a review session that preserves imported text and its existing vector index,
start the admitted application with `LIBRARIAN_MAINTENANCE=disabled`. The server
then rejects import, resume/pause, reindex and document-embedding actions, and
does not reconcile interrupted import jobs on startup or import-history reads.
The browser disables those controls and explains their availability. Search,
source downloads, reading progress, metadata edits and edition grouping remain
available. Query embeddings still work when the local model is configured.
The setting defaults to `enabled`; it applies to this HTTP application, not
separately launched CLI processes. The coordinator must continue to serialize
access to the data directory. An empty import-root list alone does not disable
reindexing of managed sources.

The regression suite is `cd app && npm test`. In the admitted Spark test image,
with the existing sandboxed Playwright installation, run
`LIBRARIAN_MAINTENANCE_BROWSER_OUTPUT=/tmp/librarian-maintenance-check node scripts/maintenance-browser.mjs`
from `app` to exercise the disabled controls and reading/search journeys at
desktop and mobile sizes. Use a fresh output directory; the driver creates its
own synthetic EPUB and library, with no corpus or model access. These commands
are for coordinator execution on Spark.

Put authorized books in the configured import folder, then use **Imports** in
the browser. For the existing Humble Bundle collection, preserve its transfer
manifest alongside the files. It defines the 491 expected binary source items;
the two transfer/source metadata manifests are not extra books. Import copies
and verifies originals into the managed data directory before publishing text.
After import, reading and source download use that copy, not NAS paths.

```sh
node src/cli.mjs import '/home/devin/librarian-input/Humble Bundle 2025'
node src/cli.mjs status
node src/cli.mjs receipts > /home/devin/librarian-data/import-receipts.jsonl
node src/cli.mjs resume JOB_ID_FROM_IMPORT
node src/cli.mjs reindex BOOK_ID
```

Keep the complete data directory outside Git. It includes the database, original
managed copies, cover derivatives and resumable state. Do not move or delete
individual SQLite WAL files while the application is open. Original NAS books
remain unchanged. A changed original is a new import decision, not an automatic
replacement of an edition already cited.

After a hard restart, **Imports** identifies interrupted jobs and offers Resume.
If every file already has a saved disposition, it recovers the completed status
without creating another extraction attempt.
An incomplete inventory needs a fresh folder scan. A live command-line importer
keeps ownership; the browser does not pretend it can pause that other process.
Reindex is an explicit action on a book's details page, or the CLI command above.
It reparses verified managed copies without consulting the original folder.
`reindex-files FILE_ID...` can retry an unsupported managed format after an
operator configures its converter. A source-identity conflict remains failed on
ordinary Resume and never substitutes an older local copy for a new manifest.

**Find passages** searches book contents and opens exact source slices without
generating an answer. Its title/author scope picker searches the full source-book
catalog, including books you have not opened. **Organize** offers conservative format/edition suggestions
and manual selection. Confirmed groups retain separate source books, citations,
and reading positions. Choose an individual source inside a group before reading
or asking; unlinking is reversible and preserves the source.

## Local AI

Configure models already installed and approved on the same Spark. No startup,
import or query command downloads a model. Empty model configuration leaves
lexical browsing/search available and explicitly reports the missing capability.

```sh
export LIBRARIAN_MODEL_PROVIDER=ollama
export LIBRARIAN_MODEL_ENDPOINT=http://127.0.0.1:11434
export LIBRARIAN_EMBEDDING_MODEL=EXACT_INSTALLED_EMBEDDING_MODEL
export LIBRARIAN_CHAT_MODEL=EXACT_INSTALLED_CHAT_MODEL
export LIBRARIAN_EMBEDDING_REVISION=VERIFIED_MODEL_DIGEST
export LIBRARIAN_CHAT_REVISION=VERIFIED_MODEL_DIGEST
node src/cli.mjs embed
node src/cli.mjs embed --once
node src/cli.mjs embed --retry-failed
node src/cli.mjs ask 'How does this collection explain bounded contexts?'
npm start
```

The alternate `openai` adapter is for a local compatible engine only; it does
not imply use of the hosted OpenAI API. Its endpoint is an API root such as
`http://127.0.0.1:8000/v1`. Separate engines on this machine can use
`LIBRARIAN_EMBEDDING_ENDPOINT` / `LIBRARIAN_EMBEDDING_PROVIDER` and
`LIBRARIAN_CHAT_ENDPOINT` / `LIBRARIAN_CHAT_PROVIDER`. Public endpoints and
redirects are rejected. Model revision, actual
runtime identity and settings must be recorded during qualification.

Embedding batches persist atomically and resume missing work. The application
does not label a bounded, cancelled or failed run complete. Ollama receives
`truncate:false`, so an overlong chunk raises a visible indexing failure. Real
model token limits must still be qualified before full indexing.
`embed --once` runs one existing slice with a 2,000 attempted-chunk limit and
an internal 120-second indexing budget. A successful partial yield exits zero while
reporting `status: "bounded"`, `execution.completed: false`, and remaining
coverage. Run the same command with the same model configuration to resume;
saved compatible vectors are retained. Cancellation, unavailable inference and
failed inputs still exit nonzero. These indexing bounds do not replace the
coordinator's outer deadline for model startup, auditing and cleanup.
The Imports page exposes Build/Resume indexing and Retry failed chunks. Input
rejections are recorded separately for the exact chunk text and model identity;
they do not prevent later valid chunks from being processed. Transport failures
remain retryable. Failed chunks stay outside completed vector coverage.

Retrieval combines FTS5 lexical results and cosine-ranked local embeddings by
reciprocal rank fusion, with book scope applied to both branches. Exact source
windows have explicit UTF-16 offsets; nearby context stays within the same book.
Answer context preserves the selected reader range when it fits, reserves one
anchor per source, and grows those anchors before their boundary passages and
additional hits from the same book. Nearby reader sections precede distant
summaries. This gives a worked example room to retain its input, operation and
result within the existing character, evidence and serialized-message budgets.
A long selected section can use an explicitly partial source window. The model
receives citation numbers, titles, readable locations and exact text; full
identities and source hashes remain in the returned citation map. Overlapping
source ranges are counted once, while gaps remain separate spans.
For all-library questions naming two to four unambiguous catalog title phrases,
retrieval reserves one candidate from each named source before filling the other
slots by rank. A bounded per-book lexical search can recover a named source absent
from the global candidate pool. If a requested source or its adjacent context
cannot fit, generation abstains. Optional extra hits cannot displace an admitted
reader range. Title matching cannot establish relevant support,
edition equivalence, or the first/last example in an entire book.
This coverage hint recognizes catalog title prefixes of at least three words and
15 characters, within the first 32 title words. Unknown or shorter titles are not
resolved. It scans at most 10,000 catalog sources and 512 title occurrences;
exceeding those bounds or finding an ambiguous edition withholds generation.
Questions outside the two-to-four detected-source reservation scope have no
automatic guarantee that every requested source is represented.
Ask requests local generation after retrieval. Answers retain linked evidence;
the model returns structured paragraphs with their own source references and
exact supporting quotations. After validating each quotation against the source
span, the application renders that paragraph's linked citation markers. The
older answer-string format still requires its original inline references and
support; missing references in historic responses remain failures.
That mechanical check does **not** establish factual
support: independent real-source answer review remains an acceptance gate.
Responses expose the actual inference outcome and distinguish a generated
answer, a useful partial answer, unavailable inference, and copied excerpts.

The reader offers **Extracted text** and **Original** views. Original PDF uses
the verified managed PDF and its physical page number. Original EPUB displays
a formatted, reflowable source section, preserving permitted local styles,
images and links between source sections. EPUB sections are not fixed pages.
Scripts, external resources and unsafe archive members are refused. Each view
keeps its own position when switching and returning from a citation.

**Ask about this page** (PDF) or **Ask about this section** (EPUB) carries the
selected source and section to Ask. The server derives the locator from its own
records and rejects mismatched identifiers, pages, members and offsets. Nearby
or other source passages may support the answer; their citations retain their
actual location. For an already selected source, a literal request such as
“only pages 185–186” restricts both retrieval branches before ranking and filters
neighbor passages to those physical pages. Unsupported page lists or reversed
ranges are rejected; this narrow syntax does not establish complete page coverage
or interpret EPUB sections as pages. The scope controls explicitly release this
reader context to the selected book or to the library. Search continues to locate
passages.

An API reader-context request identifies records, never supplies trusted text:

```json
{"question":"Explain this example.","bookId":"SOURCE_BOOK_ID","responseMode":"auto","readerContext":{"bookId":"SOURCE_BOOK_ID","sectionId":"SECTION_ID","scope":"section"}}
```

Reader and answer changes in this branch require Spark qualification before
they replace the existing review session. Preserve real model request/response
and source context, independently grade the frozen real-book cases, and verify
desktop/mobile reader, scope and citation journeys. Synthetic protocol tests
alone do not qualify useful answers or original-book rendering.

When generation is unavailable, abstains, or fails citation validation, **Ask your
library** shows **Cited excerpts**. These are exact quotations from up to six
retrieved chunks, with separate source locations, clipping notices and links to
the highlighted reader text. Each quote contains at most 1,600 UTF-16 code units.
They use the retrieved candidates independently of the smaller model prompt;
they do not establish completeness, firstness, agreement, or even full relevance
to the question. The UI keeps quotations visible instead of clipping them to a
two-line preview. Generated answers remain a separate capability and quality gate.

`POST /api/ask` also accepts `"responseMode":"excerpts"` to request quotations
without a chat call (`"auto"` is the default). Retrieval may still use a configured
local embedding model. Responses add `responseKind` (`source_excerpts`,
`no_evidence`, or `generated`); fallback quotes live in `extractive.passages`.
Existing `status`, `abstained`, `answer`, and `citations` retain the generation
outcome and exact model-context diagnostics. Excerpt citation numbers apply to
`extractive.passages`, separately from the model's citations. With no exact source
matches the UI reports that no excerpts are available, not that the books contain
no answer. No model is required for lexical excerpts.

## What has to be demonstrated

The real collection contains **491 files, not 491 established logical books**:
221 EPUB, 227 PDF, 2 MOBI, 4 PRC, 15 ZIP and 22 torrent files. It has 489 recorded
unique content hashes. Alternate formats are conservative grouping candidates;
duplicate bytes retain source aliases. ZIPs receive bounded member inventories;
torrents never initiate downloads. Unsupported formats and textless PDF pages
remain visible and do not count as indexed text or completed vector coverage.

Acceptance requires a 491-row reconciled disposition ledger, readable-book text
and vector coverage, a usable route for unique MOBI/PRC content, real questions
with source judgments, and live browser journeys through import, browse, reader,
ask and citation return. The application must continue working after the NAS
source is unavailable, with all inference on Spark. Synthetic fixtures and source
inspection are useful checks but do not satisfy those gates.

MOBI/PRC conversion is optional and uses an existing installed Calibre executable:

```sh
export LIBRARIAN_EBOOK_CONVERT=/absolute/installed/path/ebook-convert
export LIBRARIAN_BWRAP=/absolute/installed/path/bwrap
```

The default Bubblewrap route isolates networking and process/filesystem access.
`LIBRARIAN_CALIBRE_ROOT` may specify an installed Calibre tree. An operator may
set `LIBRARIAN_CONVERTER_ISOLATION=container` only when the coordinator has admitted
an isolated container with networking disabled and an owned process cgroup.
The application cannot establish that external isolation by itself. Conversion
retains a hash-addressed derived EPUB with original/executable/output provenance;
DRM is rejected, never bypassed. No converter is downloaded or installed by the
application, and these settings do not prove that the six real sources convert.

Run source tests on Spark with `npm test`. Browser evidence must use a live HTTP
server, never an injected HTML snapshot. Record import/embedding duration, source
and vector counts, peak memory, disk growth, cold/warm search, completed-answer
latency, and catalog responsiveness during background work. The current response
is not streamed, so first-token latency is not exposed or measured.

With the live app and existing approved Playwright ARM64 runtime available on
Spark, the browser driver exercises desktop/mobile navigation and saves viewport
screenshots plus a machine-readable report outside Git:

```sh
LIBRARIAN_DEMO_ORIGIN=http://127.0.0.1:3474 \
LIBRARIAN_DEMO_OUTPUT=/home/devin/librarian-evidence/live-001 \
LIBRARIAN_DEMO_DATASET=existing-catalog \
LIBRARIAN_DEMO_BOOK_ID=IMPORTED_BOOK_ID \
LIBRARIAN_DEMO_QUESTION='SOURCE_GROUNDED_QUESTION' \
node scripts/browser-smoke.mjs
```

This checks live UI behavior and records missing capabilities. It does not
certify corpus provenance or whether a generated answer is supported. Reading
position saves are part of the journey; it does not start imports or edit book
metadata. Keep private screenshots and evaluation evidence within the authorized
owner workflow.

`node scripts/recovery-browser.mjs` is a separate Spark-only regression using
owned synthetic EPUBs and a labelled mock local embedding endpoint. It kills
and restarts the actual server, removes only fixture originals, and resumes
imports and vectors through the browser. It does not establish real-model quality.
See its required output-directory environment variable in the script. Full-corpus
receipts can be independently checked by `scripts/acceptance-report.mjs`; its
bounded accounting result is distinct from product acceptance.

The fresh search-scope regression imports 29 synthetic EPUBs, then selects an
unseen source beyond the first catalog page in new desktop/mobile browser
contexts. It checks exact source identity and the citation/reader return path:

```sh
LIBRARIAN_SEARCH_SCOPE_OUTPUT=/tmp/librarian-search-scope-001 \
node scripts/search-scope-browser.mjs
```

Use a fresh output path. This uses the admitted Spark browser runtime and does
not replace the real-book evaluation.

A **16 GiB RAM, CPU-only, local SSD** profile with one query at a time, serialized
background indexing and a small quantized answer model is a portability target,
not a measured minimum. Disk must accommodate the 14.14 GB collection plus model
weights, extracted text, vectors and recovery space. Spark resource caps are not
equivalent to testing a physical smaller computer.

## Capability priorities

Calibre's [library interface](https://manual.calibre-ebook.com/gui.html) and
[browser server](https://manual.calibre-ebook.com/server.html) are the baseline
for useful library management. This project does not yet claim Calibre parity.

| Capability | Delivery priority | Current implementation direction |
| --- | --- | --- |
| Folder import, duplicate receipts, resume, errors | P0 | Managed local copies and durable per-file ledger |
| Cover catalog, title/author/format/status filters, sort | P0 | Local browser catalog |
| Details, TOC, reader, source jump, reading place | P0 | Stored sections and exact source locators |
| Local lexical + vector search and cited answers | P0 | FTS5, model-identified vectors, local adapters |
| Continuity across neighboring sections | P0 | Structural windows and bounded neighbor expansion |
| Unique MOBI/PRC access | P0 | Converter/parser route still to be qualified |
| OCR for otherwise unreadable pages | P0 when required | Visible omissions until an approved local route exists |
| Metadata and tags | P1 | Local corrections in catalog |
| Collections, annotations, safe format/edition merge | P1 | Follow core real-collection acceptance |
| Device sync, broad conversion UI, ebook editing | Later parity | Explicit backlog |
| News recipes, plugin ecosystem | Later parity | Explicit backlog |

Parsing choices use [PDF.js](https://mozilla.github.io/pdf.js/), bounded
[yauzl](https://github.com/thejoshwolfe/yauzl) ZIP streams, and
[htmlparser2](https://github.com/fb55/htmlparser2) without executing ebook HTML.
Node's [SQLite API](https://nodejs.org/docs/latest-v24.x/api/sqlite.html) is built
in but remains an API stability consideration on Node 24; runtime qualification
must check FTS5 and recovery behavior. Local embedding behavior follows the
[Ollama embedding API](https://docs.ollama.com/api/embed).

## Book thumbnails and source metadata

Every imported source book has a bounded thumbnail stored in SQLite `book_assets`, including an explicitly labelled generated placeholder when no usable cover exists. Embedded raster covers are resized to at most256×384 pixels and256KiB. PDF thumbnails show physical page1 and are identified as `first_page`; they are not a claim that a publisher cover exists. Original cover files and original books remain traceable. `GET /api/books/:id/thumbnail` serves the stored bytes; book detail includes the full source-metadata record and image provenance. Ask groups passages under a small thumbnail and title for each source book. Formats combine in this presentation only with a validated ISBN and matching title, authors, edition and publication date; every source/citation ID stays intact.

New imports capture all available bounded OPF metadata nodes/attributes/repeated values or PDF Info and XMP. Existing books use a metadata-and-cover-only backfill; it never calls text reindexing or embeddings. A failed or oversized source is reported explicitly and retains an honest fallback, rather than a claimed complete metadata result.

PDF capture retains every Info field reported by PDF.js, including its custom-field Map converted to a JSON object, plus parsed XMP fields and the parser's raw XMP string. PDF.js limits custom Info values to strings, numbers, booleans and PDF names; dictionary, stream and array values omitted by that upstream API are not represented by this capture.

Run backfill only through the Spark coordinator after stopping all readers, accepting a current snapshot/custody baseline and taking an exclusive DATA lease. Keep the active demo on its pinned app/database until testing and coordinated rollout finish. Preserve deliberate user metadata, progress and grouping edits. Use the existing installed dependencies; no external book or cover lookups are performed.

```sh
# Coordinator-owned isolated/snapshot DATA only; ordinary M6 execution is prohibited.
LIBRARIAN_ENRICHMENT_EXCLUSIVE=1 node src/cli.mjs enrich-books --limit 25
# Inspect pending/failed counts; repeat bounded batches until pending=0.
# Explicitly retry failed books after resolving the retained source error:
LIBRARIAN_ENRICHMENT_EXCLUSIVE=1 node src/cli.mjs enrich-books --limit 25 --retry-failed
```

The `--limit` flag and value are separate arguments (for example `--limit 25`). The exclusive environment value is an operator assertion, not a replacement for coordinator lease/custody checks. Backfill failures do not block later books in default batches. Before switching the demo, independently verify all book asset records, available source metadata, preserved text/vectors/progress/groups and real browser navigation/Ask/source workflows. Implementation-author tests do not constitute the independent usability verdict.

### Native page metadata recovery (source-review candidate)

Indexing completion and descriptive metadata completeness are separate. Catalog
cards now flag filename titles or missing authors as “Metadata needs review”, and
empty author arrays display “Author not identified”. Missing values remain missing;
this label is not a claim that a genuinely anonymous work has a recoverable author.

The opt-in PDF front-matter observer uses the existing PDF.js parser to capture
physical pages1–8, exact native text, UTF16 quote offsets, page-text SHA256 values
and bounded typography hints. Defaults preserve the previous extraction behavior.
The observer does not OCR, invoke a model, access the network, convert files, open
a database or create sections/chunks/vectors. Image-only pages are recorded as
empty native text. Limits are8pages,4096items/page,16384characters/page,
49152characters total and256KiB serialized evidence. Exceeding a bound fails
explicitly. Font prominence and an ISBN are review aids; neither authorizes an
inferred catalog title, author or merge across source books.

Run these commands only through the existing qualified Spark coordinator after
source review, with an immutable full checkout mounted at `/work`, installed
previously pinned dependencies, an exact retained source tree mounted read-only,
and fresh owned output outside both checkout and DATA:

```sh
cd /work/app
node --test --test-reporter=junit test/source-metadata-recovery.test.mjs test/extract.test.mjs test/book-enrichment.test.mjs
node --test --test-reporter=junit test/*.test.mjs
node scripts/collect-metadata-evidence.mjs /inputs/cases.json /retained/sources /evidence/front-matter
node scripts/audit-metadata.mjs /retained/library.sqlite > /evidence/metadata-audit.json
```

The first pilot manifest pins the18ISBN-filename PDFs from the failed candidate
catalog. The independent reviewer must read the actual page evidence, distinguish
an author/editor credit from a brand, publisher, dedication or copyright holder,
and preserve title/subtitle/edition words that are actually present. Source-page
quotes form the gold values; source filenames and prior candidate metadata do not.
Where native text is missing or role/edition is ambiguous, retain the review flag
and request a bounded actual-page image comparison using the existing approved
PDF renderer. No OCR or new vision/model runtime is enabled by this change.

The reviewer supplies a `librarian-reviewed-source-metadata-batch/v1` bundle with
1–25reviews. Each `librarian-reviewed-source-metadata/v1` review names the original
`sourceSha256`, independent `reviewer`, immutable `evidenceReceiptSha256`, and
optional `fields.title` / `fields.authors` (an array). Every value has `evidence`
entries containing physical `page`, `pageTextSha256`, UTF16 `start` / `end`, and
exact `quote`. A value must equal its source quotes joined with whitespace
normalization. No unquoted substitution or external bibliographic value is allowed.
The coordinator must verify the independent receipt's actual hash and review
ownership before admitting that bundle; supplying a64hex string alone is not
independent approval.

On a new isolated clone, under the coordinator's exclusive DATA lease:

```sh
cd /work/app
LIBRARIAN_ENRICHMENT_EXCLUSIVE=1 node scripts/apply-reviewed-metadata.mjs /inputs/reviewed-receipts.json
```

This reparses the bounded original source, checks source/page/quote identity and
updates only deficient display fields. Existing meaningful title/author fields
and all user edits win, including edits arriving during extraction. The original
Info/XMP remains separately retained, and a receipt records the reviewed values,
confidence, applied/preserved fields and unresolved fields. Existing thumbnail
bytes are kept. Invalid reviews leave existing book/asset rows unchanged.
The1MiB source-metadata cap remains enforced transactionally.

These commands are proposed for coordinator qualification; they have not run on
the actual18sources yet. Do not replace the passed c36covers/navigation candidate
or live demo with this source successor until tests, actual gold, protected-row
preservation and independent catalog acceptance have been reviewed. Full coverage
means accounting for all451books as meaningful reviewed/embedded metadata or
explicit unresolved/anonymous cases, not equating451successful source captures
with451complete bibliographic records.
# Frozen independent metadata pilot

The coordinator can apply the independently frozen 18-source gold set to a new isolated clone. `apply-frozen-gold.mjs` accepts only gold bytes with SHA256 `d023dff9723d508062c90faff7f5757815f3cae56576fc40c7ddf2d008224f80`. Supply all 27 frozen PNGs under their SHA256 filenames. The writer rechecks original PDF identities and native page quotes, preserves existing meaningful fields and user edits, keeps creator credits separate from authors, and leaves unobserved fields unresolved. Gold and rendered corpus pages stay outside Git.

Run in the existing qualified Spark environment, with exclusive clone custody and a fresh evidence directory:

```sh
node --test --test-reporter=junit test/*.test.mjs
LIBRARIAN_ENRICHMENT_EXCLUSIVE=1 LIBRARIAN_DATA_DIR=/isolated/data node scripts/apply-frozen-gold.mjs /inputs/gold.json /inputs/render-png /evidence/application.json
node scripts/frozen-gold-browser.mjs http://127.0.0.1:PORT/ /inputs/gold.json /evidence/application.json /evidence/ui
```

The coordinator binds the isolated server, existing Playwright installation and finite process-tree bounds. `application.json` records actual before/after values and cover hashes. The browser proof checks all 18 Library cards and detail fields, retains desktop screenshots plus four mobile cases, and rejects browser writes and external requests. Independent comparison and visual acceptance belong to the original critic; a driver pass alone does not establish that verdict.
