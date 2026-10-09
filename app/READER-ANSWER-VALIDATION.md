# Reader and answer proof contract

The implementation defaults Ask to `responseMode: auto`: retrieval precedes local
inference, a generated answer carries exact source citations, and unavailable or
insufficient generation leaves supporting excerpts visibly supplemental. Find
passages remains a lookup. Reader context is validated against database book,
section and native page/member/anchor/UTF-16 offsets before inference.

## Execution boundary

Run `scripts/reader-answer-browser.mjs` only as a finite job through the existing
Spark coordinator on prepared Linux ARM64 Node 24 with the existing
`/opt/playwright/node_modules/playwright`. The coordinator imposes a 420-second
process-tree bound; the driver has a 360-second work deadline. Do not run Node,
SQLite, model, browser or native jobs on the M6. Do not install models or packages,
contact the live reader service, or upload source books. Evidence paths must be
fresh absolute directories under `/tmp`, outside Git.

`report.passed` describes mechanical browser checks only. `acceptanceVerdict` stays
false; an independent reviewer assesses saved raw answers against frozen gold.
A completed HTTP/model request and an exact quoted span do not prove that a claim
is semantically correct.

## Existing isolated real-book service

The coordinator starts and owns a fresh loopback app service backed by its
isolated clone, with the authorized installed local model. This mode launches
only its owned browser; it does not create a database, import fixtures or start an
app/model service.

Required environment:

- `LIBRARIAN_READER_MODE=existing_service`
- `LIBRARIAN_READER_APP_ROOT`: absolute immutable staged `app/` source directory.
- `LIBRARIAN_READER_OUTPUT`: fresh absolute `/tmp` evidence directory.
- `LIBRARIAN_READER_ORIGIN`: coordinator-owned `http://127.0.0.1:<fresh-port>`;
  ports 3474 and 3475 are rejected.
- `LIBRARIAN_READER_CASES`: absolute frozen JSON case-wrapper path.

Case input is `{ "cases": [...] }`, with one to four entries. Each entry contains
`id`, `bookId`, `question`, and either `sectionId` or the source's native `page` or
`member`. Optional `charStart`, `charEnd`, and `fragment` select an exact passage.
An optional `expected: "no_answer"` checks honest abstention mechanically; other
entries require real completed generation and a nonempty answer. The driver does
not send expected claims or semantic gold into the model. Missing section IDs are
resolved from the server's own catalog page/member metadata.

The frozen wrapper supplied by the root coordinator contains:

- Learning SQL NULL handling in `OEBPS/ch08.html`, UTF-16 offsets 12218–14958.
- Interview Prep Book physical page 184, with adjacent Promise.all evidence.
- The unsupported publisher-secret question, expecting honest no-answer.

Both desktop (1365×900) and mobile (390×844) jobs record original-view screenshots
and metadata, EPUB scroll across toggles, source/edition/page or section scope,
the exact submitted Ask request, actual inference metadata, raw answer JSON,
exact quoted citations verified against source-section HTTP reads, citation
navigation and extracted highlighting, conversation return, and explicit
whole-book/library scope release. The only permitted app mutations are Ask and
reading progress in the coordinator-controlled clone. No import, metadata or
catalog-group endpoints are used.

Outputs include `report.json`, per-profile raw `*-ask.json`, original-view PNGs,
and citation-roundtrip PNGs. Reports pin driver and implementation source hashes,
case input hash, browser/runtime identities, response bytes/hashes, and cleanup.

## Separate synthetic proof

Use `LIBRARIAN_READER_MODE=synthetic` (the default) for an owned temporary database
with tiny generated PDF/EPUB fixtures. This mode starts only its bounded loopback
fixture app services and uses the already running, authorized local model. It
also checks model-unavailable UI and negative reader scopes before inference.

Required environment includes the source/output paths above plus:

- `LIBRARIAN_READER_MODEL_ENDPOINT`: existing loopback model URL. Ollama uses an
  origin; OpenAI-compatible mode also permits the exact `/v1` API root.
- `LIBRARIAN_READER_MODEL`: already installed model name.
- `LIBRARIAN_READER_MODEL_PROVIDER`: `ollama` or `openai`.
- `LIBRARIAN_READER_MODEL_REVISION`: authorized installed revision, if available.
- `LIBRARIAN_READER_CONTEXT_TOKENS`: configured context bound, default 8192.

Synthetic generation calls are real and bounded to twelve requests. A second
owned app instance without a configured model verifies honest unavailable UI.
Fixture import, database state and owned services are removed at completion.
Synthetic success is not real-book acceptance.

## Stable UI selectors

`reader-content` exposes `data-book-id`, `data-section-id`, `data-reader-mode`,
native `data-physical-page` or `data-member`, and `data-view-ready`.
The latter is `true` only after the selected extracted/native view is ready; it
is `false` during opening and `error` on failure. Wait for the exact requested
hash, section, mode and native page/document identity together. The global
`aria-busy` flag alone can describe the preceding route during navigation. Controls are `reader-original`,
`reader-extracted`, `reader-ask-context`, `reader-return-ask`, `reader-return-search`,
`original-pdf-canvas`, `original-pdf-viewport`, `original-pdf-zoom`,
`original-epub-frame`, `reader-contents`, `reader-source-notes`, and
`reader-source-identity`. PDF viewport `aria-busy` describes raster completion.

Ask exposes `ask-question`, `ask-submit`, `ask-pending`, `ask-result`,
`ask-scope-title`, `ask-scope-detail`, `ask-whole-book`, `ask-clear-scope`,
`ask-return-reader`, `ask-answer`, `inline-citation`, `citation-link`, and
`excerpt-link`. Responses expose `data-generated-answer` and `data-answer-status`.
An extracted citation's exact highlighted text is `#cited-passage`.

Fresh citation/TOC navigation carries `focus=1`, discards obsolete positions,
and seeks the new native fragment or exact source span. Ordinary toggles and
return links omit fresh focus and restore each view's independent position.


## Exact support from source lines

The model sees each passage once, with reversible `number|original line` prefixes.
It returns paragraph-owned support as `{ "citation": 1, "lines": [2, 5] }`.
Line numbers are inclusive and local to one passage. The application derives the
exact original quote and UTF-16 offsets, preserving blank lines, indentation,
nonbreaking spaces and repeated-line occurrence. Range, source identity and
12–2000-character quote checks remain mandatory. Disjoint spans use separate
references. Mixing `lines` and `quote` in one reference is rejected.

Legacy exact quotations remain accepted under the same strict validation. No
whitespace correction, ellipsis expansion or fuzzy quote matching salvages old
failures. Line selection and claim entailment still need independent review;
`semanticSupport` remains `not_automatically_verified`. Serialized prompt bytes
including prefixes count against the existing 6480-byte budget, with context
capacity unchanged at 8192.

## Reader cancellation and mobile evidence

Closing an original view settles its readiness with `AbortError`, including an
EPUB iframe that never finishes loading. The superseded route can then finish
without blocking a new extracted/native view. The regression uses a deliberately
never-loading iframe; browser evidence must independently qualify the roundtrip.

Contents starts collapsed on narrow reader layouts; source notices remain
available in a collapsed Source notes disclosure. Mobile PDFs start at Readable
size (at least 640 CSS pixels), with Fit width, 150% and 200% options. The original
page pans inside its own viewport. Size and both pan offsets are retained across
ordinary toggles/returns; fresh citation focus discards obsolete positions. The
raster remains bounded to 16 million pixels, and physical-page identity and exact
citation mapping are unchanged.

A separate view-only job must capture default/mobile readable width, changed
size, pan position, and visible native illustrations. Loaded resource metadata
alone does not establish image fidelity. Saved REAL004 had nine accepted positive
answers, three quotation-format failures and three honest no-answers; five of six
browser case checks passed. Those historical results do not qualify this new
prompt or reader implementation. Re-run the frozen positive/negative cases on a
fresh coordinator-owned clone after CPU qualification, then independently assess
answers and pixels. Broader frozen reader workflows remain separately pending.
