# U1: choose a book and inspect cited evidence

The human preview currently cannot browse the indexed catalog or select a book,
although CLI and MCP callers can scope questions by book ID. An independent
critic and two read-only skeptics accepted this P3 capability gap at `20f2bb2`.
This slice makes that existing scope usable through an integrated human flow.
It does not change retrieval ranking or generate answers.

## Shared catalog contract

Add optional `book_id=None` to `LibraryService.books` and `SQLiteRetriever.books`.
The existing MCP `list_books` tool accepts the same strict `BookFilter` as
`ask_library`: omitted/empty string means unfiltered; a nonempty value is a
64-character lowercase hexadecimal book hash. JSON null, booleans, numbers and
malformed IDs remain invalid MCP input. Keep the existing four tool names.

Return the same `{books, next_offset}` envelope and bounded public item fields.
Apply exact, parameterized book filtering before pagination. A valid unknown ID
returns an empty page without a global fallback. Existing limit/offset bounds,
stable ID ordering and unfiltered adapter calls remain unchanged. Unsupported
filtered adapter calls must fail rather than silently search the whole catalog.

## Human workflow

The preview shows at most 20 catalog entries at a time, with title, full book ID,
chunk count and existing per-book extraction coverage. Full IDs distinguish
identical titles and colliding shortened prefixes. No title is used as identity.
An explicit all-books choice restores the unfiltered scope.

Selecting a book resolves its current public metadata through the filtered
catalog at offset zero, independently of the displayed catalog page. The chosen
title, full ID and coverage remain visible while browsing other catalog pages,
after a question returns matches or no matches, and during excerpt continuation.
Do not trust a hidden title or scan all catalog pages to find the selection.

Browse, selection, question submission and contextual passage navigation submit
the current form state, including unsent textarea edits, selected book ID and
catalog offset. Questions are not added to navigation URLs. Browsing and selection
do not run a search automatically. Existing direct GET passage URLs remain usable.

The HTTP entry points remain loopback-only and foreground-only. New POST browse
and passage actions reuse the current request framing, body, socket and Host/Origin
guards. Reject unknown, repeated or conflicting form fields. Retain useful question
and scope state on user-correctable errors. An invalid or no-longer-indexed selection
is shown explicitly and must never become an all-library search. A contextual
passage outside the selected book is a request error. Internal failures remain
safe service errors rather than successful abstentions.

Render question/title/ID/excerpt text with existing HTML escaping. Use bounded
public metadata only; source paths and raw metadata never enter the public UI.
Preserve citation IDs, excerpt offsets, continuation semantics and C1 warnings.
Search citation numbering remains unchanged; a direct or contextual passage view
uses its own `[1]` display label, as the existing direct-lookup page does. Source
book/chunk IDs remain stable across views.
Search results remain quoted source evidence with honest no-match abstention,
without inferring that a matching excerpt answers the question.

## Verification and boundaries

Use small original synthetic PDF/EPUB books and independently authored questions.
Exercise same-title editions, exact selected-book results, a scoped miss where
another book matches, empty or removed scope, catalog pagination beyond the
selected row, unsent text preservation, long-excerpt continuation, PDF omissions,
unknown EPUB coverage, explicit return to all books, invalid/repeated inputs,
escaping and unchanged read-only database bytes. Test the actual loopback HTTP
flow and both real MCP transport modes, including exact filtered catalog lookup
and unchanged default catalog calls. Preserve source-grounded citation checks.

All application imports, installs, tests, lint and evaluation run only on Spark
through the coordinator, using the existing dependency profiles. Source editing,
review, packaging and artifact inspection stay on M6. No private corpus/index,
model download, service installation, LAN exposure, deployment or publication.

Do not change the lexer, literal guard, retrieval SQL/ranking, chunking, schema,
or frozen quality labels. The three known recovery diagnostics remain unresolved:
the keyword case ranks useful evidence behind same-book distractors; the natural
question misses a word-form difference; the paraphrase also misses. Record these
limits without claiming a ranking or semantic-quality gain from book selection.
Closed C1 and PERF1 remain attributed to their verified source; this slice makes
no new performance claim and does not rerun the large PERF1 benchmark.
