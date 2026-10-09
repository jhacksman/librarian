# C1: recorded PDF omission results

C1 is closed within the frozen omission-visibility scope and retained on
`codex/librarian-readiness-20261003`. Independent verification found no remaining
blocker. This is branch acceptance, without deployment or private-library adoption.

Spark job `librarian-20261004-coverage-001` passed at exact source
`b20a126c49b03a61896b419e47955237c633815b`. The change exposes stored extraction
limits through completed/resumed ingestion, catalog, status, search and direct
passages. CLI text and the loopback preview display warnings; existing MCP tools
return the same structured fields. It requires no migration or reindex.

## Observed behavior

The authored four-page PDF has text on pages 1 and 3 and blank pages 2 and 4.
Its report retains `[2, 4]`, count 2, and `completeness: "not_assessed"`.
The returned quotation remains `Alpha needle evidence.` on physical page 1,
with source-character span 0–22. Direct lookup reproduces that passage.
Filtered no-match and empty-normalized-query responses still report both omitted
pages. A nonexistent book filter reports an empty scope.

A separate PDF with no recorded omissions returns an empty page list and count
zero, with the same completeness caveat. EPUB metadata remains unknown for this
page-specific report. The three-book fixture's aggregate is three books, four
chunks, two reported page-coverage records, one unknown record and two omitted
pages from one book. These counts describe authored fixtures, not the private
library.

The passing cases also cover malformed, oversized and NUL-containing metadata;
full-list validation beyond the 20-page preview; resume without reparsing;
source removal; exact citations and unchanged read-only database bytes; explicit
all-empty/parser failures; safe human rendering; and optional-adapter fallback
and error propagation. The [frozen contract](../COVERAGE-POLICY.md) defines the
65,536-character metadata limit, unknown semantics and scope rules.

## Verification

| Gate | Full MCP profile | Minimal profile |
| --- | --- | --- |
| Complete suite | 204 passed | 192 passed, 12 expected SDK skips |
| New C1 cases, included above | 54 passed | 52 passed, 2 expected SDK skips |
| Dependencies | 35 existing hash-locked wheels | 12 existing hash-locked wheels |

There were no failures or errors. Scoped Ruff and both dependency checks passed.
Five existing Pydantic deprecation warnings appeared in each test profile.
The original seven-query retrieval gate passed: six positives at rank 1, one
correct abstention and six source-bound citation checks. No standalone large
PERF1 benchmark or default/held-out quality evaluation was rerun.

The new real MCP cases each made seven calls through the pinned SDK: auto mode
negotiated `2026-07-28`; legacy mode negotiated `2025-11-25`. They asserted
structured/text parity with the service, correct scoped coverage and unchanged
database bytes. Existing transport and excerpt-continuation cases also passed.

Execution used CPython 3.12.15 on Linux ARM64, two CPUs, 4 GiB and the existing
600-second job limit. The job completed in 36.17 seconds with exit zero and
cleanup verified; its container was absent. Bridge networking remained available
throughout, with reviewed use for dependency downloads and local test traffic.
This elapsed job time is not a new search-performance measurement.

## Evidence and limits

The sibling `readiness-results/librarian-20261004-coverage-001/` preserves all
23 returned output files and 18 metadata files, with `SHA256SUMS` and
`custody.json`. Source archive SHA-256:
`fa7c15ee9c14f05ef32f2aeb08755247eab558ebccc09d704b508698225e1821`.
Returned artifact archive SHA-256:
`7fdee5415ca2c99fe28633df1bdf05c99cdfa3811be34163ee21d2cfc1ff2efb`.
The separate `readiness-results/coverage-verification-20261004.json` records
independent source, artifact, dependency, result and semantic verification.

Saved C1 service/MCP records retain expected service payloads after exact wire
equality assertions pass. Raw wire envelopes, most temporary source books and
database snapshots are not retained. No-write, parser-failure and rendering
claims therefore combine reviewed source assertions with passing test results;
they are not independent replays of deleted fixtures. Generated EPUB identities
can differ between cases; comparisons preserve those differences and map only
explicit fixture IDs when comparing semantics. HTML was checked as text and by
HTTP tests, without a visual browser review.

Recorded omissions do not assess meaningful content or extraction completeness,
recover images, or provide OCR. Synthetic Linux validation does not establish
private-library quality, macOS runtime behavior, actual client registration or
LAN deployment. The private corpus/index stayed untouched. PERF1 remains closed
with its original source and timing attribution; added reporting work was not
benchmarked here.
