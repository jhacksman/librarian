# PERF1 measured results

PERF1 is closed. The substring-absence guard is accepted and retained on
`codex/librarian-readiness-20261003` in the isolated readiness worktree, within
the synthetic validation scope below. No merge, push, runtime installation,
private-index processing or deployment was performed.

The frozen performance rule passed at `51117e93957d30e7c76eb4edb32057012a33afde`
in Spark job `librarian-20261004-literal-performance-001`. That job's overall exit
1 remains recorded because Ruff found one C420 and four SIM117 violations.
The focused repair at `593920868f69ebf6c2108ceaf39b1f3c394c2b37` subsequently passed
both 58-test runs and the required Ruff scope in job
`librarian-20261004-literal-performance-002`. Separate verification closed the
final gate; no benchmark rerun was needed.

The experiment compared the correct literal predicate frozen at `c527e48` with
the substring-absence guard introduced at `977be2d`. Both versions used the same
database for each comparison. Eight original synthetic corpora covered 1, 8, 32
and 128 background books, with ordinary text and substring collisions. The two
largest corpora each had 141 books, 989,313 words and 4,144 chunks. No private
ebooks or existing library index were used.

## Measured behavior

All 56 query scenarios completed three A-B-B-A blocks, with five timed service
calls per phase: 30 observations per version and 3,360 observations overall.
The table reports pooled medians in milliseconds for the `archive++` miss;
the decision used the separately retained paired-block medians.

| Background books | FTS candidates | Plain baseline | Plain candidate | Collision baseline | Collision candidate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 4.674 | 0.287 | 5.313 | 5.259 |
| 8 | 256 | 35.345 | 1.059 | 39.382 | 39.555 |
| 32 | 1,024 | 140.109 | 3.721 | 161.197 | 160.694 |
| 128 | 4,096 | 578.617 | 15.992 | 649.230 | 650.254 |

For the largest plain miss, the three paired improvements were 97.239%, 97.254%
and 97.231%, exceeding the predeclared 20% requirement in every block. Its
30-sample nearest-rank p95 changed from 585.282 ms to 16.735 ms. Separate counters
recorded 4,096 predicate calls in both versions, with predicate lexer invocations
falling from 4,096 to zero: each absent substring was rejected by the guard.
The separate intrusive CPU profiles support that mechanism; their timings are
excluded from these service latencies.

The collision background contains `archive++17`, so the requested substring is
present inside a longer token. Both versions still invoked the predicate lexer
4,096 times and correctly rejected every candidate. The optimization does not
remove that broad collision cost.

The 128-flag invalid-boundary probes expose a real counterexample: pooled medians
increased by 0.262 ms (10.813%) in the plain corpus and 0.259 ms (10.497%) in the
collision corpus. These increases remain visible in the results. Every paired
block was below 0.268 ms, so none crossed the frozen requirement that a veto
exceed **both** 5% and 1 ms in at least two blocks. No measured scenario crossed
both thresholds in even one block. This is a bounded engineering decision,
not a statistical significance claim or latency SLA.

## Correctness and gate attribution

The executed harness checked exact A/B response equality, order, literal
provenance, abstention, citations, excerpt offsets and source-grounded
continuation. It also verified the result behind 35 rejected candidates,
book filters, Unicode boundaries, early/late literals and 128-literal cases.
All eight recorded database before/after hashes matched.

The same job passed all 150 full-environment tests and 140 minimal-environment
tests, with 10 expected optional-SDK skips. The original seven citation cases
passed. The unchanged 22-case quality suite retained its 16 passing gates and
three existing diagnostic failures: `recovery-keywords`,
`recovery-natural-question` and `recovery-paraphrase`. All 20 formerly held-out
cases passed across service, modern MCP and legacy MCP paths. These are disclosed
regression cases now; no new blind-evaluation claim is made.

PERF-H3 records the five Ruff failures. An independent critic and two skeptics
accepted the narrow repair before the respective sole fixers changed it.
Commit `593920868f69ebf6c2108ceaf39b1f3c394c2b37` replaces the untimed constant-zero
counter comprehension with `dict.fromkeys` and combines four nested test context
chains while preserving entry/unwind order. Application code, measured calls,
assertions, fixtures, policy and dependencies are unchanged. Independent static
review found no actionable issue in that diff or its focused job manifest.

Job `librarian-20261004-literal-performance-002` passed the same 58 affected tests
in both environments, with no skips, plus scoped Ruff. Its exact source archive,
dependencies, tests, command, successful exit and cleanup were independently
verified. Source comparison confirmed that application code, measured operations,
fixtures, policy and dependency locks were unchanged. PERF-H3 is closed.
All performance, quality and full-suite evidence remains attributed to `51117e9`;
the focused repair evidence belongs to `5939208`.

The guard is automatic through the existing `ask`, search and read-only service
paths; it needs no new flag, schema migration or reindex. See
[technical literal usage](../USABILITY.md#technical-literal-search) for CLI and
MCP examples, including the `--` separator for a question such as `--force`.

## Evidence and limits

The sibling `readiness-results/librarian-20261004-literal-performance-001/`
directory preserves all 21 returned output files and 13 coordinator metadata
files, with `SHA256SUMS.json`. Source archive SHA-256:
`629837bcb1c572c7c94d9b5ab42984a9b5559f4178809644452fb6c54cd8e6fc`.
The raw performance report SHA-256 is
`4a53c8a5d83c817f3f431bc9784c0af59d1675505d37eafda75d818f24c5cd3f`.
The [frozen policy](PERFORMANCE-POLICY.md) and
[reproduction commands](PERFORMANCE-RUNBOOK.md) remain separate from these results.

The separate `readiness-results/performance-verification-20261004.json` audit
recomputed every timing block, checked 72 retained passages and 22 continuations
against authored text, and verified unchanged frozen outcomes across all three
paths. Source/archive custody, all 34 saved file hashes, dependency hashes,
test counts, resource limits and cleanup evidence matched. It initially retained
the Ruff gate as the adoption blocker; its additive focused-job verification now
closes that gate while preserving the original failed-job history.

The sibling `readiness-results/librarian-20261004-literal-performance-002/`
directory preserves nine returned output files and eight coordinator metadata
files with its own `SHA256SUMS.json`. Its exact source archive SHA-256 is
`ef624134557559aa679ff0f8f8b63914ef774d1d2c81849002712d204367bdc9`.
The verifier checked all 17 copies and the same 35/12 pinned dependency sets.
The 16.15-second focused job exited zero with cleanup verified and the container
absent. Its test results supplement the earlier full-suite evidence; they are
not an additional performance measurement.

Execution used the approved Python 3.12 ARM64 image, the existing 35-wheel full
and 12-wheel minimal hash locks, two CPUs and a 4-GiB limit. The benchmark took
162.44 seconds; the complete job took 212.28 seconds and cleanup was verified.
Bridge networking remained available throughout, with reviewed use limited to
dependency downloads and local test communication. No offline isolation claim
is made.

These are warmed, sequential, synthetic Linux measurements. There is no
macOS-runtime, cold-cache, concurrency, private-library, semantic/model or
production-capacity conclusion. The generated EPUB/PDF/SQLite files and full
source oracle were temporary and are not returned artifacts. All 3,360 complete
response objects were not retained individually: equality is supported by the
reviewed executed assertions and per-sample markers, with representative
responses and continuations retained. An artifact audit can verify raw timing
arithmetic, hashes and retained responses, but cannot independently reopen those
deleted corpora or replay every response.
