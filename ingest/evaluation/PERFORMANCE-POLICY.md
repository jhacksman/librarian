# PERF1: frozen literal-predicate experiment

This is a bounded synthetic investigation, not a latency SLA or a production
capacity claim. The previously measured broad miss took 561.04 ms median and
570.07 ms p95 over 4,096 FTS anchor matches; the 15-sample nearest-rank p95 is
also the maximum. That observation does not establish predicate CPU attribution.

## Candidate and invariants

A is the current correct literal predicate, as frozen at `c527e48` and unchanged
through the verified literal increment. B adds only a necessary-condition check:
after constructing the literal set and handling the empty set, reject if any
required literal is absent as an exact source substring. Substring presence never
accepts a match; the existing maximal-token lexer still decides acceptance.

Every accepted literal span is an unchanged source substring. The guard therefore
cannot remove a true match for the application's string inputs. Keep the original
set conversion so one-shot iterables are consumed once. Keep literal grammar,
case policy, Unicode offsets, normalization version, SQL filtering before LIMIT,
book filtering, BM25 order, citation fields, schema and index format unchanged.
There is no candidate cap or exception-to-abstention fallback.

The independent critic and two skeptics accepted this narrow hypothesis before
the sole fixer changes application code. The main counterexample to a speedup is
all substrings being present: up to one additional substring search per distinct
literal precedes the original lexer. Documentation must distinguish substring
probes from at most one maximal-token lexer pass.

## Frozen workload and measurements

Use only original synthetic EPUB/PDF data, existing reviewed packages and Spark.
Retain both existing case files byte-for-byte and report their unchanged gate and
diagnostic outcomes. Those cases are disclosed development cases now, not a new
blind set. No private sources, models, services or new dependencies are involved.

Use the existing deterministic background generator at 1, 8, 32 and 128 books.
The primary query is the absent literal `archive++` over ordinary background
text. A separate collision corpus replaces background `archive` words with
`archive++17`; this keeps the anchor common while making the substring present
inside a disallowed longer token. Record actual corpus hashes and anchor counts.
Use separate named probes for early/late positives, long Unicode excerpts,
qualified/flag/Unicode boundary rejects, multiple literals with an absent term,
128 distinct one-anchor flags (early, late and invalid-boundary variants), ordinary
queries, book filtering and a valid result behind many rejected candidates.

On each same database, run three A-B-B-A blocks with five timed service calls per
phase, giving 30 observations per version per query. Warm each version first.
Keep setup, result/citation validation, instrumentation, profiling and startup
outside these timings. Use fixed `PYTHONHASHSEED=0` because set iteration affects
which absent literal short-circuits first. Preserve raw order and measurements.
Compare exact service response equality including order, provenance, abstention,
citations and excerpt offsets, and verify continuation against the source oracle.
Restore patched predicate bindings even on failure.

Separate instrumented passes count predicate calls, accept/reject results,
guard-only rejections, predicate lexer calls, predicate input characters and
lexer input characters. Input length totals are not actual characters scanned.
Count only predicate lexing, excluding query planning and excerpt selection.
Separate standard-library profiles record process CPU attribution; their timings
are intrusive and must not enter the latency comparison.

## Predeclared decision rule

All correctness checks and existing regression gates must pass. At the largest
plain-background size, require B's median to improve at least 20% over A in every
paired block for the primary query. Treat a counterexample median regression
that exceeds both 5% and 1 ms in at least two of three blocks as a veto. These are
engineering acceptance thresholds, not a product SLA or statistical significance
test. Report all smaller changes, outliers and counterexamples.

Stop expansion on correctness mismatch, timeout or resource exhaustion; retain
partial context and measurements. A timing veto remains visible and blocks
adopting this minimal optimization. The final reviewer must distinguish exact
runtime observations, source-based reasoning and remaining generalization limits.

## Harness budget and checkpoint clarification

The job's first command, before dependency installation and regression checks,
records a monotonic deadline 510 seconds later. The harness uses the remaining
time from that deadline, reserving 90 seconds against the requested 600-second
outer job budget. This is a command-relative budget: the current coordinator
starts its outer clock before container startup and does not expose that exact
deadline inside the container. Do not claim strict knowledge of remaining outer
time. No coordinator privileges or runner changes are required.

Write an initial atomic checkpoint, then checkpoints after corpus setup, each
completed measurement phase, instrumentation and profiling. These writes occur
outside timed calls. Retain the previous valid report if a replacement is
interrupted, and retain non-passing/incomplete status until the full decision.
Checkpoints record their context, elapsed time and current memory snapshot. They
preserve the last completed step if the outer deadline or an OOM kills the
process; they cannot retain an interrupted phase or measure memory at the instant
of an OOM. Owned child-process hard-exit checks on Spark exercise this limitation
without deliberately exhausting memory.
