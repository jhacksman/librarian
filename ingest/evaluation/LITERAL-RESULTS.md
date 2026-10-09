# Literal-query comparison, 2026-10-04

The frozen candidate resolves all 14 technical diagnostics in the independent
synthetic set; the previous implementation resolves seven. Both pass the six
ordinary-word and book-filter controls. This is evidence for the narrow ASCII
literal policy, not representative library or semantic retrieval quality.

The increment's verification is complete. Full regression tests passed at
`b40133e`: **86 passed** with MCP, and **76 passed with 10 expected SDK skips**
in the minimal environment. The final fixture-import repair at `4afea3e` passed
all 25 focused tests with MCP, 23 tests with two expected SDK skips without it,
and the complete scoped Ruff check. Independent artifact audits confirmed the
exact commits, dependency locks, evidence custody and container cleanup.

The first candidate job failed collection because of an invalid test decorator
colon; its independently completed retrieval evaluations remain valid. The
second job fixes that syntax and verifies the full tests, but exits one for the
fixture-import lint issue. Preserve these outcomes rather than replacing them.

## Source and evidence identity

| Item | Immutable source |
| --- | --- |
| Frozen policy | `e0a373d595337ee23bee66456415d72d4b93c4ea` |
| Frozen application candidate | `c527e480330242fdc81ac656c4900d44bccead51` |
| Candidate evaluation job | `librarian-20261004-literal-candidate-001`, commit `fcb05432b95e480c7bfa50748643e8cffba3d8cc` |
| Control evaluation job | `librarian-20261004-literal-control-001`, commit `f43a42a0cd53dc671fd2b059a85690bb1eec8e8b` |
| Control application | Identical to `c0afb83ec47e21966f7734bb8c2962e1db8a8014` |
| Test syntax repair and full tests | `b40133e4b58671ebb92332e7843de6476ada6422`, candidate002 |
| Fixture declaration repair and final lint | `4afea3ecb840da0bae56127b41b70f1b9461ca0e`, candidate003 |

The implementer opened the independent cases only after application freeze and
critic review. Their SHA-256 remains
`4327aa1c0463f49c719402411e5b4ad4b3f653144bc55e9cae7304602569b721`.
The same evaluation code, labels, dependency lock, image and resource limits
were used for both versions. Application code did not change after disclosure.
The repairs remove an invalid test decorator colon and declare the reused pytest
fixture as an explicit re-export. Neither changes test assertions, application
code, evaluation fixtures or evaluation logic.

Both runs used Spark2 Linux ARM64, two CPUs, 4 GiB and the reviewed Python 3.12
image with the 35-wheel SHA-256 lock. Both containers were confirmed absent
after completion. The control job exited zero and passed all 14 evaluator tests.
Candidate001 exited one for the test syntax error. Candidate002 exited one solely
for Ruff F811 after all tests passed; its container cleanup was also verified.

Raw artifacts, approved manifests, logs, checksums and review records are kept
outside Git in the Librarian folder's `readiness-results/` directory:

- `librarian-20261004-literal-control-001/`
- `librarian-20261004-literal-candidate-001/`
- `librarian-20261004-literal-candidate-002/`
- `librarian-20261004-literal-candidate-003/`
- `literal-verification-20261004.json`
- `literal-comparison-20261004.json`
- `literal-review-20261004.json`
- `literal-status.json`

## Independent synthetic outcomes

Each version generated the same 136 books, 983,215 words and 4,117 chunks:
eight labeled EPUBs plus 128 background EPUBs. Corpus manifest SHA-256 is
`c35c85386495f3e3ac2466cd9b7afcd2752eb1cadafa9acd891e43f9f0ce3800`.
Index size was 27,004,928 bytes. No private ebooks or existing index were used.

| Measure | Control | Candidate |
| --- | ---: | ---: |
| Ordinary/filter gates passing | 6/6 | 6/6 |
| Technical diagnostics passing | 7/14 | 14/14 |
| Source-span recall@3, 16 positive questions | 0.96875 | 1.0 |
| Reciprocal rank@3, positive questions | 1.0 | 1.0 |
| Correct no-match labels | 3/4 | 4/4 |
| Questions with extraneous hits | 7 | 0 |
| Verified returned citations per path | 30 | 19 |

Service, modern MCP (`2026-07-28`) and legacy MCP (`2025-11-25`) agree on every
case within each version. All returned citation identities/spans and book filters
were verified. Catalog enumeration found all 136 identities and both database
hashes remained unchanged. Fewer returned citations reflect removal of unrelated
hits, not loss of labeled evidence.

The seven improved diagnostics cover `F#`, `--dry-run`, `-q`, `mesh::paint`,
`retry_slot`, `queue.ready`, and the absent shorter flag `--dry`. Labels and the
gate/diagnostic distinction were retained even when the old implementation failed.
These cases are now disclosed development cases; another blind evaluation would
need separately authored, sealed material.

## Cost and remaining quality limits

The separate `archive++` probe has 4,096 matches for its broad FTS anchor
`archive`. The old implementation discards punctuation and returns ten unrelated
hits. The candidate retains the literal constraint and correctly returns no hit.
Across 15 sequential service calls, p95 was **78.04 ms** for the control and
**570.07 ms** for the candidate. This exposes the cost of literal verification
when an anchor is common. There is no hidden candidate cap or latency guarantee.
The reported candidate count is SQL anchor cardinality, not visited-row
instrumentation; the raw shared scope wording must not imply that the old
implementation performed the same literal scan.

The 60 labeled warm service samples had p95 20.28 ms for the control and 11.97 ms
for the candidate. These are descriptive observations from one sequential run
per version, not a statistical speedup or concurrency-capacity claim. The first
pass follows ingestion and is not cold-cache measurement. Whole-container memory
peaks include different preceding checks and cannot isolate application changes.

The candidate also passed the original seven-case citation exercise and all 16
gates in the unchanged 22-case default quality suite. Its `C++` and `--force`
collision diagnostics now pass. Three existing recovery diagnostics still fail:
keyword ranking returns extra evidence, a natural question has a word-form
mismatch, and a paraphrase retrieves no evidence. No stemming, synonyms,
embeddings, reranking or generated answers have been added.

## Verification closure

Independent critic and two skeptics confirmed L2, the test-collection defect,
before the sole fixer removed the colon. Candidate002 executed all 86 tests in
both environments at `b40133e`: full 86 passed, minimal 76 passed with 10 explicitly
optional-SDK skips, zero failures/errors. The separate verifier closed L2.

The same review process confirmed L3, Ruff F811 for the imported pytest fixture.
The sole fixer declared `from test_mcp_transport import sdk as sdk`, preserving
pytest's existing fixture binding. Job
`librarian-20261004-literal-candidate-003` passed the affected `test_query.py`
in both environments and scoped Ruff at `4afea3e`, exited zero, and verified
container cleanup. The independent verifier closed L3. No application or
benchmark rerun was needed solely for this import declaration.

Successful application evaluations remain attributed to candidate001, full
regression tests to candidate002, and final fixture/lint verification to
candidate003. See [README.md](README.md) for reproducible
coordinator-run commands and [QUERY-POLICY.md](../QUERY-POLICY.md) for the exact
supported grammar.
