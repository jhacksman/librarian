# Reusable retrieval evaluation contract

Current runnable baseline: `synthetic-cases.json` + `evaluate.py`. All validation
executes on Spark through the coordinator. This is original synthetic material;
no licensed corpus text or model-generated substitute for an original book.

The 2026-10-03 research round reopens model and method selection. Existing BGE
and Qdrant configuration is a historical baseline. This contract does not select
an embedding, reranker, answer model, vector store, or hardware execution plan.
No evaluation should implicitly download a model or migrate the real index.

## Existing fixture format

```json
{
  "provenance": "Original synthetic fixture; author/source and reuse terms",
  "documents": [
    {"id": "stable-document-label", "title": "Synthetic title",
     "sections": ["Original first-section text", "Original second-section text"]}
  ],
  "queries": [
    {"query": "question or search terms", "document": "stable-document-label", "section": 1},
    {"query": "unsupported question", "document": null}
  ]
}
```

Sections use one-based ordinals. `document: null` is an expected abstention.
The harness creates bounded temporary EPUB fixtures, ingests them, maps document
labels to source SHA-256 book IDs, searches with k=3, and independently checks
returned excerpt offsets against extracted source sections. It records per-query
rank/hit count, recall@3, MRR@3, correct negative abstentions, and verified citations.
Current coverage is six single-section positives and one no-match negative;
passing it is only a regression gate, not a library-quality conclusion.

## Synthetic scale and adversarial evidence exercise

`evaluate_quality.py` adds a separate development evaluation through
`LibraryService.ask` and the actual MCP stdio client/server in both reviewed
protocol modes. The original seven-case baseline remains unchanged. Labels in
`quality-cases.json` are authored before execution: 22 questions, including 16
hard contract cases and six diagnostic quality cases. The goldens name an exact
document, physical PDF page or EPUB section, and verbatim source span. Duplicate
hits cannot increase source-span recall; an excerpt from the wrong section or
only part of the golden span cannot receive credit.
Every citation ID must also exist in the indexed-chunk snapshot and agree with
that chunk's source section and containing span. A plausible ID/URI alone does
not establish citation resolution.

The eight labeled documents include conflicting editions with identical titles,
duplicate passages in distinct books, printed-versus-physical page numbering,
Unicode offsets, and quoted instruction text. Negative cases include words that
occur separately in the corpus but never in the same passage. The instruction
fixture must remain quoted evidence; this does not measure an answer model's
resistance to prompt injection. No model is involved.

Ranking, paraphrase, natural-question rewriting and punctuation collisions are
diagnostic cases. Their misses and extraneous hits remain in the report and do
not make a successful contract gate a semantic-quality claim. Report source-span
recall@3, MRR@3, precision over returned hits, no-match results and per-category
failures. A returned excerpt is retrieval evidence, not proof of answerability.
All returned citations, book filters, evidence-only responses and cross-transport
parity are hard checks, including for diagnostic cases. The gate/diagnostic split
must not be changed after a failed run merely to obtain a green result.

The fixed seed `20261004` produces 128 additional EPUBs, each with 16 sections of
480 words: 983,040 background words plus the labeled documents. This is a volume
exercise, with repeated synthetic vocabulary, not a representative library or a
held-out benchmark. ZIP timestamps are fixed and generated EPUB/PDF bytes are
checked for reproducibility. The report records source hashes, a corpus manifest,
parser/chunker version, actual words/chunks, source/index bytes and generation
plus ingestion time. Background query `archive` exercises common-term retrieval
with the maximum ten hits. No private sources or existing index are read.

Additional checks reject 17 malformed or oversized tool-argument cases through
the service and each MCP mode, then verify MCP recovery. Three valid input limits
are also exercised. Physical-page and Unicode passage starts/final characters
are checked, and one-past-the-end offsets rejected. Catalog pagination must
preserve all identities even when titles are ambiguous. These are tool-argument
tests, not raw protocol framing fuzzing. The index hash must remain unchanged
after all read-only requests.

Measurements use sequential first-pass and three warm repetitions per question,
nearest-rank p50/p95 and raw per-question timings. The first pass follows ingestion
and is **not** cold-cache measurement. Driver peak RSS includes the generator and
source oracle; exited-child peak RSS is reported separately, not added to it.
When available, cgroup memory peak and CPU/memory/PID limits are recorded. Cgroup
memory covers the entire CI container, including installation and prior checks.
There is no production latency SLO or concurrency-capacity claim.
MCP timings include the client call plus response validation; service timings
cover only the service method. Their difference is not pure transport overhead.
Phase reports are attached before work starts and populated incrementally. If a
later check fails, the JSON retains completed rows and timings with explicit
phase/check/case context. Fault-injection tests verify this for service and MCP.
The three full-harness integration tests skip when the optional MCP SDK is
absent; eleven fixture/scoring/profile checks still run with the minimal pilot packages.
The full MCP verification environment must run all fourteen with zero skips.

Run only through the coordinator in the reviewed Spark container, after installing
the existing 35-wheel hash lock. No dependencies are added:

```sh
export PYTHONPATH="$PWD/ingest/src"
/tmp/librarian-ci-venv/bin/python -m pytest -q -c /dev/null -p no:cacheprovider \
  --junitxml=ci-output/quality-tests.xml ingest/tests/test_quality_evaluation.py
/tmp/librarian-ci-venv/bin/ruff check ingest/evaluation ingest/tests/test_quality_evaluation.py
/tmp/librarian-ci-venv/bin/python ingest/evaluation/evaluate_quality.py \
  --warm-repetitions 3 --output ci-output/quality.json
```

The bounded job requests two CPUs, 4 GiB, no GPU, no host ports and a ten-minute
container timeout. Corpus and index are temporary and removed when the harness
exits. Only synthetic evidence and measurements are returned. The unchanged
application’s previous 47-test result remains attributed to its original commit;
this job verifies the new evaluation harness and additional workload.

## Independent literal-query comparison

The literal policy was frozen in `e0a373d595337ee23bee66456415d72d4b93c4ea`;
application candidate `c527e480330242fdc81ac656c4900d44bccead51` was committed
and independently reviewed before its implementer opened the separate cases.
`lexical-heldout-cases.json` was copied unchanged from the sealed author output:
SHA-256 `4327aa1c0463f49c719402411e5b4ad4b3f653144bc55e9cae7304602569b721`.
The companion notes have SHA-256
`3a79e537e2b36754b50f9fb49a8af96f2dd58f5adbc6cdc8a57bcd62a2bb33f9`.
They contain 20 questions: 14 technical diagnostics and six ordinary/filter
contract controls over eight original synthetic EPUBs. The labels are unchanged.
These independent model-authored cases are not a representative or human-authored
benchmark. They become disclosed development cases after this initial comparison.

An alternate fixture requires an explicit profile. It retains all labeled-case
scoring, citation verification, service/MCP parity, protocol discovery/status,
catalog pagination, common-term timings and database immutability. Default-only
passage-boundary and invalid-request probes are explicitly recorded as `not_run`.
Run the default full profile separately to retain those checks. Optional
`--literal-cost-probe` records 15 service calls for `archive++`, the number of FTS
`archive` candidates, returned-hit count, normalization and raw timings. This
observation has no expected answer or latency gate and does not affect labeled
quality metrics; it measures the cost of scanning broad literal candidates.

```sh
/tmp/librarian-ci-venv/bin/python ingest/evaluation/evaluate_quality.py \
  --profile query-comparison --cases ingest/evaluation/lexical-heldout-cases.json \
  --literal-cost-probe --warm-repetitions 3 --output ci-output/heldout.json
```

Run baseline and candidate only through the Spark coordinator with identical
harness/cases, background seed and count, dependency lock, image and resources.
Record each exact source commit. Baseline normalization metadata is unavailable
and recorded as null; candidate metadata records the actual transformation.
Retain all diagnostic misses. Single sequential runs support descriptive timing
comparisons, not a statistical performance conclusion. Runtime results are not
established merely by freezing this source.

## Adapter seam used by human and agent entry points

The model-agnostic `Retriever` protocol in `src/ingest/local/service.py` defines:

- `search(query, limit, book_id)` returns ranked passage dictionaries.
- `passage(chunk_id, offset)` returns one bounded indexed excerpt or no result.
- `books(limit, offset)` returns indexed identity/title/chunk-count records.
- `coverage()` returns indexed book/chunk counts.

Passage fields: `chunk_id`, `book_id`, `title`, `quote`, `chapter`, `page_start`,
`page_end`, `epub_member`, `source_sha256`, `char_start`, `char_end`, `source_uri`,
`content_kind: source_excerpt`. Character offsets refer to exact extracted section
text, not raw PDF bytes or printed page labels. The service supplies citation IDs
and an explicit evidence-only/abstention response, shared by CLI, UI and MCP tools.
The lexical baseline rewrites filler words; a semantic adapter receives the
original question. Record transformations so methods are compared transparently.

Do not mix document parsing/chunk changes with embedding changes in an unlabelled
comparison. Freeze source hashes, extraction version and chunking version first.
If a candidate changes chunking, compare source-span coverage as well as chunk IDs.
Preserve source citations through fusion, reranking, filtering and answer assembly.

## Proposed broader comparison suite (not yet executed)

Use a frozen query/label file with stable query IDs, question wording, task type,
expected relevant source spans, required book filters, and expected abstention.
Include exact lookup, paraphrases, synonyms, ambiguous queries, cross-book
comparison, multi-evidence questions, mixed technical prose/code, negatives,
near-miss topics, and edition/version ambiguity. Add PDF page-label and EPUB
location checks. Use original synthetic or verified public-domain documents for
portable public tests. Private licensed corpus evaluation stays local and is
never copied into a public benchmark or sent to an external model.

For each candidate record:

- Full model/revision, tokenizer, query instruction, normalization, dimension,
  distance, precision/quantization, software lock, hardware and resource budget.
- Corpus/source hashes; parser/chunker version; candidate k; hybrid/fusion and
  reranker configuration; any query rewriting or expansion.
- Recall@k, MRR/nDCG where labels support them, book-filter correctness,
  no-answer precision/recall, and source-span/citation correctness.
- Warm/cold latency, memory, disk/index size and indexing throughput, measured
  only on the approved Spark job—not estimated or executed on M6.
- If synthesis is added: evidence sufficiency, supported-claim coverage,
  unsupported claims, citation fidelity and calibrated abstention. Retrieval
  matches alone must not be scored as correct generated answers.

Split development and held-out questions; review labels independently before
model comparison. Do not tune thresholds on the held-out set or present the
seven easy baseline questions as representative. Approval for a candidate's
package/model execution is separate from proposing its comparison configuration.

## Research-informed comparison plan (2026-10-03; not executed)

This plan incorporates the separate *Librarian architecture research for one
DGX Spark* report dated 2026-10-03. Its candidates are hypotheses to evaluate,
not installed components or a production selection. The source report is retained
in the research task as `Librarian-research-2026-10-03.md`.

| Stage | Controlled comparison | Gate before expansion |
| --- | --- | --- |
| Text control | FTS5/BM25 alone; historical BGE base English v1.5 alone | Same immutable source spans and chunks; verify prompt, pooling, normalization and citations |
| Embedding challengers | Qwen3-Embedding-0.6B and Voyage 4 Nano independently; 512/1024 dimensions as separate configurations | Model/runtime approval and full revision locks; no live-index replacement |
| Fusion | Each dense candidate alone versus the same candidate plus BM25 reciprocal-rank fusion | Identical filters and candidate budgets; preserve exact symbol lookup and deduplicate overlapping evidence |
| Reranking | Best hybrid configurations with/without Qwen3-Reranker-0.6B | Separate approved model run; start at 30 candidates and retain original questions |
| Parsing | Current extraction versus Docling on about 40 manually checked difficult pages | Only if observed extraction failures explain missed evidence; separate parser/model approval |
| Visual retrieval | Selective Qwen3-VL page/crop retrieval on 30–50 difficult pages | Only if table/figure questions still fail; separate vector space and approval; never fabricate quotations from page relevance |

Candidate references from the research: [BGE control](https://huggingface.co/BAAI/bge-base-en-v1.5),
[Qwen3 embedding](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B),
[Voyage 4 Nano](https://huggingface.co/voyageai/voyage-4-nano),
[Qwen3 reranker](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B),
[Docling](https://github.com/docling-project/docling), and
[Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B).
Verify exact cards, licenses, full commits and ARM64/GB10 runtime compatibility
when preparing the approval packet. Do not silently use mutable `main`, accept
remote custom code, or infer runtime compatibility from a model family name.
No new installation, model execution or private corpus transfer is authorized.

First propose 30 development questions on the existing two-book sample as a
plumbing gate. After separate scope approval, choose eight representative books
by format/topic (including matched editions where available) and prepare 120
human-authored questions: 30 paraphrase/concept, 25 exact symbol/code/error,
20 table/math/figure, 15 comparison/multihop, 10 edition/negation traps, and
20 unanswerable/out-of-coverage. These are proposed quotas, not existing goldens.
Split 60 development/60 held-out, preferably by chapter/book; a non-tuning human
keeps held-out questions and source-span goldens sealed. Blindly adjudicate pooled
results from all candidates and recheck negative labels against the frozen corpus.
Do not send licensed source passages to a cloud model to generate or judge tests.

Use evidence-span Recall@5/10/20, graded nDCG@10 and MRR@10 by question class and
format. Add literal identifiers (`std::vector`, `foo_bar`, `--flag`, `C++`) to the
exact-match evaluation: FTS tokenization alone does not establish symbol fidelity.
Keep extraction/chunking fixed in embedding comparisons; evaluate new structural
chunking, parent expansion, rewriting and quantization as separate interventions.
Preserve the original question alongside any rewrite and retain original source
spans separately from generated context.

Measure cold/warm p50/p95 latency, embedding throughput with token lengths,
rerank pairs/second, peak process/system memory, index bytes, build time and
restore correctness on the approved Spark job. Compare paired per-question
outcomes with uncertainty; a leaderboard score or seven easy synthetic queries
cannot select the library's winner. Freeze the answer model and evidence budget
if synthesis is later approved, and separate retrieval evaluation from answer
faithfulness using oracle evidence. Citation resolution is not claim support.

Proposed acceptance targets need agreement before execution: 100% tested citation
resolution and restore, no outbound source content, at least 90% abstention on
unanswerable cases, at least 95% manually supported generated claims if synthesis
is enabled, no material exact-symbol regression, and a meaningful paired gain
within an agreed latency/memory budget. The report proposes a 3-second retrieval
plus reranking p95 and a 24 GB process budget; neither is a measured result or an
approved resource reservation. Small-sample percentages remain exploratory.

The approval packet must name exact model/tokenizer/code revisions and artifact
hashes, runtime/container/package locks, licenses, selected books, isolated
cache/index paths, network policy, resource limits and coordinator time slot.
Preserve the active lexical index and a restorable generation. Larger ingestion,
production model changes and authenticated LAN hosting are separate decisions.
The locality requirement also applies to MCP callers: a local server connected
to a cloud model would transmit retrieved passages outside the local system.
