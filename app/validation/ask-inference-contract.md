# Coordinated local Ask inference

Run only in the existing Spark coordinator with its isolated, writable clone of
current qualified205 data and managed original/native-review source custody.
The job root must be a dedicated ordinary /tmp directory, its standalone snapshot an ordinary non-symlink file with no existing sidecars, and its output a fresh directory. Saves are exclusive. Prepare/import approved source bytes with the existing coordinator before this
helper. The helper performs no imports, indexing, launches, downloads or probes.
It must not open the live pilot database. No execution is authorized on M6.

The coordinator runs:

```sh
LIBRARIAN_SPARK_COORDINATED=1 node app/validation/ask-inference.mjs /bound-inputs/ask-input.json
```

Input shape (paths/endpoints supplied by the coordinator's approved job):

```json
{
  "isolatedSnapshot": true,
  "jobRoot": "/tmp/librarian-bound-job",
  "dbPath": "/tmp/librarian-bound-job/library.sqlite",
  "outputDir": "/tmp/librarian-bound-job/evidence",
  "maxSeconds": 600,
  "retrieval": {
    "provider": "openai",
    "embeddingModel": "librarian-qwen3-embedding-q8",
    "embeddingRevision": "approved immutable runtime:model revision",
    "chatModel": "librarian-bonsai2-ptq1",
    "chatRevision": "approved immutable runtime:model revision",
    "contextTokens": 8192,
    "timeoutMs": 45000,
    "queryPrefix": "Instruct: Retrieve book passages that directly support an answer to the question.\nQuery: ",
    "documentPrefix": "",
    "embedding": { "provider": "openai", "endpoint": "http://127.0.0.1:18093/v1" },
    "chat": { "provider": "openai", "endpoint": "http://127.0.0.1:18094/v1" }
  },
  "bindings": [
    { "role": "isolated-snapshot", "path": "/tmp/librarian-bound-job/library.sqlite", "sha256": "sha256 of exact standalone snapshot" },
    { "role": "source-native-custody", "path": "/bound-inputs/source-custody.json", "sha256": "sha256 of exact file" },
    { "role": "model-runtime-contract", "path": "/bound-inputs/model-contract.json", "sha256": "sha256 of exact file" }
  ],
  "cases": [{
    "id": "P01",
    "question": "Explain the code and following failure discussion.",
    "bookId": "exact content identity from approved catalog",
    "readerContext": { "bookId": "same exact content identity", "sectionId": "resolved source section", "scope": "section", "page": 184 }
  }]
}
```

This example does not qualify or reserve its ports. The coordinator binds existing
approved Bonsai and Qwen runtime contracts, endpoint choices and tokenizer/context
support. Historical contracts/aliases must be revalidated by that coordinator.
The helper enforces an 8192 configured context and keeps the existing conservative
6480-byte prompt admission; it makes no expanded tokenizer/context claim.

Finite bounds: 1–12 cases, no retries, at most 32 embedding/chat transport calls,
512KiB per raw response, 2MiB per artifact, 512KiB inputJSON, 2GiB standalone snapshot binding and 16MiB other binding receipts (3GiB total streamed hashes), 600 seconds total by default (explicit
30–1200 seconds accepted), and app timeout 45 seconds per request by default. Source
prompts retain max 24,000characters and max 12 citation spans; output max 1200tokens
and max 12,000characters. Inputs must contain unique case IDs and question/source
scope only. Expected claims, suggested answers and native gold must stay outside
this input; native/source custody bindings are hashed and never parsed into the
prompt. Every case uses responseMode:auto.

Required evidence for acceptance:

- At least five useful real-book synthesized answers from the independent critic's
  frozen supported cases, including LearningSQL counts/nulls/distinct/CASE,
  InterviewPrep Promise.all185–186 with visible anchor184, and ProBash echo/printf.
- Independently inspect every substantive output claim and citation against exact
  native original source, edition and page/member. Check whether admitted source
  text is coherent and complete enough for each requested condition. Exact quote
  membership is a source receipt, not semantic entailment.
- Real-model negative evidence: absent invented secret, unsupported source/order
  claim, and instruction injection in source text. Synthetic unit transport tests
  do not qualify these behaviors. Wrong-source/page, invalid citations/quotes,
  midflight source replacement, abort and unavailable model paths have CPU tests.
- Confirm engine/runtime identity and returned usage/finish reason from coordinator
  runtime observation plus original raw requests/responses. A dispatched HTTP
  request alone does not prove engine execution. The helper never marks semantic
  or real-book quality acceptance true; independent review supplies that decision.
- Bind source archive, helper, CPU results, isolated database before/after audit,
  originals, approved model/runtime bytes, output manifest and cleanup receipts.
  Existing historical Bonsai001/002 failures and003 synthetic-only success remain
  unchanged; ranked-context replay with stub abstention is not inference evidence.

Artifacts include input.json; per-call original request/response bytes; each Ask
result with retrieval hits, sent citation spans, synthesis, support quotes,
validation/warnings/usage; and report.json with elapsed timings, errors and file
hashes. Artifact timestamps or citation syntax never establish grounding.

CPU command through the same coordinator:

```sh
node --test app/test/config.test.mjs app/test/retrieval.test.mjs app/test/answer-support.test.mjs app/test/context-allocation.test.mjs app/test/extractive-response.test.mjs app/test/named-source-coverage.test.mjs
```

Broader server/product checks belong to the root coordinator packet. Source work
and whitespace review alone do not establish that these tests pass.
