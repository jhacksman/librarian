# Reproducing the bounded literal experiment

Run this experiment only through the reviewed Spark coordinator job. The frozen
workload and adoption thresholds are in [PERFORMANCE-POLICY.md](PERFORMANCE-POLICY.md).
The experiment compares the correct `c527e48` predicate with the candidate on
each same synthetic database. It does not read the private library.

The coordinator manifest installs the existing binary hash-locked full and
minimal dependency sets, runs both complete test suites, scoped Ruff and the
original seven-case evaluation, then runs the commands below. Its first step,
before installation or tests, creates the budget file:

```sh
mkdir -p ci-output
python - <<'PY'
import json
import time
from pathlib import Path

started = time.monotonic()
Path('ci-output/performance-budget.json').write_text(json.dumps({
    'command_started_monotonic': started,
    'deadline_monotonic': started + 510.0,
}, indent=2) + '\n')
PY
export PYTHONHASHSEED=0
```

After the reviewed environment setup and preceding checks, use the full
environment's Python with `PYTHONPATH` pointing to `ingest/src`:

```sh
python ingest/evaluation/evaluate_quality.py \
  --warm-repetitions 3 --output ci-output/quality.json
python ingest/evaluation/evaluate_quality.py \
  --profile query-comparison \
  --cases ingest/evaluation/lexical-heldout-cases.json \
  --warm-repetitions 3 --output ci-output/heldout.json
python ingest/evaluation/evaluate_literal_performance.py \
  --output ci-output/literal-performance.json \
  --budget-file ci-output/performance-budget.json \
  --quality-report ci-output/quality.json \
  --quality-report ci-output/heldout.json
```

The actual coordinator command accumulates failure status across checks and
retains all outputs. The old unpaired `--literal-cost-probe` is unnecessary here.
Do not reset the budget before the paired harness, omit a failed correctness
check, or select a faster rerun to replace a veto. The two quality case files
remain unchanged; they are disclosed regression checks at this stage.

Keep the complete report, raw samples, dependency reports, source archive hash,
test and lint results, container limits, exit status and cleanup proof together.
The paired report records source and fixture hashes, SQL anchor counts, exact
responses and continuations, separate predicate counters and CPU profiles, and
the predeclared decision. A successful performance decision requires separate
green regression checks and independent verification before adoption.

An incomplete or failed report cannot support adoption. Atomic checkpoints retain
completed phases; a hard stop can lose the interrupted phase. The 510-second
deadline is relative to command startup, with a nominal 90-second reserve against
the 600-second outer job. The runner's earlier archive/container startup clock
is not exposed, so inner-first timeout is not guaranteed. Results describe warmed,
sequential, synthetic Linux ARM64 calls only.
