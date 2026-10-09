# Real stdio MCP verification

The application remains model-independent. `mcp_server.py` uses official
`mcp==2.3.0`; `requirements-mcp-ci.txt` pins its Python 3.12 dependency closure
alongside the already verified pilot requirements. The platform-specific
`requirements-mcp-linux-aarch64-py312.lock` additionally pins the SHA-256 of each
of the 35 wheels actually verified in Spark job003 at commit
`7d94997e541fe98af59c2b32484e1d418cd4f0ef`. It intentionally fails on an
unlisted platform artifact; prepare and review a separate lock for another
Python/platform combination. Optional CLI extras, model
libraries and telemetry exporters are not installed. Ordinary ingestion, CLI and
human preview remain usable with the minimal pilot dependencies.

## Dependency provenance and scope

Reviewed on 2026-10-04: [PyPI MCP 2.3.0](https://pypi.org/project/mcp/2.3.0/)
identifies the official [Python SDK](https://github.com/modelcontextprotocol/python-sdk)
with trusted publishing commit `2118f14f8a19bc158d8a1cf90af58d85d187f849`.
The universal wheel SHA-256 is
`dd0c44c089d16453e8ae31a3877a0054d7a2314caaa81f5e0541b9b1734b2377`.
API review used that immutable source, not an assumed old FastMCP interface.

Required packages include `mcp-types`, AnyIO, Pydantic, JSON Schema, HTTPX2,
Starlette/SSE, Uvicorn, PyJWT/cryptography and the OpenTelemetry API. The SDK
requires HTTP/auth packages even for stdio; this implementation does not start
those transports, configure OAuth, load exporters or make retrieval API calls.
HTTPX2/httpcore2 metadata points to the [Pydantic-maintained project](https://github.com/pydantic/httpx2).
The lock includes standard transitive dependencies; cryptography, CFFI and
rpds-py require compatible ARM64 wheels. `--only-binary=:all:` refuses source
builds. Record the resolver report and `pip freeze` in each Spark run.

All installation/test/lint execution is coordinator-only on Spark in a disposable
CPU container, no host ports, secrets, GPU, private books or live index. Network
access is requested only for the exact PyPI wheel installation. This request is
separate from pending model-download approval. No automatic SDK upgrades or
production environment installation are performed.

## Reproduce in the approved Spark job

From repository root, with Python 3.12 in the reviewed container:

```sh
python -m venv /tmp/librarian-ci-venv
mkdir -p ci-output
/tmp/librarian-ci-venv/bin/python -m pip install --only-binary=:all: \
  --require-hashes --report ci-output/install-report.json \
  -r ingest/requirements-mcp-linux-aarch64-py312.lock
/tmp/librarian-ci-venv/bin/python -m pip freeze > ci-output/pip-freeze.txt
export PYTHONPATH="$PWD/ingest/src"
export LIBRARIAN_EVIDENCE_DIR="$PWD/ci-output"
/tmp/librarian-ci-venv/bin/python -m pytest -q -c /dev/null -p no:cacheprovider \
  --junitxml=ci-output/tests.xml ingest/tests
/tmp/librarian-ci-venv/bin/ruff check ingest/src/ingest/local \
  ingest/src/ingest/__init__.py ingest/tests ingest/evaluation
/tmp/librarian-ci-venv/bin/python ingest/evaluation/evaluate.py \
  --output ci-output/retrieval.json
```

The ordinary minimal-dependency suite may skip `test_mcp_transport.py` when no
SDK is installed. The MCP verification job installs the lock and must report no
such skips. The transport transcript contains only original synthetic evidence;
never point this export fixture at the licensed collection.

The entry point accepts only `--index`; it does not offer an HTTP switch or
change the default human-preview binding. A client starts it as a subprocess
and owns its lifecycle. Do not install a client configuration that auto-downloads
packages or points a cloud model at the private index. The LAN design and its
still-pending decisions are in [LAN-PLAN.md](LAN-PLAN.md).

## Current verified usage and limits

The SDK client launched the real stdio server on Spark and exercised both
`2026-07-28` and legacy `2025-11-25` protocol modes. Job003 passed all 47 tests,
Ruff and the small synthetic retrieval/citation gate. Those results cover the
source at `7d94997`; they do not claim a persistent installation or LAN service.

With the reviewed runtime and a deliberately selected existing index, a trusted
local MCP client uses this executable and argument list (no shell expansion):

```text
command: /path/to/reviewed/python
args: ["-m", "ingest.local.mcp_server", "--index", "/path/to/index.sqlite"]
env: PYTHONPATH=/absolute/path/to/librarian/ingest/src
```

For direct human use of the same evidence service:

```sh
PYTHONPATH=ingest/src /path/to/reviewed/python -m ingest.local \
  --index /path/to/index.sqlite ask 'What are regular expressions?' --json
PYTHONPATH=ingest/src /path/to/reviewed/python -m ingest.local.web \
  --index /path/to/index.sqlite --port 8765
```

The second command is a foreground loopback preview and stops with Ctrl-C.
Starting private-library clients/hosting is not part of the synthetic verification.
Only approved local processing may receive licensed excerpts. No model download,
private-corpus reindex, authenticated LAN exposure or persistent service is
implied; see [USABILITY.md](USABILITY.md) and [LAN-PLAN.md](LAN-PLAN.md).
