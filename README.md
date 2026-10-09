# librarian
A LLM powered librarian that responsibly manages a library for Hackerspace members

## JavaScript application

The [single-machine application](app/README.md) is the active product work:
resumable imports, a browser catalog and reader, and cited local AI on one Spark.
The [October 9 sprint report](docs/sprints/2026-10-09.md) records published
functionality, verified checks, deployment status, and prioritized remaining work.
It distinguishes historical real-library receipts from synthetic regression tests.

## Local ingestion pilot

See [the offline pilot guide](ingest/PILOT.md) for tested EPUB/PDF ingestion,
metadata, traceable chunks, and cited SQLite search without API calls. The legacy
LLM/Qdrant pipeline remains unfinished.

The isolated readiness branch adds [resumable ingestion and recovery](ingest/READINESS.md),
with all milestone validation dispatched through the Spark CI coordinator.
It also provides [cited questions and technical literal search](ingest/USABILITY.md#technical-literal-search).
The [verified literal-search performance results](ingest/evaluation/PERFORMANCE-RESULTS.md)
record the accepted optimization, correctness checks and measured limits.
Search and catalog responses also expose recorded PDF extraction omissions;
see the [verified coverage results](ingest/evaluation/COVERAGE-RESULTS.md).
The working pilot returns cited source evidence. Semantic retrieval, generated
answers, wider private-library validation and LAN deployment remain separate work.
