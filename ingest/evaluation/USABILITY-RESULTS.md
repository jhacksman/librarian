# Scoped browsing verification (U1)

Spark job `librarian-20261004-usability-001` tested application/test commit
`3754a4f619641b942ee8104a257a138b7f855292`. It returned exit 0. Separate independent
verification passed and closed this bounded gate with no verification blockers.
The counts below come from returned JUnit and evidence files.

The web preview now lets a reader browse 20 catalog entries at a time, select a
book by its full identity, ask within that book and continue cited excerpts while
preserving typed question text and scope. `list_books` gains an optional exact
book filter while retaining its existing envelope and the four MCP tool names.
The [frozen contract](../USABILITY-POLICY.md) governs this interface change.

| Check | Returned result |
| --- | --- |
| Full profile | 257 passed, no skips or failures |
| Minimal profile | 243 passed, 14 expected optional-SDK skips, no failures |
| New U1 module | 53 full passes; 51 minimal passes and two SDK skips |
| Scoped Ruff | Passed |
| Dependency checks | Existing 35-wheel full and 12-wheel minimal profiles; both `pip check` outputs clean |
| Existing tiny retrieval gate | Six positives at rank 1, one correct no-match, six verified citations |
| New synthetic evidence exercise | Eleven gates pass; one separately retained paraphrase diagnostic misses |
| Real MCP modes | Auto protocol `2026-07-28` and legacy `2025-11-25`; 18 recorded calls each |

Each profile reported the same five existing Pydantic deprecation warnings. The
204 previously passing test cases remain in the full roster. There was no large
PERF1, default-22-query or held-out-20-query standalone benchmark rerun.

## What the new evidence establishes

The independent oracle predates the application implementation. Its three
original synthetic books include same-title PDF editions with different intervals,
a blank physical PDF page, and an EPUB. The twelve scenarios distinguish useful
retrieval from answerability, exact technical literals, selected-book misses,
cross-book conflict and extraction coverage. The paraphrase still has a relevant
authored source and produces no lexical match; it is not relabeled as a success.

Actual loopback HTTP checks cover catalog selection, pagination with unsent text,
full IDs, scoped errors/no-match, return to all books, Unicode excerpt continuation,
escaping, Host/Origin/framing checks and source-file removal. Read-only database
bytes remain unchanged. CLI and both real SDK transports return the expected
source-bound evidence and coverage. Each new MCP transcript has twelve successful
payloads and six invalid-filter error markers.

The retained HTTP-flow JSON contains five service-reference answers after equality
assertions against parsed HTML; it is not a raw HTTP transcript. The test also
asserts return-to-all and continuation behavior without retaining those response
bodies. The two existing `human-preview-synthetic.html` files are separate rendered
preview evidence. The subsequent narrow walkthrough run is recorded below; visual evidence
is not attributed to the original U1 test run.

## Source, runtime and artifact identity

- Source archive SHA-256:
  `bf6b09b9b9113e7deef360a02ab36b26f99ebbf8881d45b0a419fe883b609f2e`.
- Returned `ci-output` archive SHA-256:
  `da71b6b81afbe95851994f0051fca863167f9fc4a1d7f5aee98f67aa4f402163`.
- Independent oracle SHA-256:
  `e750904e58a8a180e07dae4c84e2cfdd0cf1fc6a26d5a082472ab862652aff52`.
- Python image:
  `sha256:ee2c320efc696d510c4579d9d40d5e2ece061a6cd0adc4b966db84b581bc8be0`.

The job ran on `gb10-02`, CPython 3.12.15 / Linux ARM64, with 2 CPUs, 4 GiB,
512 PIDs, a 600-second limit and no GPU. The reported interval was 46.190 seconds,
including setup; it is not a
retrieval benchmark. Bridge networking remained available for the whole container.
The result records verified cleanup and container absence. No private corpus,
models, credentials, host port publication or persistent service was involved.
The conditions sample labeled "during" was taken after completion and supplies
no evidence about concurrent host load.

There are 33 returned output files. Exact ordinary tar members were copied as
inert bytes and checked against the coordinator's member hashes. Local custody,
metadata and separate verification records live outside Git under
`20261004/librarian/usability-results/librarian-20261004-usability-001/` and
`20261004/librarian/usability-verification-20261004.json` in the m6mini project.
The separate verifier's closed report has SHA-256
`1cb2d141d072ec78688e7d7b0f69261bd72eb8968ef9225b7c0596f996c088e2`.

For focused reproduction in the already reviewed Spark environment, with
`PYTHONPATH=ingest/src` and optional `LIBRARIAN_EVIDENCE_DIR` set:

```sh
/path/to/reviewed/python -m pytest -q -c /dev/null -p no:cacheprovider \
  ingest/tests/test_usability.py
```

This result does not establish visual usability, model quality, private-library
quality, OCR or extraction completeness, live client registration, LAN deployment,
concurrent load or new performance measurements. Prior C1 and PERF1 evidence keeps
its original source attribution; the three earlier recovery diagnostics remain
unresolved. The working pilot remains lexical and evidence-only.


## Synthetic visual walkthrough

The follow-up capture job `librarian-20261004-demo-capture-001` used commit
`7b75555bbdf4485f7a16815521994c365a327027`. Its one existing HTTP-flow test and
scoped Ruff check passed. The change added capture output to that test; all 40
application source files remained identical to verified U1 commit `3754a4f`.
Five exact HTTP response bodies were retained after the source-file removal,
form-state, quotation and unchanged-database checks.

The separate browser job `librarian-20261004-demo-browser-001` rendered those
bodies from synthetic transport checkout `14fd41da01068ce3a515c1c121c50a6ca49121c8`.
It passed with five desktop images and one mobile image, zero attempted resource
requests, unchanged input hashes and verified browser/container cleanup. The job
used Linux ARM64, Node 24.20.0, Playwright 1.63.0 and Chromium 153.0.8010.12 in the
existing image `sha256:b8df542f0badff9f0829d97146c2adc5b193041a276b42a220f9eab5f7df8a0e`.
Limits were 2 CPUs, 4 GiB and 120 seconds, with networking disabled, no GPU,
Chromium sandbox enabled and no dependency installation.

Both the root reviewer and independent verifier inspected all six original PNGs.
The captured desktop layouts and 390-pixel mobile layout show readable questions,
quotes and full source identifiers without observed clipping or overlap. The
selected 2026 source returns the 42-minute quotation; a scoped blue-ledger query
returns no match; returning to all books shows the 2026/42-minute and
2022/18-minute passages with separate citations. These observations concern the
captured states, not general accessibility or usability qualification.

The browser used exact saved HTML and clicked only native disclosure summaries.
It did not submit forms or connect to a live application. Actual HTTP behavior is
established by the preceding capture test. This does not add a model, private-book
validation, performance measurement, live service or deployment.

Artifacts remain outside Git in the m6mini project's `20261004/librarian/`:

- `usability-walkthrough-20261004.html`: local gallery of all six original images.
- `demo-results/librarian-20261004-demo-browser-001/`: returned images, browser
  report, exact archives and custody records.
- `usability-demo-verification-20261004.json`: separate capture/browser verification.
- `usability-demo-visual-review-20261004.json`: root's image-by-image observations.

The returned browser archive has SHA-256
`f540457ad461091716517706d95f5ac7a5d8cf1ce4a9653e4f251e1f0662b9fb`;
its `report.json` has SHA-256
`5114779ee514ecf55a7acfdb52469f69085819dae640f4702154aa7f334c14a5`.
The transport checkout's source archive has SHA-256
`614b7f3984301d2c44bda7ddcff7d1d0d57d193b6aae7f6e84eea0fb3bf9f85a`.

The focused commands used, through the Spark coordinator in the reviewed
runtimes, were:

```sh
# Capture checkout 7b75555, existing minimal Python profile.
LIBRARIAN_EVIDENCE_DIR=ci-output PYTHONPATH=ingest/src \
  /path/to/reviewed/python -m pytest -q -c /dev/null -p no:cacheprovider \
  ingest/tests/test_usability.py::test_http_choose_same_title_editions_miss_then_return_to_all

# Separate transport checkout 14fd41d, existing prepared browser image.
node usability-browser-driver.cjs . ci-output/usability-browser
```

The second command reproduces rendering of its frozen inputs. A newly captured
HTTP run requires an updated, reviewed receipt and transport archive; it must not
silently reuse old provenance. The output directory must be fresh. The reviewed
four-line U2 presentation patch remains unapplied and is not shown in these images.
