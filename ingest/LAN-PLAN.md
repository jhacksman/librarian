# Authenticated LAN serving plan

Proposed implementation and acceptance gates, 2026-10-04. This document does
not establish LAN availability, authorization, or passed LAN tests.

## Starting boundary

`LibraryService` supplies evidence-only search, passage lookup, book listing and
coverage through the read-only SQLite retriever. The human preview binds to
loopback; the MCP adapter exposes the same four operations. Preserve that shared
service and citation contract. MCP transport readiness is independent of choosing
an embedding or answer model. See [current interfaces](USABILITY.md) and the
[retrieval evaluation contract](evaluation/README.md).

Local hosting does not guarantee local data use: a LAN MCP client can forward
returned passages to a cloud model. The initial private-library deployment must
admit only approved local callers with local processing and storage. Record each
client's model destination, logging and retention behavior. Cloud callers require
a separate explicit data-sharing decision; the server cannot revoke excerpts
after an authorized client receives them.

## Decisions before implementing or exposing a listener

- Name the Spark host, LAN address/DNS name, permitted devices, human identities,
  client registrations and allowed books. Default to no access. An advertised
  MCP client name or a user-supplied `book_id` is not an authorization grant.
- Choose the authorization provider, token lifetime/revocation method, browser
  login/session mechanism and credential store. Keep credentials outside the
  repository and index; define their owner, permissions and rotation procedure.
- Choose HTTPS termination and certificate issuance/renewal, plus any trusted
  reverse proxy. Decide the exact binding, firewall rule and service lifetime
  only in the deployment change. Do not broaden the current preview binding.
- Pin SDK/client versions and explicitly declare supported protocol versions.
  The official [latest specification](https://modelcontextprotocol.io/specification/latest)
  resolves to 2026-07-28 at review time. That revision changes request metadata
  and removes protocol sessions and the standalone GET stream. An older client's
  successful `initialize` round trip proves only its tested compatibility mode;
  it does not establish support for the latest revision.
  [Transport and compatibility rules](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http).

## Implementation sequence

1. Add a request identity and book-authorization layer shared by the human UI
   and MCP. Apply it to search, direct passage lookup, book listing and coverage
   counts. Filter before ranking and counting; reject unauthorized direct IDs
   without revealing whether the book exists. Keep the index read-only and
   filesystem paths absent from the public contract. Tool annotations describe
   behavior; they enforce neither read-only storage nor permission checks.
   [MCP tool annotations](https://modelcontextprotocol.io/specification/2026-07-28/server/tools).
2. Add Streamable HTTP through the pinned SDK and the selected provider's MCP
   authorization flow: protected-resource metadata/discovery, least-privilege
   read scopes and validated bearer tokens on every request. Check issuer,
   audience, expiry and scope; reject missing/invalid tokens with 401 and
   insufficient permission with 403. Tokens belong in authorization headers,
   never URLs; do not pass them to another service.
   [MCP authorization](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization).
   Use the provider's maintained OAuth flow, including PKCE, exact redirect
   validation and secure token storage.
   [Authorization security](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization/security-considerations).
3. Require HTTPS for LAN access. Allow only configured Host values and browser
   Origins; reject an invalid present Origin with 403. Native clients without
   Origin still need authentication. Trust forwarded headers only from the
   selected proxy. Protect browser sessions with Secure/HttpOnly/SameSite cookies
   and CSRF controls; do not expose bearer credentials in page JavaScript.
4. Retain exact quotations, source hashes, IDs, PDF/EPUB locations and offsets.
   Treat source instructions as data and escape the human rendering. Apply
   limits before expensive work and during response serialization. Proposed
   starting limits, subject to synthetic Spark measurements: 2,000 question
   characters, 1–10 passages, 16 KiB request bodies, 128 KiB complete responses,
   100 books per page, four active requests, 60 calls/minute per identity,
   five-second body-read and ten-second work deadlines. Bound aggregate requests
   too. Paginate listings; reject an oversized passage with a bounded error
   rather than silently altering its quote or offsets. Release work on timeout
   or cancellation according to the supported protocol version.
5. Return short, stable validation/not-found/unavailable errors; never serialize
   raw exceptions, local paths, SQL, credentials or stack traces. Keep empty
   retrieval's explicit abstention separate from service failure. Record only
   identity, operation, outcome, duration and a correlation ID in bounded logs;
   exclude questions, passages and tokens. Require authentication before any
   corpus titles or counts are rendered.

## Acceptance before LAN rollout

Run tests/builds/installations only on the approved Spark job, using original
synthetic documents and temporary indexes. Do not load the private corpus,
install models, reindex the live library or start a persistent LAN service as
part of this plan.

- Run real SDK/client discovery and all four calls for each declared protocol
  version; record SDK locks and wire version. Match UI/MCP evidence and verify
  every citation against original synthetic source spans, including abstentions.
- Use two identities with disjoint books. Test search with/without filters,
  direct passage IDs, listing, counts and pagination for cross-identity leaks.
  Test expired/wrong-audience/revoked credentials and least-privilege failures.
- Test Host/Origin spoofing, absent Origin on an authenticated native client,
  proxy-header spoofing, browser CSRF, escaped hostile excerpts, malformed input,
  oversized responses, slow bodies, cancellation and concurrent saturation.
  Confirm bounded safe errors, cleanup and absence of paths/secrets in output.
- Verify no database mutation and no retrieval egress; inspect the approved
  client's processing/retention separately. Demonstrate credential rotation,
  rollback to loopback-only operation and temporary-service shutdown.
- Present measured results and the exact proposed host/TLS/auth/client/book
  configuration for deployment approval. LAN exposure, firewall/service changes
  and credential provisioning remain pending that decision. Stdio success and
  retrieval regression results alone are not LAN acceptance.
