---
title: Agent runtime ingress and identity
status: source-and-local-integration-verified
date: 2026-09-04
type: explanation
---

# Agent runtime ingress and identity

Agentbox's interaction plane combines two controls: per-user identity at the proxy, and a shared daemon token beneath it. This is a substantive defence against treating loopback access as sufficient authority, but it is not per-process isolation or a universal estate identity boundary.

## Current executable evidence

The existing [proxy self-test](../../../project/agentbox/config/nip98-proxy/selftest.mjs) starts local fake upstreams and exercises HTTP, WebSocket, route selection, credentials and sessions. The initial run exited zero with three signature-related skips because `nostr-tools` could not resolve from the source layout. Repeating with the already installed `management-api/node_modules` supplied through `NODE_PATH` passed **45 assertions with zero failures and zero skips**. Both [initial](evidence/ingress-selftest.json) and [complete local](evidence/ingress-selftest-runtime-deps.json) receipts are retained.

This verifies local proxy behaviour with test credentials, not production ingress, a real AoE daemon, per-process isolation or downstream mutation authority. The dependency-resolution difference itself matters: a zero-exit test needs its skip count and runtime environment attached before it supports a signature-verification claim.

## The boundary has several modes

[verifyIdentity](../../../project/agentbox/config/nip98-proxy/proxy.mjs) accepts an explicitly configured break-glass bearer, fresh NIP-98, or an HMAC browser session established through a signed handshake. Upstream requests are authenticated before routing. The proxy-owned handshake page is a pre-auth surface; saying literally every request is already authenticated obscures that necessary distinction.

HTTP and WebSocket forwarding remove client-supplied identity headers and inject the authenticated identity. For named routes, a configured upstream bearer is injected only for a non-NIP-98 mode; fresh NIP-98 is preserved so the upstream can verify it. AoE instead receives its daemon token in every mode. Therefore the proxy establishes attribution for its routes while downstream governance still needs its own signature and authority checks.

The current [Nix source](../../../project/agentbox/flake.nix) generates loopback `aoe serve --auth token --behind-proxy`. That proves build configuration, not the active process mode. Current ADR amendments explicitly keep source verification separate from deployed activation.

## Credentials do not yet provide process isolation

The proxy reads the AoE token from its state file and retains a last-good value across transient read errors. Deleting that file is not a revocation mechanism; daemon acceptance and rotation determine validity. Its source comments acknowledge mtime-based cache limitations and the shared-user residual.

[aoe-curl.sh](../../../project/agentbox/scripts/aoe-curl.sh) constrains generated agent access to positional methods and loopback API paths rather than arbitrary URLs or curl flags. That narrows accidental or injected misuse of this wrapper. Co-resident processes that can read the same token still share its authority, and other direct consumers need their own reviewed boundary.

The [identity helper](../../../project/agentbox/management-api/lib/agent-identity.js) derives lowercase x-only public identity and writes the private key with mode 0600. It can return a valid identity with `persisted: false` after a failed write, making restart stability a separate property. A failed derivation returns null; the entrypoint can retain its placeholder identity. Canonical public formatting therefore does not prove stable ownership across restart or complete canonical migration in every pod tier.

## ADR closeout consequences

Agentbox ADR-2002, 2009, 2010 and 2011 now contain current source verification and explicit CP-01/04/05/06 acceptance conditions. Their historical evidence remains visible. ADR-2022 changes from complete to partial because forced-local authoring is outside the remote direct-load guard, as established in [agent grounding and governance](agent-grounding-and-governance.md).

Remaining acceptance includes active binary/config identity, direct tokenless rejection, token rotation, session expiry and allowlist removal, opposite authentication modes at actual governance upstreams, failed key persistence, and the permissions of each co-resident consumer. This chapter covers ingress and identity; full container lifecycle, tool permissions, memory privacy and isolation remain under investigation.

## Port-gate syntax and exposure coverage

[ADR-2013](../../../project/agentbox/docs/adr/ADR-2013-loopback-publish-except-9096.md) correctly expands the intended port inventory across root `docker-compose*.yml` overlays and names ten sanctioned mappings. The current tree passes its [CI script](../../../project/agentbox/scripts/ci/check-ports-loopback.sh), which is wired into the invariants workflow. This verifies the script's judgement of these files, not the complete active network surface.

The [isolated fixtures](evidence/ports-gate-probe.json) demonstrate a syntax gap: a block-form public mapping fails, but the same invented public port inside a service flow mapping or a whole-file JSON flow mapping passes. The walker recognises `ports:` only at the beginning of an indented line. Its global long-syntax check does not catch a nested short-syntax `ports` list. The [reproducer](evidence/ports-gate-probe.py) copies the actual script unchanged into temporary roots. Python parses the JSON control as structured data; Docker Compose resolution and service launch were not performed. No actual unsanctioned public binding is established by these fixtures.

The gate also scopes file discovery to root `docker-compose*.yml`; a claim covering arbitrary filenames, nested deployment inputs, overrides supplied externally or host-network listeners requires another inventory. A sanctioned `(file, mapping)` pair identifies an exposure exception, not the service's authentication, intended audience, host firewall or runtime bind state. The broad claim that the list alone describes every LAN door should therefore be narrowed.

CP-01/04/08 requires a parser-backed gate over the effective deployment configuration, with supported syntax and override precedence explicit. Test block/flow forms, aliases and merges, interpolation, file naming and long syntax; either reject unsupported input reliably or inspect its normalised meaning. Attach exact release/config identity and active listener evidence to sanctioned exposures, with authority and revocation requirements per service. ADR-2013 is partial against its syntax-proof/all-surface guarantee; existing exception decisions and historical activation remain visible.

## Relay admission versus inbox authorisation

The pod-bridge backend constructs an in-memory relay and passes the allowlist only to its asynchronous inbox consumer. The consumed WebSocket handler verifies signatures, calls `ingest_verified`, and returns relay OK after ingestion. That method stores/broadcasts events according to event kind without the bridge allowlist. The consumer subsequently checks exact author-key membership before unwrapping or writing the inbox. An empty list therefore denies inbox processing, not relay storage/broadcast. Ephemeral events are broadcast without retention. Self-authored events are skipped by the consumer because the egress path writes them separately.

This source trace crosses [bridge startup](../../../project/agentbox/services/nostr-pod-bridge/src/main.rs), [consumer authorisation](../../../project/agentbox/services/nostr-pod-bridge/src/lib.rs), [WebSocket admission](../../../solid-pod-rs/crates/solid-pod-rs-nostr/src/ws.rs) and [verified ingestion](../../../solid-pod-rs/crates/solid-pod-rs-nostr/src/typestate.rs). Relay OK does not acknowledge authorised or durable pod delivery. Consumer lag logs dropped events; restart/replay and delivery receipts need explicit acceptance.

Three existing native helper tests pass: listed author accepted, unknown author rejected, delegation tag does not grant access. They exercise the authorisation helper with synthetic events, not the full signature/relay/inbox journey. [The receipt](evidence/relay-admission-snapshot.json) records source hashes and command scope. No event was sent or pod written in this pass.

For the standalone backend, Nix emits no `pubkey_whitelist` setting when the list is empty, while enabling NIP-42 for non-open policy. This differs from generating an explicit empty whitelist; the consumed standalone binary's default and authentication semantics still require verification. The validator's W039 text claims only the local npub is accepted, contradicting the intended no-auto-add account and the inspected bridge helper. The text is not evidence of an implemented fallback.

ADR-2012 is partial against its relay-wide no-fallback guarantee. CP-01/04/08 requires admission policy at the intended boundary for each backend: signature validity, permission to publish, permission to subscribe/read, inbox authorisation and durable delivery are separate contracts. Test listed/unlisted/self authors, empty lists, gift wrapping, removal/restart, lag and both backends. Bind effective key lists and backend identity to release receipts; no current external reachability or deployed rejection result is inferred here.

## Custody and revocation acceptance

ADR-2027 now has a [provisional seven-role custody register](../../../project/agentbox/docs/SECURITY-profiles.md#provisional-custody-register--2026-09-04). It covers bridge/publisher identity, emergency bearer, browser-session signing, AoE token, remote execution and backup recovery. Actual custodians, deployed locations, cadences and response windows remain unconfirmed. This is a governance starting point, not a complete secret inventory or implemented lifecycle policy.

The proxy captures its emergency bearer at start and compares it before other identity methods. That branch lacks expiry and request-scope checks; returning a mode/identity is not durable per-use audit. A configured session secret and a per-boot random secret also imply different restart invalidation policies. The AoE last-good token cache requires coordinated revocation rather than file deletion alone.

Dream dispatch invokes ssh/scp with batch/timeout options and ambient identity configuration; these calls do not identify the actual deployed credential or its remote permissions. VisionClaw's backup script gathers selected configuration filenames, creates an ordinary ZIP plus file manifest, and invokes unzip integrity testing. It specifies neither encryption nor explicit permission hardening; environmental protections remain unverified. Archive integrity does not establish restoration, off-host survival or revocation of retained copies.

The [source receipt](evidence/custody-snapshot.json) contains code hashes only. No credential file, environment, backup or remote account was inspected or used. CP-01/04/07/08 requires role acceptance, complete inventory, explicit lifecycle windows and synthetic recovery/revocation exercises before declaring ADR-2027 implemented. Its proposed/none/inactive status remains accurate for that full policy.

## VisionClaw replay and operation boundaries

VisionClaw's [NIP-98 validator](../../../project/src/utils/nip98.rs) checks freshness, URL/method and signature before atomically claiming the event ID. Its map refuses new IDs at capacity without evicting live entries. The [isolated actual-helper probe](evidence/replay-cache-probe.json) confirms initial acceptance, replay rejection, capacity rejection with the prior entry retained, rejection immediately before TTL, acceptance exactly at TTL, and acceptance into a fresh map. [Reproducer](evidence/replay-cache-probe.py) extracts the helper/constants into a temporary Rust executable. It does not execute signatures, concurrent mutex access or HTTP requests; the ADR's historical 26-test run is not re-certified.

The freshness check accepts an inclusive ±60-second wall-clock window, while the cache expires at a 120-second monotonic age. Exact-boundary and clock-discontinuity behaviour needs a combined validator test, particularly for a token initially dated at the future edge. The helper receipt establishes expiry semantics, not a reproduced authenticated replay across the full validator. A new process or another replica has a separate map; deployment and restart recovery must account for that explicit scope.

Body binding is conditional: a supplied request body is compared only when a payload tag is present. The inspected Solid proxy identity helper passes no body, while NostrService passes its caller-provided optional body. This requires route-specific mutation policy rather than describing every accepted token as binding every request byte.

Single-use authentication is also distinct from exactly-once mutation. NostrService validates/claims the token before user lookup/creation; subsequent application work can fail after a successful claim. Retries need a fresh authentication event and an application idempotency or outcome-reconciliation contract. CP-04/05/08 requires full-route body binding, rejection/consumption semantics, response-loss and failed-commit recovery, capacity handling, replica/restart policy and combined clock-boundary tests. ADR-2002 retains its scoped implementation declaration with these limitations made explicit.
