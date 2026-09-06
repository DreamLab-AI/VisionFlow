---
title: Decision history, read authority and signing context
status: source-reviewed-open
date: 2026-09-05
type: explanation
---

# Decision history, read authority and signing context

The forum implements a paginated authenticated decision API and a separate client history assembled from observed relay events. Both are useful audit surfaces. Neither by itself proves that a decision was projected successfully, applied downstream or reviewed with a preserved snapshot of its original context. This qualifies the book's seventeenth commitment. [Six source hashes](evidence/decision-history-snapshot.json) fix the scope; this pass did not execute Worker/D1, signed requests, relay, browser or existing tests.

## API authority and pagination

The [auth-worker router](../../../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/lib.rs) registers `GET /api/governance/decisions`. Its [handler](../../../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/governance_api.rs) invokes `require_authed`, then reads all decisions or filters by an optional exact case ID. The [gate](../../../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/admin.rs) verifies NIP-98 and returns the signer; this handler does not use that signer to scope the query or require admin/community membership. A valid authenticated reader can request the unfiltered projection under this source contract. Whether that breadth is intended must be an explicit governance data policy, rather than inferred from the presence of authentication.

The router supplies the actual request URL and `canonical_url` preserves its path and query, so pagination parameters are included in the URL being verified. This source evidence should not be replaced by the stale assumption that only the endpoint path is signed. A live signed-request test remains required.

Pagination defaults to 100 rows, clamps limits to 1–200 and defaults invalid offsets to zero. SQL orders only by `decided_at DESC`, using limit/offset. Equal timestamps lack an explicit tie-breaker, and concurrent insertion between pages can shift offsets. The response returns rows, limit and offset, without a total, snapshot identifier or continuation cursor. Complete export requires a defined stable traversal contract and concurrent-write tests.

## What the persisted history proves

Rows expose decision/case IDs, outcome/detail, broker pubkey, reasoning, prior decision, supersession marker and decision time. They do not include the signed envelope, the original request event ID, a decision-time risk/confidence snapshot or an applied-operation receipt. A decision ID may permit separate event resolution, but the API does not perform that reconstruction.

The [relay projection](../../../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs) obtains the latest projected decision by timestamp, plans the response and inserts a decision separately from case-state updates. As established in the [forum review](forum-decisions.md), relay acknowledgement precedes this projection and its failure can be silent to the sender. The decision API therefore describes the available D1 projection, not a guaranteed complete census of accepted signed events. Missing rows need reconciliation against retained events; a successful query alone cannot establish completeness.

## UI history and signing context

The [governance page](../../../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/governance.rs) renders `SupersessionHistory` from the [panel registry](../../../nostr-rust-forum/crates/nostr-bbs-forum-client/src/stores/panel_registry.rs), rather than fetching this decision API. The registry captures observed kind-31403 events by case `d` tag, suppresses duplicate event IDs and retains signer, reason, outcome, timestamp and supersedes reference.

Its pure resolver marks any referenced event ID superseded and picks the newest non-superseded observed entry as effective. It does not query D1 projection or downstream mutation state. “Current” therefore describes this local event-chain calculation, not an applied-action receipt. The resolver also does not itself validate authority or reference chronology; those are upstream admission obligations. This is not evidence that unauthorised events reach a live client.

The history row displays outcome, shortened signer and reason, with superseded/current badges. It does not display a signing-time risk snapshot or execution result. No observed entries renders no history, which does not distinguish no decisions from incomplete replay or an unavailable source.

The request view carries agent-provided reasoning, risk tier and confidence. Ordinary approve/reject signing inserts a generated rationale such as `Human approve via governance UI` and references the request event. That reference is useful provenance, but it does not record the human's substantive rationale. Risk and confidence are not copied into the response; preserving and resolving the exact referenced request is required to reconstruct what was reviewed. Original event retention, subsequent request changes and missing-event behaviour remain acceptance obligations.

## Closeout requirements

CP-04/05/08/09 should ratify who can read cross-case reasoning and broker identities, then enforce and test that policy with unrelated users and private cases. Preserve exact request and response envelopes with a durable correlation to projection and applied/rejected results. Recover projection omissions without presenting relay acknowledgement as application.

Define stable history pagination and reconciliation watermarks, and test equal timestamps, insertion between pages, reconnect, missing predecessors and supersession references. Align the API and UI's observed, projected and applied states, including visibly incomplete history. Capture meaningful human rationale when required, preserve decision-time context and allow readers to resolve the exact signed request.

Use [ADR-2010](../../../nostr-rust-forum/docs/adr/ADR-2010-durable-governance-outcome-receipts.md) for the durable receipt contract and the [book map](book-roadmap-reconciliation.md) for the historical commitment. Source tracing resolves the formerly unexamined consumer path; complete live history and review-context acceptance remain open.
