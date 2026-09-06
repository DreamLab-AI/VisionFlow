---
title: Forum decision delivery and human review
status: source-reviewed
date: 2026-09-04
type: explanation
---

# Forum decision delivery and human review

The forum gives the estate a shared place to see agent requests, identify their authors and issue signed decisions. Its current implementation has useful controls: separate read-only member components, admin-only response admission, strict event signature verification, and a domain model that rejects self-review and decisions on terminal cases. The main unresolved boundary is between accepting a signed event and successfully applying its meaning.

[Source and test receipt](evidence/forum-snapshot.json) records the scope of this pass.

## What the human signs

The [governance page](../../../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/governance.rs) displays agent-declared risk and confidence. Low-risk requests are suppressed from the member view; this is a visibility filter, not independent risk classification. Administrators can approve or reject a request. The event includes its `d` address and referenced request event ID. The current ordinary response builds reasoning automatically as `Human approve via governance UI` or its rejection equivalent; it does not capture the human's substantive rationale.

Signing is asynchronous, allowing extension-based signers. Separate approve/reject loading flags disable only their respective buttons, so source inspection identifies a possible conflicting-action sequence while the first response is pending. A browser reproduction remains needed. Member components mount no signing/publishing path, which is a stronger separation than hiding administrative buttons with CSS.

## Sent means relay-accepted

The [client relay helper](../../../nostr-rust-forum/crates/nostr-bbs-forum-client/src/relay.rs) registers a callback before sending the event. The page shows `Response sent` only after an affirmative relay OK, and displays a retry state on rejection. This is an honest improvement over treating socket send as success. Pending callbacks are removed on OK; the inspected publish path has no per-event acknowledgement deadline. Disconnection and silent acknowledgement loss still need a browser recovery test.

The [relay handler](../../../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs) verifies event identity/signature, session/admission policy, agent registration for agent events, and admin authority for human responses. Supersessions receive an additional authority check. It then saves the event, sends OK and broadcasts **before** projecting the action response into case and decision tables.

Projection obtains the case row, plans a domain transition, inserts a decision, and updates the case in separate operations. Errors can return silently or be ignored. Consequently, relay acceptance does not establish that the decision projection succeeded, and neither establishes that a downstream mutation was applied. The ordinary planner deliberately constructs a default case when the row is absent. Comments referring to an orphan-response `RelayGovernanceGate` should not be treated as proof of such a gate: the inspected worker source contains those comments but no corresponding named implementation or call.

## Domain rules are stronger than transaction evidence

The [core governance model](../../../nostr-rust-forum/crates/nostr-bbs-core/src/governance.rs) supports approve, reject, amend, delegate, promote and precedent outcomes, plus authorised supersession and appeals. Its governance-filtered library suite passed **47 tests** using `cargo test --locked -p nostr-bbs-core governance --lib` on 2026-09-04. These tests cover local transitions and validation; they do not execute D1 projection, client signing, or agent application.

The [event store](../../../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/storage.rs) treats these kinds as parameterised replaceable events and can delete older events with the same author/kind/address. Projected decision history and retained signed-event history therefore need separate retention guarantees. An append-only domain history is not proof that every original signed envelope remains available for verification.

Agentbox's [authority consumer](../../../project/agentbox/management-api/lib/authority-consumer.js) independently checks signer allowlisting and signatures before releasing a matching waiter. That boundary must remain independent of a forum status label. See [agent governance](agent-grounding-and-governance.md) for the early-response race and the distinction between approval and committed writeback.

## Closeout requirements

A complete decision journey needs explicit receipts for signed, relay-accepted, projection-committed, consumer-received and applied/rejected. Persist the event and its projection atomically or provide a durable replay/reconciliation job. Require request/case correlation before applying a decision, retain its signed provenance, and expose partial failure to the operator. Browser tests should cover opposite decisions during an outstanding send, lost OK, reconnect, signer cancellation and revoked authority. These are roadmap requirements, not claims that a live exploit or production loss was reproduced.

The broader forum review remains open: zones, moderation, relay read visibility, device keys, passkey/session storage, search, deployment and accessible operation require further inspection. This chapter records the decision path examined before the user expanded the task into an estate-wide ADR closeout programme.

## Identity and trust decision closeout

The nine operative ADRs now have individual acceptance conditions and both governing documents carry the same qualifications. [Current receipt](evidence/forum-closeout-snapshot.json) pins the inspected source. The native key suite passes 22 tests, including a JS-parity known-answer fixture; it does not execute the current JS producer. Both workers retain independent exact-string device gates. The exact pod dependency remains `=0.5.0-alpha.7`, distinct from other estate consumers.

The trust policy protects TL3 and admin rows, but its committed outcome is weaker than its policy description. `check_demotion` can move TL2 directly to TL0; the old ADR wording of one level per sweep was corrected. Separate trust UPDATE and audit INSERT execution errors are ignored before returning the planned level, which the sweep counts as a decrease. A returned level therefore does not establish a committed transition or retained audit.

The sweep uses OFFSET pagination over an eligible set that its writes shrink. The [reproducible probe](evidence/forum-trust-probe.py) executes the actual extracted SELECT in in-memory SQLite and models qualifying TL1-to-TL0 updates: of 400 eligible rows, the first page processes 200, the next is empty, and 200 remain eligible. This demonstrates query/pagination behaviour under the fixture, not a deployed D1 incident. Use stable pagination and explicit committed/error receipts, and test recovery, tied timestamps and more than one batch. ADR-2006 becomes partial on this evidence.

Proposed [ADR-2010](../../../nostr-rust-forum/docs/adr/ADR-2010-durable-governance-outcome-receipts.md) now covers durable event/projection/application receipts. Its contract is partial and inactive pending adoption and implementation; the preceding nine records focus on corpus, crypto, identity, trust, signing and ACL policy. Frozen archives retain their historical status. Browser signing, private delivery, deployment configuration and complete cross-service revocation remain open.

## Historical routing and proposed receipt contract

The [forum historical map](../../../nostr-rust-forum/docs/adr-history-closeout.md) gives all 24 frozen records a governing-document route and specific acceptance task, with correct links to the three sprint canonical records. This repairs navigation through a companion rather than modifying frozen text. It is routing coverage, not proof of all historical claims. [Revalidation](evidence/forum-governance-closeout.json) confirms the six source hashes from the earlier governance receipt are unchanged. ADR-2010 specifies recovery and correlation acceptance conditions without claiming the complete receipt flow already exists.

## VisionClaw ACSP consumption and recovery

The [consumer source receipt](evidence/acsp-consumer-snapshot.json) qualifies VisionClaw ADR-2006's stateless claim. The old BrokerActor transport is absent, but the elevation actor maintains pending cases and the inbox projects stored enrichment proposals. Its local BrokerCase response type is a DTO, not evidence that the retained domain aggregate or DecisionOrchestrator runs on this route. Retaining a domain kernel does not establish its integration into a new transport.

The ACSP subscription requests action responses since the current timestamp. Its loop logs skipped notifications on lag; no replay request occurs in that branch. The parser checks kind, case-prefix and JSON shape, then produces CaseDecision with case ID, action, reasoning and responder pubkey. It drops the event ID, timestamp and request-event reference at this boundary. SDK signature checks and forum admission policy are separate controls; the source finding does not establish that an unauthorised event can reach a live consumer.

The elevation handler removes a matching case from its pending map before dispatching the decision. Approve enters the consistency gate and PR workflow; other actions share a record-and-skip branch. Persistence failures there are logged after the map entry has been consumed. Replayed or unknown case IDs are ignored when absent from the map. In the cycle producer, publication precedes proposal persistence and insertion into the actor's pending map. These orderings require delivery/recovery evidence; this pass did not reproduce an early-response race or lost decision.

ADR-2006 is partial against the broader stateless, kernel-driven approval-flow description. The ACSP surface and removal of the old transport remain established. CP-01/03/04/05/08 requires an explicit owner for the durable case state machine, event/request correlation retained through application, bounded replay and restart recovery, and distinct semantics for approve/reject/amend/delegate. Test publication-before-registration, duplicate/superseding responses, lag, persistence failure and replay after restart. Verify reviewer admission and case-specific authority at the chosen trust boundary. Link human intent → signed request/response → stored outcome → gate result → PR/merge or rejection receipt, without equating any intermediate stage with completed mutation.

## Forum navigation, counts and cold entry

The three sprint-canonical ADRs 090–092 remain distinct from their frozen archive stubs. The [source snapshot](evidence/forum-sprint-snapshot.json) traces their current client implementations. This is source evidence; no browser, login, service worker or real relay ran.

ADR-090's base-relative path helper and explicit service-worker URL/scope are implemented. Login returnTo handling normalises paths and rejects root/login/signup destinations. The helper uses a textual strip_prefix rather than a path-segment boundary check, and the validator accepts slash-prefixed input before normalisation. These do not prove every unusual or encoded destination is handled correctly. The invariant should be verified at the browser/router boundary under both root and /community deployments, including prefix collisions, doubled prefixes, protocol-relative forms and login loops. No external redirect was demonstrated here.

ADR-091's independent persisted message counter is removed. Store counts derive from deduplicated event vectors; broad delivery suppresses tombstoned events, and deletion folding removes stored events. This establishes useful local mechanisms. However, ChannelPage maintains a second MessageData vector: its shared-store effect appends unseen events and sorts, without removing events absent from a later shared-store snapshot. Its displayed message count derives from that local vector. Store deletion therefore does not by itself establish removal from an already-mounted page's displayed count; other local actions may update it separately. Test remote deletion, replay and navigation through the actual consumer rather than only the store fold.

ADR-092's original ensure_subscribed lifecycle is no longer the page's active bootstrap path. The method remains, but the page explicitly removed its Effect in favour of a kind-40 by-ID query. When metadata arrives, the page seeds the shared store and opens narrow kind-42 replay filters for ID, name and section. It tracks those subscriptions for cleanup. Reactive fallback resolves names/sections against the shared channel list. This credits an implemented recovery mechanism without treating the earlier wait-for-EOSE/~4-second design as current code.

That page-owned replay callback deduplicates IDs but does not consult the store's tombstones, unlike the broad subscription. It can therefore append a delivered previously-deleted event to the store at this boundary; whether a live relay supplies that event remains a separate question. The local page projection then has its own deletion reconciliation obligation. The loading fallback is eight seconds and clears loading; it is not a successful bootstrap receipt. Slug cold entry, late metadata, reconnect and rapid route changes require browser/relay evidence, including cleanup and visible error states.

CP-01/06/08/09 requires one declared navigation and message lifecycle contract, consistent tombstone handling at every insertion path, reconciliation of displayed projections, and negative/recovery journeys under the shipped deployment base. Preserve all three accepted design statuses while retaining these implementation and activation limits. The sprint records are canonical; frozen copies remain historical routing aids.
