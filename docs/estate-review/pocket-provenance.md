---
title: Pocket provenance and reference durability
status: partial-implementation-assessment
date: 2026-09-05
type: explanation
---

# Pocket provenance and reference durability

The live mirror appends a canonical session activity URN, and the digest producer includes an activity reference. The resolver can return a retained matching event. These are useful components, but they do not establish durable reconstruction of every escalated decision on a phone. [Evidence receipt](evidence/pocket-provenance-snapshot.json) contains six hashes, nine passing existing tests and five in-process route assertions. No Nostr message was sent and no phone, live database or network journey ran.

## Reference and evidence unit

The [live mirror](../../../project/agentbox/config/hooks/nostr-live-mirror.cjs) derives an activity URN from session ID and identity scope. Every turn under the same session/scope uses the same reference; an unavailable minter or session ID falls back to text-only. Body composition preserves the reference within the cap by truncating turn text. The [digest producer](../../../project/agentbox/services/nostr-pod-bridge/src/session_summary.rs) also mints a session activity reference. This pass does not rerun cross-language byte parity.

A session identity is not an exact decision/request/receipt identity. Preserving its spelling does not demonstrate that the named session record exists, includes the original evidence and uncertainty, or identifies the exact escalation the operator is considering. The existing [egress review](runtime-egress-and-profiles.md) remains relevant to filtering, truncation and encryption boundaries.

The nine passing tests verify composition, canonical form, determinism, fallbacks and construction of a resolver URL. The test described as reference resolution checks URL structure; it does not resolve a stored execution record. That evidence should be credited at its actual scope.

## Resolver retention and selection

The [URI route](../../../project/agentbox/management-api/routes/uri-resolver.js) redirects activity/event names to `/v1/agent-events?id=<urn>`. The [event route](../../../project/agentbox/management-api/routes/agent-events.js) searches the [publisher](../../../project/agentbox/management-api/utils/agent-event-publisher.js)'s in-memory buffer, retained at 1,000 events. It matches numeric event ID or exact top-level `source_urn`, `target_urn`, `activity_urn`, `event_urn` or `urn`. It does not match the CTC `handoff_id` or arbitrary nested metadata.

The newest matching record wins, and the response contains one record. Two different decisions or turns sharing a session reference are therefore not returned as a full history. Eviction returns 404; the lookup has no durable-store fallback. A new publisher starts with an empty buffer and numeric IDs restarting at one, so a bare numeric reference is process-local unless an external incarnation contract is added.

An actual isolated route probe first minted a reference without inserting a record: lookup returned 404. After injecting two matching synthetic events, it returned only the second. After 1,000 unrelated events, the same reference returned 404 again. These five assertions prove selection and eviction behaviour, not that production session references currently have matching producer records. That producer binding remains unverified.

## Authority and mobile consumption

The reviewed GET handler has no caller or resource authorisation check of its own and returns the retained event with additional fields preserved. Deployment-level access controls may apply, but they were not inspected or exercised in this pass. Encryption of the mirror message does not itself authorise the subsequent HTTP lookup. A complete mobile journey needs a reachable resolver, authenticated resource access and an explicit unavailable/expired result.

The mirror carries a URN string rather than evidence that a particular phone client resolves it. No automatic phone resolver, stable link handoff, retained original request, decision-time uncertainty or applied-action reconstruction was established here. Existing components must be joined and validated before upgrading the book's standalone claim to complete-system acceptance.

## Closeout requirements

CP-04/05/08 should persist a session manifest and distinct signed request/decision/application references with retention and revision identity. Bind each mirrored escalation to the exact original evidence and uncertainty, and link the session reference to its full ordered history rather than an arbitrary latest event.

Test authorised and denied resolution, missing producer records, multiple decisions per session, buffer eviction, process restart and retention expiry. Preserve explicit missing/partial states and avoid silently substituting another record. Then exercise an actual phone client from decryption through resource-authorised lookup to the original evidence and applied/rejected outcome. Record transport and content provenance separately.

These requirements extend [ADR-2026](../../../project/agentbox/docs/adr/ADR-2026-session-mirror-egress-boundary.md) without changing its existing privacy constraints. The [book roadmap](book-roadmap-reconciliation.md) and [execution sequence](closeout/execution-sequence.md) retain the full journey as open.
