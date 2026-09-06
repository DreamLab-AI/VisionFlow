---
title: Liveness observations and acceptance evidence
status: source-reviewed
date: 2026-09-04
type: explanation
---

# Liveness observations and acceptance evidence

VisionClaw implements the central observer proposed by historical ADR-130 D3: SQLite-backed registration and fires, HTTP registration/observation/status, local emit sites, a KG watchdog and an optional Nostr tap. The [source receipt](evidence/liveness-observer-snapshot.json) confirms boot wiring and these paths. This is meaningful implementation beyond passive health endpoints, but its output requires interpretation before it can support complete-system acceptance.

## What an observation establishes

HTTP registration accepts an identifier, description, kind and optional repository/wave/SHA. It does not accept or execute the historical wire-descriptor/fire-predicate contract. Observe accepts a free-form evidence string and forwards it to storage. In the non-debug build without dev-auth, a configured shared agent key gates writes; the debug/dev-auth branch returns true at this local check. Central route policy remains a separate gate. The key authenticates a caller credential, not the truth of the described traffic or ownership of a named canary.

The Nostr tap is more structured: it checks the event signature and maps allowed publishers, kind/tag and content into observations. Boot enables it only when CANARY_TAP_RELAY_URL is configured, and an empty publisher allowlist rejects events. Signed publisher evidence remains an attestation unless the consuming observer independently validates its claimed outcome. No relay or HTTP observation was sent in this review.

The harness stamps fires with current_sha: a runtime environment value takes precedence over the compiled value, with unknown as the fallback. Foreign observations therefore inherit the observer's revision at this boundary, not an independently verified producer revision. Registration SHA is separate and is not the status query's fire-freshness comparison. A shared server SHA cannot stand in for a complete cross-repository artefact manifest.

## Fired is neither success nor continuous health

Status counts all observations and reports the latest timestamp across them. Its fired flag means at least one stored fire has the current server SHA and a timestamp within the 30-day window; armed is its inverse. The query applies the same criterion to standing and one-shot records. Thus the kind label does not implement a distinct continuity/expiry policy for standing loops. A months-old count or newer differently-versioned observation can coexist with a different fired result; readers must retain these distinctions.

The KG watchdog records state transitions in both directions, including loss. That negative transition fires the same canary while the separate kg_backend_up gauge becomes false. Recording a failure is correct observability behaviour, but a generic fired flag cannot then be interpreted as successful loop operation. The watchdog self-polls health rather than observing business traffic, an explicit exception to the original traffic-only description.

## Closeout requirements

CP-01/03/07/08/09 needs a typed evidence contract distinguishing observation, success, failure, current health and acceptance. Define per-canary predicates, owner authority, observation identity, producer/consumer revisions and required payload correlation. Give standing loops a stated continuity window and failure/quiet-period policy; retain one-shot correctness evidence separately.

Test no traffic, invalid or unauthorised observation, stale/mismatched/unknown revision, negative transitions, restart and unavailable storage through the actual status and promotion consumers. Reconcile the historical manifest-schema and independent wire-tap commitments with current HTTP attestations and local instrumentation. No wave should be promoted solely because fired is true. These source findings do not establish that a current promotion consumer makes that mistake; tracing those consumers remains open.

## Consumer evidence and promotion scope

The [actual D1 checker fixture](evidence/canary-consumer-probe.py) substitutes only curl, using invented local responses. [Results](evidence/canary-consumer-probe.json): an empty roster exits 2; a count-only response with count 1 exits 0 after observation; the same count plus a successful HTTP response body containing fired:false still exits 0. The script does not inspect that response body's acceptance flag. These mocked cases establish checker scope, not a real server emitting that contradictory response.

D1's script proves at most that its roster-count predicate and transport-level observation request succeed. It does not run a client, inspect an agent's freshness or action state, observe a beam, or bind a render acknowledgement to a source event. Its narrower roster evidence can be useful, but cannot alone close the embodiment join.

SwarmObservabilityPanel polls status every ten seconds and displays each record as fired/armed, with a green fired label. Fetch failures retain prior state. The local D8 fire latch is set before calling observeCanary, whose errors are swallowed; that mounted instance does not retry solely because the observation failed. This preserves UI operation during harness failure, but means client-side latch state is not a durable acknowledgement. No browser transition was exercised.

The archived [sprint governance ADR](../../../VisionFlow/docs/archive/adr/ADR-004-gap-close-sprint-governance.md) places promotion at canon ratification with a new register version, and separately requires mesh evidence for federation-verified claims. A bounded search of VisionFlow scripts/workflows, VisionClaw scripts/client and agentbox scripts/management API did not identify an automatic wave-promoter consuming fired. That is not proof none exists elsewhere. The current findings therefore qualify the evidence supplied to review; they do not assert an automated false promotion occurred.

CP-01/06/08/09 requires predicates matched to the claimed journey: distinguish roster presence, observed action, rendered beam and user-visible result. Verify positive and negative observation receipts, retry/idempotency, stale status and source-to-consumer correlation. Record the canon review and compatibility evidence explicitly when promoting a wave; a checker exit code or green fired label cannot substitute for that decision.
