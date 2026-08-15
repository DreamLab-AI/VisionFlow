# ADR-007: Close the governance loop

**Status**: Proposed
**Date**: 2026-08-15
**Context**: The 2026-08-15 terminology live test (ADR-006) exercised the governed write
path end-to-end and found it broken in the middle. The ecosystem audit of the same day
(RuVector `project-state/ecosystem-audit-2026-08-15`) ranked closing it as the next
critical decision. Related: agentbox ADR-054 (bridge defects), VisionClaw ADR-110 (ACSP),
ADR-130 (broker kernel, Proposed), agentbox ADR-048/050 (decision records as graph nodes;
decision elevation), TODO-unified C-3 and L-1.

## The problem, as observed live

The mesh's core claim is that a human decision has one place where it gets signed. On
2026-08-15 a governed amendment (proposal `7de21296`, knowledge-graph class definition)
passed the Whelk consistency gate, passed the conflict gate, and staged with a
cryptographic receipt — and then became invisible. No kind-31402 action request was
emitted, so the forum governance panel never showed it to the operator whose signature it
awaits. Three adjacent failures were found in the same session:

1. The ontology-bridge is remote-primary with a **silent local-markdown fallback on
   network error**. The remote create route fails; the fallback's incompatible `propose()`
   signature turns that into a misleading `subject 'undefined'` error; the caller never
   learns the governed backend was unreachable.
2. The panel showed an elevation case approved on 2026-08-13 (kind-31403 stored on the
   relay) **still listed as a Pending Action** — decision consumption does not reliably
   close cases.
3. The stores holding all of this — `data/{kpi,enrichment,settings}.sqlite3` — have **no
   backup at all** (TODO-unified C-3). The staged proposals, the case queue and the KPI
   lineage share one unprotected file tier.

Silence, in every case, was indistinguishable from success. That is the exact property a
signing loop must not have.

## Decisions

**D1 — Staging emits, always.** Every proposal staged by the VisionClaw proposal spine
(elevation, bridge amend, bridge create, future write kinds) emits a kind-31402 ACSP
action request through the same publisher the WS6 elevation flow uses
(`lib/elevation-publisher.js` lineage). The spine's `acsp` gate value may only read
`pending` when an event has been accepted by the relay; failure to publish is a staging
**error**, surfaced to the caller — fail-open remains acceptable for telemetry, never for
governance visibility.

**D2 — Governed writes fail loud.** The bridge's local-markdown fallback is a read-path
convenience only. A write (`ontology_propose`, `ontology_axiom_add`) that cannot reach the
governed backend returns the remote failure verbatim, tagged with the route it attempted.
`AGENTBOX_ONTOLOGY_LOCAL=1` remains available as an explicit, logged opt-in for offline
work — an opt-in is not a fallback.

**D3 — Decisions consume to closure.** A stored kind-31403 Decision closes its case in
the `broker_cases` projection and produces an enactment receipt (the applied change, or
the rejection, content-addressed like the staging receipt). A case that is approved on the
relay but pending in the panel is a defect of this ADR, not a UI quirk. The still-Proposed
ADR-130 kernel cherry-pick (broker domain kernel onto the ACSP case queue) is adopted as
part of this decision.

**D4 — The governance store is durable.** Execute TODO-unified C-3: scheduled backups of
`data/{kpi,enrichment,settings}.sqlite3`, a restore runbook, and one rehearsed restore.
Signatures over state that can vanish in a disk failure are theatre.

**D5 — Key registration is checked, not remembered.** Any system key that publishes
governance events must be present in the relay's D1 `agent_registry`/allowlist. A canary
(publish-and-read-back of a kind-31405 probe per publishing key) runs in the health
surface, so an unregistered or rotated key is a red check within minutes, not a silent
drop discovered during an incident. (The `dotenv` comment "whitelist this pubkey manually"
is the anti-pattern this replaces.)

## Falsification — the loop canary

Extend live-session item **L-1** (now unblocked; K-1/K-2 done) with the full bridge round
trip, run against the live stack:

1. `ontology_propose` (amend) from agentbox → staged with receipt, **and** kind-31402
   visible on the relay within one poll interval;
2. the case appears in the forum governance panel;
3. operator approves in the panel → kind-31403 stored;
4. the case leaves Pending, and the enactment receipt resolves;
5. pull the network between steps 1 and 2 on a second run → the caller receives an
   explicit publish failure, and nothing pretends to be staged-and-visible.

The ADR is Accepted when the canary passes and is armed; it is Implemented when the canary
has fired on real traffic alongside the rest of L-1.

## Consequences

- The bridge write path (agentbox ADR-054 defects 1 and 3) gets its fix specification
  here; ADR-054 remains the defect record.
- The proposal spine gains a hard dependency on relay reachability for governed writes —
  deliberate: a governance system that degrades silently is worse than one that refuses.
- Backup infrastructure (D4) lands before the loop drives more traffic through the store.
- Out of scope, unchanged by this ADR: NIP-42 challenge/response at the relay
  (scaffolded), cross-relay federation (frozen, G2), widening ACSP beyond ontology
  elevation. The loop must close for the one use case before it earns more.
