---
title: Book roadmap commitments and current closeout evidence
status: partial-evidence-reconciliation
date: 2026-09-05
type: explanation
---

# Book roadmap commitments and current closeout evidence

The [ecosystem roadmap chapter](../../presentation/report/chapters/14a-ecosystem-roadmap.tex) contains eighteen numbered commitments, historical sprint-close accounts and a two-quarter evaluation pledge. The [chapter receipt](evidence/book-roadmap-reconciliation.json) fixes the source reviewed here. Historical shipment statements and proposed dates are not refreshed delivery commitments or live acceptance receipts.

This map connects every numbered commitment to the current nine-package closeout plan. It distinguishes findings already supported by review evidence from paths still needing examination. A mapped row is not a completed milestone.

| No. / commitment | Current evidence or review limit | Closeout package and next acceptance |
|---|---|---|
| 1 Governed-writeback security | Existing route checks do not establish the complete local/remote authoring, inherited policy and replay contract. [Authority review](agent-grounding-and-governance.md) and [storage review](storage-and-authority.md) retain material gaps. | CP-04/08: signed resource-scoped request, mutation, revocation and denied retry |
| 2 BrokerActor / queue | Current architecture intentionally supersedes the old actor transport; [ACSP consumer review](forum-decisions.md#visionclaw-acsp-consumption-and-recovery) finds case-correlation and recovery obligations. | CP-05: durable queued request through applied/rejected result; do not recreate the abandoned actor to satisfy its old name |
| 3 Contextual transaction cost | [Accounting review](transaction-cost-accounting.md) verifies emitter forwarding but finds turn-per-step versus cumulative semantics, incomplete capture/delivery and no established current per-DAG aggregation/display consumer. | CP-01/07: trace producer, aggregation, missing events and displayed measure; compare cost of verification and recovery as well as token counts |
| 4 Four KPI dashboard | [Two metrics exist and two await data](kpi-outcomes.md). Summary persistence and a canary fire do not establish nonempty observations or improved outcomes. | CP-01/05/07: agreed definitions, capture health, run boundaries and freshness |
| 5 MAST failure telemetry | [Failure review](failure-telemetry.md) verifies selected taxonomy/route paths but finds untagged schema errors, conflicting wire tags and an unverified metrics producer. QE and independent-runtime census remains open. | CP-07/08: include crashes, timeouts and missing telemetry; verify unknown classifications and durable capture |
| 6 Authority escalation defaults | [Capability enforcement](capability-instructions-and-enforcement.md) separates skill instructions from executable limits; [governance review](agent-grounding-and-governance.md) preserves admission/correlation gaps. | CP-04/05: demonstrate actual denied action then authorised release, including stale or opposite decisions |
| 7 Outcome learning | [Learning evidence](learning-evidence.md) distinguishes successful tool execution from desired outcome and records incomplete capture/recovery. A statistical floor alone does not prove causal benefit. | CP-07: complete capture and independently evaluated retrieval/routing change |
| 8 Orchestration diversity | The [consultant review](agent-grounding-and-governance.md#cross-model-consultation-and-acceptance) finds warning-only same-family handling and no production selector caller in its bounded search. This qualifies the chapter's “refusing” wording. | CP-05/07: verified executed-model identity, actual enforcement and candidate-bound verdict effect |
| 9 Pocket provenance | [Pocket provenance](pocket-provenance.md) verifies session-reference composition and a latest-match, 1,000-event in-memory lookup. Durable source binding, lookup authority and actual phone reconstruction remain unverified. | CP-04/05/08: decrypt, resolve the exact URN, authorise lookup and reconstruct the requested outcome |
| 10 Insight ingestion loop | [Governance](agent-grounding-and-governance.md), [ACSP recovery](forum-decisions.md) and [KPI review](kpi-outcomes.md) qualify end-to-end and velocity claims. Current KPI summary still lacks Mesh Velocity's source. | CP-02/03/05/09: one correlated proposal-to-merge-to-loaded-generation receipt with real timestamps |
| 11 Joined trace / data moat | [Source review](joined-provenance-trace.md) finds identity-only grouping, unconditional pod absence in the HTTP path, unbounded lower-cutoff reads and no handler-level resource restriction. This does not prove a complete action trace. | CP-01/04/08: trace completeness, authority, temporal joins and explicit partial results |
| 12 External pilot | The chapter explicitly says no pilot ran. [Commercial surfaces](commercial-surfaces.md) and release review do not establish an independent deployment. | CP-08/09: adopted pilot protocol, actual external operation and measured outcome; no new dates invented |
| 13 Agent disclosure | [Disclosure review](agent-disclosure.md) traces the public endpoint and sixteen current source mounts. One-shot lookup, invisible failure states and active-only historical attribution remain open, alongside complete rendered-surface coverage. | CP-04/06/09: ordinary-user views show agent identity and authorising principal under normal/stale/error states |
| 14 Actor DID identity | [Rendered state](rendered-state.md) and [identifier review](federation-identifiers.md) distinguish address representation, signature verification and actual action correlation. | CP-04/06: selected actor identity verified and bound to the authorised target action |
| 15 Voice-to-selected-actor | [Voice authority](agent-grounding-and-governance.md#voice-speaker-target-and-mandate-scope) records producer tests and issuer/resource/revocation limits. Dispatch and audible acknowledgement are not applied action receipts. | CP-04/05/06: microphone selection through scoped dispatch and applied/rejected feedback |
| 16 Graduated escalation | [Forum domain and relay review](forum-decisions.md) credits richer transitions but retains projection/recovery and independent pending-button state concerns. Reaching domain code does not prove three complete outcomes or reduced approval fatigue. | CP-05/09: three actual outcomes, risk-policy denial/auto-resolution and measured human review burden |
| 17 Decision audit and review context | [History review](decision-history.md) traces any-authenticated-reader API access, offset pagination, separate relay-derived UI history and incomplete signing-time context. Complete projection/application reconstruction remains unverified. | CP-04/05/09: authorised history read, reasoning/risk at signing time, and rejected/failed projection visibly distinct from applied decisions |
| 18 Headset intervention | [XR review](rendered-state.md) identifies incomplete control coverage alongside implemented rendering and identity primitives. Hardware is not the only remaining evidence obligation. | CP-06/09: real selection, verified identity, intended control action and denied/stale/recovery states in headset |

## Measurement and falsification

The chapter commits to Mesh Velocity, Augmentation Ratio, Trust Variance, HITL Precision and contextual transaction cost at stated review cadences. Preserve these definitions and the publication-relative evaluation pledge as authored commitments. To assess them fairly, first record the actual publication baseline, observation windows, component revisions and missing-data policy. This pass does not infer that a reporting deadline has elapsed or that a target was met.

The [KPI review](kpi-outcomes.md) establishes why a dashboard response cannot stand in for all five commitments. Two missing data sources must remain missing, and the implemented ratios need denominators tied to the intended completed work. [Liveness observations](liveness-observation.md) also need journey-specific predicates: a recent fire is not independent evidence that a user task succeeded.

The chapter's four-surface scoring frame is a separate evaluation obligation. Component tests and helper probes cannot supply its desktop, phone, forum or headset ratings. Keep the original criteria visible and run the actual target-client tasks before changing a score. Failure to observe an experiment and a measured lack of benefit should be reported separately.

## Remaining investigation from this map

The broad remaining investigation includes taxonomy coverage across the QE fleet and independent failure sources. The dedicated consumer reviews now trace the selected CTC, trace, history, disclosure and pocket-provenance paths, but retain explicit source-binding and deployment limits. Those limits must be resolved before current-state reconciliation or complete-system acceptance can be called complete. Other book chapters and historical/upstream ADR dispositions also remain open.

Use the [ordered execution sequence](closeout/execution-sequence.md) for delivery dependencies and the [assessment audit](closeout/assessment-requirements.md) for documentation completion. The eighteen historical commitments inform that plan; neither historical “sprint close” nor this mapping supplies owner ratification or complete-system acceptance.
