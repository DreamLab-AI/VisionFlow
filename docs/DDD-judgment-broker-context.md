# DDD: Judgment Broker Bounded Context

**Status:** Living document
**Date:** 2026-05-22 (F6 supersession-authority spec added 2026-07-08)
**Scope:** Cross-substrate decision loop between agents and humans
**Amendment:** §7a (Supersession Authority) added per [ADR-005](ADR-005-gap-close-canon-decisions.md) §Decision 5, discharging register gap **F6**; extends Invariant 5. The canon states this authority model here; nostr-rust-forum, which owns `DecisionOutcome` and the decision surface, implements against it (`PRD-gap-close-forum.md`).

---

## 1. Bounded Context

The Judgment Broker Context governs the decision loop between agents and humans. It spans four substrates (nostr-rust-forum, agentbox, VisionClaw, Nostr relay mesh) but maintains clear ownership boundaries per aggregate. The broker does not own identity, persistence, or transport --- it orchestrates the flow of governance decisions across the substrates that do.

---

## 2. Context Map

| Context | Relationship | Notes |
|---|---|---|
| **Judgment Broker** (this context) | Coordinates decision flow across substrates | Defines the decision loop lifecycle |
| **nostr-rust-forum Governance** | Upstream | Domain model owner, human decision surface, relay gating rules. Canonical source: `crates/nostr-bbs-core/src/governance.rs` (1015 lines) |
| **agentbox Agent Runtime** | Upstream | Agent event publishing, decision reception, broker-bridge proxy. Files: `relay-consumer.js`, `broker-bridge.js`, `nostr-bridge.js` |
| **VisionClaw Semantic Substrate** | Upstream | Enrichment proposals, knowledge graph mutation gating. `BrokerActor` on `crashbug` branch |
| **Nostr Relay Mesh** | Transport layer | Not a domain participant --- the medium for all governance events |
| **Solid Pod Storage** | Persistence | Governance events persisted at `pods/<npub>/events/governance/` |

### Relationship Types

- **nostr-rust-forum -> Judgment Broker:** Conformist. The broker conforms to the governance event kinds (31400--31405) and the `DecisionOutcome` enum defined in `governance.rs`.
- **agentbox -> Judgment Broker:** Customer/Supplier. Agentbox supplies the agent runtime and relay consumer; the broker context defines the protocol the consumer must follow.
- **VisionClaw -> Judgment Broker:** Anticorruption Layer. The `broker-bridge.js` in agentbox translates between VisionClaw's enrichment model and the governance event protocol.

---

## 3. Aggregates

| Aggregate | Owner Repo | Source File | Description |
|---|---|---|---|
| `BrokerCase` | nostr-rust-forum | `governance.rs` | Case lifecycle: Open -> PendingReview -> Resolved/Rejected. Contains case_id, agent_did, category, priority, content, decisions. |
| `PanelDefinition` | nostr-rust-forum | `governance.rs` | Agent-declared control panel: schema fields, allowed actions, display config. Kind 31400. |
| `ActionRequest` | nostr-rust-forum | `governance.rs` | Agent request for human decision. References panel, includes context. Kind 31402. |
| `ActionResponse` | nostr-rust-forum | `governance.rs` | Human-signed decision. `DecisionOutcome` enum: Approve, Reject, Amend, Delegate, Promote, Precedent. Kind 31403. |
| `PanelRegistry` | nostr-rust-forum | `panel_registry.rs` | Reactive store of active panels, pending actions, agent stats. Client-side aggregate. |
| `GovernanceEventPod` | agentbox | `relay-consumer.js` | Pod-local persistence of governance events at `pods/<npub>/events/governance/<event-id>.json` |
| `BrokerBridge` | agentbox | `broker-bridge.js` | REST/SSE proxy bridging VisionClaw enrichment review to management API |

---

## 4. Entities

| Entity | Identity | Owner |
|---|---|---|
| `Agent` | `did:nostr:<hex-pubkey>` | agentbox (sovereign bootstrap) |
| `Human` | `did:nostr:<hex-pubkey>` | nostr-rust-forum (passkey auth) |
| `GovernanceEvent` | Nostr event ID (hex) | Relay mesh |
| `EnrichmentProposal` | VisionClaw case ID | VisionClaw |

---

## 5. Value Objects

| Value Object | Fields | Notes |
|---|---|---|
| `NostrEventKind` | Integer kind (31400--31405) + semantic name | Parameterised replaceable events |
| `DecisionOutcome` | Approve, Reject, Amend, Delegate, Promote, Precedent | Closed enum in `governance.rs` |
| `CaseCategory` | KnowledgeEnrichment, PolicyReview, ResourceAllocation, AccessControl | Extensible |
| `CasePriority` | Critical, High, Medium, Low | |
| `CaseState` | Open, PendingReview, Resolved, Rejected | State machine on `BrokerCase` |
| `NIP98Signature` | Schnorr signature over event hash | Authentication proof |

### Nostr Event Kind Allocation

| Kind | Name | Publisher | Purpose |
|---|---|---|---|
| 31400 | PanelDefinition | Agent | Declare a control panel |
| 31401 | PanelUpdate | Agent | Update panel schema |
| 31402 | ActionRequest | Agent | Request human decision |
| 31403 | ActionResponse | Human | Signed decision |
| 31404 | CaseStatusUpdate | Agent | Case state transition |
| 31405 | CaseMetadata | Agent | Supplementary case data |

---

## 6. Domain Events

| Event | Trigger | Publisher | Consumer |
|---|---|---|---|
| `PanelPublished` | Agent declares control panel | agentbox -> relay | nostr-rust-forum (renders panel) |
| `ActionRequested` | Agent requests human decision | agentbox -> relay | nostr-rust-forum (shows pending action) |
| `ActionResponded` | Human signs decision | nostr-rust-forum -> relay | agentbox (relay-consumer) |
| `DecisionReceived` | agentbox receives kind 31403 | relay -> agentbox | Orchestrator adapter |
| `DecisionApplied` | Agent acts on decision | agentbox | **NOT YET IMPLEMENTED** |
| `EnrichmentProposed` | VisionClaw submits enrichment for review | VisionClaw -> broker-bridge | Management API |
| `EnrichmentDecided` | Human approves/rejects enrichment | broker-bridge -> VisionClaw | BrokerActor |
| `ProvenanceRecorded` | Decision linked to PROV-O activity | --- | **NOT YET IMPLEMENTED** |

---

## 7. Invariants

1. **Single agent ownership.** A `BrokerCase` must have exactly one agent DID and at least one human decision before resolution.

2. **Event reference integrity.** An `ActionResponse` (kind 31403) must reference a valid `ActionRequest` (kind 31402) via Nostr event tags. The relay gate rejects orphan responses.

3. **Agent-only publishing.** Only NIP-42 authenticated agents may publish kinds 31400, 31401, 31402, 31404, 31405.

4. **Human-only decisions.** Only NIP-42 authenticated humans may publish kind 31403.

5. **Decision immutability and supersession authority.** A `DecisionOutcome` is immutable once published. The Nostr event is the audit trail. There is no "undo" --- only new events that supersede. **Supersession is authority-gated (F6, §7a):** only the original human signer, or a human holding a governance role above the original at the time of supersession, may supersede a published `DecisionOutcome`, and only by publishing a *new signed kind-31403* that references the superseded event and carries a stated reason. No other party may change a published decision, and no published event is ever mutated.

6. **Relay as ordering authority.** The relay mesh is the source of truth for governance event ordering, not any single substrate's local store. Pod persistence is a cache, not the canonical timeline.

7. **Decision loop closure.** `handleGovernanceDecision()` on the orchestrator adapter MUST exist before the decision loop can be considered closed. **SATISFIED (2026-05-22, `e1a8d716`; verified 2026-07-10)** --- implemented at `agentbox/management-api/adapters/orchestrator/local-process-manager.js:133`, invoked from `relay-consumer.js:318`. Forum-originated decisions are applied with provenance; the agent side is no longer open-ended for that path.

---

## 7a. Supersession Authority (F6)

Invariant 5 says a published `DecisionOutcome` is superseded, never mutated, but
until now named no authority: it did not say *who* may supersede a decision or
*how*. This section supplies that model, discharging register gap **F6** per
ADR-005 §Decision 5. It extends Invariant 5; it does not weaken immutability.
Supersession remains a new signed event referencing the old — this section only
constrains who may sign it and what states result.

### 7a.1 Who may supersede

A published `DecisionOutcome` (kind 31403) may be superseded by exactly one of:

| Authorised party | Condition |
|---|---|
| **The original human signer** | The `did:nostr` that signed the original kind-31403, acting on their own decision |
| **A higher governance role** | A human holding a governance role *above* the original signer's role at the time of supersession (per the forum's role model, e.g. an admin over a member/moderator) |

No other party may supersede. In particular, **an authenticated human of equal or
lower governance role than the original signer MUST NOT supersede** that signer's
decision — the authority gradient the governance plane exists to hold is not
collapsed by supersession. An agent MUST NOT supersede a decision (Invariant 4:
only humans publish kind 31403). The relay `RelayGovernanceGate` enforces this: a
superseding kind-31403 whose signer is neither the original signer nor a
higher-role human is rejected as an unauthorised supersession, exactly as it
rejects an orphan response under Invariant 2.

### 7a.2 How a decision is superseded

Supersession is a **new signed kind-31403** (`ActionResponse`) that:

1. references the superseded event by Nostr event tag (`e`-tag), the same
   reference-integrity mechanism Invariant 2 already requires;
2. carries the superseding signer's own `DecisionOutcome`; and
3. carries a **stated reason** for the supersession.

The original event is untouched and remains on the relay as the audit record; the
superseding event points back at it. Reading the two together reconstructs the full
history — no state is lost, and nothing is edited in place (Invariant 5).

Two named conformance shapes:

- **Revoke** is supersession by a `Reject` that references a prior `Approve`. It
  withdraws a previously granted decision. The original `Approve` stays on the
  audit trail; the `Reject` supersedes it under §7a.1 authority.
- **Appeal** is *not* a supersession by the appellant. It is a fresh
  `ActionRequest` (kind 31402) that cites the superseded (or to-be-reviewed)
  decision, re-opening the case for a new human decision. An appeal reopens; it
  does not itself overturn. The overturning, if any, is a subsequent kind-31403
  under §7a.1 authority.

### 7a.3 Decision and case lifecycle — Superseded / Reopened

Supersession introduces two lifecycle states on top of the existing
`CaseState { Open, PendingReview, Resolved, Rejected }`:

| State | Meaning | Entered by |
|---|---|---|
| **Superseded** | A previously published `DecisionOutcome` has been referenced and replaced by a newer authorised kind-31403 | A superseding kind-31403 under §7a.1 (includes Revoke) |
| **Reopened** | A `Resolved`/`Rejected`/`Superseded` case is under review again after an appeal | A fresh kind-31402 (§7a.2) citing the prior decision |

Lifecycle transitions:

```
Resolved/Rejected --(superseding 31403, §7a.1)--> Superseded
Resolved/Rejected/Superseded --(appeal 31402, §7a.2)--> Reopened
Reopened --(new 31403 decision)--> Resolved / Rejected / Superseded
```

A `Superseded` decision is terminal *for that event* — it is never mutated — but
its **case** is not terminal: an appeal can reopen the case, and a new decision can
supersede the superseding one, chaining forward exactly as the register does under
Invariant 5. The current effective decision for a case is the most recent
authorised kind-31403 in the reference chain; superseded events remain visible as
history.

### 7a.4 Ownership and integration

The canon states this authority model; **nostr-rust-forum owns `DecisionOutcome`,
the `CaseState` machine and the decision surface**, and implements §7a against this
spec (`PRD-gap-close-forum.md`), citing it as its canon contract. Reaches
`integrated` when the forum cites this section and a superseding decision is
demonstrated end to end (an authorised supersede accepted, an unauthorised
supersede rejected by `RelayGovernanceGate`). This is a cross-context extension of
the Judgment Broker model through its owner, not a parallel model (DDD Gap-Close
Canon Context §10).

---

## 8. Ubiquitous Language

| Term | Meaning |
|---|---|
| **Judgment Broker** | The distributed capability that routes decisions between agents and humans across the Nostr relay mesh |
| **Panel** | An agent-declared control surface (kind 31400) that defines what humans can see and act on |
| **Action Request** | An agent's explicit request for a human decision (kind 31402) |
| **Action Response** | A human's signed decision (kind 31403) --- the immutable audit record |
| **Broker Bridge** | The agentbox REST/SSE proxy that bridges VisionClaw enrichment review to the management API |
| **Decision Loop** | The full cycle: agent publishes -> relay routes -> human decides -> relay routes -> agent applies |
| **Governance Event** | Any Nostr event of kinds 31400--31405 |
| **Case** | A `BrokerCase` --- the aggregate that tracks one decision from request to resolution |
| **Enrichment Proposal** | A VisionClaw knowledge graph mutation submitted for human review via the broker bridge |
| **Supersession** | Replacing a published decision with a new signed kind-31403 that references the old event and states a reason; authority-gated (§7a). Never a mutation |
| **Revoke** | A supersession by a `Reject` referencing a prior `Approve` — withdrawing a granted decision (§7a.2) |
| **Appeal** | A fresh kind-31402 citing a prior decision, reopening the case for a new human decision; does not itself overturn (§7a.2) |
| **Superseded** | Lifecycle state of a published decision that a newer authorised kind-31403 has referenced and replaced (§7a.3) |
| **Reopened** | Lifecycle state of a resolved/rejected/superseded case under review again after an appeal (§7a.3) |

---

## 9. Services

| Service | Responsibility | Owner | Status |
|---|---|---|---|
| `RelayGovernanceGate` | Validates event authorship and kind permissions at the relay level | nostr-rust-forum relay worker | Implemented |
| `PanelRenderer` | Renders `PanelDefinition` as interactive UI for human decision-makers | nostr-rust-forum forum client | Implemented |
| `ResponseSigner` | Signs `ActionResponse` with human's Nostr key (Schnorr/secp256k1) | nostr-rust-forum forum client | Implemented |
| `GovernanceEventRelay` | Subscribes to governance event kinds and routes to local handlers | agentbox relay-consumer | Implemented |
| `BrokerBridgeProxy` | Proxies VisionClaw enrichment cases to management API via REST/SSE | agentbox broker-bridge | Implemented |
| `DecisionApplicator` | Applies human decisions to agent state, closing the decision loop | agentbox orchestrator adapter | **NOT YET IMPLEMENTED** |
| `EnrichmentReviewer` | Submits KG mutations for human review through the broker protocol | VisionClaw BrokerActor | crashbug branch only |

---

## 10. Decision Flow Sequence

```
Agent (agentbox)          Relay Mesh          Human (nostr-rust-forum)
      |                       |                       |
      |-- PanelDefinition --->|                       |
      |   (kind 31400)        |-- PanelPublished ---->|
      |                       |                       |
      |-- ActionRequest ----->|                       |
      |   (kind 31402)        |-- ActionRequested --->|
      |                       |                       |
      |                       |<-- ActionResponse ----|
      |<-- DecisionReceived --|    (kind 31403)       |
      |                       |                       |
      |-- DecisionApplied --->|                       |
      |   (NOT YET IMPL)      |                       |
```

### VisionClaw Enrichment Path

```
VisionClaw               Broker Bridge            Management API         Human
    |                         |                         |                   |
    |-- EnrichmentProposed -->|                         |                   |
    |                         |-- REST POST ----------->|                   |
    |                         |                         |-- Render -------->|
    |                         |                         |<-- Decision ------|
    |                         |<-- SSE event -----------|                   |
    |<-- EnrichmentDecided ---|                         |                   |
```

---

## 11. Open Issues

1. **Decision loop closed (2026-05-22, `e1a8d716`; verified 2026-07-10).** ~~`handleGovernanceDecision()` does not exist on the orchestrator adapter.~~ **Resolved.** It is implemented at `agentbox/management-api/adapters/orchestrator/local-process-manager.js:133`: a human decision arriving via `relay-consumer.js:318` is parsed, minted a PROV-O activity/receipt URN, and dispatched to a matched running agent's stdin — or persisted to the pod governance directory for later pickup. **Invariant 7 is now SATISFIED.** Residual (not this gap): agentbox-*originated* cases still lack agent-side application on VisionClaw main — that loop-join landed 2026-07-10 (`ca145a1ce`); see closeout GOV-3 and the register v1.3 addendum.

2. **No PROV-O linkage.** Governance decisions are not linked to PROV-O activity records. The `ProvenanceRecorded` domain event has no publisher. This breaks the audit trail for downstream compliance.

3. **VisionClaw BrokerActor is branch-only.** The `BrokerActor` enrichment reviewer exists only on the `crashbug` branch. It has not been merged to main, so the VisionClaw -> broker path is not production-ready.

4. **No precedent replay.** The `Promote` and `Precedent` outcomes in `DecisionOutcome` imply a precedent system that does not yet exist. There is no mechanism for agents to query past decisions to avoid re-asking.

5. **Pod persistence is fire-and-forget.** `GovernanceEventPod` writes to `pods/<npub>/events/governance/` but there is no read-back path. The pod store is write-only for governance events.

---

## 12. Anti-Corruption Layers

### Broker Bridge (agentbox <-> VisionClaw)

The `broker-bridge.js` file acts as the ACL between VisionClaw's enrichment domain model and the governance event protocol. It translates:

- VisionClaw `EnrichmentProposal` -> Management API REST payload
- Management API decision response -> VisionClaw `EnrichmentDecision`

This prevents VisionClaw's internal model (graph nodes, edges, embeddings) from leaking into the governance protocol, and prevents governance protocol details (Nostr kinds, event tags) from leaking into VisionClaw.

### Relay Consumer (agentbox <-> Relay Mesh)

The `relay-consumer.js` file translates raw Nostr events into domain events (`DecisionReceived`, `PanelPublished`). It filters by kind range (31400--31405) and validates structural integrity before dispatching to local handlers.

---

## 13. Ownership Summary

| Substrate | Owns | Does Not Own |
|---|---|---|
| **nostr-rust-forum** | Domain model, governance.rs aggregates, relay gating, human UI, response signing | Agent lifecycle, enrichment model, pod persistence |
| **agentbox** | Agent runtime, relay subscription, pod persistence, broker bridge, decision reception | Domain model definition, human authentication, relay gating rules |
| **VisionClaw** | Enrichment proposals, KG mutation model, BrokerActor | Governance protocol, human decision surface, relay transport |
| **Nostr relay mesh** | Event ordering, transport, NIP-42 authentication | Domain semantics, decision logic, persistence |
| **Solid pods** | Durable per-identity storage of governance events | Event creation, validation, ordering |

---

## 14. Kind 31403 (ActionResponse) consumer map — verified 2026-07-10

A signed kind-31403 `ActionResponse` fans out to **three** independently
verified consumers. This is the authoritative map; each row is code-cited.

| # | Consumer | Path | Effect on a decision |
|---|---|---|---|
| a | VisionClaw `ElevationActor` | `src/actors/elevation_actor.rs:698` (`impl Handler<Decision>`) | On **Approve** of a `vc-elev-` case, drafts a class page and opens a **GitHub PR** to the corpus repo via `GitHubPRService` (`last_pr_url` set); replayed/foreign decisions are dropped |
| b | Forum relay decision projector | `nostr-rust-forum` `crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs` → `project_action_response()` (dispatched from the kind-31403 gate branch) | Routes the response through `DecisionOrchestrator` into **D1** `broker_decisions` / `broker_cases` |
| c | agentbox orchestrator | `agentbox/management-api/adapters/orchestrator/local-process-manager.js:133` `handleGovernanceDecision()`, invoked from `mcp/nostr-bridge/relay-consumer.js:318` | Parses the outcome, **mints PROV-O activity/receipt URNs**, and dispatches to a matched running **agent's stdin** — else persists to the pod governance directory |

> Citation note: an earlier recon cited the forum consumer as
> `nip_handlers.rs:1581`. That line is a kind-1059 relay-gate **test** in the
> pinned checkout; the real consumer `project_action_response()` sits near
> `:1196` and moves by revision, so it is cited here by function name, not by a
> brittle line number.

### Honest gaps landing today (2026-07-10) — sibling repos

- **VisionClaw (main).** Loop-join landed: pending-case producer + REST
  decision → forum projection (`ca145a1ce`, closes the GOV-3 residual — there is
  **no** distributed `BrokerActor` by design per ADR-130; the residual was the
  loop-join, not a missing actor). `/api/ingest/writeback` registered, closing
  the git-bridge `WriteBackSaga` 404 (`78759494e`, GOV-4).
- **agentbox (local, being pushed today).** Authority gate on the broker decide
  path (ADR-037 D2, `a70dc4fb`) — broker-bridge now keys write-back closure off
  VisionClaw's `writeback_committed` and forwards the real deciding pubkey
  (GOV-1); plus the wait-registry seam `management-api/lib/governance-decision-waiter.js`
  (reuses the one relay-consumer 31403 subscription to resolve the gate,
  fail-closed to DENY on timeout).
- **Still open.** GOV-2 (no `ConceptElevated` event — elevation is claimed at
  PR-*creation*, not merge; grep across all four repos = 0 hits),
  GOV-7 (ElevationActor opens approval cases without the documented Whelk EL++
  consistency gate), and the C4 `ShareOrchestratorActor` executor — a
  `ShareTransitionPlan` is *produced* by `src/domain/broker/broker_decision.rs:139-167`
  but has **no executor** (see the `:202` NOTE: "execution ... lives in
  `ShareOrchestratorActor` (agent C4 — follow-up sprint)"; scaffold only).
