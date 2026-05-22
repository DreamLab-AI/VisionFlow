# DDD: Judgment Broker Bounded Context

**Status:** Living document
**Date:** 2026-05-22
**Scope:** Cross-substrate decision loop between agents and humans

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

5. **Decision immutability.** A `DecisionOutcome` is immutable once published. The Nostr event is the audit trail. There is no "undo" --- only new events that supersede.

6. **Relay as ordering authority.** The relay mesh is the source of truth for governance event ordering, not any single substrate's local store. Pod persistence is a cache, not the canonical timeline.

7. **Decision loop closure.** `handleGovernanceDecision()` on the orchestrator adapter MUST exist before the decision loop can be considered closed. **This is currently missing** --- the loop is open-ended on the agent side.

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

1. **Decision loop is not closed.** `handleGovernanceDecision()` does not exist on the orchestrator adapter. Human decisions arrive at agentbox via `relay-consumer.js` but are persisted to pods without being acted upon. This is the highest-priority gap.

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
