# PRD: Judgment Broker

**Owner:** DreamLab AI
**Status:** Active
**Date:** 2026-05-22
**Version:** 1.0

## TL;DR

The Judgment Broker is not a service. It is an emergent capability produced by the coordination of four repositories: VisionClaw (case submission), agentbox (agent orchestration and MCP tooling), nostr-rust-forum (human decision UI and relay gating), and dreamlab-ai-website (panel visibility). No single repo owns the full loop. The broker exists when all four substrates correctly publish, relay, gate, decide, and route governance events across Nostr kinds 31400--31405.

65% of this capability is implemented today. The forum-side loop is closed: agents publish action requests, humans render decisions in the governance UI, signed responses propagate back through the relay. What remains is the last-mile wiring: routing decisions back into agent behaviour, giving agents MCP tools to compose governance events, merging VisionClaw's BrokerActor to main, recording provenance, and proving the full loop end-to-end.

## Goals

| Goal | Outcome |
|---|---|
| G1: Close the decision-to-agent loop | A human Approve/Reject decision (kind 31403) arriving at agentbox triggers concrete orchestrator action |
| G2: Agent-authored governance events | Agents compose and publish valid PanelDefinition (31400) and ActionRequest (31402) events via MCP tools |
| G3: BrokerActor on main | VisionClaw ships the BrokerActor on the main branch with CI coverage |
| G4: Provenance chain | Every broker decision produces a PROV-O activity record linking the decision event to the resulting agent action |
| G5: End-to-end proof | A repeatable test exercises the full mesh: VisionClaw submits case, forum renders panel, human decides, agentbox routes decision, provenance is recorded |

## Current Implementation Status

### Forum -- nostr-rust-forum (95%)

| Component | Path | Status |
|---|---|---|
| Domain model (PanelDefinition, ActionRequest, ActionResponse, BrokerCase, 6 DecisionOutcome variants, state machine) | `crates/nostr-bbs-core/src/governance.rs` (1015 lines) | Shipped |
| Leptos governance UI (active panels, pending actions, agent stats, real-time subscription) | `crates/nostr-bbs-forum-client/src/pages/governance.rs` (310 lines) | Shipped |
| Reactive panel registry store (kinds 31400--31405) | `crates/nostr-bbs-forum-client/src/stores/panel_registry.rs` (155 lines) | Shipped |
| NIP-98-gated REST API (agent registry, broker case queries) | `crates/nostr-bbs-auth-worker/src/governance_api.rs` | Shipped |
| Relay gating (agents: 31400/31402, humans: 31403) | `crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs` | Shipped |
| Response signing (human click -> kind 31403 -> relay -> D1 broker_decisions) | Auth worker + forum client | Shipped |

### Agentbox (45%)

| Component | Path | Status |
|---|---|---|
| Kind constants 31400--31405, subscribe/publish | `mcp/servers/nostr-bridge.js` (577 lines) | Shipped |
| Relay consumer (subscribes governance kinds, writes to pods, calls orchestrator) | `mcp/nostr-bridge/relay-consumer.js` (683 lines) | Shipped, but calls nonexistent `orchestrator.handleGovernanceDecision()` |
| Broker bridge (proxies BrokerActor REST, SSE relay, content enrichment) | `management-api/routes/broker-bridge.js` (598 lines) | Shipped |
| Agent unsigned event outbox -> signed publish | Nostr bridge pipeline | Shipped |
| `handleGovernanceDecision()` on orchestrator adapter | -- | **Missing** |
| MCP tools for `governance_publish_panel` / `governance_request_action` | -- | **Missing** |

### VisionClaw (40%)

| Component | Path | Status |
|---|---|---|
| BrokerActor (~400 lines, publishes 31400 on startup, 31402 on case submission) | `crashbug` branch | Not on main |
| NIP-98 + enterprise role middleware | `enterprise_auth.rs` | Shipped |
| IS-Envelope spec ownership (ADR-075) | Docs + fixtures | Shipped |
| Enterprise drawer UI | `docs/design/2026-04-17-enterprise-drawer.md` (design doc only) | Deferred (ADR-090) |

### dreamlab-ai-website (functional)

| Component | Status |
|---|---|
| Governance panel visibility in forum embed | Functional via nostr-rust-forum integration |

## Functional Requirements

### FR1: Decision Routing (agentbox)

Implement `handleGovernanceDecision(event)` on the orchestrator adapter interface. When a kind 31403 event arrives via the relay consumer:

1. Deserialise the ActionResponse from the event content.
2. Match the `case_id` to the originating BrokerCase.
3. For `Approve`: execute the pending action via the appropriate adapter slot.
4. For `Reject`: mark the case closed, notify the originating agent.
5. Emit an adapter-level span and log line per ADR-005 observability requirements.
6. All six DecisionOutcome variants must be handled at the type level; Amend/Delegate/Promote/Precedent log a warning and take no action in M1.

### FR2: Agent Governance MCP Tools (agentbox)

Expose two new MCP tools in the nostr-bridge server:

- **`governance_publish_panel`**: Accepts panel name, description, required roles, decision quorum. Composes a valid PanelDefinition, signs via agent identity, publishes as kind 31400.
- **`governance_request_action`**: Accepts panel reference, action description, context payload, urgency. Composes a valid ActionRequest, signs via agent identity, publishes as kind 31402.

Both tools must validate against the domain model in `nostr-bbs-core/src/governance.rs`. Invalid payloads fail before publish.

### FR3: BrokerActor on Main (VisionClaw)

Merge the BrokerActor from the `crashbug` branch to `main`. Requirements for merge:

- Unit tests for panel publication and case submission.
- Integration test with a local relay (can use the existing fixture relay).
- CI gate passing.
- No regressions to existing graph operations.

### FR4: Provenance Recording (agentbox)

When FR1 routes a decision to an action:

1. Mint a `urn:agentbox:activity:<scope>:decision-<id>` via `uris.js`.
2. Record a PROV-O-aligned activity linking the inbound kind 31403 event ID, the case ID, the decision outcome, and the resulting action URN.
3. Write the activity record to the pod via the beads adapter.

### FR5: End-to-End Test

A mesh smoke test (extending [mesh-smoke-test.md](protocol/mesh-smoke-test.md)) that exercises:

1. VisionClaw BrokerActor publishes a PanelDefinition (31400).
2. VisionClaw BrokerActor submits an ActionRequest (31402).
3. nostr-rust-forum relay accepts and gates the event.
4. Forum UI renders the pending action (verified via test assertion or browser automation).
5. A simulated human decision publishes kind 31403.
6. agentbox relay consumer receives the decision.
7. `handleGovernanceDecision()` routes the decision.
8. Provenance activity record exists in the pod.

## Milestones

### M1: Close the Loop

**Delivers:** FR1 + FR2

- agentbox ships `handleGovernanceDecision()` on the orchestrator adapter.
- agentbox ships `governance_publish_panel` and `governance_request_action` MCP tools.
- Agents can publish governance events and receive routed decisions.
- The agent-human-agent round-trip is functional.

### M2: BrokerActor on Main

**Delivers:** FR3

- VisionClaw merges BrokerActor from `crashbug` to `main` with tests and CI.
- The full four-repo mesh can run from released branches (no feature branches required).

### M3: Provenance and Proof

**Delivers:** FR4 + FR5

- Every routed decision produces a PROV-O activity record.
- The end-to-end mesh smoke test passes as a repeatable qualification gate.

## Out of Scope

- **Enterprise drawer UI.** Design doc exists (`docs/design/2026-04-17-enterprise-drawer.md`), intentionally deferred per ADR-090. The broker loop does not depend on it.
- **Amend/Delegate/Promote/Precedent decision workflows.** The forum domain model supports all six DecisionOutcome variants. Runtime handling beyond Approve/Reject is deferred to a follow-up PRD. M1 logs a warning for the four deferred variants.
- **Relay federation.** The broker loop operates over a single relay. Multi-relay federation is out of scope.
- **Payment enforcement.** Server-side payment gating (P1-28 in agentbox security audit) is orthogonal to governance event flow.

## Protocol Reference

- Nostr kinds: 31400 (PanelDefinition), 31401 (PanelMembership), 31402 (ActionRequest), 31403 (ActionResponse), 31404 (CaseState), 31405 (AuditTrail)
- Identity: `did:nostr:<hex-pubkey>` (shared across all four repos)
- URN grammar: `urn:agentbox:<kind>:<scope>:<local>` per ADR-013
- Mesh smoke test: [protocol/mesh-smoke-test.md](protocol/mesh-smoke-test.md)
- IS-Envelope: VisionClaw ADR-075
- Adapter contract: agentbox ADR-005
