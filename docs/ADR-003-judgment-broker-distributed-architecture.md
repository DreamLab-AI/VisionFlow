# ADR-003: Judgment Broker Distributed Architecture

**Status:** Accepted (2026-07-03: 2 of 3 identified gaps closed — see amendment below)
**Date:** 2026-05-22
**Decision Owners:** DreamLab AI maintainers
**Related:** [ADR-002 Ecosystem Alignment Governance](ADR-002-ecosystem-alignment-governance.md), [ADR-075 IS-Envelope (VisionClaw)](https://github.com/DreamLab-AI/VisionClaw), [agentbox ADR-005 Pluggable Adapters](https://github.com/DreamLab-AI/agentbox/blob/main/docs/reference/adr/ADR-005-pluggable-adapter-architecture.md), [closeout final-design](closeout/final-design.md)

> **2026-07-03 closeout amendment.** The "35% gap" inventory below is stale: 2 of its 3 gaps are closed. Gap 1 (decision application) is **implemented** — `handleGovernanceDecision` mints PROV-O activity/receipt URNs and dispatches to agent stdin or the pod (`agentbox/management-api/adapters/orchestrator/local-process-manager.js:122`, `stdio-bridge.js:72`), merged 2026-05-22 (`e1a8d716`). Gap 2 (agent MCP tools) is **shipped** — five governance tools (`publish_panel`/`request_action`/`update_panel`/`retire_panel`/`list_decisions`) ship in `agentbox/mcp/mcp.json`. Only Gap 3 (the distributed BrokerActor merge, D5) remains open, and note that main shipped a *different* architecture — an inline decide/inbox handler + ADR-110 ElevationActor with a real Oxigraph write-back — so the distributed BrokerActor on `crashbug` is now an unmerged alternative, not the only path to closure. The "65% implemented" framing understates current completion.

## Context

The Judgment Broker mediates human-in-the-loop decisions for autonomous agents. Previous documentation described it as specification-only, but runtime research (2026-05-22) found that 65% of the broker is already implemented, distributed across three repositories and the Nostr relay mesh.

The implementation is not concentrated in one place because the broker's responsibilities span three distinct domains: human decision-making (forum), agent orchestration (agentbox), and semantic enrichment (VisionClaw). Nostr event kinds 31400-31405 serve as the wire protocol binding these substrates together.

### Current Implementation Inventory

| Substrate | Component | Lines | Maturity |
|---|---|---|---|
| nostr-rust-forum | Domain model (`governance.rs`) | 1015 | integrated |
| nostr-rust-forum | Panel rendering (Leptos) | 310 | integrated |
| nostr-rust-forum | Panel registry | 155 | integrated |
| nostr-rust-forum | Response signing, relay gating, REST API, D1 persistence | -- | integrated |
| agentbox | Relay subscription/publishing (`relay-consumer.js`) | 683 | standalone |
| agentbox | Broker-bridge REST/SSE proxy (`broker-bridge.js`) | 598 | standalone |
| agentbox | Kind constants, event pod storage | -- | standalone |
| VisionClaw | BrokerActor (crashbug branch) | ~400 | scaffolded |
| VisionClaw | IS-Envelope spec (ADR-075) | -- | integrated |

### Identified Gaps (original inventory — see 2026-07-03 amendment above for current status)

1. **Decision application.** ~~`handleGovernanceDecision` on the agentbox orchestrator adapter is stubbed.~~ **RESOLVED (2026-05-22, `e1a8d716`).** `handleGovernanceDecision` is fully implemented: it parses the decision, mints PROV-O activity/receipt URNs, and dispatches to agent stdin or persists to the pod.
2. **Agent MCP tools.** ~~No composable tool interface exists for agents to publish panels or request actions.~~ **RESOLVED.** Five governance MCP tools ship in `agentbox/mcp/mcp.json` (`publish_panel`, `request_action`, `update_panel`, `retire_panel`, `list_decisions`).
3. **BrokerActor merge.** The VisionClaw BrokerActor lives on the `crashbug` branch and has not been merged to main. **OPEN** — but main shipped an inline decide/inbox + ElevationActor architecture instead (with a real Oxigraph write-back), so this is now a choice between merging the distributed actor or descoping it, not a blocking gap in the decision loop.

## Decision

The Judgment Broker is distributed by design. No single repository owns or should own the complete broker. Each substrate owns its natural responsibility within the decision loop.

### D1: Distribution is intentional

The broker is a distributed capability that emerges from the coordination of four substrates (nostr-rust-forum, agentbox, VisionClaw, Nostr relay mesh). This is a deliberate architectural choice, not an accident of incremental development.

### D2: Kinds 31400-31405 are the wire protocol

nostr-rust-forum owns the kind definitions and relay gating rules. Agents (agentbox) write kinds 31400 (PanelDefinition) and 31402 (ActionRequest). Humans (forum) write kind 31403 (Decision). The relay enforces this separation via NIP-42/NIP-98 auth.

### D3: The decision loop spans three repos plus relay

```
Agent (agentbox) --[31400/31402]--> Relay ---> Forum (nostr-rust-forum)
                                                    |
                                              Human decides
                                                    |
Forum (nostr-rust-forum) --[31403]--> Relay ---> Agent (agentbox)
                                                    |
                                         VisionClaw enters when
                                         enrichment/mutation of
                                         the knowledge graph is
                                         involved
```

### D4: broker-bridge is a convenience layer

The broker-bridge in agentbox proxies VisionClaw's enrichment review to the management API, bridging WebSocket to SSE. It is not the canonical path. The canonical path is relay-mediated event flow.

### D5: BrokerActor should be merged to main

VisionClaw's BrokerActor (crashbug branch, ~400 lines) publishes PanelDefinition and ActionRequest events. It should be merged to main and stabilised as part of closing the 35% gap.

### D6: Decision application is the critical gap

**Resolved (2026-05-22).** The `handleGovernanceDecision` method on the agentbox orchestrator adapter is implemented, minting PROV-O provenance and dispatching decisions to agent stdin or the pod. The original premise that decisions were "silently dropped" no longer holds. (Note: agentbox-*originated* cases still route to a BrokerActor absent from main — see GOV-3 in the closeout register — but forum-originated decisions are applied.)

### D7: Agent MCP tools for governance are needed

**Resolved.** Five composable governance MCP tools ship in `agentbox/mcp/mcp.json` (`publish_panel`, `request_action`, `update_panel`, `retire_panel`, `list_decisions`), superseding the outbox-directory mechanism as a stable interface.

## Alternatives Considered

| Alternative | Verdict | Rationale |
|---|---|---|
| Single-service broker | Rejected | Creates a centralised bottleneck and fights the federated architecture. Every decision would route through one service regardless of domain. |
| Broker in VisionClaw only | Rejected | The human decision surface (panel rendering, response signing, relay gating) naturally belongs to the forum. VisionClaw has no user-facing UI. |
| Broker in agentbox only | Rejected | Enrichment review requires knowledge graph context that only VisionClaw holds. Duplicating the KG into agentbox would violate the single-source-of-truth principle. |

## Consequences

### Positive

- Each substrate evolves independently within its domain responsibility.
- No single point of failure in the decision loop.
- Naturally scales with federation -- adding relay nodes or forum instances extends capacity without architectural change.
- Aligns with the existing `did:nostr` identity mesh; no new identity primitives required.

### Tradeoffs

- The decision loop spans three repositories plus relay, making end-to-end integration testing harder. The mesh smoke test (ADR-002) must be extended to cover the full broker loop.
- Debugging a failed decision requires correlating events across substrates. Observability must include relay event tracing.

### Risks

- ~~The 35% implementation gap means decisions currently reach agentbox but are not applied.~~ **Superseded (2026-07-03):** D6 and D7 are resolved; forum-originated decisions are applied with provenance. The residual risk is narrower — agentbox-originated cases still lack a consumer on main (closeout GOV-3), and the flagship elevation loop has no `ConceptElevated` closing event (GOV-2).
- BrokerActor remaining on a feature branch (D5) means the distributed event-publishing path is not part of any release; main runs the inline decide/inbox + ElevationActor path instead.

## Substrate Ownership Summary

| Responsibility | Owner | Consumers |
|---|---|---|
| Kind definitions (31400-31405) | nostr-rust-forum | agentbox, VisionClaw |
| Relay gating rules | nostr-rust-forum | all substrates |
| Human decision surface | nostr-rust-forum | -- |
| Relay subscription/publishing | agentbox | -- |
| Decision application (orchestrator adapter) | agentbox | VisionClaw (via enrichment feedback) |
| Enrichment proposals, KG mutation gating | VisionClaw | agentbox (via broker-bridge) |
| BrokerActor event publishing | VisionClaw | nostr-rust-forum, agentbox |
| Transport | Nostr relay mesh | all substrates |
