# Status Reconciliation

**Status:** Source-reconciled current view with dated historical notes
**Reviewed:** 2026-09-07

This note separates source-supported mechanisms from historical estimates and deployed acceptance. The [compatibility matrix](compatibility-matrix.md#current-compatibility--2026-09-07) is the current cross-repository view; the [estate audit](../estate-review/2026-09-07-estate-audit.md) and its source reports provide evidence and remaining work.

## Current reconciliation — 2026-09-07

| Earlier claim | Current source-supported reading |
|---|---|
| Replay protection exists only in forum/CF pod | False for current source: VisionClaw has a process-local single-use cache and native Solid has a replay-store seam. None establishes shared replay protection across every replica/restart. |
| Canonical did:nostr is the Multikey string | Agentbox keeps `did:nostr:<64-hex>` canonical and offers the Multikey verification material alongside. Verify representation at each boundary. |
| Forum mesh has no transport and is not a relay dependency | Core/transport types, dependency, admission and fan-out planning exist. Outbound connector/accept-path joins remain deferred, so complete runtime federation is not proven. |
| Judgment Broker is 65% implemented and only on crashbug | Retire the percentage. The kernel, ACSP producer and REST case/decision surfaces are current source; BrokerActor/Neo4j was deliberately excluded. End-to-end applied decisions remain an acceptance obligation. |
| Ontology bridge currently exposes 12 MCP tools | Current `ontology-bridge.js::TOOLS` advertises eleven entries; `ontology_propose` has a dispatcher branch but is absent from that list. Historical twelve-tool records do not describe the selected working tree. |
| Ontology retrieval is only an Agentbox-to-VisionClaw proxy | Loom is also a configured backend. Its generation endpoint exists server-side, but Agentbox does not consume/verify it; provenance scope and route-specific recall remain open. |
| Beam and gluon are both delivered | Beam path is source-wired; transient attractive-edge gluon remains deferred. Hardware/client liveness needs separate evidence. |
| A complete ADR or passing source fixture proves deployment | Source, tests, configuration, image and loaded processes are different claims. Staged Agentbox corrections and pending-live register canaries retain their labels. |

Source evidence: [VisionClaw](../estate-review/2026-09-07-visionclaw-audit.md), [Agentbox](../estate-review/2026-09-07-agentbox-audit.md), [federation](../estate-review/2026-09-07-federation-audit.md), and [imported dependency scope](../estate-review/2026-09-07-imported-adr-scope.md). The ontology count was checked directly against the tracked TOOLS array, not private session settings. No new live cross-system workflow was run for this reconciliation.

## Historical notes — 2026-05-22 with May follow-ups

Everything below preserves earlier observations and estimates. Headings saying “current”, completion percentages, version strings, tool counts and “integrated” labels describe that earlier record and must not override the current reconciliation above. Published registers remain immutable; new evidence chains forward.

## VisionFlow Website Docs

| Claim area | Reconciled status |
|---|---|
| Tailwind Play CDN in ADR-001 | Superseded in implementation by local static CSS; ADR-001 now carries a 2026-05-20 amendment |
| GitHub Pages deployment via `gh-pages` push | Superseded in implementation by Pages artifact upload/deploy actions; ADR-001 now carries a 2026-05-20 amendment |
| Lighthouse and axe acceptance criteria | Axe and browser smoke checks now run through the external Chrome DevTools sidecar; Lighthouse's local-Chrome gate is superseded until it can attach cleanly to the sidecar |
| All ten website sections in nav | Static nav now links the primary PRD sections present in the page |
| Contact/demo form and hashed assets | Still deferred; tracked in [Site Verification Status](../site-verification.md) |

## VisionClaw PRD-010 / PRD-014 / PRD-015

| Document | Reconciled reading |
|---|---|
| PRD-010 DID:Nostr Mesh Federation | The early “current-state evidence” section intentionally records audit findings from before Phase 0. Later sections and memory indicate crypto fixes/absorption work progressed, but federation remains “implementation in progress.” Treat PRD-010 as historical audit plus target architecture, not proof that all listed gaps are still open. |
| PRD-014 Ecosystem Productionisation | Completion checklist marks the 60% -> 80% productionisation work complete, including zero critical gaps, substrate CI, runbooks, fixture sync, health aggregation, and security hardening. Remaining 80% -> 100% items are mesh/runtime/observability/accessibility/release-process work. |
| PRD-015 Ecosystem Code Hygiene | Success criteria show many hygiene items completed. Cross-substrate NIP-98 convergence remains explicitly deferred pending WASM-compatible shared surfaces. |

### Historical high-confidence/open-item list

| Item | Why it remains open |
|---|---|
| Mesh federation defaults vary by substrate | nostr-rust-forum (3.0.0-rc11) and dreamlab-ai-website default federated; agentbox and solid-pod-rs default standalone. Federation claims should be substrate-specific, not ecosystem-wide. |
| IS-Envelope ownership resolved | VisionClaw owns the spec (ADR-075), JSON Schema, and 11 test vectors. agentbox implements runtime decode/dispatch. No longer an open item. |
| Shared NIP-98 implementation | PRD-015 explicitly leaves this unchecked |
| CF Workers/native pod convergence | Forum docs still describe two-tier behavior and portability blockers |
| Judgment Broker runtime | Judgment Broker is 65% implemented as a distributed system. Forum: closed decision loop (governance.rs 1015 lines, panel registry, response signing). Agentbox: event relay + broker-bridge proxy. VisionClaw: BrokerActor on crashbug branch. Gaps: decision→agent application, agent MCP tools, provenance. |
| Agent Control Surface completeness | Kinds 31400-31405 are fully implemented in nostr-rust-forum (`governance.rs`) and dreamlab-ai-website (`/governance` dashboard). agentbox publishes and subscribes via nostr-bridge. Status: `integrated`. |
| Ontology bridge | 12 MCP tools in agentbox (`ontology-bridge.js`) proxy SPARQL to VisionClaw Oxigraph. Shipped cross-substrate integration not previously reflected in coordination docs. |
| BC20 anti-corruption layer | **No longer paper-only (2026-05-29).** Real bidirectional code: agentbox `management-api/lib/bc20-provenance-bridge.js` (20 tests) holds the closed kind map (`activity`⇄`execution`, `agent`⇄`did:nostr`, `thing`⇄`kg`, `memory`⇄`concept`) + durable `UrnMapping`; VisionClaw mirrors the ingest schema (`src/agent_events/schema.rs`). Older PRD-010 "paper-only" language is historical. |
| Embodied agent loop (`/wss/agent-events`) | **Phase 2a wired and cargo-verified (2026-05-29).** agentbox emits canonical `notifications/agent_action` (identity preserved per ADR-013) via one builder; VisionClaw authenticates the WS upgrade, validates, and publishes to a process-global broadcast hub (7/7 tests). **Phase 2b beam render code-wired (2026-05-30).** VisionClaw `agent_beam_actor.rs` (spawned in `app_state.rs`) subscribes to the hub and emits the `0x23` agent-action frame; the client decodes it in `TransientBeamsLayer.tsx` with semantic colour/shape/motion from `semanticEncoding.ts` (`a190fa3ab` + `a099d58fe`, unit-verified, visual E2E pending host client rebuild). Still open: gluon transient-edge render, live did:nostr-keyed actor nodes, `ConceptElevated`, `:9500` state-poll cutover. Contract: agentbox ADR-014 + VisionClaw ADR-059. |
| Gluon render mechanism | **Corrected (2026-05-29).** The gluon is a **transient attractive edge** on the spring kernel, not a per-node `class_charge` modulation (that buffer is bulk ontology-clustering metadata loaded at construction, with no per-node update path). VisionClaw ADR-059 §4 / agentbox PRD-014 §8 carry the correction. |
| Browser verification sidecar reachability | The expected endpoint is `browsercontainer:9223` from Docker-network runtimes or `localhost:9222` from the host; sidecar CDP was verified on 2026-05-21 after resolving `browsercontainer` to its Docker-network IP |
