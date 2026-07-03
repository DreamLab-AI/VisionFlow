# PRD: Mandate-at-Grant Governance

**Owner:** DreamLab AI
**Status:** Speculative
**Date:** 2026-06-23
**Version:** 0.1 (speculative — design exploration, not committed to implementation)
**Provenance:** Cross-reference of "The Moment of Authorisation" principle (mandate-at-grant) against VisionFlow ecosystem audit (5-agent mesh, 2026-06-23). Extends ADR-004 harness engineering framework and ADR-122 two-speed governance routing.

## TL;DR

The VisionFlow Judgment Broker gates agent behaviour at **action time** — an agent publishes an ActionRequest (kind 31402) describing what it *wants to do*, and a human approves or rejects. The "Moment of Authorisation" principle argues this is insufficient: **authorization risk begins at the moment access is granted (the mandate), not when the agent acts**. Three channels grant mandates: accounts (identity credentials), files (pod/filesystem ACLs), and capabilities (MCP tools, adapter slots).

Our ecosystem currently grants mandates silently. When `spawnAgent()` is called, the agent receives a `did:nostr` identity, access to all registered MCP tools, pod write paths, and adapter slots — with no governance event, no capability graph, no blast radius analysis, and no stakeholder review. The harness engineering framework (ADR-004) provides the guide/sensor/pairing discipline for regulating agent *execution*, but has no equivalent for regulating agent *provisioning*.

This PRD explores extending the Judgment Broker and harness template system with **mandate-time governance**: the ability to inspect, gate, and audit what agents CAN do, not just what they're asking to do right now.

## Goals

| Goal | Outcome |
|---|---|
| G1: Mandate-time governance event | When an agent is provisioned or re-provisioned, a governance event declares what was granted, the blast radius, and affected stakeholders |
| G2: Three-channel mandate inventory | Every harness template declares the account, file, and capability mandates required by its topology |
| G3: Capability composition graph | MCP tools and adapter slots are modelled as a directed graph with transitive reach analysis, not a flat list |
| G4: Blast radius analysis | Before an agent receives capabilities, the system calculates what data, systems, and stakeholders are reachable through capability composition |
| G5: Multi-stakeholder mandate review | Certain mandate grants require acceptance from multiple stakeholder roles, not just the provisioner |
| G6: Mandate-aware precedent system | The precedent system (PRD-harness-engineering M6) evaluates mandate-level patterns, auto-approving routine re-provisioning and flagging novel capability combinations |

## Current State — Mandate Gaps

*Findings from 5-agent mesh examination (6 agents, 112 tool invocations) across agent bootstrap, governance bridge, file/pod access, MCP tool surface, and harness template schema.*

### Account Channel (did:nostr identity)

| Aspect | Current | Gap | Evidence |
|---|---|---|---|
| Identity provisioning | `signerFromHex()` creates signer with `{ sign, skBytes, pubkey }` granting full Schnorr signing authority | No governance event at identity creation. Agent can sign ANY Nostr kind — no kind restriction on the keypair. | `per-user-agent.js` lines 142-146, `run-per-user-agent.cjs` lines 142-150 |
| NIP-42 AUTH | Signer registered on NostrBridge via `setAuthSigner()` BEFORE subscribing | Relay-level authentication granted silently before any governance event | `per-user-agent.js` line 800 |
| NIP-98 token minting | `nip98Token(signer, url, method)` available to agent with no per-invocation gate | Agent can forge HTTP auth headers for ANY URL path — no path restriction at the signer level | `per-user-agent.js` lines 134-165 |
| Pod-signer key sharing | `pod-signer.js` loads one key per agentbox stack | Multiple agents authenticate as the SAME `did:nostr` — the per-agent mandate model assumes distinct identities but the signing infrastructure collapses them | `pod-signer.js` constructor |
| Event attribution | `agent_event_auth` defaults to `'off'` | Any caller of `POST /v1/agent-events/emit` can attribute actions to arbitrary agents. `source_urn` is caller-asserted, not verified. | `agentbox.toml` line 132, `agent-event-auth.js` line 25 |
| Identity rotation | Not implemented | No revocation event. Orphaned identities persist indefinitely. | — |

### File Channel (pod/filesystem ACLs)

| Aspect | Current | Gap | Evidence |
|---|---|---|---|
| Mandate system | `mandate.js` creates scoped ACL grants via WAC Turtle fragments. `signMandate()` wraps in NIP-33 kind 30078 for revocability. | `createMandate()` emits NO governance event. The signed Nostr event is replaceable but no ActionRequest surfaces it for human review before the ACL is PUT. | `mandate.js` lines 99-127 |
| WAC evaluation | Solid LDP WAC is default-deny. Every pod request evaluated against `acl:agent` grants (strong action-time gate). | ACLs are configured, not governed. No governance event when an ACL is modified. `acl:Control` is issuable without elevated governance — an agent with Control can rewrite its own ACL. | `mandate.js` — no `acl:Control` differentiation |
| Process.env inheritance | `spawnAgent()` passes `{ ...process.env, ...(spec.env) }` to child process | Spawned agents inherit ALL environment secrets: `MANAGEMENT_API_KEY`, all Nostr relay credentials, all API keys. | `local-process-manager.js` line 47 |
| API key passthrough | `ProcessManager.spawnTask()` explicitly passes 8 API keys | `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `GOOGLE_GEMINI_API_KEY`, `OPENROUTER_API_KEY`, `GITHUB_TOKEN`, `BRAVE_API_KEY`, `CONTEXT7_API_KEY` — all passed with no governance event | `process-manager.js` lines 173-184 |
| Mandate expiry | `createMandate()` defaults `expiresAt` to null | No maximum TTL enforcement. Mandates without expiry grant permanent access until manually revoked. | `mandate.js` |
| Mandate-ACL drift | Mandate record (Nostr event) and pod ACL (`.acl` Turtle) are separate artefacts | A revoked mandate can leave its ACL in place. Suspension/archival endpoints return 501. No reconciliation daemon. | — |

### Capability Channel (MCP tools, adapter slots)

| Aspect | Current | Gap | Evidence |
|---|---|---|---|
| MCP tool surface | 108+ tools across 25+ MCP servers. All globally available to every agent. | No per-agent tool scoping. A doc-planner agent sees identical tools to a governance agent: `governance_publish_panel`, `ontology_axiom_add`, `browser_evaluate_script`, `precedent_promote`. | `.mcp.json` (workspace-global), `skills/mcp.json` (25+ servers) |
| ALLOWED_TOOLS | ProcessManager constrains claude-flow tasks to `Read/Write/Edit/Bash(grep|find|ls|cat|head|tail|wc)` | This is the ONLY capability-constraining mechanism in the codebase. It applies only to the claude-flow provider path, not to per-user agents or direct MCP tool invocation. | `process-manager.js` ALLOWED_TOOLS constant |
| Precedent namespace | `memory_store` has no namespace ACL. Any agent can write to `governance-precedents` namespace. | An agent can inject synthetic precedent records that auto-approve future governance decisions. This bypasses human-in-the-loop entirely. | `memory_store` tool — no namespace write restriction |
| Agent spawn hook | `agent-action-hooks.js` emits `AgentActionType.CREATE` event | This is OBSERVABILITY ONLY (emitted AFTER spawn completes). It cannot block or condition the spawn. Not a governance gate. | `agent-action-hooks.js` lines 222-236 |
| Adapter slots | 5 adapter types globally registered in `agentbox.toml` | No per-agent adapter restriction. An agent with the orchestrator adapter can spawn other agents. | `agentbox.toml` adapter configuration |
| Autonomous DM loop | Per-user agent processes incoming DMs through LLM and publishes signed, gift-wrapped responses | The entire `_onDM → _think → _sendDm` cycle runs without human approval. An LLM hallucination becomes an attributed, signed message from the agent's `did:nostr` identity. | `per-user-agent.js` autonomous reply pipeline |

### Composition Risk Matrix (ranked by blast radius)

| Risk | Chain | Composed Effect | Evidence |
|---|---|---|---|
| **CRITICAL** | `memory_store(governance-precedents)` → `precedent_match(future ActionRequest)` → auto-approved | **Self-Approving Governance Loop.** Agent writes synthetic precedent, future matching governance decisions auto-apply. Human-in-the-loop bypassed entirely. Subverts the governance layer itself. | `memory_store` has no namespace ACL; `precedent_match` reads same namespace |
| **CRITICAL** | `spawnAgent(spec)` → child inherits `process.env` → unrestricted network | **Full Process Exfiltration.** ALLOWED_TOOLS constrains Claude CLI tool use but NOT the raw process's syscall-level capabilities. Spawned process can POST secrets to any HTTP endpoint or publish via Nostr. | `local-process-manager.js` line 47, `process-manager.js` lines 173-184 |
| **HIGH** | `createMandate(acl:Control)` → agent PUT modified ACL → `acl:default` cascades | **Self-Escalating Pod Access.** Agent with `acl:Control` rewrites its own ACL, grants access to other agents, removes restrictions. Invisible to mandate-level revocation due to mandate-ACL drift. | `mandate.js` — no `acl:Control` differentiation |
| **HIGH** | `agent_event_auth=off` → POST arbitrary `source_urn` → WebSocket to VisionClaw | **Governance Visualisation Spoofing.** Fabricated agent activity in operator's situational awareness display. Could mask real attacks or create phantom activity. | `agentbox.toml` line 132, `agent-event-auth.js` line 25 |
| **HIGH** | `callLlm()` + `bridge.publish()` + NIP-59 gift wrap | **Autonomous Message Fabrication.** Agent autonomously reasons, produces responses published as encrypted DMs. LLM hallucination → attributed, signed, gift-wrapped message from agent's `did:nostr` identity. | `per-user-agent.js` `_onDM`/`_think`/`_sendDm` |
| **HIGH** | `email_gateway.refresh_inbox()` → `browser_navigate(attacker-URL)` → `browser_fill_form(content)` | **Email-to-Browser Exfiltration.** 36 browser automation tools + email gateway read compose into a full exfiltration pipeline. No per-agent URL allowlist. | `.mcp.json` — both servers registered globally |
| **HIGH** | `AGENTBOX_ONTOLOGY_DIRECT_LOAD=true` → `ontology_axiom_add` | **Ontology Bypass.** Process-level env var disables Whelk consistency, human approval, and PR workflow for ALL agents in the session. | Global env var, not per-request |
| **MEDIUM** | `governance_publish_panel` → `governance_request_action` → `governance_list_decisions` | **Self-Referencing Governance.** Agent controls both the question and the monitoring. Human still decides, but agent frames the decision space. | All three tools globally available |
| **MEDIUM** | `memory_store` + `memory_search` (any namespace) | **Cross-Session State Injection.** Agent writes to memory in one session, retrieves in another. Persistent state without governance. By design, but no audit of what's stored. | `memory_store` — no write audit trail |

## Milestones

### M1: Mandate Inventory in Harness Templates (P1)

Extend the harness template schema with a `mandates` section declaring required capabilities per topology.

| Requirement | Acceptance Criteria |
|---|---|
| FR1.1: `mandates` schema section | JSON Schema extended with `mandates: { account: AccountMandate, file: FileMandate[], capability: CapabilityMandate[] }`. Validated by `harness_validate` MCP tool. |
| FR1.2: Three-channel declaration | Each channel declares: what is granted, why it is needed, the blast radius (reachable data/systems), and affected stakeholders. |
| FR1.3: Canonical templates updated | All 4 existing templates (`agentbox-agent-task`, `visionclaw-enrichment`, `governance-decision`, `pod-mutation`) extended with `mandates` section. |
| FR1.4: `harness_inspect` returns mandates | Agents querying their active template receive the mandate inventory alongside guides and sensors. |

### M2: Capability Composition Graph (P1)

Model MCP tools and adapter slots as a directed graph with transitive reach analysis.

| Requirement | Acceptance Criteria |
|---|---|
| FR2.1: Tool taxonomy | Each MCP tool classified: read-only, write/mutation, governance-write, infrastructure-modify. Classification stored in harness template or separate registry. |
| FR2.2: Composition edges | Edges declared between tools that can chain: e.g. `governance_publish_panel → governance_request_action` (creates-then-submits). |
| FR2.3: Transitive reach calculation | Given an agent's tool set, compute the full set of reachable systems and data stores through tool composition. |
| FR2.4: Blast radius MCP tool | New `mandate_blast_radius` MCP tool accepts a set of capabilities and returns the reachable graph. |

### M3: Mandate-Time Governance Event (P2)

Emit a governance event when an agent is provisioned or re-provisioned with new capabilities.

| Requirement | Acceptance Criteria |
|---|---|
| FR3.1: MandateGranted event | New Nostr event kind (proposed: 31406 or extension of kind 31400) emitted when `spawnAgent()` or equivalent is called. Carries: agent identity, granted capabilities (3-channel), blast radius summary, requesting context. |
| FR3.2: Mandate review panel | Kind 31400 PanelDefinition extended (or new panel type) that renders the mandate inventory for human review. Shows blast radius, composition risks, affected stakeholders. |
| FR3.3: Gate modes | Three modes: `audit` (emit event, don't block), `advisory` (emit event, warn on high-risk), `blocking` (require human approval before agent receives capabilities). Default: `audit`. |
| FR3.4: Mandate provenance | Each mandate grant mints a `urn:agentbox:activity:mandate:grant-<id>` PROV-O record linking the mandate event to the provisioning context and the granting authority. |

### M4: Per-Agent Capability Scoping (P2)

Restrict MCP tool access per agent based on topology and harness template.

| Requirement | Acceptance Criteria |
|---|---|
| FR4.1: Template-driven tool allowlist | `spawnAgent()` reads the active harness template for the agent's topology and restricts MCP tool access to the declared `mandates.capability` set. |
| FR4.2: Tool access sensor | Computational sensor that detects tool invocations outside the agent's declared mandate. Fail mode: `advisory` (log violation) or `blocking` (reject call). |
| FR4.3: Adapter slot restriction | Agent adapter access restricted to slots declared in the template. An enrichment agent should not have access to the orchestrator adapter. |
| FR4.4: Principle of least privilege audit | Automated scan comparing agents' actual tool usage (from learning service) against their declared mandates. Report over-provisioned and under-used capabilities for mandate tightening. |

### M5: Multi-Stakeholder Mandate Review (P3)

Certain mandate grants require acceptance from multiple stakeholder roles.

| Requirement | Acceptance Criteria |
|---|---|
| FR5.1: Stakeholder mapping | Each mandate channel maps to affected stakeholders: account → identity owner + security; file → data owner + compliance; capability → tool owner + governance. |
| FR5.2: Quorum requirement | Mandate grants above a risk threshold require approval from ≥2 distinct stakeholder roles via kind 31403 responses. |
| FR5.3: Escalation chain | If quorum is not reached within a timeout, the mandate is escalated or the agent is provisioned with reduced capabilities (degraded mandate). |

### M6: Mandate-Aware Precedent System (P3)

Extend the precedent system to evaluate mandate-level patterns.

| Requirement | Acceptance Criteria |
|---|---|
| FR6.1: Mandate precedents | Approved mandate grants stored as precedents in RuVector (`governance-mandates` namespace). |
| FR6.2: Re-provisioning auto-approval | When an agent is re-provisioned with the same capability set, the system matches against mandate precedents and auto-approves if similarity ≥ threshold. |
| FR6.3: Novel combination detection | If the requested mandate includes a capability combination not seen in any precedent, the system flags for human review regardless of individual capability precedents. |

## Architectural Alignment

### Relationship to ADR-004 (Harness Engineering)

ADR-004 established guides, sensors, pairings, and harness templates as the regulatory surface for agent *execution*. Mandate-at-grant extends this to agent *provisioning*:

| ADR-004 Concept | Execution Domain | Provisioning Domain (this PRD) |
|---|---|---|
| Guide | `hooks.pre` — shapes behaviour before acting | `mandates.capability` — shapes what the agent CAN do |
| Sensor | `hooks.validate` — checks behaviour after acting | Mandate audit sensor — checks what was granted vs what's used |
| Pairing | Guide-sensor binding per execution concern | Mandate-sensor binding per grant channel |
| Template | Regulation bundle per topology | Extended with mandate inventory per topology |
| Fitness gate | Computational sensor → CI merge block | Mandate composition check → provisioning block |

### Relationship to ADR-122 (Two-Speed Governance)

ADR-122 routes governance by **epistemic class** (structural truth → human gate; volatile facts → automatic). The mandate-at-grant equivalent routes by **grant channel risk**:

| ADR-122 Lane | Epistemic Routing | Mandate Routing (this PRD) |
|---|---|---|
| L1 (human-gated) | TBox / structural truth | Novel capability combinations, cross-substrate mandates, governance tool grants |
| L2 (automatic + consistency) | Volatile ABox facts | Routine re-provisioning matching a standing precedent |
| L3 (automatic) | Derived output | Read-only tool grants, status queries |

### Relationship to Existing Governance Kinds

| Kind | Current Purpose | Mandate Extension |
|---|---|---|
| 31400 PanelDefinition | Declares a decision panel for human UI | Extended with `mandate_inventory` field showing what the panel's agent CAN do |
| 31401 PanelState | Current panel state | No change |
| 31402 ActionRequest | Agent requests a specific action | Extended with `current_mandates` field showing what the requesting agent already has |
| 31403 ActionResponse | Human decision | Extended with `mandate_adjustment` field for grant/revoke/modify mandate |
| 31404 PanelUpdate | Incremental panel changes | No change |
| 31405 Precedent | Standing policy | Extended to cover mandate precedents |
| **31406 MandateGrant** (new) | **Declares that an agent received capabilities** | **New kind: carries 3-channel mandate inventory + blast radius + stakeholder map** |

## Existing Mandate Infrastructure

The ecosystem already has partial mandate-at-grant infrastructure that this PRD extends rather than replaces:

| Component | What It Does | What It Lacks |
|---|---|---|
| `mandate.js` + `signMandate()` | Creates scoped WAC ACL grants for pod containers. Signs as NIP-33 kind 30078 (revocable). Renders Turtle fragments via `mandateToAclTurtle()`. | No governance event emitted. No blast radius analysis. `acl:Control` not differentiated. No expiry enforcement. No reconciliation with actual ACLs. |
| `ALLOWED_TOOLS` whitelist | Constrains claude-flow spawned tasks to a narrow tool set (Read/Write/Edit/Bash subset). | Applies only to claude-flow provider path. Per-user agents and direct MCP invocations are ungated. |
| `agent-action-hooks.js` spawn hook | Emits `AgentActionType.CREATE` telemetry event after spawn. Visible in VisionClaw visualisation. | Post-spawn observability only. Cannot block. Not a governance gate. |
| `agent_registry` D1 table | Maps pubkey → active status. Relay kind gate rejects events from unregistered pubkeys. | No Nostr event emitted when a pubkey is activated. No capability scope declaration. |
| S6 WoT Thing Description | Models MCP capabilities as W3C WoT Actions/Properties/Events. Machine-readable capability surface. | Emit-only (`direction: 'emit'`). Not consumed by governance decision flow. Disconnected from kinds 31400-31405. |
| S8 Payment mandate (ODRL) | W3C Verifiable Credential with ODRL Permission vocabulary for financial authorisation. Scoped by `amount_sats`, rate limits. | Completely disconnected from Nostr governance events. Uses VC vocabulary, not the Agent Control Surface protocol. |

The mandate-at-grant model unifies these partial systems under the harness engineering framework, connecting them to the governance event protocol via the proposed kind 31406 MandateGranted event.

## Immediate Remediation (Configuration Changes, No New Abstractions)

These items can be addressed immediately by changing defaults and adding guards, independent of the full mandate-at-grant architecture:

| Item | Change | Risk Closed | Effort |
|---|---|---|---|
| IR1: `agent_event_auth` default | Change from `'off'` to `'nip98'` in `agentbox.toml` | Event attribution spoofing (Chain 4) | Config change + integration test |
| IR2: Precedent namespace protection | Add `governance-precedents` to a protected-namespace list in `memory_store` tool. Reject writes from non-admin callers. | Self-approving governance loop (Chain 1) | ~20 lines in `ruvector-mcp.cjs` |
| IR3: Explicit env var passthrough | Replace `{ ...process.env, ...(spec.env) }` in `spawnAgent()` with explicit allowlist from template mandate | Full process exfiltration (Chain 2) | Modify `local-process-manager.js` + `process-manager.js` |
| IR4: Mandatory mandate expiry | Add `MAX_MANDATE_TTL` (default 30 days) to `mandate.js`. Reject `createMandate()` with null expiry. | Indefinite silent access (mandate-ACL drift) | ~10 lines in `mandate.js` |
| IR5: Spawn-time telemetry event | Emit a structured telemetry event BEFORE spawn (not after) carrying the capability surface. Not a gate — observability only. | Visibility gap (no record of what was granted) | Modify `agent-action-hooks.js` |

## Non-Goals

- **Runtime capability enforcement.** This PRD governs what agents RECEIVE, not what they DO with it. Runtime enforcement is ADR-004's domain (guides, sensors, pairings). The boundary: mandate-at-grant controls the provisioning surface; harness engineering controls the execution surface.
- **Zero-trust agent architecture.** Full zero-trust (verify every call, no standing access) is architecturally possible but premature. The mandate model is a step toward it — making grants visible and auditable — without requiring per-call verification.
- **Automated capability shrinking.** Auto-revoking unused capabilities is a future optimisation. This PRD makes over-provisioning visible; remediation is human-directed.
- **Cross-organisation mandates.** Mandate governance applies within the VisionFlow ecosystem. External agent federation would require a mandate interchange protocol not designed here.

## Risk / Reward Assessment

| Dimension | Score | Notes |
|---|---|---|
| **Reward: security posture** | 8/10 | Making silent capability grants visible is a fundamental security improvement. Most agent security incidents in the industry trace to over-provisioning, not to action-level failures. |
| **Reward: governance efficiency** | 6/10 | Mandate precedents reduce human steering burden for routine re-provisioning. But the initial mandate review adds friction to agent bootstrap. |
| **Reward: architectural clarity** | 9/10 | Extends the guide/sensor model to provisioning — conceptually clean, uses existing abstractions. |
| **Risk: implementation complexity** | 5/10 | Capability graph and blast radius analysis are genuinely hard to get right. Composition chains in a dynamic MCP tool registry create a moving target. |
| **Risk: performance impact** | 3/10 | Mandate checks happen at spawn time (infrequent), not per-action. Negligible latency impact. |
| **Risk: developer friction** | 4/10 | Adding tools to `.mcp.json` now requires mandate inventory updates. Mitigated by making `audit` mode the default (observe, don't block). |
| **Overall risk/reward** | **7.5/10** | High security and architectural value. Medium implementation complexity. Low runtime cost. |

## Success Metrics

| Metric | Baseline (2026-06-23) | Target |
|---|---|---|
| Mandate grants with governance event | 0% (all silent) | ≥90% of agent provisioning emits MandateGrant |
| Capability composition coverage | 0% (no graph) | ≥80% of MCP tools classified and composition-mapped |
| Over-provisioned agents detected | Unknown | ≥50% of agents have mandate tightening recommendations |
| Mandate precedent auto-approval rate | 0% | ≥60% of routine re-provisioning auto-approved |
| Stakeholder coverage | Single approver | ≥2 stakeholder roles for high-risk mandates |
| Blast radius calculation accuracy | N/A | ≥90% of tool composition chains correctly mapped |

## Dependencies

| Dependency | Owner | Status |
|---|---|---|
| PRD-harness-engineering M2 (template schema) | VisionFlow | Shipped |
| PRD-harness-engineering M6 (precedent system) | agentbox | Shipped (MCP bridge + service) |
| ADR-004 guide/sensor model | VisionFlow | Accepted |
| ADR-122 two-speed governance | VisionClaw | Proposed |
| Governance bridge MCP server | agentbox | Shipped (5 tools) |
| `spawnAgent()` orchestrator adapter | agentbox | Shipped (local-process-manager) |
| Nostr relay kind gate | nostr-rust-forum | Shipped (31400-31405) |

## Open Questions

1. **Kind allocation.** Should the MandateGranted event be kind 31406 (extending the Agent Control Surface sequence) or a separate kind range? Using 31406 keeps it in the governance family; a separate range signals it's a distinct concern.
2. **Capability graph maintenance.** Who maintains the tool composition edges — template authors, tool authors, or automated analysis? Manual maintenance risks staleness; automated analysis risks incompleteness.
3. **Degraded mandate.** When a full mandate cannot be granted (missing stakeholder approval, composition risk too high), should the agent receive a reduced capability set or not be provisioned at all?
4. **Retrospective mandate audit.** Should existing running agents be retroactively audited against the mandate model, or does it apply only to new provisioning events?
5. **ADR-122 integration.** Should mandate grants be classified by the same epistemic classifier (L1/L2/L3), or does mandate governance need its own routing logic?
6. **Mandate scope lifetime.** Does a mandate grant have a TTL (like ADR-122's `:observed` graph), or does it persist until explicitly revoked?
