# ADR-005: Mandate-at-Grant Governance

**Status:** Speculative — implementation is not committed (kind 31406 MandateGrant is absent from `agentbox.toml` `allowed_kinds`, and the harness schema has no `mandates` section)
**Date:** 2026-06-23
**Decision Owners:** DreamLab AI maintainers
**Related:** [ADR-004 Harness Engineering Framework](ADR-004-harness-engineering-framework.md), [ADR-122 Two-Speed Governance](../../project/docs/adr/ADR-122-two-speed-writeback-governance-routing.md), [PRD-mandate-at-grant](PRD-mandate-at-grant.md), [PRD-harness-engineering](PRD-harness-engineering.md), [closeout final-design](../closeout/final-design.md)
**Provenance:** "The Moment of Authorisation" principle (Sheridan et al.) cross-referenced against VisionFlow governance architecture audit, 2026-06-23. Five-agent mesh examination across agent bootstrap, governance bridge, file/pod access, MCP tool surface, and harness template schema.

> **2026-07-03 closeout amendment.** This ADR remains **Speculative** (not built), but one Context claim is now stale and corrected below: the `governance-precedents` namespace **is** write-protected. `mcp/servers/lib/memory-tools.js:60-68` defines `PROTECTED_NAMESPACES` (default `governance-precedents`) and rejects writes with `namespace "…" is write-protected (IR2 mandate-at-grant)`. The precedent-poisoning vector this ADR cites as its most critical motivation is therefore already mitigated at the capability channel; the remaining account/file/env-secret grant channels are unchanged. This note is not a build commitment — D2's `mandates` schema section and kind 31406 remain unimplemented.

## Context

The VisionFlow Judgment Broker implements a mature action-time governance model: agents publish ActionRequest events (kind 31402), humans review and respond via ActionResponse (kind 31403), and the orchestrator applies decisions via `handleGovernanceDecision()` with full PROV-O provenance. ADR-004 extended this with harness templates that pair feedforward guides with feedback sensors across the agent execution lifecycle.

Both systems share a blind spot: **they govern what agents DO, not what agents CAN do**. The "Moment of Authorisation" principle (Sheridan et al.) identifies this as a fundamental gap:

> Authorization risk begins at the moment access is granted — the mandate — not when the agent acts. The appropriate method is to map what the agent can do (capability graph), connect the side-effects of those capabilities, and surface the risks to all affected stakeholders.

Five-agent mesh examination of the VisionFlow ecosystem confirms three classes of silent mandate grants:

**Account channel.** `signerFromHex()` in `per-user-agent.js` creates a signer with `{ sign, skBytes, pubkey }` granting full Schnorr signing authority under a `did:nostr` identity — no governance event emitted (`per-user-agent.js` lines 142-146). The NIP-42 AUTH signer is registered on the bridge BEFORE subscribing (line 800), and `nip98Token(signer, url, method)` can forge HTTP auth headers for ANY URL path. The agent can sign any Nostr event kind. `agent_event_auth` defaults to `'off'` (`agentbox.toml` line 132), meaning event attribution is caller-asserted, not verified.

**File channel.** `mandate.js` provides a partial mandate system — `createMandate()` generates WAC ACL grants and `signMandate()` wraps them in NIP-33 kind 30078 for revocability. However, `createMandate()` emits NO governance event, `acl:Control` is issuable without elevated governance (enabling self-escalation), and `expiresAt` defaults to null (indefinite access). More critically, `spawnAgent()` in `local-process-manager.js` passes `{ ...process.env, ...(spec.env) }` to child processes (line 47), silently granting ALL environment secrets. `ProcessManager.spawnTask()` explicitly passes 8 API keys (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, etc., lines 173-184) with no governance event.

**Capability channel.** 108+ tools across 25+ MCP servers (`skills/mcp.json`) are globally available to every agent. The `ALLOWED_TOOLS` whitelist in ProcessManager is the sole capability-constraining mechanism and applies only to the claude-flow provider path. Per-user agents receive `NostrBridge.subscribe()`, `callLlm()`, `RuVector` memory access, and NIP-98-authed pod access — all granted at constructor time with zero governance events. The precedent-poisoning vector originally cited here — "`memory_store` has no namespace ACL, so any agent can write to `governance-precedents`" — **has since been closed**: `memory-tools.js:60-68` defines `PROTECTED_NAMESPACES` (default `governance-precedents`) and rejects such writes with an IR2 mandate-at-grant error. The remaining ungoverned grants are the account, file, and environment-secret channels above.

**Existing partial infrastructure.** The S6 WoT Thing Description surface already models MCP capabilities as W3C WoT Actions/Properties/Events but is emit-only and disconnected from governance. The S8 Payment mandate uses ODRL Permission VCs for financial authorisation but is disconnected from the Nostr event protocol. Neither feeds the governance decision flow.

The harness template schema (ADR-004 D2) has no `mandates` or `capabilities` section. Templates declare guides, sensors, and pairings for execution-time regulation but do not declare what capabilities the topology requires or what blast radius those capabilities produce.

## Decision

### D1: Mandate-at-grant is a left-shift of the governance model

The current governance model is:

```
agent provisioned (silent)
  → agent acts (ActionRequest kind 31402)
    → human reviews action (ActionResponse kind 31403)
      → decision applied (handleGovernanceDecision)
```

The mandate-at-grant model inserts a governance checkpoint at provisioning:

```
agent provisioned (MandateGrant event kind 31406)
  → mandate reviewed (MandateResponse, gate mode determines blocking)
    → agent receives scoped capabilities
      → agent acts (ActionRequest kind 31402, carries current_mandates)
        → human reviews action (ActionResponse kind 31403)
          → decision applied (handleGovernanceDecision)
```

**Rationale:** Governing at action time answers "should this agent do this thing?" Governing at grant time answers "should this agent be ABLE to do this thing?" The latter is a superset — it evaluates the full capability space, not a single point within it. This is the same "left-shift" pattern that ADR-004 applied to quality (fitness gates before merge) now applied to authorization (mandate review before provisioning).

### D2: Three-channel mandate inventory in harness templates

The harness template schema gains a `mandates` section with three channel declarations:

```
mandates:
  account:
    identity_type: string         # e.g. "did:nostr", "api-key"
    signing_scope: string[]       # Nostr kinds the agent may sign
    identity_lifetime: string     # "session", "persistent", "ttl:24h"
    stakeholders: string[]        # Roles affected by this identity grant
  file:
    - path: string                # Pod or filesystem path
      access: string[]            # ["read", "write", "append", "control"]
      purpose: string             # Why this access is needed
      blast_radius: string        # What data/systems this path reaches
      stakeholders: string[]
  capability:
    - tool: string                # MCP tool name or adapter slot
      classification: string      # "read_only", "write_mutation", "governance_write", "infrastructure_modify"
      purpose: string
      composes_with: string[]     # Other tools this chains with
      transitive_reach: string[]  # Systems/data reachable through composition
      stakeholders: string[]
```

**Rationale:** The three-channel model from the "Moment of Authorisation" maps directly onto VisionFlow's architecture: account = `did:nostr` + `AGENTBOX_PUBKEY`, file = `AGENTBOX_POD_ROOT` + WAC, capability = `.mcp.json` + adapter slots. Declaring mandates in harness templates makes them machine-readable, inspectable via `harness_inspect`, and pairable with sensors (per ADR-004 D1).

### D3: Capability composition is modelled as a directed graph, not a flat list

MCP tools and adapter slots are classified by mutation type and linked by composition edges:

```
ToolNode:
  id: string                       # MCP tool name
  classification: enum(read_only | write_mutation | governance_write | infrastructure_modify)
  side_effects: string[]           # Systems/data modified by this tool
  
CompositionEdge:
  from: string                     # Tool that produces output
  to: string                       # Tool that consumes it
  effect: string                   # What the composition achieves
  risk: enum(low | medium | high)  # Risk of the composed effect
```

The composition graph enables **transitive reach analysis**: given an agent's tool set, compute the full set of reachable systems and data stores through tool composition. This is the poster's "connect the side-effects" step.

**Rationale:** A flat tool list makes over-provisioning invisible. An agent with 8 tools could have 2 high-risk composition chains, but this is undetectable without the graph. The existing BC20 anti-corruption layer monitors boundary crossings at runtime; the composition graph evaluates them at grant time.

### D4: Tool classification taxonomy

Every MCP tool is classified by its mutation characteristics:

| Classification | Definition | Default gate behaviour | Examples |
|---|---|---|---|
| `read_only` | Returns data, no side-effects | No mandate review needed | `harness_inspect`, `harness_list`, `memory_search`, `ontology_query` |
| `write_mutation` | Modifies data in a single system | Mandate audit (log) | `memory_store`, `ontology_propose`, `precedent_promote` |
| `governance_write` | Creates or modifies governance infrastructure | Mandate advisory (warn) | `governance_publish_panel`, `governance_request_action`, `governance_update_panel` |
| `infrastructure_modify` | Changes system configuration or spawns processes | Mandate blocking (require approval) | `spawnAgent` (orchestrator adapter), `governance_retire_panel` |

**Rationale:** This taxonomy parallels ADR-004 D4's computational/inferential sensor split. Both classify artefacts by their operational characteristics to determine appropriate gate behaviour. The principle: **the higher the mutation scope, the stronger the mandate review.**

### D5: Mandate-time governance event (kind 31406 MandateGrant)

A new Nostr event kind carries the mandate inventory when an agent is provisioned:

```json
{
  "kind": 31406,
  "tags": [
    ["d", "<mandate-id>"],
    ["p", "<agent-pubkey>"],
    ["e", "<provisioning-context-event>", "", "context"]
  ],
  "content": {
    "agent_identity": "urn:agentbox:agent:proc-<id>",
    "topology": "agentbox-agent-task",
    "mandates": {
      "account": { ... },
      "file": [ ... ],
      "capability": [ ... ]
    },
    "blast_radius": {
      "reachable_systems": ["pod", "relay", "ontology-store", "memory"],
      "reachable_data": ["governance events", "ontology triples", "learning patterns"],
      "composition_risks": [
        {
          "chain": "governance_publish_panel → governance_request_action",
          "effect": "Agent can create and populate governance panels",
          "risk": "medium"
        }
      ]
    },
    "stakeholders": ["operator", "data-owner", "security"],
    "gate_mode": "audit"
  }
}
```

Three gate modes control the mandate review lifecycle:

| Mode | Behaviour | When to use |
|---|---|---|
| `audit` | Emit MandateGrant event, don't block provisioning. Event available for retrospective review. | Default. Low-risk topologies, established agents. |
| `advisory` | Emit event + warn operator via Nostr mirror if composition risk > threshold. Don't block. | Medium-risk topologies, agents with governance tools. |
| `blocking` | Emit event, require kind 31403 approval before provisioning completes. | High-risk: novel capability combinations, cross-substrate mandates, infrastructure-modify tools. |

The `capability` channel of the MandateGrant event should reference the S6 WoT Thing Description (`s06-wot.js`), which already models MCP tools as W3C WoT Actions/Properties/Events. The `encode()` output provides a machine-readable capability surface that can be embedded in the 31406 content. Similarly, the S8 Payment mandate (`s08-payments.js`) ODRL Permission VC should be linked via `evidence` references so the financial blast radius (`amount_sats`, rate limits) is part of the mandate envelope.

**Rationale:** The three-mode approach mirrors ADR-004's `blocking/advisory/learning` validation modes. Starting in `audit` mode makes mandate governance observable without impacting existing workflows. Operators promote to `advisory` or `blocking` as confidence in the model grows — the same maturity trajectory as sensor promotion in ADR-004 D4. Leveraging S6 WoT and S8 ODRL avoids creating a third capability vocabulary — the MandateGrant event composes existing standards rather than inventing new ones.

### D6: Mandate routing follows the ADR-122 epistemic pattern

ADR-122 routes governance decisions by epistemic class (L1 human-gated, L2 automatic-with-consistency, L3 automatic-derived). Mandate grants are routed by an analogous classification:

| Lane | What routes here | Gate | Mandate equivalent |
|---|---|---|---|
| **M1 — Human-gated** | Novel capability combinations not covered by any precedent. Cross-substrate mandates spanning ≥3 substrates. Any grant including `infrastructure_modify` tools. | Kind 31406 → human review panel → kind 31403 response | New agent roles, first-time provisioning of powerful tool combinations |
| **M2 — Precedent-matched** | Re-provisioning with a capability set matching a standing mandate precedent. Same agent type, same topology, same tool set. | Auto-applied from `governance-mandates` namespace. PROV-O audit trail. | Routine agent restarts, known topology re-entry |
| **M3 — Automatic** | Read-only tool grants. Status queries. Agents receiving only `read_only` classified tools. | No governance event (or minimal audit log) | Health check agents, monitoring-only agents |

The classifier defaults to M1 (human gate) unless the mandate clearly qualifies for M2/M3. This mirrors ADR-122's default-conservative classification.

**Rationale:** Reusing the lane routing pattern from ADR-122 maintains architectural consistency. The same mental model applies: structural/high-stakes decisions are human-gated; routine/low-stakes decisions are auto-applied; derived/read-only operations are automatic. The mandate classifier is simpler than the epistemic classifier because tool classification (D4) is deterministic, not probabilistic.

### D7: Mandate sensors extend the guide/sensor pairing model

Each mandate channel becomes a guide (the declared mandate in the template) paired with a sensor (the mandate audit that validates compliance):

| Mandate Guide | Mandate Sensor | Pairing Mode |
|---|---|---|
| `mandates.account.signing_scope` | Signing audit: agent signed only permitted Nostr kinds | Advisory → Blocking |
| `mandates.file[].path` + access | Pod access audit: agent accessed only declared paths | Advisory |
| `mandates.capability[].tool` | Tool usage audit: agent invoked only declared tools | Advisory → Blocking |
| Composition graph `composes_with` edges | Composition audit: no undeclared tool chains executed | Learning |
| `mandates.*.stakeholders` | Stakeholder coverage: all listed stakeholders were notified | Advisory |

These pairings follow ADR-004 D3's three-phase lifecycle:

```
mandate declared (guide, in harness template)
  → agent provisioned (execution)
    → mandate audit (sensor, checks actual vs declared usage)
      → mandate learning (pattern stored for precedent extraction)
```

**Rationale:** The mandate-sensor pairing is the natural extension of ADR-004's pairing discipline to the provisioning domain. Without it, mandates would be declared but unvalidated — the same "feedforward-only" failure mode ADR-004 D3 identified for agent execution.

### D8: Blast radius is Ashby-bounded by topology commitment

ADR-004 D7 invoked Ashby's Law of Requisite Variety: topology commitment narrows the regulation space. The same principle applies to mandates: **an agent's blast radius is bounded by the topology it commits to.**

An agent provisioned for the `agentbox-agent-task` topology should only have capabilities needed for that topology. If it also has `governance_publish_panel` (a `governance-decision` topology tool), the mandate audit flags this as cross-topology over-provisioning.

**Rationale:** Without topology-bounded mandates, every agent has the blast radius of the full tool registry. Topology commitment constrains the mandate to what the harness template declares, making over-provisioning detectable and auditable.

### D9: ActionRequest carries current mandates

Kind 31402 ActionRequest events are extended with a `current_mandates` field summarising what the requesting agent already has:

```json
{
  "kind": 31402,
  "content": {
    "title": "Propose ontology enrichment",
    "description": "...",
    "current_mandates": {
      "account": { "signing_scope": [31400, 31402] },
      "file": [{ "path": "/var/lib/agentbox/pods/...", "access": ["read", "write"] }],
      "capability": ["ontology_propose", "memory_store", "harness_inspect"]
    }
  }
}
```

This allows the human reviewer to evaluate the action request in the context of what the agent CAN do, not just what it's ASKING to do.

**Rationale:** The poster's core insight: examining the action without knowing the mandate is evaluating a symptom without the diagnosis. A request to "propose ontology enrichment" reads differently when the reviewer can see the agent also has `governance_publish_panel` + `memory_store` — composition risks become visible at decision time.

## Consequences

### Positive

- **Silent grants become visible.** Every capability grant is either declared in a harness template (design time) or emitted as a MandateGrant event (runtime). No agent receives capabilities without a record.
- **Composition risks are calculable.** The tool composition graph enables automated blast radius analysis — a capability that doesn't exist today and has no manual equivalent at scale.
- **Left-shifted security.** Over-provisioning is detected before the agent acts, not after damage occurs. This is the same quality-left-shift that fitness gates (ADR-004 D4) applied to correctness.
- **Consistent governance model.** Mandate routing reuses the ADR-122 lane pattern. Mandate sensors reuse the ADR-004 pairing model. No new architectural concepts — existing abstractions extended to a new domain.
- **Precedent efficiency.** Routine re-provisioning auto-approves from mandate precedents, keeping human steering burden proportional to novelty, not volume.

### Negative

- **Capability graph maintenance.** The composition graph requires ongoing maintenance as MCP tools are added, modified, or removed. Mitigated by: (a) tool authors declare `composes_with` at registration time; (b) janitor agent (PRD-harness-engineering M7) audits for undeclared edges.
- **Bootstrap friction.** New agent types require mandate inventory before first provisioning. Mitigated by: (a) `audit` mode as default — observe before blocking; (b) template scaffolding generates mandate stubs from tool registry.
- **Schema complexity.** The harness template schema gains a substantial new section. Mitigated by: (a) mandate section is optional — existing templates continue to validate; (b) the `mandates` schema is a separate `$def` that can be validated independently.
- **Retrospective gap.** Running agents were provisioned without mandate governance. Retrospective audit requires scanning current agent states against the new mandate model.

### Neutral

- Kind 31406 extends the Agent Control Surface sequence. Relay kind gates need updating to accept 31406 events.
- The `harness_inspect` MCP tool returns mandates alongside guides and sensors. No breaking change — additional fields in the response.
- Existing governance flows (31400-31405) continue to function. Mandate governance is additive, not a replacement.

## Alternatives Considered

### A1: Action-time-only governance (status quo)

Continue governing at action time via kind 31402/31403.

**Rejected (speculatively):** The poster's argument is compelling: action-time governance can never catch over-provisioning because it only sees individual requests, not the capability space. An agent with governance tools + memory tools + ontology tools has a blast radius invisible to action-time review.

### A2: Zero-trust per-call verification

Require governance approval for every MCP tool invocation, not just ActionRequests.

**Rejected:** Latency is prohibitive. An agent invoking 50 tools per task would require 50 governance roundtrips. The mandate model achieves equivalent security by constraining what tools are available, not by gating every invocation.

### A3: Static capability configuration (no governance events)

Configure per-agent tool allowlists in `agentbox.toml` without governance events.

**Rejected:** Configuration without governance events is invisible to the Judgment Broker, precedent system, and provenance trail. It solves the technical problem (restrict tools) but not the governance problem (who approved the restriction, when, and why).

### A4: Embed mandates in DID documents only

Declare agent capabilities in the `did:nostr` service endpoints and let DID verification handle mandate governance.

**Rejected as sole mechanism:** DID documents are the right place for cross-organisation capability declarations, but they're too coarse for topology-specific mandate governance. A DID document might declare "this agent can sign kind 31402" but won't capture the composition risks of specific tool combinations. DID declarations complement but don't replace harness template mandates.

## Implementation Notes (Speculative)

This ADR is speculative — implementation is not committed. If progressed:

1. **Phase 1 (observe):** Add `mandates` section to harness template schema. Populate canonical templates. Ship `mandate_blast_radius` MCP tool. All in `audit` mode — emit events, don't block.
2. **Phase 2 (classify):** Build tool taxonomy. Declare composition edges for high-risk tool combinations. Enable `advisory` mode for templates with governance tools.
3. **Phase 3 (enforce):** Per-agent tool scoping via template-driven allowlists. `blocking` mode for `infrastructure_modify` tools. Mandate precedent system.
4. **Phase 4 (sustain):** Multi-stakeholder review. Mandate TTLs. Janitor audits mandate drift.

Each phase is independently valuable and can be shipped without committing to later phases.
