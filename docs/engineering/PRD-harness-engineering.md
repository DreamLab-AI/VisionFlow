# PRD: Harness Engineering

**Owner:** DreamLab AI
**Status:** Active
**Date:** 2026-06-23
**Version:** 1.0
**Provenance:** Gap analysis of Martin Fowler's *Harness Engineering for Coding Agent Users* (2025) cross-referenced against VisionFlow ecosystem audit (4-agent mesh, 2026-06-23).

## TL;DR

The VisionFlow ecosystem already operates 400+ feedforward control artifacts (agent specs, skills, hooks, CLAUDE.md layers, Judgment Broker control panels) and a solid computational sensor foundation (contract tests, fixture parity, IS-Envelope validation, accessibility checks). What it lacks is the **pairing discipline** — every guide should have a corresponding sensor that validates the guide was followed. Note: the steering loop (M1) was found to be already implemented during the implementation sprint — `handleGovernanceDecision()` is shipped on both orchestrator adapters (local-process-manager and stdio-bridge) with PROV-O provenance minting, and the governance-bridge MCP server provides all 5 agent-facing governance tools. M1 is reclassified as **complete**.

This PRD formalises the upgrade from ad-hoc guide and sensor accumulation to a **harness engineering discipline** across the multi-repo ecosystem. The core deliverable is the **harness template** — a machine-readable bundle of `{structure + guides + sensors}` per topology that agents can inspect, not just humans can read.

## Goals

| Goal | Outcome |
|---|---|
| G1: Close the steering loop | Human decisions (kind 31403) arriving at agentbox trigger concrete orchestrator action via `handleGovernanceDecision()` and produce PROV-O provenance |
| G2: Harness template specification | A declarative schema for bundling guide+sensor pairs per topology, inspectable by agents via MCP tools |
| G3: Executable fitness functions | ADR compliance scores, SLO thresholds, and fixture parity checks promoted from dashboards to blocking CI gates |
| G4: Sensor-guide pairing | Every feedforward guide in the ecosystem has a corresponding feedback sensor; gaps are tracked in the compatibility matrix |
| G5: Cross-substrate harness | Mesh-spanning harness templates that declare the full topology (agentbox → VisionClaw → pod) not just per-repo fragments |
| G6: Precedent system | Recurring identical governance decisions auto-elevate to standing policy, eliminating repetitive human steering |

## Current Harness Inventory

### Feedforward Controls (Guides) — 400+ artifacts

| Layer | Count | Location | Coverage |
|---|---|---|---|
| Instruction hierarchy | 3 tiers | `~/.claude/CLAUDE.md` → `~/workspace/CLAUDE.md` → project | Global → workspace → project override chain |
| Agent specifications | 110 | `~/.claude/agents/` | YAML specs with `hooks.pre` (pattern search, memory injection) |
| Skills | 106 | `~/.claude/skills/` + `SKILL-DIRECTORY.md` | Decision tree for tool selection |
| Commands | 139+ | `~/.claude/commands/` (17 subdirs) | Skill-specific CLI entry points |
| Hook scripts | 40+ | `~/.claude/helpers/` | Routing, memory, intelligence, session management |
| Router patterns | 9 agent types | `router.js` | Task → agent classification |
| Ecosystem docs | 20+ | `VisionFlow/docs/` | ADRs, PRDs, DDDs, architecture maps |
| Judgment Broker | 5 Nostr kinds | 31400–31405 | Agent Control Surface protocol |

### Feedback Sensors — partial coverage

| Sensor Type | Status | Implementation |
|---|---|---|
| Contract tests (adapter suites) | Shipped | 5 slots × 3 impls, SLO validation (ADR-031) |
| Fixture parity | Shipped | 54 vectors, SHA-256 checksums, upstream pins |
| IS-Envelope validation | Shipped | 11 vectors, JSON Schema (VisionClaw ADR-075) |
| Accessibility (axe-core) | Shipped | WCAG 2.1 AA, zero violations |
| Performance budget | Shipped | ≤800KB initial payload |
| Security scanner | Shipped | Secrets detection, vuln patterns |
| ADR compliance scoring | Shipped | `adr-compliance.sh`, 4h throttle |
| Ontology bridge health | Shipped | 10 MCP tools, Oxigraph availability |
| BC20 anti-corruption metrics | Shipped | Drop/crossing Prometheus counters |
| Learning service | Shipped | HNSW + MiniLM-L6-v2, pattern lifecycle |
| AI code review | Shipped | `/code-review` skill, multi-effort |
| Mutation testing | **Missing** | No mutation score gates |
| Linter/type-checker gates | **Missing** | No ESLint, Prettier, clippy CI gates |
| Cyclomatic complexity | **Missing** | No complexity thresholds |
| Over-engineering detection | **Missing** | No API surface bloat metrics |
| Architecture fitness functions (blocking) | **Missing** | ADR scores exist but don't block merges |
| Decision→agent application | **Shipped** | `handleGovernanceDecision()` on both adapters, PROV-O provenance, E2E tested |
| Precedent system | **Missing** | `Promote`/`Precedent` outcomes designed but not implemented |

## Milestones

### M1: Steering Loop Closure (P0)

**Status: COMPLETE** (discovered during implementation sprint 2026-06-23)

Implementation audit revealed that all M1 requirements were already shipped:
- `handleGovernanceDecision()` — implemented on both `local-process-manager.js` (285 lines, PROV-O provenance, agent stdin routing, pod persistence) and `stdio-bridge.js` (JSON-RPC forwarding)
- Agent MCP tools — `governance-bridge.js` provides `governance_publish_panel`, `governance_request_action`, `governance_update_panel`, `governance_retire_panel`, `governance_list_decisions`
- Contract tests — `governance-flow.spec.js` (782 lines) covers the full E2E loop including provenance URN verification
- BrokerActor status — requires VisionClaw `crashbug` branch merge (deferred to VisionClaw sprint)

Wire the decision-to-agent feedback loop. Without this, all other harness improvements are feedforward-only.

| Requirement | Acceptance Criteria |
|---|---|
| FR1.1: `handleGovernanceDecision()` | Orchestrator adapter receives `ActionResponse` (kind 31403), deserialises `DecisionOutcome`, dispatches to appropriate adapter slot. Unit test coverage ≥80%. |
| FR1.2: Decision provenance | Each applied decision mints a `urn:agentbox:activity:governance:decision-<id>` PROV-O record linking decision event → case → resulting action. Written to pod. |
| FR1.3: Agent MCP tools | `governance_publish_panel` and `governance_request_action` MCP tools exposed in agentbox. Agents compose valid kind 31400/31402 events without raw outbox writes. |
| FR1.4: BrokerActor on main | VisionClaw merges BrokerActor from `crashbug` branch with CI coverage. |
| FR1.5: End-to-end smoke test | Mesh smoke test (`docs/protocol/mesh-smoke-test.md`) passes: VisionClaw submits → forum renders → human decides → agentbox routes → provenance recorded. |

**Depends on:** PRD-judgment-broker M1 (this supersedes and extends it)

### M2: Harness Template Specification (P1)

Define and implement the harness template schema.

| Requirement | Acceptance Criteria |
|---|---|
| FR2.1: Template schema | JSON Schema defining `HarnessTemplate` with fields: `topology`, `structure`, `guides[]`, `sensors[]`, `pairings[]` (guide↔sensor bindings), `escalation_rules[]`. Published under `docs/engineering/schemas/`. |
| FR2.2: Four canonical templates | Templates for: `agentbox-agent-task`, `visionclaw-enrichment`, `governance-decision`, `pod-mutation`. Each bundles ≥3 guide-sensor pairs. |
| FR2.3: MCP tool for template inspection | `harness_inspect` MCP tool returns the active harness template for the current topology. Agents query this before acting. |
| FR2.4: Template registry | Templates discoverable via `harness_list` MCP tool. Cross-substrate templates reference multiple repos. |
| FR2.5: Compatibility matrix integration | Harness coverage tracked per substrate in the compatibility matrix. Each cell shows guide count, sensor count, pairing ratio. |

### M3: Executable Fitness Functions (P1)

Promote quality checks from dashboards to merge gates.

| Requirement | Acceptance Criteria |
|---|---|
| FR3.1: ADR compliance gate | `adr-compliance.sh` runs in CI. Score below threshold blocks merge. Threshold configurable per repo. |
| FR3.2: SLO enforcement | Adapter contract SLO thresholds (p95 latency, throughput floor, error ceiling per ADR-005) enforced in CI, not just test output. |
| FR3.3: Fixture parity gate | SHA-256 fixture parity check runs in CI for all repos consuming upstream vectors. Drift blocks merge. |
| FR3.4: Mutation testing | Mutation score gate on adapter contract suites and IS-Envelope validation. Minimum surviving mutant kill rate: 80%. |
| FR3.5: Linter/type-checker gates | ESLint (agentbox, website JS), `cargo clippy` (VisionClaw, solid-pod-rs, forum Rust) enforced in CI. |

### M4: Sensor-Guide Pairing (P2)

Formalise the discipline that every guide has a corresponding sensor.

| Requirement | Acceptance Criteria |
|---|---|
| FR4.1: `hooks.validate` phase | Agent execution lifecycle extended: `hooks.pre` (guide) → execution → `hooks.validate` (sensor) → `hooks.post` (learning). Validate phase checks whether guide constraints were respected. |
| FR4.2: Pairing audit | Automated scan of all agent specs, skills, and ecosystem docs. Report: guides without sensors, sensors without guides, unpaired count. Target: ≤20% unpaired. |
| FR4.3: In-task trajectory monitoring | During agent execution, a lightweight monitor checks intermediate outputs against declared guide constraints. Violations trigger self-correction before completion. |
| FR4.4: Pairing ratio in compatibility matrix | Each substrate's harness coverage cell shows `paired/total` ratio. Matrix column added. |

### M5: Cross-Substrate Harness (P2)

Harness templates that span the full mesh topology.

| Requirement | Acceptance Criteria |
|---|---|
| FR5.1: Mesh harness template | A single template describing the full enrichment flow: agentbox (agent submits) → relay (routes) → forum (human decides) → VisionClaw (enrichment gating) → pod (persistence). Includes guide-sensor pairs for each hop. |
| FR5.2: Cross-repo sensor aggregation | A coordinator service (or MCP tool) that queries sensor status across all substrates and reports mesh-wide harness health. |
| FR5.3: Topology-aware agent briefing | When an agent receives a cross-substrate task, the harness template is injected into its context. The agent knows the full topology it operates within, not just its local substrate. |

### M6: Precedent System (P3)

Eliminate repetitive governance decisions through learned precedents.

| Requirement | Acceptance Criteria |
|---|---|
| FR6.1: Precedent storage | Approved decisions with `Promote` outcome stored in RuVector (`governance-precedents` namespace) with semantic embedding. |
| FR6.2: Precedent matching | Before routing a new `ActionRequest` to a human, the system searches precedents. If similarity ≥ threshold, auto-applies the precedent and records a `Precedent` outcome. |
| FR6.3: Precedent override | Humans can override or retire precedents via kind 31403 with `Reject` + precedent reference. |
| FR6.4: Audit trail | Every auto-applied precedent produces a PROV-O record linking the precedent source decision to the current application. |

### M7: Janitor Automation (P3)

Continuous drift detection and cleanup.

| Requirement | Acceptance Criteria |
|---|---|
| FR7.1: Scheduled janitor agent | A periodic agent runs `CompareFixtureCorpus`, `EvaluateMaturityStatus`, `CheckOntologyBridgeHealth`, and pairing audit. Findings filed as issues or auto-fixed. |
| FR7.2: Dead code detection | Cross-repo scan for unused exports, deprecated adapter implementations, orphaned fixtures. |
| FR7.3: Drift alerting | Janitor findings above severity threshold trigger Nostr notification to operator (via existing mirror hook). |

## Non-Goals

- **Replacing the Judgment Broker.** This PRD extends it with harness engineering concepts; it does not redesign the decision loop.
- **Universal harness templates.** We build topology-specific templates for VisionFlow's substrates, not a generic harness framework for arbitrary projects.
- **Automated code generation from templates.** Templates inform agents; they do not scaffold code. The article's "harness template" is a regulation artefact, not a code generator.

## Dependencies

| Dependency | Owner | Status |
|---|---|---|
| PRD-judgment-broker M1 | agentbox, VisionClaw | 65% (FR1 subsumes) |
| ADR-005 pluggable adapter architecture | agentbox | Shipped |
| ADR-031 adapter contract enforcement | agentbox | Shipped |
| ADR-075 IS-Envelope spec | VisionClaw | Shipped |
| Mesh smoke test protocol | VisionFlow | Defined, not automated |
| RuVector memory backend | Infrastructure | Shipped |

## Success Metrics

| Metric | Baseline (2026-06-23) | Target |
|---|---|---|
| Steering loop closure | ~95% (implemented on both adapters, E2E tested; BrokerActor merge pending) | 100% (decisions applied + provenance recorded) |
| Guide-sensor pairing ratio | 100% paired / 82.5% source-backed (measured by `harness-audit.sh`, 2026-07-03; the earlier "~30%" was an unmeasured estimate) | ≥80% source-backed across all substrates |
| Fitness function CI coverage | 0 repos with blocking gates | All 6 repos |
| Mutation kill rate | 0% (no mutation testing) | ≥80% on contract suites |
| Precedent auto-application rate | 0% | ≥40% of recurring decision types |
| Mean time to detect drift | Manual (days) | Automated (≤4 hours via janitor) |
