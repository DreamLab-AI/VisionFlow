# DDD: Harness Engineering Bounded Context

**Status:** Living document
**Date:** 2026-06-23
**Scope:** Cross-substrate regulation of agent behaviour through paired feedforward/feedback controls

---

## 1. Bounded Context

The Harness Engineering Context governs the lifecycle of guides (feedforward controls), sensors (feedback controls), their pairings, and their bundling into topology-specific harness templates. It spans all six VisionFlow substrates but does not own execution, identity, or persistence — it orchestrates the **regulation surface** across the substrates that do.

This context is adjacent to but distinct from the Judgment Broker Context. The Judgment Broker governs **individual decisions** (approve/reject a specific agent action). The Harness Engineering Context governs the **systemic regulation** that shapes all agent behaviour — the standing rules, not the case-by-case rulings.

---

## 2. Context Map

| Context | Relationship | Notes |
|---|---|---|
| **Harness Engineering** (this context) | Coordinates regulation across substrates | Owns template schema, pairing discipline, fitness gates |
| **Judgment Broker** | Downstream | Consumes harness templates to determine what controls apply to a governance case. Precedent system feeds back into harness as new feedforward controls. |
| **agentbox Agent Runtime** | Upstream supplier | Supplies agent specs (`hooks.pre`, `hooks.validate`, `hooks.post`), MCP tool registry, adapter contract suites. Consumes templates via `harness_inspect` tool. |
| **VisionClaw Semantic Substrate** | Upstream supplier | Supplies ontology constraints (OWL 2 EL), SPARQL sensors (Oxigraph health), IS-Envelope spec (ADR-075). Consumes enrichment harness template. |
| **nostr-rust-forum Governance** | Upstream supplier | Supplies governance domain model, relay gating rules, decision outcomes. Consumes governance harness template. |
| **solid-pod-rs** | Upstream supplier | Supplies pod validation (WAC ACLs, NIP-98, git-marks). Consumes pod-mutation harness template. |
| **dreamlab-ai-website** | Consumer | Consumes harness health metrics for dashboard display. |
| **Ecosystem Alignment** (ADR-002) | Sibling | Shares the compatibility matrix. Harness coverage columns extend the matrix; maturity vocabulary applies to harness template status. |

### Relationship Types

- **Judgment Broker → Harness Engineering:** Customer/Supplier. The broker queries active templates to know what sensors apply to a decision case. The precedent system (Promote/Precedent outcomes) creates new feedforward entries in harness templates.
- **Substrates → Harness Engineering:** Conformist. Substrates own their guides and sensors but conform to the harness template schema when bundling them. The template schema is the shared contract.
- **Harness Engineering → Compatibility Matrix:** Shared Kernel. Both contexts contribute to and read from the same matrix. Harness engineering adds columns; ecosystem alignment maintains rows.

---

## 3. Ubiquitous Language

| Term | Definition |
|---|---|
| **Guide** | A feedforward control that shapes agent behaviour before execution. Examples: agent spec `hooks.pre`, CLAUDE.md instruction, IS-Envelope schema, OWL domain/range constraint. |
| **Sensor** | A feedback control that observes agent output after execution and enables correction. Examples: contract test, fixture parity check, linter, AI code review. |
| **Computational sensor** | A sensor that produces deterministic results in milliseconds via CPU. Tests, linters, type checkers, structural analysis. Default: blocking. |
| **Inferential sensor** | A sensor that produces probabilistic results in seconds via GPU/LLM. AI code review, semantic duplication, over-engineering detection. Default: advisory. |
| **Pairing** | A binding between a guide and its corresponding sensor. The sensor validates that the guide's constraints were respected. |
| **Harness template** | A declarative bundle of `{topology, structure, guides[], sensors[], pairings[], escalation_rules[]}` for a specific substrate topology. The unit of agent governance. |
| **Topology** | A named configuration of substrates, event kinds, and data flows that a harness template governs. Examples: `governance-decision`, `visionclaw-enrichment`, `pod-mutation`. |
| **Fitness function** | A computational sensor promoted to a merge gate. Failures block integration. |
| **Steering loop** | The iterative cycle: agent acts → sensor observes → human/system corrects → guide updated → agent acts differently. |
| **Precedent** | A feedforward control generated from recurring identical feedback. A promoted governance decision that auto-applies to future matching cases. |
| **Pairing ratio** | `paired_controls / total_controls` per substrate. The primary coverage metric. |
| **Trajectory monitoring** | In-execution sensor that checks intermediate agent outputs against guide constraints before completion. |
| **Janitor** | An automated agent that periodically scans for drift, dead code, unpaired controls, and fixture divergence. |

---

## 4. Aggregates

| Aggregate | Owner | Description |
|---|---|---|
| `HarnessTemplate` | VisionFlow (this repo) | The canonical regulation bundle for a topology. Contains guides, sensors, pairings, and escalation rules. Immutable per version; new versions supersede. |
| `Guide` | Owning substrate | A feedforward control with source location, type, applicability scope. Owned by the substrate that defines it but referenced by templates owned here. |
| `Sensor` | Owning substrate | A feedback control with source location, type, frequency, gate behaviour. Owned by the substrate that defines it but referenced by templates owned here. |
| `Pairing` | VisionFlow (this repo) | The binding between a guide and a sensor. Owned here because pairings are an ecosystem concern — the substrate that owns the guide may not own the corresponding sensor. |
| `FitnessGate` | VisionFlow (this repo) | A computational sensor promoted to blocking status with a threshold and CI integration config. |
| `Precedent` | Judgment Broker (downstream) | A promoted governance decision stored in RuVector. Referenced by harness templates as an auto-generated feedforward control. |
| `HarnessAudit` | VisionFlow (this repo) | A periodic snapshot of pairing ratios, sensor health, fitness gate status, and drift findings across all substrates. |

---

## 5. Entities

| Entity | Identity | Owner |
|---|---|---|
| `HarnessTemplate` | `urn:visionflow:harness:<topology>:<version>` | VisionFlow |
| `Guide` | `urn:<substrate>:guide:<id>` | Owning substrate |
| `Sensor` | `urn:<substrate>:sensor:<id>` | Owning substrate |
| `Pairing` | `urn:visionflow:pairing:<guide-id>:<sensor-id>` | VisionFlow |
| `FitnessGate` | `urn:visionflow:gate:<sensor-id>` | VisionFlow |
| `Precedent` | `urn:agentbox:precedent:<decision-id>` | Judgment Broker |
| `HarnessAudit` | `urn:visionflow:audit:<timestamp>` | VisionFlow |
| `PairingGap` | `urn:visionflow:gap:<guide-id>` | VisionFlow |

---

## 6. Value Objects

| Value Object | Fields | Notes |
|---|---|---|
| `GuideType` | `instruction`, `schema`, `constraint`, `hook`, `ontology_axiom` | Classification of feedforward controls |
| `SensorType` | `computational`, `inferential` | Per ADR-004 D4 |
| `SensorFrequency` | `per_change`, `per_commit`, `per_release`, `continuous` | When the sensor runs |
| `GateBehaviour` | `blocking`, `advisory`, `learning` | What happens when a sensor fires |
| `EscalationAction` | `block`, `warn`, `file_issue`, `auto_fix`, `escalate_human` | Response to sensor threshold breach |
| `ValidationMode` | `blocking`, `advisory`, `learning` | How a pairing enforces the guide |
| `PairingRatio` | `{ paired: int, total: int, ratio: float }` | Coverage metric per substrate |
| `TopologyScope` | `{ substrates: string[], event_kinds: int[], data_flows: Flow[] }` | What a template governs |
| `TemplateMaturity` | `planned`, `scaffolded`, `standalone`, `integrated`, `verified` | Reuses ecosystem alignment maturity vocabulary |

---

## 7. Domain Events

| Event | Trigger | Consumers | Effect |
|---|---|---|---|
| `HarnessTemplatePublished` | New template version committed to VisionFlow | All substrates | Substrates update their guide/sensor registrations against new template |
| `PairingCreated` | Guide linked to sensor | Compatibility matrix | Pairing ratio recalculated for affected substrate |
| `PairingGapDetected` | Janitor audit finds guide without sensor | Issue tracker, operator notification | Gap filed as issue; operator notified via Nostr mirror |
| `FitnessGatePromoted` | Sensor promoted to blocking | CI pipeline config | CI config updated to include blocking check |
| `FitnessGateFired` | Blocking sensor fails in CI | Merge blocked | Developer must fix before merge |
| `SensorAdvisoryFired` | Advisory sensor reports finding | Learning service | Finding recorded for pattern extraction; no merge block |
| `PrecedentCreated` | `DecisionOutcome::Promote` applied in Judgment Broker | Harness template, RuVector | New feedforward control auto-generated from feedback |
| `PrecedentApplied` | Incoming `ActionRequest` matches stored precedent | PROV-O audit trail | Auto-applied decision with full provenance linkage |
| `PrecedentRetired` | Human overrides precedent with `Reject` + precedent ref | RuVector, harness template | Precedent removed from active templates; audit trail preserved |
| `DriftDetected` | Janitor finds fixture divergence, dead code, or maturity regression | Issue tracker, operator | Findings filed; severity above threshold triggers alert |
| `HarnessAuditCompleted` | Scheduled janitor finishes scan | Compatibility matrix, operator | Matrix updated with current pairing ratios and sensor health |
| `TrajectoryViolation` | In-execution monitor detects guide constraint breach | Agent self-correction | Agent retries with violation context; violation recorded for learning |

---

## 8. Services

| Service | Type | Description |
|---|---|---|
| `PublishTemplate` | Command | Validates and commits a new `HarnessTemplate` version. Checks schema conformance, guide/sensor existence, pairing validity. |
| `InspectHarness` | Query | Returns the active template for a given topology. Exposed as `harness_inspect` MCP tool. Agents call this before acting. |
| `ListHarnesses` | Query | Returns all registered templates with maturity status. Exposed as `harness_list` MCP tool. |
| `AuditPairings` | Query | Scans all guides and sensors across substrates. Returns `PairingRatio` per substrate and list of `PairingGap` entries. |
| `PromoteFitnessGate` | Command | Promotes a computational sensor to blocking status. Generates CI config. Requires human approval (kind 31403). |
| `RunJanitor` | Command | Executes the periodic drift scan: `CompareFixtureCorpus`, `EvaluateMaturityStatus`, `CheckOntologyBridgeHealth`, `AuditPairings`. Files findings. |
| `MatchPrecedent` | Query | Searches RuVector `governance-precedents` namespace for semantic match against an incoming `ActionRequest`. Returns matching precedent or null. |
| `ApplyPrecedent` | Command | Auto-applies a matched precedent. Mints PROV-O record. Publishes `PrecedentApplied` event. |
| `ValidateTrajectory` | Query | Checks intermediate agent output against guide constraints from the active template. Returns violations or pass. |
| `AggregateMeshHealth` | Query | Queries sensor status across all substrates. Returns mesh-wide harness health report. |

---

## 9. Invariants

| Invariant | Enforcement |
|---|---|
| Every `Pairing` references an existing `Guide` and `Sensor` | `PublishTemplate` validates referential integrity at commit time |
| A `FitnessGate` can only be created from a computational sensor | `PromoteFitnessGate` rejects inferential sensors (per ADR-004 D4) |
| Precedents require PROV-O audit trail | `ApplyPrecedent` fails if provenance record cannot be minted |
| Template topology scope must name at least one substrate | Schema validation on `PublishTemplate` |
| Pairing `validation_mode: blocking` requires a computational sensor | Schema validation — inferential sensors cannot be blocking pairings |
| Janitor findings above severity threshold trigger operator notification | `RunJanitor` checks severity; notification via Nostr mirror hook (fail-open) |
| Template versions are append-only | No delete or in-place mutation; new versions supersede |
| Cross-substrate templates must reference the mesh smoke test | `PublishTemplate` checks that cross-substrate templates include a reference to a smoke test protocol |

---

## 10. Integration with Existing Contexts

### 10.1 Judgment Broker Integration

The harness engineering context extends the Judgment Broker in two directions:

**Downstream (consuming decisions):**
- When `handleGovernanceDecision()` is wired (M1), each applied decision is checked against the active harness template for the relevant topology. The sensor that corresponds to the decision's guide is triggered to validate the application.

**Upstream (generating feedforward):**
- When a `DecisionOutcome::Promote` occurs, the harness engineering context creates a new `Guide` entry from the promoted decision, pairs it with a `Sensor` (precedent match check), and adds the pairing to the relevant template.

### 10.2 Ecosystem Alignment Integration

The compatibility matrix (`docs/architecture/compatibility-matrix.md`) gains new columns:

| Substrate | ... existing columns ... | Guides | Sensors | Paired | Ratio | Fitness Gates |
|---|---|---|---|---|---|---|
| agentbox | ... | 14 | 8 | 6 | 43% | 3 |
| VisionClaw | ... | 11 | 5 | 4 | 36% | 2 |
| ... | ... | ... | ... | ... | ... | ... |

Maturity vocabulary applies to harness templates:
- `planned` — template schema drafted, no guides/sensors registered
- `scaffolded` — guides and sensors identified, pairings not validated
- `standalone` — template works for single-substrate operations
- `integrated` — cross-substrate pairings validated by ≥2 substrates
- `verified` — end-to-end smoke test passes with all pairings active

### 10.3 Agent Runtime Integration

Agent specs gain an optional `hooks.validate` field:

```yaml
name: coder
hooks:
  pre: |
    npx claude-flow@v3alpha memory search --query "$TASK" --limit 5 --use-hnsw
    npx claude-flow@v3alpha hooks intelligence --action pattern-search
  validate: |
    harness_inspect --topology "$TOPOLOGY" --check-output "$OUTPUT"
  post: |
    npx claude-flow@v3alpha hooks intelligence --action pattern-store
    npx claude-flow@v3alpha neural train --pattern-type coordination
```

Agents without `hooks.validate` continue in two-phase mode. The three-phase lifecycle is opt-in per agent spec, enforced per harness template.

---

## 11. Canonical Harness Templates

### 11.1 `agentbox-agent-task`

| Guide | Sensor | Pairing Mode |
|---|---|---|
| Agent spec `hooks.pre` (capability constraints) | Adapter contract test suite (ADR-031) | Blocking |
| CLAUDE.md instruction hierarchy (3-tier) | `hooks.validate` trajectory check | Advisory |
| Skill routing decision tree | Task router confidence threshold | Advisory |
| MCP tool constraints (per skill) | BC20 anti-corruption drop counter | Learning |
| Adapter SLO thresholds (ADR-005) | SLO p95 latency / throughput / error CI gate | Blocking |

### 11.2 `visionclaw-enrichment`

| Guide | Sensor | Pairing Mode |
|---|---|---|
| OWL 2 EL ontology constraints (domain/range) | Oxigraph health check (class count, axiom count) | Blocking |
| IS-Envelope schema (ADR-075) | IS-Envelope fixture validation (11 vectors) | Blocking |
| Enrichment proposal schema | BrokerActor enrichment gating | Blocking |
| Ontology bridge MCP tool constraints | Ontology bridge health check (12 tools respond) | Blocking |
| PROV-O activity record schema | Provenance chain integrity check | Advisory |

### 11.3 `governance-decision`

| Guide | Sensor | Pairing Mode |
|---|---|---|
| Control Surface kinds 31400–31405 schema | Relay gating rules (NIP-42/NIP-98 auth) | Blocking |
| Panel definition schema (kind 31400) | Governance UI rendering validation | Advisory |
| Decision outcome enum (6 variants) | Decision chain integrity (prior_decision_id links) | Blocking |
| Precedent library (auto-generated) | Precedent match check | Advisory |
| Mesh smoke test protocol | End-to-end smoke test | Blocking (release gate) |

### 11.4 `pod-mutation`

| Guide | Sensor | Pairing Mode |
|---|---|---|
| Solid LDP protocol | WAC ACL evaluation | Blocking |
| NIP-98 signing requirement | Schnorr signature verification | Blocking |
| Content-negotiation rules (Turtle/JSON-LD/N-Triples/N-Quads) | Content-type validation | Blocking |
| Git-marks append-only invariant | Git history integrity check | Blocking |
| Block-trail anchoring rules (BIP-341) | Taproot anchor verification | Advisory |

---

## 12. Open Questions

1. **Template versioning.** When a substrate updates a guide or sensor, does the template version bump automatically? Or is template versioning a manual coordination step?
2. **Inferential sensor promotion.** What false-positive rate threshold qualifies an inferential sensor for promotion to blocking? The 5% target in ADR-004 D4 is a starting point, not a validated threshold.
3. **Precedent similarity threshold.** What embedding similarity score justifies auto-applying a precedent? Too low risks incorrect auto-decisions; too high eliminates the precedent system's value.
4. **Trajectory monitoring cost.** How much latency does in-execution validation add? Is it acceptable for all topologies or only high-risk operations?
5. **Janitor scope creep.** The janitor agent (FR7.1) could expand indefinitely. What is the maintenance budget for automated drift cleanup?
