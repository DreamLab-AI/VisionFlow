# ADR-004: Harness Engineering Framework

**Status:** Accepted (with 2026-07-03 closeout amendments — see note below)
**Date:** 2026-06-23
**Decision Owners:** DreamLab AI maintainers
**Related:** [ADR-002 Ecosystem Alignment Governance](../archive/adr/ADR-002-ecosystem-alignment-governance.md), [ADR-003 Judgment Broker](../archive/adr/ADR-003-judgment-broker-distributed-architecture.md), [PRD-harness-engineering](PRD-harness-engineering.md), [DDD-harness-engineering-context](DDD-harness-engineering-context.md), [closeout final-design](../closeout/final-design.md)
**Provenance:** Martin Fowler, *Harness Engineering for Coding Agent Users* (2025). Cross-referenced against VisionFlow ecosystem audit, 2026-06-23.

> **2026-07-03 closeout amendment.** This ADR, its templates, schema, `scripts/harness-audit.sh` and the `harness-fitness-gates` CI workflow were untracked in git until committed as part of the ecosystem closeout — until then the "Accepted" status had no history and the fitness gate had never run. Two corrections to the original 2026-06-23 text: (1) the "open steering loop" motivation (former gap #1) was **false** — `handleGovernanceDecision()` was fully implemented in agentbox on 2026-05-22 (commit `e1a8d716`), a month before this ADR was written, minting PROV-O activity/receipt URNs and dispatching to agent stdin or the pod; decision application is not a gap. (2) Decision **D3** (the `hooks.validate` phase) is **Deferred**, not implemented — no production agent spec wires a `hooks.validate` phase (grep returns zero matches across the ecosystem). The two harness controls that depend on it (`hooks-validate-trajectory`, and the never-built `slo-ci-gate`) are marked `source_status: planned` in the templates.

## Context

The VisionFlow ecosystem governs agent behaviour through an extensive but informally structured collection of feedforward controls (110 agent specs, 115 skills, 139 commands, 3-tier CLAUDE.md hierarchy, Judgment Broker control panels) and feedback sensors (contract tests, fixture parity, IS-Envelope validation, learning service). These controls grew organically as the ecosystem expanded from one repo to six.

Fowler's harness engineering framework provides the missing organising principle: the distinction between **guides** (feedforward — shape behaviour before acting) and **sensors** (feedback — observe and correct after acting), their **pairing** (every guide should have a corresponding sensor), and their **bundling** into **harness templates** per topology.

Three gaps motivate this ADR. (The original 2026-06-23 draft listed a fourth — an "open steering loop" premised on `handleGovernanceDecision()` being stubbed — which the 2026-07-03 review disproved and withdrew; see the closeout amendment above.)

1. **Unpaired controls.** Agent specs carry `hooks.pre` (guide) and `hooks.post` (learning) but no `hooks.validate` (sensor). Guides are not validated during execution. (The `hooks.validate` remedy is Decision D3 below, which remains Deferred.)
2. **No executable fitness functions.** ADR compliance scores and SLO thresholds exist but don't block merges. Quality is not "left."
3. **Per-repo silos.** Each substrate has its own guides and sensors with no cross-substrate harness template.

## Decision

### D1: Adopt the guide/sensor/pairing model

Every control in the ecosystem is classified as either a **guide** (feedforward) or a **sensor** (feedback). Controls are tracked with their pairing status. The compatibility matrix gains a harness coverage column showing `paired/total` per substrate.

**Rationale:** The informal accumulation of controls makes it impossible to answer "is this aspect of agent behaviour regulated?" The classification and pairing model makes gaps visible.

### D2: Harness templates are the unit of agent governance

A **harness template** is a declarative bundle:

```
HarnessTemplate:
  topology: string           # e.g. "governance-decision"
  structure:                 # what exists in this topology
    substrates: string[]     # repos involved
    event_kinds: integer[]   # Nostr kinds (if applicable)
    data_flows: Flow[]       # directed graph of data movement
  guides: Guide[]            # feedforward controls
    - id: string
      type: enum(instruction | schema | constraint | hook)
      source: string         # file path or MCP tool
      applies_to: string[]   # substrates this guide governs
  sensors: Sensor[]          # feedback controls
    - id: string
      type: enum(computational | inferential)
      source: string         # test file, MCP tool, or hook
      applies_to: string[]
      frequency: enum(per_change | per_commit | per_release | continuous)
  pairings: Pairing[]        # guide ↔ sensor bindings
    - guide_id: string
      sensor_id: string
      validation_mode: enum(blocking | advisory | learning)
  escalation_rules: Rule[]   # when sensors fire, what happens
    - sensor_id: string
      threshold: any
      action: enum(block | warn | file_issue | auto_fix | escalate_human)
```

**Rationale:** Agents need machine-readable governance contracts, not prose documents. The template schema makes harness coverage inspectable via MCP tools.

### D3: Three-phase agent execution lifecycle

**Status: Deferred (2026-07-03).** This decision is not implemented. No production agent spec wires a `hooks.validate` phase; the ecosystem runs the two-phase `hooks.pre → execution → hooks.post` lifecycle today. It is retained here as the accepted *design* for closing the unvalidated-execution gap, to be implemented when the `hooks.validate` phase and its per-guide constraint checker land. Until then the dependent harness sensor `hooks-validate-trajectory` is `source_status: planned`.

Agent execution is extended from two phases to three:

```
hooks.pre (guide injection)
  → execution
    → hooks.validate (sensor verification)
      → hooks.post (learning extraction)
```

The `hooks.validate` phase checks whether the guide's constraints were respected during execution. Validation failures trigger self-correction before the learning phase records the execution as successful.

**Rationale:** The current two-phase model (pre/post) records what happened but never checks whether guidance was followed. This is the article's "feedback-only produces repetitive errors" failure mode in reverse — feedforward-only encodes rules without validation.

### D4: Computational sensors are blocking; inferential sensors are advisory

Sensors are classified by execution type:

| Type | Characteristics | Default gate behaviour |
|---|---|---|
| **Computational** | Deterministic, milliseconds, CPU | Blocking (merge gate) |
| **Inferential** | Probabilistic, seconds-minutes, GPU/LLM | Advisory (report, don't block) |

Computational sensors (tests, linters, type checkers, fixture parity, SLO thresholds) block merges in CI. Inferential sensors (AI code review, semantic duplication, over-engineering detection) report findings but do not block, because their non-determinism makes false positives likely.

**Rationale:** Blocking on non-deterministic results creates flaky pipelines that erode trust. Inferential sensors earn blocking status only after sustained false-positive rates below 5%.

### D5: Cross-substrate templates reference the mesh smoke test

The mesh smoke test protocol (`docs/protocol/mesh-smoke-test.md`) is the seed of the cross-substrate harness. Cross-substrate harness templates extend it with per-hop guide-sensor pairs.

**Rationale:** The smoke test already validates the full decision loop across four repos. Enriching it with guide-sensor pairs at each hop creates a mesh-wide regulation surface without inventing a parallel coordination mechanism.

### D6: Precedents are feedforward controls generated from feedback

When a `DecisionOutcome::Promote` is applied, the approved decision is stored as a standing precedent in RuVector (`governance-precedents` namespace) with semantic embedding. Future matching `ActionRequest` events check precedents before routing to a human. If a precedent matches above threshold, it is auto-applied with a `Precedent` outcome and full PROV-O audit trail.

**Rationale:** This is the article's "steering loop" at its most concrete: recurring feedback (repeated identical human approvals) generates new feedforward (automatic policy). The precedent system converts the most common governance decisions into self-regulating controls, reducing human steering burden without removing human override capability.

### D7: Ashby's Law — topology commitment narrows the regulation space

VisionFlow's multi-substrate architecture initially appears to violate Ashby's Law of Requisite Variety (the regulator must model the full variety of the system). The mitigation is **topology commitment**: each harness template declares the exact substrate set, event kinds, and data flows it governs. By committing to a topology, the harness narrows the production space to what it can comprehensively regulate.

**Rationale:** A harness that claims to govern "everything" governs nothing. Topology-specific templates are honest about their scope and can achieve complete coverage within that scope.

### D8: Harness templates are owned by VisionFlow, not substrates

VisionFlow (this repo) owns the harness template schema and the canonical template instances. Individual substrates own their guides and sensors, but the **pairing** and **bundling** is an ecosystem concern.

**Rationale:** Cross-substrate templates cannot be owned by any single substrate. VisionFlow already owns the compatibility matrix, release manifests, and smoke test protocol — harness templates are the same class of ecosystem coordination artefact.

## Consequences

### Positive

- **Visible coverage.** The guide/sensor pairing model makes regulation gaps visible in the compatibility matrix.
- **Machine-readable governance.** Agents inspect harness templates via MCP tools, receiving structured constraints instead of parsing prose.
- **Quality left.** Computational sensors as CI gates catch issues at merge time, not release time.
- **Reduced steering burden.** Precedent system eliminates repetitive human decisions.
- **Cross-substrate coherence.** Mesh harness templates ensure agents understand the full topology they operate within.

### Negative

- **Template maintenance.** Harness templates are a new artefact class that must be kept in sync with evolving guides and sensors. Mitigated by janitor automation (FR7.1).
- **Validation overhead.** The `hooks.validate` phase adds latency to agent execution. Mitigated by making validation optional per template (some topologies may run validate only on high-risk operations).
- **CI gate friction.** Blocking computational sensors may slow development velocity. Mitigated by D4's separation — only deterministic checks block.

### Neutral

- Agent template format gains a `hooks.validate` field. Existing templates without it continue to operate in two-phase mode.
- The compatibility matrix gains columns. No existing data is removed.

## Alternatives Considered

### A1: Informal harness accumulation (status quo)

Continue adding guides and sensors organically without formal pairing or templates.

**Rejected:** Informal accumulation leaves regulation coverage unmeasurable. Once templated, `scripts/harness-audit.sh` reports it directly: 100% of declared guides are paired to a sensor, but only 33/40 controls (82.5%) are source-backed — the remaining 7 are `planned`. (The "30% pairing" figure in the original draft was an unmeasured estimate, superseded by the audit script's output.) As the ecosystem grows, only a running audit keeps this honest.

### A2: Centralised harness service

Build a dedicated harness service that mediates all agent-guide-sensor interactions.

**Rejected:** Violates the distributed architecture principle (ADR-003 D1). The harness should be an emergent property of coordinated substrate behaviour, not a single point of failure.

### A3: LLM-only inferential validation

Use LLM-as-judge for all validation, eliminating the computational/inferential split.

**Rejected:** Non-deterministic validation as a merge gate creates flaky pipelines. Computational sensors are cheap, fast, and reliable — they should be the default gate. Inferential sensors complement but do not replace them.


## Closeout extension — 2026-09-05

The [section-level engineering assessment](../estate-review/engineering-governance.md) supplies current dispositions and CP-01/04/05/07/08/09 acceptance obligations. Preserve this record's original accepted/deferred/speculative distinctions. Declared pairing, existing helper code and signed mandate primitives are not complete runtime governance evidence.

The [actual audit-script probes](../estate-review/evidence/harness-audit-probe.json) show that nonexistent source paths still count as source-backed and repeated pairing edges can yield 200% coverage. The workflow adds ID/reference checks but does not resolve control sources or invoke the full schema. No hosted CI, agent provisioning, precedent write or mandate operation ran.

Closeout requires source-bound distinct coverage, actual consumer enforcement, authorised durable outcomes and revocation/recovery evidence. See the assessment table for every numbered decision in this record; unresolved proposals must retain explicit adoption or deferral.

## Acceptance progress — 2026-09-05

The **source-bound distinct coverage** obligation is discharged. The other three
(consumer enforcement, authorised durable outcomes, revocation/recovery) are
runtime-governance obligations and remain open; nothing here converts a script
result into runtime evidence.

`scripts/harness-audit.sh` had three defects, each reproduced in
[harness-audit-probe.json](../estate-review/evidence/harness-audit-probe.json):

**1. Source paths were never resolved.** A control counted as "source-backed"
whenever `source_status` was absent or `"present"` — an unchecked
self-declaration. The probe's `missing_sources_default_present` case scored a
template naming two fabricated files at "2/2 controls source-backed (100.0%
present)". Sources are now **resolved** against the substrate checkouts. The
resolver handles the `SUBSTRATE:LOCATOR` form actually in use: `#anchor`
fragments, `~/`-anchored runtime paths, globs, directories, and `' + '`-joined
composites where every part must exist. Each source lands in one of five states
— `resolved`, `unresolved` (declared present, path absent — **not**
source-backed), `planned`, `descriptor` (a non-filesystem artefact such as an
MCP namespace, which no path check can settle), or `unverifiable` (the substrate
is not checked out).

Running it immediately found four declared-present sources that did not exist,
all the same "moved to archive" class as the ADR-2005 allowlist defect. All four
repaired in the templates: `agentbox:docs/reference/adr/ADR-005-…` →
`docs/archive/adr/…`; `visionclaw:docs/adr/ADR-075-…` → `docs/archive/adr/…`;
`agentbox:~/.claude/agents/*.md` → the in-repo `agentbox:agents/*.md` (the home
path is empty in this container, the repo path is real and checkable); and the
CLAUDE.md-hierarchy composite's ambiguous `project/CLAUDE.md` third tier
disambiguated to `~/workspace/project/CLAUDE.md`.

**2. Pairing edges were counted with duplicates.** The ratio was
`raw_pairing_count / max(guides, sensors)`, so the probe's `duplicate_pairings`
case — one guide, one sensor, the same pairing twice — scored **200.0%** and
passed any target. Edges are now de-duplicated on `(guide_id, sensor_id)`, and
coverage counts **distinct paired guides and sensors** over the controls that
exist. Since the paired sets are subsets of the declared sets, the ratio is
bounded by 100% by construction rather than by hoping nobody repeats an edge.
Edges referencing a non-existent control are reported as dangling and cannot
evidence coverage.

**3. Backing was counted per control, not per source.** Two controls naming one
file counted as two backed controls. Backing is now capped at **distinct
sources**; per-control figures are still printed alongside.

Substrate availability follows the estate's partial-source failure mode
(ADR-005 §Decision 2, as the drift counter already uses it): an absent checkout
yields `unverifiable`, excluded from the denominator and reported, rather than
counted as failure. `--strict-sources` turns unresolved and unverifiable into
exit 1 for environments where every substrate is present.

**Result.** The honest figure is lower than the fabricated one, which is the
point: the old script reported 33/40 controls source-backed (82.5%); the audit
now reports **31/39 distinct sources resolved (79.5%)** with **zero unresolved**,
7 planned and 1 descriptor. Pairing coverage stays 100% (40 of 40 distinct
paired controls over 20 de-duplicated edges) — that figure was always true, it
just could not previously be distinguished from a false one.

**Tests.** `tests/gates/harness-audit.test.sh`, 26 assertions, all passing, and
wired into `harness-fitness-gates.yml` to run *before* the audit so CI proves the
gate can fail before trusting it to pass. It replays both probe cases: [1] the
duplicate-pairing template now scores 100% with the duplicate reported and no
200% anywhere; [2] the fabricated-source template scores 0/2 backed and lists
both, with [2b] exiting 1 under `--strict-sources`. [4] pins distinct-source
capping, [5] the unverifiable-substrate path, [6] dangling edges, [7] the real
templates at zero unresolved.

Receipts: [harness audit log](../estate-closeout/2026-09-05/logs/harness-audit.log),
[gate self-test log](../estate-closeout/2026-09-05/logs/gate-tests.log),
[gate closeout receipt](../estate-closeout/2026-09-05/gate-closeout-receipt.json).

**Remaining.** No hosted CI run; no agent provisioning, precedent write or
mandate operation was executed, so consumer enforcement and durable-outcome
evidence are untouched. The audit still does not validate templates against
`harness-template.schema.json` — `harness-fitness-gates.yml` checks required
fields, ID uniqueness and pairing references by hand rather than invoking a JSON
Schema validator. D3 (`hooks.validate`) remains **Deferred**: its two dependent
controls stay `source_status: planned`, now with resolvable paths. The
`ruvector:governance-precedents namespace` guide is classified `descriptor` —
honestly unverifiable by a filesystem check, and it needs a queryable source to
become backed. §Context's "115 skills" is left as written: it is a dated
statement of the situation at decision time, and the drift counter's scan list
was narrowed to exclude ADR Context paragraphs for exactly that reason (the
current sourced figure is 126).

Governed paths changed: `scripts/harness-audit.sh`,
`docs/engineering/templates/agentbox-agent-task.json`,
`docs/engineering/templates/visionclaw-enrichment.json`,
`.github/workflows/harness-fitness-gates.yml`,
`tests/gates/harness-audit.test.sh` (new).
