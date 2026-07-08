# DDD: Gap-Close Canon Slice

**Status:** Living document
**Date:** 2026-07-08
**Scope:** The VisionFlow canon's owned slice of the gap-close sprint — the register, the counter, the diagram gate, the position documents, the reconciliations
**Governed by:** [PRD Gap-Close Canon](PRD-gap-close-canon.md), [ADR-005 Gap-Close Canon Decisions](ADR-005-gap-close-canon-decisions.md)
**Conformist to:** [DDD Gap-Close Context](DDD-gap-close-context.md) (parent), [ADR-002 Ecosystem Alignment](ADR-002-ecosystem-alignment-governance.md), [DDD Judgment Broker Context](DDD-judgment-broker-context.md)

---

## 1. Bounded Context

This is the canon's view inside the Gap-Close Sprint context, not a new context. It owns the parts of the register-to-closure flow the canon is responsible for: publishing and superseding the `GapRegister`, gating waves, running the `DriftCounter`, holding the diagram-render gate, authoring the `theory→canon` position documents, and reconciling the canon's own claims. It adds nothing to the parent Gap-Close model or to the upstream Ecosystem Alignment and Judgment Broker models; it consumes all three and names the canon's slice of them.

The canon owns claims, counts, renders and adjudication about the whole. It does not own any substrate's implementation or maturity claim. Where this slice touches the judgment-broker domain (F6 supersession authority), it extends that context's aggregate through its owner, it does not fork a parallel model.

---

## 2. Context Map

| Context | Relationship | Notes |
|---|---|---|
| **Gap-Close Sprint** ([DDD](DDD-gap-close-context.md)) | Parent (Conformist) | This slice conforms to the parent's aggregates, events and invariants verbatim; it specialises `RegisterKeeper`, `WaveGate` and `DriftCounter` to the canon owner |
| **Ecosystem Alignment** ([ADR-002](ADR-002-ecosystem-alignment-governance.md)) | Upstream (Conformist) | Supplies the seven-tier maturity vocabulary, the compatibility matrix, the release-manifest machinery, and the canon-owns-cross-repo rule |
| **Judgment Broker** ([DDD](DDD-judgment-broker-context.md)) | Upstream (Conformist) | Owns `DecisionOutcome` and Invariant 5; the canon's F6 supersession authority extends §7 of that context through its owner (the forum), not here |
| **agentbox / VisionClaw / nostr-rust-forum** | Downstream (Supplier) | Expose script-queryable count sources to the `DriftCounter`; cite the canon's position documents and disclosure norm |

### Relationship Types

- **Gap-Close → Canon slice:** Conformist. Every aggregate, event and invariant below is the parent's, scoped to the canon owner.
- **Ecosystem Alignment → Canon slice:** Conformist. Maturity tiers and the compatibility matrix are used verbatim; the canon reflects a substrate's tier, it does not redefine the vocabulary.
- **Canon slice → substrates:** Customer/Supplier. The substrates supply count sources and cite canon positions; the canon defines the contract those supplies meet.

---

## 3. Aggregates

Conformist to the parent Gap-Close context (its §3). The canon is the owner of the context root.

| Aggregate | Root | Canon's slice |
|---|---|---|
| `GapRegister` | Yes (context root) | The canon publishes and supersedes it. **Immutable once published**; the ADR-111 and count reconciliations chain forward as a new version, never edit the published register in place |
| `SprintWave` | Yes | The canon's `WaveGate` admits items and opens a wave when the prior wave's canaries are green in the touched substrates |
| `RepoWorkPackage` | Yes | This trio (`PRD-gap-close-canon.md`, `ADR-005`, this DDD) is the canon's work package; its consistency boundary is the canon's owned items reconciled together at each wave |
| `ReconciledClaim` | No (member of `RepoWorkPackage`) | One canon assertion the sprint touches: its old wording, its new wording, and the child document that discharges it. Chains forward when the register is superseded |
| `CountedAxis` | No (member of the `DriftCounter`) | One self-description number (skill count, ontology class count, MCP ontology-bridge tool count, roster size) with exactly one script-queryable source |
| `ClosureEvidence` | No (member of `RepoWorkPackage`) | The receipt, maturity claim and canary result for one closed canon item |

---

## 4. Entities

| Entity | Identity | Owner |
|---|---|---|
| `Gap` | Register ID (`F10`, `V2`) | `GapRegister` (canon) |
| `Residual` | Residual ID (`RES-b`, `RES-d`, `RES-e`) | `GapRegister` (canon) |
| `ReconciledClaim` | Claim locus (file:line or file) | Canon `RepoWorkPackage` |
| `PositionDocument` | `docs/protocol/` slug (`forum-social-dynamics.md`, `voice-addressing.md`) | Canon |
| `CountedAxis` | Axis name (skills, ontology-classes, mcp-ontology-tools, roster) | Canon `DriftCounter` |
| `LivenessCanary` | Canary ID (`CANARY-CANON-DIAGRAM`, `CANARY-CANON-DRIFT`, `CANARY-CANON-CLAIMS`) | VisionClaw (`LivenessHarness`), canon (registration) |

---

## 5. Value Objects

Conformist to the parent (its §5). Canon-local additions below the line.

| Value Object | Fields | Notes |
|---|---|---|
| `GapType` | theory→canon, canon→practice, theory→practice | F10 and V2 are both `theory→canon`: the remedy is a design position |
| `MaturityTier` | historical, planned, scaffolded, standalone, integrated, federation-verified, released | ADR-002 vocabulary, verbatim. A position reaches `standalone` when authored, `integrated` when ≥2 substrates cite it |
| `Wave` | P0, P1, P2 | RES-b and COM-13 are P0; RES-d is P1; F10, V2, RES-e, COM-18, D7, F6, F9 are P2 |
| `FalsificationStatement` | Predicate whose truth means *not done* | Authored in `PRD-gap-close-canon.md` before any work |
| `ForkCriterion` | Predicate over cross-repo state | The F9 build-or-park condition; all three sub-predicates must hold to build |
| `DisclosureNorm` | The `did:nostr` self-identification MUST | Written into `identity-spine.md`; cited by forum and VisionClaw |

---

## 6. Domain Events

Conformist to the parent (its §6). The canon publishes or consumes the subset below; the three canon-local events specialise `CanaryFired` and the reconciliation flow.

| Event | Trigger | Publisher | Consumer |
|---|---|---|---|
| `RegisterPublished` | Meta-PRD inventory ratified | canon `RegisterKeeper` | all `RepoWorkPackage`s |
| `WorkPackageMinted` | This trio authored | canon | Gap-Close context |
| `FalsificationStated` | Falsification written before work | canon | Gap-Close context |
| `CanaryRegistered` | A canon loop item registers a canary | canon | `SprintWave`, `LivenessHarness` |
| `CanaryFired` | A canon CI gate blocks on its counter-example probe | `LivenessHarness` | `SprintWave`, `ClosureEvidence` |
| `ClaimReconciled` | A canon claim's wording is corrected and forward-chained | canon `RegisterKeeper` | compatibility matrix, all consumers |
| `WaveOpened` | Prior wave's canaries green in touched substrates | canon `WaveGate` | owning repositories |
| `RegisterSuperseded` | A new versioned register corrects a published one | canon `RegisterKeeper` | all consumers |

---

## 7. Invariants

Conformist to the parent Gap-Close context (its §7, verbatim); the canon-specific readings below make each concrete for this slice.

1. **One owner, one child document.** F10, V2, RES-b, RES-d, RES-e and each reconciliation are discharged by exactly this trio. An unowned or double-owned canon item is a register defect.

2. **Closure is code-verified at the stated tier.** No canon item closes on documentation alone. RES-b closes on a fired `CANARY-CANON-DIAGRAM`, RES-d on a fired `CANARY-CANON-DRIFT`, the reconciliation set on a fired `CANARY-CANON-CLAIMS`. A position document closes to `standalone` on being authored and to `integrated` only when ≥2 substrates cite it.

3. **Falsification precedes the work.** The falsification statement in `PRD-gap-close-canon.md` exists before any counter, gate or page is built.

4. **No canary, no closure.** A canon loop item without a fired canary registers as `Open`, visibly. An accepted counter that never runs in CI is the canon's "built, and unwired".

5. **The register is immutable once published.** The ADR-111 and count corrections chain forward as a superseding `GapRegister` version; `PRD-gap-close-sprint.md` and `docs/closeout/unified-findings-register.json` are never edited in place.

6. **Maturity is claimed conservatively.** F10 and V2 stay `planned` in this minting phase because their pages are not yet written. A deferred sub-feature is labelled `scaffolded`/`planned`, never folded into a closed parent.

7. **Waves gate.** The canon's P0 items (RES-b, COM-13) close before its P1 (RES-d) and P2 (F10, V2, RES-e, COM-18, D7, F6, F9) enter closure. Severity does not override wave order.

8. **The verifier is not the producer.** The party that confirms a canon closure sits on a different model family from the one that produced it and runs at least one counter-example probe (the injected drift, the missing-node `.mmd`, the over-claimed tier).

9. **One number, one source (canon-local).** Every `CountedAxis` has exactly one script-queryable source of truth; a second distinct figure for an axis anywhere in the tree is a drift defect the counter fails CI on.

---

## 8. Ubiquitous Language

Conformist to the parent (its §8). Canon-local terms below.

| Term | Meaning |
|---|---|
| **Reconciled Claim** | One canon assertion the sprint corrects: old wording, new wording, and the child document that discharges it. Chains forward, never edited in place |
| **Counted Axis** | One self-description number (skill count, ontology class count, MCP ontology-bridge tool count, roster size) with a single script-queryable source |
| **Drift Counter** | The canon-owned script and CI gate that reads the four counted axes from substrate-exposed sources and fails CI on disagreement or a second figure |
| **Diagram Gate** | The canon-owned render + visual-regression CI job over `presentation/report/diagrams/*.mmd`, resolving the phantom ADR-111 |
| **Position Document** | A `docs/protocol/` page stating a `theory→canon` position (F10, V2) a substrate cites as its canon contract |
| **Disclosure Norm** | The `did:nostr` self-identification MUST the canon states and the forum enforces (COM-13) |
| **Fork Criterion** | The F9 build-or-park predicate; federation is built only when all three sub-predicates hold |
| **Register Keeper** | The canon service that publishes and supersedes the immutable `GapRegister` |
| **Wave Gate** | The canon service that opens a wave only on the prior wave's green canaries in the touched substrates |

---

## 9. Services

Conformist to the parent Gap-Close context (its §9); the canon owns four of the six named there and specialises them here.

| Service | Responsibility | Owner | Status |
|---|---|---|---|
| `RegisterKeeper` | Publishes and supersedes the immutable `GapRegister`; forward-chains the ADR-111 and count corrections | VisionFlow canon | `planned` → `integrated` on first publish + supersede |
| `WaveGate` | Admits items to a wave; opens a wave on the prior wave's green canaries | VisionFlow canon | `planned` → `integrated` on first gated promotion |
| `DriftCounter` | Single source of truth for the four counted axes; CI-enforced against substrate-exposed sources | VisionFlow canon (RES-d) | `planned` → `integrated` on fired `CANARY-CANON-DRIFT` → `released` on manifest pin |
| `DiagramGate` | Renders `.mmd` sources and fails PRs on render error or invisible-text regression | VisionFlow canon (RES-b) | `planned` → `integrated` on fired `CANARY-CANON-DIAGRAM` → `released` on manifest pin |
| `LivenessHarness` | Registers canaries, records `CanaryFired` in live sessions | VisionClaw (RES-a) | `planned` (canon registers against it) |
| `AntiFoxVerifier` | Confirms a canon closure on a different model family with a counter-example probe | Cross-repo (build-with-quality) | `planned` |

---

## 10. Cross-Context Extension: F6 Supersession Authority

F6 extends the Judgment Broker context, it does not model a parallel one. The canon states the authority model (ADR-005 §Decision 5); the forum, which owns `DecisionOutcome` and the decision surface, writes it into `DDD-judgment-broker-context.md` §7 and implements it.

- **Extends** Judgment Broker Invariant 5 ("no undo — only new events that supersede") with the authority it omits: who may supersede, and how.
- **Authority.** The original human signer, or a human holding a governance role above the original at the time of supersession, publishes a new signed kind-31403 referencing the superseded event with a stated reason.
- **Conformance.** Revoke is supersession by a `Reject` referencing a prior `Approve`. Appeal is a fresh kind-31402 citing the superseded decision. No mutation of any published event; the Nostr event stays the audit trail.

The canon holds the position; the forum's `PRD-gap-close-forum.md` implements and cites it. Reaches `integrated` when the forum cites it and a superseding decision is demonstrated end to end.

---

## 11. Ownership Summary

| Repository | Owns in the canon slice | Does not own |
|---|---|---|
| **VisionFlow** (canon) | `GapRegister`, `RegisterKeeper`, `WaveGate`, `DriftCounter`, `DiagramGate`, F10/V2 position documents, the COM-13 disclosure norm, the F6 supersession-authority position, the F9 fork criteria, the reconciled claims | Any substrate's implementation or maturity claim; the forum's F6/COM-13 implementation; the `LivenessHarness` |
| **agentbox** | Exposes the skill-count manifest and the `ontology-bridge.js` tool-registry count to the `DriftCounter` | The counter, the gate, the canon positions |
| **VisionClaw** | Exposes the ontology class count (Oxigraph query or committed file) to the `DriftCounter`; owns the `LivenessHarness` the canon registers against; cites `voice-addressing.md` for COM-15 | The counter, the diagram gate, the canon register |
| **nostr-rust-forum** | Exposes the agent roster to the `DriftCounter`; implements the COM-13 disclosure badge and the F6 supersession authority citing the canon; implements F9 only if the fork resolves to build | The counter, the register, the fork criteria |

---

## 12. Open Issues

1. **Register versioning cadence.** Conformant to the parent's default (a new version at each wave boundary, corrections chaining forward). The first forward-chained corrections are the ADR-111 resolution and the count reconciliations.
2. **Canary durability.** `CANARY-CANON-DRIFT` and `CANARY-CANON-CLAIMS` are standing monitors (a CI gate that must stay green on every PR). `CANARY-CANON-DIAGRAM` is standing on any PR touching a `.mmd`. None is one-shot; the canon's canaries are all wire-standing, not fire-once.
3. **Roster axis source.** The agent roster count is not yet single-sourced; the forum must expose `agent_registry` as script-queryable before the counter can track that fourth axis. Until it does, the counter runs on three axes and the roster axis stays `planned`.
4. **Dual ADR numbering.** `docs/ADR-005-*` (this) and `docs/engineering/ADR-005-*` (mandate-at-grant) share a number across two sequences. Flagged as a housekeeping hazard; not resolved in this sprint.
