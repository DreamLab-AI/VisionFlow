# DDD: Gap-Close Sprint Bounded Context

**Status:** Living document
**Date:** 2026-07-08
**Scope:** The cross-repository sprint that closes the four-surface gap register
**Governed by:** [PRD Gap-Close Sprint](PRD-gap-close-sprint.md), [ADR-004 Gap-Close Sprint Governance](ADR-004-gap-close-sprint-governance.md)

---

## 1. Bounded Context

The Gap-Close Context governs the coordinated remediation of the four-surface gap register (book Chapter 14b) across six repositories. It owns the vocabulary and invariants of *how a measured gap becomes a verified closure* — the register, the waves, the work packages, the evidence, the canaries. It does not own the code that closes any gap; that belongs to the substrate contexts. This context orchestrates the flow of a gap from published-and-owned, through falsification-stated and worked, to canary-fired and closed at a stated maturity tier.

It is deliberately downstream of two existing contexts. The **Ecosystem Alignment** context supplies the maturity vocabulary, the compatibility matrix, and the rule that the canon owns the cross-repo view while repositories own implementation. The **Judgment Broker** context supplies the decision-loop domain that most of the critical gaps live inside. This context adds nothing to either model; it consumes both.

---

## 2. Context Map

| Context | Relationship | Notes |
|---|---|---|
| **Gap-Close Sprint** (this context) | Coordinates remediation flow across substrates | Defines the register-to-closure lifecycle |
| **Ecosystem Alignment** ([DDD](DDD-ecosystem-alignment-context.md), [ADR-002](ADR-002-ecosystem-alignment-governance.md)) | Upstream | Owns the maturity vocabulary, compatibility matrix, and canon-owns-cross-repo rule this context obeys |
| **Judgment Broker** ([DDD](DDD-judgment-broker-context.md), [ADR-003](ADR-003-judgment-broker-distributed-architecture.md)) | Upstream | Owns the decision-loop domain most critical gaps sit within (forum admin-only, desktop no-ACSP-surface, voice loop) |
| **VisionClaw / agentbox / solid-pod-rs / nostr-rust-forum / dreamlab-ai-website** | Downstream (Supplier) | Each supplies a RepoWorkPackage and its child documents; this context defines the closure protocol they follow |

### Relationship Types

- **Ecosystem Alignment → Gap-Close:** Conformist. This context conforms to the ADR-002 seven-tier maturity vocabulary and the release-manifest machinery without redefining them.
- **Judgment Broker → Gap-Close:** Conformist. Gaps that touch the decision loop are typed and closed against the broker context's aggregates (`BrokerCase`, `DecisionOutcome`), not a parallel model.
- **Gap-Close → substrate repositories:** Customer/Supplier. Each repository supplies its work package and evidence; this context defines the falsification, receipt and canary protocol the supply must meet.

---

## 3. Aggregates

| Aggregate | Root | Description |
|---|---|---|
| `GapRegister` | Yes (context root) | The consolidated inventory: 28 register gaps, 18 commitments, 5 residuals. **Immutable once published.** It is superseded by a new versioned register, never edited in place — mirroring the broker context's decision-immutability invariant. A correction is a new register that cites the old. |
| `SprintWave` | Yes | An ordered closure set (P0, P1, P2). Holds the items admitted to it and the canaries that gate its closure. A wave opens only when the prior wave's canaries are green in the substrates it touched. |
| `RepoWorkPackage` | Yes | One repository's owned slice of the register, its child PRD/ADR pair, and its falsification statements. Consistency boundary: a repository's closures are evidenced and reconciled together, at the canon, per ADR-002. |
| `Gap` | No (member of `GapRegister`) | A single typed, severity-rated finding with exactly one owning repository, one wave, and one child document. |
| `Commitment` | No (member of `GapRegister`) | A scheduled roadmap item; may subsume one or more Gaps by cross-reference. |
| `ClosureEvidence` | No (member of `RepoWorkPackage`) | The receipt (command, raw output, timestamp, git SHA) plus the maturity claim and the canary result for one closed item. |

---

## 4. Entities

| Entity | Identity | Owner |
|---|---|---|
| `Gap` | Register ID (e.g. `F1`, `D4`, `M1`, `V1`) | GapRegister (VisionFlow canon) |
| `Commitment` | Roadmap ID (`REC-1`…`REC-12`, `COM-13`…`COM-18`) | GapRegister (VisionFlow canon) |
| `Residual` | Residual ID (`RES-a`…`RES-e`) | GapRegister (VisionFlow canon) |
| `ChildDocument` | Repository slug + `gap-close` + type (PRD/ADR) | Owning repository |
| `LivenessCanary` | Canary ID registered against the harness | VisionClaw (harness), owning repo (registration) |

---

## 5. Value Objects

| Value Object | Fields | Notes |
|---|---|---|
| `GapType` | theory→canon, canon→practice, theory→practice | The remedy differs by type: design position, engineering (or honest retraction), or both. |
| `Severity` | Critical, Major, Minor | Fixed by the register's adversarial judges; not re-rated in this context. |
| `Wave` | P0, P1, P2 | Correctness/security preconditions; measurement + embodiment join + governance surfaces; extension. |
| `MaturityTier` | historical, planned, scaffolded, standalone, integrated, federation-verified, released | The ADR-002 vocabulary, used verbatim. A closure states target and current tier. |
| `ExitCriterion` | Predicate over live system state | The observable condition that, met, closes the item. Carried from the register and roadmap. |
| `FalsificationStatement` | Predicate whose truth means *not done* | Authored before the work. |
| `EvaluationLens` | {Forum:40, Desktop:30, MR:20, Voice:10} | The scoring frame from book Chapter 12a. |

---

## 6. Domain Events

| Event | Trigger | Publisher | Consumer |
|---|---|---|---|
| `RegisterPublished` | Meta-PRD inventory ratified | VisionFlow canon | All RepoWorkPackages |
| `WorkPackageMinted` | Repository authors its child PRD/ADR pair | Owning repository | Gap-Close context |
| `FalsificationStated` | Falsification statement written before work | Owning repository | Gap-Close context |
| `CanaryRegistered` | Loop item registers a liveness canary | Owning repository | SprintWave |
| `CanaryFired` | Canary observes live traffic on the wire | Liveness harness | SprintWave, ClosureEvidence |
| `ClosureEvidenced` | Receipt + maturity claim + canary result recorded | Owning repository | Anti-fox verifier |
| `ClosureVerified` | Different-family verifier confirms against a counter-example probe | Verifier | GapRegister, compatibility matrix |
| `WaveOpened` | Prior wave's canaries green in touched substrates | Gap-Close context | Owning repositories |
| `RegisterSuperseded` | A new versioned register corrects a published one | VisionFlow canon | All consumers |

---

## 7. Invariants

1. **One owner, one child document.** Every `Gap`, `Commitment` and `Residual` has exactly one owning repository and is discharged by exactly one `ChildDocument`. An unowned or double-owned item is a register defect.

2. **Closure is code-verified at the stated tier.** An item is `Closed` only with `ClosureEvidence` carrying a receipt at the claimed `MaturityTier`. Documentation alone never closes an item — the register's D5 (fabricated status) and desktop-beam ("wired at boot" over a dead wire) findings are the standing counter-examples this invariant forbids.

3. **Falsification precedes the work.** A `FalsificationStatement` for an item MUST exist before implementation of that item begins. Work started without one is out of process.

4. **No canary, no closure.** A loop-closing item closed without its `LivenessCanary` having fired in a live session is not closed, regardless of evidence. An accepted design whose canary never fires registers as `Open`, visibly — the structural answer to the codebase's own history of designs accepted and abandoned (ADR-043's three dormant months).

5. **The register is immutable once published.** `GapRegister` is superseded by a new version, never edited in place. The published counts, file paths and severities stand as the checkable claim; corrections chain forward.

6. **Maturity is claimed conservatively.** No item is labelled above the tier its evidence supports. A deferred sub-feature is labelled `scaffolded` or `planned`, never folded silently into a `Closed` parent.

7. **Waves gate.** A `SprintWave` opens only after the prior wave's canaries are green in the substrates it touched. Severity does not override wave order; a critical gap in P2 waits on P0 and P1.

8. **The verifier is not the producer.** `ClosureVerified` is published by a party distinct from the one that produced the closure, on a different model family, having run at least one adversarial counter-example probe.

---

## 8. Ubiquitous Language

| Term | Meaning |
|---|---|
| **Gap** | A single typed, severity-rated finding from the four-surface register: published knowledge unadopted (theory→canon), a documented capability the code lacks (canon→practice), or both (theory→practice). |
| **Commitment** | A scheduled roadmap item (book Chapter 14a) that may subsume one or more Gaps; the promise the next edition is marked against. |
| **Residual** | A gap surfaced by the July 2026 book-production pass, outside the register and roadmap (RES-a…RES-e). |
| **Wave** | An ordered closure set — P0 preconditions, P1 measurement-and-surfaces, P2 extension — that gates the next. |
| **Exit Criterion** | The observable live-system condition that, met, closes an item; carried from the register and roadmap unchanged. |
| **Evaluation Lens** | The 40/30/20/10 forum/desktop/MR/voice weighting that scores sprint progress. |
| **Child Document** | The PRD/ADR pair a repository mints for its slice, reconciled at the canon per ADR-002. |
| **Closure Evidence** | A receipt (command, raw output, timestamp, git SHA) plus a maturity claim and a canary result — the only thing that closes an item. |
| **Liveness Canary** | A probe that must observe live traffic on a wired loop before that loop is closed; the answer to "built, and unwired". |
| **Falsification Statement** | The pre-authored condition under which a work package is judged not done. |

---

## 9. Services

| Service | Responsibility | Owner | Status |
|---|---|---|---|
| `RegisterKeeper` | Publishes and supersedes the immutable `GapRegister`; holds the consolidated inventory | VisionFlow canon | `planned` (this PRD ratifies it) |
| `WaveGate` | Admits items to a wave; opens a wave when the prior wave's canaries are green | Gap-Close context | `planned` |
| `LivenessHarness` | Registers canaries and records `CanaryFired` in live sessions | VisionClaw (RES-a) | `planned` |
| `EvidenceLedger` | Records `ClosureEvidence` receipts and maturity claims per item | Owning repository | `planned` |
| `AntiFoxVerifier` | Independently confirms a closure on a different model family with a counter-example probe | Cross-repo (build-with-quality discipline) | `planned` |
| `DriftCounter` | Single source of truth for skill count, ontology class count, roster size; CI-enforced | VisionFlow (RES-d) | `planned` |

---

## 10. Ownership Summary

| Repository | Owns in this context | Does not own |
|---|---|---|
| **VisionFlow** (canon) | `GapRegister`, `RegisterKeeper`, `WaveGate`, `DriftCounter`, canon reconciliations, the diagram-render gate | Any substrate's implementation or maturity claim |
| **VisionClaw** | Desktop/MR/voice work packages, `did:nostr` keying, the embodiment join, the `LivenessHarness` | The forum decision surface, the agent runtime, the pod layer |
| **agentbox** | Authority model, MAST telemetry, outcome learning, the voice-intent producer, `did:nostr` source-at-spawn | The render surfaces, the governance UI, the canon |
| **solid-pod-rs** | PATCH correctness, the shared NIP-98 verifier, the pod provenance trail | Governance, embodiment, agent runtime |
| **nostr-rust-forum** | The member surface, disclosure, decision integrity, escalation, roster admin, NIP-42 | Embodiment, agent runtime, pod persistence |
| **dreamlab-ai-website** | The kit cutover and the operator overlay for disclosure and roster | Any protocol source — it is a consumer, not an owner |

---

## 11. Open Issues

1. **Register versioning cadence.** The register is immutable-and-superseded; the trigger for cutting a new version (per-wave? on any correction?) is not yet fixed. Default: a new version at each wave boundary, corrections chaining forward.
2. **Canary durability.** Whether a fired canary must remain green (a standing monitor) or firing once suffices for closure. The measurement commitments imply standing monitors for the KPI-feeding loops; one-shot firing may suffice for correctness fixes. Resolve per item in the child ADRs.
3. **Cross-repo item accounting.** REC-1 and COM-15 span more than one repository; the meta-PRD assigns a primary owner, but the sub-item boundary between owners is fixed only in the child documents, not here.
