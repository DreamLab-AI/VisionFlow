# ADR-004: Gap-Close Sprint Governance

**Status:** Proposed
**Date:** 2026-07-08
**Decision Owners:** DreamLab AI maintainers
**Related:** [PRD Gap-Close Sprint](PRD-gap-close-sprint.md), [DDD Gap-Close Context](DDD-gap-close-context.md), [ADR-002 Ecosystem Alignment Governance](ADR-002-ecosystem-alignment-governance.md), [ADR-003 Judgment Broker Distributed Architecture](ADR-003-judgment-broker-distributed-architecture.md), book chapters "The Gap Register" (14b) and "Evaluating the Living Experiment" (14a)

## Context

The four-surface audit in book Chapter 14b fixed the numbers: 28 typed gaps across forum, desktop, mixed reality and voice. Chapter 14a converted them into 18 scheduled commitments, and the July 2026 book-production pass added 5 residuals. The material is now diagnosed and scheduled but not yet governed as work: 51 line items sit across six repositories with no decision about how they are closed together.

Two prior decisions constrain the answer. ADR-002 already owns the umbrella alignment process — release manifests, the compatibility matrix, the seven-tier maturity vocabulary, and the rule that repository-local docs stay authoritative for implementation while the canon owns the cross-repo view. ADR-003 already established that the judgment broker is distributed by design and no single repository owns the full loop. Any sprint that closes these gaps must respect both: it cannot centralise ownership, and it cannot let each repository drift while claiming ecosystem-level closure.

The register's own through line sharpens the risk. The failures are overwhelmingly "built, and unwired" — a capability complete to one missing call, then declared done in a closeout that the code does not support (the desktop beam actor marked "wired at boot" while `connect()` is called nowhere; the "MCP Connected" dot hardcoded `true`). A governance process that accepts documentation as closure would reproduce exactly the defect it is meant to close.

## Decision

Run **one coordinated cross-repository sprint**, governed by the meta-PRD (`PRD-gap-close-sprint.md`), in which each repository **mints its own child PRD/ADR pair** for its slice of the register and reconciles it at the VisionFlow canon through the existing ADR-002 machinery.

Specifically:

1. **The canon owns one register, each repository owns its slice.** The meta-PRD holds the consolidated inventory and assigns every item exactly one owning repository and one wave. Each owning repository authors a child PRD (its slice as a product increment) and a child ADR (the decisions that slice forces). The canon owns the cross-repo view; the children own the implementation, per ADR-002.

2. **Closure is code-verified at a stated maturity tier, never documentation alone.** Every closed item carries an execution receipt (command, raw output, timestamp, git SHA) and a maturity claim in the ADR-002 vocabulary. A claim above the tier its evidence supports is a governance defect, treated as the register treats D5.

3. **Every loop-closing item ships with a liveness canary.** A loop-closing item — anything that wires an agent action, decision, beam, voice command or provenance record end to end — is not closed until a canary fires in a live session proving the wire carries traffic. This is a hard gate, not a recommendation, because the codebase's history is the standing proof that an accepted design is not a running one.

4. **Waves gate one another.** P0 (correctness and security preconditions), then P1 (measurement, the embodiment join, the governance surfaces), then P2 (MR parity, the voice conversation layer, federation). A wave opens only when the prior wave's canaries are green in the substrates it touched.

5. **The four-surface lens is the scoring frame.** Progress is scored on the 40/30/20/10 forum/desktop/MR/voice weighting, against the per-surface "closed" definitions carried in the meta-PRD. The falsification condition is fixed now, before the data exists.

6. **Anti-fox verification.** The party that verifies a closure is not the party that produced it, and sits on a different model family — the register's auditor-and-audited discipline made operational per the build-with-quality skill.

7. **Conflict resolution follows ADR-002's division.** Where a child ADR and the canon register disagree, the canon wins on cross-repo scope, wave assignment and maturity tier; the child wins on local implementation detail. This is ADR-002's existing rule — repository-local docs authoritative for implementation, canon authoritative for the cross-repo view — applied to the sprint. A child that needs to move an item's wave or owner raises it at the canon; it does not silently re-scope its own slice.

## What This Decision Does Not Govern

- **Implementation of any gap.** The child PRD/ADR pairs own that. This ADR governs the coordination, not the code.
- **Re-rating severity or re-typing a gap.** The register's adversarial judges fixed those; this context conforms to them (DDD Invariant, Gap-Close context).
- **Reopening ADR-003's distributed-broker finding or the standalone-first federation freeze.** Both are inputs. Federation appears once, as a P2 rescope-or-build fork.

## Alternatives Considered

| Alternative | Verdict | Rationale |
|---|---|---|
| Per-repo ad-hoc fixes with no canon-level register | Rejected | Reproduces the drift ADR-002 exists to catch. Each repository would close its own gaps against its own reading, the shared `did:nostr`-keying blocker (COM-14, which unlocks desktop, MR and voice at once) would be solved three incompatible ways, and no single place would hold the cross-repo score. The register's whole value is that it is one checkable list. |
| One monolithic cross-repo pull request | Rejected | Fights the federated architecture ADR-003 ratified and the repository-local authority ADR-002 protects. A single PR spanning six repositories has no clean review boundary, no per-substrate maturity claim, and no way to land P0 in one repo without blocking on P2 in another. It also concentrates failure: one broken slice stalls the whole sprint. |
| Defer the sprint until after an external pilot | Rejected | Inverts the dependency. Recommendation 12's pilot is a P2 item that depends on the P0 security floor and the P1 governance surfaces existing first — piloting a stack whose ontology `/propose` endpoint is unauthenticated and whose governance plane is admin-only would export the very defects the register found. The pilot is evidence *of* closure, not a precondition for starting it. |

## Consequences

### Positive

- Ecosystem truth stays in one place (the meta-register) without flattening repository ownership — the ADR-002 posture extended to a bounded sprint.
- Every closure is auditable through a receipt and a tier, and the four-surface score is computable at any point in the sprint.
- The shared blockers (`did:nostr` keying, the security floor) are solved once, in the owning repository, and consumed by the rest — not re-solved per surface.
- The liveness-canary gate makes the register's signature failure mode ("built, and unwired") impossible to reproduce as a false closure.

### Tradeoffs

- Six repositories must each author and maintain a child PRD/ADR pair and keep it reconciled at the canon — real coordination overhead, the same tradeoff ADR-002 already accepted for coordinated releases.
- The wave gating serialises some work that could otherwise run in parallel; P1 measurement waits on P0 correctness even where the two do not technically conflict.

### Risks

- **The standing risk the codebase demonstrates on itself: accepted designs sit unbuilt.** ADR-043 specified the four KPIs verbatim with a complete lineage data model in April 2026, and three months later not one line of implementation existed. The self-improving writeback flywheel (ADR-121) and two-speed governance routing (ADR-122) were fully specified and frozen days before the book edition. The codebase carries a ghost flag (`writeback_triggered`) reporting writes it never performs. This ADR's response is structural, not exhortatory: the liveness-canary gate (Decision 3) means an accepted child ADR cannot be scored as closed until its canary fires, so a design accepted and then abandoned registers as *open*, visibly, rather than as a documentation success. If this sprint produces the same outcome, the four-surface score will show it, and the next edition will report it in the terms fixed now.
- **Selection bias.** Auditor and audited are the same organisation. The anti-fox rule and the receipt discipline mitigate but do not eliminate this; ADR-002's conservative-maturity rule remains the backstop.

## Governance Cadence and Exit

- **Wave promotion is ratified at the canon.** A wave is promoted when its canaries are green in the substrates it touched and the compatibility matrix reflects the closed items at their evidenced tier. Promotion cuts a new register version.
- **The sprint's own governance is time-boxed to two quarters**, matching the book's falsification window: if, two quarters from publication, the four surfaces still score as they did in the register, that outcome is published under the terms fixed in the meta-PRD's evaluation lens — the sprint does not quietly extend to avoid reporting a null result.
- **This decision is revisited only on a structural change** — a new substrate joining the mesh, or ADR-002/ADR-003 being superseded. Item-level movement (wave, owner) is a canon-register edit, not an ADR revision.

## Implementation Notes

- Child documents follow the required-section and falsification-statement structure in the meta-PRD's per-repository work packages.
- Register the liveness canaries against the harness owned by VisionClaw's child PRD (RES-a); a loop item with no registered canary cannot enter a wave's closure set.
- Reconcile each child ADR into `docs/architecture/compatibility-matrix.md` at the tier its evidence supports, per ADR-002's release-qualification path.
- Treat the mesh smoke test as the promotion gate from `integrated` to `federation-verified` for any item that claims cross-substrate proof, per ADR-002.
