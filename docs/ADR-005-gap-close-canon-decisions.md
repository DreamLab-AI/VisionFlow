# ADR-005: Gap-Close Canon Decisions

**Status:** Proposed
**Date:** 2026-07-08
**Decision Owners:** DreamLab AI maintainers (VisionFlow canon)
**Related:** [PRD Gap-Close Canon](PRD-gap-close-canon.md), [DDD Gap-Close Canon Context](DDD-gap-close-canon-context.md), [ADR-004 Gap-Close Sprint Governance](ADR-004-gap-close-sprint-governance.md), [ADR-002 Ecosystem Alignment Governance](ADR-002-ecosystem-alignment-governance.md), [ADR-003 Judgment Broker Distributed Architecture](ADR-003-judgment-broker-distributed-architecture.md), [DDD Judgment Broker Context](DDD-judgment-broker-context.md)

**Numbering note.** This is ADR-005 in the top-level canon sequence (`docs/ADR-001…005`). A separate `docs/engineering/` sequence carries its own `ADR-004`/`ADR-005` (harness-engineering, mandate-at-grant); the two counters do not share a namespace. A reader resolving "ADR-005" must check the directory: `docs/ADR-005-*` is this canon decision, `docs/engineering/ADR-005-*` is mandate-at-grant governance.

## Context

The canon's slice of the gap-close sprint forces decisions the meta-PRD deliberately left to the child. Five need a real choice with alternatives weighed, not a restatement of the PRD:

1. The meta-PRD cites "ADR-111 execution outstanding" as the diagram-render gate (`PRD-gap-close-sprint.md:100,148`). ADR-111 does not exist in the canon ADR sequence and cannot be verified as a VisionClaw document from this checkout. Something concrete has to replace the phantom before "ADR-111 execution" can mean anything.
2. RES-d is a drift across three axes at once (skills, ontology classes, MCP ontology-bridge tools), spanning three repositories. Where the counter runs and who exposes the sources is a boundary decision.
3. The diagram gate needs a render mechanism that survives the broken `mmdc`/puppeteer chain and a regression check that catches the specific invisible-text failure that shipped.
4. F6 requires a supersession-authority model in a domain (the judgment broker) whose Invariant 5 says decisions supersede but names no authority.
5. F9's rescope-or-build fork needs criteria a reviewer can decide from, sitting inside the standalone-first freeze ADR-003 and ecosystem-map G2 ratified.

ADR-002 and ADR-003 constrain all five: the canon owns the cross-repo view and the substrates own implementation; the broker is distributed by design; federation is parked standalone-first. None of these decisions reopens those.

## Decision 1 — Resolve ADR-111 by adopting the render gate as a canon decision

The Diagram-as-Code Gate is a canon decision, made in Decision 3 below, not an import of a phantom. The register citation "ADR-111 execution outstanding" is corrected forward: because `GapRegister` is immutable-and-superseded (DDD Gap-Close Invariant 5), the correction is recorded as a Reconciled Claim in `PRD-gap-close-canon.md` and lands in the next register version cut by `RegisterKeeper`. `PRD-gap-close-sprint.md` and `docs/closeout/unified-findings-register.json` are not edited in place. If a VisionClaw ADR-111 is later confirmed to define a compatible gate, this decision cites it as prior art; it does not block on it.

**Alternatives considered.**

| Alternative | Verdict | Rationale |
|---|---|---|
| Import a VisionClaw ADR-111 by reference | Rejected | VisionClaw is not checked out here (no submodule); the reference is unverifiable and would re-import the exact "cited but absent" defect the register exists to catch. |
| Edit `PRD-gap-close-sprint.md` in place to fix the citation | Rejected | Violates DDD Gap-Close Invariant 5 (the register is immutable once published). Corrections chain forward through a superseding version, they do not overwrite the published claim. |
| Mint a standalone `ADR-111` in the canon to honour the number | Rejected | The canon sequence runs 001–005; a 111 would be a fabricated number matching VisionClaw's range, reproducing the cross-repo numbering confusion. The gate belongs in this ADR at its real number. |

## Decision 2 — One canon counter, substrate-exposed sources, four axes

The drift counter is a single script plus a CI gate, both owned by the canon. agentbox and VisionClaw expose script-queryable count sources; they do not run the gate. The counter tracks four axes: skill count (agentbox manifest), ontology class count (VisionClaw Oxigraph query or committed count file), MCP ontology-bridge tool count (agentbox `ontology-bridge.js` registry length), and agent roster size (forum `agent_registry`). The gate fails CI when a prose figure in the canon tree disagrees with its queried source, or when a second distinct figure for one axis appears anywhere in the tree.

The MCP ontology-bridge axis is folded in deliberately. The recon found `README.md:130` reads "7" against "12" at `compatibility-matrix.md:13,15` and `ecosystem-map.md:30`; the earlier "10" was retired by commit `edad233`. A counter that tracks only skills and classes would leave this axis to re-drift; folding it in closes it under the same gate.

**Alternatives considered.**

| Alternative | Verdict | Rationale |
|---|---|---|
| Each substrate counts and asserts its own figures in prose | Rejected | This is the current state that produced the drift. Three prose figures for one number, no single source, nothing enforcing agreement. |
| A shared library each repo imports to print its counts | Rejected | Spreads the counting logic across six repos and six CI configs; a version skew reintroduces drift. One canon script reading exposed sources keeps the logic in one place. |
| Track only skills and ontology classes (the meta-PRD's two named axes) | Rejected | The MCP ontology-bridge tool count is live drift today (7 vs 12); leaving it out reopens RES-d the day the gate lands. Absorbing it costs one query and closes the axis. |
| Canon queries substrate internals directly (grep their source) | Rejected | Couples the canon to each substrate's file layout. A published `counts.json` or query endpoint is a stable contract; internal layout is not. |

## Decision 3 — Diagram render gate: pinned mmdc plus visual-regression baseline

`scripts/render-diagrams.sh` wraps `mmdc` with a pinned puppeteer and a pinned chromium, so the render is reproducible and does not depend on the ambient Nix `mmdc` that throws `ERR_MODULE_NOT_FOUND`. A new `.github/workflows/diagram-render.yml` renders every `presentation/report/diagrams/*.mmd` a PR touches and fails on render error or on a pixel diff against a committed baseline. The regression check additionally asserts rendered text nodes carry a non-background fill, targeting the invisible-text failure that shipped in published PDFs specifically, since a pixel diff alone can miss white-on-white text that matches a white baseline.

**Alternatives considered.**

| Alternative | Verdict | Rationale |
|---|---|---|
| Fix the ambient `mmdc` in the Nix store | Rejected | The store path is content-addressed and shared; patching it is not reproducible and does not travel with the repo. A pinned local puppeteer/chromium under the repo's control is reproducible per PR. |
| Pixel-diff only, no text-fill assertion | Rejected | The failure that shipped was invisible text, which a pixel diff against an equally-invisible baseline passes. The text-fill assertion is the specific counter-example probe for that failure. |
| Render at publish time only, no PR gate | Rejected | Reproduces the register's signature defect: a step that exists but never runs where it would block. The gate must fail a PR, which is what `CANARY-CANON-DIAGRAM` proves. |
| Replace mermaid with a different diagram tool | Rejected | Ten `.mmd` sources and their book references would need rewriting; out of scope for a residual fix and net-new risk beyond the register. |

## Decision 4 — Position documents as protocol pages, standalone-first

F10 and V2 land as `docs/protocol/` pages (`forum-social-dynamics.md`, `voice-addressing.md`) alongside `identity-spine.md`, not as new ADRs. They are `theory→canon` positions, cited by substrates, not decisions with alternatives; the protocol directory is where the canon states what a conformant substrate must honour. Each is authored standalone (a canon position not yet consumed) and reaches `integrated` when a named substrate cites it. This minting phase specifies them; it does not write them.

**Alternatives considered.**

| Alternative | Verdict | Rationale |
|---|---|---|
| Mint F10 and V2 as numbered ADRs | Rejected | They record a position, not a decision between weighed options. The protocol directory already holds this document class (`identity-spine.md`); ADRs would inflate the sequence with non-decisions. |
| Fold both into `identity-spine.md` | Rejected | `identity-spine.md` is DID and verification. Social influence and voice addressing are distinct concerns; folding them dilutes all three and hides the addressing model VisionClaw's COM-15 needs to cite by name. |
| Write the pages now, in the minting phase | Rejected | The parent PRD scopes this phase to documents that specify the work, matching the F10/V2 instruction to specify but not yet author the protocol pages. Writing them now would claim `standalone` before the position is reviewed. |

## Decision 5 — F6 supersession authority in the judgment-broker DDD

The supersession-authority model lands as a new subsection in `DDD-judgment-broker-context.md` §7, extending Invariant 5 ("no undo — only new events that supersede") with the authority it currently omits. The model names who may supersede a published `DecisionOutcome`: the original human signer, or a human holding a governance role above the original at the time of supersession, publishing a new signed kind-31403 that references the superseded event and carries a stated reason. Revoke is supersession by a `Reject` that references a prior `Approve`; appeal is a fresh `ActionRequest` (kind 31402) that cites the superseded decision. The forum owns the decision surface and implements it; the canon states the authority model it must honour. This minting phase specifies the model; it does not edit the broker DDD yet.

**Alternatives considered.**

| Alternative | Verdict | Rationale |
|---|---|---|
| Any authenticated human may supersede any decision | Rejected | Collapses the authority gradient the governance plane exists to hold; an ordinary member could overturn an admin decision. Supersession must respect the role that made the original. |
| A mutable decision record with an edit history | Rejected | Violates Invariant 5 and the broker's Nostr-event-as-audit-trail model. Supersession is a new signed event referencing the old, never a mutation. |
| Define supersession authority in a new canon ADR | Rejected | The authority belongs in the broker context that owns `DecisionOutcome`, not in a parallel canon doc that would fork the model ADR-003 already governs. |

## Decision 6 — F9 fork criteria, federation stays parked until all three hold

Federation is built only when all three hold: (a) two or more independent relays under distinct operators require cross-relay propagation of governance kinds 31400–31405; (b) a `MeshTransport` contract is ratified against the IS-Envelope spec; (c) the standalone-first freeze is explicitly lifted at the canon. Absent any one, F9 stays `planned`, the forum ships nothing, and the 38xxx federation kinds stay off the forum allow-list. The canon writes and owns these criteria; the forum implements only if the fork resolves to build.

**Alternatives considered.**

| Alternative | Verdict | Rationale |
|---|---|---|
| Build federation now as a P2 deliverable | Rejected | Inverts the standalone-first freeze (ecosystem-map G2, ADR-003) the sprint takes as an input, and the federated-by-default tally is 1/4 today (`compatibility-matrix.md:11`). No standing demand justifies the build. |
| Drop F9 entirely | Rejected | The register carries it; dropping it hides a known gap rather than parking it with a stated condition. The fork keeps it visible and decidable. |
| Let the forum decide build-or-park itself | Rejected | Federation is a cross-repo scope question; ADR-004 Decision 7 puts cross-repo scope with the canon. The forum implements, it does not set the fork. |

## What This Decision Does Not Govern

- The implementation of any counter, gate, position page or authority model. This ADR decides the shape; the later waves write the code and the pages, and `PRD-gap-close-canon.md` carries the acceptance criteria.
- Any substrate's maturity claim. The canon reflects a tier once a substrate evidences it; it does not assign it.
- Re-rating any gap's severity or type. Fixed by the register's judges (ADR-004, What This Decision Does Not Govern).
- The dual ADR-numbering namespaces. Flagged here as a housekeeping hazard for a future doc; not resolved in this sprint.

## Consequences

### Positive

- The phantom ADR-111 is replaced by a concrete, checkable gate, and the correction chains forward without violating register immutability.
- One counter closes three drift axes at once, with the counting logic in one place and stable source contracts for the substrates.
- The invisible-text failure has a specific probe (the text-fill assertion), not a generic pixel diff that already passed it once.
- F6 gains an authority model that respects the role gradient, and F9 gains criteria a reviewer can decide from, both without reopening ADR-003.

### Tradeoffs

- The counter depends on three repositories exposing source contracts; a substrate that does not publish its `counts.json` blocks the axis until it does.
- The render gate adds a pinned puppeteer/chromium to CI, a maintenance surface that must track upstream mermaid.
- Parking F9 behind three conditions means a real future federation need waits on the canon lifting the freeze, by design.

### Risks

- The standing risk the meta-register names applies here: an accepted counter or gate that never runs in CI is the canon's "built, and unwired". The liveness canaries (`CANARY-CANON-DRIFT`, `CANARY-CANON-DIAGRAM`, `CANARY-CANON-CLAIMS`) are the structural answer — each proves its wire by a counter-example probe before the item is scored closed. If the gate is declared closed without its canary firing, the falsification statement in `PRD-gap-close-canon.md` catches it.
- Selection bias persists: the canon audits its own claims. The anti-fox rule (ADR-004 Decision 6) puts the closure verifier on a different model family from the producer.

## Governance Cadence and Exit

- `RegisterKeeper` cuts a new register version at each wave boundary; corrections chain forward. The ADR-111 and count reconciliations are the first forward-chained corrections.
- `WaveGate` opens a wave only when the prior wave's canaries are green in the substrates that wave touched. The canon's P0 canaries (`CANARY-CANON-DIAGRAM`) gate the canon's own P0 closure.
- This decision is revisited only on a structural change: a new substrate joining, or ADR-002/ADR-003 being superseded. Item-level movement is a register edit, not an ADR revision.
