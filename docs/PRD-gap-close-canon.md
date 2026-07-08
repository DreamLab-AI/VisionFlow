# PRD: Gap-Close Sprint — VisionFlow Canon Slice

**Owner:** DreamLab AI (VisionFlow canon)
**Status:** Proposed
**Date:** 2026-07-08
**Version:** 1.0
**Parent:** [PRD Gap-Close Sprint](PRD-gap-close-sprint.md) (meta-register)
**Governed by:** [ADR-005 Gap-Close Canon Decisions](ADR-005-gap-close-canon-decisions.md), [ADR-004 Gap-Close Sprint Governance](ADR-004-gap-close-sprint-governance.md), [ADR-002 Ecosystem Alignment Governance](ADR-002-ecosystem-alignment-governance.md)
**Bounded context:** [DDD Gap-Close Canon Context](DDD-gap-close-canon-context.md), conformist to [DDD Gap-Close Context](DDD-gap-close-context.md)
**Book cross-reference:** Chapter "The Gap Register" (14b), "Evaluating the Living Experiment" (14a)

## TL;DR

This is the canon's own work package under the gap-close sprint. The canon owns two register gaps (F10, V2), three residuals (RES-b, RES-d, RES-e), five reconciliations of positions the canon has not yet taken (COM-13 disclosure MUST, COM-18 XR-oversight, D7 intent-legibility, F6 supersession authority, F9 federation rescope), and stewardship of the register itself (`RegisterKeeper`, `WaveGate`). The two gaps the canon owns are both `theory→canon`: a position only the canon can state. The three residuals are the canon's housekeeping debts surfaced by the July 2026 book-production pass. None of this closes another repository's code; it fixes what the canon asserts, counts, renders and adjudicates about the whole.

The through line the meta-register named applies to the canon in its own idiom. Where a substrate fails on a dead wire, the canon fails on a stale or unbacked claim: a skill count that reads three ways in one tree on one day, an ontology class count that reads 5,975 in the book and 7,445 on the website, a diagram pipeline that shipped PDFs with invisible text, a phantom ADR-111 cited as the render gate that this repository does not contain. Closure for the canon means a claim single-sourced and CI-enforced, a render step proven to block on regression, and a position document a substrate can cite. Every loop-closing item here ships a liveness canary, for the same reason the meta-register requires one: an accepted counter script that never runs in CI is the canon's version of "built, and unwired".

## Owned Items

| Item | Type | Severity / class | Wave | Current tier | Target tier |
|---|---|---|---|---|---|
| F10 multi-agent social-influence position | theory→canon | Minor | P2 | `planned` | `standalone` (page authored) → `integrated` (≥2 substrates cite) |
| V2 voice addressing / turn-taking position | theory→canon | Major | P2 | `planned` | `standalone` (page authored) → `integrated` (VisionClaw COM-15 cites) |
| RES-b diagram-as-code render gate | residual | P0 | P0 | `planned` | `integrated` (gate blocks in CI, canary fired) → `released` (pinned in a manifest) |
| RES-d self-description drift counter | residual | P1 | P1 | `planned` | `integrated` (CI fails on injected drift, canary fired) → `released` (pinned in a manifest) |
| RES-e Wardley export quality | residual | P2 | P2 | `scaffolded` (`.owm` sources exist, exports broken) | `integrated` (scripted clean export ≥300 DPI) |
| COM-13 disclosure MUST (canon norm) | canon reconciliation | P0 | P0 | `planned` | `standalone` (MUST written) → `integrated` (forum + VisionClaw cite) |
| COM-18 XR-oversight position | canon reconciliation | P2 | P2 | `planned` | `standalone` → `integrated` (VisionClaw COM-18 cites) |
| D7 intent-legibility position | canon reconciliation | P2 | P2 | `planned` | `standalone` → `integrated` (VisionClaw D7 cites) |
| F6 supersession authority (canon norm) | canon reconciliation | P2 | P2 | `planned` | `standalone` (authority model written) → `integrated` (forum implements + cites) |
| F9 federation rescope fork | canon reconciliation | P2 | P2 | `planned` (federation parked standalone-first) | `standalone` (fork criteria written); federation stays `planned` unless the fork resolves to build |
| `RegisterKeeper` (register stewardship) | service | — | P0 | `planned` | `integrated` (publishes v1, supersedes ≥1) |
| `WaveGate` (wave admission) | service | — | P0 | `planned` | `integrated` (gates ≥1 promotion on green canaries) |

Maturity in this phase is honest about what a document can claim. This is the minting phase: it authors the PRD/ADR/DDD trio and specifies the position pages, the counter and the render gate. It does **not** write `protocol/forum-social-dynamics.md`, `protocol/voice-addressing.md`, the disclosure MUST into `identity-spine.md`, or the supersession subsection into `DDD-judgment-broker-context.md`. Those edits are the work the later waves discharge, so every current tier above stays at `planned`/`scaffolded` until that work lands with its receipt.

## Reconciled Claims

Each row is a canon assertion the sprint touches, its wording today, and the wording the canon adopts. `GapRegister` is immutable once published (DDD Gap-Close Invariant 5), so these corrections chain forward into the next register version; `PRD-gap-close-sprint.md` and `docs/closeout/unified-findings-register.json` are **not** edited in place. Each reconciled claim cites the child document that discharges it, per the parent PRD acceptance criteria (line 149).

| Claim locus | Today | Reconciled to | Discharged by |
|---|---|---|---|
| `PRD-gap-close-sprint.md:100,148`; `docs/closeout/unified-findings-register.json` | "ADR-111 execution outstanding" (diagram render gate) | ADR-111 does not exist in the canon ADR sequence (001–005) and is unverifiable as a VisionClaw doc from this checkout; the render gate is adopted as a canon decision | ADR-005 §Decision 1, §Decision 3 |
| `README.md:130` | "7 MCP Ontology Tools" | 12 tools, matching `compatibility-matrix.md:13,15` and `ecosystem-map.md:30`; the 2026 pre-`edad233` "10" is retired | RES-d counter (this PRD); ADR-005 §Decision 2 |
| `presentation/the-coordination-collapse.md:601,607,620`; `google-analysis.md:561` | "83+ skills" | The single-sourced skill count the counter emits | RES-d counter |
| `docs/engineering/ADR-004-harness-engineering-framework.md:13` | "106 skills" | The single-sourced skill count the counter emits | RES-d counter |
| `README.md:219,395,473,780`; `ecosystem-map.md:30`; `docs/PRD-website.md:122` | "90+ skills" | The single-sourced skill count the counter emits | RES-d counter |
| `website/static/index.html:583`; `website/dist/index.html:535` | "7,445 OWL classes" | The single-sourced ontology class count the counter emits, reconciled against the book's 5,975 / 4,952 `owl:Class` + 9,766 stubs | RES-d counter; `presentation/report/chapters/13e-ontology-binding.tex:16,18` |
| `docs/protocol/identity-spine.md` (descriptive tables, no normative language) | No agent-disclosure MUST | A `did:nostr` self-identification MUST for agents at interaction time | COM-13 reconciliation (this PRD); forum `PRD-gap-close-forum.md` |
| `docs/DDD-judgment-broker-context.md:108` (Invariant 5) | "no undo — only new events that supersede" (no authority model) | A revoke/appeal/supersede authority model naming who may supersede a published `DecisionOutcome` | F6 reconciliation (this PRD); ADR-005 §Decision 5 |
| `docs/architecture/compatibility-matrix.md` | Per-area posture, no post-sprint tier column for gap-close items | Every canon-owned item reflected at its evidenced tier once closed | This PRD, per wave promotion |

## Disclosure Norm (COM-13)

`docs/protocol/identity-spine.md` is descriptive today: it tables the canonical DID form and verification surfaces with zero normative language. COM-13 requires the canon to state the agent-disclosure norm the forum then enforces. This PRD specifies the clause; the forum's `PRD-gap-close-forum.md` owns the badge that renders it. The clause to be written into `identity-spine.md` (P0, not written in this minting phase):

> An agent participating in any governed surface MUST publish its `did:nostr` and be resolvable to the human or organisation that authorised it, at interaction time, before its contribution is acted upon. A surface that renders an agent-authored item without that disclosure is non-conformant.

Acceptance: the clause is present in `identity-spine.md` as a MUST; the forum's disclosure badge (COM-13) cites it as its canon contract; VisionClaw's `did:nostr` keying (COM-14) cites the same clause for headset and desktop surfaces. The norm reaches `integrated` when both cite it.

## Automated Self-Description Counter (RES-d)

Three skill counts and two ontology class counts were live in one tree on one day, plus a third drift axis the recon surfaced: the MCP ontology-bridge tool count reads **7** at `README.md:130` against **12** at `compatibility-matrix.md:13,15` and `ecosystem-map.md:30`. The counter absorbs four axes, each with one machine-queryable source of truth. The canon owns the counting script and the CI gate; the substrates expose the source, per the cross-repo boundary the sprint fixes below.

| Axis | Single source (script-queryable) | Exposed by | Current drift |
|---|---|---|---|
| Skill count | Skill manifest (`counts.json` or equivalent), not prose | agentbox | 90+ / 83+ / 106 |
| Ontology class count | Oxigraph `SELECT (COUNT(DISTINCT ?c) AS ?n) WHERE { ?c a owl:Class }` or a committed count file | VisionClaw | 5,975 / 7,445 |
| MCP ontology-bridge tool count | `mcp/servers/ontology-bridge.js` tool registry length | agentbox | 7 / 12 |
| Agent roster size | Forum `agent_registry` / `dreamlab.toml` roster (distinct from the 610 external turbo-flow templates and from skill count) | nostr-rust-forum | not yet single-sourced |

**Cross-repo boundary (queen).** The canon owns the counter script and the CI gate. agentbox and VisionClaw expose script-queryable count sources; they do not run the gate. The gate greps the canon tree for any prose figure on a tracked axis and fails CI when a figure disagrees with its queried source, or when a second distinct figure for one axis appears anywhere in the tree.

Acceptance: the counter runs as a CI job; an injected second skill/class/tool count on a probe branch turns the build red; the six drifting prose assertions in the reconciled-claims table are replaced by a generated include or badge; `CANARY-CANON-DRIFT` has fired. The counter reaches `released` only when pinned in a release manifest under `docs/releases/`.

## Diagram-as-Code Gate (RES-b)

`mmdc` is confirmed broken locally (`ERR_MODULE_NOT_FOUND: Cannot find package 'puppeteer'`), and published mermaid-derived PDFs shipped with invisible text. Ten `.mmd` sources live under `presentation/report/diagrams/`; no `.github/workflows/*.yml` renders them. The phantom ADR-111 cited as the gate is resolved in ADR-005 §Decision 1: the gate is a canon decision here, not an import.

The gate: a `scripts/render-diagrams.sh` wrapping `mmdc` with a pinned puppeteer/chromium, plus a new `.github/workflows/diagram-render.yml` that fails any PR touching `presentation/report/diagrams/*.mmd` on render error or on a visual-regression diff against a committed baseline. The regression check asserts rendered text nodes carry a non-background fill, catching the invisible-text failure specifically.

Acceptance: a `.mmd` referencing a missing node fails the workflow on a probe branch; a clean `.mmd` renders and passes; no published PDF ships with invisible or unrendered diagram text; `CANARY-CANON-DIAGRAM` has fired. Reaches `released` when pinned in a manifest.

## Position Documents (F10, V2)

Two positions the canon has never taken. This PRD specifies their scope and acceptance; it does **not** write the pages in this minting phase.

- **F10 — `docs/protocol/forum-social-dynamics.md`.** States the canon position on multi-agent social influence in forum threads: how the register treats agent-to-agent amplification, quorum-shaping and disclosure of influence, and what a conformant forum surface must show a human about it. Cited by the forum's `PRD-gap-close-forum.md` and VisionClaw's desktop observation surface. Reaches `integrated` when both cite it.
- **V2 — `docs/protocol/voice-addressing.md`.** Defines the PTT-to-actor addressing and turn-taking model that VisionClaw's COM-15 implementation cites as its canon contract: how a spoken command names a selected actor, how turns are arbitrated across agents, and how the addressed actor is confirmed. `identity-spine.md` is DID-focused and does not cover addressing; this page is its addressing counterpart. Reaches `integrated` when VisionClaw COM-15 cites it.

Both are `theory→canon`: the remedy is a design position, not code. Neither is loop-closing, so neither carries a liveness canary; each carries an acceptance check (the page exists and states a position a named substrate can cite).

## Wardley Export Quality (RES-e)

Five `.owm` sources live under `presentation/report/wardley/`. The exported `wardley-01-coordination-value-chain.png` is 1400x1094 (sub-print) and bakes the onlinewardleymaps tool's own UI chrome into the raster (Export SVG / Export PNG / Toggle Grid buttons), a full-page screenshot rather than a clean export. A clean matplotlib path already exists for the pitch deck (`scripts/generate-wardley-map.py` → 2983x2082).

Acceptance: each of the five `.owm` re-exports via the tool's Export action (not a screenshot) at ≥300 DPI with app chrome cropped; scripted if the tool exposes a CLI or API. One-shot content correction, not a standing wire; acceptance check, no canary.

## Federation Rescope Fork (F9)

Federation is `planned`, parked standalone-first (ecosystem-map G2, `compatibility-matrix.md:11`: federated-by-default tally is 1/4). F9 appears once as a P2 rescope-or-build fork. **Cross-repo boundary (queen):** the canon writes the fork criteria; the forum implements only if the fork resolves to build.

Fork criteria the canon states (ADR-005 §Decision 6): federation is built only when (a) two or more independent relays under distinct operators require cross-relay propagation of governance kinds 31400–31405, and (b) a `MeshTransport` contract is ratified against the IS-Envelope spec, and (c) the standalone-first freeze is explicitly lifted at the canon. Absent all three, F9 stays `planned` and the forum ships nothing. The 38xxx federation kinds remain off the forum allow-list until the fork resolves to build.

Acceptance: the fork criteria are written and testable (a reviewer can decide build-or-park from them); the forum's `PRD-gap-close-forum.md` cites them and keeps F9 at `planned` until they resolve.

## Register Stewardship (RegisterKeeper, WaveGate)

The canon stewards the register itself. Two `planned` services from the Gap-Close DDD become the canon's responsibility.

- **`RegisterKeeper`** publishes and supersedes the immutable `GapRegister`. It cuts a new version at each wave boundary; corrections chain forward and never edit a published register in place. Reaches `integrated` when it has published v1 and superseded it at least once with a forward-chained correction (the ADR-111 and count reconciliations are the first candidates).
- **`WaveGate`** admits items to a wave and opens a wave only when the prior wave's canaries are green in the substrates that wave touched. Reaches `integrated` when it has gated at least one promotion on a green canary set.

## Liveness Canaries

Only loop-closing items — a wire that carries traffic — register a canary. The canon's position documents (F10, V2), the reconciliation norms (COM-13, COM-18, D7, F6, F9) and the one-shot content fix (RES-e) are not loop-closing; they carry acceptance checks, not canaries, and are labelled so. Three canon wires are loop-closing, each a CI gate proven live by a counter-example probe.

| Canary ID | Item | Wire observed | Firing means |
|---|---|---|---|
| `CANARY-CANON-DIAGRAM` | RES-b | `.github/workflows/diagram-render.yml` render + visual-regression job | A `.mmd` with a missing-node reference (or an invisible-text regression) turns the build red on a probe branch, and a clean `.mmd` passes — the render gate blocks, it does not merely run |
| `CANARY-CANON-DRIFT` | RES-d | The drift-counter CI job against the four counted axes | A second distinct skill/class/tool count injected on a probe branch turns the build red — the counter is wired to CI, not a decorative script |
| `CANARY-CANON-CLAIMS` | Reconciliation set | The claims-audit CI job over `compatibility-matrix.md` and the canon tree | A canon page edited to assert a tier above its child's evidenced tier turns the build red, and every reconciled claim carries a child-doc citation — the tier-honesty check is live |

Canaries register against the `LivenessHarness` VisionClaw owns (RES-a); a loop item with no registered canary cannot enter a wave's closure set (ADR-004 Decision 3).

## Package Acceptance Criteria

- Every reconciled claim in the table above cites the child document that discharges it.
- `docs/architecture/compatibility-matrix.md` reflects the post-sprint tier of every canon-owned item.
- The drift counter fails CI on any injected second count across the four axes.
- No published PDF ships with invisible or unrendered diagram text.
- The ADR-111 dangling reference is resolved to a concrete decision, and the register correction is chained forward, not edited in place.
- `RegisterKeeper` has cut and superseded at least one register version; `WaveGate` has gated at least one promotion on green canaries.

## Falsification Statement

*This package is falsified if any canon page still asserts a capability at a tier its owning repository's child document has not evidenced; if a second skill, ontology-class or MCP-ontology-tool count can appear anywhere in the tree without CI failing; if a mermaid-derived PDF ships with invisible text after the gate is declared closed; if "ADR-111" is still cited as an unresolved gate; or if a canon-owned loop item is declared closed without its canary having fired in a live session.*

## Maturity

Canon reconciliation is `integrated` when at least two substrates' child docs cite it. The drift counter and the diagram gate are `released` only when pinned in a release manifest under `docs/releases/`; until then they are `integrated` on a fired canary, or `planned` before the wire exists. F10 and V2 reach `standalone` when their pages are authored and `integrated` on citation; neither is claimed above `planned` in this minting phase, because neither page is written yet. F9 stays `planned` unless its fork resolves to build.
