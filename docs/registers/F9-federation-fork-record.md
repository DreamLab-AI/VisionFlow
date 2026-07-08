# F9 Federation Rescope — Fork Record

**Item:** F9 federation rescope fork (`canon reconciliation`, P2)
**Status:** Fork evaluated 2026-07-08 — **all three build criteria fail → federation stays `planned`**
**Governed by:** [ADR-005](../ADR-005-gap-close-canon-decisions.md) §Decision 6; [PRD Gap-Close Canon](../PRD-gap-close-canon.md) §"Federation Rescope Fork (F9)"
**Register trail:** Forward-chains into the next `RegisterKeeper` cut (v1.2, P2 boundary). Does **not** edit `gap-register-v1.1.md` in place (Invariant 5) or `PRD-gap-close-sprint.md`.

## What this is

ADR-005 §Decision 6 made F9 a **rescope-or-build fork**: federation is built only
when all three stated criteria hold; absent any one, F9 stays `planned`, the forum
ships nothing, and the 38xxx federation kinds stay off the forum allow-list. This
record evaluates the three criteria against the current truth of the tree and
records the decision a reviewer can re-derive from the evidence. The fork resolves
to **park**: 0 of 3 criteria hold.

## Criteria evaluation

| # | Criterion (ADR-005 §Decision 6) | Holds? | Current truth | Evidence |
|---|---|---|---|---|
| (a) | Two or more independent relays under **distinct operators** require cross-relay propagation of governance kinds 31400–31405 | **NO** | The mesh is single-operator. Only `dreamlab-ai-website` fans out; the federated-by-default tally is **1/4** (agentbox, solid-pod-rs and nostr-rust-forum all default standalone). `MESH_FEDERATED_KINDS` is one forum relay-worker's allow-list; there is no second independent-operator relay demanding cross-relay 31400–31405 propagation | `docs/architecture/compatibility-matrix.md:11` (tally 1/4); `docs/ecosystem-map.md:117` (three of four default standalone); `docs/protocol/event-kind-registry.md:84,88` (single relay allow-list) |
| (b) | A **`MeshTransport` contract is ratified** against the IS-Envelope spec | **NO** | `nostr-bbs-mesh` is scaffold-only with **no `MeshTransport` implementation**, and is not even a dependency of the relay-worker (lands Sprint v12+). The IS-Envelope spec is owned and fixture-tested (VisionClaw ADR-075, 11 vectors), but **IS-Envelope routing is unimplemented in every substrate — only conformance vectors exist.** A spec plus vectors is not a ratified transport contract | `docs/architecture/compatibility-matrix.md:11` (`nostr-bbs-mesh` scaffold-only, no `MeshTransport`); `docs/ecosystem-map.md:119` (IS-Envelope routing unimplemented in every substrate); `docs/ADR-002-ecosystem-alignment-governance.md:47` (IS-Envelope = spec + schema + vectors, VisionClaw ADR-075) |
| (c) | The **standalone-first freeze is explicitly lifted** at the canon | **NO** | Ecosystem-map **G2 is FROZEN**: standalone-first is declared the supported deployment mode; mesh federation (ADR-073/PRD-010, IS-Envelope routing) is *designed, not shipped* and parked. The freeze is in force, not lifted; ADR-005 does not reopen ADR-003 | `docs/ecosystem-map.md:115` (G2 "…Standalone-First Is the Supported Mode (FROZEN)"), `:121` (closeout FREEZE decision); `docs/ADR-005-gap-close-canon-decisions.md:88` (freeze taken as input, not reopened) |

## Decision

**0 of 3 criteria hold. The fork resolves to park. F9 stays `planned`.**

Consequences, per ADR-005 §Decision 6:

- Federation is **not** built this sprint. The forum implements nothing for F9
  (cross-repo boundary: the canon writes the criteria; the forum builds only if the
  fork resolves to build — it has not).
- The **38xxx federation kinds stay off the forum allow-list.** `MESH_FEDERATED_KINDS`
  is not widened for F9. (The separate, owner-policy decision to federate the
  agentbox 383xx marketplace kinds documented in `event-kind-registry.md §3` is
  orthogonal to F9 and likewise stays open by deliberate deferral, not resolved
  here.)
- F9 remains **visible and decidable**: this record keeps the gap on the register
  with a stated re-evaluation condition rather than dropping it. The fork is
  re-run when any of (a), (b), (c) changes — a second independent-operator relay
  standing up and demanding cross-relay governance propagation, a `MeshTransport`
  contract ratified against IS-Envelope, or the canon explicitly lifting the G2
  freeze.

## Register-trail placement

Per DDD Gap-Close Invariant 5, this decision **chains forward**: it is recorded here
as an immutable fork record and forward-chains into the next `RegisterKeeper`
version (v1.2, cut at the P2 wave boundary), which will carry F9 at `planned` with
this record as its evidence. It does **not** edit `gap-register-v1.1.md`,
`PRD-gap-close-sprint.md` or `docs/closeout/unified-findings-register.json` in place.

## Acceptance mapping (PRD §"Federation Rescope Fork (F9)")

| Criterion | State |
|---|---|
| The fork criteria are written and testable (a reviewer can decide build-or-park) | Done — ADR-005 §Decision 6; each of the three predicates evaluated above with file:line evidence |
| A reviewer can decide build-or-park from them | Done — 0/3 hold → park; the evidence is re-checkable against the cited files |
| The forum cites them and keeps F9 at `planned` until they resolve | Cross-repo: the forum's `PRD-gap-close-forum.md` cites this record; the canon's disposition is `planned` |

## Falsification exposure

This record is falsified if any of the three criteria in fact holds against the
cited evidence (a second independent-operator relay demanding cross-relay 31400–31405,
a ratified `MeshTransport` contract, or a lifted G2 freeze) and F9 is nonetheless
kept parked; if F9 is claimed built or the 38xxx kinds added to `MESH_FEDERATED_KINDS`
while the fork reads park; or if this decision is written by editing the published
v1.1 register in place rather than chaining forward.
