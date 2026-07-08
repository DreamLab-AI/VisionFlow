# Gap Register v1.1 — P0 Wave Close

**Version:** 1.1 (immutable once published)
**Status:** Published — status register, superseding the v1.0 inventory at the P0→P1 wave boundary
**Date:** 2026-07-08
**Cut by:** `RegisterKeeper` (VisionFlow canon)
**Supersedes:** v1.0 = the meta-PRD consolidated inventory in [`PRD-gap-close-sprint.md`](../PRD-gap-close-sprint.md) §"Consolidated Inventory"
**Governed by:** [ADR-004 Gap-Close Sprint Governance](../ADR-004-gap-close-sprint-governance.md) (§Governance Cadence — "promotion cuts a new register version"), [ADR-005 Gap-Close Canon Decisions](../ADR-005-gap-close-canon-decisions.md), [PRD Gap-Close Canon](../PRD-gap-close-canon.md)
**DDD:** [DDD Gap-Close Canon Context](../DDD-gap-close-canon-context.md) §3 (`GapRegister` aggregate), Invariant 5 (immutable once published)

## What this is

This is the versioned **status register** ADR-004 requires at a wave boundary. It
supersedes the v1.0 inventory **as a status record**, not as a re-scoping: the
meta-PRD (`PRD-gap-close-sprint.md`) stays authoritative for item scope, ownership
and wave assignment (ADR-004 Decision 7). v1.1 records where each **P0** item
stands after the P0 wave closed, forward-chains the corrections that the wave
re-derived, and states the P1 open set.

Per DDD Gap-Close Invariant 5 and ADR-005 §Decision 1, corrections **chain
forward** into this superseding version; `PRD-gap-close-sprint.md` and
`docs/closeout/unified-findings-register.json` are **not** edited in place. All
SHAs below are on `gap-close/2026-07` branches, **committed, not pushed** (local).

## Immutability

Once published, v1.1 is not edited. The next wave boundary (P1 close) cuts v1.2,
which forward-chains the P1 closures and any further re-derived corrections. This
is the first `RegisterKeeper` publish-and-supersede: v1.0 published (the meta-PRD
inventory), v1.1 supersedes it with the P0-close status and the forward-chained
corrections below.

## Correction note — 2026-07-08 (RegisterKeeper precondition)

*Added post-publication as a visible correction (not a silent edit); the "Cut by"
and "Immutability" claims above are left in place as originally written.*

**Defect (adversarially verified).** The claim above that this is the first
`RegisterKeeper` publish-and-supersede — the basis for `RegisterKeeper` reaching
`integrated` (`PRD-gap-close-canon.md` §"Register Stewardship"; `DDD-gap-close-canon-context.md`
§9) — rested on v1.0 (the meta-PRD inventory) having been **published**. Under the
DDD Gap-Close model the `RegisterPublished` event fires only on **"Meta-PRD
inventory ratified"** (`DDD-gap-close-context.md` §6; `DDD-gap-close-canon-context.md`
§6). At v1.1's cut, `PRD-gap-close-sprint.md` still read **Status: Proposed**, so
the inventory was not ratified, the `RegisterPublished` precondition was unmet, and
the `integrated` claim for `RegisterKeeper` was unsupported.

**Resolution.** `PRD-gap-close-sprint.md` and `ADR-004-gap-close-sprint-governance.md`
were ratified (**Status: Proposed → Accepted**, dated 2026-07-08) within the canon's
own authority under ADR-004 — matching reality, the sprint being already in execution
on the `gap-close/2026-07` branches. The `RegisterPublished` trigger now has its
precondition; v1.0 is validly published; and `RegisterKeeper`'s publish-and-supersede
(v1.0 → v1.1) is a supportable basis for the `integrated` tier. No P0 status row,
maturity tier or canary state below is changed by this correction.

## P0 item status (post-wave)

The floor. Maturity uses the ADR-002 vocabulary. **Maturity** is the evidenced
tier of the landed code; **Canary state** is separate — a batch of liveness
canaries was deliberately stacked to one live-session run at sprint end (a
recorded deviation from strict wave gating). A canon **loop** item is not scored
`integrated` until its canary fires in a live session (DDD Invariant 4); RES-b and
RES-d are therefore held at `scaffolded` with their canary state stated honestly.

| Item | Gap(s) subsumed | Owner(s) | Closure SHA(s) | Maturity | Canary | Canary state |
|---|---|---|---|---|---|---|
| RES-a KG-backend liveness + canary harness | — (residual) | VisionClaw | `6f4eb1b0a`, `1492bc17b` | `integrated` | `LivenessHarness` self + `kg_backend_up` gauge; Nostr tap | Harness live; 6 P0 canaries seeded; Nostr-tap relay round-trip **PENDING-LIVE** |
| RES-b diagram-as-code render gate | — (residual) | VisionFlow (canon) | `f72d173cd` | `scaffolded` | `CANARY-CANON-DIAGRAM` | Both halves proven **locally**; has **not** fired in a live GitHub Actions session (box cannot push) — **PENDING-LIVE** |
| COM-13 agent disclosure | F2 | nostr-rust-forum (+ canon norm) | forum `7157a92`, `fb7826859`; canon `identity-spine.md` (this wave) | Forum reference impl `integrated`; canon clause `standalone` | Forum badge live render | Endpoint + `AgentBadge` census-verified (12 files / 15 mounts); live render **PENDING-LIVE** |
| COM-14 `did:nostr` keying of agent nodes | D4, M1 (V1 addressing root) | VisionClaw (consumer) + agentbox (source) | VC `4a595cc8f`; AB `6189f47d` | `integrated` | COM-14 live Schnorr round-trip | Source mint + consumer carry + BIP-340 verify (4 tests) landed; live round-trip **PENDING-LIVE** |
| REC-2 broker kernel + case-queue events | D2, D3 | VisionClaw (agentbox wiring) | `c9f2e3539` (+ docs honesty `b4b78f5e0`, `c65cd8058`) | `integrated` (P0 slice) | broker `case_decided` / case queue | Kernel (965 L) + ACSP events + `ElevationActor` dev-default-on + 42 tests; live case e2e **PENDING** (COM-15 e2e batch) |
| D5 fabricated MCP-status honesty | D5 | VisionClaw | `6f4eb1b0a` | `integrated` | status-honesty | `ControlCenter.tsx` real WS subscription; `mcp_status_label` from `check_mcp_metrics` — hardcoded-`true` dot gone (landed) |
| REC-1 governed-writeback security floor | ontology-auth, `/load` backdoor, PATCH, NIP-42 | VisionClaw · solid-pod-rs · nostr-rust-forum | see **Forward-chained corrections** | `integrated` / `reconciled` | route-guard | REC-1a/1b verify-only (`tests/rec1_route_guard.rs` green); PATCH + NIP-42 reconciled pre-sprint (below) |

**Adjacent source landing.** The agentbox side of RES-d (the script-queryable
skill-count source, `scripts/skill-count-check.js`) landed in the P0 validator
pass at agentbox `d13f8688`. RES-d proper — the canon counter and CI gate — is a
P1 canon item (see the P1 open set).

## Forward-chained corrections

Each row is a v1.0 assertion the P0 wave re-derived. Per Invariant 5 the correction
lands **here**, in the superseding version, chaining forward with its evidence; the
v1.0 wording is not overwritten in place.

| # | v1.0 register wording | Re-derived finding | Disposition | Evidence |
|---|---|---|---|---|
| REC-1a/1b | REC-1: "close governed-writeback security gaps (ontology auth, `/load` backdoor)" | The ontology route-guard and the `/load` backdoor were already closed; the P0 work is **verify-only** — an assertion that the guard holds, not a re-implementation | Closed / reconciled pre-sprint | VisionClaw `6f4eb1b0a`, `tests/rec1_route_guard.rs` |
| PATCH | REC-1 sub-item: "PATCH bug" | The non-destructive-PATCH fix already landed **before** the sprint with 3 passing tests; this is closure-with-receipt against a **stale register entry**, not new work | Closed pre-sprint | solid-pod-rs `791977a`, `patch_non_destructive_integration.rs` (3/3) |
| NIP-42 | REC-1 / F9 sub-item: "NIP-42 reconciliation" | The edge relay gates by **pubkey whitelist** (standalone-first, `auth_required:false`), not NIP-42 AUTH; NIP-42 answering exists only in the WASM browser client. The register's "reconcile NIP-42" wording is superseded by the standalone-first design finding — the edge relay is conformant by design, not deficient | Reconciled pre-sprint | `compatibility-matrix.md` Identity + Mesh rows; ecosystem-map G2 / ADR-003 standalone-first freeze |
| D1 | "Embodiment channel inert; live activity never reaches screen" | The beam **ships** over `/wss/agent-events` (ADR-059); the channel exists and is wired. The residual is **live beam traffic**, not an inert channel — the v1.0 wording is superseded; the gap narrows to a P1 liveness item | Wording superseded; live traffic P1 | ADR-059; `ecosystem-map.md:29` (beam shipped over `/wss/agent-events`); D1 beam-traffic **PENDING-LIVE** canary |
| D5 | "Fabricated 'MCP Connected' status miscalibrates trust" | Status is now sourced from `check_mcp_metrics` and `ControlCenter.tsx` subscribes to a **real** WS; the hardcoded-`true` dot is gone. The v1.0 wording is superseded and the gap is closed | Closed; wording superseded | VisionClaw `6f4eb1b0a` (`ControlCenter.tsx` WS + `mcp_status_label` from `check_mcp_metrics`) |

## P1 open set

The wave the P0 close opens. From the meta-PRD P1 wave (`PRD-gap-close-sprint.md`
§"Wave Sequencing"), carried forward unchanged in scope:

- **REC-3** contextual transaction cost (CTC) instrumentation — VisionClaw · agentbox
- **REC-4** four-KPI control-centre dashboard (ADR-043) — VisionClaw
- **REC-5** MAST-aligned failure telemetry — agentbox · VisionClaw
- **REC-6** escalation-default authority boundaries — agentbox · nostr-rust-forum (overlaps COM-16)
- **REC-7** real outcome learning (feed_retrieval / feed_routing) — agentbox
- **D1** embodiment join — live activity reaches the desktop screen (beam traffic) — VisionClaw
- **D2 / D3** steering / interruption / case surfaces (via REC-2) — VisionClaw
- **D8** swarm-level observability — VisionClaw
- **COM-15 / V1 / D6 / M5** PTT voice-to-selected-actor governed loop — VisionClaw · agentbox
- **COM-16 / F3 / F7** graduated escalation + approval-fatigue response — nostr-rust-forum
- **COM-17 / F4 / F5** decision-audit read API + trust-calibration + optimistic-send fix — nostr-rust-forum
- **F1** member read-only governance view — nostr-rust-forum
- **F8** agent-roster admin UI — nostr-rust-forum
- **RES-c** solid-pod-rs diagram refresh — solid-pod-rs (gated on RES-b render gate)
- **RES-d** self-description drift counter + CI gate — **VisionFlow (canon)** — *landed this wave* (`scaffolded`; `CANARY-CANON-DRIFT` logic proven locally, live CI fire pending)
- **COM-13 canon norm** — the disclosure MUST written into `identity-spine.md` — **VisionFlow (canon)** — *landed this wave* (`standalone`)

Canon P1 progress landed under this register cut: RES-d (drift counter, this
document's companion work) and the COM-13 canon disclosure clause. Both are held at
their honest tiers pending their live canary / second-substrate citation.

## Canary ledger (PENDING-LIVE batch)

Deferred to one stack-up live-session run at sprint end (recorded deviation from
strict wave gating): D1 beam traffic, COM-14 live Schnorr round-trip, COM-15 e2e,
RES-a live relay-tap round-trip, forum badge live render, RES-b CI fire, and — added
this wave — `CANARY-CANON-DRIFT` live CI fire (the drift gate turning a probe PR red
in GitHub Actions). Until a loop item's canary fires live, it is not scored
`integrated` (DDD Invariant 4).

## Compatibility-matrix reflection

`docs/architecture/compatibility-matrix.md` §"Gap-Close Sprint — P0 Item Tiers"
reflects the P0-closed items at their evidenced tiers, per the PRD acceptance
criteria and this register.

## Next version

v1.2 is cut at the P1 wave boundary: it forward-chains the P1 closures (including
RES-d reaching `integrated` on a live `CANARY-CANON-DRIFT` fire and the COM-13 norm
reaching `integrated` on both substrates citing the canon clause) and any further
re-derived corrections.
