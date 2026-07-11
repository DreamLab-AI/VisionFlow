# Gap Register v1.3 — Broker-Governance Forward-Chain Addendum

**Version:** 1.3 (addendum; forward-chains from v1.2, does not edit it in place)
**Status:** Published — broker-canon code-landing addendum, 2026-07-10
**Date:** 2026-07-10
**Cut by:** `RegisterKeeper` (VisionFlow canon)
**Supersedes:** nothing in place — [`gap-register-v1.2.md`](gap-register-v1.2.md) stays
authoritative for every P0/P1/P2 status row, tier and canary state. Per v1.2
§Immutability and §"Next version", corrections **chain forward** into this cut;
v1.2 is not edited.
**Governed by:** [ADR-004 Gap-Close Sprint Governance](../ADR-004-gap-close-sprint-governance.md),
[ADR-003 Judgment Broker Distributed Architecture](../ADR-003-judgment-broker-distributed-architecture.md)
(2026-07-03 closeout amendment); VisionClaw ADR-130 Gap-Close VisionClaw Decisions
(`project/docs/adr/ADR-130-gap-close-visionclaw-decisions.md`, sibling repo — cited, not linked cross-tree)
**Closeout source:** [`docs/closeout/unified-findings-register.json`](../closeout/unified-findings-register.json)
(GOV-1 … GOV-7)

## What this is — and what it is NOT

This addendum records the **broker-governance (GOV-*) code closures landing
2026-07-10** across the sibling repos (agentbox, VisionClaw). It is a *code*
landing addendum, cut against verified trees, not the live-session promotion.

It does **NOT** fire the pending-live batch. Every `pending-live-session` canary
in v1.2 — including the **case e2e** (D3 / REC-2 broker case round-trip, v1.2
:113,145) — **stays `pending-live-session`**. No tier is promoted to `integrated`
from a desk. The stack-up live-session promotion reserved by v1.2 §"Next version"
remains owed and forward-chains (into this version when it fires, or a later
stamp). What a live run of the broker case e2e needs is stated in §"Pending-live
boundary (unchanged)".

Maturity tiers use the ADR-002 seven-tier vocabulary; the **Canary** column is
the liveness state, separate from the code tier (v1.2 §"Maturity vocabulary").
All sibling-repo SHAs are on each repo's `main`/`gap-close` line as of
2026-07-10; agentbox authority-gate work is **committed local + being pushed
today** (agentbox `main` is ahead of `origin/main`).

## Broker-canon truth-pass (docs corrected this cut)

The stale claim that agentbox `relay-consumer.js` "calls nonexistent
`orchestrator.handleGovernanceDecision()`" and that FR1 / DDD §11(1) /
DDD Invariant 7 were *missing* is **falsified and corrected in place**:

- `handleGovernanceDecision()` is real at
  `agentbox/management-api/adapters/orchestrator/local-process-manager.js:133`
  (merged 2026-05-22, `e1a8d716`), invoked from
  `agentbox/mcp/nostr-bridge/relay-consumer.js:318`.
- `PRD-judgment-broker.md` — status header + amendment, TL;DR, agentbox and
  VisionClaw status rows corrected (2 of 3 M1/M2 gaps closed).
- `DDD-judgment-broker-context.md` — §11(1) reframed to *closed*; **Invariant 7
  SATISFIED**; new §14 records the verified kind-31403 three-consumer map.

## GOV-item status this cut

| Item | Disposition (2026-07-10) | Maturity | Canary | Evidence |
|---|---|---|---|---|
| **GOV-1** broker-bridge false write-back closure | **Closed-by-verification.** broker-bridge now keys closure off VisionClaw's `writeback_committed` (not the local verb) and forwards the real deciding `did:nostr` hex (never `'unknown'`); authority gate added on the decide path (ADR-037 D2). Fix present in agentbox commits, being pushed today. | `integrated` (write-back attribution path wired agentbox↔VisionClaw) | pending-live-session (relay round-trip proving committed closure) | agentbox `a70dc4fb` (`management-api/routes/broker-bridge.js` reads `writeback_committed`/`decidingPubkey`) + uncommitted `management-api/lib/governance-decision-waiter.js` |
| **GOV-3** agent-side application unwired → "routes to BrokerActor absent from main" | **Reframed per ADR-130: there is NO distributed `BrokerActor` by design.** Main ships the ADR-110 ElevationActor + inline decide/inbox. The actual residual was the **loop-join** (pending-case producer + REST decision → forum projection), which **landed today**. | `integrated` (loop-join wired) | pending-live-session (case e2e — unchanged from v1.2 :113) | VisionClaw `ca145a1ce` (`elevation_actor.rs`, `enrichment_proposals_handler.rs`, `acsp/events.rs`) |
| **GOV-4** git-bridge `WriteBackSaga` → `/api/ingest/writeback` 404 | **Closed today.** Route registered on VisionClaw main with a route test; git-bridge no longer 404s. | `integrated` (route + handler + test landed) | registered (`tests/gov4_ingest_writeback_route.rs` present; live write-back round-trip pending-live; test-green not re-run in this cut) | VisionClaw `78759494e` (`src/handlers/ingest_writeback_handler.rs`, `handlers/mod.rs`, `main.rs`) |
| **GOV-2** `ConceptElevated` absent; elevation claimed at PR-creation | **Open — honest.** Grep across `project/src`, nostr-rust-forum, agentbox, dreamlab-ai-website = **0 hits**. `elevation_actor.rs` sets `last_pr_url` at PR *creation* (`GitHubPRService`), no merge-watch, no closing event. | `planned` (canon terminus unimplemented) | — (no closure observation exists) | `project/src/actors/elevation_actor.rs` (no `ConceptElevated`, no merge-watch) |
| **GOV-7** ElevationActor skips Whelk EL++ consistency gate | **Open — honest.** No `whelk`/`consistency`/`reasoner` invocation anywhere in `elevation_actor.rs`; approval cases are opened on a drafted axiom without the documented correctness pre-gate. | `historical` (documented gate never built on this path) | — | `project/src/actors/elevation_actor.rs` (frontier-select → `draft_class_page` → open case; no gate) |
| **C4 `ShareOrchestratorActor` executor** (ShareTransitionPlan) | **Open — scaffolded, recorded not omitted.** `broker_decision.rs` *produces* a `ShareTransitionPlan` for `ContributorMeshShare` approves but **has no executor**: the `:202` NOTE states execution "lives in `ShareOrchestratorActor` (agent C4 — follow-up sprint)". `ShareOrchestratorActor` does not exist. | `scaffolded` | — | `project/src/domain/broker/broker_decision.rs:139-167` (`decide()` returns `share_plan`), `:202` NOTE |

Not re-touched this cut (unchanged from closeout): GOV-5 (docs-stale, P2 — the
merged inline decide/inbox is the shipped architecture; ecosystem-map /
compatibility-matrix framing tracked separately), GOV-6 (AgentActionEnvelope
embodiment channel is a distinct live concern, not pending retirement — P3).

## Kind 31403 three-consumer map (verified)

Recorded in full in `DDD-judgment-broker-context.md` §14. Summary:

1. **VisionClaw `ElevationActor`** → GitHub PR — `src/actors/elevation_actor.rs:698`
   (`impl Handler<Decision>`, opens a corpus PR via `GitHubPRService` on approve).
2. **Forum relay decision projector** → D1 `broker_decisions`/`broker_cases` —
   `nostr-rust-forum` `crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs`
   `project_action_response()` via `DecisionOrchestrator`. (Cited by function
   name: the earlier `nip_handlers.rs:1581` recon citation is a kind-1059 gate
   *test* in the pinned checkout, not the consumer, which sits near `:1196`.)
3. **agentbox `handleGovernanceDecision`** → agent stdin / pod —
   `agentbox/management-api/adapters/orchestrator/local-process-manager.js:133`,
   invoked from `mcp/nostr-bridge/relay-consumer.js:318`; mints PROV-O URNs.

## Pending-live boundary (unchanged)

The broker **case e2e** canary stays `pending-live-session`. A live run needs:
`FORUM_RELAY_URL` set to a running relay, `ACSP_PANEL_NOSTR_PRIVKEY` for a panel
key **registered** as an authorised broker pubkey, and a running VisionClaw
server (ElevationActor + broker inbox) alongside the relay. Until that single
stack-up run observes the relay round-trip, GOV-1's committed-closure and GOV-3's
loop-join are `integrated` at code but their loop-closing observation is batched,
exactly as v1.2 records for D3/REC-2.

## Next version

The stack-up live-session promotion reserved by v1.2 §"Next version" is still
owed. When it fires the pending-live batch it forward-chains — promoting the
loop items (including the broker case e2e that clears GOV-1/GOV-3 liveness) to
`integrated` on green canaries, and re-opening any observation that fails to
fire. No status row, tier or canary state in v1.2 or in this v1.3 code-landing
section is edited by that promotion; it appends.

---

## Forward-stamp — 2026-07-11 (post-cut landings + in-container live-loop canary)

Appended per this document's own rule ("it appends"); no row above is edited.

### Code landings after the 2026-07-10 cut

| Item | Disposition (2026-07-11) | Maturity | Canary | Evidence |
|---|---|---|---|---|
| **GOV-2** `ConceptElevated` terminal event | **Closed at code.** kind-31404 `CaseStatusUpdate` builder (`acsp/events.rs`), per-case PR tracking, 120s `GitHubPRService::pr_state` merge poll; merged → `concept_elevated` + store status `elevated`; closed-unmerged → `elevation_abandoned` + `abandoned`; degraded-visible when no GitHub token. Supersedes the `planned` row above. | `integrated` (code) | registered (merge-poll observation not yet fired — needs a real PR merge) | VisionClaw `ada2069a9` + `43a12e401` |
| **GOV-7** Whelk EL++ consistency gate | **Closed at code.** `WhelkInferenceEngine::check_axiom_set` (fresh reasoner, Arc-safe) gates the ElevationActor approve arm; base ∪ draft checked; inconsistent OR gate-unavailable → fail-closed reject with recorded reason, no PR. No advisory pass. | `integrated` (code) | registered (unit canaries: contradictory pair blocks; gate-unavailable blocks) | VisionClaw `43a12e401` (`elevation_actor.rs`, `whelk_inference_engine.rs`) |
| GOV-1 evidence note | The `governance-decision-waiter.js` listed above as "uncommitted" is committed and pushed in agentbox `a70dc4fb` (11/11 broker-bridge + 13/13 authority tests green); agentbox `main` == `origin/main`, tree clean. VisionClaw submodule pointer bumped at `ce8c78524`. | — | — | agentbox `a70dc4fb`; VisionClaw `ce8c78524` |

### In-container live-loop canary (2026-07-11, evidence — NOT the live-session promotion)

A full REST-loop pass was fired against a **live server process + live local relay**
(VisionClaw `43a12e401` debug build, `ELEVATION_ACTOR_ENABLED=1`,
`FORUM_RELAY_URL=ws://127.0.0.1:7777` embedded nostr-pod-bridge relay):

1. ElevationActor started; ACSP decision-projection client connected (boot log).
2. `POST /api/ingest/writeback` (GOV-4 route, git-bridge payload shape) →
   `{success:true, attributed:true, writeback_triggered:true, writeback_committed:true,
   forum_projection:"published"}` with PROV-O activity URN
   `urn:visionclaw:execution:sha256-12-16ffed302021`.
3. `GET /api/broker/inbox` shows the case `canary-case-e2e-001` decided (loop-join read side).
4. kind-31403 event `40aebfb815a2…` **read back from the relay by an independent
   WS subscriber** (`REQ kinds:[31403]` → EVENT match) — the `published` claim is
   verified at the relay layer, not just client-ACKed.

Finding recorded en route: the local pod-bridge "not allow-listed" rejection is the
**pod-ingress consumer** declining pod-inbox ingestion of broadcast events — the
relay itself accepts and serves them. On the **production forum relay**, however,
write gating is real: the ACSP panel pubkey must be registered in the relay
`agent_registry` / allowlist before the stack-up session (operator item; allowlist
work observed in progress in a parallel session, 2026-07-10).

**Boundary honoured:** this run used a local relay and a curl-driven decision, not
the production forum relay + forum-UI human. The **case e2e canary therefore stays
`pending-live-session`**; this stamp contributes fired-in-test-grade evidence that
the loop mechanics (persist → attribute → commit → project → subscribe) are sound.
