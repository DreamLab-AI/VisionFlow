# Gap Register v1.2 — Sprint Close (P1 + P2 Waves)

**Version:** 1.2 (immutable once published)
**Status:** Published — status register, superseding v1.1 at the P1/P2 wave boundary; cuts the sprint
**Date:** 2026-07-08
**Cut by:** `RegisterKeeper` (VisionFlow canon)
**Supersedes:** [`gap-register-v1.1.md`](gap-register-v1.1.md) (P0-close status); which supersedes v1.0 = the meta-PRD consolidated inventory in [`PRD-gap-close-sprint.md`](../PRD-gap-close-sprint.md) §"Consolidated Gap Inventory"
**Governed by:** [ADR-004 Gap-Close Sprint Governance](../ADR-004-gap-close-sprint-governance.md) (§Governance Cadence — "promotion cuts a new register version"), [ADR-005 Gap-Close Canon Decisions](../ADR-005-gap-close-canon-decisions.md), [PRD Gap-Close Canon](../PRD-gap-close-canon.md)
**DDD:** [DDD Gap-Close Canon Context](../DDD-gap-close-canon-context.md) §3 (`GapRegister` aggregate), Invariant 5 (immutable once published), Invariant 4 (a loop item is not `integrated` until its canary fires in a live session)

## What this is

This is the wave-boundary status register ADR-004 requires when the P1 and P2
waves close. It records where every one of the 51 line items stands at sprint
end, forward-chains the corrections the two waves re-derived, and states the
pending-live-session boundary honestly. It supersedes v1.1 **as a status
record**, not as a re-scoping: `PRD-gap-close-sprint.md` stays authoritative for
item scope, ownership and wave assignment (ADR-004 Decision 7).

Per DDD Gap-Close Invariant 5 and ADR-005 §Decision 1, corrections **chain
forward** into this superseding version; v1.1, `PRD-gap-close-sprint.md` and
`docs/closeout/unified-findings-register.json` are **not** edited in place. All
SHAs below are on the `gap-close/2026-07` branches (per repo), **committed, not
pushed** (local). The Godot XR chain was built on the `gap-close/2026-07-xr`
worktree branch and merged back at `485bb8f9f`; that worktree is removed and the
branch deleted post-merge (five-of-five harness tests green after a union-resolve
of the canary-seed and `app_state` conflicts).

## Immutability

Once published, v1.2 is not edited. It closes the sprint's three waves. Any
residual re-derivation (for instance, the pending-live canaries firing in the
stack-up session) chains forward into a v1.3 addendum cut at that session; it does
not edit v1.2 in place. Publish-and-supersede lineage: v1.0 (meta-PRD inventory,
ratified) → v1.1 (P0 close) → v1.2 (P1 + P2 close, this document).

## Maturity vocabulary

Tiers use the ADR-002 seven-tier vocabulary: `historical`, `planned`,
`scaffolded`, `standalone`, `integrated`, `federation-verified`, `released`. The
**Maturity** column below is the honest post-fixup tier of the landed code: the
tier its evidence supports after the wave's adversarial verifiers forced the
corrections recorded in §"Forward-chained corrections". It is separate from the
**Canary** column, which is the liveness state:

- **fired-in-test** — the canary logic is exercised by a passing local test (the mapper, the gate, the resolver proven against fixtures) but has not fired against a live stack.
- **registered** — the canary is seeded in the `LivenessHarness` (or the CI gate is wired) and will fire in the stack-up session.
- **pending-live-session** — the loop-closing observation itself (relay round-trip, beam traffic, GitHub Actions gate turning a probe red, headset session) awaits the one stack-up run described in §"Pending-live-session batch".

A canon **loop** item is not scored `integrated` until its canary fires in a live
session (DDD Invariant 4). Where the code and tests land but the loop-closing
observation is batched, the item is tiered on its evidenced code (`integrated` for
a wired cross-substrate path, `scaffolded` for a canon-loop item whose only
closure proof is the pending canary) and its pending observation is stated in the
Canary column, not folded into the tier.

## Register-drift: items already closed before the sprint

The v1.0 inventory audited an older tree. A pre-sprint reconnaissance
(`wf_44dd4e9e`, 2026-07-08) falsified four register/meta-PRD closure claims by
opening the current code: the work was **already done on `main`**, so the sprint's
task for these was closure-with-receipt against a stale entry, not new
engineering. All four were recorded in v1.1's forward-chain; they are restated
here as the register-drift set so the sprint's true net-new footprint is legible.

| Drift item | v1.0 wording | Reality on `main` | Evidence |
|---|---|---|---|
| REC-1a ontology `/propose` auth | "authenticate the unauthenticated `/propose` endpoint" | Already gated: `ontology_agent_handler.rs:344-362` `RequireAuth::authenticated()` + `RateLimit`, wired `main.rs:928` | VisionClaw `main`; verify-only `tests/rec1_route_guard.rs` |
| REC-1b `/load` backdoor | "remove or hard-gate the `/load` validation bypass" | Already single-scoped: `api_handler/ontology/mod.rs:1339-1401` `RequireAuth::power_user().mutations_only()`, weaker duplicate scope deleted | VisionClaw `main` |
| PATCH non-destructive bug | "fix the solid-pod-rs non-destructive PATCH bug" | Fixed pre-sprint 2026-05-29 (`seed_graph_from_patch_target`), 3/3 regression tests | solid-pod-rs `791977a`, `patch_non_destructive_integration.rs` |
| NIP-42 reconciliation | "perform the AUTH round-trip or reconcile the claim" | Deliberately reconciled: edge relay advertises `auth_required:false` and gates by pubkey whitelist (`nip11.rs:126-172` + `nostr_bbs.write_policy` block); NIP-42 answering exists only in the WASM browser client — a canon wording fix, not a deficiency | `compatibility-matrix.md` Identity/Mesh rows; ecosystem-map G2 standalone-first freeze |

Two further v1.0 framings were re-derived as **wrong**, not as work: **D1** was
described as an inert embodiment channel, but `AppInitializer.tsx:356` awaits
`websocketService.connect()` at boot and `useAgentNodes` polls `/api/bots/agents`
over one multiplexed WS (there is no separate `/wss/agent-events` client), so D1's
residual is live beam **traffic**, a canary, not code. **D5**'s "MCP Connected"
badge is honest (a real `/bots/status` poll); the fabricated indicator was the
separate WS dot (`ControlCenter.tsx:139` passed a literal `websocketStatus="connected"`),
which is the locus the P0 fix corrected. Both framings are superseded here, not
edited into v1.0.

## Full item status: the 51 line items

Grouped as the meta-PRD counts them: 28 register gaps + 18 commitments + 5
residuals = 51. Six commitments (COM-13 to COM-18) subsume register gaps; the
subsumed gap carries its status from the commitment that owns it and is
cross-referenced. **Final status** is the sprint-end disposition; **Maturity** is
the honest post-fixup tier; **Canary** is the liveness state; **SHA(s)** are the
owning commits.

### Register gaps: Forum (F1–F10)

| Item | Final status | Maturity | Canary | SHA(s) |
|---|---|---|---|---|
| F1 governance plane admin-only | Closed (P1) | `integrated` | fired-in-test (member read-only `/governance` route-split) | forum `0b8a1c`, `6986276` |
| F2 no agent disclosure | Closed via COM-13 (P0) | forum impl `integrated`; canon clause `standalone` | pending-live-session (badge live render) | forum `7157a92`, `fb7826859`; canon `identity-spine.md` |
| F3 decisions binary; escalation dead code | Closed via COM-16 (P1) | `integrated` | fired-in-test (`DecisionOrchestrator`→31403 delegate/promote/precedent) | forum `0b8a1c`, `6986276` |
| F4 optimistic-only; rejected reads as sent | Closed via COM-17 (P1) | `integrated` | fired-in-test (`publish_with_ack` both sites) | forum `0b8a1c`, `6986276` |
| F5 no trust-calibration; no decision-audit API | Closed via COM-17 (P1) | `integrated` | fired-in-test (`GET /api/governance/decisions`; reasoning/confidence/risk_tier schema) | forum `0b8a1c`, `6986276` |
| F6 no revoke/appeal/supersede authority | Closed (P2) | `scaffolded` (canon-gated) | fired-in-test (authorised-signer-or-higher supersede via new signed 31403; projection marks Superseded) | forum `35dbb1b`; tier fixup `696fd233`; canon spec `7513e58` (DDD-judgment-broker) |
| F7 no approval-fatigue response | Closed via COM-16 (P1) | `integrated` | fired-in-test (risk-tier Memo suppression of low-risk panels) | forum `0b8a1c`, `6986276` |
| F8 no roster admin UI | Closed (P1) | `integrated` | fired-in-test (Agents roster admin tab; nine existing endpoints) | forum `0b8a1c`, `6986276` |
| F9 cross-relay federation absent | Rescoped; stays parked (P2) | `planned` | n/a (fork record; criteria unmet) | canon `7513e58` (F9 fork record, standalone-first freeze) |
| F10 silent on multi-agent social influence | Closed (P2 canon position) | `standalone` | n/a (theory→canon position authored) | canon `7513e58` (`forum-social-dynamics.md`) |

### Register gaps: Desktop (D1–D8)

| Item | Final status | Maturity | Canary | SHA(s) |
|---|---|---|---|---|
| D1 embodiment channel "inert" | Closed; v1.0 framing superseded (P1) | `integrated` (transport + poll wired at boot; register-drift correction) | pending-live-session (beam traffic) | VisionClaw `4aca6f729` (liveness instrument); drift note above |
| D2 no steering/interruption affordance | Closed with documented boundary (P1→P2) | `integrated` | fired-in-test (`AgentDetailPanel` on node-select; `/bots/submit-task`+interrupt+task-status; `InterruptAgentTask` resolver) | VisionClaw `e0f582403`, `453bd41b1`, `b6c1c43b5` |
| D3 no client ACSP case surface | Closed via REC-2 (P1) | `integrated` | pending-live-session (case e2e, COM-15 batch) | VisionClaw `e0f582403` (broker case queue UI vs `/api/broker/inbox`; ambient ACSP indicator) |
| D4 nodes keyed by `task_id` not `did:nostr` | Closed via COM-14 (P0) | `integrated` | pending-live-session (Schnorr round-trip) | VisionClaw `4a595cc8f`; agentbox `6189f47d` |
| D5 fabricated "MCP Connected" status | Closed (P0) | `integrated` | fired-in-test (real WS subscription; `mcp_status_label` from `check_mcp_metrics`) | VisionClaw `6f4eb1b0a` |
| D6 PTT globally scoped | Closed via COM-15 (P1) | `integrated` | pending-live-session (voice e2e) | VisionClaw `f6e1d58f9`; agentbox `9673624`, `1fc47a14` |
| D7 no pre-action intent legibility | Closed (P2) | `standalone` | fired-in-test (envelope intent field + panel display; canon page) | VisionClaw `774ffa05e`; canon `2511f994` (`intent-legibility.md`) |
| D8 no swarm-level observability | Closed (P1) | `integrated` | fired-in-test (`SwarmObservabilityPanel`, MAST counts + canary status) | VisionClaw `e0f582403` |

### Register gaps: Mixed reality (M1–M6)

| Item | Final status | Maturity | Canary | SHA(s) |
|---|---|---|---|---|
| M1 identity-blind in headset | Closed (P2) | `standalone` (Godot DID badge; no headset in box) | pending-live-session (M1-HUD rides P0 COM-14) | XR `57b32faee`, `0f3a1b60c`, `348172d7d` (merged `485bb8f9f`) |
| M2 no interaction/governance affordance in MR | Closed via COM-18 (P2) | `standalone` | pending-live-session (COM18-INTERV standing) | XR `0f3a1b60c` (HUD intervention panel POSTs signed decide) |
| M3 copresence theory unrepresented | Closed (P2) | `standalone` (explicit — no headset) | fired-in-test (gaze/proxemics/avatar_state property-tested) | XR `57b32faee` (`gaze.rs`, `proxemics.rs`, `avatar_state.rs`) |
| M4 input layer broken (world-origin ray, no gaze) | Closed (P2) | `standalone` | pending-live-session (M4-RAY one-shot) | XR `57b32faee` (`selection.rs` 3-resolver arbiter) |
| M5 voice PTT-to-actor binding, zero consumers | Closed via COM-15 (P1) | `integrated` (binding now consumed) | pending-live-session (voice e2e) | VisionClaw `f6e1d58f9` |
| M6 `enterVR()` never sets `isXRMode` | Closed (P2) | `standalone` | pending-live-session (eye-gaze guard #113717; on-device) | XR `0f3a1b60c` |

### Register gaps: Voice (V1–V4)

| Item | Final status | Maturity | Canary | SHA(s) |
|---|---|---|---|---|
| V1 canonical PTT loop absent | Closed via COM-15 (P1) | `integrated` | pending-live-session (voice e2e) | VisionClaw `f6e1d58f9`; agentbox `9673624`, `1fc47a14` |
| V2 no multi-agent addressing/turn-taking | Closed (P2 canon position) | `standalone` | n/a (theory→canon position authored) | canon `7513e58` (`voice-addressing.md`) |
| V3 no conversational grounding/repair | Closed (P2) | `standalone` | fired-in-test (`voice_clarification.rs` confidence-gate kernel, 492 L) | VisionClaw `774ffa05e` |
| V4 docs describe deprecated voice-to-swarm as live | Closed (P2) | `integrated` (docs-honesty) | n/a (documentation fix) | VisionClaw `774ffa05e` |

### Commitments (REC-1 … REC-12, COM-13 … COM-18)

| Item | Final status | Maturity | Canary | SHA(s) |
|---|---|---|---|---|
| REC-1 governed-writeback security floor | Closed / reconciled (P0) | `integrated` / `reconciled` | fired-in-test (`tests/rec1_route_guard.rs` green); PATCH + NIP-42 reconciled pre-sprint (drift set) | VisionClaw `6f4eb1b0a`; solid-pod `791977a` |
| REC-2 BrokerActor + case queue | Closed; v1.0 "Implemented" claim corrected (P0+P1) | `integrated` (P0 kernel + P1 case queue) | pending-live-session (case e2e) | VisionClaw `c9f2e3539` (965 L kernel), `e0f582403`; docs honesty `b4b78f5e0`, `c65cd8058` |
| REC-3 contextual transaction cost | Closed (P1) | `integrated` | fired-in-test (CTC envelope + agentbox emitter; serde aligned) | VisionClaw `4aca6f729`, `453bd41b1`; agentbox `9673624`, `ceb3401b` |
| REC-4 four-KPI dashboard (ADR-043) | Closed (P1) | `integrated` (2 KPIs computed; 2 honest "awaiting data source" tiles) | fired-in-test (`GET /api/kpi/summary`; Augmentation Ratio + Trust Variance live) | VisionClaw `4aca6f729` |
| REC-5 MAST-aligned failure telemetry | Closed (P1) | `integrated` | fired-in-test (14-mode `lib/failure-taxonomy.js`; agent-events tagged) | agentbox `9673624`, `1fc47a14`, `ceb3401b` |
| REC-6 escalation-default authority | Closed; tier corrected (P1) | `standalone` (re-tiered honestly in fixup) | fired-in-test (`buildAuthorityGate` block-on-31402 / release-on-31403 in `POST /v1/llm/revoke`; `authority_class` on envelope) | agentbox `9673624`, `1fc47a14`, `ceb3401b` |
| REC-7 real outcome learning | Closed; consumers gated pending floor (P1) | `standalone` (learning runs; does not yet influence a live second consumer) | fired-in-test (Wilson floor; `feed_retrieval`/`feed_routing` gated; stale-comment purge) | agentbox `9673624`, `1fc47a14` |
| REC-8 engineered orchestration diversity | Closed; tier corrected (P2) | `standalone` (re-tiered in fixup) | fired-in-test (cross-family anti-fox consultant wrapper, 19 tests) | agentbox `9ebff750`, `a8dd21a5` |
| REC-9 provenance-to-pocket | Closed; id-param fixed (P2) | `standalone` (re-tiered in fixup) | fired-in-test (urn-in-mirror-DM; `/v1/agent-events?id=` resolves urns with 404 branch) | agentbox `9ebff750`, `a8dd21a5` |
| REC-10 Insight Ingestion Loop v1 | Closed (P2) | `integrated` | fired-in-test (`ontology_propose`→broker→merged-enrichment monotonic timestamps; Mesh Velocity computable; trace endpoint) | VisionClaw `1c462f492` |
| REC-11 data-moat consolidation | Closed; one source contract-only (P2) | `standalone` (unified `/api/trace` joins agent-events + broker + pod-marks, absent-source tolerant; pod `_prov/` endpoint contract-defined) | fired-in-test | VisionClaw `1c462f492`; solid-pod `40043b0` (+WP-2) |
| REC-12 external pilot / kit cutover | Closed; pilot blocked-operational (P2) | `standalone` (kit cutover done; external pilot staged as runbook, no real keys) | registered (D3 key-split runbook, blocked-operational) | website `ca06ab3` |
| COM-13 agent disclosure (subsumes F2) | Closed (P0) | forum impl `integrated`; canon clause `standalone` | pending-live-session (badge live render) | forum `7157a92`, `fb7826859`; canon `identity-spine.md` (@ `8aabb4c`) |
| COM-14 `did:nostr` keying (subsumes D4, M1, V1 root) | Closed (P0) | `integrated` | pending-live-session (Schnorr round-trip) | VisionClaw `4a595cc8f`; agentbox `6189f47d` |
| COM-15 PTT voice loop (subsumes V1, D6, M5) | Closed (P1) | `integrated` | pending-live-session (voice e2e) | VisionClaw `f6e1d58f9`; agentbox `9673624`, `1fc47a14` |
| COM-16 graduated escalation + approval-fatigue (subsumes F3, F7) | Closed (P1) | `integrated` | fired-in-test | forum `0b8a1c`, `6986276` |
| COM-17 decision-audit API + trust-calibration (subsumes F4, F5) | Closed (P1) | `integrated` | fired-in-test | forum `0b8a1c`, `6986276` |
| COM-18 headset intervention + verifiable identity (subsumes M2, M4, M6) | Closed (P2) | `standalone` (no headset in box) | pending-live-session (COM18-INTERV standing) | XR `57b32faee`, `0f3a1b60c`, `348172d7d` |

### Residuals (RES-a … RES-e)

| Item | Final status | Maturity | Canary | SHA(s) |
|---|---|---|---|---|
| RES-a KG-backend liveness + canary harness | Closed (P0) | `integrated` | pending-live-session (Nostr relay-tap round-trip); harness live, 6 P0 canaries seeded | VisionClaw `6f4eb1b0a`, `1492bc17b` |
| RES-b diagram-as-code render gate | Closed at mechanism; live gate pending (P0) | `scaffolded` | pending-live-session (`CANARY-CANON-DIAGRAM` GitHub Actions fire; box cannot push) | canon `f72d173cd` |
| RES-c solid-pod-rs diagram refresh | Closed (P1) | `integrated` (9 diagrams regenerated via canon renderer, visible text) | fired-in-test (gated on RES-b renderer, proven locally) | solid-pod `40043b0` (+WP-2) |
| RES-d self-description drift counter + CI gate | Closed at mechanism; live gate pending (P1) | `scaffolded` | pending-live-session (`CANARY-CANON-DRIFT` GitHub Actions fire on a probe PR) | canon `b47c5fd` (`drift-counter.mjs`); sources VisionClaw `4aca6f729`, agentbox `d13f8688` |
| RES-e Wardley export quality | Closed (P2) | `integrated` | n/a (5 Wardley maps re-exported clean, no UI chrome) | canon `7513e58` |

**Tally.** 51/51 items dispositioned. 50 closed (28 gaps, 17 commitments, 5
residuals); 1 (F9 federation) deliberately parked `planned` per the standalone-first
fork. Honest tiers: `integrated` × 27, `standalone` × 15, `scaffolded` × 3
(RES-b, RES-d, F6 canon-gated), `planned` × 1 (F9); plus the two split-tier items
(F2/COM-13 forum `integrated` + canon `standalone`) counted at their forum tier.
No item is scored above the tier its evidence supports.

## Forward-chained corrections (v1.1 → v1.2)

Each row is an assertion the P1 or P2 wave re-derived. Per Invariant 5 the
correction lands **here**, chaining forward with its evidence; the prior wording
is not overwritten. v1.1's forward-chain (REC-1a/1b, PATCH, NIP-42, D1, D5) stands
and is not repeated; these are the new links.

| # | Prior wording | Re-derived finding | Disposition | Evidence |
|---|---|---|---|---|
| REC-2 / ADR-041 | ADR-041 stated the `BrokerActor` was **"Implemented"** on `main` | Reconnaissance found **no** `BrokerActor` on `main` — only on the orphan `crashbug` branch; `main` ships the ADR-110 stateless ACSP producer with the forum owning the queue. The claim was false. The P0 wave cherry-picked the 965 L domain kernel onto `gap-close/2026-07` and corrected the lying docs (CHANGELOG, ADR-033, ecosystem-convergence.md, rest-api.md) | Closure real; doc claim corrected | VisionClaw `c9f2e3539`; docs honesty `b4b78f5e0`, `c65cd8058` |
| D2 interrupt (MCP-native) | P1 declared D2 steering/interrupt closed | Re-verification refuted it: the interrupt join key was **still missing** for MCP-native agents — the MCP surface exposes no terminate verb and the Management API carries only role labels, so a subset of agents could not be interrupted. Folded into the P2 stage-0 fix: a `claude_flow_agent_id` join key in both repos, plus a client disclosure state marking non-interruptible external agents rather than pretending they can be stopped | Refuted then fixed; honest boundary documented | VisionClaw `b6c1c43b5` |
| REC-6 tier | P1 first-pass tiered REC-6 above its evidence | Escalation-default authority runs standalone in agentbox; it does not yet demonstrate a live blocked-then-released round-trip across substrates. Re-tiered `standalone` honestly | Tier corrected | agentbox `ceb3401b` |
| REC-8 / REC-9 tier | P2 first-pass over-tiered the diversity wrapper and provenance-to-pocket | Both operate within agentbox and were not yet cross-substrate-proven; REC-9's `/v1/agent-events?id=` did not resolve urns (404 path missing). Fixed the id-param and re-tiered both `standalone` | Tier corrected; bug fixed | agentbox `a8dd21a5` |
| F6 tier | P2 first-pass tiered F6 supersession `integrated` | F6 is canon-gated (the supersession authority is specified in DDD-judgment-broker but the forum impl is conformist to a canon clause not yet second-sourced). Re-tiered `scaffolded (canon-gated)` per its own compound bar | Tier corrected | forum `696fd233` |
| D7 canon page | P2 first-pass claimed D7 closed | The `intent-legibility.md` canon page was **missing** from the first pass — only the envelope field and panel display had landed. Authored in fixup, so the theory→canon half is discharged | Refuted then fixed | canon `2511f994` |

## Adversarial verification: false closures caught per wave

The anti-fox discipline (a refute-mandate verifier on a different model family
from the implementer, ADR-004 Quality Gate 3) ran on every wave. It caught false
closures in all three:

- **P0:** three adversarial catches: two false closures fixed and re-confirmed (COM-13 badge coverage under-counted the author-render sites; the REC-2 `ServerNostrActor` doc sweep missed live-voice regex hits), and one fabricated verifier claim caught and discarded (an RES-b report asserted a `puppeteer-core` removal that was never committed anywhere; the repo evidence was honest, so the claim was set aside, not actioned).
- **P1:** six false closures caught first-pass (7 CONFIRMED / 13 PARTIAL-honest / 6 REFUTED); all six fixed, then re-verified (4 CONFIRMED / 2 PARTIAL-endorsing / 1 REFUTED; the surviving refutation was the D2 MCP-native interrupt join key, folded into P2 stage-0 above).
- **P2:** three real false closures caught and fixed (D7 canon page missing; REC-9 id-param; REC-8/REC-9/F6 tier overclaims), of five first-pass refutations; the other two were stale-audit artefacts against tiers the implementers had already self-corrected in the same wave. Re-verification all green.

The meta-PRD and ADR-004 were ratified (Status: Proposed → Accepted, 2026-07-08)
citing the caught false closures as acceptance evidence: the sprint's own auditor
catching its own audited is the honesty discipline made operational, and the
receipt that the gate has teeth.

## Pending-live-session batch

Strict wave gating (a wave's canaries green before the next starts, ADR-004
§Governance Cadence) was **deliberately relaxed** and that deviation is recorded
here, because no live stack runs in the build container: the Docker socket is
mounted but source bind-mounts resolve on the host filesystem, so a locally
launched stack bakes in image code rather than the current edits, and there is no
headset, no push credential to GitHub Actions, and no live relay peer in the box.
Rather than block three waves on infrastructure the container cannot provide, the
liveness canaries were **stacked to one live-session run at sprint end**. Each is
mechanism-complete and fired-in-test locally; each awaits the same stack-up.

The batch:

1. **D1** beam traffic reaching the desktop screen (transport wired at boot).
2. **COM-14 / D4** live Schnorr 31402 round-trip against a selected node.
3. **COM-15 / V1 / D6 / M5** voice e2e: spoken command → accepted signed 31402 → Kokoro ack.
4. **D3 / REC-2** broker case e2e (rides the COM-15 stack-up).
5. **RES-a** Nostr relay-tap round-trip (canary evidence lands as a kind-1 tagged event).
6. **COM-13 / F2** forum agent badge live render.
7. **RES-b** `CANARY-CANON-DIAGRAM` firing in GitHub Actions (turning a bad-diagram probe PR red).
8. **RES-d** `CANARY-CANON-DRIFT` firing in GitHub Actions (turning a drift-injecting probe PR red).
9. **P2 MR:** `M4-RAY` (one-shot, in-headset ray resolves a node) and `COM18-INTERV` (standing, in-headset intervention panel POSTs a signed decide); `M1-HUD` rides the P0 COM-14 identity primitive.

Until each fires live, its item is held at the tier its offline evidence supports
and its Canary column reads `pending-live-session`. The MR surface is the clearest
case: the Godot copresence chain is complete and property-tested (`standalone`),
and `integrated` is one sidecar fire away: a single headset session, not more
code.

## Four-surface state against the 40/30/20/10 lens

Scored on the Chapter 12a weighting carried into the meta-PRD's Evaluation Lens
(Forum 40, Desktop 30, Mixed reality 20, Voice 10), against each surface's
per-surface definition of "closed":

- **Forum (40): closed at code, one live render pending.** All six lens conditions are met at the evidenced tier: agent badges + read-only member panels (COM-13 + F1), three decision outcomes reachable (COM-16 delegate/promote/precedent), decision-audit read API live (COM-17 `GET /api/governance/decisions`), relay-confirmed decisions replacing optimistic-send (COM-17 `publish_with_ack`), roster manageable in-UI (F8 Agents tab), NIP-42 claim reconciled (drift set). The only pending observation is the badge's live render (batch item 6).
- **Desktop (30): closed at code; two live canaries + one honest boundary.** Beam-and-avatar poll wired at boot (D1, register-drift correction), node-select opens steering + approval controls (D2/D3 via REC-2), MCP status dot reflects real reachability (D5, landed), nodes carry a verified `did:nostr` (COM-14). Pending live: D1 beam traffic and the COM-14 Schnorr round-trip. One honest boundary stands: MCP-native external agents cannot be interrupted (no terminate verb) and the client now discloses that state rather than offering a dead control.
- **Mixed reality (20): built to `standalone`; `integrated` gated on a headset.** In-headset identity badge (M1/COM-18), spatial intervention panel + ambient ACSP indicator (M2/COM-18), gaze/pinch/controller selection arbiter (M4), `isXRMode` handling (M6) are all implemented and property-tested in the Godot chain. None can reach `integrated` in the build container: there is no headset. This is the surface most honestly short of its target tier, and it is short by hardware, not by code.
- **Voice (10): closed at code; one live e2e pending.** The PTT-to-selected-actor governed loop is wired end to end (COM-15/V1: VoiceIntentClient → mandate-gated `/v1/voice-intent` → Kokoro ack), the low-confidence clarification turn exists (V3 `voice_clarification.rs`), and the docs match shipped behaviour (V4). Pending live: the voice e2e (batch item 3), which also clears D3's broker case e2e and M5's MR-side consumption.

The lens reading at sprint end: three of four surfaces are closed to the tier
their code evidences, with their loop-closing observations batched to one live
session; the fourth (MR) is complete in code at `standalone` and blocked on a
physical headset. The two-quarter evaluation window fixed in the meta-PRD stands
unchanged. The falsification condition is that the KPIs cannot be computed from
their sources, or can be computed and show no compounding, and the surfaces still
score as they did in the register. This register is the baseline that window is
measured against.

## Compatibility-matrix reflection

`docs/architecture/compatibility-matrix.md` §"Gap-Close Sprint — P1/P2 Item Tiers"
reflects the P1/P2-closed items at their evidenced tiers, per the PRD acceptance
criteria and this register. The P0 tier table there is authoritative to v1.1 and
is left in place (immutability); the P1/P2 table is authoritative to this
document.

## Next version

v1.3 is an addendum, not a new wave cut: it is stamped when the stack-up
live-session run fires the pending-live batch, promoting the affected loop items
from their current tiers to `integrated` (and the MR chain to `integrated` on a
headset session) on green canaries, and recording any observation that fails to
fire as a re-opened item chaining forward. No P0/P1/P2 status row, tier or canary
state in v1.2 is changed by that addendum; it forward-chains.
