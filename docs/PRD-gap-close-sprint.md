# PRD: Final Gap-Close Sprint (Meta-PRD)

**Owner:** DreamLab AI
**Status:** Accepted (ratified 2026-07-08; execution commenced same day)
**Date:** 2026-07-08
**Version:** 1.0
**Governed by:** [ADR-004 Gap-Close Sprint Governance](ADR-004-gap-close-sprint-governance.md), [ADR-002 Ecosystem Alignment Governance](ADR-002-ecosystem-alignment-governance.md)
**Bounded context:** [DDD Gap-Close Context](DDD-gap-close-context.md)
**Book cross-reference:** Chapter "The Gap Register" (chapters/14b), "Evaluating the Living Experiment" (chapters/14a), "The Four Surfaces" weighting (chapters/12a)
**Ratification:** This inventory is ratified as the sprint's authoritative register under [ADR-004](ADR-004-gap-close-sprint-governance.md); the P0/P1 work on the `gap-close/2026-07` branches (per-repo) is the execution record, and satisfies the DDD Gap-Close `RegisterPublished` trigger ("Meta-PRD inventory ratified") that `RegisterKeeper`'s first publish depends on.

## TL;DR

The product of this PRD is not code. It is a coordinated set of child PRD/ADR pairs, one per repository, that together close the measured distance between the VisionFlow ecosystem's substrate and its surfaces. The book fixed the numbers first: a four-surface audit found 28 typed gaps (7 critical, 17 major, 4 minor) and the roadmap chapter converted them into 18 scheduled commitments, and a July 2026 book-production pass surfaced 5 further residuals. This document consolidates all three registers into one inventory, assigns every item exactly one owning repository and one sprint wave, and directs each repository to mint its own child documents under the shared machinery of ADR-002. It stays deliberately at the level of the full sweep; the engineering detail lives in the children.

The through line the register named governs the whole sprint: the ecosystem's failures are rarely "it does not work", they are "it is built, and unwired". A cryptographic identity that is never read at interaction time; a governance plane real enough to reject a forged approval yet open to one operator; a beam transport complete to the last actor and starved by an absent `connect()` call. Closure therefore means a wire proven live, not a design accepted. Every loop-closing item in this sprint ships with a liveness canary, because the codebase's own history (ADR-043's KPI model, accepted April 2026, unbuilt for three months) is the standing proof that an accepted design is not a shipped one.

## Goals

| Goal | Outcome |
|---|---|
| G1: One register, one owner each | Every gap, commitment and residual has exactly one owning repository and one child document; no item is unowned or double-owned |
| G2: Child documents, not central edicts | Each repository authors its own PRD/ADR pair for its slice, reconciled at the VisionFlow canon per ADR-002; this document never prescribes repo-local implementation detail |
| G3: Closure is code-verified at a stated tier | No item is "closed" on documentation alone; closure carries evidence at the ADR-002 maturity tier claimed, and a fired liveness canary for every loop-closing item |
| G4: Falsification precedes work | Every work package states, before implementation, the condition under which it is judged not done — matching the book's own falsifiability commitment |
| G5: Scored on the four-surface lens | Progress is measured on the 40/30/20/10 forum/desktop/MR/voice weighting fixed in book Chapter 12a, against the per-surface definitions of "closed" carried forward here |

## Non-Goals

- **A monolithic cross-repo pull request.** Rejected in ADR-004. Each substrate retains local authorship; the canon owns only the cross-repo view.
- **New capability beyond the register.** This sprint closes measured gaps and honours standing commitments. Net-new product ideas outside the 28+18+5 inventory are out of scope and belong in fresh PRDs.
- **Re-opening settled architecture.** ADR-003's finding that the broker is distributed by design, and the standalone-first freeze of mesh federation (ecosystem-map G2), are inputs, not questions. Where a gap touches them it is scoped to what those decisions already permit.
- **Relay federation as a P0 or P1 target.** Federation is `planned`, parked standalone-first. It appears here only as a single P2 extension item with an explicit rescope-or-build fork.
- **Descoping the honesty discipline.** The register's value is that auditor and audited are the same organisation and every claim is checkable. Nothing in this sprint may soften a maturity claim to make a wave look closed.

## Consolidated Gap Inventory

Three source registers feed one inventory. Deduplication key: the book's roadmap chapter already promoted six register gaps into scheduled commitments 13–18; those gaps are listed once with a **covered-by** cross-reference to their commitment, and the commitment carries the **subsumes** back-reference. An item is counted against its owning repository once, at the wave of the commitment that carries it where one exists, otherwise at its own assessed wave. Maturity of every current claim uses the ADR-002 vocabulary (`historical`, `planned`, `scaffolded`, `standalone`, `integrated`, `federation-verified`, `released`).

### Table A — The 28 register gaps (book Chapter 14b)

| Gap | Surface | Type | Severity | Owning repo | Wave | Covered-by |
|---|---|---|---|---|---|---|
| F1 Governance plane admin-only; ordinary humans excluded | Forum | canon→practice | Critical | nostr-rust-forum | P1 | — |
| F2 No agent disclosure; agents indistinguishable from humans | Forum | theory→practice | Critical | nostr-rust-forum | P0 | COM-13 |
| F3 Decisions binary; graduated-escalation lifecycle dead code | Forum | canon→practice | Major | nostr-rust-forum | P1 | COM-16 |
| F4 Forum-client decisions optimistic-only; rejected reads as sent | Forum | canon→practice | Major | nostr-rust-forum | P1 | COM-17 |
| F5 No trust-calibration context; no decision-audit read API | Forum | theory→practice | Major | nostr-rust-forum | P1 | COM-17 |
| F6 No revoke/appeal/supersede authority for a published decision | Forum | theory→canon | Major | nostr-rust-forum | P2 | — |
| F7 No design response to approval fatigue / automation complacency | Forum | theory→canon | Major | nostr-rust-forum | P1 | COM-16 |
| F8 No web UI for agent-roster administration | Forum | canon→practice | Major | nostr-rust-forum | P1 | — |
| F9 Cross-relay federation of governance events absent | Forum | canon→practice | Major | nostr-rust-forum | P2 | — |
| F10 Silent on multi-agent social influence in threads | Forum | theory→canon | Minor | VisionFlow | P2 | — |
| D1 Embodiment channel inert; live activity never reaches screen | Desktop | canon→practice | Critical | VisionClaw | P1 | — |
| D2 No steering/interruption/approval affordance in running client | Desktop | theory→practice | Critical | VisionClaw | P1 | REC-2 |
| D3 No client-side ACSP case surface; pending judgment invisible | Desktop | theory→canon | Major | VisionClaw | P1 | REC-2 |
| D4 Agent nodes keyed by `task_id`, not `did:nostr` | Desktop | canon→practice | Major | VisionClaw | P0 | COM-14 |
| D5 Fabricated "MCP Connected" status miscalibrates trust | Desktop | canon→practice | Major | VisionClaw | P0 | — |
| D6 Push-to-talk globally scoped, not directed at selected agent | Desktop | canon→practice | Major | VisionClaw | P1 | COM-15 |
| D7 No pre-action intent legibility; embodiment shows only past | Desktop | theory→canon | Major | VisionClaw | P2 | — |
| D8 No swarm-level observability vs 2026 AgentOps table stakes | Desktop | theory→practice | Minor | VisionClaw | P1 | — |
| M1 Identity-blind; agent nodes carry no `did:nostr` in headset | MR | canon→practice | Critical | VisionClaw | P0 | COM-14 |
| M2 No agent-interaction or governance affordance reachable in MR | MR | theory→canon | Critical | VisionClaw | P2 | COM-18 |
| M3 Copresence theory unrepresented (no avatar, gaze, proxemics) | MR | theory→canon | Major | VisionClaw | P2 | — |
| M4 Input layer broken; targeting ray at world-origin, gaze absent | MR | canon→practice | Major | VisionClaw | P2 | COM-18 |
| M5 Voice PTT-to-actor binding is complete code, zero consumers | MR | canon→practice | Major | VisionClaw | P1 | COM-15 |
| M6 `enterVR()` never sets `isXRMode`; XR renders as desktop | MR | canon→practice | Minor | VisionClaw | P2 | COM-18 |
| V1 Canonical PTT voice-to-selected-actor governed loop absent | Voice | canon→practice | Critical | VisionClaw | P1 | COM-15 |
| V2 No multi-agent addressing or turn-taking model | Voice | theory→canon | Major | VisionFlow | P2 | — |
| V3 No conversational grounding or repair | Voice | theory→canon | Major | VisionClaw | P2 | — |
| V4 Docs describe a deprecated voice-to-swarm path as live | Voice | canon→practice | Minor | VisionClaw | P2 | — |

### Table B — The 18 roadmap commitments (book Chapter 14a)

| Commitment | Priority→Wave | Owning repo(s) | Subsumes |
|---|---|---|---|
| REC-1 Close governed-writeback security gaps (ontology auth, `/load` backdoor, PATCH bug, NIP-42 reconciliation) | P0 | VisionClaw · solid-pod-rs · nostr-rust-forum | (F9 NIP-42 sub-item) |
| REC-2 Finish the BrokerActor; surface case queue in control centre | P0 | VisionClaw (agentbox wiring) | D2, D3 |
| REC-3 Instrument contextual transaction cost per DAG | P1 | VisionClaw · agentbox | — |
| REC-4 Ship the four KPIs as a control-centre dashboard (ADR-043) | P1 | VisionClaw | — |
| REC-5 Adopt MAST-aligned failure telemetry | P1 | agentbox · VisionClaw | — |
| REC-6 Default AgentBox authority boundaries to escalation | P1 | agentbox · nostr-rust-forum | (overlaps COM-16) |
| REC-7 Make outcome learning real (feed_retrieval / feed_routing) | P1 | agentbox | — |
| REC-8 Engineer diversity in orchestration (multi-model verification) | P2 | agentbox | — |
| REC-9 Carry provenance to the human's pocket (mirror + digest) | P2 | agentbox | — |
| REC-10 Run the Insight Ingestion Loop v1 across the mesh | P2 | VisionClaw · agentbox · nostr-rust-forum | — |
| REC-11 Consolidate the data moat into one queryable trace | P2 | VisionClaw · agentbox · solid-pod-rs | — |
| REC-12 Pilot the stack beyond one team (kit cutover + mesh Phase 0) | P2 | dreamlab-ai-website · VisionFlow | — |
| COM-13 Agent disclosure for ordinary forum users | P0 | nostr-rust-forum | F2 |
| COM-14 Key the actor nodes by `did:nostr` | P0 | VisionClaw (agentbox source) | D4, M1, (V1 addressing root) |
| COM-15 Close the PTT voice-to-selected-actor loop | P1 | VisionClaw · agentbox | V1, D6, M5 |
| COM-16 Graduated escalation and an approval-fatigue response | P1 | nostr-rust-forum | F3, F7 |
| COM-17 Decision-audit read API and trust-calibration context | P1 | nostr-rust-forum | F4, F5 |
| COM-18 Headset intervention affordance and verifiable identity | P2 | VisionClaw | M2, M4, M6 |

### Table C — The 5 book-production residuals (July 2026 audit)

| Residual | Description | Owning repo | Wave |
|---|---|---|---|
| RES-a KG-backend liveness | `visionclaw-server:4000` was unreachable a full working day; fail-open worked, but liveness monitoring of the KG backend is itself absent. Underpins the sprint-wide liveness-canary requirement | VisionClaw | P0 |
| RES-b Diagram-as-code render | Local mermaid pipeline broken (`mmdc ERR_MODULE_NOT_FOUND`); published mermaid-derived PDFs shipped with invisible text. Needs a verified render step and a visual regression check (ADR-111 execution outstanding) | VisionFlow | P0 |
| RES-c solid-pod-rs stale diagrams | Rendered architecture diagrams are stale/broken versus their `.mmd` sources | solid-pod-rs | P1 |
| RES-d Self-description drift | Three skill counts and three ontology class counts were live in one tree on one day; the drift ADR-002 reconciles needs an automated counter | VisionFlow · agentbox · VisionClaw | P1 |
| RES-e Wardley export quality | Wardley map exports carry application UI chrome and sub-print resolution | VisionFlow | P2 |

**Inventory totals.** 28 register gaps + 18 commitments + 5 residuals = 51 line items, resolving after cross-reference to 28 gaps owned once each, 12 net-new recommendations (of the 18 commitments, six — COM-13 to COM-18 — subsume register gaps), and 5 residuals.

### Distribution summary

Every line item lands in exactly one wave. The register gaps distribute across owning repositories without ambiguity; commitments and residuals are attributed to their lead repository, with shared items noted.

| | P0 | P1 | P2 | Total |
|---|---|---|---|---|
| Register gaps | 4 | 13 | 11 | 28 |
| Commitments | 4 | 8 | 6 | 18 |
| Residuals | 2 | 2 | 1 | 5 |
| **Wave total** | **10** | **23** | **18** | **51** |

| Owning repo | Register gaps | Lead commitments / residuals |
|---|---|---|
| nostr-rust-forum | 9 (F1–F9) | COM-13, COM-16, COM-17; shares REC-1 (NIP-42), REC-6, REC-10 |
| VisionClaw | 17 (D1–D8, M1–M6, V1, V3, V4) | REC-2, REC-4, COM-14, COM-15, REC-10; RES-a; shares REC-1, REC-3, REC-11 |
| VisionFlow (canon) | 2 (F10, V2) | RES-b, RES-d, RES-e; co-owns REC-12; owns all canon reconciliations |
| agentbox | 0 | REC-5, REC-6, REC-7, REC-8, REC-9; shares COM-14, COM-15, REC-3, RES-d |
| solid-pod-rs | 0 | RES-c; shares REC-1 (PATCH), REC-11 |
| dreamlab-ai-website | 0 | REC-12 (lead) |

The concentration is expected and load-bearing: nostr-rust-forum and VisionClaw own 26 of the 28 gaps because the forum owns the decision surface and VisionClaw owns the three observation-and-interaction surfaces (desktop, MR, voice). The two gaps VisionFlow owns are both `theory→canon` — positions the canon never took, which only the canon can take.

## Sprint Waves

Waves gate one another. A wave is not started until the prior wave's liveness canaries are green in the substrates that wave touched.

**P0 — Correctness and security preconditions.** The floor. Nothing above is worth measuring until these close (roadmap Recommendation 1's own framing). Contains: the governed-writeback security gaps (REC-1), the BrokerActor loop (REC-2), agent disclosure (COM-13/F2), `did:nostr` keying of actor nodes (COM-14/D4/M1 — the single shared unblocker for desktop, MR and voice addressing), the fabricated MCP-status honesty fix (D5), KG-backend liveness monitoring and the sprint-wide canary harness (RES-a), and the verified diagram render step (RES-b). Security front-loaded is an enabler, not a blocker.

**P1 — Measurement, the embodiment join, and the governance surfaces.** The instrumentation and the surfaces that carry the decision. Contains: CTC (REC-3), the four-KPI dashboard (REC-4), MAST telemetry (REC-5), escalation defaults (REC-6), outcome learning (REC-7); the embodiment join that makes live activity reach the desktop screen (D1) and its steering/case surfaces (D2/D3 via REC-2); the PTT voice-to-selected-actor loop (COM-15/V1/D6/M5); graduated escalation and the approval-fatigue response (COM-16/F3/F7); the decision-audit read API, trust-calibration context and optimistic-send fix (COM-17/F4/F5); the member read-only governance view (F1); the agent-roster admin UI (F8); swarm observability (D8); the self-description counter (RES-d); and the solid-pod-rs diagram refresh (RES-c).

**P2 — Extension: MR parity, the voice conversation layer, and federation.** Reaches beyond the core loop. Contains: model diversity (REC-8), provenance-to-pocket (REC-9), the Insight Ingestion Loop v1 (REC-10), the data-moat consolidation (REC-11), the external pilot (REC-12); the headset intervention affordance and copresence work (COM-18/M2/M4/M6, M3); pre-action intent legibility (D7); supersession authority (F6); multi-agent social-influence and voice addressing/grounding positions (F10, V2, V3); the voice-docs honesty fix (V4); and the Wardley export cleanup (RES-e). Federation (F9) sits here with an explicit rescope-or-build fork per ecosystem-map G2.

## Per-Repository Work Packages

Each repository mints a **child PRD** (its slice as a product increment) and a **child ADR** (the decisions its slice forces), reconciled at the canon per ADR-002. Every child document MUST contain: (1) the required sections below; (2) explicit acceptance criteria per owned item; (3) a **falsification statement** per work package, written before implementation; and (4) a **maturity claim** per item using the ADR-002 seven-tier vocabulary, with the target tier stated and the current tier cited from the register.

**Minting and reconciliation cadence.** A repository mints its child pair before entering its first wave, not before the sprint. Reconciliation at the canon is the ADR-002 loop already in service: each closed item updates `docs/architecture/compatibility-matrix.md` at the tier its evidence supports, and a coordinated release manifest under `docs/releases/` pins the substrate SHAs when a wave is promoted beyond local draft. A new register version (immutable, superseding) is cut at each wave boundary; corrections chain forward and never edit a published register in place. Where an item spans repositories (REC-1, COM-14, COM-15, REC-10, REC-11), the meta-PRD names the lead; the sub-item boundary between co-owners is fixed in the child documents, not here.

### VisionFlow (canon) — child: PRD-gap-close-canon + ADR

- **Owns:** F10, V2 (position), RES-b, RES-d (counter), RES-e; the canon reconciliations behind COM-13 disclosure MUST, COM-18 XR-oversight ADR, D7 intent-legibility position, F6 supersession-authority specification (into DDD-judgment-broker-context), F9 federation rescope; and this meta-register itself.
- **Required child sections:** Reconciled Claims (each canon assertion the sprint touches, old wording → new wording); Disclosure Norm (the `did:nostr` self-identification MUST written into `protocol/identity-spine.md`); Automated Self-Description Counter (skill count, ontology class count, agent roster size — one source of truth, CI-enforced); Diagram-as-Code Gate (verified render + visual regression per ADR-111).
- **Acceptance criteria:** every reconciled claim cites the child document that discharges it; `docs/architecture/compatibility-matrix.md` reflects the post-sprint tier of every item; the counter fails CI on drift; no published PDF ships with invisible or unrendered diagram text.
- **Falsification statement:** *This package is falsified if any canon page still asserts a capability at a tier its owning repository's child document has not evidenced, or if a second skill/class count can appear anywhere in the tree without CI failing.*
- **Maturity:** canon reconciliation is `integrated` when at least two substrates' child docs cite it; the counter is `released` only when pinned in a release manifest.

### VisionClaw (`project`) — child: PRD-gap-close-visionclaw + ADR

- **Owns:** D1–D8, M1–M6, V1, V3, V4; REC-1 (ontology `/propose` auth, `/load` backdoor), REC-2 (BrokerActor + control-centre case queue), REC-3 (agent-events CTC schema), REC-4 (four-KPI dashboard, ADR-043 resurrection), COM-14 (`did:nostr` keying), COM-15 (STT-side binding), COM-18 (headset), REC-10/REC-11 shares; RES-a (KG-backend liveness + canary harness).
- **Required child sections:** Embodiment Join (the `connect()` wire and `.nodes` fix that makes D1 live); Identity Keying (the `did:nostr` field on the `Agent` struct, sourced from agentbox at spawn, verified before trust); Steering Surface (mount `AgentDetailPanel`/`BotsControlPanel` behind selection; the missing `/bots/submit-task` + interrupt routes); Control-Centre Governance (broker case queue, ambient ACSP indicator); Liveness Canary (the harness every P0/P1 loop item registers against).
- **Acceptance criteria:** selecting any agent node yields a verified `did:nostr` addressable by a signed 31402 (COM-14 exit); live agent activity draws beams and avatars on screen with the poll started at boot (D1); the MCP status dot reflects a real `test_connection()` (D5); a spoken command to a selected agent produces an accepted signed 31402 and a Kokoro TTS acknowledgement end to end (COM-15/V1 exit); the KG backend's reachability is monitored and its loss fires the canary rather than silently failing open (RES-a).
- **Falsification statement:** *This package is falsified if any surface still keys agents by `task_id`, if `resolveAgentPosition` returns false at boot with live data present server-side, if any status indicator is decoupled from ground truth, or if a loop item is declared closed without its canary having fired in a live session.*
- **Maturity:** D1/COM-14/COM-15 target `federation-verified` (end-to-end runtime proof across substrates); D5/M6/V4 target `integrated`; MR copresence (M3) may land `scaffolded` if instantiation is deferred, but must be labelled so.

### agentbox — child: PRD-gap-close-agentbox + ADR

- **Owns:** REC-5 (MAST telemetry), REC-6 (escalation-by-default authority boundaries), REC-7 (outcome learning: `feed_retrieval`, `feed_routing`), REC-8 (model diversity), REC-9 (provenance-to-pocket); COM-14 source-side (`did:nostr` at spawn), COM-15 producer (`/v1/voice-intent` un-gating + scene-selection binding), REC-3 hook fields, RES-d skill-count source.
- **Required child sections:** Authority Model (recoverable vs zero-tolerance action classes; the blocking-on-signed-response pattern); Failure Taxonomy (the 14 MAST modes replacing free-text error strings); Outcome Learning (the statistical-sample floor, the two gated consumers, deletion of stale "learns" comments); Voice-Intent Producer (un-gate behind mandate, accept a scene-selected actor `did:nostr`, wire callers).
- **Acceptance criteria:** new skills default to escalation with each action classified; every failure through the pipeline carries a MAST tag; `feed_retrieval`/`feed_routing` influence live retrieval and routing once the floor clears, evidenced not asserted; `/v1/voice-intent` accepts and dispatches a scene-bound signed 31402.
- **Falsification statement:** *This package is falsified if the intelligence banner or router confidence still stands in for outcome learning that does not run, if any comment still claims the old path learns, or if voice-intent remains gated-off or caller-less.*
- **Maturity:** REC-7 target `integrated` (learning observably influences a second consumer); REC-8/REC-9 target `integrated`; MAST telemetry `integrated` when the QE fleet and agent-events both emit the taxonomy.

### solid-pod-rs — child: PRD-gap-close-solid-pod + ADR

- **Owns:** REC-1 non-destructive PATCH bug; the NIP-98 verifier consolidation (ecosystem-map G3/G5 residual — move the replay-store abstraction into `core`); REC-11 pod git-mark/block-trail share; RES-c stale diagrams.
- **Required child sections:** PATCH Correctness (the non-destructive merge fix with a regression vector); Shared Verifier (one NIP-98 verifier across tiers, or the documented edge-local exception); Provenance Trail (the pod contribution to the unified queryable trace).
- **Acceptance criteria:** a PATCH that previously destroyed sibling data preserves it, proven by a fixture; the NIP-98 verifier is either single-sourced or the exception is recorded in the compatibility matrix; diagrams regenerate from `.mmd` under the RES-b render gate.
- **Falsification statement:** *This package is falsified if the shipped server still binds only a slice the library implements without that being documented, or if a PATCH regression re-appears without a failing test.*
- **Maturity:** PATCH fix target `integrated`; shared verifier `integrated` only when a second tier consumes it, otherwise the exception is `standalone` and labelled.

### nostr-rust-forum — child: PRD-gap-close-forum + ADR

- **Owns:** F1, F3–F9; COM-13 (disclosure badge), COM-16 (graduated escalation + risk-tiering), COM-17 (decision-audit read API + optimistic-send fix), REC-1 NIP-42 sub-item, REC-6 relay-projection share, REC-10 discovery-surface share.
- **Required child sections:** Member Surface (read-only panel view behind `AuthGated`, response controls behind `AdminGatedGovernance`); Disclosure (agent badge driven by `agent_registry`, naming the authorising principal); Decision Integrity (`publish_with_ack` port; `GET /api/governance/decisions`; reasoning/confidence/risk-tier in the panel schema); Escalation (wire `DecisionOrchestrator` delegate/promote/precedent into the 31403 projection; risk-tiering that suppresses low-risk panels); Roster Admin (an Agents tab calling the nine existing endpoints); NIP-42 (perform the AUTH round-trip or reconcile the claim).
- **Acceptance criteria:** an ordinary member sees agent-authored items badged and read-only panels but no Approve/Reject (F1, COM-13 exits); at least three decision outcomes reachable end to end and a documented risk tier suppresses low-risk panels (COM-16 exit); a panel displays reasoning and confidence at decision time and an operator can query decision history (COM-17 exit); a relay-rejected decision no longer reads as "sent".
- **Falsification statement:** *This package is falsified if a non-admin still cannot view a panel, if any agent-authored item renders without a badge, if a non-binary action still parks a case in `under_review` forever, or if the relay still advertises `auth_required:false` while the canon claims an authenticated signer.*
- **Maturity:** F1/COM-13/COM-16/COM-17 target `integrated`; F9 federation stays `planned` unless a `MeshTransport` is wired, in which case `scaffolded` → `integrated` on evidence.

### dreamlab-ai-website (DreamLab Edge) — child: PRD-gap-close-edge + ADR

- **Owns:** REC-12 kit cutover (PRD-012) and the operator overlay for the disclosure badge and roster admin; the `dreamlab.toml` roster legibility (making bot authority visible, forum gap F8's out-of-band configuration).
- **Required child sections:** Kit Cutover (completing the de-specialisation into a thin forum-kit consumer); Overlay Surfaces (rendering the forum's new disclosure badge and Agents tab in the branded deployment); Roster Legibility (surfacing which principal authorised each agent).
- **Acceptance criteria:** the branded deployment renders the forum's post-sprint disclosure and roster surfaces without local reimplementation; a kit compatibility record pins the forum SHA per production deployment.
- **Falsification statement:** *This package is falsified if the Edge deployment reimplements any surface the forum kit now owns, or if the production deployment cannot state which forum-kit SHA it runs.*
- **Maturity:** kit cutover target `integrated`; overlay surfaces `integrated` when they consume the forum kit's shipped components rather than local copies.

## The Evaluation Lens

The next edition scores this sprint on the four collaboration surfaces, weighted as fixed in book Chapter 12a and carried into Chapter 14a: **Forum 40, Desktop 30, Mixed Reality 20, Voice 10.** The forum owns the decision, the desktop owns observation, the headset extends it at room scale, and voice is the narrowest ingress. The weighting is the scoring frame; a surface's score is the fraction of its "closed" definition met, times its weight.

| Surface | Weight | What "closed" means |
|---|---|---|
| Forum | 40 | Ordinary members see agent badges and read-only panels; at least three decision outcomes reachable; decision-audit read API live; relay-confirmed decisions replace optimistic-send; agent roster manageable in-UI; the NIP-42 claim reconciled with the mechanism. |
| Desktop | 30 | Live agent activity reaches the screen (beam and avatar poll actually starts); node selection opens per-agent steering and approval controls; the MCP status dot reflects real reachability; agent nodes carry a verified `did:nostr`. |
| Mixed reality | 20 | In-headset actor identity verifiable on a selected agent; a spatial intervention affordance plus an ambient ACSP indicator; hand or gaze targeting that resolves a real selection; `isXRMode` set correctly during a session. |
| Voice | 10 | The PTT-to-selected-actor governed loop wired end to end with audible confirmation; a clarification turn on low-confidence input; documentation matching shipped behaviour. |

The falsification condition is fixed now, before the data exists, so it cannot be quietly redefined: if two quarters from publication the KPIs cannot be computed from the sources below, or can be computed and show no compounding, and the four surfaces still score as they did in the register, that is evidence against the mesh thesis as deployed here, and the next edition says so in these terms.

## Measurement Commitments

Carried forward verbatim from book Chapter 14a; the child PRDs for VisionClaw and agentbox own the data sources.

| Measure | Concrete data source | Cadence |
|---|---|---|
| Mesh Velocity | Timestamps across the insight loop: `ontology_propose` event to broker decision to merged enrichment | Quarterly |
| Augmentation Ratio | Agent-action volume from `/wss/agent-events` against ACSP escalation volume | Monthly roll-up |
| Trust Variance | Dispersion of broker decision outcomes (approve, amend, reject rates) over a rolling 30-day window | Monthly |
| HITL Precision | Proportion of ACSP escalations where the human decision materially changed the outcome | Monthly |
| Contextual transaction cost | Handoff counts, token burden and verification outcomes per DAG from the extended agent-events envelope | Quarterly |

## Quality Gates (per build-with-quality)

Applied to every child document and its implementation, adapting the skill's Expectation-Driven Development discipline to a cross-repo sprint:

1. **Expectation before code.** Each owned item is expressed as a falsifiable expectation (the falsification statement) authored before implementation, with counter-examples — the behaviour that would prove it not done.
2. **Evidence with receipts.** Closure carries an execution receipt: command, raw output, timestamp, git SHA. Narrative evidence ("tested, works") is auto-rejected.
3. **Anti-fox separation.** The agent or reviewer that verifies a closure is not the one that produced it, and sits on a different model family — the register's own auditor-and-audited discipline made operational.
4. **Liveness canary.** Every loop-closing item registers a canary that must fire in a live session before closure; a green canary is the difference between `integrated` and a claim.
5. **Maturity honesty.** No item is labelled above the tier its evidence supports; a deferred sub-feature is labelled `scaffolded` or `planned`, never quietly folded into a "closed" parent (the register's D5 and desktop-beam findings are the standing counter-examples).
6. **Staleness blocks the gate.** Evidence older than the SHA it was captured against, or than 30 days, is stale and re-opens the item.

## Protocol and Cross-Reference

- Nostr governance kinds: 31400–31405 (ACSP); federation kinds 38xxx (dropped by the forum allow-list today, F9).
- Identity: `did:nostr:<hex-pubkey>`, canonicalised on the Multikey form by VisionClaw ADR-125.
- Embodiment: `/wss/agent-events`, `0x23` binary frame, `AgentBeamActor` (VisionClaw ADR-059); gluon force `planned`.
- Governance machinery: ADR-002 (maturity vocabulary, release manifests, compatibility matrix), ADR-003 (distributed broker), ADR-004 (this sprint's governance decision).
- Book: Chapter "The Gap Register" (14b) is the register; "Evaluating the Living Experiment" (14a) is the schedule; "The Four Surfaces" (12a) is the weighting.
