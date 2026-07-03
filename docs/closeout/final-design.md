# VisionFlow Ecosystem Closeout — Final Design

**Status:** Strategic closeout analysis — code-verified, cross-validated by two independent audit swarms
**Date:** 2026-07-03
**Method:** Ruflo mesh (`swarm_1783067180317_99wd065`), Fable queen coordination. 22 Opus gap analysts + 10 Sonnet 5 / agentic-qe auditors (32 agents, ~3.1M tokens, 1,045 tool invocations), two GLM-5.2 adversarial collaboration passes. Every finding carries file:line or command-output evidence; the two swarms ran independently and converged on the same systemic diagnosis.
**Supersedes:** the Gap Register (G1–G8) in `docs/ecosystem-map.md` (dated 2026-05-22/30), which this analysis found stale in both directions.
**Machine-readable register:** [`unified-findings-register.json`](unified-findings-register.json) — 286 findings (25 P0 / 79 P1 / 110 P2 / 72 P3), tagged by source swarm, scope, kind, severity, evidence.

---

## 1. Executive Summary

The VisionFlow ecosystem is **substantially built and dangerously misdescribed**. Across 323 ADRs audited in nine repo slices, 202 (63%) are genuinely implemented and verify in code — including work the canon still lists as open. The governance decision loop, the BC20 provenance bridge, the no-Tokio pod core extraction, and the 2026-05-30 provenance loops all shipped and merged. This is a real system, not vapourware.

The closeout problem is not missing product code. It is a **failed immune system**. The 25 P0 findings cluster almost entirely in enforcement and truth infrastructure:

- **CI theatre** — thousands of tests exist that no CI trigger ever runs.
- **Truth decay** — 90 doc-drift findings; the canonical ecosystem map simultaneously understates shipped work and asserts capabilities (mesh federation, NIP-42, IS-Envelope routing, NIP-26) that exist nowhere.
- **Trust-layer overstatement** — the PROV-O provenance emitter is dead code with zero production callers, and the SHACL shapes are never loaded, while current marketing commits advertise both.
- **Silent degradation** — the ecosystem's semantic memory runs on hash-placeholder embeddings, and RuView's sensing/inference runs on simulated data and rule-based heuristics, all reporting success.

**Closeout verdict:** the system can be honestly closed out in one focused push, but the sequence matters. Enforcement first (Step Zero), then truth reconciliation, then the six remaining mechanism gaps. Fixing docs before CI, or features before either, re-drifts within weeks.

---

## 2. What Is Genuinely Done (and under-claimed)

The audit's most consistent surprise: the ecosystem **understates its own progress**. Confirmed shipped and merged to main:

| Claimed open in canon | Verified reality |
|---|---|
| "Broker write-back local/unmerged, agent decision application the remaining gap" (ecosystem-map) | `POST /api/enrichment-proposals/{id}/decide` merged (`023c847b0`), gated (`72bd6ec05`), hardened (`bed1565…`); `handleGovernanceDecision` fully implemented in agentbox with PROV-O activity/receipt URNs (`local-process-manager.js:122`, `stdio-bridge.js:72`) since `e1a8d716` (2026-05-22) |
| "Beam+gluon render (ADR-059 Phase 2b) still open" | Beam actor shipped and wired; only the gluon attractive-force sub-feature remains a deferred no-op |
| "Real ACSP 31402 dispatcher still open" | Dispatcher shipped (the `AgentActionEnvelope` coexists by design rather than being retired) |
| "did:nostr canonicalisation unresolved" (Open Question 1) | Answered by VisionClaw ADR-125 |
| "G1: umbrella docs sparse" | Referenced entry-point docs now exist; premise resolved |
| "G5: CF-portability core extraction undecided" | The no-Tokio `core` feature shipped; forum consumes published solid-pod-rs `0.5.0-alpha.3` with `default-features=false` |
| ADR-003 "Judgment Broker 65% / 35% gap" | 2 of 3 listed gaps closed; only the BrokerActor main-merge remains |
| ~40 "Proposed" ADRs across RuView, ruvector, agentbox, VisionClaw | Fully implemented in code (e.g. RuView ADR-029/030/031/033, ruvector ADR-045/057, agentbox ADR-023/034, VisionClaw ADR-112) |

**ADR closeout table (per audited slice):**

| Repo slice | ADRs | Implemented | Partial | Unimplemented | Stale/superseded |
|---|---|---|---|---|---|
| VisionFlow | 5 | 1 | 1 | 1 | 2 |
| VisionClaw (011–074) | 50 | 35 | 6 | 8 | 1 |
| VisionClaw (078–127) | 47 | 16 | 9 | 20 | 2 |
| agentbox | 35 | 27 | 6 | 0 | 2 |
| nostr-rust-forum | 20 | 13 | 1 | 3 | 3 |
| solid-pod-rs + website | 30 | 18 | 4 | 5 | 3 |
| ruvector (first half) | 58 | 47 | 4 | 5 | 2 |
| ruvector (second half) | 36 | 29 | 3 | 0 | 4 |
| RuView | 42 | 16 | 15 | 10 | 1 |
| **Total** | **323** | **202 (63%)** | **49** | **52** | **20** |

The unimplemented cluster is concentrated and legible: VisionClaw's second-half ADR range (the forum-kit convergence cluster 078–085, self-improving ontology stack 121–123, mobile bridge 092–097) and RuView's speculative security/consensus specs (007/008/010) account for most of it.

---

## 3. The Ten Themes

Each theme aggregates cross-validated findings. Full evidence in the register JSON.

### T1 — Truth decay (90 doc-drift findings)
The canon is stale **in both directions**. `ecosystem-map.md` misreports merged provenance loops as local/unmerged, lists shipped features as open, and simultaneously asserts mesh/NIP-42/IS-Envelope capabilities that no substrate implements. `VisionFlow/README.md` contradicts VisionClaw's own canonical figures (92 vs 82 CUDA kernels; "114 CQRS handlers" vs 44 hexser handlers with CQRS explicitly retired by VisionClaw ADR-089). The same production graph is described with three different node counts (934 / ~998 / 17k+). solid-pod-rs states three different parity percentages across three canonical docs. ADR status fields are systematically unreliable: ~40 Proposed-but-shipped, plus Accepted-but-unbuilt.

### T2 — Enforcement theatre (21 ci-gap findings; the P0 cluster)
- VisionClaw: ~163 Rust test files + 42 Vitest suites, **no CI workflow builds or tests the backend at all**; the README build badge cites a `ci.yml` that has never existed.
- ruvector: full workspace tests run **only on tag push** (not since v0.1.16); RVF, hyperbolic-HNSW and micro-hnsw crates are workspace-excluded — their suites can never run.
- solid-pod-rs: CI never builds/tests/lints 6 of 7 workspace crates.
- agentbox: ADR-005's "a red contract suite blocks merge" is contradicted by the actual required-check list; the nine SLO assertions are `it.todo()` stubs.
- dreamlab-ai-website: the `ci-pass` aggregator gates on a `continue-on-error` Vitest job — structurally always green.
- RuView: zero CI for the Rust workspace the README calls "primary".

### T3 — Contract sync broken (fixture corpus)
The "strongest cross-repo alignment signal" (compatibility matrix) is illusory: checksums pass while `did-doc-conformance.json` has diverged four ways (7/8/9/7 vectors; two variants test a **superseded** DID canonicalisation). `sync-fixtures.sh` in all three consumer repos points at `docs/specs/fixtures/` — deleted from VisionClaw. Forum fixtures fail their own checksum manifest today. The release manifest is six weeks stale and the generator emits a broken (zeroed) agentbox entry. The event-kind registry is unowned and already drifting: agentbox federates kinds 38300–38304; the forum relay drops everything above 38100. Crypto verification in conformance tests is `test.skip`'d, and missing fixtures pass silently.

### T4 — Identity fragmentation
NIP-98 is independently reimplemented in four repos (VisionClaw `nostr_sdk`, solid-pod-rs `k256`, forum `k256`, agentbox `nostr-tools`) with divergent URL-matching, replay, and signature-gating semantics — G3 unresolved at the security-critical layer. Replay protection exists only in the forum and CF pod tier; the native git-capable tier has none. NIP-26 delegation is listed in the compatibility matrix and implemented in **no repo**. VisionClaw `/wss/agent-events` stamps frames "Signed"/attributed from a structural pubkey string **without signature verification**. solid-pod-rs Schnorr verification is off-by-default and fail-open, surviving only via implicit cargo feature-unification.

### T5 — Mesh federation is scaffold
`nostr-bbs-mesh` (forum, ADR-073's counterpart) has no `MeshTransport` implementation and is not even a dependency of the relay-worker. The forum relay advertises `auth_required:false` — the claimed NIP-42 gate is false; it gates by pubkey whitelist. VisionClaw has zero NIP-42. IS-Envelope **routing** (ADR-075) is implemented in no substrate (the spec, schema and vectors exist; the routing does not). Three of four runtime substrates default standalone, not two of four.

### T6 — Governance last-mile (four wires + one bleed)
Real progress (see §2), but: **(bleed)** agentbox `broker-bridge` reports false write-back closure — it ignores VisionClaw's `writeback_committed` and stamps `broker_pubkey:'unknown'`, a silent data-loss vector. **(wires)** `ConceptElevated` exists in no repo — the flagship elevation loop terminates at PR-creation with no closing event; agent-side application of returned 31403 decisions for agentbox-originated cases routes to a BrokerActor absent from main; the git-bridge `WriteBackSaga` posts to `/api/ingest/writeback`, a route not registered on VisionClaw main; `ElevationActor` skips the documented Whelk EL++ consistency gate.

### T7 — Trust layer overstated (the Opus P0)
The PROV-O reification emitter is **dead code: zero production callers** — no ingest, inference or BC20 crossing produces queryable provenance triples. The five `.shacl.ttl` NodeShapes are never loaded; the running validator is a hardcoded SHACL-lite Rust matcher. The ingest write path runs the SHACL gate **advisory-only** while the trust-status endpoint reports `writePaths='enforcing'`. VisionFlow's most recent commits advertise exactly these capabilities.

### T8 — Silently degraded ML/infra substrates
- ruvector: the default AgentDB embedding pipeline is a **hash placeholder, not MiniLM-L6-v2** (the local Candle MiniLM path is a stub that errors) — semantic search is degraded ecosystem-wide, including the RuVector memory system this workspace itself relies on. `ruvector-dag` ships forgeable placeholder ML-DSA/ML-KEM by default. Postgres multi-tenancy is a facade (quota accounting, no data movement). IVFFlat indexing reports success while indexing nothing; mincut gating decides on a hardcoded constant.
- RuView: every real WiFi-CSI hardware adapter is unimplemented — sensing runs exclusively on simulated data; the NN inference layer silently substitutes rule-based heuristics while faking model loading; RVF export writes sine-wave placeholder weights; "hardware-validated on ESP32-S3" claims have no in-repo artifact.
- All of this **succeeds silently**. Nothing warns operators that they are running degraded paths.

### T9 — Harness engineering uncommitted and double-broken
ADR-004 (Accepted) and ADR-005, the `docs/engineering/` template tree, `scripts/harness-audit.sh`, and the harness-fitness-gates CI workflow are **all untracked in git** — the gate has never run and cannot run. The pairing ratio contradicts across three sources (30% in ADR-004, 60% in the compatibility matrix, 100% from the audit script) — and the 100% is vacuous: it counts ID cross-references in hand-authored JSON whose `source` paths mostly do not exist. ADR-004's founding premise (`handleGovernanceDecision` stubbed) was disproven by code committed a month before the ADR was written.

### T10 — Embodiment partial
Beam ships; gluon is a deferred no-op. Live agent-actor nodes are keyed by management-API `task_id`, not `did:nostr`, breaking the addressed-actor contract of the flagship voice loop. `ConceptElevated` (the loop's closing event) is missing (see T6). Per-user filter/settings persistence in VisionClaw is stubbed across three seams — authenticated writes silently discarded. `/layout/zones` is a live no-op returning `success:true`.

---

## 4. Closeout Triage — Finish / Freeze / Delete

Framework (adopted from GLM-5.2 collaboration, corrected by queen review): **Architectural Centrality** (is it on the flagship-flow critical path?) × **Replacement Cost**. Fix the mechanism before the brochure — a closeout that only edits claims leaves the system structurally dishonest.

### FINISH (high centrality — mechanism work, ordered)

| # | Work | Why / contents |
|---|---|---|
| F0 | **Stop the bleed:** broker-bridge false write-back closure | Silent data loss on the governance write-back; honour `writeback_committed`, propagate real `broker_pubkey` |
| F1 | **CI immune system (T2)** | Step Zero for everything else. VisionClaw backend CI; ruvector workspace tests on PR; solid-pod-rs all-crate CI; remove `continue-on-error` from gating jobs; un-exclude or explicitly quarantine excluded crates. A red suite must actually block merge |
| F2 | **Trust layer minimal viable path (T7)** | Wire the PROV-O emitter into ingest/inference/BC20 crossings; load the real `.shacl.ttl` shapes; make the write-path gate enforce or make the status endpoint stop claiming it does |
| F3 | **Identity spine minimal (T4)** | Adopt the shared solid-pod-rs 0.5.0 verifier in all four substrates; compile-guard Schnorr verification on; replay nonce on the native pod tier; real signature verification (or explicit unauthenticated downgrade) on `/wss/agent-events`; delete the NIP-26 row from the matrix until code exists |
| F4 | **Fixture re-canonicalisation (T3)** | Pick the canonical `did-doc-conformance.json`, fix `sync-fixtures.sh` paths in all three consumers, un-skip crypto verification, make missing fixtures fail, fix the release-manifest generator, assign an owner to the event-kind registry |
| F5 | **Governance wires (T6)** | Define and emit `ConceptElevated`; register or retire `/api/ingest/writeback`; merge or descope BrokerActor (agent-side 31403 application); wire the Whelk gate into `ElevationActor` |
| F6 | **Truth reconciliation sprint (T1)** | One pass over `ecosystem-map.md`, `compatibility-matrix.md`, READMEs, and ~75 ADR status fields. Normalise status vocabulary; fold amendments into decisions; regenerate the harness table programmatically |
| F7 | **Harness commit-or-downgrade (T9)** | Commit ADR-004/005 + templates + script + workflow with honest pairing data and real source paths, or downgrade ADR-004 to Proposed. An "Accepted" ADR must exist in git history |
| F8 | **Loud degradation (T8, GLM blind-spot)** | Unmissable `WARN` logs + deployment gates on hash-embedding mode, rule-based inference fallback, placeholder crypto, simulated-CSI mode. Silent fake success is banned |

### FREEZE (low centrality, high replacement cost — quarantine honestly)

- **Mesh federation (T5):** declare **standalone-first** as the supported deployment mode. Park ADR-073/PRD-010 mesh features, forum-kit convergence cluster (VisionClaw ADR-078..085), mobile bridge cluster (ADR-092..097), IS-Envelope *routing* (spec and vectors remain canonical). Matrix says "federation: designed, not shipped".
- **VisionClaw physics-v2** (~6k LOC feature-gated engine rewrite) — keep gated, mark experimental.
- **RuView hardware adapters:** declare **simulation-first** publicly; hardware validation is future work with named artifacts required to unfreeze.
- **ruvector excluded crates** (RVF, hyperbolic-HNSW, micro-hnsw) — either re-include with CI or mark experimental/unsupported in the workspace manifest.
- **ADR-005 Mandate-at-Grant** — remains Speculative; correct its context (governance-precedents namespace is already write-protected).
- **Spec-only ADR clusters** (RuView 007/008/010; VisionClaw 057/065/067; ruvector 010/011/028/036) — mark Deferred with explicit unfreeze criteria, or move to a `speculative/` register.

### DELETE (zero centrality, or active dishonesty)

- Dead code: `_moved_to_visionflow_actors/` (orphaned 31KB), parked pod-worker shims, headroom dead abstractions.
- **ruvector-mcp fake-success tools** — 15 of 21 advertised orchestration tools fabricate success/empty data; unknown tools report success. Delete or return honest `unimplemented` errors.
- Phantom claims: NIP-26 matrix row; "150×–12,500×" performance claims (contradicted by the repo's own committed benchmarks); "hardware-validated" assertions without artifacts; the never-existed CI badge.
- Empty shell crates (RuView api/config/db one-liners) — delete or mark scaffold explicitly.
- Superseded fixture variants after F4 re-canonicalisation.

---

## 5. Sequenced Closeout Roadmap

The P0 pattern (enforcement/truth, not product) dictates the order. Do not reorder — every later phase re-drifts without the earlier one.

| Phase | Contents | Exit criterion |
|---|---|---|
| **0. Immune system** | F0 + F1 (+ F8 warn-gates) | CI builds and tests every substrate on PR; red blocks merge; degraded modes scream |
| **1. Truth** | F6 + F7 | Canon docs match code both directions; ADR statuses reliable; harness in git history |
| **2. Contracts** | F3 + F4 | One NIP-98 verifier; fixtures byte-identical across consumers with a canary test importing from the canonical source |
| **3. Mechanism** | F2 + F5 | Queryable PROV-O on the hot path; real SHACL; governance loop closes end-to-end with `ConceptElevated` |
| **4. Verification** | Promoted runtime probes (§6) + end-to-end governance and embodied-loop smoke tests (already P1 in the old roadmap, still unbuilt) | Flagship flow demonstrated on a clean deployment, evidence archived |
| **5. Declaration** | Release manifest with repo SHAs, contract versions, fixture sync evidence; freeze register published | Ecosystem closeout declared with a reproducible manifest |

**Definition of Closed** (adopted, incl. GLM additions): (1) every Accepted ADR exists in git history and its claims verify in code, or it is downgraded; (2) **red-to-green mandate** — nothing is closed by deleting a test or excluding a crate; (3) **contract-test canary** — consumers import conformance fixtures from the canonical source so local drift fails builds immediately; (4) **zero advisory write paths** — the `advisory-but-reports-enforcing` pattern is banned; (5) degraded-mode operation is loud; (6) the compatibility matrix is generated from code/templates, never hand-maintained.

---

## 6. Residual Risk — Runtime Probe Backlog

Static analysis cannot verify runtime behaviour. Promoted probes (from GLM pass 1, filtered by evidence):

1. **Forged-identity probe (PROMOTED, security):** send a structurally valid but cryptographically garbage frame to `/wss/agent-events`. Evidence says it will be stamped "Signed" — confirm and fix under F3.
2. **Semantic degradation baseline (PROMOTED):** ingest a known corpus into AgentDB, measure the distance matrix under hash-placeholder embeddings. Quantifies how degraded RuVector memory recall currently is; sets the acceptance bar for the MiniLM fix.
3. Broker deadlock on mesh loss mid-judgment; BC20 URN collision from duplicate ACSP events; governance fork finality (stale Approve after Reject); two-tier pod write consistency on mid-write kill; WAC-vs-NIP-98 evaluation ordering — **run during Phase 4**.
4. Mooted by evidence: mesh-routing and NIP-42 probes (nothing to probe — the capabilities don't exist); hardware/physics validation (simulation-only confirmed).

---

## 7. Appendix — Method and Provenance

- **Coordination:** ruflo mesh `swarm_1783067180317_99wd065`; roles: `fable-queen` (strategy/synthesis only), `opus-gap-lead`, `sonnet-audit-lead`.
- **Workflows:** `wf_68bbcb7e-e09` (22 Opus agents: 9 ADR slices, 6 stub sweeps, 6 contract dimensions, 1 register re-verify; 815s) and `wf_f5eb0dce-998` (10 Sonnet agents: 7 repo audits, fixtures, harness, AQE fleet; 397s). 32/32 returned, zero errors.
- **GLM-5.2** (Z.AI): pass 1 blind-spot hypotheses + triage framework; pass 2 adversarial critique (three triage corrections adopted; sequencing insight adopted as Phase 0).
- **agentic-qe fleet** v3.11.3: ran health/init/quality/coverage/defect-predict; itself surfaced P0 tool defects (unscopeable `quality_assess`, invalid JSON from `aqe coverage`, inoperable coverage-gap analysis) — recorded as toolchain findings.
- **RuVector memory keys** (`project-state`): `visionflow-closeout-gap-analysis-mission-2026-07-03`, `glm52-blindspot-hypotheses-2026-07-03`, `sonnet-qe-audit-digest-2026-07-03`, `opus-gap-analysis-digest-2026-07-03`, `visionflow-closeout-final-design-2026-07-03`.
- **Caveat:** severity ratings are analyst judgements; the two-swarm convergence raises confidence but findings marked from a single source carry single-witness weight. Runtime behaviour (§6) remains unverified by construction.
