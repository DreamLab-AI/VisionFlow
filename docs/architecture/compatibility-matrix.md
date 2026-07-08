# Ecosystem Compatibility Matrix

**Status:** Docs, metadata, code spot-check, and runtime research synthesis
**Date:** 2026-05-22, reconciled against code 2026-07-03

This matrix records the current compatibility posture verified by cross-repo runtime research, not only sibling documentation. The 2026-07-03 pass corrected the mesh, governance, identity and ontology-bridge rows against merged code; the full closeout analysis is in [`docs/closeout/final-design.md`](../closeout/final-design.md).

| Area | VisionClaw | agentbox | nostr-rust-forum | solid-pod-rs | dreamlab-ai-website | Compatibility status |
|---|---|---|---|---|---|---|
| Identity | NIP-98, NIP-07, DID:Nostr, Solid/WAC documented; enterprise SSO remains a known future path. NIP-26 delegation is **not implemented** (deferral comments only) | Bootstrap BIP-340 key becomes canonical `did:nostr`; accepted by relay, pods, WAC, URNs | Passkey PRF derives Nostr keys; NIP-98 across workers; NIP-42 answering exists only in the WASM browser client, not the edge relay | Foundation for DID:Nostr, NIP-98, Solid-OIDC, Tier 1/Tier 3 DID resolution | WebAuthn PRF + NIP-98; configured agent identities | NIP-98 is reimplemented independently in four repos with divergent URL-matching/replay semantics (G3 open). `did:nostr` is canonicalised on the Multikey form by ADR-125. NIP-26 delegation is unimplemented everywhere; replay protection exists only in the forum + CF pod tier |
| Mesh | ADR-073 describes client/subscriber mode by default unless a local relay is added | Standalone default (`[mesh] mode = "standalone"`); embedded relay at `:7777` with pod-bridge; federated/client modes available | `nostr-bbs-mesh` is **scaffold-only** (no `MeshTransport` impl, not a dependency of the relay-worker; lands Sprint v12+). The relay gates by pubkey whitelist, **not** NIP-42 (`auth_required:false`). Peer discovery and IS-Envelope routing are not shipped | Standalone with an embedded NIP-01 relay (no NIP-42 AUTH); "native mesh in alpha.15" is not a real feature — federation is README prose only | Federated architecture; CF Workers relay fan-out to public relays for censorship resistance | **Standalone-first is the supported deployment mode** (closeout FREEZE). Only dreamlab-ai-website defaults to fan-out — the federated-by-default tally is **1/4**, not 2/4 (agentbox, solid-pod-rs and nostr-rust-forum all default standalone). Mesh federation is designed, not shipped |
| Pod | Embedded `solid-pod-rs` at local `:8484`; WAC against `did:nostr` | Embedded Solid pod and native pod overlay via Cloudflare Tunnel | CF Worker pod tier mirrors Solid/JSS; native tier supports git via `solid-pod-rs-server` | Canonical LDP/WAC/WebID/auth/storage implementation | CF pod tier plus native agentbox pod tier for git | Strongest compatibility area; two-tier behavior is the main gap |
| Governance | Merged main runs an **inline decide/inbox handler + ADR-110 ElevationActor**, not the distributed BrokerActor (that lives on the unmerged `crashbug` branch only). The `POST /api/enrichment-proposals/{id}/decide` endpoint performs a real fenced Oxigraph write on attributed approval and flips `writeback_committed`. Agent Control Surface event kinds defined | Publishes and subscribes to kinds 31400-31405 via nostr-bridge; **12-tool** ontology bridge proxies SPARQL to VisionClaw Oxigraph; `handleGovernanceDecision` applies returned decisions with PROV-O URNs | Full implementation of kinds 31400-31405 in `governance.rs`; primary governance UI/API: agent registry, cases, roles, signed responses | Access-control foundation, not workflow owner | Governance dashboard at `/governance` renders kinds 31400-31405; NIP-98 signed responses; feature-gated | Agent Control Surface implemented in forum + website; agentbox publishes/subscribes and applies decisions. Governance is **shipped inline on main** (decision application + KG write-back live); the distributed BrokerActor is an unmerged alternative. Remaining wire: `ConceptElevated` closing event (elevation terminates at PR creation, not merge) |
| Deployment | Docker Compose multi-profile and docs CI; productionisation PRDs still list hardening gaps | Nix-built multi-arch image, runtime contracts, image scan/SBOM, flake checks | Cloudflare Workers kit with CI/audit docs | Server/library deployment docs; CI scope may miss sibling crates | GitHub Pages + five Workers; kit clone/overlay deployment | Usable but split; kit pinning and cross-repo version policy remain weak |
| Ontology Bridge | Oxigraph SPARQL store; OWL 2 EL reasoning (Whelk-rs); 82 CUDA kernels across 9 `.cu` files (5,854 LOC); canonical knowledge graph | **12 MCP tools** (`mcp/servers/ontology-bridge.js`) proxy SPARQL queries to VisionClaw; gated via `[skills.ontology]` in `agentbox.toml` | Not applicable (forum does not query KG directly) | Not applicable (pod stores data, does not query KG) | Not applicable (website does not query KG) | Shipped cross-substrate integration: agentbox → VisionClaw. **Bidirectional** (governed propose + query): `ontology_propose` runs a Whelk EL++ consistency gate then opens a GitHub PR for human review — axiom-submission governance is implemented, not deferred |
| Tests/Ops | Build/test commands and known issues documented; master cross-substrate fixtures relocated to `tests/fixtures/` on 2026-06-29 (13 protocol fixtures + 14 schemas — the old `docs/specs/fixtures/` path is gone); IS-Envelope spec (ADR-075) | Upstream fixture vectors under `tests/contract/upstream_vectors/` (13 JSON fixtures / 126 vectors / 30 checksummed files) with SHA-256 checksums and upstream pins; adapter contract test suites | Workspace tests, audit workflow, Worker build docs; fixture copies under `tests/fixtures/` — the forum's own `CHECKSUMS.txt` is **red** (5 files fail, manifest not regenerated after the 2026-06-15 refresh) | JSS parity tests/docs; fixture copies in `solid-pod-rs-didkey` and `solid-pod-rs-nostr` (did-doc frozen at the pre-refresh state, masked by passing self-checksums) | README/test docs conflict; some gates advisory | `is-envelope-v1.json` is byte-identical across all five locations, but `did-doc-conformance.json` has **4-way drift** (7/8/9/7 vectors, two testing a superseded DID spec) — the "checksum verification is the strongest signal" claim does not hold for the security-sensitive DID fixture. `sync-fixtures.sh` in all three consumers still points at the deleted `docs/specs/fixtures/` path |

## Minimum Compatibility Record

Every coordinated release should publish:

| Field | Required value |
|---|---|
| Repo SHAs | VisionFlow, VisionClaw, agentbox, solid-pod-rs, nostr-rust-forum, dreamlab-ai-website |
| Identity contract | DID method, public-key form, verification suite string |
| Relay contract | NIP list, write auth, read auth, peer relay list, allowed pubkeys |
| Pod contract | Pod tier, WAC mode, NIP-98 replay behavior, git/export capability |
| Governance contract | Supported Agent Control Surface kinds and schema versions |
| Test evidence | Fixture sync status, smoke test status, audit status |
| Operations | Health endpoints, backup/restore plan, DR owner |

## Release Manifest

Generate the current local repository SHA set with:

```sh
scripts/generate-release-manifest.sh > docs/releases/ecosystem-release.local.json
```

See [Release Manifests](../releases/README.md) for the schema and promotion rules.

## Harness Coverage

Per ADR-004, each substrate's guide-sensor pairing status is tracked here. **This table is generated from `scripts/harness-audit.sh`** (run 2026-07-03) — do not hand-edit; re-run the script and paste its output. See `docs/engineering/templates/` for full template definitions.

Two distinct metrics are reported, and they are not the same thing. **Paired** counts declared guide→sensor cross-references inside a template (a coverage-of-intent measure). **Source-backed** counts the controls whose `source` field resolves to a real artifact in the named substrate today; the remainder carry `source_status: planned` (declared but not yet implemented). A template can be 100% paired while some controls are still planned, so both columns are published to keep the metric honest.

Maturity labels below are the templates' own `maturity` fields — not hand-typed. `governance-decision` is a single shared template covering agentbox, nostr-rust-forum and VisionClaw (`structure.substrates`); it is listed once against all three rather than credited only to the forum.

| Substrate(s) | Harness Template | Guides | Sensors | Paired | Ratio | Source-backed | Maturity |
|---|---|---|---|---|---|---|---|
| agentbox | `agentbox-agent-task` | 5 | 5 | 5 | 100% | 8/10 (2 planned) | standalone |
| VisionClaw | `visionclaw-enrichment` | 5 | 5 | 5 | 100% | 8/10 (2 planned) | standalone |
| agentbox + nostr-rust-forum + VisionClaw | `governance-decision` (shared) | 5 | 5 | 5 | 100% | 9/10 (1 planned) | integrated |
| solid-pod-rs | `pod-mutation` | 5 | 5 | 5 | 100% | 8/10 (2 planned) | standalone |
| dreamlab-ai-website | — | 0 | 0 | 0 | — | — | planned |

**Totals:** 20 guides / 20 sensors / 20 paired = 100.0% pairing; 33/40 controls source-backed (7 planned). Audit verdict: PASS (target 80%).

### Planned controls (declared, not yet source-backed)

These carry `source_status: planned` in the templates and are tracked for implementation:

- `agentbox-agent-task`: `hooks-validate-trajectory` (the ADR-004 D3 `hooks.validate` phase, explicitly deferred), `slo-ci-gate` (no `adapter-slo.yml`; the SLO assertions are `it.todo()` stubs)
- `governance-decision`: `governance-ui-rendering` (no dedicated render test; the governance UI lives in the WASM client)
- `pod-mutation`: `block-trail-rules` + `taproot-verification` (no Taproot implementation exists in solid-pod-rs — documentation only)
- `visionclaw-enrichment`: `enrichment-proposal-schema` (no standalone JSON Schema fixture yet), `provenance-chain-integrity` (PROV-O is not yet reified on the ingest path — see closeout T7)

Fitness-gate ADR citations are repo-qualified to avoid ADR-number collisions: agentbox SLO thresholds are **agentbox** ADR-005 (pluggable-adapter architecture), a different document from **VisionFlow** ADR-005 (mandate-at-grant governance).

Run `scripts/harness-audit.sh` for current status.

## Gap-Close Sprint — P0 Item Tiers

The area rows above record per-area compatibility posture. Per ADR-002 and the
Gap-Close Canon PRD, each gap-close item is also reflected here at its **evidenced
tier** once its wave closes. This table is the P0 wave; it is authoritative to
[`docs/registers/gap-register-v1.1.md`](../registers/gap-register-v1.1.md) (the
immutable status register cut at the P0→P1 boundary). Tiers use the ADR-002
vocabulary; `scaffolded` here means the mechanism is complete and its canary is
proven locally but has **not** fired in a live session (a canon loop item is not
scored `integrated` until its canary fires live — DDD Gap-Close Invariant 4).

| Item | Owner(s) | Closure SHA(s) | Evidenced tier | Canary state |
|---|---|---|---|---|
| RES-a KG liveness + canary harness | VisionClaw | `6f4eb1b0a`, `1492bc17b` | `integrated` | Harness live; Nostr-tap round-trip PENDING-LIVE |
| RES-b diagram render gate | VisionFlow (canon) | `f72d173cd` | `scaffolded` | `CANARY-CANON-DIAGRAM` proven locally; live CI fire PENDING |
| COM-13 agent disclosure | nostr-rust-forum (+ canon norm) | forum `7157a92`, `fb7826859`; canon `identity-spine.md` | Forum impl `integrated`; canon clause `standalone` | Badge census-verified; live render PENDING-LIVE |
| COM-14 `did:nostr` keying | VisionClaw + agentbox | VC `4a595cc8f`; AB `6189f47d` | `integrated` | Live Schnorr round-trip PENDING-LIVE |
| REC-2 broker kernel + case queue | VisionClaw | `c9f2e3539` | `integrated` (P0 slice) | Case e2e PENDING (COM-15 batch) |
| D5 MCP-status honesty | VisionClaw | `6f4eb1b0a` | `integrated` | Real WS + `check_mcp_metrics`; landed |
| REC-1 governed-writeback floor | VisionClaw · solid-pod-rs · forum | `6f4eb1b0a`; solid-pod `791977a` | `integrated` / `reconciled` | Route-guard tests green; PATCH + NIP-42 reconciled pre-sprint |
| RES-d drift counter (canon) | VisionFlow (canon) | this wave (P1) | `scaffolded` | `CANARY-CANON-DRIFT` proven locally; live CI fire PENDING |

The forward-chained corrections behind REC-1a/1b, PATCH, NIP-42, D1 and D5 are
recorded in `gap-register-v1.1.md` §"Forward-chained corrections", not edited into
the v1.0 inventory in place (register immutability, DDD Gap-Close Invariant 5).

## Gap-Close Sprint — P1/P2 Item Tiers

The P1 and P2 waves closed the remaining 41 line items. This table reflects them
at their evidenced tiers and is authoritative to
[`docs/registers/gap-register-v1.2.md`](../registers/gap-register-v1.2.md) (the
immutable status register cut at the P1/P2 boundary, superseding v1.1). Tiers use
the ADR-002 vocabulary. `standalone` means the mechanism runs within one substrate
but is not yet cross-substrate-proven; `scaffolded` means the mechanism is complete
and proven locally but its live gate has not fired; a canon loop item is not scored
`integrated` until its canary fires in a live session (DDD Gap-Close Invariant 4).
The P0 table above is authoritative to v1.1 and is left in place; the two do not
overwrite one another.

| Item | Owner(s) | Closure SHA(s) | Evidenced tier | Canary state |
|---|---|---|---|---|
| F1 member read-only governance view | nostr-rust-forum | `0b8a1c`, `6986276` | `integrated` | fired-in-test |
| F3/F7 via COM-16 graduated escalation | nostr-rust-forum | `0b8a1c`, `6986276` | `integrated` | fired-in-test |
| F4/F5 via COM-17 decision-audit API | nostr-rust-forum | `0b8a1c`, `6986276` | `integrated` | fired-in-test |
| F6 supersession authority | nostr-rust-forum · canon | forum `35dbb1b`, `696fd233`; canon `7513e58` | `scaffolded` (canon-gated) | fired-in-test |
| F8 roster admin UI | nostr-rust-forum | `0b8a1c`, `6986276` | `integrated` | fired-in-test |
| F9 cross-relay federation | VisionFlow (canon) | `7513e58` (fork record) | `planned` | parked (criteria unmet) |
| F10 multi-agent social influence | VisionFlow (canon) | `7513e58` | `standalone` (position authored) | n/a |
| D1 embodiment beam liveness | VisionClaw | `4aca6f729` | `integrated` (wired at boot; drift correction) | PENDING-LIVE (beam traffic) |
| D2/D3/D8 steering + case surface + observability | VisionClaw | `e0f582403`, `453bd41b1`, `b6c1c43b5` | `integrated` | fired-in-test; case e2e PENDING-LIVE; MCP-native interrupt a documented boundary |
| D6 PTT directed at selected actor (via COM-15) | VisionClaw · agentbox | `f6e1d58f9`; AB `9673624`, `1fc47a14` | `integrated` | PENDING-LIVE (voice e2e) |
| D7 pre-action intent legibility | VisionClaw · canon | `774ffa05e`; canon `2511f994` | `standalone` | fired-in-test |
| M1–M6 / COM-18 headset copresence + intervention | VisionClaw (Godot XR) | `57b32faee`, `0f3a1b60c`, `348172d7d` (merged `485bb8f9f`) | `standalone` (no headset in box) | PENDING-LIVE (`M4-RAY`, `COM18-INTERV`; `M1-HUD` rides P0) |
| V1/M5 via COM-15 PTT governed loop | VisionClaw · agentbox | `f6e1d58f9`; AB `9673624`, `1fc47a14` | `integrated` | PENDING-LIVE (voice e2e) |
| V2 multi-agent voice addressing | VisionFlow (canon) | `7513e58` | `standalone` (position authored) | n/a |
| V3 conversational grounding/repair | VisionClaw | `774ffa05e` | `standalone` | fired-in-test |
| V4 voice-docs honesty | VisionClaw | `774ffa05e` | `integrated` (docs) | n/a |
| REC-2 broker case queue (P1 slice) | VisionClaw | `e0f582403` | `integrated` | case e2e PENDING-LIVE |
| REC-3 contextual transaction cost | VisionClaw · agentbox | `4aca6f729`, `453bd41b1`; AB `9673624`, `ceb3401b` | `integrated` | fired-in-test |
| REC-4 four-KPI dashboard (ADR-043) | VisionClaw | `4aca6f729` | `integrated` (2 KPIs live; 2 honest pending tiles) | fired-in-test |
| REC-5 MAST failure telemetry | agentbox · VisionClaw | AB `9673624`, `1fc47a14`, `ceb3401b` | `integrated` | fired-in-test |
| REC-6 escalation-default authority | agentbox | `9673624`, `1fc47a14`, `ceb3401b` | `standalone` (re-tiered honestly) | fired-in-test |
| REC-7 outcome learning | agentbox | `9673624`, `1fc47a14` | `standalone` (consumers gated pending floor) | fired-in-test |
| REC-8 orchestration diversity | agentbox | `9ebff750`, `a8dd21a5` | `standalone` (re-tiered) | fired-in-test |
| REC-9 provenance-to-pocket | agentbox | `9ebff750`, `a8dd21a5` | `standalone` (re-tiered) | fired-in-test |
| REC-10 Insight Ingestion Loop v1 | VisionClaw · agentbox · forum | `1c462f492` | `integrated` | fired-in-test |
| REC-11 data-moat unified trace | VisionClaw · solid-pod-rs | `1c462f492`; SP `40043b0` (+WP-2) | `standalone` (pod `_prov/` contract-only) | fired-in-test |
| REC-12 kit cutover + external pilot | dreamlab-ai-website | `ca06ab3` | `standalone` (pilot blocked-operational) | registered (D3 key-split runbook) |
| COM-15 PTT voice-to-actor loop | VisionClaw · agentbox | `f6e1d58f9`; AB `9673624`, `1fc47a14` | `integrated` | PENDING-LIVE (voice e2e) |
| COM-16 graduated escalation + fatigue | nostr-rust-forum | `0b8a1c`, `6986276` | `integrated` | fired-in-test |
| COM-17 decision-audit + trust-calibration | nostr-rust-forum | `0b8a1c`, `6986276` | `integrated` | fired-in-test |
| COM-18 headset intervention + identity | VisionClaw (Godot XR) | `57b32faee`, `0f3a1b60c`, `348172d7d` | `standalone` (no headset) | PENDING-LIVE (`COM18-INTERV`) |
| RES-c solid-pod-rs diagram refresh | solid-pod-rs | `40043b0` (+WP-2) | `integrated` (9 diagrams re-rendered) | fired-in-test |
| RES-d self-description drift counter + CI | VisionFlow (canon) | `b47c5fd`; sources VC `4aca6f729`, AB `d13f8688` | `scaffolded` | PENDING-LIVE (`CANARY-CANON-DRIFT` CI fire) |
| RES-e Wardley export quality | VisionFlow (canon) | `7513e58` | `integrated` | n/a |

The P1/P2 forward-chained corrections (REC-2/ADR-041 "Implemented" claim, the D2
MCP-native interrupt boundary, and the REC-6/REC-8/REC-9/F6 tier re-derivations)
are recorded in `gap-register-v1.2.md` §"Forward-chained corrections", not edited
into v1.1 or the v1.0 inventory in place. The pending-live canaries are batched to
one stack-up session at sprint end (a recorded deviation from strict wave gating,
because no live stack, headset or GitHub Actions push runs in the build
container); see `gap-register-v1.2.md` §"Pending-live-session batch".
