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
