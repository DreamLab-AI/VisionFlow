# PRD: Ecosystem Alignment and Release Readiness

**Owner:** DreamLab AI
**Status:** Active
**Date:** 2026-05-22
**Version:** 1.1

## Purpose

Turn the VisionFlow ecosystem audit into an operational alignment program across VisionFlow, VisionClaw, agentbox, solid-pod-rs, nostr-rust-forum, and dreamlab-ai-website.

The ecosystem already has real shared contracts: `did:nostr`, Nostr event kinds, Solid pod storage, governance events, broker flows, and a shared fixture corpus. The remaining product need is to make compatibility, ownership, maturity, and release evidence explicit enough that downstream teams can ship against the ecosystem without reading every repository first.

## Problem

The repositories are directionally aligned, but the alignment is unevenly visible:

- Identity and pod work are concrete and broadly consistent.
- Governance event kinds exist in docs, code, config, and fixtures.
- Mesh federation is specified; nostr-rust-forum and dreamlab-ai-website default to federated mode, while agentbox and solid-pod-rs default to standalone. Runtime maturity varies by substrate.
- Version compatibility is partly documented, but coordinated release manifests are not yet the normal source of truth.
- Shared fixtures exist, but drift detection is not consistently treated as a release gate.
- IS-Envelope is specified and fixture-tested (VisionClaw ADR-075, 11 test vectors, JSON Schema), but the canonical ownership was not previously reflected in VisionFlow coordination docs.
- Historical PRDs and ADRs can read more complete than the current deployed posture.

## Goals

| Goal | Outcome |
|---|---|
| G1: Release compatibility manifest | Every coordinated release pins repo SHAs, branches, dirty state, and protocol compatibility claims |
| G2: Mesh maturity visibility | Each substrate reports whether mesh behavior is standalone, scaffolded, integrated, or federation-verified |
| G3: Protocol ownership registry | Shared contracts have exactly one canonical owner and named consumers |
| G4: End-to-end governance proof | A smoke test proves agentbox -> relay -> forum -> VisionClaw -> solid-pod-rs provenance flow |
| G5: Fixture sync gates | Shared protocol fixture drift fails CI or release qualification |
| G6: Pod tier clarity | Cloudflare, embedded, native, and git-capable pod tiers are documented with supported features |
| G7: Documentation status reconciliation | Active docs distinguish shipped, integrated, scaffolded, experimental, and historical claims |

## Non-Goals

- Implement full relay federation in this PRD.
- Replace repository-local PRDs, ADRs, or runbooks.
- Redesign the DreamLab website or product narrative.
- Migrate user data between pod tiers.
- Change Nostr, Solid, DID, or WAC protocol semantics.

## Users

| User | Need |
|---|---|
| Ecosystem maintainer | Know which repository versions and protocol claims can be shipped together |
| Agent runtime maintainer | Know which governance events and relay modes are safe to emit |
| Forum maintainer | Know which human-agent collaboration events must be rendered and verified |
| Pod maintainer | Know which Solid and provenance capabilities are required by upstream consumers |
| Website/operator | Present ecosystem status without overstating maturity |
| Contributor | Find canonical contracts before making cross-repo changes |

## Functional Requirements

| ID | Requirement | Acceptance Criteria |
|---|---|---|
| F1 | Publish ecosystem release manifests | `docs/releases/` contains a committed candidate or release manifest for coordinated releases using `ecosystem-release.schema.json` |
| F2 | Maintain compatibility matrix | `docs/architecture/compatibility-matrix.md` records identity, mesh, pod, governance, deployment, and test posture by repo |
| F3 | Maintain mesh status table | Each substrate has a status of `standalone`, `scaffolded`, `integrated`, or `federation-verified`, with evidence links |
| F4 | Pin protocol owners | Shared contracts including `did:nostr`, Agent Control Surface kinds, IS-Envelope, pod auth, and fixture schemas have one owner and listed consumers |
| F5 | Define governance smoke test | `docs/protocol/mesh-smoke-test.md` specifies event path, assertions, blockers, and required services |
| F6 | Gate fixture drift | Fixture checksum verification is implemented via `CHECKSUMS.txt` (SHA-256) and `UPSTREAM_PINS.md` in agentbox. Release qualification includes cross-repo fixture parity checks. |
| F7 | Document pod tiers | Pod capability docs identify Cloudflare, embedded local, native Rust, and git-capable behavior |
| F8 | Reconcile historical claims | Active docs link to status reconciliation when older PRDs/ADRs describe deferred or partial implementation |

## Non-Functional Requirements

| ID | Requirement | Acceptance Criteria |
|---|---|---|
| N1 | Traceability | Every compatibility claim links to a repository doc, config, test, or code path |
| N2 | Repeatability | Release manifest generation is scriptable from the local workspace |
| N3 | Minimal duplication | Repository-local docs own implementation detail; VisionFlow owns cross-repo coordination |
| N4 | Conservative maturity language | Docs do not call a feature shipped or federation-complete without runtime evidence |
| N5 | Auditability | Release candidates preserve the exact repo SHAs used for verification |

## Gap Register

| Gap | Impact | Target Resolution |
|---|---|---|
| No committed coordinated release manifest | Release manifest schema and generator script exist (`scripts/generate-release-manifest.sh`); candidate manifests are generated but not yet part of release qualification | Add candidate manifests during release qualification |
| Mesh defaults vary by substrate | nostr-rust-forum and dreamlab-ai-website are federated by default; agentbox and solid-pod-rs default standalone. Federation claims should be substrate-specific. | Track mesh status per substrate and only promote after smoke proof |
| ✓ IS-Envelope ownership resolved | VisionClaw owns the spec (ADR-075), JSON Schema, and 11 fixture vectors. agentbox decodes at runtime. Ownership is no longer implicit. | Resolved — update coordination docs to reflect VisionClaw ownership |
| Fixture corpus lacks universal gates | Shared protocol changes can silently diverge | Add fixture sync checks to CI/release gates |
| Pod tier capability matrix is scattered | solid-pod-rs alpha.15 ships native mesh with CORS and PSK admin provisioning. Cloudflare Workers tier has documented feature differences. Two-tier behavior is clearer but not fully reconciled. | Publish tier matrix and migration notes |
| Historical docs are richer than current runtime | Readers can confuse aspiration with shipped behavior | Keep status reconciliation active and linked |
| ✓ Ontology bridge reflected in coordination docs | agentbox ships 12 MCP tools proxying to VisionClaw Oxigraph SPARQL (ontology-bridge.js). This cross-substrate integration is absent from compatibility and alignment docs. | Resolved — added to compatibility matrix, ADR-002, and DDD |
| Judgment Broker is 65% implemented as a distributed system | The decision loop is closed on the forum side (nostr-rust-forum governance.rs, 1015 lines). agentbox relays events and bridges VisionClaw enrichment review. Critical gaps: decision→agent application (handleGovernanceDecision missing), agent MCP tools for panel publishing, BrokerActor not on main. | Close the 35% gap per PRD-judgment-broker.md milestones |

## Milestones

| Milestone | Exit Criteria |
|---|---|
| M0: Documentation baseline | PRD, ADR, DDD, compatibility matrix, status reconciliation, and smoke-test plan are linked from the docs index |
| M1: Owner registry | Shared protocol owners and consumers are listed in the compatibility matrix or a dedicated registry |
| M2: Release manifest candidate | A candidate manifest is generated and reviewed for the current workspace |
| M3: Fixture gate | At least one CI or local release command verifies shared fixture parity |
| M4: Governance smoke proof | The full agentbox -> relay -> forum -> VisionClaw -> pod path runs with captured evidence |
| M5: Federation qualification | Mesh status for all required substrates reaches `federation-verified` |

## Success Metrics

- 100% of cross-repo compatibility claims have evidence links.
- 100% of coordinated releases include a release manifest.
- Fixture drift is detected before release.
- Mesh status language matches runtime configuration.
- A new contributor can identify canonical protocol owners from VisionFlow docs without inspecting every repository.
- Ontology bridge integration reflected in compatibility matrix with evidence links.

## Open Questions

- IS-Envelope canonical ownership resolved: VisionClaw (ADR-075). agentbox implements runtime decode; solid-pod-rs and nostr-rust-forum consume fixture vectors.
- Which CI system should run fixture parity checks across externally mounted repositories?
- What is the minimum acceptable relay topology for `federation-verified`?
- Should DreamLab website deployments consume only released manifests, or are candidate manifests acceptable for staging?
