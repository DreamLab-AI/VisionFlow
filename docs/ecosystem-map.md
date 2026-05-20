# VisionFlow Ecosystem Map

**Status:** Docs-only synthesis
**Date:** 2026-05-20
**Scope:** Recently modified sibling repositories under `/home/devuser/workspace`

This document maps the VisionFlow ecosystem from repository documentation only. It does not verify implementation code. When a gap is listed, it means the docs either explicitly identify it or describe an architecture whose required counterpart is absent, deferred, or inconsistent in the available docs.

## Source Set

| Path | Interpreted repository | Documentation used |
|---|---|---|
| `../VisionFlow` | VisionFlow | README, website PRD/ADR/DDD |
| `../project` | VisionClaw mount | README, PRD-010/014/015, ADRs, DDD/integration research docs |
| `../agentbox` | agentbox | README, developer ecosystem/identity/sovereign mesh docs |
| `../solid-pod-rs` | solid-pod-rs | README, ecosystem integration, parity/gap analysis |
| `../nostr-rust-forum` | nostr-rust-forum | README, architecture, consumer surface map, ADR-086/087/089/093 |
| `../dreamlab-ai-website` | DreamLab deployment | README, deployment and forum-consumer docs |

## System Shape

VisionFlow is not a single executable. It is a coordination architecture that emerges from five product substrates and one branded deployment:

| Substrate | Primary responsibility | Boundary |
|---|---|---|
| VisionFlow | Ecosystem narrative, public positioning, repository map, shared coordination model | Documentation and website surface |
| VisionClaw | Knowledge engineering: OWL 2 EL reasoning, GPU graph physics, XR, ontology MCP tools, Judgment Broker | GPU host / graph backend / semantic workbench |
| agentbox | Reproducible sovereign agent runtime with Nix, tools, skills, privacy filtering, Nostr/Solid identity | Agent container and harness |
| solid-pod-rs | Solid/JSS foundation: LDP, WAC, NIP-98, DID:Nostr, WebID, OIDC, git pods | Shared protocol library and native pod server |
| nostr-rust-forum | Governance UI and relay kit: passkey auth, Cloudflare Workers, Agent Control Surface events | Human decision surface and Nostr relay edge |
| dreamlab-ai-website | Branded DreamLab deployment and operator overlay for the forum kit | Public site and Cloudflare deployment config |

The common primitive is `did:nostr:<hex-pubkey>`. Docs consistently describe it as the identity used for relay authentication, HTTP request signing, WAC ACL subjects, provenance, and cross-substrate routing.

## Core Flows

### Human-Governed Agent Action

1. An agent in agentbox publishes an Agent Control Surface event, normally one of kinds `31400-31405`.
2. The Nostr relay mesh routes that event to the forum.
3. The forum renders the event as a governance panel or action request.
4. A human signs an approval/rejection response.
5. VisionClaw's Judgment Broker and related enrichment/write-back flows use the decision as the control point before mutation.

### Sovereign Data Access

1. A user or agent signs an HTTP request using NIP-98.
2. A pod tier verifies the signature against the same secp256k1 pubkey used by `did:nostr`.
3. WAC evaluates access against that identity.
4. The resource is served from either Cloudflare Workers storage or a native `solid-pod-rs` server.

### Knowledge Ingestion and Provenance

1. VisionClaw ingests knowledge from Logseq/GitHub and pod-backed sources.
2. OWL reasoning and graph physics derive semantic structure.
3. Agents propose enrichments or actions.
4. Human decisions gate mutation.
5. Provenance beads and URNs preserve attribution.

## Gap Register

### G1: The Umbrella Docs Are Sparse

The VisionFlow README references richer documentation paths such as `docs/architecture/repository-map.md`, `docs/protocol/identity-spine.md`, and licensing architecture. The local checkout only contained website PRD/ADR/DDD docs before this synthesis. That creates a trust gap between the top-level narrative and navigable technical reference.

**Impact:** New contributors cannot easily find the canonical ecosystem contract from the VisionFlow repo alone.

**Next action:** Keep this document, the repository map, and the identity spine as the top-level docs entry points.

### G2: Mesh Federation Is Still Partly Aspirational

VisionClaw PRD-010 describes the target DID:Nostr mesh, relay federation modes, NIP-42 write gates, NIP-26 delegation, DID service endpoints, and IS-Envelope routing. PRD-014 explicitly defers major pieces: IS-Envelope runtime, full relay mesh, NIP-26 unification, distributed tracing, shared type crate, coordinated releases, and cross-substrate integration tests.

**Impact:** The ecosystem can be described as federation-ready, but not yet federation-complete.

**Next action:** Track a compatibility matrix for relay capabilities, supported NIPs, event kinds, and DID document service endpoints per substrate.

### G3: Protocol Implementations Are Duplicated

VisionClaw PRD-015 identifies cross-substrate duplication in NIP-98 auth, Solid pod clients, DID:Nostr resolution, WAC/ACL evaluation, Nostr key management, and URN minting.

**Impact:** Security fixes and protocol behavior can diverge across repos. This is especially risky for auth, replay protection, and DID verification.

**Next action:** Promote shared crates/contracts for NIP-98, DID:Nostr, WAC/pod client behavior, and ecosystem event types.

### G4: VisionClaw Has Documented Integration Debt

VisionClaw docs identify several still-important risks: URI resolver redirects to missing routes, BC20 anti-corruption layer described as paper-only in PRD-010, historical multi-keypair drift, missing or incomplete NIP-42 support, auth hardening gaps, dead/stub code, and parallel service implementations.

**Impact:** VisionClaw is the semantic center of the ecosystem, so gaps there block end-to-end provenance and governance flows.

**Next action:** Reconcile PRD-010/014/015 status against current code and update docs with completed versus open items.

### G5: Cloudflare Workers Portability Drives Duplication

solid-pod-rs docs and forum ADRs repeatedly identify the same structural issue: native/Tokio features cannot be linked directly into Cloudflare Workers. The forum therefore mirrors or reimplements pod behavior for edge deployment.

**Impact:** The forum gets edge deployment benefits, but duplicates protocol logic and cannot expose every native pod feature.

**Next action:** Decide whether to extract no-Tokio `core` surfaces in solid-pod-rs or accept permanent two-tier behavior.

### G6: Two-Tier Pods Add Operational Complexity

nostr-rust-forum ADR-093 defines a hybrid pod architecture: Cloudflare Workers pods for edge LDP/R2 and native agentbox-hosted `solid-pod-rs-server` pods for git/app capabilities.

**Impact:** Users can land on different pod tiers with different capabilities. Operators must manage native provisioning, Cloudflare Tunnel, PSK rotation, and tier-aware WebID routing.

**Next action:** Maintain a pod tier matrix and a migration story for users moving from CF-tier pods to native git-capable pods.

### G7: agentbox Mesh Docs Need Runtime Status Reconciliation

agentbox docs describe the sovereign relay, pod inbox bridge, identity root, and Solid pod integration as core architecture. VisionClaw PRD-010 and earlier ecosystem audits call out relay exposure, boot wiring, allowlist, and identity bootstrap issues as gaps.

**Impact:** It is unclear from docs alone which agentbox mesh items are currently shipped, fixed, or still deferred.

**Next action:** Add an agentbox mesh status table: embedded relay exposure, bridge boot wiring, NIP-42 support, allowed pubkey source, DID document relay endpoint, and multi-agent identity status.

### G8: DreamLab Website Is a Consumer Overlay, Not a Protocol Source

dreamlab-ai-website owns branding, static React pages, Cloudflare config, and forum operator overlay. It consumes nostr-rust-forum rather than owning forum source.

**Impact:** Deployment behavior depends on kit pinning, workflow clone behavior, Cloudflare resource mapping, and config compatibility.

**Next action:** Pin and publish a kit compatibility record for every production deployment.

## Recommended Roadmap

| Priority | Work | Why |
|---|---|---|
| P0 | Security/auth status reconciliation across PRD-014 gaps | Prevents publishing an architecture that implies stronger guarantees than deployed systems provide |
| P0 | Cross-substrate compatibility matrix | Makes mesh readiness explicit and testable; see [Compatibility Matrix](architecture/compatibility-matrix.md) |
| P1 | Shared protocol contracts for NIP-98, DID:Nostr, WAC, IS-Envelope | Reduces drift and duplicated security-sensitive logic |
| P1 | End-to-end governance smoke test: agentbox -> relay -> forum -> VisionClaw -> pod/provenance | Validates the central VisionFlow claim |
| P1 | solid-pod-rs CF Workers portability decision | Determines whether duplication is temporary or permanent |
| P2 | Coordinated release/version policy | Lets consumers know which repo versions work together; see [Roadmap](roadmap.md) |
| P2 | Unified operations docs: health, backup, DR, pod tiers, relay status | Moves the ecosystem from impressive components to operable system |

## Open Questions

1. Which repository owns the canonical IS-Envelope schema and event kind registry?
2. Is `did:nostr` resolution canonicalized in solid-pod-rs, forum core, or a new shared crate?
3. Are Cloudflare Workers pods and native pods expected to converge, or remain separate tiers?
4. What is the minimum supported deployment: single operator, team, or cross-organization federation?
5. What exact repo/version set defines the current production DreamLab deployment?
