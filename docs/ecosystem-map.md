# VisionFlow Ecosystem Map

**Status:** Docs plus targeted code/manifest spot-checks, updated with runtime research 2026-05-22
**Date:** 2026-05-22
**Scope:** Repositories mounted under `/home/devuser/workspace`

This document maps the VisionFlow ecosystem from repository documentation, with targeted implementation checks for the integration claims that affect cross-repo alignment. When a gap is listed, it means the docs or spot-checked code/manifests identify an absent, deferred, or inconsistent counterpart.

## Source Set

| Path | Interpreted repository | Documentation used |
|---|---|---|
| `../VisionFlow` | VisionFlow | README, website PRD/ADR/DDD |
| `../project` | VisionClaw mount | README, PRD-010/014/015, ADRs, DDD/integration research docs |
| `../project/agentbox` | agentbox | README, developer ecosystem/identity/sovereign mesh docs, relay/pod bridge code |
| `../solid-pod-rs` | solid-pod-rs | README, ecosystem integration, parity/gap analysis |
| `../nostr-rust-forum` | nostr-rust-forum | README, architecture, consumer surface map, ADR-086/087/089/093 |
| `../dreamlab-ai-website` | DreamLab deployment | README, deployment and forum-consumer docs |

## System Shape

VisionFlow is not a single executable. It is a coordination architecture that emerges from five product substrates and one branded deployment:

| Substrate | Primary responsibility | Boundary |
|---|---|---|
| VisionFlow | Ecosystem narrative, public positioning, repository map, shared coordination model | Documentation and website surface |
| VisionClaw | Knowledge engineering: OWL 2 EL reasoning, GPU graph physics, XR, 7 native MCP ontology tools, IS-Envelope spec owner (ADR-075), Judgment Broker (distributed: enrichment gating, BrokerActor on crashbug branch) | GPU host / graph backend / semantic workbench |
| agentbox | Reproducible sovereign agent runtime with Nix, 90+ skills, 180+ MCP tools, 10-tool ontology bridge to VisionClaw SPARQL, browser setup wizard, privacy filtering, Nostr/Solid identity | Agent container and harness |
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
5. VisionClaw's Judgment Broker and related enrichment/write-back flows use the decision as the control point before mutation. (Judgment Broker is 65% implemented as a distributed system; the decision loop is closed on the forum side, decision application to agents is the critical gap)

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

### G2: Mesh Federation Maturity Varies by Substrate

nostr-rust-forum (3.0.0-rc11) defaults to federated NIP-05 mode. dreamlab-ai-website uses federated CF Workers relay fan-out. agentbox and solid-pod-rs default to standalone. VisionClaw PRD-010 describes the target mesh; PRD-014 defers some pieces (NIP-26 unification, distributed tracing, shared type crate).

**Impact:** The ecosystem is federation-capable but maturity is substrate-specific, not uniform.

**Next action:** The compatibility matrix now tracks mesh status per substrate. Promote agentbox and solid-pod-rs to federated defaults when smoke test evidence exists.

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

### G7: agentbox Mesh Is Real but Still Needs Runtime Status Reconciliation

agentbox docs describe the sovereign relay, pod inbox bridge, identity root, and Solid pod integration as core architecture. Spot checks found concrete implementation surfaces, including `mcp/nostr-bridge/relay-consumer.js`, `management-api/adapters/pods/local-solid-rs.js`, `management-api/routes/broker-bridge.js`, and manifest settings for governance kinds `31400-31405`. VisionClaw PRD-010 and earlier ecosystem audits still call out relay exposure, boot wiring, allowlist, and identity bootstrap issues as historical or unresolved gaps depending on deployment mode.

**Impact:** agentbox is not merely aspirational, but the docs still need a single status table that distinguishes shipped loopback/private-mesh behavior from full federated operation.

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

## Recently Resolved

- **IS-Envelope canonical ownership:** Resolved. VisionClaw owns the spec (ADR-075, JSON Schema, 11 test vectors). Runtime consumers: agentbox, solid-pod-rs, nostr-rust-forum. Event kind registry remains unowned.

## Open Questions

1. Is `did:nostr` resolution canonicalized in solid-pod-rs, forum core, or a new shared crate?
2. Are Cloudflare Workers pods and native pods expected to converge, or remain separate tiers?
3. What is the minimum supported deployment: single operator, team, or cross-organization federation?
4. What exact repo/version set defines the current production DreamLab deployment?
