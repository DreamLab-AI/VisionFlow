# ADR-002: Ecosystem Alignment Governance

**Status:** Accepted
**Date:** 2026-05-22
**Decision Owners:** DreamLab AI maintainers
**Related:** [Ecosystem Alignment PRD](PRD-ecosystem-alignment.md), [Compatibility Matrix](architecture/compatibility-matrix.md), [Mesh Smoke Test](protocol/mesh-smoke-test.md), [Release Manifests](releases/README.md)

## Context

VisionFlow coordinates a multi-repository ecosystem:

- VisionClaw provides the semantic substrate, broker concepts, contracts, and many canonical fixtures.
- agentbox provides agent runtime, MCP, relay bridge, and local pod integration.
- solid-pod-rs provides Rust Solid pod, WebID, WAC, DID, and storage capability.
- nostr-rust-forum provides Nostr forum, governance, mesh, and human-agent collaboration middleware.
- dreamlab-ai-website provides brand surface, deployment configuration, and an integration test surface.

The audit found substantial implementation alignment, especially around `did:nostr`, Solid pod integration, governance event kinds, and shared fixtures. It also found that runtime maturity is uneven: mesh federation is well specified, but default deployments and some crates still represent standalone or scaffolded behavior.

Without a coordination decision, each repository can remain locally correct while the ecosystem-level story drifts.

## Decision

VisionFlow owns the umbrella alignment process. Repository-local docs remain authoritative for implementation detail, but VisionFlow owns the cross-repo view of compatibility, maturity, and release evidence.

Specifically:

1. Coordinated ecosystem releases must publish a machine-readable release manifest under `docs/releases/` when they are promoted beyond local draft.
2. The compatibility matrix is the human-readable source of truth for cross-repo identity, mesh, pod, governance, deployment, and verification posture.
3. Shared protocol contracts must have a named canonical owner and listed consumers.
4. Mesh federation claims require evidence from the end-to-end smoke path, not only design docs.
5. Shared fixtures remain part of the compatibility contract and must be checked for drift during release qualification.
6. Active docs must label maturity conservatively as shipped, integrated, scaffolded, experimental, standalone, or historical.
7. Historical PRDs and ADRs should be preserved, but active index pages must route readers through current status reconciliation.

## Protocol Ownership Policy

| Contract | Canonical Owner | Consumers |
|---|---|---|
| `did:nostr` identity spine | VisionFlow protocol docs, with implementation evidence from solid-pod-rs and nostr-rust-forum | VisionClaw, agentbox, solid-pod-rs, nostr-rust-forum, dreamlab-ai-website |
| Solid pod, WebID, WAC, NIP-98 pod auth | solid-pod-rs | VisionClaw, agentbox, dreamlab-ai-website |
| Agent Control Surface event kinds `31400`-`31405` | nostr-rust-forum core governance model | agentbox, VisionClaw, dreamlab-ai-website |
| Judgment Broker domain model | nostr-rust-forum (domain model + human surface); VisionClaw (enrichment gating); agentbox (agent wiring + broker-bridge) | All substrates via Nostr relay mesh |
| Release compatibility manifest | VisionFlow | All ecosystem repositories |
| Shared fixture corpus | VisionClaw for master fixtures until a standalone fixture package exists | agentbox, solid-pod-rs, nostr-rust-forum |
| Ontology bridge (SPARQL proxy) | agentbox (`mcp/servers/ontology-bridge.js`, 10 MCP tools) | VisionClaw (Oxigraph backend) |
| IS-Envelope | VisionClaw (ADR-075, JSON Schema, 11 test vectors). agentbox implements runtime decode/dispatch. | VisionClaw, agentbox, nostr-rust-forum |

## Maturity Vocabulary

| Status | Meaning |
|---|---|
| `historical` | Documented for context but not a current implementation claim |
| `planned` | Product or architecture intent without implementation evidence |
| `scaffolded` | Code or config shape exists, but end-to-end behavior is not proven |
| `standalone` | Works locally without cross-service federation guarantees |
| `integrated` | Cross-repo contract is implemented by at least two substrates |
| `federation-verified` | End-to-end runtime proof exists across the required substrates |
| `released` | Pinned in a coordinated release manifest |

## Consequences

Positive:

- Readers have one place to understand ecosystem truth without flattening repository ownership.
- Release claims become auditable through repo SHAs and compatibility assertions.
- Mesh maturity cannot be overstated without smoke-test evidence.
- Protocol drift has a defined place to be detected and resolved.

Tradeoffs:

- VisionFlow docs become a release dependency for coordinated ecosystem claims.
- Maintainers must update both local implementation docs and umbrella coordination docs for cross-repo changes.
- IS-Envelope ownership is resolved (VisionClaw ADR-075); runtime consumers should pin to the VisionClaw fixture schema version.

## Implementation Notes

- Use `scripts/generate-release-manifest.sh` and `docs/releases/ecosystem-release.schema.json` for release manifest generation and validation.
- Keep the compatibility matrix short enough to review, but link to repository-local docs, configs, and code evidence.
- Treat the mesh smoke test as the promotion gate from `integrated` to `federation-verified`.
- Do not remove historical PRDs solely because they are aspirational; route them through current status reconciliation.
- The agentbox upstream fixture corpus (`tests/contract/upstream_vectors/`, 54 vectors) includes SHA-256 checksums (`CHECKSUMS.txt`) and upstream pin tracking (`UPSTREAM_PINS.md`). Use these as the baseline for fixture drift detection.
- The ontology bridge (10 MCP tools in `agentbox/mcp/servers/ontology-bridge.js`) is a shipped cross-substrate integration between agentbox and VisionClaw. Include it in compatibility evidence.

## Deferred Decisions

- IS-Envelope ownership resolved: VisionClaw (ADR-075). Runtime consumers: agentbox (relay-consumer.js decode), solid-pod-rs (fixture vectors), nostr-rust-forum (fixture vectors).
- CI topology for checking externally mounted repositories.
- Fixture parity is currently byte-for-byte via SHA-256 checksums. Whether semantic comparison is additionally needed remains open.
- Exact relay topology required for federation qualification.
