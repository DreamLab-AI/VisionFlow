# VisionFlow Documentation

VisionFlow is the coordination architecture spanning the DreamLab repositories. Start here for the local, repo-owned documentation entry points.

## Ecosystem

| Document | Purpose |
|---|---|
| [Ecosystem Alignment PRD](PRD-ecosystem-alignment.md) | Requirements for compatibility, maturity, ownership, fixture, and release-readiness work |
| [Ecosystem Alignment ADR-002](ADR-002-ecosystem-alignment-governance.md) | Decision record for cross-repo alignment governance and maturity vocabulary |
| [Ecosystem Alignment DDD](DDD-ecosystem-alignment-context.md) | Bounded context, aggregates, invariants, and language for ecosystem alignment |
| [Ecosystem Map](ecosystem-map.md) | Docs and spot-check synthesis of the sibling repositories, system flows, and gap register |
| [Repository Map](architecture/repository-map.md) | Local path and role map for the federated repositories |
| [Compatibility Matrix](architecture/compatibility-matrix.md) | Current identity, mesh, pod, governance, deployment, and tests/ops posture |
| [Status Reconciliation](architecture/status-reconciliation.md) | Current reading of older PRD/ADR claims versus implementation status |
| [Pod Tier Matrix](architecture/pod-tier-matrix.md) | Capability comparison across Cloudflare Workers, embedded, native, and git-capable pod tiers |
| [Identity Spine](protocol/identity-spine.md) | Shared `did:nostr` identity contract and compatibility checklist |
| [Mesh Smoke Test](protocol/mesh-smoke-test.md) | End-to-end governance proof path and current blockers |
| [Release Manifests](releases/README.md) | Machine-readable manifest process for coordinated ecosystem releases |
| [Licensing](architecture/licensing.md) | Licence boundaries across the ecosystem |
| [Roadmap](roadmap.md) | Phased roadmap from docs honesty to mesh proof and operations |

## Website

| Document | Purpose |
|---|---|
| [Website PRD](PRD-website.md) | Product requirements for `visionflow.info` |
| [Website DDD](DDD-website-context.md) | Bounded contexts for the website |
| [Website ADR-001](ADR-001-website-technology.md) | Static site and Rust/WASM technology decision |
| [Site Verification](site-verification.md) | Current verification status against website PRD acceptance criteria |
