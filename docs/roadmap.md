# Roadmap

**Status:** Working roadmap from ecosystem docs and code spot-check review
**Date:** 2026-05-22

This roadmap is governed by [Ecosystem Alignment PRD](PRD-ecosystem-alignment.md), [ADR-002](ADR-002-ecosystem-alignment-governance.md), and [Ecosystem Alignment DDD](DDD-ecosystem-alignment-context.md).

## Phase 0: Honesty and Traceability

| Work | Outcome |
|---|---|
| Keep top-level docs linked and current | README claims are backed by local docs |
| Maintain compatibility matrix | Operators can tell which repo versions work together |
| Reconcile PRD status against implementation | Deferred features are not presented as shipped |
| Record verification status for the website | Sidecar browser testing is active; Lighthouse, form, and asset-hashing gaps are explicit |
| Keep PRD/ADR/DDD alignment docs current | Cross-repo scope, decisions, and domain language are explicit |

## Phase 1: Mesh Contract

| Work | Outcome |
|---|---|
| Define canonical IS-Envelope schema owner | One event envelope contract across repos |
| Publish event kind registry | Agent Control Surface and mesh event kinds are unambiguous |
| Normalize DID document service fields | Pod, WebID, and relay discovery work without out-of-band config |
| Add NIP-42/NIP-98/NIP-26 status per substrate | Mesh auth is inspectable |

## Phase 2: End-to-End Proof

| Work | Outcome |
|---|---|
| agentbox -> relay -> forum governance smoke test | Agents can ask humans for decisions |
| forum -> VisionClaw broker response smoke test | Human decisions reach the semantic substrate |
| VisionClaw -> pod provenance write smoke test | Approved mutations are persisted with identity/provenance |
| Cross-substrate fixture sync gate | Protocol drift is caught before release |

The first smoke-test contract is defined in [Mesh Smoke Test](protocol/mesh-smoke-test.md).

## Phase 3: Operational Readiness

| Work | Outcome |
|---|---|
| Versioned ecosystem release manifest | Deployments can pin compatible repo SHAs |
| Pod tier migration plan | Users can move from CF pods to native git-capable pods |
| Unified health dashboard | Operators can see mesh, pod, relay, and broker status |
| Backup and DR runbooks | Recovery is documented across Neo4j, pods, relay stores, D1/KV/R2 |
