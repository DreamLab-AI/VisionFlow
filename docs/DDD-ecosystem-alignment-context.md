# Domain-Driven Design: Ecosystem Alignment Context

**Owner:** DreamLab AI
**Status:** Draft
**Date:** 2026-05-22
**Related:** [Ecosystem Alignment PRD](PRD-ecosystem-alignment.md), [ADR-002](ADR-002-ecosystem-alignment-governance.md)

## Bounded Context

The Ecosystem Alignment Context governs compatibility, maturity, ownership, and release evidence across the VisionFlow repository set.

It does not own the implementation of agents, pods, relays, brokers, forums, or websites. It owns the language and artifacts that say which versions of those systems work together and what level of maturity has been proven.

## Context Map

| Context | Upstream / Downstream Relationship |
|---|---|
| VisionFlow Alignment | Upstream for compatibility language, release manifests, and maturity status |
| VisionClaw Semantic Substrate | Upstream for broker contracts and master fixtures; downstream of alignment release gates |
| agentbox Runtime | Upstream for agent runtime behavior; downstream of Agent Control Surface and release manifests |
| solid-pod-rs Pod Platform | Upstream for pod, WebID, WAC, and auth capability |
| nostr-rust-forum Collaboration Middleware | Upstream for governance event model and forum/mesh behavior |
| dreamlab-ai-website Brand/Test Surface | Downstream consumer of ecosystem status and deployment posture |
| Ontology Bridge (agentbox ↔ VisionClaw) | Cross-cutting integration: 10 MCP tools in agentbox proxy SPARQL queries to VisionClaw's Oxigraph store |

## Core Domain Concepts

### Aggregates

| Aggregate | Responsibility |
|---|---|
| `CompatibilityManifest` | Pins repository refs and compatibility assertions for a coordinated release |
| `Substrate` | Represents one participating repository and its role, path, version, branch, and maturity state |
| `ProtocolContract` | Defines a shared protocol surface, canonical owner, consumers, schema/version, and evidence |
| `MeshCapabilityStatus` | Records mesh maturity per substrate and the evidence required for promotion |
| `FixtureCorpus` | Tracks canonical fixtures, copied fixtures, checksums, semantic version, and drift status |
| `GovernanceSmokeTest` | Defines and records the end-to-end agentbox -> relay -> forum -> broker -> pod proof |
| `PodTierMatrix` | Describes supported storage/auth/provenance features by pod deployment tier |
| `ReleaseCandidate` | Bundles manifest, compatibility matrix snapshot, smoke-test evidence, and open gaps |
| `OntologyBridgeStatus` | Tracks health and availability of the 10-tool SPARQL proxy between agentbox and VisionClaw Oxigraph |

### Entities

| Entity | Identity |
|---|---|
| `RepositoryRef` | Repository name plus local path |
| `SubstrateVersion` | Git SHA plus branch and dirty flag |
| `ProtocolOwner` | Owning repository or coordination doc |
| `FixtureSet` | Fixture namespace plus version/checksum |
| `SmokeTestRun` | Test run id plus timestamp |
| `EvidenceLink` | File path, test command, config path, or code path |
| `OntologyBridgeTool` | Tool name plus endpoint URL |

### Value Objects

| Value Object | Fields |
|---|---|
| `GitSha` | 40-character commit hash |
| `MaturityLevel` | `historical`, `planned`, `scaffolded`, `standalone`, `integrated`, `federation-verified`, `released` |
| `NostrEventKind` | Integer kind plus semantic name |
| `DidNostrIdentifier` | DID method, pubkey, verification method |
| `FixtureChecksum` | Algorithm plus digest |
| `CompatibilityAssertion` | Domain, claim, status, evidence |
| `PodCapability` | Feature name, tier, supported flag, constraints |
| `ReleaseStatus` | `local-draft`, `candidate`, `released` |
| `SparqlEndpoint` | URL, health status, last-checked timestamp |

## Domain Events

| Event | Trigger |
|---|---|
| `CompatibilityManifestGenerated` | A local manifest is generated from workspace repositories |
| `ReleaseCandidateOpened` | A candidate manifest is committed for review |
| `ReleasePromoted` | A manifest is marked released |
| `ProtocolOwnerAssigned` | A shared protocol receives a canonical owner |
| `ProtocolOwnerChanged` | Ownership moves to a different repository or package |
| `FixtureDriftDetected` | Fixture copy differs from canonical source |
| `FixtureDriftResolved` | Fixture parity is restored |
| `MeshStatusPromoted` | A substrate moves to a higher maturity level |
| `MeshStatusDemoted` | Evidence no longer supports a maturity claim |
| `GovernanceSmokeTestPassed` | The full governance path succeeds |
| `GovernanceSmokeTestFailed` | Any required assertion in the smoke path fails |
| `HistoricalClaimReconciled` | An older doc claim is mapped to current status |
| `OntologyBridgeHealthChanged` | Bridge health check detects status change |

## Invariants

- A `ReleaseCandidate` cannot be promoted to `released` without a valid `CompatibilityManifest`.
- A `CompatibilityManifest` must include VisionFlow, VisionClaw, agentbox, solid-pod-rs, nostr-rust-forum, and dreamlab-ai-website.
- A `ProtocolContract` must have exactly one canonical owner at a time.
- A protocol owner may be VisionFlow only as a temporary coordination owner; implementation ownership must be resolved before release if the protocol is runtime-critical.
- Mesh status cannot be `federation-verified` without a passing `GovernanceSmokeTest` or equivalent linked runtime evidence.
- A `FixtureCorpus` cannot be current when canonical and copied fixtures differ without an explicit compatibility exception.
- A repository with a dirty working tree can appear in a `local-draft` or `candidate` manifest, but not in a final `released` manifest unless the release policy explicitly allows it.
- Historical docs must not be used as release evidence unless current implementation evidence is also linked.
- A `ProtocolContract` for IS-Envelope must reference VisionClaw ADR-075 as canonical owner. Temporary VisionFlow ownership is superseded.
- The Judgment Broker is a distributed capability at 65% implementation. The forum decision loop is closed; the agent decision application loop is not. Cite as `integrated` for the forum↔relay↔agent event flow, but `scaffolded` for decision application and provenance.

## Ubiquitous Language

| Term | Meaning |
|---|---|
| Alignment | The ecosystem-level agreement that repo versions, protocols, and maturity claims fit together |
| Substrate | A participating repository that provides runtime, protocol, storage, governance, brand, or coordination capability |
| Compatibility Claim | A statement that a protocol or integration works across named substrates |
| Evidence | A doc, config, code path, fixture, test, or manifest supporting a claim |
| Fixture Drift | Any mismatch between canonical protocol fixtures and repository-local copies |
| Federation Verified | Runtime proof that a mesh path works across required services |
| Standalone | Local or single-service operation without federation guarantees |
| Release Manifest | Machine-readable artifact pinning the compatible repository set |
| Protocol Owner | The canonical place where a shared contract is defined and versioned |
| Ontology Bridge | The 10-tool MCP server in agentbox that proxies SPARQL queries to VisionClaw's Oxigraph knowledge graph |
| Browser Setup Wizard | The standalone SPA in agentbox that edits `agentbox.toml` before container boot, with zero compiled dependencies |

## Services

| Service | Responsibility |
|---|---|
| `GenerateReleaseManifest` | Reads workspace repository refs and emits a manifest draft |
| `ValidateReleaseManifest` | Checks manifest schema, required repositories, and release policy |
| `CompareFixtureCorpus` | Compares canonical and copied fixtures |
| `EvaluateMaturityStatus` | Determines whether evidence supports a maturity level |
| `RunGovernanceSmokeTest` | Executes or records the end-to-end governance proof |
| `ReconcileHistoricalClaims` | Maps older claims to current implementation status |
| `CheckOntologyBridgeHealth` | Verifies agentbox can reach VisionClaw Oxigraph and all 10 tools respond |

## Repositories and Persistence

The primary persistence model is documentation plus versioned release artifacts:

- `docs/releases/*.json` for manifests.
- `docs/architecture/compatibility-matrix.md` for human review.
- `docs/architecture/status-reconciliation.md` for historical claim mapping.
- `docs/protocol/mesh-smoke-test.md` for runtime proof definition.
- Repository-local docs and tests for implementation evidence.

If the alignment context becomes automated, these artifacts should remain the public interface even if generated from a structured store.
