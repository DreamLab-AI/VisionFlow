---
id: VF-08
title: Compatibility matrix and dependency direction across the estate
area: visionflow
governing:
  - docs/architecture/compatibility-matrix.md
  - docs/architecture/repository-map.md
  - docs/BASELINE-visionflow.md
adrs: [ADR-2006, ADR-2007]
sources:
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/mesh.rs
  - ../project/src/utils/nip98.rs
  - ../project/src/domain/broker/mod.rs
  - ../project/agentbox/management-api/lib/agent-identity.js
  - ../project/agentbox/mcp/servers/ontology-bridge.js
  - docs/estate-review/2026-09-07-imported-adr-scope.md
  - docs/estate-review/2026-09-07-visionclaw-audit.md
  - docs/estate-review/2026-09-07-federation-audit.md
  - docs/estate-review/2026-09-07-agentbox-audit.md
  - docs/estate-review/2026-09-07-estate-audit.md
  - docs/architecture/compatibility-matrix.md
  - docs/architecture/repository-map.md
  - docs/architecture/pod-tier-matrix.md
  - docs/architecture/licensing.md
  - docs/architecture/status-reconciliation.md
  - docs/ecosystem-map.md
  - docs/registers/gap-register-v1.1.md
  - docs/registers/gap-register-v1.2.md
  - docs/registers/gap-register-v1.3.md
  - docs/registers/F9-federation-fork-record.md
  - docs/adr/ADR-2006-canon-owns-crossrepo-view-not-implementation.md
  - docs/adr/ADR-2007-estate-closeout-evidence-roadmap.md
  - docs/estate-review/evidence/adr-inventory.json
  - scripts/generate-release-manifest.sh
  - scripts/estate-health/roster.json
  - scripts/drift-counter/allowlist.json
  - docs/releases/candidate-2026-05-22.json
  - docs/BASELINE-visionflow.md
  - MAINTAINERS.md
  - LICENSES/README.md
  - ./README.md
verified_commit: bec06dc3a
---

Source reconciliation: 2026-09-07. These panels follow the current sections of the compatibility matrix and status reconciliation, plus the source audits named in frontmatter. Historical metrics and register cuts remain historical. This review did not certify a live federation, image, hardware session or replica-wide authentication contract.

## VF-08.1 Dependency direction without a fictional complete mesh

```mermaid
flowchart TB
    VF["VisionFlow: canon and evidence"] -.-> VC["VisionClaw: graph, GPU, XR, broker kernel"]
    VF -.-> AB["Agentbox: agent runtime"]
    SPR["solid-pod-rs source and selected packages"] --> VC
    SPR --> AB
    NRF["Forum kit: relay, auth, governance"] --> DLW["DreamLab deployment overlay"]
    AB -->|agent-event producer to ingest| VC
    AB <-->|broker REST and ACSP integration| VC
    VC <-->|configured governance paths| NRF
    AB <-->|configured relay and inbox paths| NRF
    LOOM["Loom: ontology facade"] -->|configured retrieval| AB
    RV["RuVector: selected core or image"] --> LOOM
    RV -->|MCP and Postgres boundary| AB
    SCOPE["Arrows identify implemented source seams, not a verified complete mesh"]
    VF -.-> SCOPE
```

Evidence: `repository-map.md`, the Agentbox audit and imported-scope review. Published corpus/explorer, upstream sensing and ancillary repositories remain in the extended estate inventory; this interaction view is not an exhaustive repository count.

## VF-08.2 Identity representation and replay boundaries

```mermaid
flowchart LR
    KEY["BIP-340 x-only public key"] --> DID["Agentbox canonical did:nostr:64-hex"]
    KEY --> MULTI["Multikey verification material offered alongside"]
    REQ["Signed NIP-98 request"] --> VC["VisionClaw process-local single-use cache"]
    REQ --> POD["Native Solid replay-store seam"]
    REQ --> EDGE["Forum and Worker-specific verifier contracts"]
    REQ --> AB["Agentbox proxy identity boundary"]
    VC --> LIMIT["Restart, replicas, URL semantics and caller wiring require separate proof"]
    POD --> LIMIT
    EDGE --> LIMIT
    AB --> LIMIT
```

Evidence: VisionClaw `src/utils/nip98.rs`, Agentbox `agent-identity.js`, and the federation audit. A canonical identity representation does not unify every verifier. VisionClaw NIP-26 delegation remains deferred; that is narrower than saying no upstream delegation implementation exists anywhere.

## VF-08.3 Estate inventories have different scopes

```mermaid
flowchart TB
    MAP["Repository map: architectural roles"] --> REVIEW["Compare declared scope and evidence date"]
    RELEASE["Release manifest: selected repository and revision identities"] --> REVIEW
    HEALTH["Health roster: probeable service/repository targets"] --> REVIEW
    ADR["ADR inventory: documents and lineage"] --> REVIEW
    REVIEW --> DIFFER["Different membership can be deliberate"]
    REVIEW --> DRIFT["Missing owned dependency or contradictory role needs correction"]
    REVIEW --> RULE["Do not reuse a historical nine/fourteen count as today's estate total"]
```

Use the current inventory and source records. A release roster, runtime health roster and ADR corpus are different sets; neither matching counts nor unequal counts alone prove correctness or drift.

## VF-08.4 Mesh and governance are partly connected

```mermaid
flowchart LR
    CORE["Forum mesh core and transport types"] --> PLAN["Relay configuration and fan-out planner"]
    PLAN -.-> JOIN["Deferred outbound connector and accept-path joins"]
    KERNEL["VisionClaw broker kernel"] --> ACSP["ACSP and inbox/decide REST surfaces"]
    TASK["Agentbox task spawn"] --> JOURNAL["Journalled local fast path"]
    ACSP --> STAGES["Signed / accepted / projected / received / applied"]
    STAGES -.-> E2E["Complete correlated applied-decision proof remains open"]
    JOURNAL -.-> LIMIT["Does not mediate all tools or nightly egress"]
```

Source: forum `mesh.rs`, VisionClaw `src/domain/broker/mod.rs`, Agentbox `action-plane.js`, and the federation audit. BrokerActor/Neo4j was deliberately left out; the current broker is not merely an unmerged branch. Relay acknowledgement does not prove external application.

## VF-08.5 Versions and tool counts require selected-source identity

```mermaid
flowchart TB
    OLD["Historical candidate manifest and twelve-tool prose"] --> DATE["Retain date and revision"]
    CURRENT["Current ontology-bridge.js TOOLS"] --> ELEVEN["Eleven advertised entries"]
    DISPATCH["ontology_propose dispatcher branch"] --> SEPARATE["Dispatchable branch is not an advertised registry entry"]
    PIN["Drift-counter pinned sibling revision"] --> PINNED["Gate result applies to that revision"]
    CURRENT --> QUALIFY["Working tree, pinned source and loaded MCP may differ"]
    PINNED --> QUALIFY
    DATE --> QUALIFY
```

The current TOOLS array and ListTools handler were read directly. Historical twelve-tool statements and candidate package versions are preserved in the matrix's dated snapshot, not claimed as current. A moving checkout must not silently update the gate's approved pin.

## VF-08.6 Licence evidence follows the consumed component

```mermaid
flowchart LR
    REPO["Repository licence statements"] --> COMPONENT["Selected crate, package or asset"]
    COMPONENT --> SOURCE["Its manifest and licence files"]
    COMPONENT --> LINK["Actual dependency or process boundary"]
    SOURCE --> REVIEW["Record component-specific obligations and unresolved questions"]
    LINK --> REVIEW
    IMPORT["Imported upstream ADR or copied tree"] -.-> NO["Does not grant a new licence or prove adopted code identity"]
```

Keep `licensing.md` as the policy route, with repository manifests and licence files as the evidence. Agentbox service crates and its wider repository have different declared terms. The imported-scope review distinguishes Loom's sibling core, RuView's registry dependencies and Agentbox's image. This panel does not issue a new legal determination or infer estate-wide terms from one badge.

## VF-08.7 Pod compatibility is a consumer contract

```mermaid
flowchart TB
    LIB["solid-pod-rs library and server capabilities"] --> VC["VisionClaw embedded library selection"]
    LIB --> AB["Agentbox supervised native server selection"]
    LIB --> CF["Forum Worker tier and feature selection"]
    VC --> CHECK["Verify selected version, features and caller wiring"]
    AB --> CHECK
    CF --> CHECK
    CHECK --> AUTH["WAC, OIDC and replay semantics"]
    CHECK --> DATA["Storage policy, cache, notifications and git capability"]
    AUTH --> TEST["Consumer-level fixtures and deployment evidence"]
    DATA --> TEST
```

The federation audit qualifies native OIDC, process-local replay and Worker caching. The historical pod-tier matrix is a capability guide, not proof that every tier implements the full library surface or shares its replay store. Agentbox's supervised native server must not be relabelled as an in-process embedded library.

## VF-08.8 Reconciliation preserves evidence dates

```mermaid
sequenceDiagram
    participant OLD as Historical claim
    participant CODE as Selected source and tests
    participant NOW as Current compatibility section
    participant HISTORY as Dated snapshots and registers
    OLD->>CODE: Check mechanism, revision and scope
    CODE-->>NOW: Source-supported correction with limitations
    OLD->>HISTORY: Preserve dated original
    NOW->>NOW: Separate implementation from deployed acceptance
    Note over NOW,HISTORY: New prose never fires a pending-live canary
```

`status-reconciliation.md` now separates September findings from the May notes. The matrix separates its current source comparison from the July compatibility and harness snapshots. Completion percentages and stale version strings in those snapshots are not live status.

## VF-08.9 Published register cuts remain immutable

```mermaid
stateDiagram-v2
    [*] --> v1_0
    v1_0 --> v1_1 : historical P0 cut
    v1_1 --> v1_2 : historical P1/P2 cut
    v1_2 --> v1_3 : historical broker addendum
    v1_3 --> NewEvidence : source and runtime observations
    NewEvidence --> NextDisposition : forward-chained correction
    note right of v1_3
        Historical integrated/scaffolded tiers and pending-live
        canaries retain their recorded evidence scope.
    end note
    note right of NextDisposition
        Promotion needs a new receipt, not a rewrite of an old cut.
        Federation fork criteria are evaluated at their own date.
    end note
```

Evidence: the v1.1–v1.3 registers and F9 fork record. This audit neither overwrites those cuts nor declares their pending-live acceptance complete.

## VF-08.10 Canon claims require bounded evidence

```mermaid
flowchart TB
    CLAIM["Cross-estate claim"] --> ID["Name repository, path, revision and consumer"]
    ID --> SOURCE["Describe implemented mechanism"]
    SOURCE --> TEST["Attach actual test or measurement"]
    TEST --> LIMIT["State missing runtime, replica, hardware or integration proof"]
    LIMIT --> VIEW["Update current canon view and closeout roadmap"]
    PROPOSAL["Proposed ADR or imported design"] -.-> SCOPE["Explicit proposal/adoption disposition"]
    SCOPE --> VIEW
    VIEW --> BOUNDARY["No blanket completion from diagrams, hashes or passing helper tests"]
```

VisionFlow owns the cross-repository view and evidence vocabulary. Source audits can correct that view, while implementation changes belong to the owning repository. Source hashes bind observations; they do not turn a planned feature or a helper assertion into a running end-to-end system.
