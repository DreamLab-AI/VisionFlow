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
  - ../project/agentbox/agentbox.toml
  - ../project/crates/vault/src/main.rs
  - ../project/AGENTS.md
  - docs/estate-review/2026-09-07-imported-adr-scope.md
  - docs/estate-review/2026-09-07-visionclaw-audit.md
  - docs/estate-review/2026-09-07-federation-audit.md
  - docs/estate-review/2026-09-07-agentbox-audit.md
  - docs/estate-review/2026-09-07-estate-audit.md
  - docs/architecture/compatibility-matrix.md
  - docs/architecture/repository-map.md
  - docs/architecture/pod-tier-matrix.md
  - docs/architecture/status-reconciliation.md
  - docs/adr/ADR-2006-canon-owns-crossrepo-view-not-implementation.md
  - docs/adr/ADR-2007-estate-closeout-evidence-roadmap.md
  - docs/BASELINE-visionflow.md
  - ./README.md
verified_commit: {visionflow: 4d361b58ce2b3ee17e1570361eaf78270b630012, visionclaw: 8692b12ba987ecb87165566beb7c320a2446e4b4, agentbox: 6500582dfae23bf8a38a6942916d208154e43625, nostr-rust-forum: 1b27f79d08537b61b3d5a5d02cac9ee5999b79ed}
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
    LOOM["Loom: ontology facade, HTTP only"] -->|configured retrieval| AB
    RV["RuVector: selected core or image"] --> LOOM
    RV -->|MCP and Postgres boundary| AB
    VAULT["vault CLI, VisionClaw crates/vault<br/>agentbox.toml:1167 cli=true<br/>main.rs:49 enum Command"] --> AB
    SCOPE["Arrows identify implemented source seams, not a verified complete mesh"]
    VF -.-> SCOPE
```

**INVARIANT: no MCP inside the estate for the corpus (BASELINE-visionflow.md:341, Invariant 9 since the sidestr invariant was inserted as 6).** Agentbox retired `ontology-bridge.js` (deleted; see AB-25); agents and the Loom reach the corpus only through the `vault` binary over Bash and the Loom's HTTP surface, never a registered MCP server.

**INVARIANT: `verified_commit` is a `{repo: sha}` map, not one sha, for any topic whose `sources:` span more than one repo (`compatibility-matrix.md:8`).** A repository present in `sources:` but missing from that map resolves its citations against the working tree without complaint — a silent source-identity gap, not a verified stamp.

Evidence: `repository-map.md`, the Agentbox audit and imported-scope review. This interaction view is not an exhaustive repository count.

## VF-08.2 Identity representation and replay boundaries

```mermaid
flowchart LR
    KEY["BIP-340 x-only public key"] --> DID["Agentbox canonical did:nostr:64-hex<br/>agent-identity.js:22"]
    KEY --> MULTI["Multikey verification material offered ALONGSIDE,<br/>never as a replacement DID string<br/>agent-identity.js:18"]
    REQ["Signed NIP-98 request"] --> VC["VisionClaw process-local single-use cache —<br/>the event id is recorded atomically after validation<br/>nip98.rs:164"]
    REQ --> POD["Native Solid replay-store seam"]
    REQ --> EDGE["Forum and Worker-specific verifier contracts"]
    REQ --> AB["Agentbox proxy identity boundary"]
    VC --> LIMIT["Restart, replicas, URL semantics and caller wiring require separate proof.<br/>EXTERNAL: the cache is declared single-process by invariant in VisionClaw itself<br/>nip98.rs:210 — see VC-03"]
    POD --> LIMIT
    EDGE --> LIMIT
    AB --> LIMIT
```

Evidence: VisionClaw `src/utils/nip98.rs`, Agentbox `agent-identity.js`, and the federation audit. A canonical identity representation does not unify every verifier. VisionClaw NIP-26 delegation remains deferred; that is narrower than saying no upstream delegation implementation exists anywhere.

## VF-08.4 Mesh and governance are partly connected

```mermaid
flowchart LR
    CORE["Forum mesh core and transport types"] --> PLAN["Relay configuration and fan-out planner<br/>mesh.rs:185 plan_fanout, gated by MESH_FEDERATED_KINDS<br/>mesh.rs:59"]
    PLAN -.-> JOIN["Deferred outbound connector and accept-path joins"]
    KERNEL["VisionClaw broker kernel — broker_case, broker_decision,<br/>precedent_registry<br/>broker/mod.rs:31"] --> ACSP["ACSP and inbox/decide REST surfaces"]
    TASK["Agentbox task spawn"] --> JOURNAL["Journalled local fast path"]
    ACSP --> STAGES["Signed / accepted / projected / received / applied"]
    STAGES -.-> E2E["Complete correlated applied-decision proof remains open"]
    JOURNAL -.-> LIMIT["Does not mediate all tools or nightly egress"]
```

Source: forum `mesh.rs`, VisionClaw `src/domain/broker/mod.rs`, Agentbox `action-plane.js`, and the federation audit. BrokerActor/Neo4j was deliberately left out; the current broker is not merely an unmerged branch. Relay acknowledgement does not prove external application.

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

## VF-08.10 Canon claims require bounded evidence

```mermaid
flowchart TB
    CLAIM["Cross-estate claim"] --> ID["Name repository, path, revision and consumer"]
    ID --> SOURCE["Describe implemented mechanism"]
    SOURCE --> TEST["Attach actual test or measurement"]
    TEST --> LIMIT["State missing runtime, replica, hardware or integration proof"]
    LIMIT --> VIEW["Update current canon view under the ACCEPTED evidence rule:<br/>an item closes only on evidence naming its repository,<br/>the revision it was checked at, its evidence class and its<br/>acceptance boundary — ADR-2007-estate-closeout-evidence-roadmap.md:24"]
    PROPOSAL["Proposed ADR or imported design"] -.-> SCOPE["Explicit proposal/adoption disposition"]
    SCOPE --> VIEW
    VIEW --> BOUNDARY["No blanket completion from diagrams, hashes or passing helper tests"]
```

VisionFlow owns the cross-repository view and evidence vocabulary. Source audits can correct that view, while implementation changes belong to the owning repository. Source hashes bind observations; they do not turn a planned feature or a helper assertion into a running end-to-end system. ADR-2007 was accepted on 2026-10-02, narrowed to exactly this evidence rule (`docs/adr/ADR-2007-estate-closeout-evidence-roadmap.md:24`); the estate closeout roadmap it originally proposed is recorded there as overtaken, not accepted, so this panel states the accepted rule rather than the abandoned coordinating contract.

## VF-08.11 Supporting repositories sit beside the substrates, not inside the mesh

```mermaid
flowchart TB
    CS["Codescape: code graph, layout, agent activity, voice collaboration<br/>its own repository, mounted into VisionClaw as the codescape/ submodule<br/>../project/AGENTS.md:15 and docs/architecture/repository-map.md:37"]
    VCWIRE["VisionClaw coupling is three things only: the published visionclaw-wire crate (crates.io),<br/>the runtime CUDA PTX and the host compose contract<br/>docs/architecture/compatibility-matrix.md:20"]
    ISO["a build-graph isolation test forbids any other VisionClaw dependency, and the root workspace<br/>may never list Codescape as a member or depend on it by path<br/>../project/AGENTS.md:15"]
    VT["visiontouch-rs: private haptic study, no runtime coupling to the substrates,<br/>node-grab input into VisionClaw is intended and not built<br/>docs/architecture/compatibility-matrix.md:20 and docs/architecture/repository-map.md:38"]
    CS --> VCWIRE
    VCWIRE --> ISO
    N1["INVARIANT: the agent-activity hook (agentbox ADR-2144) is on a branch, not agentbox main, and the repository map draws it as a dashed edge labelled branch only<br/>docs/architecture/compatibility-matrix.md:20 and docs/architecture/repository-map.md:83"]
    N2["OPEN: neither supporting repository carries hardware or headset acceptance, so neither is a federated substrate<br/>docs/architecture/compatibility-matrix.md:20"]
    CS --- N1
    VT --- N2
```

**What it shows.** Two repositories admitted since the earlier stamp are drawn at the edge of the dependency picture: Codescape reaches VisionClaw only through a published wire crate, a runtime PTX file and a compose contract, and visiontouch-rs reaches nothing.
**Why it is this way.** The matrix row says to read neither as a federated substrate; the isolation rule in VisionClaw's agent instructions makes the Codescape boundary a build-graph property rather than a convention.
