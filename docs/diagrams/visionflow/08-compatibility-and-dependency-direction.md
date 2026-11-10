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
  - ../project/agentbox/flake.nix
  - ../project/agentbox/lib/ontomark.nix
  - ../project/agentbox/management-api/lib/ontology-apply.js
  - scripts/diagram-index-gen.cjs
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
verified_commit: {visionflow: a4a6c20c34c0e298bc4a6725e71e58e00685f64c, visionclaw: 46edc196b58a46c64fffa5c10e3dd72c0a36238e, agentbox: 720e0d89656a4360998b3a716d0f3ee8cac66f93, nostr-rust-forum: 6064622dbb1a7cc1ba2d072109f9e3e8a96157df}
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
    CLI["Ontomark corpus operations, pin 0.8.0<br/>lib/ontomark.nix:46<br/>ontology-apply.js:157"] --> AB
    VAULT["vault governance transport: propose and panel<br/>ontology-apply.js:153"] --> AB
    GATE["Both packages share vaultCliActive<br/>flake.nix:952 and flake.nix:970"] --> CLI
    GATE --> VAULT
    SCOPE["Arrows identify implemented source seams, not a verified complete mesh"]
    VF -.-> SCOPE
```

**Current Agentbox corpus dispatch, 2026-10-31:** `binFor` routes `propose` and `panel` through `vault`, and other corpus verbs through `ontomark` (`../project/agentbox/management-api/lib/ontology-apply.js:153`). Both are selected under the same Nix gate (`../project/agentbox/flake.nix:952`, `../project/agentbox/flake.nix:970`). The old vault-only statement did not describe the migrated consumer. This source trace is not a universal absence claim about every estate MCP adapter.

**INVARIANT: `verified_commit` is a `{repo: sha}` map, not one sha, for any topic whose `sources:` span more than one repo (`scripts/diagram-index-gen.cjs:215-225`).** A missing repository pin can fall back to working-tree bytes, but the current checker reports that as unverified (`scripts/diagram-index-gen.cjs:579`) and strict mode rejects warnings (`scripts/diagram-index-gen.cjs:912`). Such a fallback must not be presented as revision-backed verification.

Evidence: `repository-map.md`, the Agentbox audit and imported-scope review. This interaction view is not an exhaustive repository count.

## VF-08.2 Identity representation and replay boundaries

```mermaid
flowchart LR
    KEY["BIP-340 x-only public key"] --> DID["Agentbox canonical did:nostr:64-hex<br/>agent-identity.js:22"]
    KEY --> MULTI["Multikey verification material offered ALONGSIDE,<br/>never as a replacement DID string<br/>agent-identity.js:22-25"]
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
    LIMIT --> VIEW["Update current canon view under the ACCEPTED evidence rule:<br/>an item closes only on evidence naming its repository,<br/>the revision it was checked at, its evidence class and its<br/>acceptance boundary — ADR-2007-estate-closeout-evidence-roadmap.md:26"]
    PROPOSAL["Proposed ADR or imported design"] -.-> SCOPE["Explicit proposal/adoption disposition"]
    SCOPE --> VIEW
    VIEW --> BOUNDARY["No blanket completion from diagrams, hashes or passing helper tests"]
```

VisionFlow owns the cross-repository view and evidence vocabulary. Source audits can correct that view, while implementation changes belong to the owning repository. Source hashes bind observations; they do not turn a planned feature or a helper assertion into a running end-to-end system. ADR-2007 was accepted on 2026-10-02, narrowed to exactly this evidence rule (`docs/adr/ADR-2007-estate-closeout-evidence-roadmap.md:26`); the estate closeout roadmap it originally proposed is recorded there as overtaken, not accepted, so this panel states the accepted rule rather than the abandoned coordinating contract.

## VF-08.11 Supporting repositories sit beside the substrates, not inside the mesh

```mermaid
flowchart TB
    CS["Codescape: code graph, layout, agent activity, voice collaboration<br/>its own repository, mounted into VisionClaw as the codescape/ submodule<br/>../project/AGENTS.md:15 and docs/architecture/repository-map.md:37"]
    VCWIRE["VisionClaw coupling is three things only: the published visionclaw-wire crate (crates.io),<br/>the runtime CUDA PTX and the host compose contract<br/>docs/architecture/compatibility-matrix.md:20"]
    ISO["a build-graph isolation test forbids any other VisionClaw dependency, and the root workspace<br/>may never list Codescape as a member or depend on it by path<br/>../project/AGENTS.md:15"]
    VT["visiontouch-rs: private haptic study, no runtime coupling to the substrates,<br/>node-grab input into VisionClaw is intended and not built<br/>docs/architecture/compatibility-matrix.md:20 and docs/architecture/repository-map.md:40"]
    CS --> VCWIRE
    VCWIRE --> ISO
    N1["INVARIANT: the agent-activity hook (agentbox ADR-2144) is on a branch, not agentbox main, and the repository map draws it as a dashed edge labelled branch only<br/>docs/architecture/compatibility-matrix.md:20 and docs/architecture/repository-map.md:85"]
    N2["OPEN: neither supporting repository carries hardware or headset acceptance, so neither is a federated substrate<br/>docs/architecture/compatibility-matrix.md:20"]
    CS --- N1
    VT --- N2
```

**What it shows.** Two repositories admitted since the earlier stamp are drawn at the edge of the dependency picture: Codescape reaches VisionClaw only through a published wire crate, a runtime PTX file and a compose contract, and visiontouch-rs reaches nothing.
**Why it is this way.** The matrix row says to read neither as a federated substrate; the isolation rule in VisionClaw's agent instructions makes the Codescape boundary a build-graph property rather than a convention.


**Review boundary, 2026-10-31:** the Agentbox corpus-dispatch slice and the checker-fallback wording were refreshed then. The session at that time exposed Ontomark 0.7.0 and vault 0.1.0, recorded with binary hashes in the [consumer audit](../../estate-review/2026-10-31-corpus-consumer.md). No full mesh or universal deployment claim follows from those local versions.

**Restamp, 2026-11-02:** all four repository pins moved to the audit wave 2 revisions: VisionFlow `5f8b1da`, VisionClaw `86cdd76`, Agentbox `c57d9c0` and nostr-rust-forum `c39f41f`. Every citation was re-read at those revisions. The cross-repository lines were byte-identical to the earlier pins. Two VisionFlow lines moved, and are now `ADR-2007-estate-closeout-evidence-roadmap.md:26` and the checker lines in VF-08.1. ADR-2007 is now `complete` and `live` (`docs/adr/ADR-2007-estate-closeout-evidence-roadmap.md:6-7`), with the evidence gate in all five ledgers, so the rule VF-08.10 states is also mechanically enforced at closing transitions. That is source tracing; no hosted run was checked.

**Drift (anchors, corrected 2026-11-02):**
- VF-08.1's map invariant cited a matrix sentence that says nothing about `verified_commit`. It now cites the checker that enforces the form (`scripts/diagram-index-gen.cjs:215-225`).
- VF-08.2's Multikey node cited the key-handling line. It now cites the lines that say the Multikey is offered alongside the unchanged DID (`../project/agentbox/management-api/lib/agent-identity.js:22-25`).
