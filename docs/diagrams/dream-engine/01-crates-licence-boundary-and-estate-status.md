---
id: DE-01
title: The extracted dream-engine repository — crate DAG, licence boundary and estate status
area: dream-engine
governing:
  - docs/architecture/repository-map.md
adrs: [ADR-DE-001, ADR-DE-002, ADR-DE-005, ADR-DE-006, ADR-DE-007]
sources:
  - ../dream-engine/Cargo.toml
  - ../dream-engine/deny.toml
  - ../dream-engine/README.md
  - ../dream-engine/crates/dream-core/Cargo.toml
  - ../dream-engine/crates/dream-gate/Cargo.toml
  - ../dream-engine/crates/dream-cli/Cargo.toml
  - ../dream-engine/crates/pairwise-gate/Cargo.toml
  - ../dream-engine/crates/failure-fingerprint/Cargo.toml
  - ../dream-engine/docs/adr/ADR-DE-001-standalone-rust-engine.md
  - ../dream-engine/docs/adr/ADR-DE-002-licence-and-provenance.md
  - ../dream-engine/docs/adr/ADR-DE-006-automerge-is-an-option.md
  - ../dream-engine/docs/adr/ADR-DE-007-a-night-is-identified-by-its-directory.md
  - ../project/agentbox/services/dream-engine/Cargo.toml
  - ../project/agentbox/docs/adr/ADR-2141-extract-the-dream-engine-core-into-a-standalone-permissive-repo.md
  - scripts/estate-health/roster.json
  - scripts/generate-release-manifest.sh
  - tests/gates/release-manifest.test.sh
  - scripts/estate-browser-journeys.mjs
verified_commit: {dream-engine: 0f6a66913c072603b6e29a80bdc67350737b9b21, agentbox: 1565f477326bcb853886994421e6071c56e27d94, visionflow: 555dffad5d3e8fa4db182c68ba2c65d52dc8399b}
---

The repository's own corpus (eight topics in four areas: nightly, gate, crates, selection) lives at `../dream-engine/docs/diagrams/README.md` and is not duplicated here. This area records only the estate's view of it: the crate DAG, the licence fence, the consumer and the pending items. The estate adapter is AB-23. Both stamps are provisional: the agentbox revision is the head of an unmerged worktree branch and the dream-engine revision is an unpushed local head, so re-stamp after the merge and the push.

## DE-01.1 The crate DAG, the licence fence and the consumer

```mermaid
flowchart TB
    subgraph REPO["DreamLab-AI/dream-engine, MIT OR Apache-2.0 (pending first push)"]
        direction TB
        PG["pairwise-gate, leaf<br/>../dream-engine/crates/pairwise-gate/Cargo.toml:2"]
        FF["failure-fingerprint, leaf<br/>../dream-engine/crates/failure-fingerprint/Cargo.toml:2"]
        DG["dream-gate: verdict, manifest, receipts, witness, gate, guard, patch, outcome, config<br/>../dream-engine/crates/dream-gate/Cargo.toml:23-24"]
        DC["dream-core: runner, ledger, readiness, runstate, roster, sweep, compile, source, context,<br/>candidate, persist, seams, testfmt, streak, replay, automerge<br/>../dream-engine/crates/dream-core/Cargo.toml:17-18"]
        CLI["dream-cli, the dream binary, a thin shell<br/>../dream-engine/crates/dream-cli/Cargo.toml:24-25"]
        DG --> PG
        DG --> FF
        DC --> DG
        DC --> FF
        CLI --> DG
        CLI --> DC
    end
    DENY["INVARIANT: cargo-deny keeps the forum stack and database crates out of the permissive workspace,<br/>nostr-bbs-core and sqlx are banned<br/>../dream-engine/deny.toml:29-30"]
    REPO --- DENY
    FENCE["LICENCE FENCE: a permissive crate must never depend on an AGPL crate, the adapter and its zone-crypto<br/>envelope stay in the consuming product<br/>../dream-engine/docs/adr/ADR-DE-002-licence-and-provenance.md:34"]
    DENY --- FENCE
    subgraph ESTATE["agentbox services/dream-engine, AGPL-3.0-only (the adapter, AB-23)"]
        direction TB
        AD["engine.rs composition root, llm, SshRunner, MemorySink, notifiers,<br/>governance, relay, digest, inbox, zone_crypto<br/>services/dream-engine/Cargo.toml:8"]
    end
    AD -->|"path dependency on dream-core and dream-gate"| DC
    AD -->|"path dependency"| DG
    FENCE --- ESTATE
    PIN["TENSION: both lines must flip to a git revision together, a path and git mix links two copies of dream-gate<br/>whose types do not unify. services/dream-engine/Cargo.toml:22-26"]
    AD --- PIN
```

**Invariant:** the five crates are one licence expression, `MIT OR Apache-2.0`, and the fence is physical because the code that signs Nostr events is not in this repository at all (`../dream-engine/Cargo.toml:8`, `../dream-engine/docs/adr/ADR-DE-002-licence-and-provenance.md:23-33`).

## DE-01.2 What is pending, and where the estate still says otherwise

```mermaid
flowchart TB
    L["built clean-room on the build machine, one workspace, edition 2021<br/>../dream-engine/Cargo.toml:3, ../dream-engine/Cargo.toml:6"]
    P1["PENDING: first push to DreamLab-AI/dream-engine, rename of the TypeScript fork to<br/>DreamLab-AI/dream-machine and its archive"]
    P2["PENDING: crates.io publication. The README already carries badges for pairwise-gate<br/>and failure-fingerprint<br/>../dream-engine/README.md:9-10"]
    P3["PENDING: the adapter flips both path dependencies to one git revision, adds the pin to<br/>lib/dream-engine.nix and rebuilds the image<br/>../project/agentbox/docs/adr/ADR-2141-extract-the-dream-engine-core-into-a-standalone-permissive-repo.md:40"]
    LIVE["NOT YET LIVE (after the rebuild): guard before apply, the finished-night rule and the<br/>corrected dry streak run in the supervised engine"]
    L --> P1 --> P2
    P1 --> P3 --> LIVE
    subgraph STALE["DRIFT: estate files that still describe the fork"]
        direction TB
        R1["roster row: provenance imported, role fork of ruvnet/dream-machine<br/>scripts/estate-health/roster.json:21"]
        R2["release manifest generator seeds the same row as imported<br/>scripts/generate-release-manifest.sh:102"]
        R3["the release-manifest gate asserts exactly two imported repositories<br/>tests/gates/release-manifest.test.sh:68-69"]
        R4["DEBT: the browser journey probes the fork's Pages site, the new repository has no such page<br/>scripts/estate-browser-journeys.mjs:24"]
        R1 --- R2 --- R3 --- R4
    end
    P1 -.-> STALE
    subgraph LIMITS["OPEN: recorded limits of the new engine"]
        direction TB
        K1["publish outcomes are not reconciled after a crash, at-most-once<br/>../dream-engine/docs/adr/ADR-DE-007-a-night-is-identified-by-its-directory.md:34"]
        K2["the automerge forge step is unit-tested against a stub and was not exercised against real GitHub<br/>../dream-engine/README.md:67-69"]
        K1 --- K2
    end
    LIVE -.-> LIMITS
```

**Tension (pending push):** nothing outside the build machine can resolve the adapter's dependencies until the repository is pushed, and the image keeps the pre-extraction binary until then (`../project/agentbox/services/dream-engine/Cargo.toml:27-28`).

**Tension (roster vs rename):** after the rename the slug names a first-party repository, so the next collection would measure it under a provenance label that no longer describes it (`scripts/estate-health/roster.json:21`), and the gate asserting two imported rows fails the day the row is corrected (`tests/gates/release-manifest.test.sh:68-69`), so row, generator and test must move in one change.

**Open:** crates.io publication is recorded nowhere as done, though the README displays badges for two crates (`../dream-engine/README.md:9-10`), and the adapter publishes through the draft path only and never applies the automerge option (AB-23.6).
