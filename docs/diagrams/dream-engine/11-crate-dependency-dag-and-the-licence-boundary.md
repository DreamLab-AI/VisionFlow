---
id: DE-11
title: The crate dependency DAG and the licence boundary
area: dream-engine
governing: [../dream-engine/docs/adr/ADR-DE-002-licence-and-provenance.md, ../dream-engine/docs/adr/ADR-DE-001-standalone-rust-engine.md]
adrs: [ADR-DE-001, ADR-DE-002]
sources:
  - ../dream-engine/Cargo.toml
  - ../dream-engine/deny.toml
  - ../dream-engine/crates/dream-core/Cargo.toml
  - ../dream-engine/crates/dream-gate/Cargo.toml
  - ../dream-engine/crates/dream-cli/Cargo.toml
  - ../dream-engine/crates/pairwise-gate/Cargo.toml
  - ../dream-engine/crates/failure-fingerprint/Cargo.toml
  - ../dream-engine/.github/workflows/ci.yml
  - ../dream-engine/README.md
  - ../dream-engine/crates/dream-gate/src/gate.rs
  - ../dream-engine/docs/adr/ADR-DE-002-licence-and-provenance.md
  - ../dream-engine/docs/adr/ADR-DE-001-standalone-rust-engine.md
verified_commit: b40d50724bb6344dc00d57270f7166b138714406
---
## For developers

The workspace holds five crates, all under one licence expression, and the
dependency edges only point downwards (`../dream-engine/Cargo.toml:3`, `../dream-engine/Cargo.toml:8`). Two
leaf crates, `pairwise-gate` and `failure-fingerprint`, know nothing of the
engine. `dream-gate` judges using both, `dream-core` does the impure work around
the judgement using `dream-gate` and `failure-fingerprint`, and `dream-cli` is a thin
shell over the last two. A consuming product brings its own adapter, which
depends on the core crates and never the other way round
(`docs/adr/ADR-DE-002-licence-and-provenance.md:28`).

## For the business

The five workspace packages declare `MIT OR Apache-2.0`. Their dependency direction and CI policy keep host-specific components outside the engine graph. These are inspectable packaging controls, not a proof of source provenance or a guarantee about every downstream dependency configuration.

## DE-11.1 Crate dependencies inside the workspace

```mermaid
flowchart TB
    CLI["dream-cli<br/>dream-cli/Cargo.toml:25"] --> CORE["dream-core<br/>dream-core/Cargo.toml:17"]
    CLI --> GATE["dream-gate<br/>dream-gate/Cargo.toml:23"]
    CORE --> GATE
    CORE --> FF["failure-fingerprint<br/>dream-core/Cargo.toml:18"]
    GATE --> PW["pairwise-gate<br/>dream-gate/Cargo.toml:23"]
    GATE --> FF
```

**What it shows.** Versioned path dependencies between workspace members: no cycles, and
the two leaves have no internal dependencies of their own
(`crates/pairwise-gate/Cargo.toml:20`, `crates/failure-fingerprint/Cargo.toml:13`).
**Why it is this way.** The gate is separated from the impure loop so its decision function can
stay pure and be rerun during replay
(`crates/dream-gate/src/gate.rs:8`). The selection policies and the failure
identity are separate crates because each is useful to a loop that has no
other part of this engine (`README.md:75`, `README.md:76`).
The four core libraries in ADR-DE-002 plus the CLI make five workspace members; these counts describe different scopes and are consistent (`docs/adr/ADR-DE-002-licence-and-provenance.md:23`, `../dream-engine/Cargo.toml:3`).
Resolved since the earlier stamp: the README crate table no longer calls
`dream-gate` Planned; it describes the built gate (`README.md:77`), which
`dream-core` depends on (`crates/dream-core/Cargo.toml:17`).

## DE-11.2 The licence boundary

```mermaid
flowchart TB
    subgraph PERM["Declared MIT OR Apache-2.0"]
        direction TB
        C1["dream-core"]
        C2["dream-gate"]
        C3["pairwise-gate"]
        C4["failure-fingerprint"]
        C5["dream-cli"]
    end
    BAN["cargo-deny: allow list of permissive licences, ban list of host-tying crates<br/>deny.toml:10 and deny.toml:26"]
    subgraph PRODUCT["External product"]
        direction TB
        AD["Adapter: implements the seams, strong-copyleft licence<br/>ADR-DE-002 line 28"]
        ENV["Private encryption envelope, never published<br/>ADR-DE-002 line 30"]
    end
    AD -->|"depends on"| C1
    AD -->|"depends on"| C2
    BAN -.->|"checks resolved dependencies against policy"| PERM
    C1 -.->|"never depends on"| AD
```

**What it shows.** Dependencies cross the boundary in one direction only: the
adapter depends on the core, and a permissive crate never depends on the
adapter (`docs/adr/ADR-DE-002-licence-and-provenance.md:34`).
**Why it is this way.** The licence list in `deny.toml` has no deny list; anything
not allowed is denied, which keeps copyleft licences out of the graph
(`deny.toml:7`). The ban list also names crates that would tie the engine to one
host, such as database or client crates for a particular forum stack
(`deny.toml:26`).
Resolved since the earlier stamp: the ADR now says `cargo deny check` runs in
CI (`docs/adr/ADR-DE-002-licence-and-provenance.md:42`), and CI has the `deny`
job (`.github/workflows/ci.yml:71`).
**Debt:** the workspace declares a minimum Rust version of 1.89 (`../dream-engine/Cargo.toml:7`)
because `NightLock` uses `File::try_lock`, but every CI job installs `stable`
(`.github/workflows/ci.yml:26`); nothing builds on 1.89, so the declared floor is
untested.
**Open:** nothing in this repository verifies that the product's adapter
follows the rule in the other direction; the boundary is a convention plus a
dependency ban in the permissive tree.
