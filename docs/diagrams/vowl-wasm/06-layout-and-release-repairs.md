---
id: VW-06
title: Layout termination, reload state and release retry repairs
area: vowl-wasm
governing: []
adrs: [vowl-wasm:ADR-001]
sources:
  - ../vowl-wasm/src/layout/quadtree.rs
  - ../vowl-wasm/src/layout/csr_sim.rs
  - ../vowl-wasm/src/bindings/mod.rs
  - ../vowl-wasm/.github/workflows/release.yml
verified_commit: 0779174612a3942b15331356422f58bf8379912b
---

## For developers

This scoped successor covers VW-03.3, VW-04.1 and VW-05.3. Native regressions and local WASM tests passed. The [official release run](https://github.com/DreamLab-AI/vowl-wasm/actions/runs/41589928674) subsequently passed the pinned Rust 1.99.0 build, inventory and tests and uploaded the GitHub bundle; npm publication returned a permission/package E404. At 0779174 the workflow names the cause, a missing npm Trusted Publisher entry, and v0.3.0 was the first version published through it (release.yml:37-45, release.yml:19-27); see VW-05.3.

## For the business

Coincident graph nodes no longer cause unbounded subdivision. Loading another graph clears interaction state, and a failed npm publication can retry the existing reviewed release bundle.

## VW-06.1 Bound subdivision and reject non-finite controls

```mermaid
flowchart TB
    POINTS["coincident points"] --> LEAF["depth-limited leaf<br/>quadtree.rs:192"]
    PARAM["simulation control input"] --> FINITE["ignore non-finite values<br/>csr_sim.rs:845"]
    FINITE --> REHEAT["same rule for reheat<br/>csr_sim.rs:934"]
```

**What it shows:** legacy subdivision terminates, while CSR setters preserve valid state when supplied NaN or infinity. **Why it is this way:** a valid graph may contain many identical positions. Invalid controls must not poison future frames.

**Invariant:** the legacy tree stops subdividing at its fixed depth bound (`quadtree.rs:192`).

## VW-06.2 A newly loaded ontology starts a new interaction state

```mermaid
flowchart LR
    LOAD["load parsed ontology"] --> PINS["new PinManager<br/>bindings/mod.rs:76"]
    PINS --> SIM["simulation.reset: alpha restarts, settings kept<br/>bindings/mod.rs:80"]
    SIM --> CH["CHANGED at 0779174 (0.3.0): was a default ForceSimulation,<br/>which dropped setLinkDistance, setChargeStrength and setCenter"]
```

**What it shows:** a reused WebVowl instance does not retain old pins or a settled alpha, but keeps the caller's simulation settings (bindings/mod.rs:54-60). **Why it is this way:** node identifiers reused by another graph must not inherit an earlier graph's user interaction.

**Invariant:** successful loading replaces the pin manager and restarts the simulation (`bindings/mod.rs:76`, `bindings/mod.rs:80`).

## VW-06.3 Reuse the immutable release archive on registry retry

```mermaid
flowchart TB
    REVIEW["verify reviewed inventory<br/>release.yml:141"] --> EXISTS["existing GitHub bundle<br/>release.yml:155"]
    EXISTS --> CHECK["download and check checksum<br/>release.yml:160"]
    CHECK --> NPM["publish verified archive if absent<br/>release.yml:254"]
```

**What it shows:** an existing GitHub release no longer skips the independent npm publication step. **Why it is this way:** retrying registry delivery must preserve the reviewed archive rather than silently rebuild it with a different toolchain.

**Invariant:** archive checksum validation precedes registry publication (`release.yml:163`).
