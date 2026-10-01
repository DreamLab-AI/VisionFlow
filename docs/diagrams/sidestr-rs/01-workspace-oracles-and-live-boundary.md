---
id: SR-01
title: sidestr-rs workspace, reference-oracle ladder, and live estate boundary
area: sidestr-rs
governing:
  - ../sidestr-rs/docs/adr/ADR-0001-reference-oracle-is-the-compatibility-contract.md
  - ../sidestr-rs/docs/adr/ADR-0002-keep-optional-execution-and-services-behind-crate-boundaries.md
  - docs/adr/ADR-2012-sidestr-settlement-is-ecosystem-canon.md
adrs: [ADR-2012, ADR-2096, ADR-2112]
sources:
  - ../sidestr-rs/Cargo.toml
  - ../sidestr-rs/README.md
  - ../sidestr-rs/.github/workflows/ci.yml
  - ../sidestr-rs/docs/adr/ADR-0001-reference-oracle-is-the-compatibility-contract.md
  - ../sidestr-rs/docs/adr/ADR-0002-keep-optional-execution-and-services-behind-crate-boundaries.md
  - docs/adr/ADR-2012-sidestr-settlement-is-ecosystem-canon.md
  - ../project/agentbox/config/sidechain/README.md
  - ../project/agentbox/config/sidechain/dreamlab/chain.json
  - ../project/docs/TODO-unified.md
verified_commit: {sidestr-rs: 3aadeb7a26ff60c18614113bb986c1ba0d33d115, visionflow: e5826a84e37651e3a915fdf6df0b5ad30f2b033d, visionclaw: d5ecd38a2de3012e42509c27b950f6cd17e3fee4, agentbox: 5d5d083e2e77ea448d27e6e2dae95822eb960922}
---

## For developers

`sidestr-rs` now has ten crates. Seven published crates carry the reusable chain, wallet, Nostr, federation, agent and Hitch surfaces; EVM and the two reserve crates remain separate, unpublished components. Compatibility is judged against pinned upstream revisions and regenerated fixtures rather than inferred from similar APIs.

## For the business

New source expands what the estate can validate and eventually operate, but it does not change what is live. `sidestr:dreamlab` is still a test-only chain produced by the upstream JavaScript engine. Publication, source completeness and deployment are tracked as separate facts.

## SR-01.1 Workspace ownership

```mermaid
flowchart LR
    subgraph published["Seven published reusable crates"]
        C["core"]
        H["header"] -. "optional core feature" .-> C
        W["wallet"] --> C
        W --> H
        N["nostr"] --> C
        R["round"] --> C
        R --> N
        A["agent"] --> W
        A --> N
        CH["hitch"] --> C
    end
    subgraph private["Three unpublished components"]
        E["evm overlay"]
        RA["origin-neutral reserve"] --> L["Liquid adapter"]
    end
    E --> C
    E --> W
    WS["ten members<br/>../sidestr-rs/Cargo.toml:3"] --> published
    WS --> private
```

**What it shows.** The root manifest names ten members and explicitly keeps EVM, reserve attestations and the Liquid adapter out of the published core graph (`../sidestr-rs/Cargo.toml:28`, `../sidestr-rs/Cargo.toml:32`).

**Why it is this way.** Optional execution and origin adapters can evolve without pulling their dependency trees into every validator. The local crate-boundary decision records this separation (`../sidestr-rs/docs/adr/ADR-0002-keep-optional-execution-and-services-behind-crate-boundaries.md:27`).

## SR-01.2 The compatibility ladder

```mermaid
flowchart TB
    P["Pinned revisions<br/>spec, schema, blaketestnode, Hitch<br/>../sidestr-rs/.github/workflows/ci.yml:14"]
    RT["Reference project tests<br/>../sidestr-rs/.github/workflows/ci.yml:143"]
    FX["EVM fixtures regenerated<br/>and checked for drift<br/>../sidestr-rs/.github/workflows/ci.yml:156"]
    O["Rust oracle and interop suites<br/>../sidestr-rs/.github/workflows/ci.yml:178"]
    Q["fmt, clippy, tests, rustdoc,<br/>no_std and wasm gates<br/>../sidestr-rs/.github/workflows/ci.yml:41"]
    P --> RT --> FX --> O --> Q
```

**What it shows.** CI checks the reference repositories out at fixed revisions, runs their relevant tests, regenerates the EVM corpus, then runs the Rust workspace against those sources (`../sidestr-rs/.github/workflows/ci.yml:97`, `../sidestr-rs/.github/workflows/ci.yml:162`).

**Why it is this way.** The reference engine is the compatibility contract. A green Rust-only test suite cannot prove wire or consensus parity on its own (`../sidestr-rs/docs/adr/ADR-0001-reference-oracle-is-the-compatibility-contract.md:27`).

## SR-01.3 Source, publication and activation are separate states

```mermaid
stateDiagram-v2
    [*] --> Source
    Source: Source implemented and reference-tested
    Source --> Published: seven reusable crates released
    Source --> Unpublished: EVM and reserve components private
    Published --> Adopted: host integrates a released crate
    Unpublished --> Adopted: explicit private integration
    Adopted --> Active: chain document and runtime evidence
    note right of Source
        docs/adr/ADR-2012-sidestr-settlement-is-ecosystem-canon.md:98
    end note
    Active --> [*]
```

**What it shows.** VisionFlow records source implementation as partial while keeping the ecosystem decision proposed and activation inactive (`docs/adr/ADR-2012-sidestr-settlement-is-ecosystem-canon.md:98`, `docs/adr/ADR-2012-sidestr-settlement-is-ecosystem-canon.md:112`).

**Why it is this way.** A library can be correct and published without any estate service importing it or any chain activating its rules. The master board closes source parity in N-9 and keeps estate adoption open in N-10 (`../project/docs/TODO-unified.md:100`, `../project/docs/TODO-unified.md:101`).

**Invariant:** source evidence may promote implementation status; it cannot promote activation status.

## SR-01.4 The live boundary

```mermaid
flowchart LR
    AB["agentbox supervisor<br/>config/sidechain/README.md:52"] --> JP["upstream JavaScript producer<br/>config/sidechain/README.md:60"]
    AB --> M["mirror"]
    AB --> F["Rust faucet"]
    JP --> D["sidestr:dreamlab<br/>testnet4, level 1<br/>config/sidechain/dreamlab/chain.json:5"]
    M --> D
    F --> D
    RS["sidestr-rs libraries"] -. "validated components;<br/>native producer not deployed" .-> AB
```

**What it shows.** Agentbox supervises the producer, mirror and faucet, but the producer runs the upstream JavaScript engine (`../project/agentbox/config/sidechain/README.md:52`, `../project/agentbox/config/sidechain/README.md:60`). The sealed chain is level 1, and its document reaches `containmentDigest` without a `rules` field (`../project/agentbox/config/sidechain/dreamlab/chain.json:5`, `../project/agentbox/config/sidechain/dreamlab/chain.json:28`).

**Why it is this way.** The Rust workspace is currently a validated library component. A native producer, authenticated chain proxy and host adoption need their own implementation and runtime receipts (`../project/agentbox/config/sidechain/README.md:77`).

**Open:** deploy and prove a native producer before describing the Rust engine as the running estate chain.
