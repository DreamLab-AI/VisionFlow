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
verified_commit: {sidestr-rs: cd177ecc08f4907541a55518263bae342f7ba8e5, visionflow: e5987acc8337ddd64c72f775750d61fef46d8e0b, visionclaw: 7d3ea2edb067432a57e6fe1fd951fd8254380bb8, agentbox: 5ab197a9d49e9721b85b791bf9efe30842c9e047}
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
    P["Pinned revisions<br/>spec, schema, blaketestnode, Hitch, Reef<br/>../sidestr-rs/.github/workflows/ci.yml:14"]
    RT["Reference project tests<br/>../sidestr-rs/.github/workflows/ci.yml:152"]
    FX["EVM fixtures regenerated<br/>and checked for drift<br/>../sidestr-rs/.github/workflows/ci.yml:165"]
    O["Rust oracle and interop suites<br/>../sidestr-rs/.github/workflows/ci.yml:187"]
    Q["fmt, clippy, tests, rustdoc,<br/>no_std and wasm gates<br/>../sidestr-rs/.github/workflows/ci.yml:43"]
    P --> RT --> FX --> O --> Q
```

**What it shows.** CI checks the reference repositories out at fixed revisions, runs their relevant tests, regenerates the EVM corpus, then runs the Rust workspace against those sources (`../sidestr-rs/.github/workflows/ci.yml:99`, `../sidestr-rs/.github/workflows/ci.yml:171`). Since `cd177ecc` a fifth reference joins the ladder: Reef's BIP 21 parser, pinned at `648487a`, is the oracle for `sidestr-wallet`'s payment requests and is handed to the oracle suites alongside Hitch (`../sidestr-rs/.github/workflows/ci.yml:23`, `../sidestr-rs/.github/workflows/ci.yml:127`, `../sidestr-rs/.github/workflows/ci.yml:199`). The same push moved the Hitch pin to `62f8e39` and blaketestnode to `f1da4a6` (`../sidestr-rs/.github/workflows/ci.yml:19`, `../sidestr-rs/.github/workflows/ci.yml:22`).

**Why it is this way.** The reference engine is the compatibility contract. A green Rust-only test suite cannot prove wire or consensus parity on its own (`../sidestr-rs/docs/adr/ADR-0001-reference-oracle-is-the-compatibility-contract.md:27`). **Drift (ADR-0001 vs CI):** the record says CI pins "all four reference repositories", but since `cd177ecc` the workflow pins five: spec, schema, blaketestnode, Hitch and Reef (`../sidestr-rs/docs/adr/ADR-0001-reference-oracle-is-the-compatibility-contract.md:27`, `../sidestr-rs/.github/workflows/ci.yml:24`).

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

**Why it is this way.** A library can be correct and published without any estate service importing it or any chain activating its rules. The master board closes source parity in N-9 and keeps estate adoption open in N-10 (`../project/docs/TODO-unified.md:100`, `../project/docs/TODO-unified.md:101`). Two releases after that closure, `sidestr-wallet` 0.5.1 (BIP 21 with the Reef oracle) and `sidestr-hitch` 0.2.0, are recorded as an annotation on the closed N-9 row, with adoption left under N-10 (`../project/docs/TODO-unified.md:114`).

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
