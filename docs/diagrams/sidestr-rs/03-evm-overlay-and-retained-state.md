---
id: SR-03
title: In-chain EVM execution, withdrawal accounting, and retained snapshots
area: sidestr-rs
governing:
  - ../sidestr-rs/docs/adr/ADR-0002-keep-optional-execution-and-services-behind-crate-boundaries.md
  - docs/adr/ADR-2012-sidestr-settlement-is-ecosystem-canon.md
  - ../project/agentbox/docs/adr/ADR-2096-sidestr-sidechains-are-the-sole-value-instrument.md
adrs: [ADR-2012, ADR-2096]
sources:
  - ../sidestr-rs/sidestr-evm/src/lib.rs
  - ../sidestr-rs/sidestr-evm/src/rule.rs
  - ../sidestr-rs/sidestr-evm/src/state.rs
  - ../sidestr-rs/sidestr-evm/tests/oracle.rs
  - ../sidestr-rs/sidestr-evm/tests/chain.rs
  - ../sidestr-rs/.github/workflows/ci.yml
  - ../sidestr-rs/README.md
  - docs/adr/ADR-2012-sidestr-settlement-is-ecosystem-canon.md
  - ../project/agentbox/docs/adr/ADR-2096-sidestr-sidechains-are-the-sole-value-instrument.md
  - ../project/agentbox/config/sidechain/dreamlab/chain.json
verified_commit: {sidestr-rs: 3aadeb7a26ff60c18614113bb986c1ba0d33d115, visionflow: e5826a84e37651e3a915fdf6df0b5ad30f2b033d, agentbox: 5d5d083e2e77ea448d27e6e2dae95822eb960922}
---

## For developers

`sidestr-evm` implements the chain's optional `evm` rule using `revm`. Sidechain transactions carry deposits and Ethereum transactions; the coinbase commits the resulting state root and pays withdrawals. Rule state commits only after the enclosing sidechain block is accepted.

## For the business

This adds smart-contract execution inside a sidestr chain. It does not create a separate Ethereum settlement rail: value enters and leaves as the same chain's sats, and the chain's signers still order and validate every block. The crate is unpublished and the estate chain has not activated it.

## SR-03.1 One chain, two coordinated state views

```mermaid
flowchart LR
    STX["sidechain transaction<br/>sidestr-evm/src/lib.rs:19"] --> DEP["deposit<br/>1 sat equals 1 gwei"]
    STX --> CAR["EVM carrier transaction"]
    DEP --> EVM["revm world state"]
    CAR --> EVM
    EVM --> W["withdrawal obligations"]
    EVM --> ROOT["state root"]
    W --> CB["sidechain coinbase pays sats"]
    ROOT --> CB
    CB --> BLOCK["accepted sidestr block"]
```

**What it shows.** The overlay recognises deposits, carried Ethereum transactions, withdrawals and the root committed by the sidechain coinbase (`../sidestr-rs/sidestr-evm/src/lib.rs:19`). One sat maps to one gwei inside the overlay (`../sidestr-rs/sidestr-evm/src/lib.rs:10`).

**Why it is this way.** Deposits and withdrawals remain accountable in the host chain's block. Estate policy therefore treats EVM as in-chain execution rather than a second value rail (`../project/agentbox/docs/adr/ADR-2096-sidestr-sidechains-are-the-sole-value-instrument.md:116`).

**Invariant:** an EVM withdrawal is valid only when the same block's coinbase pays the required sidechain satoshis.

## SR-03.2 Check, prepare, then commit

```mermaid
sequenceDiagram
    participant R as EvmRule
    participant S as staged EVM state
    participant C as sidechain coinbase
    R->>S: execute deposits and carriers
    S-->>R: root, receipts, withdrawals
    R->>C: verify root and withdrawal outputs
    alt enclosing block applied
        R->>S: commit by block hash
    else refused or sequencing-only
        R->>S: discard staged state
    end
    Note over R,S: sidestr-evm/src/rule.rs:62
```

**What it shows.** `EvmRule` exposes validation, coinbase allowance and applied-block commit through the core rule interface (`../sidestr-rs/sidestr-evm/src/rule.rs:62`). Preparation checks both the root and withdrawal payments before state can commit (`../sidestr-rs/sidestr-evm/src/state.rs:459`).

**Why it is this way.** Validation and producer sequencing must not mutate durable state. The crate deliberately commits only an applied block (`../sidestr-rs/sidestr-evm/src/lib.rs:78`).

## SR-03.3 Retained-height snapshots

```mermaid
flowchart TB
    BH["retained block height"] --> SNAP["versioned snapshot<br/>sidestr-evm/src/state.rs:161"]
    SNAP --> W["accounts and storage worlds"]
    SNAP --> T["timestamps and block hashes"]
    SNAP --> RC["receipts"]
    SNAP --> META["configuration and root"]
    RESTORE["restore"] --> V{"version, configuration,<br/>root and time agree?"}
    V -->|yes| LIVE["resume at retained height"]
    V -->|no| REF["refuse snapshot"]
```

**What it shows.** A snapshot contains the versioned execution worlds, times, block hashes and receipts (`../sidestr-rs/sidestr-evm/src/state.rs:161`). Restore verifies version, configuration, root and time, then clears transient execution state (`../sidestr-rs/sidestr-evm/src/state.rs:291`).

**Why it is this way.** A follower can resume from any retained sidechain height without treating unverified serialised state as consensus truth.

## SR-03.4 Evidence and activation boundary

```mermaid
flowchart LR
    J["ethereumjs reference<br/>fixture generator<br/>.github/workflows/ci.yml:156"] --> F["accepted and refused<br/>block fixtures"]
    F --> R["Rust replay"]
    R --> EQ["roots, withdrawals,<br/>receipts and refusals agree"]
    EQ --> SRC["source capability"]
    SRC -. "separate release and deployment" .-> LIVE["estate activation"]
```

**What it shows.** CI regenerates the EVM fixtures against the pinned reference and fails on drift (`../sidestr-rs/.github/workflows/ci.yml:156`). The oracle suite checks roots, withdrawals, hashes, receipts and refused blocks (`../sidestr-rs/sidestr-evm/tests/oracle.rs:1`).

**Why it is this way.** Byte-level parity supports a source claim. It does not publish the crate or alter a sealed chain document. VisionFlow records both limits explicitly (`docs/adr/ADR-2012-sidestr-settlement-is-ecosystem-canon.md:106`).

**Open:** `sidestr-evm` is unpublished, and the `sidestr:dreamlab` document reaches `containmentDigest` without an `evm` rule (`../project/agentbox/config/sidechain/dreamlab/chain.json:5`, `../project/agentbox/config/sidechain/dreamlab/chain.json:28`).
