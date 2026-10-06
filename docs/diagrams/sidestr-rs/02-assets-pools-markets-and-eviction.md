---
id: SR-02
title: Ordered asset, pool and market rules with exact producer eviction
area: sidestr-rs
governing:
  - ../sidestr-rs/docs/adr/ADR-0001-reference-oracle-is-the-compatibility-contract.md
  - docs/adr/ADR-2012-sidestr-settlement-is-ecosystem-canon.md
adrs: [ADR-2012, ADR-2096]
sources:
  - ../sidestr-rs/sidestr-core/src/rules.rs
  - ../sidestr-rs/sidestr-core/src/overlays.rs
  - ../sidestr-rs/sidestr-core/src/assets.rs
  - ../sidestr-rs/sidestr-core/src/pool.rs
  - ../sidestr-rs/sidestr-core/src/markets.rs
  - ../sidestr-rs/sidestr-core/src/state.rs
  - ../sidestr-rs/sidestr-core/tests/eviction.rs
  - ../sidestr-rs/README.md
  - ../project/agentbox/config/sidechain/dreamlab/chain.json
verified_commit: {sidestr-rs: a7aadd68d536167507e00e1ce6237fbecb5b46fa, agentbox: 5ab197a9d49e9721b85b791bf9efe30842c9e047}
---

## For developers

`BlockRule` is the extension seam for rule-local validation, coinbase allowances and applied-block commits. The lightweight overlays always execute as assets, pool, then markets, and producer selection evaluates candidates against temporary state before final block validation.

## For the business

Source now represents issued assets, automated pools and binary markets without changing the base UTXO ledger. These are source capabilities only: the live estate chain has no rules field and therefore offers none of them.

## SR-02.1 The ordered rule pipeline

```mermaid
flowchart LR
    TX["candidate transactions"] --> AS["assets<br/>conservation and carry trace<br/>sidestr-core/src/overlays.rs:1"]
    AS --> PO["pool<br/>constant product"]
    PO --> MA["markets<br/>collateral lifecycle"]
    MA --> CB["coinbase allowances"]
    CB --> CM["commit only after<br/>block application"]
```

**What it shows.** The compositor installs assets before pool before markets so later rules can inspect the same candidate block's asset carry trace (`../sidestr-rs/sidestr-core/src/overlays.rs:1`, `../sidestr-rs/sidestr-core/src/overlays.rs:65`). `BlockRule` separates checking from committing applied state (`../sidestr-rs/sidestr-core/src/rules.rs:534`).

**Why it is this way.** Pool shares and market outcome tokens are authorised asset mints. Fixed ordering lets those rules grant a mint while the asset rule still enforces conservation.

**Invariant:** overlay order is consensus behaviour and cannot be rearranged as an implementation detail.

## SR-02.2 Assets and constant-product pools

```mermaid
flowchart TB
    I["issue or transfer records"] --> T["asset input and output tallies<br/>sidestr-core/src/assets.rs:160"]
    T --> C{"conserved?"}
    C -->|yes| TRACE["carry trace for later rules"]
    C -->|no| MP{"mint policy allows<br/>pool or market mint?"}
    MP -->|no| REF["refuse transaction"]
    MP -->|yes| TRACE
    TRACE --> P["pool transition"]
    P --> K["integer constant-product<br/>boundary"]
```

**What it shows.** The asset rule classifies transactions, totals inputs and outputs, and delegates exceptional minting to a policy supplied by the pool and market views (`../sidestr-rs/sidestr-core/src/assets.rs:160`, `../sidestr-rs/sidestr-core/src/overlays.rs:34`). Pools then enforce their transition and constant-product boundary (`../sidestr-rs/sidestr-core/src/pool.rs:80`, `../sidestr-rs/sidestr-core/src/pool.rs:213`).

**Why it is this way.** One conservation rule stays authoritative while purpose-built rules can create only the assets their own state transition justifies.

## SR-02.3 Binary market lifecycle

```mermaid
stateDiagram-v2
    [*] --> Open
    Open --> Open: split locks collateral and mints YES plus NO
    Open --> Open: merge burns a complete pair and releases collateral
    Open --> Resolved: resolver proof within expiry plus grace
    Open --> Refunding: redeem pairs after expiry plus grace
    Resolved --> Resolved: split, merge, or redeem the winning outcome
    Refunding --> Refunding: redeem paired outcomes at half a sat each
    note right of Open
        sidestr-core/src/markets.rs:138
    end note
```

**What it shows.** A market is opened with its collateral and resolver terms, then supports split, merge, resolve, redeem and refund transitions (`../sidestr-rs/sidestr-core/src/markets.rs:138`, `../sidestr-rs/sidestr-core/src/markets.rs:260`). Resolver proof is checked before the status can advance (`../sidestr-rs/sidestr-core/src/markets.rs:202`).

**Why it is this way.** Every creation or release of outcome assets stays traceable to locked collateral and the market's current status.

## SR-02.4 Admission is not block eligibility

```mermaid
sequenceDiagram
    participant A as submit<br/>sidestr-core/src/state.rs:636
    participant M as mempool
    participant S as temporary overlay state
    participant B as block builder
    A->>A: refuse immature BIP 68 or nLockTime<br/>state.rs:717-728
    A->>M: store eligible transaction<br/>state.rs:729-732
    M->>S: replay candidate in order
    alt valid in the candidate block
        S-->>B: keep transaction
    else invalid under block rule
        S-->>M: evict exact serialised transaction
    end
    B->>B: validate final block and apply once
    Note over M,B: sidestr-core/src/state.rs:874
```

**What it shows.** Admission refuses a transaction whose BIP 68 relative lock or `nLockTime` has not matured at the next height, so an immature channel sweep or HTLC refund never enters the mempool and production is never asked to evict it (`../sidestr-rs/sidestr-core/src/state.rs:628`, `../sidestr-rs/sidestr-core/src/state.rs:721`). When admission and block validation still disagree, the builder walks the mempool in order and removes transactions that admission accepted but current block state rejects (`../sidestr-rs/sidestr-core/src/state.rs:874`). Eviction matches the exact transaction bytes, so a witness-repaired transaction with the same txid is not removed accidentally (`../sidestr-rs/sidestr-core/src/state.rs:759`, `../sidestr-rs/sidestr-core/tests/eviction.rs:16`).

**Why it is this way.** Rule state changes as earlier candidates enter a block. Turning immature locks away at admission means they can be submitted again once their height arrives instead of being evicted from a produced block (`../sidestr-rs/sidestr-core/src/state.rs:631`). Exact-byte eviction prevents one invalid witness form from suppressing a repaired form that shares its txid.

**Open:** assets, pools and markets remain inactive on `sidestr:dreamlab`; its sealed document reaches `containmentDigest` without a `rules` field (`../project/agentbox/config/sidechain/dreamlab/chain.json:5`, `../project/agentbox/config/sidechain/dreamlab/chain.json:28`).
