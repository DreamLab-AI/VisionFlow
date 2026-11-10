---
id: NF-15
title: Two-chain wallets — identity, freshness and delivery
area: nostr-rust-forum
governing:
  - ../nostr-rust-forum/docs/adr/ADR-2021-two-pinned-sidestr-chains-dreamlab-and-dreamlab-txbt4.md
adrs: [ADR-2021]
sources:
  - ../nostr-rust-forum/docs/adr/ADR-2021-two-pinned-sidestr-chains-dreamlab-and-dreamlab-txbt4.md
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/wallet/chain.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/wallet/profile.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/wallet/mod.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/wallet/relays.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/poker/mod.rs
  - ../nostr-rust-forum/crates/nostr-bbs-poker-citizen/src/chain.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/components/poker_schedule.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/components/tip_button.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/components/thread_view.rs
verified_commit: {nostr-rust-forum: 6064622dbb1a7cc1ba2d072109f9e3e8a96157df}
---

## For developers

These panels follow the website-consumed kit revision. The [section audit](../../estate-review/2026-11-01-forum-multichain-wallet.md) also compares the newer ADR close-out, which changes the activation trigger without changing these decisions.

## For the business

Members can use the same identity on two configured testnet chains. Reliable receipts need a known asset, fresh state and recovery across account changes and interrupted delivery. The compiled chain pins provide only part of that assurance.

## NF-15.1 Profiles choose among compiled chains

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/wallet/chain.rs:38
    A["Two compiled pins<br/>DREAM and dreamlab-txbt4"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/wallet/profile.rs:188
    B["Runtime selects known pins<br/>unknown chains are ignored"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/wallet/profile.rs:215
    C["DEBT: asset override also changes DREAM<br/>while its default ticker stays DREAM"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/wallet/profile.rs:279
    D["Typed entry parse failure drops the entry<br/>if no entries survive, use DREAM"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/wallet/profile.rs:136
    E["DEBT: URL checks only inspect prefixes<br/>invalid hosts can pass"]
    A --> B
    B --> C
    B --> D
    B --> E
```

**What it shows:** The native cases preserve the no-third-chain boundary but contradict the fixed DREAM asset assertion. A malformed field type can change which chain is offered.
**Why it is this way:** ADR-2021 separates chain trust from deployment settings, but asset identity and malformed configuration need stronger validation.

## NF-15.2 Replay validates history but does not establish freshness

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/wallet/mod.rs:379
    A["Load mirror blocks with cache disabled<br/>and a time-bucket query"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/wallet/chain.rs:342
    B["Replay against the compiled document<br/>dispatch Stock or Blake2bV2"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/wallet/mod.rs:397
    C["Successful replay replaces snapshot<br/>loaded_at records download time"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/wallet/mod.rs:680
    D["Ready requires snapshot and signer<br/>no maximum snapshot age"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/wallet/mod.rs:397
    E["DEBT: no prior-tip rollback check<br/>a valid older prefix can replace newer state"]
    A --> B
    B --> C
    C --> D
    D --> E
```

**What it shows:** A native test accepts height 26 and then a valid height-zero prefix. The replacement path is source evidence; this audit did not execute a browser rollback.
**Why it is this way:** A chain pin establishes identity. It does not establish that a mirror supplies the latest state or that its signer is honest.

## NF-15.3 Pending state is partitioned but migration is fallible

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/wallet/mod.rs:239
    A["Create one wallet store per profile<br/>bind all stores to the current account"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/wallet/mod.rs:308
    B["Pending storage key contains<br/>chain ID and account pubkey"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/wallet/mod.rs:454
    C["DREAM may restore the legacy key<br/>then save under the new key"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/wallet/mod.rs:475
    D["DEBT: remove legacy key after save attempt<br/>write errors are ignored"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/wallet/mod.rs:431
    E["Reconcile inclusion or old unspent inputs<br/>conflicting inputs have no terminal state"]
    A --> B
    B --> C
    C --> D
    D --> E
```

**What it shows:** Per-chain keys are implemented. Source inspection finds no successful-write requirement before migration deletion and no conflict-expiry branch.
**Why it is this way:** ADR-2021 preserves old DREAM pending lists. Migration and receipt recovery need explicit success and failure states.

## NF-15.4 Signing and delivery have separate guarantees

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/wallet/mod.rs:632
    A["Extension signs the selected chain<br/>returned transaction and signatures checked"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/wallet/mod.rs:734
    B["Build from snapshot and held coins<br/>then await signing and relay delivery"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/wallet/relays.rs:135
    C["Publish kind 23500 to configured relays<br/>count matching accepted event IDs"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/wallet/mod.rs:661
    D["DEBT: zero ACKs says nothing was sent<br/>timeout cannot establish non-delivery"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/wallet/mod.rs:673
    E["Only after ACK insert pending and persist<br/>no account epoch or early coin reservation"]
    B --> A
    A --> C
    C -->|No accepted ACK| D
    C -->|Accepted ACK| E
```

**What it shows:** Existing signer tests reject changed transactions and invalid signatures. The late reservation and account association are source-based concurrency concerns, not reproduced browser exploits.
**Why it is this way:** Relay acknowledgement is an intermediate receipt. Persisted intent and coin reservation should precede an operation that may already have reached a peer.

## NF-15.5 Poker binds a table through configuration

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/poker/mod.rs:189
    A["House key priority: chain map<br/>then profile then DREAM scalar"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/poker/mod.rs:265
    B["Asset and house key create a table<br/>bound to that chain wallet"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-poker-citizen/src/chain.rs:176
    C["House verifies producer document equality<br/>and replays the compiled pin"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/components/poker_schedule.rs:160
    D["Calendar event adds the chain tag<br/>events link to its table fragment"]
    %% Source: ../nostr-rust-forum/docs/adr/ADR-2021-two-pinned-sidestr-chains-dreamlab-and-dreamlab-txbt4.md:70
    E["Protocol 1 still names the asset dream<br/>chain and asset binding depend on the seat"]
    A --> B
    B --> C
    B --> D
    B --> E
```

**What it shows:** Two correctly configured identities distinguish the current tables. The wire tag alone does not identify a chain or asset; the preceding poker audit covers payment and recovery gaps.
**Why it is this way:** ADR-2021 deliberately retains protocol 1. Configuration validation and a verifiable service capability receipt should guard that compatibility choice.

## NF-15.6 Tips select and display each asset separately

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/components/tip_button.rs:126
    A["Offer one token row per asset profile<br/>default highlight follows wallet selection"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/components/tip_button.rs:156
    B["Load each asset chain for tip totals<br/>selection can differ from wallet page"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/components/tip_button.rs:97
    C["Keep totals separate by asset<br/>display configured tickers"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/components/thread_view.rs:185
    D["Tip control sits at right of reply row<br/>or reply footer"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/components/tip_button.rs:421
    E["Hide own-post tip button<br/>send via selected token wallet"]
    A --> B
    B --> C
    C --> D
    D --> E
```

**What it shows:** The consumed client implements the picker, separate totals and own-post hiding. Each asset chain adds a download and replay, and optimistic pending totals remain distinct from confirmed inclusion.
**Why it is this way:** The useful outcome is a recognisable contribution receipt in the conversation. Clear asset identity and payment status matter more than adding another chain.
