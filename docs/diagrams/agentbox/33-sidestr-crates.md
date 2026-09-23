---
id: AB-33
title: The four sidestr crates — dependency graph, the AGPL boundary, the ports and their oracles
area: agentbox
governing:
  - ../project/agentbox/docs/BASELINE-container.md
adrs: [ADR-2030, ADR-2096, ADR-2101, ADR-2103, ADR-2106]
sources:
  - ../project/agentbox/crates/sidestr/Cargo.toml
  - ../project/agentbox/crates/sidestr/sidestr-header/Cargo.toml
  - ../project/agentbox/crates/sidestr/sidestr-header/src/lib.rs
  - ../project/agentbox/crates/sidestr/sidestr-core/Cargo.toml
  - ../project/agentbox/crates/sidestr/sidestr-core/src/lib.rs
  - ../project/agentbox/crates/sidestr/sidestr-nostr/Cargo.toml
  - ../project/agentbox/crates/sidestr/sidestr-nostr/src/lib.rs
  - ../project/agentbox/crates/sidestr/sidestr-wallet/Cargo.toml
  - ../project/agentbox/crates/sidestr/sidestr-wallet/src/lib.rs
  - ../project/agentbox/crates/sidestr/sidestr-core/tests/oracle.rs
  - ../project/agentbox/crates/sidestr/sidestr-core/tests/interop.rs
  - ../project/agentbox/crates/sidestr/sidestr-core/tests/xcheck.mjs
  - ../project/agentbox/crates/sidestr/sidestr-wallet/tests/oracle.rs
  - ../project/agentbox/crates/sidestr/sidestr-nostr/tests/live.rs
  - ../project/agentbox/.github/workflows/sidestr-crates.yml
  - ../project/agentbox/docs/adr/ADR-2106-sidestr-crates-are-agpl-derivatives-of-siding-published-and-consumed.md
  - ../project/agentbox/docs/developer/licensing.md
  - ../project/agentbox/docs/BASELINE-container.md
verified_commit: ec60a8f14f4544520b4b1f6e8f5de2def4cfedcf
---

## For developers

Four crates under `crates/sidestr/` (`crates/sidestr/Cargo.toml:3-19`), all published at 0.1.0 and all `AGPL-3.0-only` because they are ports of Melvin Carvalho's `siding` rather than clean-room work from prose — ADR-2106 reversed ADR-2096 D2's permissive plan on the day the first seal showed a prose-only port would forgo the reference's tests and fixtures (`ADR-2106-sidestr-crates-are-agpl-derivatives-of-siding-published-and-consumed.md:22-32`). Every departure from the reference is listed in the crate's own rustdoc, and each is held down by an oracle test rather than by assertion.

**Drift (this topic vs agentbox since ad45e7bf8):** the crates this topic cites under `crates/sidestr/` no longer live in agentbox — on 2026-09-23 ADR-2112 moved them with their history and their CI to `DreamLab-AI/sidestr-rs`, leaving a pointer README; they are now five (`sidestr-round` joined) at 0.2.x. Citations here stay true at `ec60a8f14` and are not re-stamped; the current crate graph, oracle ladder and level status are SR-01.

## For the business

The estate can now check its own settlement chain with code it owns, instead of trusting a single reference implementation it did not write. The licence is the copyleft one the original carries, which means these four pieces are not reusable by anyone who wants to keep their own work private — accepted, because the estate is the consumer, and it keeps the estate's obligations to the original author clean.

## AB-33.1 The dependency graph and the AGPL boundary

```mermaid
flowchart TB
    subgraph ws["crates/sidestr workspace - crates/sidestr/Cargo.toml:1"]
        H["sidestr-header 0.1.0, AGPL-3.0-only<br/>no_std, RustCrypto only, NO bitcoin crate<br/>sidestr-header/Cargo.toml:8, sidestr-header/Cargo.toml:26"]
        C["sidestr-core 0.1.0, AGPL-3.0-only<br/>sidestr-core/Cargo.toml:8"]
        N["sidestr-nostr 0.1.0, AGPL-3.0-only<br/>sidestr-nostr/Cargo.toml:8"]
        W["sidestr-wallet 0.1.0, AGPL-3.0-only<br/>sidestr-wallet/Cargo.toml:8"]
    end
    N -->|"path plus version 0.1"| C
    W -->|"path plus version 0.1"| C
    H -.->|"NO EDGE today"| C
    C --> BTC["bitcoin 0.32, bech32, serde, sha2, hex, thiserror<br/>crates/sidestr/Cargo.toml:30"]
    W --> HM["hmac 0.12, the ADR-2101 D3 domain-separated derivation<br/>sidestr-wallet/Cargo.toml:33"]
    subgraph rule["The boundary ADR-2106 D3 draws"]
        R1["consumers take the crates FROM crates.io, never by path,<br/>in any repository other than agentbox<br/>ADR-2106-sidestr-crates-are-agpl-derivatives-of-siding-published-and-consumed.md:46"]
        R2["a consumer that links one is AGPL-3.0 in effect and<br/>DECLARES it - :47-48"]
        R3["NO permissive crate on a crates.io path may depend on one<br/>:49"]
        R4["they live under crates/, never services/, so ADR-2030's<br/>permissive default for services/ is untouched - :50"]
        R1 ~~~ R2 ~~~ R3 ~~~ R4
    end
    ws --> rule
```

**Tension (ADR-2103 D2 vs the workspace):** D2 says both header profiles are first-class in `sidestr-header`, "a crate `sidestr-core` depends on" (`../project/agentbox/docs/adr/ADR-2103-parent-chain-and-header-profile-are-configuration-behind-the-p21-gate.md:50`), but `sidestr-core` declares no such dependency (`../project/agentbox/crates/sidestr/sidestr-core/Cargo.toml:23-30`) and carries its own `HeaderFamily` instead, refusing a BLAKE2b parent as `Error::UnsupportedFamily` "until `sidestr-header` lands" (`../project/agentbox/crates/sidestr/sidestr-core/src/lib.rs:110-114`) — the convergence is 0.2's.

**Drift (BASELINE-container vs the workspace):** the governing document still describes the crates as a proposed permissive set with internal `sidestr-producer`, `sidestr-bridge` and `sidestr-mcp` members (`../project/agentbox/docs/BASELINE-container.md:340`, `../project/agentbox/docs/BASELINE-container.md:347`); four AGPL crates exist and none of those three does.

## AB-33.2 What each crate holds

```mermaid
flowchart TB
    subgraph h["sidestr-header - the part of consensus the parent's library does not own"]
        H1["stock 80-byte SHA-256d and Knots 164-byte v2 BLAKE2b,<br/>the two families SPEC 3.2 fixes<br/>sidestr-header/src/lib.rs:11-14"]
        H2["strict decode and encode, hash which IS the pow hash,<br/>compact Target and check_pow pinned to powLimit<br/>sidestr-header/src/lib.rs:31-37"]
        H3["the BIP-325 block-data preimage over the first 72 bytes,<br/>the same layout in both families, so ONE function<br/>sidestr-header/src/lib.rs:60-63"]
        H4["version bit 31 SELECTS the family, and a header with the<br/>wrong setting is refused at decode<br/>sidestr-header/src/lib.rs:41-42"]
    end
    subgraph c["sidestr-core - the chain"]
        C1["parents, document, block, marker, rules, state,<br/>blockfile, chain, address<br/>sidestr-core/src/lib.rs:31-39"]
        C2["level 1 and the stock family only<br/>sidestr-core/src/lib.rs:110-114"]
    end
    subgraph n["sidestr-nostr - the chain has no peer-to-peer network"]
        N1["own NIP-01 event, its id, BIP-340 verify, the sealed<br/>Signer port - sidestr-nostr/src/lib.rs:29"]
        N2["tip 33333 with the mirror trust rule, tx 23500, faucet<br/>23501, rules 33500, genesis 33501<br/>sidestr-nostr/src/lib.rs:32-34"]
        N3["33502 decoded as PegRecord, Pledge or Ambiguous,<br/>NEVER guessed - sidestr-nostr/src/lib.rs:35"]
        N4["the level-2 round codecs 23510 to 23514, and the<br/>estate's 38420 to 38425<br/>sidestr-nostr/src/lib.rs:36-37"]
    end
    subgraph w["sidestr-wallet - builds and signs, does not decide"]
        W1["coins, largest-first selection, key-path spends,<br/>peg-out burns, the parent-side peg-in shape, delivery<br/>sidestr-wallet/src/lib.rs:33-38"]
        W2["SpendSigner port and derive_spend_key<br/>sidestr-wallet/src/lib.rs:39"]
        W3["SpendPolicy consulted by EVERY builder before signing<br/>sidestr-wallet/src/lib.rs:40"]
    end
    h ~~~ c
    c --> n
    c --> w
```

**Invariant:** a key is reached only through named operations — `sidestr-nostr`'s `SignRequest` has no public constructor, so a signer is driven only through the `sign_*` functions and always sees the whole event (`../project/agentbox/crates/sidestr/sidestr-nostr/src/lib.rs:97-101`), and a `sidestr-wallet` builder computes the BIP-341 sighash and asks the port, never holding a secret (`../project/agentbox/crates/sidestr/sidestr-wallet/src/lib.rs:95-96`).

## AB-33.3 Where the ports depart from siding

```mermaid
flowchart TB
    subgraph core["sidestr-core - sidestr-core/src/lib.rs:123"]
        D1["FAILS CLOSED on script verification: the reference reports<br/>an unknown witness version as unverifiable and lets the<br/>block through. This verifies taproot key-path spends and<br/>REFUSES anything else - sidestr-core/src/lib.rs:128-134"]
        D2["overlay records commit on APPLY, not while validating, so<br/>a block that later fails leaves nothing behind<br/>sidestr-core/src/lib.rs:135-140"]
        D3["record_text checks the push length, which the reference<br/>has commented out - sidestr-core/src/lib.rs:141-143"]
        D4["a document naming assets, pool or evm, or a level-2<br/>federation, is REFUSED rather than misjudged<br/>sidestr-core/src/lib.rs:146-149"]
    end
    subgraph nostr["sidestr-nostr - sidestr-nostr/src/lib.rs:115"]
        D5["BOTH header families parse. Upstream parseTip accepts only<br/>length mod 328, so it returns null for every stock-family<br/>chain, the live dreamlab announcement included<br/>sidestr-nostr/src/lib.rs:117-121"]
        D6["refusals are typed Errors, not null<br/>sidestr-nostr/src/lib.rs:125-126"]
        D7["33502 decoded by STRUCTURE, where upstream reads every<br/>33502 as a pledge - sidestr-nostr/src/lib.rs:127-128"]
    end
    subgraph wallet["sidestr-wallet - sidestr-wallet/src/lib.rs:114"]
        D8["plain BIP 341 SIGHASH_DEFAULT, 64 bytes, where spend.mjs<br/>uses the kernel's UNIFIED sighash that sidestr-core<br/>refuses. BOTH engines accept the standard form, so<br/>acceptance is the bar, not byte equality<br/>sidestr-wallet/src/lib.rs:116-124"]
        D9["dust is refused rather than minting an unspendable coin<br/>sidestr-wallet/src/lib.rs:125-129"]
        D10["a fixed fee is checked against minFeeRate before signing<br/>sidestr-wallet/src/lib.rs:130-131"]
    end
    core --> EV["Each departure is deliberate and small, and the<br/>byte-for-byte genesis and the interop tests are what say<br/>they are harmless - sidestr-core/src/lib.rs:125-126"]
    nostr --> EV
    wallet --> EV
```

**Invariant:** every crate signs with zero BIP-340 auxiliary randomness, so a block, an event and a spend are each a pure function of their inputs and the key, and two producers with the same key and mempool make the same block (`../project/agentbox/crates/sidestr/sidestr-core/src/lib.rs:103-106`).

**Debt:** `sidestr-core` verifies only taproot key-path spends and refuses everything else, so a block valid to the reference is invalid here — a full script interpreter is deferred to 0.2 (`../project/agentbox/crates/sidestr/sidestr-core/src/lib.rs:133-134`).

## AB-33.4 The oracle ladder — what proves the port

```mermaid
sequenceDiagram
    autonumber
    participant F as sealed fixtures<br/>crates/sidestr/sidestr-core/tests/oracle.rs:1
    participant R as the Rust crates
    participant J as the reference engine<br/>crates/sidestr/sidestr-core/tests/xcheck.mjs:1
    participant L as public relays<br/>crates/sidestr/sidestr-nostr/tests/live.rs:1

    Note over F,R: 1 trial: a throwaway chain WITH its disposable key, so the<br/>genesis is rebuilt and compared byte for byte
    F->>R: chain.json, trial.key, blocks.dat
    R-->>F: block 0 reproduced, 317 bytes, height 0 (sidestr-core/tests/oracle.rs:38)
    Note over F,R: that equality is what proves the BIP-325 preimage, the<br/>coinbase layout and the zero-aux Schnorr path (sidestr-core/tests/oracle.rs:5)
    Note over F,R: 2 dreamlab: the estate's sealed genesis, key WITHHELD,<br/>replayed and checked against genesisHash (sidestr-core/tests/oracle.rs:7)
    R->>J: 3 interop, produce: seal one empty block with Siding<br/>crates/sidestr/sidestr-core/tests/xcheck.mjs:16
    J-->>R: height, hash and tx count of the JS-sealed block
    R->>J: replay: open the Rust-written directory and validate every block
    J-->>R: genesisHash, height, tip hash, utxo size (xcheck.mjs:17)
    Note over R,J: both directions use the trial fixture, so the genesis is the<br/>SAME sealed block on both sides (interop.rs:4-5)
    Note over R,J: without SIDESTR_SIDING, SCHEMA and BLAKETESTNODE the test<br/>reports itself SKIPPED and passes (interop.rs:40-45)
    R->>J: 4 wallet: a spend and a burn accepted by State.submit and<br/>by Siding.submit, same txid, fee and size<br/>crates/sidestr/sidestr-wallet/tests/oracle.rs:2-5
    J-->>R: and a tampered signature refused by both (sidestr-wallet/tests/oracle.rs:4)
    L->>R: 5 live: kind-33333 announcements fetched read-only on<br/>2026-09-22, one per event id, with relay and receipt time
    R-->>L: every one verifies and parses, at least ten of them<br/>crates/sidestr/sidestr-nostr/tests/live.rs:32
    Note over L,R: including the live dreamlab announcement, whose single<br/>80-byte header siding's parseTip rejects (live.rs:5-7)
```

**Invariant:** the interop test skips rather than fails when the reference checkouts are absent, so the gate stays green on a runner while still being a real cross-check where the checkouts exist (`../project/agentbox/crates/sidestr/sidestr-core/tests/interop.rs:2-3`).

**Open:** nothing in this repository pins the reference checkouts, so which upstream revision the interop and wallet oracles ran against is not recorded by the test — only the ported commit `2de40bda…` in each crate's rustdoc is (`../project/agentbox/crates/sidestr/sidestr-core/src/lib.rs:19`).

## AB-33.5 The CI gate on the workspace

```mermaid
flowchart TB
    TRIG["pull_request or push to main touching crates/sidestr/**<br/>.github/workflows/sidestr-crates.yml:5-12"]
    TRIG --> FMT["cargo fmt --all --check<br/>.github/workflows/sidestr-crates.yml:33"]
    FMT --> CLIP["cargo clippy --all-targets with -D warnings<br/>.github/workflows/sidestr-crates.yml:36"]
    CLIP --> TEST["cargo test: the ported siding suites plus the sealed<br/>fixtures - the trial genesis rebuilt byte for byte and<br/>the dreamlab genesis replayed<br/>.github/workflows/sidestr-crates.yml:38-43"]
    TEST --> DOC["cargo doc --no-deps with RUSTDOCFLAGS -D warnings,<br/>because AGPL crates ship FULL rustdoc (ADR-2106 D1)<br/>.github/workflows/sidestr-crates.yml:45-49"]
    DOC --> OK["green"]
    subgraph skip["What CI cannot reach"]
        S1["the JS interop half skips without the reference<br/>checkouts - .github/workflows/sidestr-crates.yml:41"]
        S2["the on-disk block file, which is host state<br/>see AB-32.3"]
        S1 ~~~ S2
    end
    TEST --> skip
```

**Invariant:** the docs step is a gate, not a courtesy — ADR-2106 D1 requires full inline rustdoc and a clean `cargo doc --no-deps` before publication, and the workflow enforces it with `-D warnings` (`../project/agentbox/.github/workflows/sidestr-crates.yml:47-49`).

## AB-33.6 How the licensing decision was reached and where it is written down

```mermaid
stateDiagram-v2
    [*] --> Planned
    Planned: ADR-2096 D2 and PRD-024 D3 and S3 planned four MIT OR Apache-2.0<br/>clean-room crates, written from SPEC prose ONLY
    note right of Planned
        ADR-2106-sidestr-crates-are-agpl-derivatives-of-siding-published-and-consumed.md:22-24
    end note
    Planned --> Confronted : the first seal, 2026-09-22
    Confronted: none of the four existed, and all three repositories the<br/>reference toolchain is made of are AGPL-3.0
    note right of Confronted
        A prose-only port forgoes the reference's tests, fixtures
        and the consensus detail that lives only in code, which is
        where the first seal found ADR-2103's errors.
        ADR-2106-sidestr-crates-are-agpl-derivatives-of-siding-published-and-consumed.md:26-30
    end note
    Confronted --> Decided : the owner directs, same day
    Decided: where a functional crate can be published and consumed, do so,<br/>with inline docs and proper attribution, decided case by case
    note right of Decided
        For sidestr, stick with AGPL-3.0 and consume.
        ADR-2106-sidestr-crates-are-agpl-derivatives-of-siding-published-and-consumed.md:30-32
    end note
    Decided --> Amended
    Amended: ADR-2096 D2's MIT OR Apache-2.0 and never-from-the-AGPL-JS are<br/>withdrawn. Clean-room survives only as written by us, attributed,<br/>tested against the reference
    note right of Amended
        ADR-2106-sidestr-crates-are-agpl-derivatives-of-siding-published-and-consumed.md:52-57
    end note
    Amended --> Recorded
    Recorded: docs/developer/licensing.md gains the crates/sidestr section,<br/>and the invariants workflow gate enforces the services/ rule<br/>it deliberately does NOT extend here
    note right of Recorded
        licensing.md:36-44, licensing.md:46-50
    end note
    Recorded --> [*]
```

**Invariant:** the crates sit under `crates/`, never `services/`, so ADR-2030's permissive default for service crates is untouched and `check-crate-licensing.sh` keeps enforcing it unchanged (`../project/agentbox/docs/developer/licensing.md:43-44`, `../project/agentbox/docs/developer/licensing.md:46-48`).

**Open:** ADR-2106's consequences flag that `solid-pod-rs` becomes AGPL-3.0 in effect the moment it links `sidestr-core` for `AnchorConfirmer`, and its manifest must say so before that edge is added — the edge does not exist yet (`../project/agentbox/docs/adr/ADR-2106-sidestr-crates-are-agpl-derivatives-of-siding-published-and-consumed.md:62-65`).
