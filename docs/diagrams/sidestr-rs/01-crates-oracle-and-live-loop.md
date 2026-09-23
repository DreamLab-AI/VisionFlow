---
id: SR-01
title: sidestr-rs — five AGPL crates, what each owns, the oracle ladder, and the first live agent loop on sidestr:dreamlab
area: sidestr-rs
governing:
  - ../project/agentbox/docs/adr/ADR-2112-sidestr-crates-live-in-sidestr-rs.md
  - ../project/agentbox/docs/BASELINE-container.md
adrs: [ADR-2096, ADR-2101, ADR-2106, ADR-2112]
sources:
  - ../sidestr-rs/Cargo.toml
  - ../sidestr-rs/sidestr-header/Cargo.toml
  - ../sidestr-rs/sidestr-header/src/lib.rs
  - ../sidestr-rs/sidestr-header/tests/core_family.rs
  - ../sidestr-rs/sidestr-core/Cargo.toml
  - ../sidestr-rs/sidestr-core/src/lib.rs
  - ../sidestr-rs/sidestr-core/tests/oracle.rs
  - ../sidestr-rs/sidestr-core/tests/interop.rs
  - ../sidestr-rs/sidestr-core/tests/consensus_oracle.rs
  - ../sidestr-rs/sidestr-nostr/Cargo.toml
  - ../sidestr-rs/sidestr-nostr/src/lib.rs
  - ../sidestr-rs/sidestr-nostr/tests/live.rs
  - ../sidestr-rs/sidestr-wallet/Cargo.toml
  - ../sidestr-rs/sidestr-wallet/src/lib.rs
  - ../sidestr-rs/sidestr-wallet/tests/oracle.rs
  - ../sidestr-rs/sidestr-round/Cargo.toml
  - ../sidestr-rs/sidestr-round/src/lib.rs
  - ../sidestr-rs/sidestr-round/tests/interop_round.rs
  - ../sidestr-rs/sidestr-round/tests/interop_pegout.rs
  - ../project/agentbox/docs/adr/ADR-2112-sidestr-crates-live-in-sidestr-rs.md
  - ../project/agentbox/config/sidechain/dreamlab/chain.json
  - ../project/agentbox/config/sidechain/run-producer.sh
  - ../project/agentbox/config/sidechain/README.md
verified_commit: {sidestr-rs: 59a3f391836b6e824b906384f91d4812091bebd3, agentbox: ad45e7bf8371628c9e77c94e2eb9a344ee5f275f}
---

## For developers

sidestr-rs — Rust port of Melvin Carvalho's sidestr sidechains, AGPL-3.0-only: the economic engine for did:nostr agents. A did:nostr key is a sidechain wallet. The five crates were built in agentbox under `crates/sidestr/` (AB-33) and split out with their history on 2026-09-23; ADR-2112 moves source, CI, changelogs, audits and releases to this repository and leaves agentbox the chain *instance* in `config/sidechain/` (`ADR-2112-sidestr-crates-live-in-sidestr-rs.md:34-38`). This topic reads the workspace at sidestr-rs `59a3f39`, the `origin/main` head on the day of writing, and the chain instance at agentbox `ad45e7bf8`.

Each crate names the reference function it ports, carries upstream's licence, and is held to the reference engine by tests rather than by assertion. The table of what each crate owns is its crate-level rustdoc; the diagrams below cite those tables rather than restating them.

## For the business

This is the estate's settlement code as a stand-alone library: anyone running a `did:nostr` agent can check a sidestr chain, hold coins on it and pay with it, with the same key the agent already logs in with. It is copyleft (AGPL-3.0-only) because it is a faithful port of the original author's reference, and it is tested against that reference so the two cannot quietly disagree. Everything here is on test networks only; no real funds are held or moved.

## SR-01.1 The crate graph and the AGPL boundary

```mermaid
flowchart TB
    subgraph ws["sidestr-rs workspace, five members - ../sidestr-rs/Cargo.toml:3-22"]
        C["sidestr-core 0.2.2, AGPL-3.0-only<br/>sidestr-core/Cargo.toml:3, sidestr-core/Cargo.toml:8"]
        H["sidestr-header 0.2.1, AGPL-3.0-only<br/>no_std and RustCrypto only without feature core<br/>sidestr-header/Cargo.toml:3, sidestr-header/Cargo.toml:8"]
        N["sidestr-nostr 0.2.2, AGPL-3.0-only<br/>sidestr-nostr/Cargo.toml:3, sidestr-nostr/Cargo.toml:8"]
        W["sidestr-wallet 0.2.2, AGPL-3.0-only<br/>sidestr-wallet/Cargo.toml:3, sidestr-wallet/Cargo.toml:8"]
        RD["sidestr-round 0.1.1, AGPL-3.0-only<br/>sidestr-round/Cargo.toml:3, sidestr-round/Cargo.toml:8"]
    end
    H -->|"optional, feature core - sidestr-header/Cargo.toml:38"| C
    N -->|"path plus version 0.2 - sidestr-nostr/Cargo.toml:18"| C
    W -->|"path plus version 0.2 - sidestr-wallet/Cargo.toml:29"| C
    RD -->|"core, header and nostr - sidestr-round/Cargo.toml:43-45"| N
    C --> BTC["bitcoin 0.32, the consensus serialisation and BIP-340<br/>../sidestr-rs/Cargo.toml:33"]
    H --> RC["sha2 and blake2, default-features off<br/>sidestr-header/Cargo.toml:34-35"]
    subgraph rule["The boundary, restated by ADR-2112"]
        R1["consumers take the crates from crates.io only,<br/>never by path or git - agentbox included<br/>ADR-2112-sidestr-crates-live-in-sidestr-rs.md:39-41"]
        R2["a component that links one declares AGPL-3.0-only<br/>ADR-2112-sidestr-crates-live-in-sidestr-rs.md:40-41"]
        R3["licence, attribution, publish case by case and<br/>audit before publish are UNCHANGED<br/>ADR-2112-sidestr-crates-live-in-sidestr-rs.md:44-45"]
        R1 ~~~ R2 ~~~ R3
    end
    ws --> rule
```

**Invariant:** the header edge points one way only — `sidestr-header` depends on `sidestr-core` behind feature `core`, and `sidestr-core` never depends back (`../sidestr-rs/sidestr-header/Cargo.toml:37-38`, `../sidestr-rs/sidestr-header/src/lib.rs:32`), which is what lets the header crate stay `no_std` for a consumer that wants only the proof of work.

**Drift (ADR-2112 vs the workspace metadata):** ADR-2112 moves the crates to sidestr-rs (`../project/agentbox/docs/adr/ADR-2112-sidestr-crates-live-in-sidestr-rs.md:34-35`), but at `59a3f39` the workspace still declares `repository = "https://github.com/DreamLab-AI/agentbox"` (`../sidestr-rs/Cargo.toml:27`), and that is the repository every published version on crates.io links to.

## SR-01.2 What each crate owns

```mermaid
flowchart TB
    subgraph h["sidestr-header - the part of consensus the parent's library does not own"]
        H1["stock 80-byte SHA-256d beside btc and tbtc4, Knots 164-byte<br/>v2 BLAKE2b beside xbt and txbt4<br/>sidestr-header/src/lib.rs:11-14"]
        H2["hash IS the proof-of-work hash, compact Target, powLimit,<br/>BIP-325 block data, version bit 31 selects the family<br/>sidestr-header/src/lib.rs:39-51"]
    end
    subgraph c["sidestr-core - the chain"]
        C1["parents, document, block, sighash, parent, federation, marker,<br/>rules, state, blockfile, chain, address<br/>sidestr-core/src/lib.rs:29-42"]
        C2["peg-in marker and claim, peg-out burn, the mempool policy<br/>sidestr-core/src/lib.rs:50-60"]
        C3["tips and relays are NOT here: kinds 33333 and 23500 are<br/>sidestr-nostr's - sidestr-core/src/lib.rs:61-62"]
    end
    subgraph n["sidestr-nostr - the chain has no peer-to-peer network"]
        N1["event, kinds, tags, tip 33333, tx 23500 and faucet 23501,<br/>rules, the 33502 record, round codecs 23510 to 23514,<br/>the estate's 38420 to 38425, relay messages<br/>sidestr-nostr/src/lib.rs:28-39"]
        N2["verify first, decode second<br/>sidestr-nostr/src/lib.rs:105-108"]
    end
    subgraph w["sidestr-wallet - builds and signs, does not decide"]
        W1["coins, selection, spend, burn, the parent-side peg-in,<br/>delivery as POST /tx or kind 23500, key, policy<br/>sidestr-wallet/src/lib.rs:31-40"]
        W2["keys stay behind SpendSigner, the identity key never spends<br/>sidestr-wallet/src/lib.rs:95-100"]
    end
    subgraph r["sidestr-round - level 2, the protocol between core and nostr"]
        R1["Round, PegoutRound, VoteJournal, BlockSigner, ChainView,<br/>relay and node behind features<br/>sidestr-round/src/lib.rs:39-47"]
        R2["the cosign binary, one federation signer<br/>sidestr-round/Cargo.toml:34-40"]
    end
    h ~~~ c
    c --> n
    c --> w
    n --> r
```

**Invariant:** a key is reached only through named operations — `sidestr-nostr`'s `SignRequest` has no public constructor, so a signer is driven only through the `sign_*` functions and sees the whole event (`../sidestr-rs/sidestr-nostr/src/lib.rs:98-102`); a wallet builder computes the sighash and asks the port, never holding a secret (`../sidestr-rs/sidestr-wallet/src/lib.rs:95-97`).

**Open:** the sixth crate, `sidestr-agent` (the `did:nostr` key as a wallet, generalised from the tool that ran SR-01.5), is being added now and is not on `origin/main` at `59a3f39` (`../sidestr-rs/Cargo.toml:3-22` lists five members); this topic records nothing about it until it lands.

## SR-01.3 The oracle ladder against upstream

```mermaid
sequenceDiagram
    autonumber
    participant F as sealed fixtures<br/>sidestr-core/tests/oracle.rs:1
    participant R as the Rust crates
    participant J as the reference engine, siding<br/>sidestr-core/tests/interop.rs:1
    participant K as Bitcoin Core interpreter<br/>sidestr-core/tests/consensus_oracle.rs:1
    participant M as live mirrors and relays<br/>sidestr-header/tests/core_family.rs:1

    Note over F,R: 1 trial: a throwaway chain WITH its key, genesis rebuilt byte<br/>for byte, dreamlab: the estate genesis replayed, key withheld<br/>sidestr-core/tests/oracle.rs:4-8
    R->>J: 2 interop: each engine accepts the other's blocks<br/>sidestr-core/tests/interop.rs:1-5
    R->>J: 3 wallet: a spend and a burn accepted by State.submit AND by<br/>Siding.submit, same txid, fee and size<br/>sidestr-wallet/tests/oracle.rs:1-7
    J-->>R: acceptance is the bar, not byte equality, because the<br/>reference signs with the kernel's unified sighash (sidestr-wallet/tests/oracle.rs:9-12)
    R->>K: 4 the multi_a verifier judged against Core 26.0 on every<br/>subset, parity and malformed case (sidestr-core/tests/consensus_oracle.rs:1-9)
    M->>R: 5 BLAKE2b family: the live sidestr:txbt4-siding mirror,<br/>229 blocks, replayed with every rule on (sidestr-header/tests/core_family.rs:6-16)
    M->>R: 6 kind-33333 announcements fetched read-only from public relays,<br/>the dreamlab one included - sidestr-nostr/tests/live.rs:1-8
    R->>J: 7 level 2: three signers mixed Rust and JS, rotation, one down<br/>tolerated, two halts - sidestr-round/tests/interop_round.rs:1-9
    R->>J: 8 peg-out round mixed, every finalised parent tx verified<br/>under BIP 342 - sidestr-round/tests/interop_pegout.rs:1-8
    Note over R,J: the JS rungs skip and pass without SIDESTR_SIDING, SCHEMA<br/>and BLAKETESTNODE (sidestr-core/tests/interop.rs:2-3)
```

**Invariant:** every port names the upstream revision it was taken from — siding at `2de40bda…` for the core (`../sidestr-rs/sidestr-core/src/lib.rs:17-22`) and the schema kernel at `b8cbf633…` for the header (`../sidestr-rs/sidestr-header/src/lib.rs:126-135`) — so a divergence can be read side by side.

**Debt:** at `59a3f39` the repository carries no CI workflow at all; ADR-2112 deleted agentbox's `sidestr-crates.yml` with the move and records that sidestr-rs must carry that CI, which its remote did not have at the split (`../project/agentbox/docs/adr/ADR-2112-sidestr-crates-live-in-sidestr-rs.md:50-52`), so every rung above runs only where someone runs it.

## SR-01.4 Where the ports deliberately depart from siding

```mermaid
flowchart TB
    subgraph core["sidestr-core"]
        D1["script verification FAILS CLOSED: taproot key path verified,<br/>anything else refused, where the reference lets an unknown<br/>witness version through - sidestr-core/src/lib.rs:179-185"]
        D2["the genesis is judged by every height-0 rule before its<br/>hash is held to the pin - sidestr-core/src/lib.rs:150-159"]
        D3["overlay records commit on APPLY, not while validating<br/>sidestr-core/src/lib.rs:230-235"]
        D4["a document naming assets, pool or evm is REFUSED<br/>sidestr-core/src/lib.rs:243-246"]
    end
    subgraph nostr["sidestr-nostr"]
        D5["both header families parse a tip, a length that fits both<br/>is refused, never guessed - sidestr-nostr/src/lib.rs:118-128"]
    end
    subgraph wallet["sidestr-wallet"]
        D6["plain BIP 341 SIGHASH_DEFAULT, accepted by both engines<br/>sidestr-wallet/src/lib.rs:116-124"]
        D7["dust refused, a fixed fee checked against minFeeRate<br/>sidestr-wallet/src/lib.rs:125-131"]
    end
    core --> EV["deliberate and small, and the byte-for-byte genesis and<br/>the interop tests are what say they are harmless<br/>sidestr-core/src/lib.rs:147-148"]
    nostr --> EV
    wallet --> EV
```

**Invariant:** zero BIP-340 auxiliary randomness everywhere, so a block, an event and a spend are each a pure function of their inputs and the key (`../sidestr-rs/sidestr-core/src/lib.rs:106-109`, `../sidestr-rs/sidestr-wallet/src/lib.rs:132-133`).

## SR-01.5 The first agent loop on sidestr:dreamlab, 2026-09-23

```mermaid
sequenceDiagram
    autonumber
    participant A as agent Alice, a did:nostr key<br/>sidestr-wallet/src/lib.rs:39
    participant T as Bitcoin testnet4, the parent<br/>config/sidechain/dreamlab/chain.json:4
    participant P as producer, upstream JS siding<br/>config/sidechain/run-producer.sh:33
    participant B as agent Bob, a did:nostr key<br/>sidestr-wallet/src/lib.rs:39
    participant MI as mirror, GitHub Pages<br/>config/sidechain/README.md:62

    A->>T: peg-in, a peg output plus OP_RETURN pegin marker (sidestr-core/src/lib.rs:50-53)
    Note over A,T: EXTERNAL evidence, testnet4 block 153653, tx 2c4c5941..e4a57d,<br/>output 0 carries 50,000 test sats
    P->>T: scans the parent from the funding height (run-producer.sh:38)
    P->>MI: at pegConfirmations 6 (chain.json:10) the coinbase claims it with<br/>a claim marker - block 131 carries claim 2c4c5941..e4a57d:0
    A->>B: three trades, each a signed tx carried as a kind-23500 event<br/>sidestr-nostr/src/lib.rs:34
    B->>P: the producer follows kind 23500 and includes what validates<br/>sidestr-wallet/src/lib.rs:12-15
    Note over P,MI: EXTERNAL evidence, non-empty blocks 240, 243 and 245 on the<br/>mirror, 12h05 to 12h07 UTC
    A->>P: peg-out, a burn OP_RETURN pegout of at least pegoutMin<br/>sidestr-core/src/lib.rs:54-57
    Note over P,MI: EXTERNAL evidence, block 248 burns to a P2TR parent script
    P->>T: the peg holders pay the burn on the parent (sidestr-core/src/lib.rs:56-57)
    Note over T: EXTERNAL evidence, testnet4 block 153672, tx 0ceb01d3..2c8f98<br/>pays 20,000 test sats to the burned-to address
```

**Invariant:** the chain is test-only by construction — the sealed document's own comment says level 1, one signer, and coins carrying no value, with mainnet parents behind the P21 gate (`../project/agentbox/config/sidechain/dreamlab/chain.json:5`).

**Tension (the component vs the running chain):** the loop ran on a chain the upstream JavaScript engine produces (`../project/agentbox/config/sidechain/run-producer.sh:33-34`; ADR-2112 says nothing in agentbox links the crates, `../project/agentbox/docs/adr/ADR-2112-sidestr-crates-live-in-sidestr-rs.md:28-29`), so what sidestr-rs contributed was the agents' side; a Rust producer is still unbuilt (`../project/agentbox/config/sidechain/README.md:66-69`).

## SR-01.6 Level 1 runs, level 2 is usable, BFT is not built

```mermaid
stateDiagram-v2
    [*] --> Level1
    Level1: LEVEL 1, LIVE - one signer, the BIP-325 challenge,<br/>sidestr:dreamlab beside tbtc4 since 2026-09-22
    note right of Level1
        chain.json:5 level 1, one signer, depth 0
        chain.json:18 the sealed genesisHash
    end note
    Level1 --> Level2 : a federation document, signers and threshold
    Level2: LEVEL 2, USABLE - sidestr-round co-signs k of n with<br/>reference signers on the wire, cosign is runnable
    note right of Level2
        sidestr-round/src/lib.rs:1-5
        interop proven both ways - sidestr-round/src/lib.rs:76-83
        journal stops a restart double-signing - sidestr-round/src/lib.rs:91
    end note
    Level2 --> Limits
    Limits: AVAILABILITY tolerance only - n minus k signers may be<br/>DOWN, none may be WRONG
    note right of Limits
        sidestr-round/src/lib.rs:119-125
    end note
    Limits --> BFT : later
    BFT: NOT BUILT - views, decision certificates, the ADR-2101<br/>consensus crate, which replaces Round, not the codecs
    note right of BFT
        sidestr-round/src/lib.rs:125-129
        sidestr-core/src/lib.rs:135-136
    end note
    BFT --> [*]
```

**Open:** no estate chain runs at level 2 yet — `sidestr:dreamlab` is sealed as level 1 with one signer (`../project/agentbox/config/sidechain/dreamlab/chain.json:5`), so `sidestr-round`'s live evidence is the mixed-engine tests, not a federation the estate operates.
