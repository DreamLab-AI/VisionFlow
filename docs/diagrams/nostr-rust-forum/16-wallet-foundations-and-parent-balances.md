---
id: NF-16
title: Wallet foundations — custody, attribution and parent balances
area: nostr-rust-forum
governing:
  - ../nostr-rust-forum/docs/adr/ADR-2015-member-wallets-on-sidestr-dreamlab-and-dream-tips.md
  - ../nostr-rust-forum/docs/adr/ADR-2019-member-wallet-shows-blake2b-testnet4-coins-read-only.md
adrs: [ADR-2015, ADR-2019]
sources:
  - ../nostr-rust-forum/docs/adr/ADR-2015-member-wallets-on-sidestr-dreamlab-and-dream-tips.md
  - ../nostr-rust-forum/docs/adr/ADR-2019-member-wallet-shows-blake2b-testnet4-coins-read-only.md
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/wallet/mod.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/auth/session.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/auth/mod.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/wallet/chain.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/wallet.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/components/tip_button.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/wallet/parent.rs
verified_commit: {nostr-rust-forum: 6064622dbb1a7cc1ba2d072109f9e3e8a96157df}
---

## For developers

The [section audit](../../estate-review/2026-11-01-forum-wallet-foundations.md) distinguishes native parser and fee tests from unexecuted browser interleavings. NF-15 covers multichain pending state and delivery.

## For the business

A recognised forum identity makes receiving testnet tokens convenient. Clear receipts still require the correct recipient, an accurate fee and a balance labelled according to the evidence behind it. The parent card remains an optional read-only backend view.

## NF-16.1 The identity key serves different custody paths

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/wallet/mod.rs:588
    A["Wallet reuses the forum identity key<br/>when the auth session holds it"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/auth/session.rs:126
    B["Local-key login uses sessionStorage by default<br/>explicit remember=true selects localStorage"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/wallet/mod.rs:557
    C["Wallet-only unlock checks the member pubkey<br/>stores zeroising bytes in tab memory"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/wallet/mod.rs:599
    D["Without a tab key, use capable extension<br/>extension response is validated"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/auth/mod.rs:676
    E["Logout clears auth key and storage<br/>owner binding clears wallet unlock"]
    A --> B
    A --> C
    A --> D
    B --> E
    C --> E
    D --> E
```

**What it shows:** Auth-key persistence and a wallet-only unlock are different paths. The parent card uses only a public key.
**Why it is this way:** ADR-2015 deliberately reuses identity keys on testnet; it does not introduce an independent spend-key boundary.

## NF-16.2 Asset carriers and the fee preview have different checks

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/wallet/mod.rs:800
    A["Plain sends select coins carrying no asset<br/>asset sends preserve carrier tallies"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/wallet/chain.rs:646
    B["Starter pack builds asset carriers<br/>plus plain sats in one transaction"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/wallet/chain.rs:726
    C["Asset-view check runs before return<br/>local signing may already have occurred"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/wallet.rs:598
    D["Review shows a fixed about-250-sat hint<br/>not the built transaction fee"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/wallet/chain.rs:710
    E["Native synthetic fragmented-input pack:<br/>21 inputs, 1500-sat fee"]
    A --> B
    B --> C
    D -.-> E
```

**What it shows:** The actual builder protects carriers, but the confirmation screen uses a constant fee hint. The test is unsigned construction with synthetic coins, not a chain transfer.
**Why it is this way:** A useful confirmation should bind the recipient, amount, selected inputs and actual fee to the transaction later signed.

## NF-16.3 A tip memo identifies a post but does not prove its author was paid

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/components/tip_button.rs:130
    A["UI knows the post author<br/>and selects that recipient script"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/wallet/mod.rs:696
    B["Send asset with tip memo<br/>through the selected wallet"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/wallet/chain.rs:459
    C["Replay reads the memo event ID<br/>and first non-record output asset amount"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/wallet/chain.rs:468
    D["DEBT: replay tally has no post-author input<br/>recipient binding is not checked here"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/wallet/mod.rs:539
    E["Display adds this browser pending tips<br/>so optimistic totals can differ by viewer"]
    A --> B
    B --> C
    C --> D
    C --> E
```

**What it shows:** This is source evidence. A memo-bearing transfer is a candidate receipt; attributing it to an author requires the recipient comparison. No forged tip was broadcast.
**Why it is this way:** ADR-2015 relies on chain records, while D5 overstates uniform displayed totals and does not specify how to bind a transfer to the post author.

## NF-16.4 Parent reads are gated but optional work can delay all results

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/wallet/parent.rs:50
    A["Configured HTTPS-shaped base<br/>and valid public-key address enable reads"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/wallet/parent.rs:295
    B["GET address, then optional UTXOs<br/>then tip height and pair response"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/wallet/parent.rs:263
    C["No explicit fetch deadline or abort<br/>errors on optional routes become absent data"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/wallet.rs:261
    D["Effect launches request for current account<br/>completion writes one shared signal"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/wallet.rs:269
    E["DEBT: no current-account generation check<br/>a late earlier response can overwrite the result"]
    A --> D
    D --> B
    B --> C
    D --> E
```

**What it shows:** The source exposes neither a timeout for optional requests nor an account check at completion. Browser interleavings and stalled HTTP responses were not executed.
**Why it is this way:** ADR-2019 keeps the card read-only and backend-reported. Partial results and response identity still need explicit handling.

## NF-16.5 Maturity classification does not reconcile the complete coin set

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/wallet/parent.rs:149
    A["Require UTXO JSON and a tip height<br/>otherwise maturity is unknown"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/wallet/parent.rs:156
    B["Inspect confirmed coins for flag and height<br/>sum only immature coinbase values"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/wallet/parent.rs:168
    C["Spendable equals confirmed balance<br/>minus that immature subtotal"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/wallet/parent.rs:168
    D["DEBT: empty list plus balance 100<br/>reports 100 spendable"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/wallet.rs:719
    E["UI labels the result spendable now<br/>despite unvalidated backend state"]
    A --> B
    B --> C
    C --> D
    C --> E
```

**What it shows:** Native tests accept empty and inconsistent UTXO sets and an ordinary coin above the tip. The parser does reject missing flags on present confirmed coins.
**Why it is this way:** Maturity is only one restriction on spending. Reconcile snapshots and preserve unknown when the inputs disagree.

## NF-16.6 Backend shape checking leaves arithmetic and duplicate gaps

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/wallet/parent.rs:189
    A["Address must match exactly<br/>confirmed funded-minus-spent is checked"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/wallet/parent.rs:197
    B["DEBT: unconfirmed u64 values cast to i64<br/>before unchecked subtraction"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/wallet/parent.rs:240
    C["Pair parser reads branch subtotals<br/>and sums value from both entries"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/wallet/parent.rs:249
    D["DEBT: no outpoint deduplication<br/>and no checked pair total"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/wallet.rs:735
    E["UI says coins on both are spendable on either<br/>pair membership does not check maturity"]
    A --> B
    A --> C
    C --> D
    D --> E
```

**What it shows:** Native fixtures reproduce a negative wrapped cast, two caught debug overflow panics and a duplicated pair coin. Extreme arithmetic inputs test parser hardening, not realistic supply.
**Why it is this way:** A read-only backend contract should reject inconsistent or out-of-range figures before giving them spending language.
