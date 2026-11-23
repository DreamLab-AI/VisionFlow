---
id: DW-12
title: Poker settlement repairs — pins, buy-in reservation, recovery and replay-bound payment
area: dreamlab-ai-website
governing:
  - ../dreamlab-ai-website/docs/adr/ADR-2009-dream-poker-table-house-seat.md
  - ../nostr-rust-forum/docs/adr/ADR-2020-dream-poker-table-house-seat-and-dealt-member-hands.md
adrs: [ADR-2009, nostr-rust-forum:ADR-2020]
sources:
  - ../dreamlab-ai-website/forum-config/dreamlab.toml
  - ../dreamlab-ai-website/.github/workflows/deploy.yml
  - ../dreamlab-ai-website/.github/workflows/workers-deploy.yml
  - ../dreamlab-ai-website/.github/workflows/rust-ci.yml
  - ../dreamlab-ai-website/docs/architecture/kit-compatibility-record.md
  - ../project/agentbox/lib/poker-citizen.nix
  - ../project/agentbox/agentbox.toml
  - ../project/agentbox/config/poker/run-citizen.sh
  - ../nostr-rust-forum/crates/nostr-bbs-poker-citizen/src/table.rs
  - ../nostr-rust-forum/crates/nostr-bbs-poker-citizen/src/ledger.rs
  - ../nostr-rust-forum/crates/nostr-bbs-poker-citizen/src/state.rs
  - ../nostr-rust-forum/crates/nostr-bbs-poker-citizen/src/run.rs
  - ../nostr-rust-forum/crates/nostr-bbs-poker/src/bind.rs
  - ../nostr-rust-forum/crates/nostr-bbs-poker/src/protocol.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/poker/mod.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/poker/live.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/poker/choices.rs
verified_commit: {dreamlab-ai-website: 389a644c23e9f5ed0b571242f3e8e5418aae0a7c, nostr-rust-forum: 8a0ca418b9f129b219431880d9fa5a81603428fc, agentbox: 76ad253ae575d61eb43a2bdefeb163f8247cabb8}
---

## For developers

This is the successor to DW-10.1 to DW-10.4, which stay frozen at the revisions they were verified at because their citations span three repositories. It records the kit commit `8a0ca41` (buy-in reservation, persisted hands, idempotent payouts, replay-bound client payment), the agentbox house-seat pin that follows it and the website `KIT_REF` that matches. DW-10.5 and DW-10.6 are not repaired and remain valid in DW-10. The panels below come from reading source; the native tests they name were not run for this revision.

## For the business

The house no longer promises the same chips twice: each hand held against the house is set aside from its bankroll and its daily limit when the hand is accepted. A restart resumes a hand only if it proves itself, and a member's browser pays only what the verified hand and the member's own table choice support. The payment-claim timestamp can still be renewed by a malformed transaction id.

## DW-12.1 The house seat and the website build the same kit commit

```mermaid
flowchart TB
    %% Source: ../dreamlab-ai-website/forum-config/dreamlab.toml:320
    A["Website names a house identity<br/>for each sidestr chain"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/poker/mod.rs:266
    B["Client offers a table for each chain<br/>with an asset and a house key"]
    %% Source: ../project/agentbox/agentbox.toml:2091
    C["Agentbox enables the DREAM seat and a BLAKES7 seat<br/>(agentbox.toml:2104) with their own keys and caps"]
    %% Source: ../project/agentbox/config/poker/run-citizen.sh:24
    D["Runner picks the chain and per-chain state files<br/>then executes the baked binary (run-citizen.sh:61)"]
    %% Source: ../project/agentbox/lib/poker-citizen.nix:31
    E["Seat is built from forum commit 8a0ca41<br/>nix comment: must equal the website KIT_REF (nix:30)"]
    %% Source: ../dreamlab-ai-website/.github/workflows/deploy.yml:131
    F["Website KIT_REF is 8a0ca41 at deploy.yml:131,<br/>workers-deploy.yml:45 and rust-ci.yml:20"]
    %% Source: ../dreamlab-ai-website/docs/architecture/kit-compatibility-record.md:30
    G["CANONICAL_KIT_SHA records the same commit"]
    %% Source: ../project/agentbox/lib/poker-citizen.nix:15
    H["DEBT: agreement is a rule in a comment,<br/>no script in either repository compares the two pins"]
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    E --> H
```

**What it shows:** The house service and the website client now name the same kit commit, so both speak the same poker protocol. The earlier DRIFT between the service pin and the client pin is gone; the unchanged exposure is that nothing but a comment enforces it. Equal pins do not certify the dependency graph either.
**Why it is this way:** ADR-2009 puts game behaviour in the kit and service custody in agentbox. The website's pin-parity gate covers the workflow sites and the compatibility record (DW-05.10) and does not read agentbox.

## DW-12.2 Acceptance reserves the buy-in against the bankroll and the daily cap

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-poker-citizen/src/table.rs:869
    A["sit refuses an overdue debtor or a short member"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-poker-citizen/src/table.rs:564
    B["committed = house hands in play<br/>(table.rs:555) plus payouts still owing (ledger.rs:254)"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-poker-citizen/src/table.rs:879
    C["House balance must cover committed plus the new buy-in,<br/>and so must the remaining daily cap (table.rs:886)"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-poker-citizen/src/table.rs:949
    D["deal inserts the hand with its buy-in,<br/>which is the reservation"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-poker-citizen/src/table.rs:1221
    E["settle removes the hand and so releases it"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-poker-citizen/src/table.rs:1561
    F["tick folds a member hand idle past 30 minutes (table.rs:36)<br/>and voids a stuck house turn (table.rs:1544)"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-poker-citizen/src/table.rs:747
    G["DEBT: a 64-character non-hex transaction id passes the length check"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-poker-citizen/src/ledger.rs:291
    H["claim overwrites the claim time on every report,<br/>so the grace period can be renewed (table.rs:605)"]
    A --> B
    B --> C
    C --> D
    D --> E
    D --> F
    G --> H
    H --> A
```

**What it shows:** The reservation is derived from the hands the house holds, so two buy-ins answered from one chain read cannot claim the same coins, and an abandoned hand releases its share. Hands between two members reserve nothing because the house moves no funds (table.rs:551-554). The native tests named `concurrent_buy_ins_cannot_overdraw_the_house_bankroll` (table.rs:1880) and `concurrent_buy_ins_cannot_exceed_the_daily_cap` (table.rs:1903) pin both limits.
**Why it is this way:** ADR-2009 funds the house from treasury, so its worst loss in a hand is the buy-in. **Debt:** the claim-timestamp weakness of DW-10.2 is unchanged at this revision by source reading; no probe was run.

## DW-12.3 Hands, the book and payouts are saved before anything leaves

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-poker-citizen/src/run.rs:606
    A["on_event hands the message to the table"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-poker-citizen/src/run.rs:614
    B["checkpoint saves before run_effects sends (run.rs:615)"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-poker-citizen/src/state.rs:257
    C["save writes a private 0600 temporary, syncs it,<br/>renames it and syncs the directory (state.rs:271-284)"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-poker-citizen/src/run.rs:304
    D["a failed save stops the house"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-poker-citizen/src/table.rs:402
    E["restart: restore resumes a saved hand only if its secret,<br/>seed, seats and replayed actions match (table.rs:403-463)"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-poker-citizen/src/table.rs:309
    F["otherwise void reveals the deal to the member<br/>and moves no funds (table.rs:332)"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-poker-citizen/src/run.rs:486
    G["payout: signed transfer is kept and saved (run.rs:489)<br/>before the POST (run.rs:490)"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-poker-citizen/src/ledger.rs:123
    H["a retry resends the same bytes; a new transfer needs chain proof<br/>of coins spent elsewhere or an invalid inclusion"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-poker-citizen/src/ledger.rs:424
    I["rebuilds are capped at one for rejection and three in all,<br/>then the debt halts (ledger.rs:85, :87)"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-poker-citizen/src/run.rs:570
    J["OPEN: the alert for a halted or repeatedly refused payout<br/>is printed to stderr, the only handler found in run.rs"]
    A --> B
    B --> C
    C --> D
    C --> E
    E --> F
    E --> G
    G --> H
    H --> I
    I --> J
```

**What it shows:** Live hands, the ledger and the processed message ids are one saved document, written whole, so a restart sees a hand either in play or settled. An accepted payout stays a debt until the chain shows the payment (ledger.rs:360 keeps the debt, ledger.rs:308 clears it). The tests `a_restart_mid_hand_resumes_it_and_no_funds_are_lost_or_counted_twice` (table.rs:2028) and `a_rejected_payout_is_replaced_once_through_the_cap_and_then_the_debt_halts` (table.rs:2404) cover the paths.
**Why it is this way:** ADR-2009's house is a trusted dealer, so a crash must never lose a debt or pay one twice. **Open:** whether an operator sees a halted debt depends on who reads the process's standard error; no other alert path was found.

## DW-12.4 The member's payment is bound to the replay and the table they chose

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/poker/live.rs:569
    A["sit stores what the member chose for the hand:<br/>opponent, blinds and buy-in (live.rs:245)"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/poker/choices.rs:21
    B["choices are kept in localStorage per commitment<br/>for a day and reloaded on open (live.rs:366)"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/poker/live.rs:1034
    C["Done reply arrives with settlement, root and opponent"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/poker/live.rs:1137
    D["browser replays the record against its own nonce<br/>(live.rs:1124)"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-poker/src/bind.rs:128
    E["bind_settlement checks opponent, root, blinds and buy-in<br/>against the choice (bind.rs:151-169)"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-poker/src/bind.rs:170
    F["amount and payee come from the replay, and a different<br/>reported settlement is refused (bind.rs:172-184)"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/poker/live.rs:1188
    G["no stored choice means no payment"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/poker/live.rs:259
    H["pay_from refuses a house script that is not its key's (live.rs:278)<br/>and auto-pay needs a verified hand (live.rs:1261)"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/poker/live.rs:727
    I["wallet checks for an existing payment for the root<br/>before it sends another"]
    A --> B
    A --> E
    C --> D
    D --> E
    E --> F
    D --> G
    F --> H
    H --> I
```

**What it shows:** The Done message is no longer a payment instruction. Opponent, kind of hand, blinds, buy-in, root, amount and payee are each held to the member's own choice or to the verified replay, and a member who lost the stored choice is not asked to pay. The tests in bind.rs from line 290 (tampered amount, destination, hand kind, stakes) and `a_hand_that_did_not_bind_is_never_paid` (live.rs:1365) cover the refusals.
**Why it is this way:** ADR-2009 delegates the game to the kit, and a wallet transfer is irreversible, so the amount and the payee are derived from what was verified. The protocol keeps `VERSION` at 1 (protocol.rs:51), adds an optional `log_len` to Act (protocol.rs:126) and a Voided reply (protocol.rs:282).
