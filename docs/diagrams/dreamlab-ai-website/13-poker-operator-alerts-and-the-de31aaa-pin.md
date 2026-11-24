---
id: DW-13
title: Poker operator alerts and the de31aaa pin — house seat and website pins, alert DMs
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
  - ../project/agentbox/flake.nix
  - ../project/agentbox/config/poker/run-citizen.sh
  - ../nostr-rust-forum/crates/nostr-bbs-poker-citizen/src/alert.rs
  - ../nostr-rust-forum/crates/nostr-bbs-poker-citizen/src/run.rs
  - ../nostr-rust-forum/crates/nostr-bbs-poker-citizen/src/table.rs
  - ../nostr-rust-forum/crates/nostr-bbs-poker-citizen/src/ledger.rs
  - ../nostr-rust-forum/crates/nostr-bbs-poker-citizen/src/state.rs
  - ../nostr-rust-forum/crates/nostr-bbs-poker/src/protocol.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/poker/mod.rs
verified_commit: {dreamlab-ai-website: 5dc4c644517ff36219b7fe6cc1a70f20dc6a8d83, nostr-rust-forum: de31aaa6e6c6b672cc310a2659cf33fd4887602c, agentbox: ca07b59603137ca76159634cf664a622c34ab9ec}
---

## For developers

Successor to DW-12.1 and DW-12.3, which stay frozen at the revisions they were verified at. It records kit `c8b8c4e` (the house seat turns `Effect::Alert` into a rate-limited, gift-wrapped DM to an operator) and the website repin to kit `de31aaa` (website PR #101, `5dc4c64`). DW-12.2 and DW-12.4 are unchanged in substance: no poker crate or client poker file differs between `8a0ca41` and `de31aaa` except the house seat's alert work. The agentbox pin is read at `ca07b59`; no native test was run.

## For the business

When the house cannot settle a payout by itself (a debt halted after repeated rebuilds, or the block producer refusing the same payout again and again) it can now message an operator directly on the forum, at most once per debt and cause every six hours. On this deployment that message is not switched on: the operator key is commented out, so the alert still lands only in the service log. The house seat also now runs one kit commit behind the website; the poker code is identical between the two, so play is unaffected.

## DW-13.1 The website and the house seat name different kit commits again

```mermaid
flowchart TB
    %% Source: ../dreamlab-ai-website/forum-config/dreamlab.toml:320
    A["Website names a house identity<br/>for each sidestr chain"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/poker/mod.rs:266
    B["Client offers a table for each chain<br/>with an asset and a house key"]
    %% Source: ../dreamlab-ai-website/.github/workflows/deploy.yml:131
    C["Website KIT_REF is de31aaa at deploy.yml:131,<br/>workers-deploy.yml:45 and rust-ci.yml:20"]
    %% Source: ../dreamlab-ai-website/docs/architecture/kit-compatibility-record.md:30
    D["CANONICAL_KIT_SHA records the same commit"]
    %% Source: ../project/agentbox/agentbox.toml:2091
    E["Agentbox enables the DREAM seat and a BLAKES7 seat<br/>(agentbox.toml:2105)"]
    %% Source: ../project/agentbox/config/poker/run-citizen.sh:63
    F["Runner executes the baked binary with key, relay,<br/>producer and state (run-citizen.sh:64-67)"]
    %% Source: ../project/agentbox/lib/poker-citizen.nix:33
    G["Seat is built from forum commit c8b8c4e"]
    %% Source: ../project/agentbox/lib/poker-citizen.nix:32
    H["DRIFT: the nix comment says the rev must equal the website KIT_REF,<br/>and it does not - c8b8c4e against de31aaa"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-poker/src/protocol.rs:51
    I["Protocol VERSION 1 on both sides; no poker crate or client poker file<br/>differs between c8b8c4e and de31aaa"]
    %% Source: ../project/agentbox/lib/poker-citizen.nix:15
    J["DEBT: agreement is a rule in a comment,<br/>no script in either repository compares the two pins"]
    A --> B
    B --> C
    C --> D
    E --> F
    F --> G
    G --> H
    C --> H
    H --> I
    H --> J
```

**What it shows:** The website moved to kit `de31aaa` while the agentbox house seat stayed at `c8b8c4e`, so the DRIFT that DW-12.1 recorded as gone has reopened. The two commits carry the same poker code (checked with `git diff --stat c8b8c4e de31aaa` over the poker crates and the client's poker module, empty), so the exposure is the unenforced rule, not a protocol mismatch today.
**Why it is this way:** ADR-2009 puts game behaviour in the kit and service custody in agentbox; the website's pin-parity gate covers its own workflow sites and record (DW-05.11) and does not read agentbox. **Debt:** unchanged from DW-12.1, nothing but a comment ties the two pins.

## DW-13.2 Hands and payouts are saved before anything leaves, and alerts can reach an operator by DM

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-poker-citizen/src/run.rs:629
    A["on_event hands the message to the table"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-poker-citizen/src/run.rs:668
    B["checkpoint saves before run_effects sends (run.rs:669)"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-poker-citizen/src/state.rs:257
    C["save writes a private 0600 temporary, syncs it,<br/>renames it and syncs the directory (state.rs:271-284)"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-poker-citizen/src/run.rs:337
    D["a failed save stops the house"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-poker-citizen/src/run.rs:533
    E["payout: signed transfer is kept and saved (run.rs:536)<br/>before the POST (run.rs:539)"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-poker-citizen/src/ledger.rs:424
    F["rebuilds are capped at one for rejection and three in all,<br/>then the debt halts (ledger.rs:85, :87)"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-poker-citizen/src/table.rs:1436
    G["replace_payout raises Alert with cause Halted<br/>house_pay_refused raises Refused every 5th refusal (table.rs:1472, ledger.rs:89)"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-poker-citizen/src/run.rs:218
    H["startup: one Summary alert listing halted debts<br/>queued for the first connection (run.rs:279)"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-poker-citizen/src/run.rs:618
    I["run_effects always prints ALERT to stderr"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-poker-citizen/src/alert.rs:138
    J["Alerter.route: none without a key, none if this debt and cause<br/>went out in the last 6 h (alert.rs:20, :143), none past 20 per window (alert.rs:23)"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-poker-citizen/src/run.rs:621
    K["a due Dm goes to the front of the effect queue"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-poker-citizen/src/run.rs:428
    L["send_dm gift-wraps from the house key and publishes on the forum relay (run.rs:435)"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-poker-citizen/src/alert.rs:146
    M["DEBT: route marks the alert sent before the DM is wrapped or published,<br/>and a failure is only logged (run.rs:436), so it is not retried inside the window"]
    %% Source: ../project/agentbox/agentbox.toml:2098
    N["OPEN: alert_pubkey is commented out for the DREAM seat, so the flake<br/>projects no POKER_CITIZEN_ALERT_PUBKEY (flake.nix:473) and alerts stay on stderr (run.rs:274)"]
    A --> B
    B --> C
    C --> D
    B --> E
    E --> F
    F --> G
    G --> I
    H --> I
    I --> J
    J --> K
    K --> L
    L --> M
    J --> N
```

**What it shows:** Persistence is as DW-12.3 drew it; what changed is the end of the chain. An alert carries its debt and cause (alert.rs:50), the runner still logs it, and with an operator key set it also sends one gift-wrapped NIP-17 DM per debt and cause per six-hour window. The DM recipient must be admitted by the relay's gift-wrap gate (run.rs:93), and the dedupe map is in memory only (alert.rs:100), so a restart can repeat an alert inside the window.
**Why it is this way:** ADR-2009's house is a trusted dealer whose halted debts need a human, so c8b8c4e closed DW-12.3's OPEN by giving alerts a destination beyond stderr. **Open:** this deployment has not set the key, so whether an operator sees a halted debt still depends on who reads the service log.
