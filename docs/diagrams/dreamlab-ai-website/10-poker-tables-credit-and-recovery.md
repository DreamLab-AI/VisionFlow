---
id: DW-10
title: Poker tables — deployment, credit, recovery and verification
area: dreamlab-ai-website
governing:
  - ../dreamlab-ai-website/docs/adr/ADR-2009-dream-poker-table-house-seat.md
  - ../nostr-rust-forum/docs/adr/ADR-2020-dream-poker-table-house-seat-and-dealt-member-hands.md
adrs: [ADR-2009, nostr-rust-forum:ADR-2020]
sources:
  - ../nostr-rust-forum/docs/adr/ADR-2020-dream-poker-table-house-seat-and-dealt-member-hands.md
  - ../dreamlab-ai-website/docs/adr/ADR-2009-dream-poker-table-house-seat.md
  - ../dreamlab-ai-website/forum-config/dreamlab.toml
  - ../dreamlab-ai-website/.github/workflows/deploy.yml
  - ../dreamlab-ai-website/scripts/seed/provision-poker-citizen.mjs
  - ../dreamlab-ai-website/scripts/seed/test-poker-citizen.mjs
  - ../project/agentbox/lib/poker-citizen.nix
  - ../project/agentbox/agentbox.toml
  - ../project/agentbox/config/poker/run-citizen.sh
  - ../nostr-rust-forum/crates/nostr-bbs-poker-citizen/src/table.rs
  - ../nostr-rust-forum/crates/nostr-bbs-poker-citizen/src/ledger.rs
  - ../nostr-rust-forum/crates/nostr-bbs-poker-citizen/src/run.rs
  - ../nostr-rust-forum/crates/nostr-bbs-poker/src/verify.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/poker/mod.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/poker/live.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/table.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/components/poker_schedule.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/relay.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/dm/mod.rs
verified_commit: {dreamlab-ai-website: b5eb1ecf96599ebe154ebf23bbe7c703d2e653a5, nostr-rust-forum: 9c36b6032640c89879ae52fe25da99493979a0b6, agentbox: 9dfc27b16e82e9786535a55e790710493dd51fdf}
---

## For developers

Follow the website overlay into the consumed kit's house state machine and browser settlement path. The associated estate review separates native counterexamples, mocked script results, source-only findings and process observations. The kit sources here are the website's consumed revision; 42 poker files are byte-identical at the agentbox service pin.

## For the business

Members have practice chips and two configured asset tables, with the house acting as a trusted dealer. A completed hand, an accepted transfer and a recoverable record are separate outcomes. The current receipts and recovery behaviour leave gaps between them.

## DW-10.1 Deployment selects separate house seats

```mermaid
flowchart TB
    %% Source: ../dreamlab-ai-website/forum-config/dreamlab.toml:320
    A["Website enables poker and configures<br/>DREAM and BLAKES7 house identities"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/poker/mod.rs:265
    B["Client offers a table for each configured<br/>chain with an asset and house key"]
    %% Source: ../project/agentbox/agentbox.toml:2034
    C["Agentbox enables two house services<br/>with separate keys, chains and daily caps"]
    %% Source: ../project/agentbox/config/poker/run-citizen.sh:23
    D["Runner chooses chain-specific state path<br/>and executes the baked binary"]
    %% Source: ../project/agentbox/lib/poker-citizen.nix:32
    E["DRIFT: service pin dd454474 differs from<br/>website client pin 9c36b603"]
    %% Source: ../dreamlab-ai-website/.github/workflows/deploy.yml:130
    F["Website consumes KIT_REF independently<br/>equal poker source does not prove all dependencies equal"]
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
```

**What it shows:** Configuration has expanded beyond the single DREAM seat described by ADR-2009. The source comparison found no changes in the 42 poker files across the two pins; it did not certify the complete dependency graph or running binary provenance.
**Why it is this way:** ADR-2009 places forum behaviour in the kit and service custody in agentbox. Its exact-commit lockstep wording is stronger than the current separately pinned deployment.

## DW-10.2 Admission grants unsecured play credit

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-poker-citizen/src/table.rs:289
    A["Member receives commitment then submits<br/>nonce, stake and requested seat"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-poker-citizen/src/table.rs:260
    B["Refuse unclaimed debt or a claim<br/>older than the configured grace period"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-poker-citizen/src/table.rs:399
    C["DEBT: a non-hex 64-byte transaction ID<br/>can renew the payment claim timestamp"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-poker-citizen/src/table.rs:521
    D["Check member balance and house balance<br/>less already settled unpaid obligations"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-poker-citizen/src/table.rs:535
    E["Check remaining daily payout cap<br/>then deal without reserving active buy-ins"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-poker-citizen/src/table.rs:225
    F["DEBT: two simultaneous buy-ins can use<br/>the same available house balance and cap"]
    A --> B
    C --> B
    B --> D
    D --> E
    E --> F
```

**What it shows:** The native fixtures renew a false payment claim and admit two 200-unit hands against 200 units of house balance and a 200-unit cap. They show exposure without reservation, not a demonstrated breach of the payout cap.
**Why it is this way:** ADR-2009 funds the house from treasury. The kit implements balance checks and post-hand debt rather than escrow; its grace mechanism trusts the member temporarily.

## DW-10.3 Recovery persists the book but not the active game

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-poker-citizen/src/table.rs:148
    A["Active hands, offers and sessions<br/>live in the Citizen maps"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-poker-citizen/src/table.rs:1029
    B["Tick expires challenges and stale sessions<br/>but retains sessions with a live hand"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-poker-citizen/src/table.rs:189
    C["DEBT: restart reconstructs Citizen from Ledger<br/>and starts with empty active-hand maps"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-poker-citizen/src/table.rs:846
    D["Finished hand emits Done and optional Pay<br/>before its Persist effect"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-poker-citizen/src/run.rs:460
    E["Runner performs effects in queue order<br/>and clears house debt on producer acceptance"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-poker-citizen/src/run.rs:276
    F["DEBT: failed file write is logged<br/>without stopping subsequent dealing"]
    A --> B
    A --> C
    A --> D
    D --> E
    E --> F
```

**What it shows:** An idle active hand survives a simulated month of ticks but disappears after reconstruction from the saved ledger. The source also exposes a send/pay-before-save crash window; no production crash or duplicate payment was induced.
**Why it is this way:** The service separates its pure state machine from network effects. Its persisted book covers settled obligations, while the in-progress game and held transaction inputs remain in memory.

## DW-10.4 Replay verification and payment fields have different coverage

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/poker/live.rs:840
    A["Done reply carries commitment, record,<br/>root, settlement amount and opponent"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/poker/live.rs:911
    B["Browser matches its remembered nonce<br/>and replays record with the engine"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-poker/src/verify.rs:151
    C["Verifier checks commitment, seed, actions,<br/>cards, result and deterministic house play"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/poker/live.rs:968
    D["DEBT: payment uses supplied settlement,<br/>root and destination without deriving all from replay"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/poker/live.rs:1036
    E["Verified status allows automatic payment<br/>otherwise user sees a payment action"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/poker/live.rs:554
    F["Wallet checks an existing hand payment<br/>before attempting another transfer"]
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
```

**What it shows:** The verifier is substantial, but its Claim has no settlement amount, destination or supplied root. The source-only finding is the missing binding between replay and payment instructions; this review did not execute a malicious browser payout.
**Why it is this way:** ADR-2009 delegates the game to the kit. Verifiable dealing and safe settlement require different checks even when both messages come from the configured house.

## DW-10.5 Scheduling and invitation delivery are separate

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/table.rs:146
    A["Schedule button requires client is_admin<br/>rather than an explicit moderator predicate"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/components/poker_schedule.rs:154
    B["Create calendar event with poker hashtag,<br/>chain tag and invited participant keys"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/relay.rs:640
    C["Register event ACK callback<br/>then queue publication"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/components/poker_schedule.rs:200
    D["Immediately start individual invitation DMs<br/>without awaiting event acceptance"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/components/poker_schedule.rs:202
    E["DEBT: count synchronous DM acceptance<br/>as invitations sent, then close modal"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/dm/mod.rs:472
    F["DM submission first updates local state<br/>before its asynchronous encrypted delivery"]
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
```

**What it shows:** Queued work can be labelled sent before the calendar event or invitation is accepted. These are source findings; no live invitation was sent.
**Why it is this way:** ADR-2009 reuses NIP-52 events and NIP-17 invitations. Reuse supplies the transports but does not make their acknowledgements atomic.

## DW-10.6 Operational probes can certify too much

```mermaid
flowchart TB
    %% Source: ../dreamlab-ai-website/scripts/seed/provision-poker-citizen.mjs:59
    A["Profile provisioner sends AUTH<br/>then delays kind-0 publication"]
    %% Source: ../dreamlab-ai-website/scripts/seed/provision-poker-citizen.mjs:62
    B["DEBT: first OK resolves the run<br/>without matching the kind-0 event ID"]
    %% Source: ../dreamlab-ai-website/scripts/seed/provision-poker-citizen.mjs:66
    C["DEBT: rejected publication still resolves<br/>instead of returning a failed exit status"]
    %% Source: ../dreamlab-ai-website/scripts/seed/test-poker-citizen.mjs:86
    D["Live-hand probe remembers offered commitment<br/>but Done check does not compare against it"]
    %% Source: ../dreamlab-ai-website/scripts/seed/test-poker-citizen.mjs:105
    E["Checks self-consistent secret, seed and nonce<br/>without engine replay or settlement confirmation"]
    %% Source: ../dreamlab-ai-website/scripts/seed/test-poker-citizen.mjs:84
    F["DEBT: next Offer prints played and settled<br/>even when the member still owes the transfer"]
    A --> B
    B --> C
    D --> E
    E --> F
```

**What it shows:** Mocked execution reproduces early profile success, a zero exit on rejection, and a hand-probe pass with a substituted commitment and no transfer. All keys and transport responses in these tests were synthetic.
**Why it is this way:** ADR-2009 lists operational probes as verification. A useful receipt must state precisely whether it proved authentication, publication, a completed game, verification or settlement.
