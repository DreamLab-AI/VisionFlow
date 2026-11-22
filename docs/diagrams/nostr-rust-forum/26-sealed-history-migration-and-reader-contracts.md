---
id: NF-26
title: Sealed history migration and reader contracts
area: nostr-rust-forum
governing:
  - ../nostr-rust-forum/docs/adr/ADR-2017-sealed-original-history-migration.md
adrs: [ADR-2017]
sources:
  - ../nostr-rust-forum/docs/adr/ADR-2017-sealed-original-history-migration.md
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/sealed.rs
  - ../nostr-rust-forum/crates/nostr-bbs-zone-migrate/src/relay.rs
  - ../nostr-rust-forum/crates/nostr-bbs-zone-migrate/src/engine.rs
  - ../nostr-rust-forum/crates/nostr-bbs-zone-migrate/src/plan.rs
  - ../nostr-rust-forum/crates/nostr-bbs-zone-migrate/src/cli.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/admin_events.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/audit.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/mod.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/note_view.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/lib.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/whitelist.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/auth.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/stores/tombstones.rs
  - ../project/agentbox/management-api/lib/zone-keys.js
  - ../project/agentbox/scripts/dream-forum-suggestions.mjs
verified_commit: {nostr-rust-forum: af4f0d56f735ca29dddf9be2549d5c0e9f987891, agentbox: 720e0d89656a4360998b3a716d0f3ee8cac66f93}
---

## For developers

[The sealed-history audit](../../estate-review/2026-11-01-forum-sealed-history.md) records source-backed ADR-2017 dispositions and synthetic fault cases. Core cryptography and migration code execute locally; no production relay, key or purge is involved. The fixture relay does not replace Worker/D1 integration evidence.

## For the business

History can remain signed by its original authors after encryption. An operator still needs separate evidence that every message was discovered, members can read the replacement, plaintext was removed and the audit receipt survived. A single success counter does not establish all four.

## NF-26.1 Discovery warnings survive only in the report

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-zone-migrate/src/relay.rs:144
    Q["Read pages of up to 1000 events<br/>inclusive timestamp cursor and ID deduplication"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-zone-migrate/src/relay.rs:167
    D["Full page adds no new IDs<br/>record ambiguous second and step past it"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-zone-migrate/src/engine.rs:97
    P["Plan carries incomplete_seconds"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-zone-migrate/src/engine.rs:183
    S["Seal records discovered originals<br/>no completeness refusal"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-zone-migrate/src/cli.rs:492
    W["CLI prints warning<br/>recorded-set success can leave plaintext undiscovered"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-zone-migrate/src/plan.rs:124
    C["Already sealed count means marker found<br/>adoption must still open and verify"]
    Q --> D --> P --> S --> W
    P --> C
```

## NF-26.2 Migrator and member reader require different key checks

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-zone-migrate/src/engine.rs:580
    A["Adopt existing envelope<br/>select key by tag zone and epoch"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-zone-migrate/src/engine.rs:558
    O["Verify outer signature and channel<br/>open sealed original and compare identity"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-core/src/sealed.rs:375
    I["Core checks inner kind, signature, ID,<br/>no zk, channel and timestamp"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-zone-migrate/src/engine.rs:732
    V["Verify recorded digest<br/>declared zk public key is not bound here"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/mod.rs:565
    R["Member reader first compares<br/>held key public key with zk public key"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/mod.rs:566
    F["Mismatch gives Failed<br/>even when migrator can open the inner event"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-zone-migrate/src/engine.rs:611
    K["Since 6064622, seal guard: refuse unless the zone's current key matches the<br/>highest observed zk epoch and pubkey of its live posts"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-zone-migrate/src/engine.rs:224
    S["Checked per zone before anything is sealed<br/>adoption and verify are unchanged"]
    A --> O --> I --> V
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-zone-migrate/src/engine.rs:604
    DB["DEBT authors' FOLLOW-UP: a zone with no live encrypted post is refused,<br/>blocking FIRST encryption of an all-plaintext zone until an opt-in lands"]
    K --> S
    K -.-> DB
    S -. binds new envelopes only .-> R
    R --> F
    V -. mismatched admin envelope .-> F
```

## NF-26.3 Purge state trusts an incomplete success response

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-zone-migrate/src/engine.rs:438
    G["Require confirmation and nonempty state<br/>all entries Verified or Purged"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-zone-migrate/src/engine.rs:484
    V["Re-fetch and open this chunk's envelopes<br/>failure blocks that chunk"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-zone-migrate/src/engine.rs:499
    D["Ask relay to delete original IDs"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-zone-migrate/src/engine.rs:513
    P["Every ID not skipped becomes Purged<br/>including an unaccounted-for ID"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-zone-migrate/src/engine.rs:517
    W["Deleted-count mismatch only warns"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-zone-migrate/src/engine.rs:380
    N["Later verify excludes Purged entries<br/>zero checks can report no failures"]
    G --> V --> D --> P --> W --> N
```

## NF-26.4 Silent deletion, relay tombstone and audit receipt are separate operations

```mermaid
sequenceDiagram
    participant C as Admin client
    participant H as Delete handler
    participant D as D1
    participant A as Audit logger
    C->>H: NIP-98 authenticated IDs and reason
    Note over H: admin_events.rs:219<br/>validates auth and body
    H->>D: Read each ID's stored kind
    Note over H,D: admin_events.rs:255-257<br/>absent goes to notFound, other kinds skipped
    H->>D: kind 42 - tombstone under the row's own author, fail-open
    Note over H,D: admin_events.rs:279-284<br/>relay-only, never served to clients
    H->>D: then DELETE the kind-42 row
    Note over H,D: admin_events.rs:286<br/>no kind-5 or broadcast
    D-->>H: Per-ID outcomes or partial failure
    H->>A: Attempt events.delete receipt
    Note over H,A: admin_events.rs:314<br/>discards logger error
    A->>D: Insert admin_log row
    Note over A,D: audit.rs:38<br/>insertion can fail
    H-->>C: 200 on no delete failure<br/>otherwise 500 with partial outcomes
    Note over H,C: admin_events.rs:327<br/>status does not attest audit persistence
```

The tombstone stops the purged plaintext being published again, including after the zone's encryption is turned off; a D1 backup taken before the purge still restores it (`../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/admin_events.rs:26-34`).

## NF-26.5 Malformed-input outcomes differ across readers

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-core/src/sealed.rs:424
    R["Rust strict inner deserialisation<br/>duplicate and unknown fields rejected"]
    %% Source: ../project/agentbox/management-api/lib/zone-keys.js:553
    J["JavaScript JSON.parse first<br/>duplicate property names collapse"]
    %% Source: ../project/agentbox/management-api/lib/zone-keys.js:556
    K["Check remaining seven property names<br/>then verify final signed event"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/mod.rs:555
    S["Rust notices sealed marker without zk<br/>returns Failed"]
    %% Source: ../project/agentbox/management-api/lib/zone-keys.js:598
    P["JavaScript without zk returns Plain<br/>before checking sealed marker"]
    R -. input-parity gap .-> J --> K
    S -. outcome-parity gap .-> P
```

## NF-26.6 Preserved identity does not restore every lookup

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-core/src/sealed.rs:479
    E["Relay stores envelope ID<br/>public sealed tag names original ID"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-core/src/sealed.rs:416
    I["After decryption restore original<br/>author, time, ID and reply tags"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/note_view.rs:121
    N["Permalink asks relay for original ID<br/>purged row is absent"]
    %% Source: ../project/agentbox/scripts/dream-forum-suggestions.mjs:300
    T["Suggestions query original root ID<br/>and reply tags hidden inside ciphertext"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-core/src/sealed.rs:483
    C["Retained old IDs can correlate<br/>envelopes with old plaintext and authors"]
    E --> I
    E --> C
    N -. no matching envelope retrieval .-> I
    T -. no matching envelope retrieval .-> I
```

## NF-26.7 Deleting a sealed original — relay gate, moderator envelope removal, client honour rule

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1320
    K5["kind 5 naming a sealed original's id"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:3054
    MEM["Member below TL3: no row, a sealed tag names it<br/>SealedOriginal, refused whole"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:2952
    WHY["Why: the author is inside the ciphertext<br/>the relay cannot check it"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:3629
    MOD["Admin or TL3: stored, then process_deletion<br/>also removes the wrapping envelopes"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/lib.rs:838
    SHAPE["Only rows with an envelope's whole wire shape<br/>well-formed sealed tag plus a zk tag"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:3446
    ADM["Only envelopes whose author is an admin NOW<br/>fail-open on any lookup error"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:3453
    TB["Each envelope tombstoned under its real author<br/>then deleted, never in one batch"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/stores/tombstones.rs:164
    FOLD["Client fold_deletion records id and deleter<br/>for every kind 5 it receives"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/stores/tombstones.rs:120
    HON["Restored sealed original hidden only when the deleter<br/>is its verified inner author or a relay-vouched admin"]
    K5 --> MEM --> WHY
    K5 --> MOD --> SHAPE --> ADM --> TB
    K5 -.->|"whatever the relay accepted"| FOLD --> HON
```

`classify_member_target` reads the sealed flag before the tombstone flag (`../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:3054-3055`), and the member refusal says the purge does not yet record original authors (`../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:2960-2962`). A moderator's removal is fail-open (`../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:3401-3403`), and envelopes go one by one, tombstone then delete (`../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:3450-3461`).

## NF-26.8 Open items on sealed-original deletion

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:2963
    O19["OPEN kit O19, KF-19: an author cannot delete their own migrated post<br/>the member gate refuses every sealed original"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/lib.rs:870
    O19B["The purge's row-author tombstone exists<br/>but the classifier reads the sealed flag first"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/stores/tombstones.rs:121
    DEAD["For a new kind 5 the client's author branch is reachable<br/>only when the author is also admin or TL3"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:3396
    O18["DEBT kit O18, KF-18: a migrator who lost admin leaves<br/>envelopes, any current admin's envelope is removable"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/stores/tombstones.rs:17
    O25["OPEN kit O25, KF-27: the client cannot identify TL3<br/>moderators, so on a restored sealed original<br/>their deletions hide nothing client-side"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/whitelist.rs:180
    O25B["check-whitelist isAdmin reads whitelist.is_admin only<br/>whitelist.rs:135"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/auth.rs:184
    O25C["The relay also honours static ADMIN_PUBKEYS admins<br/>auth.rs:184-189, which the client never sees as admin"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/stores/tombstones.rs:13
    DRIFT1["DOC-DRIFT: says the relay accepts a kind 5 naming a<br/>restored original from anyone - members are now refused"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/admin_events.rs:18
    DRIFT2["DOC-DRIFT: names fold_deletions, which no longer exists<br/>also relay_do/nip_handlers.rs:2941-2942"]
    O19 --> O19B --> DEAD
    O25 --> O25B --> O25C
    O18
    DRIFT1
    DRIFT2
```

The two `fold_deletions` comments are recorded as left alone in ADR-2017's 2026-11-22 section (`../nostr-rust-forum/docs/adr/ADR-2017-sealed-original-history-migration.md:262`); `tombstones::fold_deletion` replaced the function (`../nostr-rust-forum/crates/nostr-bbs-forum-client/src/stores/tombstones.rs:164`).
