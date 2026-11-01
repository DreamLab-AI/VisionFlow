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
  - ../project/agentbox/management-api/lib/zone-keys.js
  - ../project/agentbox/scripts/dream-forum-suggestions.mjs
verified_commit: {nostr-rust-forum: c39f41f104b114d6a3fa29013daaf6fa58b45080, agentbox: 9dfc27b16e82e9786535a55e790710493dd51fdf}
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
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-zone-migrate/src/engine.rs:179
    S["Seal records discovered originals<br/>no completeness refusal"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-zone-migrate/src/cli.rs:462
    W["CLI prints warning<br/>recorded-set success can leave plaintext undiscovered"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-zone-migrate/src/plan.rs:115
    C["Already sealed count means marker found<br/>adoption must still open and verify"]
    Q --> D --> P --> S --> W
    P --> C
```

## NF-26.2 Migrator and member reader require different key checks

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-zone-migrate/src/engine.rs:566
    A["Adopt existing envelope<br/>select key by tag zone and epoch"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-zone-migrate/src/engine.rs:544
    O["Verify outer signature and channel<br/>open sealed original and compare identity"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-core/src/sealed.rs:375
    I["Core checks inner kind, signature, ID,<br/>no zk, channel and timestamp"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-zone-migrate/src/engine.rs:654
    V["Verify recorded digest<br/>declared zk public key is not bound here"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/mod.rs:489
    R["Member reader first compares<br/>held key public key with zk public key"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/mod.rs:490
    F["Mismatch gives Failed<br/>even when migrator can open the inner event"]
    A --> O --> I --> V
    R --> F
    V -. mismatched admin envelope .-> F
```

## NF-26.3 Purge state trusts an incomplete success response

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-zone-migrate/src/engine.rs:424
    G["Require confirmation and nonempty state<br/>all entries Verified or Purged"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-zone-migrate/src/engine.rs:470
    V["Re-fetch and open this chunk's envelopes<br/>failure blocks that chunk"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-zone-migrate/src/engine.rs:485
    D["Ask relay to delete original IDs"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-zone-migrate/src/engine.rs:499
    P["Every ID not skipped becomes Purged<br/>including an unaccounted-for ID"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-zone-migrate/src/engine.rs:503
    W["Deleted-count mismatch only warns"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-zone-migrate/src/engine.rs:366
    N["Later verify excludes Purged entries<br/>zero checks can report no failures"]
    G --> V --> D --> P --> W --> N
```

## NF-26.4 Silent deletion and audit receipt are separate operations

```mermaid
sequenceDiagram
    participant C as Admin client
    participant H as Delete handler
    participant D as D1
    participant A as Audit logger
    C->>H: NIP-98 authenticated IDs and reason
    Note over H: admin_events.rs:209<br/>validates auth and body
    H->>D: Read kind then delete each kind-42 ID
    Note over H,D: admin_events.rs:246<br/>no kind-5 or broadcast
    D-->>H: Per-ID outcomes or partial failure
    H->>A: Attempt events.delete receipt
    Note over H,A: admin_events.rs:294<br/>discards logger error
    A->>D: Insert admin_log row
    Note over A,D: audit.rs:38<br/>insertion can fail
    H-->>C: 200 on no delete failure<br/>otherwise 500 with partial outcomes
    Note over H,C: admin_events.rs:307<br/>status does not attest audit persistence
```

## NF-26.5 Malformed-input outcomes differ across readers

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-core/src/sealed.rs:424
    R["Rust strict inner deserialisation<br/>duplicate and unknown fields rejected"]
    %% Source: ../project/agentbox/management-api/lib/zone-keys.js:551
    J["JavaScript JSON.parse first<br/>duplicate property names collapse"]
    %% Source: ../project/agentbox/management-api/lib/zone-keys.js:554
    K["Check remaining seven property names<br/>then verify final signed event"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/mod.rs:479
    S["Rust notices sealed marker without zk<br/>returns Failed"]
    %% Source: ../project/agentbox/management-api/lib/zone-keys.js:596
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
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/note_view.rs:106
    N["Permalink asks relay for original ID<br/>purged row is absent"]
    %% Source: ../project/agentbox/scripts/dream-forum-suggestions.mjs:296
    T["Suggestions query original root ID<br/>and reply tags hidden inside ciphertext"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-core/src/sealed.rs:483
    C["Retained old IDs can correlate<br/>envelopes with old plaintext and authors"]
    E --> I
    E --> C
    N -. no matching envelope retrieval .-> I
    T -. no matching envelope retrieval .-> I
```
