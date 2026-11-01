---
id: NF-17
title: Relay read caches — authority, cost and activity persistence
area: nostr-rust-forum
governing:
  - ../nostr-rust-forum/docs/adr/ADR-2018-relay-read-caches-and-activity-throttle.md
adrs: [ADR-2018]
sources:
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/read_cache.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/broadcast.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/mod.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/session.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/storage.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/auth.rs
verified_commit: {nostr-rust-forum: c39f41f104b114d6a3fa29013daaf6fa58b45080}
---

## For developers

The [ADR-2018 section audit](../../estate-review/2026-11-01-forum-read-cache-and-activity.md) separates native counterexamples from source-only failure paths. All cited implementation files match the website-consumed kit pin.

## For the business

The optimisation saves repeated successful lookups. Privacy, activity credit and operating cost still need explicit failure and restart rules; the present code does not establish the ADR's universal cost or five-minute persistence guarantees.

## NF-17.1 Successful cache hits and deliberate misses

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:2156
    A["Resolve viewer from session key"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:2216
    B["Device-owner memo: 60 seconds<br/>including None"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/auth.rs:46
    C["Admin cache: 30 seconds"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:2203
    D["Cohorts plus admin flag: 60 seconds<br/>including empty and false"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:2189
    E["Zone memo: 60 seconds<br/>only positive channel bindings"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:2194
    F["Unbound channel: query again next event<br/>zone cardinality is per channel, not viewer"]
    A --> B --> C --> D
    D --> E --> F
```

Native wrapper tests: one bound channel costs one lookup across 50 calls; an unbound channel or 50 distinct bound channels costs 50. The separate admin flag can remain true through the cohort cache.

## NF-17.2 Missing binding and failed lookup share a public fallback

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:2332
    A["Channel ID from kind 40, 41 or 42"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:2350
    B["Read cached binding or query channel_zones"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:2352
    C["Some zone: apply member and visibility rules"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust.rs:584
    D["Absent row OR database/bind/query failure: None"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:2372
    E["None skips this zone gate<br/>event reaches Deliver if other gates pass"]
    A --> B
    B --> C
    B --> D --> E
```

Source-only failure finding: distinguish unavailable from unbound. A whole-database failure may prevent history retrieval first; this panel concerns an event already available to authorisation.

## NF-17.3 History and live metadata do not share the same dispatch

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1883
    A["History: every returned event goes to authoriser"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:2354
    B["Kind 41 in hidden zone: withhold from non-member"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/broadcast.rs:76
    C["Live needs_gate includes 40 and 42<br/>plus calendar and kanban; omits 41"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/broadcast.rs:149
    D["Gated kinds consult authoriser per recipient"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/broadcast.rs:162
    E["Kind 41 skips zone context<br/>matching subscription gets original event"]
    A --> B
    C -->|40 or 42| D
    C -->|41| E
```

The unchanged native dispatcher sends kind 41 despite a fixture authoriser that would withhold; kind-40 and kind-42 controls consult the gate and send nothing. This is metadata routing evidence, not a live relay probe.

## NF-17.4 Activity flush is triggered by another call

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1912
    A["Authenticated REQ with delivered events"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/read_cache.rs:128
    B["Add delivered count to pending reads"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/read_cache.rs:136
    C["First call, elapsed 300 seconds<br/>or at least 50 pending reads?"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/read_cache.rs:139
    D["No: retain count in memory<br/>no deadline timer in this ledger"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/read_cache.rs:162
    E["Yes: remove count and advance last_flush"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:2246
    F["Attempt read increment, last-active update<br/>then promotion check"]
    A --> B --> C
    C -->|no| D
    C -->|yes| E --> F
    D -. next activity .-> C
```

Native tests retain 49 pending reads through a day of other-user activity. A consumed but unpersisted Flush is not retried; repeated 50-read calls can flush every time at one timestamp.

## NF-17.5 Reconstruction restores different kinds of state

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/mod.rs:113
    A["New Durable Object constructor"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/mod.rs:123
    B["New empty caches and activity ledger"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/session.rs:115
    C["Recover socket session and persisted subscriptions"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/read_cache.rs:115
    D["Old pending activity is not restored"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/mod.rs:371
    E["Idle alarm clears rate and connection maps<br/>does not flush activity"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/read_cache.rs:71
    F["Expired cache entries miss but remain allocated"]
    A --> B --> D
    A --> C
    B --> F
    E -. no activity drain .-> D
```

Native reconstruction loses the old pending count; 10,000 expired cache entries remain allocated after misses. These are state-model tests, not actual Cloudflare eviction or resource-exhaustion tests.

## NF-17.6 Post creation counts follow save success, including no-op inserts

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1552
    A["Admitted event enters save_event"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/storage.rs:82
    B["INSERT ON CONFLICT id DO NOTHING"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/storage.rs:142
    C["Regular insert returns success when run succeeds"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1558
    D["Qualifying kinds increment posts_created<br/>after ACK and broadcast"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust.rs:432
    E["Counter update ignores database result"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1649
    F["Existing-ID check suppresses push fan-out<br/>not the creation counter above"]
    A --> B --> C --> D --> E
    D --> F
```

Source-only exactness gap: an admitted duplicate can increment the counter again, and a failed update can lose activity. Full admission/rate limits still apply; no duplicate event was published in this audit.
