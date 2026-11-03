---
id: NF-18
title: Web Push — registration, wake-up ordering and display receipts
area: nostr-rust-forum
governing:
  - ../nostr-rust-forum/docs/adr/ADR-2023-content-free-web-push-for-pwa-alerts.md
adrs: [ADR-2023]
sources:
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/push.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/push_fanout.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/lib.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/utils/push.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/js/push.js
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/sw.js
verified_commit: {nostr-rust-forum: c39f41f104b114d6a3fa29013daaf6fa58b45080}
---

## For developers

The historical drain/bump/logout findings below have a [scoped source-repair successor, NF-29](29-external-review-repairs.md#nf-293-notification-claims-and-bounded-sweeps-retain-concurrent-work). This topic retains its declared audit revision; remaining device/runtime boundaries are unchanged.

The [ADR-2023 section audit](../../estate-review/2026-11-01-forum-web-push.md) records 85 selected native/worker checks, revision differences and untested device boundaries. These panels describe current code and demonstrated interleavings, not a deployed push receipt.

## For the business

Empty push requests keep message text away from providers. Reliable alerts still require consistent account teardown, quiet-hour enforcement and a distinction between provider acceptance, device fetch and visible notification. Activation remains staged.

## NF-18.1 Registration stores one credential copy but teardown expects two

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/push.rs:1043
    A["Public config reports enablement and public key"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/utils/push.rs:295
    B["Permission and browser subscription<br/>NIP-98 registration rotates drain token"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/js/push.js:145
    C["Store drain credentials in IndexedDB"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/utils/push.rs:314
    D["Set localStorage active and refreshed flags<br/>no push_cfg write in register"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/utils/push.rs:364
    E["Logout reads localStorage push_cfg<br/>fresh registration has no such copy"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/utils/push.rs:377
    F["Browser unsubscribe and IndexedDB cleanup<br/>still attempted in background"]
    A --> B
    B --> C --> D
    D --> E --> F
```

A bridge fixture confirms the IndexedDB-only write. Source tracing establishes that the Rust caller never creates the localStorage copy used by token-based server teardown. Browser unsubscribe can still succeed.

## NF-18.2 Fan-out is bounded before pending counts exist

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/push_fanout.rs:47
    A["After save and sender OK<br/>check push enabled and eligible event kind"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/push_fanout.rs:57
    B["Named recipients plus opted-in candidates"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/push_fanout.rs:77
    C["Load prefs and suppress categories<br/>missing or failed lookup uses defaults"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/push_fanout.rs:85
    D["Truncate candidates before read checks"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/push_fanout.rs:94
    E["Kinds 40 and 42 use shared read authoriser"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/push_fanout.rs:116
    F["Load devices and cap jobs<br/>omitted work has no pending counter"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/push_fanout.rs:139
    G["wait_until delivery bumps counters<br/>then claims and sends wake-ups"]
    A --> B --> C --> D --> E --> F --> G
```

Caps limit work but are not a durable queue. The shared read gate retains the failure boundary documented in NF-17. Candidate resolution occurs before wait_until; only delivery is detached.

## NF-18.3 A fast drain can precede successful-send bookkeeping

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/push.rs:234
    A["Counter bump advances bump_seq"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/push.rs:686
    B["Claim push slot, then send empty request"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/push.rs:1311
    C["Worker drains the first count"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/push.rs:234
    D["Later event bumps count while first send<br/>still awaits result bookkeeping"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/push.rs:197
    E["Success sets pushed_seq to CURRENT bump_seq"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/push.rs:172
    F["Pending count remains but due query<br/>requires bump_seq greater than pushed_seq"]
    A --> B --> C --> D --> E --> F
```

Actual SQLite statements reproduce this ordering: one undrained count, no due wake-up. This is an allowed interleaving test, not a measured production frequency. Completion should bind to the sequence actually claimed.

## NF-18.4 Draining selects and subtracts without an atomic claim

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/push.rs:1300
    A["Read subscription counts and token hash"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/push.rs:1307
    B["Validate bearer token against selected hash"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/push.rs:161
    C["Separately subtract snapshot counts<br/>mutation matches id only"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/push.rs:1325
    D["Return the earlier counts and deep links"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/push.rs:158
    E["Concurrent drains can subtract twice<br/>and consume a newer event"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/push.rs:138
    F["Token/owner may rotate before subtraction<br/>old request can mutate replacement row"]
    A --> B --> C --> D
    E -. interleaves .-> C
    F -. interleaves .-> C
```

Two native interleavings demonstrate count loss and mutation after registration rotation. Subtracting a snapshot preserves one intervening bump only when competing drains/registrations do not invalidate that snapshot.

## NF-18.5 A drained count is not an acknowledged display

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/sw.js:306
    A["Push wakes worker<br/>read credentials then await drain"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/sw.js:307
    B["Any visible forum window?"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/sw.js:311
    C["Yes: return without showing notification<br/>login page also qualifies"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/sw.js:316
    D["No: compose generic category notices<br/>empty or failed drain gets generic fallback"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/sw.js:323
    E["Await showNotification<br/>rejection has no application retry or ack"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/sw.js:244
    F["Unresolved drain has no deadline<br/>fallback is never reached while pending"]
    A --> B
    A -. unresolved fetch .-> F
    B -->|yes| C
    B -->|no| D --> E
```

Worker fixtures reproduce visible-login suppression, rejected display after drain, and no fallback during a stalled fetch. WebKit requires user-visible notification for each push; actual device enforcement remains untested.

## NF-18.6 Trailing flush and provider acceptance have limited guarantees

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/lib.rs:1181
    A["Five-minute cron invokes trailing flush"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/push.rs:762
    B["Select at most 100 due devices<br/>old undrained counts and sequence gap"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/push.rs:778
    C["No current preference or read-authority check<br/>queued alerts can survive later quiet hours"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/push.rs:781
    D["Send without bumping counts again"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/push.rs:442
    E["Successful provider status means accepted<br/>not proof of device display"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/push.rs:744
    F["Empty body still carries category Topic<br/>plus category-dependent TTL and urgency"]
    A --> B --> C --> D --> E
    D --> F
```

The actual due-query test still selects a queued DM after preferences disable it. RFC 8030 distinguishes provider acceptance from delivery; the clear Topic header carries the category, even with no message payload.
