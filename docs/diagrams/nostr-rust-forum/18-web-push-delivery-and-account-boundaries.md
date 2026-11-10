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
verified_commit: {nostr-rust-forum: 6064622dbb1a7cc1ba2d072109f9e3e8a96157df}
---

## For developers

The historical drain/bump/logout findings below have a [scoped source-repair successor, NF-29](29-external-review-repairs.md#nf-293-notification-claims-and-bounded-sweeps-retain-concurrent-work). The drain, bump and logout findings below are marked resolved at the re-verified stamp 6064622; remaining device/runtime boundaries are unchanged.

The [ADR-2023 section audit](../../estate-review/2026-11-01-forum-web-push.md) records 85 selected native/worker checks, revision differences and untested device boundaries. These panels describe current code and demonstrated interleavings, not a deployed push receipt.

## For the business

Empty push requests keep message text away from providers. Reliable alerts still require consistent account teardown, quiet-hour enforcement and a distinction between provider acceptance, device fetch and visible notification. Activation remains staged.

## NF-18.1 Registration stores one credential copy but teardown expects two

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/push.rs:1064
    A["Public config reports enablement and public key"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/utils/push.rs:295
    B["Permission and browser subscription<br/>NIP-98 registration rotates drain token"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/js/push.js:145
    C["Store drain credentials in IndexedDB"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/utils/push.rs:314
    D["RESOLVED at nostr-rust-forum 6064622: register also writes the<br/>localStorage push_cfg copy, then sets the active flag"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/utils/push.rs:369
    E["Logout reads localStorage push_cfg<br/>which register now writes"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/utils/push.rs:382
    F["Browser unsubscribe and IndexedDB cleanup<br/>still attempted in background"]
    A --> B
    B --> C --> D
    D --> E --> F
```

At the earlier stamp a bridge fixture confirmed an IndexedDB-only write and the Rust caller never created the localStorage copy used by token-based server teardown. At 6064622 register stores that copy (`../nostr-rust-forum/crates/nostr-bbs-forum-client/src/utils/push.rs:314-318`) and logout reads it for the beacon (`../nostr-rust-forum/crates/nostr-bbs-forum-client/src/utils/push.rs:373-377`). Browser unsubscribe still runs in the background.

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
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/push.rs:237
    A["Counter bump advances bump_seq"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/push.rs:705
    B["Claim push slot, then send empty request"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/push.rs:1340
    C["Worker drains the first count"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/push.rs:237
    D["Later event bumps count while first send<br/>still awaits result bookkeeping"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/push.rs:198
    E["RESOLVED at nostr-rust-forum 6064622: success sets pushed_seq<br/>to the bump_seq read before the send"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/push.rs:173
    F["The later bump stays due because the due query<br/>requires bump_seq greater than pushed_seq"]
    A --> B --> C --> D --> E --> F
```

At the earlier stamp actual SQLite statements reproduced this ordering: one undrained count, no due wake-up. At 6064622 completion binds to the sequence read before the send (`../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/push.rs:700`, `../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/push.rs:722`) and `SQL_MARK_PUSHED` takes the larger of the stored and claimed sequence, so the later bump remains due.

## NF-18.4 Draining selects and subtracts without an atomic claim

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/push.rs:1328
    A["Read subscription counts and token hash"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/push.rs:1335
    B["Validate bearer token against selected hash"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/push.rs:1340
    C["RESOLVED at nostr-rust-forum 6064622: claim the exact snapshot in one<br/>conditional UPDATE on counts, token hash and bump_seq"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/push.rs:1361
    D["Return the claimed counts and deep links"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/push.rs:1326
    E["RESOLVED at nostr-rust-forum 6064622: a racing drain fails the claim<br/>and retries, at most eight times"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/push.rs:200
    F["RESOLVED at nostr-rust-forum 6064622: a rotated token no longer matches<br/>so an old request cannot mutate the replacement row"]
    A --> B --> C --> D
    E -. interleaves .-> C
    F -. interleaves .-> C
```

At the earlier stamp two native interleavings demonstrated count loss and mutation after registration rotation. At 6064622 the drain claims the snapshot atomically, so a competing drain, bump or registration invalidates the comparison and the loser retries (`../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/push.rs:1338-1356`).

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
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/push.rs:783
    B["Select at most 100 due devices<br/>old undrained counts and sequence gap"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/push.rs:799
    C["No current preference or read-authority check<br/>queued alerts can survive later quiet hours"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/push.rs:802
    D["Send without bumping counts again"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/push.rs:445
    E["Successful provider status means accepted<br/>not proof of device display"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/push.rs:765
    F["Empty body still carries category Topic<br/>plus category-dependent TTL and urgency"]
    A --> B --> C --> D --> E
    D --> F
```

The actual due-query test still selects a queued DM after preferences disable it. RFC 8030 distinguishes provider acceptance from delivery; the clear Topic header carries the category, even with no message payload.
