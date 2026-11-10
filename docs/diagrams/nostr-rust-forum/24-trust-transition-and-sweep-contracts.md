---
id: NF-24
title: Trust transitions, sweep fairness and audit consistency
area: nostr-rust-forum
governing:
  - ../nostr-rust-forum/docs/adr/ADR-2006-trust-demotion-cron-tl3-never.md
  - ../nostr-rust-forum/docs/archive/adr/ADR-102-trust-demotion-inactivity-sweep.md
adrs: [ADR-2006]
sources:
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust_sweep.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/lib.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/wrangler.toml
verified_commit: {nostr-rust-forum: 6064622dbb1a7cc1ba2d072109f9e3e8a96157df}
---

## For developers

The bounded-prefix starvation finding has a [scoped source-repair successor, NF-29](29-external-review-repairs.md#nf-293-notification-claims-and-bounded-sweeps-retain-concurrent-work). The starvation finding is marked resolved at the re-verified stamp 6064622; other trust consistency findings remain separate.

[The trust-transition audit](../../estate-review/2026-11-01-forum-trust-transitions.md) records 32 native trust cases, five existing demotion SQL tests and six promotion SQL scenarios. Native cases call the actual policy/engine through its existing in-memory store. Promotion probes execute extracted SQL with chosen interleavings, not the async Worker handler or live D1.

## For the business

Members need stable rights and intelligible reasons for changes. Operators need an audit record that agrees with committed state and a sweep that eventually reaches every eligible member. A balanced counter alone establishes neither fairness nor an explanation.

## NF-24.1 Demotion commits only against the observed snapshot

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/wrangler.toml:123
    A["Configured five-minute cron"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust_sweep.rs:544
    B["Start from the stored cursor, none on the first tick<br/>page eligible TL1/TL2 rows"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust.rs:326
    C["Pure policy checks exclusions,<br/>inactivity and hysteresis"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust_sweep.rs:434
    D["Guarded UPDATE compares level,<br/>admin flag, timestamps and counters"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust_sweep.rs:441
    E["Same batch: audit INSERT SELECT<br/>only WHERE changes() = 1"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust_sweep.rs:519
    F["Both results must report one change<br/>otherwise return failure/conflict"]
    A --> B --> C -->|Demote| D --> E --> F
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust_sweep.rs:297
    C -->|Hold| H["Increment held total<br/>discard named reason"]
```

## NF-24.2 Advancing within a sweep does not guarantee progress across ticks

```mermaid
flowchart LR
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust_sweep.rs:544
    A["RESOLVED at nostr-rust-forum 6064622: a new tick resumes<br/>from the stored cursor"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust_sweep.rs:294
    B["Held or failed prefix<br/>still counts as scanned"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust_sweep.rs:315
    C["Cap reached: return truncated<br/>the tail waits for the next tick"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust_sweep.rs:562
    D["RESOLVED at nostr-rust-forum 6064622: persist the resume cursor<br/>in trust_sweep_state for the next tick"]
    A --> B --> C --> D --> A
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust_sweep.rs:66
    E["Production cap: 50,000<br/>Probe: cap 3, four rows, three ticks"]
    E -.-> C
```

At the earlier stamp the native fixture reproduced starvation at a smaller configurable cap, because each tick restarted with no cursor. At 6064622 a truncated or aborted sweep stores its resume cursor and the next tick reads it back (`../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust_sweep.rs:544-557`, `../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust_sweep.rs:560-564`), so a held or failed prefix no longer hides the tail. No real membership population was inspected.

## NF-24.3 Promotion can overwrite a grant made after its read

```mermaid
sequenceDiagram
    participant P as check_promotion
    participant D as whitelist
    participant A as Concurrent admin change
    participant L as admin_log
    P->>D: Read TL0 with qualifying counters trust.rs:207
    P->>P: Snapshot is not TL3, compute TL2 trust.rs:221 trust.rs:225
    A->>D: Selected fixture schedule grants TL3
    P->>D: UPDATE by pubkey only, writes TL2 trust.rs:238
    P->>L: Separate INSERT claims TL0 to TL2 trust.rs:252
    Note over P,L: Execution errors are ignored, returns planned level trust.rs:268<br/>SQLite fixtures also cover failed writes and deleted rows<br/>No live Worker race was executed
```

## NF-24.4 Inactivity opens a gate; it does not age the counters

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust.rs:334
    A["TL3, floor and admin guards"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust.rs:349
    B["Inactive for full configured policy window?"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust.rs:359
    C["Counters below hysteresis band<br/>or TL2 moderation history?"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust.rs:366
    H["No: hold ClearsHysteresis<br/>time alone does not lower counters"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust.rs:377
    D["Yes: TL2 to TL1 or TL0<br/>TL1 to TL0"]
    A --> B -->|yes| C
    C -->|no| H
    C -->|yes| D
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust_sweep.rs:297
    H --> R["Sweep reports aggregate held count<br/>reason is not retained"]
```
