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
verified_commit: {nostr-rust-forum: c39f41f104b114d6a3fa29013daaf6fa58b45080}
---

## For developers

[The trust-transition audit](../../estate-review/2026-11-01-forum-trust-transitions.md) records 32 native trust cases, five existing demotion SQL tests and six promotion SQL scenarios. Native cases call the actual policy/engine through its existing in-memory store. Promotion probes execute extracted SQL with chosen interleavings, not the async Worker handler or live D1.

## For the business

Members need stable rights and intelligible reasons for changes. Operators need an audit record that agrees with committed state and a sweep that eventually reaches every eligible member. A balanced counter alone establishes neither fairness nor an explanation.

## NF-24.1 Demotion commits only against the observed snapshot

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/wrangler.toml:123
    A["Configured five-minute cron"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust_sweep.rs:253
    B["Start with no cursor<br/>page eligible TL1/TL2 rows"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust.rs:326
    C["Pure policy checks exclusions,<br/>inactivity and hysteresis"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust_sweep.rs:418
    D["Guarded UPDATE compares level,<br/>admin flag, timestamps and counters"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust_sweep.rs:425
    E["Same batch: audit INSERT SELECT<br/>only WHERE changes() = 1"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust_sweep.rs:503
    F["Both results must report one change<br/>otherwise return failure/conflict"]
    A --> B --> C -->|Demote| D --> E --> F
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust_sweep.rs:281
    C -->|Hold| H["Increment held total<br/>discard named reason"]
```

## NF-24.2 Advancing within a sweep does not guarantee progress across ticks

```mermaid
flowchart LR
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust_sweep.rs:253
    A["New tick starts before first row"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust_sweep.rs:275
    B["Held or failed prefix<br/>still counts as scanned"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust_sweep.rs:299
    C["Cap reached: return truncated<br/>tail never visited"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust_sweep.rs:162
    D["Report last pubkey only<br/>no persisted cursor or resume input"]
    A --> B --> C --> D --> A
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust_sweep.rs:66
    E["Production cap: 50,000<br/>Probe: cap 3, four rows, three ticks"]
    E -.-> C
```

The native fixture reproduces starvation at a smaller configurable cap. Production starvation requires a cap-sized eligible prefix that remains held or failed; no real membership population was inspected. Increasing the fixture budget reaches the tail, which is the positive control.

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
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust_sweep.rs:281
    H --> R["Sweep reports aggregate held count<br/>reason is not retained"]
```
