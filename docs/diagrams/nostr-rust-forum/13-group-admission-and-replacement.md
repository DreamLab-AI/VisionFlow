---
id: NF-13
title: Current group admission and replacement gaps
area: nostr-rust-forum
governing:
  - ../nostr-rust-forum/docs/BASELINE-architecture.md
adrs: [ADR-2030, ADR-2031, ADR-2032]
sources:
  - ../nostr-rust-forum/docs/BASELINE-architecture.md
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/groups.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/nip11.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip42.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/filter.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/storage.rs
verified_commit: {nostr-rust-forum: 6064622dbb1a7cc1ba2d072109f9e3e8a96157df}
---

## For developers

The [forum audit](../../estate-review/2026-11-01-forum-spaces-and-replication.md) tests unchanged pure read/filter modules, real signed builders and extracted production SQL in SQLite. Full Worker admission and live sockets were not executed.

## For the business

An advertised protocol and signed events do not establish private group membership. The proposed engine must add authority checks and reliable replacement before spaces depend on it.

## NF-13.1 The advertised protocol has incompatible metadata builders and a legacy gate

```mermaid
flowchart TD
    ADV["NIP-11 still advertises 29<br/>nip11.rs:207"]
    OLD["create_group_metadata emits h, no d<br/>groups.rs:137,146"] --> GATE["Current group management gate<br/>requires h and admin<br/>nip_handlers.rs:1299,1303,1307"]
    NEW["build_group_metadata emits d, no h<br/>groups.rs:254,263"] --> DENY["Current gate rejects missing h<br/>nip_handlers.rs:1304"]
    GATE --> TODO["Relay-key-generated metadata remains TODO<br/>nip_handlers.rs:1300,1301"]
```

**What it shows.** Both builders sign valid events, but only the legacy tag shape satisfies the current group-tag check. **Why it is this way.** The existing admin gate predates the proposed relay-signed group engine; NIP-11 advertisement does not close that gap.

## NF-13.2 Current read protection does not include NIP-29 group membership

```mermaid
flowchart TD
    REQ["Query matching stored events<br/>nip_handlers.rs:1865"] --> EACH["Authorise each event before delivery<br/>nip_handlers.rs:1883,1885"]
    EACH --> PROTECTED["Protected-kind predicate<br/>nip42.rs:85,92"]
    PROTECTED --> ZONE["Channel-zone extraction handles 40, 41, 42<br/>filter.rs:337,339,340,341"]
    ZONE --> DELIVER["No matching gate ends in Deliver<br/>nip_handlers.rs:2372"]
```

**What it shows.** Anonymous group kinds pass the tested pure predicates; correspondence remains protected. **Why it is this way.** Current gates cover other protocols. The proposed on-to-off rollback must preserve protection for any private group data stored after activation.

## NF-13.3 Current replacement deletes older timestamps then inserts by event id

```mermaid
flowchart TD
    REP["Replaceable: pubkey and kind<br/>storage.rs:98,100"] --> BATCH["Batch deletion then insertion<br/>storage.rs:117"]
    ADDR["Addressable: pubkey, kind and d_tag<br/>storage.rs:119,121"] --> BATCH2["Batch deletion then insertion<br/>storage.rs:139"]
    INSERT["Insertion ignores only duplicate id<br/>storage.rs:80,82"] --> BATCH
    INSERT --> BATCH2
    BATCH --> READ["Read by descending created_at<br/>storage.rs:320,322,323"]
    BATCH2 --> READ
```

**What it shows.** Real SQLite retains and returns late older rows; same-second rows coexist, and later timestamps erase higher proposed revisions. **Why it is this way.** Revision tags are not part of the current replacement key or comparison. The ADR-2031 migration is still a prerequisite.
