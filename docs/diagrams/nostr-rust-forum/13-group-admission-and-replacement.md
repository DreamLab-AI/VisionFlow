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
verified_commit: {nostr-rust-forum: de31aaa6e6c6b672cc310a2659cf33fd4887602c}
---

## For developers

The [forum audit](../../estate-review/2026-11-01-forum-spaces-and-replication.md) tests unchanged pure read/filter modules, real signed builders and extracted production SQL in SQLite. Full Worker admission and live sockets were not executed.

## For the business

An advertised protocol and signed events do not establish private group membership. The proposed engine must add authority checks and reliable replacement before spaces depend on it.

## NF-13.1 Neither metadata builder is used, and NIP-11 no longer advertises 29

```mermaid
flowchart TD
    ADV["NIP-11 drops 29: the relay gates NIP-29 admin kinds<br/>but implements no group semantics<br/>nip11.rs:16,212"]
    OLD["create_group_metadata emits h, no d<br/>deprecated, no caller<br/>groups.rs:138,139,148"] --> GATE["Current group management gate<br/>requires h and admin<br/>nip_handlers.rs:1329,1333,1337"]
    NEW["build_group_metadata emits d, no h<br/>no caller elsewhere in crates/<br/>groups.rs:262,271"] --> DENY["Current gate rejects missing h<br/>nip_handlers.rs:1334"]
    GATE --> TODO["Relay-key-generated metadata remains TODO<br/>nip_handlers.rs:1330,1331"]
```

**What it shows.** Both builders sign valid events, but only the legacy tag shape satisfies the current group-tag check, and the legacy builders are now deprecated as callerless. **Why it is this way.** The admin gate predates the proposed relay-signed engine; the relay stopped advertising NIP-29 because the ADR-2029 room dialect is a-scoped (commit 75e48b4, merged by a80c572).

## NF-13.2 Current read protection does not include NIP-29 group membership

```mermaid
flowchart TD
    REQ["Query matching stored events<br/>nip_handlers.rs:1939"] --> EACH["Authorise each event before delivery<br/>nip_handlers.rs:1961,1963"]
    EACH --> PROTECTED["Protected-kind predicate, inbox keys only widen 1059<br/>nip42.rs:99,107"]
    PROTECTED --> ZONE["Channel-zone extraction handles 40, 41, 42<br/>filter.rs:337,339,340,341"]
    ZONE --> DELIVER["No matching gate ends in Deliver<br/>nip_handlers.rs:2738"]
```

**What it shows.** Anonymous group kinds pass the tested pure predicates; correspondence remains protected. **Why it is this way.** Current gates cover other protocols. The proposed on-to-off rollback must preserve protection for any private group data stored after activation.

## NF-13.3 Replacement now follows the NIP-01 order, still with no revision key

```mermaid
flowchart TD
    REP["Replaceable: key is pubkey and kind<br/>storage.rs:143,59"] --> INS["Insert only if no stored row is newer,<br/>or equal created_at with a lower id<br/>storage.rs:65,67"]
    ADDR["Addressable: key adds d_tag<br/>storage.rs:174,38"] --> INS2["Same guard keyed on d_tag<br/>storage.rs:44,46"]
    INS --> PRUNE["Then prune the rows the incoming event beats<br/>storage.rs:71,74"]
    INS2 --> PRUNE2["Then prune by the same order<br/>storage.rs:51,54"]
    PRUNE --> KEPT["Batch reports whether the insert landed<br/>storage.rs:165,170"]
    PRUNE2 --> KEPT2["Refused stale event is not acknowledged as kept<br/>storage.rs:205,210"]
    KEPT --> SAVE["Caller broadcasts only a kept event<br/>nip_handlers.rs:1618"]
    KEPT2 --> SAVE
    SAVE --> READ["Read by descending created_at<br/>storage.rs:393,395,396"]
```

**What it shows.** A late older event is now refused and a same-second tie keeps the lowest id (commit 75e48b4 for addressable kinds, 22ad7e2 for plain replaceable ones). The audit's SQLite result of retained late rows and coexisting same-second rows was measured at the earlier stamp and no longer describes this code; it was not re-run. **Why it is this way.** Revision tags are still not part of the replacement key, so a later timestamp still erases a higher proposed revision; the ADR-2031 migration remains a prerequisite.
