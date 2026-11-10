---
id: NF-27
title: Zone key distribution, rotation and recovery
area: nostr-rust-forum
governing:
  - ../nostr-rust-forum/docs/adr/ADR-2016-end-to-end-encrypted-zones.md
  - ../nostr-rust-forum/docs/adr/ADR-2026-zone-key-recovery-master.md
adrs: [ADR-2016, ADR-2026]
sources:
  - ../nostr-rust-forum/docs/adr/ADR-2016-end-to-end-encrypted-zones.md
  - ../nostr-rust-forum/docs/adr/ADR-2026-zone-key-recovery-master.md
  - ../nostr-rust-forum/README.md
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/mod.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/store.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/auto_grant.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/grant_ledger.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/admin/zone_encryption.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/stores/indexed_db.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/auth/mod.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/relay.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/zone_config.rs
verified_commit: 6064622dbb1a7cc1ba2d072109f9e3e8a96157df
---

## For developers

[The audit report](../../estate-review/2026-11-01-forum-zone-key-distribution.md) assesses all sections of ADR-2016 and superseded ADR-2026. Native probes establish planning, ledger and crypto behaviour; account switching and full relay/browser acceptance remain untested. The NIP-171 upgrade branch is separate from the main implementation drawn here.

## For the business

A generated key, an accepted grant and a member able to read are different outcomes. Administrators need a clear activation state, repairable history delivery and tested device recovery before they can rely on removal or recovery promises.

## NF-27.1 Rotation does not force every writer onto the new epoch

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/admin/zone_encryption.rs:320
    A["Admin chooses local latest epoch plus one<br/>fresh roster, random key, local insertion"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/admin/zone_encryption.rs:344
    G["Send grants to remaining eligible members<br/>no shared epoch activation barrier"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/store.rs:380
    W["Stale writer still holds older latest key"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/mod.rs:568
    E["Encrypt new text with locally supplied key"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/zone_config.rs:208
    R["Relay checks ciphertext and zk shape<br/>no current epoch or key binding here"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/mod.rs:465
    D["Old key holder can decrypt obtained ciphertext<br/>read ACL remains a separate boundary"]
    A --> G
    W --> E --> R --> D
    G -. delayed or absent grant .-> W
```

## NF-27.2 Newest-key acknowledgement can hide missing history

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/auto_grant.rs:145
    P["Select recipients without newest-epoch ACK"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/auto_grant.rs:163
    H["For selected recipients only<br/>include missing older epochs"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/relay.rs:640
    Q["Publish queues or sends and returns<br/>it does not wait for relay ACK"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/auto_grant.rs:214
    N["Positive ACK records that key<br/>negative ACK returns without changing report"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/auto_grant.rs:148
    X["Newest accepted, older missing<br/>next sweep skips recipient entirely"]
    P --> H --> Q --> N --> X
    X -. retry uses same newest filter .-> P
```

## NF-27.3 Grant-ledger repair is scoped to one instance

```mermaid
sequenceDiagram
    participant A as Tab A ledger
    participant K as Shared zone KV
    participant B as Tab B ledger
    A->>K: Load empty recipient set
    B->>K: Load empty recipient set
    Note over A,B: grant_ledger.rs:72<br/>loaded sets remain local to each instance
    A->>K: Record Alice and write local snapshot
    Note over A,K: grant_ledger.rs:151<br/>compare snapshot<br/>with own memory
    B->>K: Record Bob and write local snapshot
    Note over K,B: grant_ledger.rs:164<br/>does not merge<br/>another instance's later write
    K-->>B: Reopened ledger contains Bob only
    Note over A,B: grant_ledger.rs:168<br/>one-instance callback repair does not repair this case
```

## NF-27.4 Account identity and zone-key custody have different lifetimes

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/stores/indexed_db.rs:25
    D["Origin database members-forum<br/>key ID is zone and epoch"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/store.rs:187
    H["Root hydration loads all stored zone keys<br/>no account or relay selection"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/store.rs:217
    S["Grant sync captures signer and member<br/>starts once per page"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/auth/mod.rs:676
    L["Logout clears identity and notifications<br/>zone keys and sync state are not reset here"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/auth/mod.rs:699
    R["Navigation reload only away from home<br/>persisted zone keys survive either way"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/stores/indexed_db.rs:334
    C["Different key with same zone/epoch<br/>replaces stored entry"]
    D --> H --> S --> L --> R
    R --> D
    C --> D
```

## NF-27.5 Recovery proposal is superseded and unimplemented on main

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/docs/adr/ADR-2026-zone-key-recovery-master.md:6
    S["ADR-2026 superseded by ADR-2028<br/>no implementation or activation"]
    %% Source: ../nostr-rust-forum/docs/adr/ADR-2026-zone-key-recovery-master.md:35
    M["Proposed 12-word zone master<br/>derive epoch key with domain and revision"]
    %% Source: ../nostr-rust-forum/docs/adr/ADR-2026-zone-key-recovery-master.md:42
    C["Choose first revision not seen in relay history<br/>read-check sequence has no atomic reservation"]
    %% Source: ../nostr-rust-forum/docs/adr/ADR-2026-zone-key-recovery-master.md:47
    K["Recovery must not overwrite conflicting held key<br/>old random keys remain unrecoverable from master"]
    %% Source: ../nostr-rust-forum/docs/adr/ADR-2026-zone-key-recovery-master.md:55
    F["Master holder can derive future epochs<br/>owner replacement and empty epochs need policy"]
    %% Source: ../nostr-rust-forum/README.md:124
    B["NIP-171 implementation belongs to upgrade branch<br/>main status must not imply deployed recovery"]
    S --> M --> C --> K --> F
    S --> B
```
