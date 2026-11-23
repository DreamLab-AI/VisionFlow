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
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/keyset.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/key_owner.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/admin.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/admin/zone_encryption.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/stores/indexed_db.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/auth/mod.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/relay.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/zone_config.rs
verified_commit: 4a73a7a94bfb4d92fea03f04926cccd38c14e9af
---

## For developers

[The audit report](../../estate-review/2026-11-01-forum-zone-key-distribution.md) assesses all sections of ADR-2016 and superseded ADR-2026. Native probes establish planning, ledger and crypto behaviour; full relay/browser acceptance remains untested. Re-verified at af4f0d5: the epoch guards (kit PRs #83, #84, #87) and per-identity key rows (#92) are drawn in NF-27.1, NF-27.4 and NF-27.6; the audit predates them and was not re-run. The NIP-171 key plane is on main since a80c572 but dormant (NF-28); the flag-off path is drawn here.

## For the business

A generated key, an accepted grant and a member able to read are different outcomes. Administrators need a clear activation state, repairable history delivery and tested device recovery before they can rely on removal or recovery promises.

## NF-27.1 Rotation guards the creator, not every writer

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/admin/zone_encryption.rs:216
    A["Create or Rotate asks plan_mint<br/>Wait, Blocked or Mint above every epoch in use, see NF-27.6"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/admin/zone_encryption.rs:489
    P["New key saved pending_delivery before any grant<br/>refused if this device holds a different key"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/admin/zone_encryption.rs:414
    G["Creator adopts it only after the relay accepted<br/>every wrap of the run, its own included"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/keyset.rs:96
    W["Other writers send under their highest held key<br/>no zone-wide activation barrier"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/mod.rs:648
    E["Encrypt new text with locally supplied key"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/zone_config.rs:238
    R["Relay checks ciphertext and zk shape<br/>epoch binding only with ZONE_GROUPS_ENABLED, see NF-28.2"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/mod.rs:545
    D["Old key holder can decrypt obtained ciphertext<br/>read ACL remains a separate boundary"]
    A --> P --> G
    W --> E --> R --> D
    G -. delayed or absent grant .-> W
```

**What it shows.** Since kit PR #83 a rotation whose grants fail no longer leaves new posts readable only on the creator's device. A member who has not yet received the new key still writes under the old one, and the relay does not refuse it.

## NF-27.2 Newest-key acknowledgement can hide missing history

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/auto_grant.rs:146
    P["Select recipients without newest-epoch ACK"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/auto_grant.rs:164
    H["For selected recipients only<br/>include missing older epochs"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/relay.rs:811
    Q["Publish queues or sends and returns<br/>it does not wait for relay ACK"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/auto_grant.rs:285
    N["Positive ACK records that key<br/>negative ACK returns without changing report"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/auto_grant.rs:149
    X["Newest accepted, older missing<br/>next sweep skips recipient entirely"]
    P --> H --> Q --> N --> X
    X -. retry uses same newest filter .-> P
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/admin.rs:613
    ADD["Add User form now also calls after_allocation<br/>kit PR #82, as the cohort editor does"]
    ADD --> P
```

## NF-27.3 Grant-ledger repair is scoped to one instance

```mermaid
sequenceDiagram
    participant A as Tab A ledger
    participant K as Shared zone KV
    participant B as Tab B ledger
    A->>K: Load empty recipient set
    B->>K: Load empty recipient set
    Note over A,B: grant_ledger.rs:86<br/>loaded sets remain local to each instance
    A->>K: Record Alice and write local snapshot
    Note over A,K: grant_ledger.rs:220<br/>compare snapshot<br/>with own memory
    B->>K: Record Bob and write local snapshot
    Note over K,B: grant_ledger.rs:233<br/>does not merge<br/>another instance's later write
    K-->>B: Reopened ledger contains Bob only
    Note over A,B: grant_ledger.rs:237<br/>one-instance callback repair does not repair this case
```

## NF-27.4 Zone keys follow the signed-in identity, legacy rows stay shared

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/stores/indexed_db.rs:32
    D["Origin database members-forum<br/>shared by every account on the profile"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/key_owner.rs:69
    O["New rows carry owner in field and id<br/>owner slash store id"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/key_owner.rs:60
    A["Owner set only while authenticated<br/>a locked session loads no keys"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/store.rs:225
    S["switch_owner drops keys, conflicts, inbox key,<br/>grant subscriptions, plaintext back to placeholders"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/key_owner.rs:172
    V["Load own rows, then legacy rows<br/>for slots without one"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/store.rs:589
    G["Grant sync once per sign-in<br/>only for the identity switched to"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/key_owner.rs:13
    L["OPEN O24 KF-26: legacy rows have no owner<br/>every signed-in identity on the browser sees them"]
    D --> O
    A --> S --> V --> G
    O --> V
    L -.-> V
```

**What it shows.** Logout itself still resets no key state (`../nostr-rust-forum/crates/nostr-bbs-forum-client/src/auth/mod.rs:676`); the store's auth effect does it (`../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/store.rs:1226-1230`). Rows are never deleted, so isolation is per identity in the UI, not access control (`../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/key_owner.rs:21-24`).

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
    B["RESOLVED at nostr-rust-forum 4a73a7a: README now says NIP-171<br/>merged into main at a80c572, dormant behind the flag"]
    S --> M --> C --> K --> F
    S --> B
```

## NF-27.6 Four guards keep one key per zone and epoch

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/keyset.rs:169
    PLAN["plan_mint: Wait until message sync and keys load<br/>keyset.rs:177"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/keyset.rs:220
    BLK["Blocked: keyless identity sees only far-ahead epochs<br/>beyond MAX_EPOCH_SKIP 8, keyset.rs:125"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/keyset.rs:234
    MINT["Mint above own latest, believable post epochs<br/>and every account's rows on this browser"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/admin/zone_encryption.rs:444
    ASK["Below an epoch seen in posts: explicit confirmation"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/key_owner.rs:131
    MD["mint_decision in one transaction: Conflict or Superseded<br/>indexed_db.rs:466, store.rs:968,977"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/mod.rs:177
    CF["conflicts: different secret or pubkey for one store id<br/>is a conflicting mint, never an update"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/store.rs:857
    RX["Received grant: memory check, then put_decision<br/>legacy row counts, store.rs:865,869, key_owner.rs:105"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/store.rs:912
    AD["Storage accepted, memory differs: memory adopts storage"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/keyset.rs:81
    PD["pending_delivery key decrypts, never the send key<br/>cleared by mark_delivered store.rs:382"]
    PLAN --> BLK
    PLAN --> MINT --> ASK --> MD --> PD
    CF --> MD
    CF --> RX --> AD
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/mod.rs:211
    DR["RESOLVED at nostr-rust-forum 4a73a7a: store_id comment now says<br/>flag-off conflicts refuse and keep the held key"]
    DR -.-> CF
```

**What it shows.** Kit PRs #84 and #87: a held key is never replaced by a different one for its slot, and a refused grant is recorded and settled (`../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/store.rs:869-879`). Kit PR #99 corrected the `store_id` comment that predated that guard (`../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/mod.rs:208-212`).
