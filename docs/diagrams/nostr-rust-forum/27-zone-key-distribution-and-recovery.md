---
id: NF-27
title: Zone key distribution, rotation and recovery
area: nostr-rust-forum
governing:
  - ../nostr-rust-forum/docs/adr/ADR-2016-end-to-end-encrypted-zones.md
  - ../nostr-rust-forum/docs/adr/ADR-2026-zone-key-recovery-master.md
adrs: [ADR-2016, ADR-2026, ADR-2028, ADR-2034]
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
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/grant_paging.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/grant_backlog.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/wrap_queue.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/tickets.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/stores/deletion_sync.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/stores/channels.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/app.rs
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/event.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs
  - ../nostr-rust-forum/docs/security/known-findings.md
  - ../nostr-rust-forum/docs/adr/ADR-2034-absence-is-reported-only-after-an-authenticated-answer.md
  - ../nostr-rust-forum/docs/adr/ADR-2028-zones-adopt-nip171-key-plane.md
verified_commit: de31aaa6e6c6b672cc310a2659cf33fd4887602c
---

## For developers

[The audit report](../../estate-review/2026-11-01-forum-zone-key-distribution.md) assesses all sections of ADR-2016 and superseded ADR-2026. Native probes establish planning, ledger and crypto behaviour; full relay/browser acceptance remains untested. Re-verified at af4f0d5: the epoch guards (kit PRs #83, #84, #87) and per-identity key rows (#92) are drawn in NF-27.1, NF-27.4 and NF-27.6; the audit predates them and was not re-run. The NIP-171 key plane is on main since a80c572 but dormant (NF-28); the flag-off path is drawn here. Re-verified at de31aaa: kit PR #95 pages the grant backlog past the relay's 1,000-row cap and records per-identity, per-scope backfill markers (NF-27.7 to NF-27.10, superseding NF-27.4); its open limits are KF-31 to KF-35 in the known-findings table (`../nostr-rust-forum/docs/security/known-findings.md:108-112`), collected in NF-32.

## For the business

A generated key, an accepted grant and a member able to read are different outcomes. Administrators need a clear activation state, repairable history delivery and tested device recovery before they can rely on removal or recovery promises. Since kit PR #95 a member with many direct messages no longer silently misses their oldest zone keys on a new device: the forum keeps fetching older pages until it has seen everything, or says the keys are late. On 2026-11-24, after the website moved to this kit commit, the operator opened a Family topic link on a fresh device three times: "Fetching your keys" appeared at 4.5 to 5.4 seconds and the topic with decrypted posts at 11.7 to 13.0 seconds, never "Topic Not Found" (two days earlier the same test took 9.2 and 28.4 seconds). That is a live observation, not a code citation.

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
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/mod.rs:654
    E["Encrypt new text with locally supplied key"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/zone_config.rs:238
    R["Relay checks ciphertext and zk shape<br/>epoch binding only with ZONE_GROUPS_ENABLED, see NF-28.2"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/mod.rs:551
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
> **Superseded** by NF-27.7 at de31aaa (frozen at 4a73a7a).

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
    MD["mint_decision in one transaction: Conflict or Superseded<br/>indexed_db.rs:466, store.rs:1429,1438"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/mod.rs:183
    CF["conflicts: different secret or pubkey for one store id<br/>is a conflicting mint, never an update"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/store.rs:1318
    RX["Received grant: memory check, then put_decision<br/>legacy row counts, store.rs:1326,1330, key_owner.rs:105"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/store.rs:1373
    AD["Storage accepted, memory differs: memory adopts storage"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/keyset.rs:81
    PD["pending_delivery key decrypts, never the send key<br/>cleared by mark_delivered store.rs:472"]
    PLAN --> BLK
    PLAN --> MINT --> ASK --> MD --> PD
    CF --> MD
    CF --> RX --> AD
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/mod.rs:217
    DR["RESOLVED at nostr-rust-forum 4a73a7a: store_id comment now says<br/>flag-off conflicts refuse and keep the held key"]
    DR -.-> CF
```

**What it shows.** Kit PRs #84 and #87: a held key is never replaced by a different one for its slot, and a refused grant is recorded and settled (`../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/store.rs:1330-1340`). Kit PR #99 corrected the `store_id` comment that predated that guard (`../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/mod.rs:214-218`).

## NF-27.7 Zone keys follow the signed-in identity, and so does the paged grant sync

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/stores/indexed_db.rs:32
    D["Origin database members-forum<br/>shared by every account on the profile"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/key_owner.rs:69
    O["New rows carry owner in field and id<br/>owner slash store id"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/key_owner.rs:60
    A["Owner set only while authenticated<br/>a locked session loads no keys"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/store.rs:307
    S["switch_owner drops keys, conflicts, inbox key, grant subscriptions,<br/>backfill markers, deferred pages and the wrap queue generation"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/key_owner.rs:172
    V["Load own rows, then legacy rows<br/>for slots without one"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/app.rs:818
    K["Signer kind passed in: a NIP-07 extension can refuse a decrypt<br/>start_grant_sync store.rs:691, SignerKind store.rs:709"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/store.rs:721
    G["Grant sync once per sign-in, identity scope always<br/>inbox scope too with RECIPIENT_KEYS_ENABLED store.rs:763"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/key_owner.rs:13
    L["OPEN O24 KF-26: legacy rows have no owner<br/>every signed-in identity on the browser sees them"]
    D --> O
    A --> S --> V --> K --> G
    O --> V
    L -.-> V
```

**What it shows.** The auth effect still drives the switch (`../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/store.rs:1747-1752`); each scope pages as NF-27.8 draws. Rows are never deleted, so isolation is per identity in the UI, not access control (`../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/key_owner.rs:21-24`).

## NF-27.8 One grant scope pages past the relay's row cap

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/store.rs:782
    RS["run_scope: page 0 is the long-lived subscription, no until"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/store.rs:1637
    F["wrap_filter: kind 1059 to p, limit 1000, until when paging<br/>channels.rs:963"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/store.rs:839
    AP["after_page at each EOSE: ScopePager.eose decides<br/>grant_paging.rs:394, page subscription closed store.rs:852-857"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/grant_paging.rs:281
    E{"page empty?"}
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/grant_paging.rs:288
    E0["page 0 empty: Done, unverified<br/>no wraps, or a failed query read as empty"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/grant_paging.rs:291
    ER["later page empty: same until again, then until minus 1<br/>grant_paging.rs:295, then Done unverified grant_paging.rs:299"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/grant_paging.rs:305
    R["deletion sync's rule: next_page_within<br/>deletion_sync.rs:431"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/grant_paging.rs:108
    MAX["GRANT_MAX_PAGES 50 after page 0, then GiveUp"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/store.rs:863
    N["Next page opened only while the wrap queue has room<br/>else deferred wrap_queue.rs:165"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/store.rs:875
    DN["Done: scope_delivered, verified unless a boundary was skipped<br/>grant_paging.rs:415, then try_write_markers store.rs:885"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/store.rs:887
    GU["GiveUp: paging_gave_up, the backlog reads stalled<br/>grant_backlog.rs:312"]
    RS --> F --> AP --> E
    E -->|yes| E0 --> DN
    E -->|yes| ER --> DN
    E -->|no| R
    R -->|older rows| N --> AP
    R -->|nothing older| DN
    R --> MAX --> GU
```

**What it shows.** Kit PR #95: a scope holds at most two subscriptions, page 0 and the current page (`../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/store.rs:774-781`). **Debt (KF-31):** the relay answers a failed query as empty plus EOSE, so an empty page 0 is only unverified, never an error (`../nostr-rust-forum/docs/security/known-findings.md:108`). **Debt (KF-32, KF-33):** past 51,000 wraps, or a full page of one timestamp, the scope gives up every session (`../nostr-rust-forum/docs/security/known-findings.md:109-110`).

## NF-27.9 The backfill marker and the eight-day warm stop

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/grant_paging.rs:209
    ID["marker id per owner, scope and addressed key<br/>grant_backfill_complete v1 grant_paging.rs:214"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/grant_paging.rs:122
    M["Marker: done_at on this device's clock, optional floor"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/grant_backlog.rs:266
    W["Writable only when nothing is in flight or retrying<br/>take_due_markers grant_backlog.rs:288"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/store.rs:905
    P["try_write_markers: flush the processed set, then put_zone_kv<br/>store.rs:940 store.rs:943"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-core/src/event.rs:28
    DR["MAX_CREATED_AT_DRIFT_SECS seven days, shared with the relay's<br/>ingress bound nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:44"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/grant_paging.rs:99
    MG["WARM_STOP_MARGIN_SECS: drift plus one day of clock skew<br/>grant_paging.rs:91 grant_paging.rs:95"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/grant_paging.rs:278
    WS["Warm stop: a full page, nothing unprocessed, newest wrap<br/>below done_at minus the margin and below the floor grant_paging.rs:143-146"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/grant_paging.rs:112
    SW["No warm stop while GRANT_SWEEP_ENABLED is on"]
    ID --> M
    W --> P --> M
    DR --> MG --> WS
    M --> WS
    SW -.-> WS
```

**What it shows.** A wrap that reached the relay after the last backfill is dated at most seven days before it landed, so it sorts above any page eight days below the marker (`../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/grant_paging.rs:48-71`). **Invariant:** the client's margin and the relay's ingress bound read one constant (`../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/grant_paging.rs:91`, `../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:44`).

## NF-27.10 How each wrap ends, and when the backlog may settle

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/wrap_queue.rs:8
    Q["Wrap queue: newest-queued first, at most four at once<br/>one until a NIP-07 signer succeeds wrap_queue.rs:49"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/grant_backlog.rs:144
    UF["unwrap_failure_outcome: invalid wrap, or local key fails: Unreadable<br/>NIP-07 first failure: Refused grant_backlog.rs:150"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/store.rs:1113
    PK["Refused wraps parked, queued again after another decrypt succeeds<br/>wrap_queue.rs:189"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/grant_backlog.rs:151
    UB["Fails again: UnreadableBySigner, settles but sets the marker floor<br/>grant_backlog.rs:242-243"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/store.rs:1209
    GR["Grant: admin check, validate, persist, then Settled, else Retry<br/>store.rs:1213"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/store.rs:1277
    TK["Ticket verdict: Retry, Unbound as Unreadable store.rs:1280,<br/>SignerFailed classified store.rs:1283, Refuse marked processed store.rs:1286"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/grant_backlog.rs:105
    RT["Retry and Refused are retryable: they hold the backlog<br/>and withhold the marker grant_backlog.rs:237-238"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/grant_backlog.rs:324
    TH["Through: no give-up, identity and inbox scopes delivered,<br/>nothing in flight, nothing retrying grant_backlog.rs:329-333"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/grant_backlog.rs:339
    PH["phase: Stalled after a give-up grant_backlog.rs:343, or 20 s idle<br/>or 120 s from the later of start and last delivery grant_backlog.rs:347"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/tickets.rs:580
    K34["OPEN KF-34: an Unbound ticket whose binding appears later<br/>is lost on a warm device once the warm stop passes it"]
    %% Source: ../nostr-rust-forum/docs/security/known-findings.md:112
    K35["OPEN KF-35: a junk wrap to a NIP-07 member pins the floor,<br/>so every session walks everything newer"]
    Q --> UF
    UF --> PK --> UB
    Q --> GR --> RT
    Q --> TK --> RT
    RT --> TH --> PH
    TK -.-> K34
    UB -.-> K35
```

**What it shows.** A refused NIP-07 prompt is never taken as an unreadable wrap on its first attempt, so the backlog cannot settle over a grant that was only refused (`../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/grant_backlog.rs:131-143`). **Open (KF-34):** a prerequisite for enabling `ZONE_GROUPS_ENABLED` (`../nostr-rust-forum/docs/security/known-findings.md:111`, `../nostr-rust-forum/docs/adr/ADR-2028-zones-adopt-nip171-key-plane.md:153-160`).
