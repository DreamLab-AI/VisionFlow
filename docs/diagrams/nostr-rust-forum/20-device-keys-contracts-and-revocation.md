---
id: NF-20
title: Device keys — producer contracts, authority and revocation
area: nostr-rust-forum
governing:
  - ../nostr-rust-forum/docs/adr/ADR-2004-device-keys-default-off-dual-worker-gate.md
adrs: [ADR-2004]
sources:
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/devices.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/utils/devices.rs
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/feature_gate.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/read_cache.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/session.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/settings.rs
verified_commit: {nostr-rust-forum: c39f41f104b114d6a3fa29013daaf6fa58b45080}
---

## For developers

The registration proof/list-number and re-enrolment findings below have a [scoped source-repair successor, NF-29](29-external-review-repairs.md#nf-292-device-lifecycle-and-dynamic-administration-settle-at-their-owners). This topic retains its declared audit revision; cache/transport findings remain separate.

[ADR-2004 section review](../../estate-review/2026-11-01-forum-device-keys.md) records 76 selected checks, including producer-consumer mismatches, actual SQLite statements and extracted cache methods. Mocked lookup fixtures do not certify deployed Workers or browser behaviour.

## For the business

A lost-phone control needs a working device list and a measurable point at which delegated access stops. Correct boolean parsing alone establishes neither. Device membership also differs from master-key messaging and administrative authority.

## NF-20.1 Independent settings do not guarantee rollout agreement

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/devices.rs:100
    A["Auth worker reads its own setting"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:2570
    B["Relay worker reads its own setting"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-core/src/feature_gate.rs:64
    C["Shared rule: exact string true only"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/utils/devices.rs:111
    D["Browser separately accepts boolean true<br/>or normalised true and 1"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-core/src/feature_gate.rs:127
    E["Auth on, relay off: no owner remap"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-core/src/feature_gate.rs:128
    F["Relay on, auth off: mappings honoured<br/>device API including revoke is unavailable"]
    A --> C
    B --> C
    A -. can differ .-> B
    D -. does not prove worker state .-> A
    C --> E
    C --> F
```

An independently whitelisted raw device key may still pass ordinary write policy when owner remapping is off. The helper name AuthOnlyFailsClosed does not establish universal device denial.

## NF-20.2 Both registration and non-empty listing have contract gaps

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/utils/devices.rs:144
    A["Settings registration sends<br/>device_pubkey and label only"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/utils/devices.rs:186
    B["Recovery-sheet registration sends<br/>the same two fields"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/devices.rs:385
    C["Handler requires device_proof<br/>missing proof returns 400"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/devices.rs:326
    D["Server row uses f64 for<br/>created_at and revoked"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/devices.rs:470
    E["List serialises those rows as JSON"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/utils/devices.rs:264
    F["Client parser expects u64 and u8<br/>non-empty server list fails"]
    A --> C
    B --> C
    D --> E --> F
```

Five native contract tests use the two extracted producer expressions and actual server/client types and parser. Empty lists pass, concealing the numeric mismatch. HTTP dispatch and browser rendering were not executed.

## NF-20.3 Registration can reactivate or transfer a device

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/devices.rs:383
    A["Verify device proof bound to<br/>authenticated owner and short expiry"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/devices.rs:398
    B["Probe principal exclusion<br/>separately from the later upsert"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/devices.rs:416
    C["Existing device row: replace owner<br/>reset revoked to zero"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/devices.rs:511
    D["Revoke matches current owner only"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/devices.rs:293
    E["Concurrent new owner mapping can appear<br/>after principal-exclusion probe"]
    A --> B --> C --> D
    B -. interleaving .-> E
    E -. no atomic recheck .-> C
```

SQL tests reproduce reactivation, transfer and a principal-probe race. Real registration still requires caller authentication and a valid device proof; these tests do not bypass either. Original-owner revocation matches no row after a transfer.

## NF-20.4 AUTH authenticates the raw key, not the owner mapping

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:2057
    A["Verify NIP-42 challenge and signed event"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:2069
    B["Set session authed_pubkey to raw author"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/session.rs:334
    C["Persist raw public key for hibernation"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:2614
    D["Later effective_pubkey applies feature gate"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:2596
    E["Registry query selects only revoked = 0"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:974
    F["No mapping: retain raw public key<br/>no explicit revoked-key rejection here"]
    A --> B --> C
    B --> D --> E --> F
```

Owner mapping is cached between the later lookup and D1. Revoking a mapping does not invalidate the device’s cryptographic signature or make AUTH itself fail. Subsequent policy decides what the remaining raw identity can do.

## NF-20.5 Cached decisions and snapshots extend beyond the database write

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:2216
    A["Owner lookup caches positive and negative results"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/read_cache.rs:73
    B["Reuse entry while age is less than 60 seconds"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/devices.rs:511
    C["Auth worker revokes database row<br/>no relay cache invalidation here"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:2160
    D["Resolved viewer context keeps<br/>owner, cohorts and admin flags"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1883
    E["Request loop reuses the same context<br/>while awaiting per-event checks"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:2221
    F["Expired next lookup refreshes mapping<br/>lookup failure becomes negative cache"]
    A --> B --> D --> E
    C -. does not clear existing context .-> D
    B --> F
```

Native fixtures retain owner mapping at t=159 after a t=100 fill and drop it at t=160; an old admin context still permits a zone read at t=500. Those synthetic times prove code behaviour, not a measured production propagation limit.

## NF-20.6 Delegated authority is different across read and write paths

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1121
    A["Non-gift-wrap event resolves owner<br/>for membership and trust checks"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1226
    B["Write admin status uses raw event author"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:2165
    C["Read context checks admin status<br/>on the resolved owner"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:2114
    D["Owner admin flag permits private-zone reads"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:2138
    E["Sealed-DM filter stays bound to<br/>raw authenticated public key"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1914
    F["Read activity is charged to raw session key"]
    A --> B
    C --> D
    C -. distinct identity rule .-> E
    C -. separate accounting .-> F
```

An admin owner’s device inherits zone-read admin scope in the fixture, while write administration remains raw-key scoped. Master-addressed encrypted messages and owner read-activity credit do not follow automatically from forum membership delegation.
