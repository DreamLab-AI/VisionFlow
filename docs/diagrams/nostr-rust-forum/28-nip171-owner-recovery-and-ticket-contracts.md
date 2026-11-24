---
id: NF-28
title: NIP-171 owner recovery and ticket contracts
area: nostr-rust-forum
governing:
  - ../nostr-rust-forum/docs/adr/ADR-2028-zones-adopt-nip171-key-plane.md
adrs: [ADR-2028]
sources:
  - ../nostr-rust-forum/docs/adr/ADR-2028-zones-adopt-nip171-key-plane.md
  - ../nostr-rust-forum/docs/runbooks/deploy-phase-0-4.md
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/nip171.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/tickets.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/keyset.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/store.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/mod.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/auto_grant.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/zone_groups.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/recipient_keys.rs
  - ../nostr-rust-forum/crates/nostr-bbs-zone-migrate/src/grants.rs
  - ../project/agentbox/management-api/lib/zone-keys.js
verified_commit: {nostr-rust-forum: de31aaa6e6c6b672cc310a2659cf33fd4887602c, agentbox: 59b1aca1cc3541b767872ecbc161c89e65d6ebca}
---

## For developers

These panels were drawn from the local upgrade branch and are re-derived here from **main**, which merged it at a80c572: the code ships dormant, since both sides read `ZONE_GROUPS_ENABLED` as on only for the exact string "true" (`../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/zone_groups.rs:47-48`, `../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/tickets.rs:51`, `../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/mod.rs:107-108`), and the shipped relay template does not set it. [The audit](../../estate-review/2026-11-01-forum-nip171-keyplane.md) (220 bounded native/Node tests) was run on the branch, not re-run on main. Owner flows and scratch migration remain incomplete. NF-27 draws the flag-off path.

Resolved at nostr-rust-forum 4a73a7a: ADR-2028 now records `implementation_status: partial`, its key plane on main behind the flag (`../nostr-rust-forum/docs/adr/ADR-2028-zones-adopt-nip171-key-plane.md:6`).

## For the business

A recovery phrase and passing derivation tests do not yet constitute a recoverable zone. Owners need a verified restore workflow; ordinary members need the intended history without receiving owner authority. Conflicting keys must produce consistent behaviour across browsers and agents before activation.

## NF-28.1 Implemented primitives stop short of owner activation

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/nip171.rs:19
    P["Derivation, records and owner backup helpers<br/>owner consumers explicitly deferred"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/tickets.rs:604
    H["Register held group signer<br/>only a test caller on main"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/tickets.rs:635
    I["Ticket issuer requires held signer<br/>and matching authoritative binding"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/auto_grant.rs:261
    F["Otherwise issue legacy 21453 grant"]
    %% Source: ../nostr-rust-forum/docs/adr/ADR-2028-zones-adopt-nip171-key-plane.md:148
    B["Owner-only numbered binding required<br/>keep ZONE_GROUPS_ENABLED off"]
    P -. missing owner workflow .-> H --> I
    I --> F
    B -. blocks activation .-> I
```

## NF-28.2 A refused multi-claim event can leave a reservation

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/zone_groups.rs:79
    C["Two zk claims<br/>same free epoch, different pubkeys"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:2570
    P["Preflight reads stored rows<br/>both claims see no existing binding"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/zone_groups.rs:38
    A["First insert reserves key A"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/zone_groups.rs:118
    R["Second insert finds existing binding<br/>lookup returns A; key B is refused"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:2586
    S["Event rejected; key A remains reserved<br/>SQLite algorithm fixture confirms"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/recipient_keys.rs:409
    O["Admin recipient registration<br/>has no epoch number or sole-owner check"]
    C --> P --> A --> R --> S
    O -. does not replace this reservation path .-> P
```

## NF-28.3 Browser conflict storage and historical lookup diverge

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/mod.rs:229
    S["Flag-on store id adds group and pubkey<br/>so competing keys for one epoch both persist"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/keyset.rs:101
    W["Selected send epoch is contested<br/>send_key refuses"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/keyset.rs:111
    L["Read lookup chooses one deterministic key<br/>message pubkey is not a lookup argument"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/mod.rs:575
    R["Message encrypted under the other held key<br/>read path fails despite usable key in store"]
    S --> W
    S --> L --> R
```

## NF-28.4 Owner restore and member history require different delivery paths

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/nip171.rs:544
    C["Validate supplied recovery chain<br/>a valid older prefix can pass"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/nip171.rs:654
    B["Owner backup protects recovery master"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/nip171.rs:764
    I["Imported random epoch keys<br/>self-encrypted to group identity"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/nip171.rs:788
    O["Opening imported history requires<br/>group identity secret held by owner"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/tickets.rs:145
    M["Member ticket contains epoch secret<br/>does not convey group identity secret"]
    %% Source: ../nostr-rust-forum/docs/adr/ADR-2028-zones-adopt-nip171-key-plane.md:76
    H["Promised member history still needs<br/>owner-mediated delivery of authorised old keys"]
    B -. owner workflow missing .-> C
    I --> O
    M -. cannot open imported owner backup .-> O
    O -. distribution workflow missing .-> H
```

## NF-28.5 Dormant deployment does not prove reader readiness

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/docs/runbooks/deploy-phase-0-4.md:9
    D["Deploy dormant code<br/>all new feature flags off"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/tickets.rs:526
    T["1014 arrives while browser flag is off<br/>route returns Ignore"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/store.rs:1174
    P["Wrap marked processed<br/>later activation needs explicit replay policy"]
    %% Source: ../nostr-rust-forum/docs/runbooks/deploy-phase-0-4.md:90
    R["Runbook starts 30-day clock<br/>when dormant deployment lands"]
    %% Source: ../nostr-rust-forum/docs/adr/ADR-2028-zones-adopt-nip171-key-plane.md:128
    A["ADR starts clock at live Phase 4<br/>exact activation date still absent"]
    D --> T --> P
    D --> R
    A -. reconcile trigger before rollout .-> R
```

## NF-28.6 Dual readers do not yet share one conflict contract

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/tickets.rs:386
    B["Browser ticket reader<br/>bound group, member and usable epoch"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-zone-migrate/src/grants.rs:178
    M["Migrator ticket reader<br/>explicit zone-to-group bindings"]
    %% Source: ../project/agentbox/management-api/lib/zone-keys.js:301
    A["Agentbox ticket reader<br/>configured zone-to-group bindings"]
    %% Source: ../project/agentbox/management-api/lib/zone-keys.js:506
    C["Competing valid ticket rejected<br/>first key retained without conflict marker"]
    %% Source: ../project/agentbox/management-api/lib/zone-keys.js:625
    W["writePlan still selects first key<br/>new probe encrypts successfully"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/keyset.rs:102
    X["Browser refuses contested send epoch<br/>cross-consumer policy remains inconsistent"]
    A --> C --> W
    B --> X
    M -. reader exists; not rollout proof .-> B
    W -. differs from .-> X
```
