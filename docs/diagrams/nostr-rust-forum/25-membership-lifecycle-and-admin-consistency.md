---
id: NF-25
title: Membership lifecycle, cohort grants and admin consistency
area: nostr-rust-forum
governing:
  - ../nostr-rust-forum/docs/adr/ADR-2014-member-lifecycle-one-source-of-truth.md
adrs: [ADR-2014, ADR-2016, ADR-2033]
sources:
  - ../nostr-rust-forum/docs/adr/ADR-2014-member-lifecycle-one-source-of-truth.md
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/whitelist_sql.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/admin/membership.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/admin/mod.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/admin/invites.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/admin.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/invites.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/username.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/auth.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/whitelist.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/user_admin.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/admin.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/auto_grant.rs
verified_commit: {nostr-rust-forum: af4f0d56f735ca29dddf9be2549d5c0e9f987891}
---

## For developers

The dynamic admin split and invite revoke/cohort findings have a [scoped source-repair successor, NF-29](29-external-review-repairs.md#nf-292-device-lifecycle-and-dynamic-administration-settle-at-their-owners). The dynamic admin split and invite cohort findings are marked resolved at the re-verified stamp 6064622; other membership findings remain separate. Re-verified at af4f0d5: the user purge now writes NIP-09 event tombstones (NF-25.5) and the Add User form now sends zone keys (NF-25.7); nothing else in these sources changed.

[The membership audit](../../estate-review/2026-11-01-forum-membership-lifecycle.md) records 37 selected native cases and seven extracted-SQL scenarios. The pager uses the actual Rust implementation with synthetic pages. SQL cases exercise chosen schedules on SQLite, without running the async Worker handlers or live D1. Consolidation below is proposed, not deployed.

## For the business

Approving, demoting and removing a person should give an operator one intelligible result. Today the interface can derive Pending from an incomplete list and report deletion despite incomplete cleanup. A unified lifecycle needs receipts that explain those outcomes.

## NF-25.1 Pending depends on complete and consistent list reads

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/username.rs:492
    A["Auth: active handle reservations"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/whitelist.rs:266
    B["Relay: non-expired whitelist<br/>added_at descending, OFFSET pages"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/admin/membership.rs:91
    C["Pager stops on short page,<br/>total reached or 500-page cap"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/admin/membership.rs:109
    D["Pending = reservations minus<br/>collected case-folded pubkeys"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/admin/membership.rs:95
    E["Returns users only<br/>no incomplete-result marker"]
    A --> D
    B --> C --> E --> D
```

The native fixtures reproduce a missing 50,001st member and a live row skipped after deletion shifts the second OFFSET page. An expired row also stays invisible after a cohort grant because the shared statement preserves expiry. None of these cases used a production membership population.

## NF-25.2 Invite replacement can erase an intervening grant

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/invites.rs:238
    A["Invite seeds a whitelist row if absent"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-core/src/whitelist_sql.rs:73
    B["Concurrent shared SQL grant<br/>commits members + zoneB"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/invites.rs:263
    C["Invite merges zoneA inside one UPDATE<br/>over the cohorts stored at that moment"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/invites.rs:254
    D["RESOLVED at nostr-rust-forum 6064622: members, zoneA and zoneB<br/>all survive, no replacement from a stale snapshot"]
    A --> B --> C --> D
```

At the earlier stamp this SQLite schedule lost zoneB. At 6064622 the invite merges inside one UPDATE and a native test asserts that members, zone-a and zone-b all survive (`../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/invites.rs:267-278`). The shared grant SQL also tolerates a malformed legacy cohort value (`../nostr-rust-forum/crates/nostr-bbs-core/src/whitelist_sql.rs:66-72`).

## NF-25.3 A relay-only demotion leaves another source of authority

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/admin/mod.rs:439
    A["Members UI sends set-admin false<br/>to relay endpoint"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/whitelist.rs:450
    B["whitelist.is_admin becomes 0"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/auth.rs:194
    C["Relay dynamic admin query<br/>now returns false"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/admin.rs:72
    D["RESOLVED at nostr-rust-forum 6064622: auth members.is_admin may remain 1<br/>but the auth resolver no longer reads it"]
    A --> B --> C
    B -.-> D
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/admin.rs:67
    E["Fixture assumes no static admin grant<br/>static authority is a separate source"]
    E -.-> C
    E -.-> D
```

At the earlier stamp the auth resolver also consulted members. At 6064622 it reads only the static set and the relay whitelist (`../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/admin.rs:62-83`), so a relay demotion is authoritative for both resolvers and a failed relay lookup returns false rather than falling through (`../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/admin.rs:85`). The fixture executes the write and both store queries.

## NF-25.4 Invite creation and revocation use different identifiers

```mermaid
flowchart LR
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/invites.rs:362
    A["Create invite<br/>ID: 8 characters<br/>code: 16 characters"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/admin/invites.rs:175
    B["Client sends code<br/>in revoke URL"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/invites.rs:733
    C["Handler looks up<br/>WHERE id = input OR code = input"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/invites.rs:743
    D["No row: 404<br/>Invite not found"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/invites.rs:761
    E["DEBT: the UPDATE matches WHERE id = the URL value<br/>so a code changes no row yet the reply is ok"]
    A --> B --> C --> D
    C --> E
```

At the earlier stamp the lookup matched id only, so a revoke by code returned 404. At 6064622 the lookup also matches code, but the UPDATE still matches id only (`../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/invites.rs:761-762`), so a revoke by code can report ok without changing a row; this is read from source, not executed.

## NF-25.5 Current deletion has separate cleanup outcomes

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/admin/mod.rs:488
    A["Relay deletion must succeed"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/user_admin.rs:199
    B["DELETE whitelist row<br/>no removed-state membership record"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/user_admin.rs:224-231
    B2["delete_events only: NIP-09 tombstones for the author's rows,<br/>then DELETE events and profiles, fail-open"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/user_admin.rs:268-270
    B3["reply and audit row carry tombstoned only when a purge ran"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/admin/mod.rs:487
    B4["TENSION: the client checks only for an error and never reads tombstoned"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/admin/mod.rs:506
    C["Primary pod deprovision<br/>failure logs a warning"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/admin/mod.rs:524
    D["Auth member cleanup<br/>failure logs a warning"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/admin/mod.rs:533
    E["UI reports Deleted<br/>refetch is best effort"]
    A --> B --> C --> D --> E
    B -.-> B2 --> B3 -.-> B4
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/user_admin.rs:233
    OPEN1["OPEN, kit O17 / KF-16: the purge tombstones and deletes by pubkey only,<br/>so device-key and gift-wrap rows are neither tombstoned nor deleted"]
    B2 -.-> OPEN1
```

The storage fixture can grant a new row after deletion. It establishes missing removed-state enforcement, without demonstrating unauthorised re-admission or exercising pod deletion. The NIP-09 tombstones added at da08b53 and db6ddfc are event-level: they stop a re-admitted member re-publishing purged events (`../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/user_admin.rs:221-223`), not re-admission itself, and no route removes one short of `reset-db` (kit O15 / KF-14, NF-08.10).

## NF-25.6 Consolidation remains a proposed migration

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/docs/adr/ADR-2014-member-lifecycle-one-source-of-truth.md:148
    A["PROPOSED: relay owns members<br/>auth retains real names and handles"]
    %% Source: ../nostr-rust-forum/docs/adr/ADR-2014-member-lifecycle-one-source-of-truth.md:159
    B["PROPOSED: one transition module<br/>state and audit in one batch"]
    %% Source: ../nostr-rust-forum/docs/adr/ADR-2014-member-lifecycle-one-source-of-truth.md:180
    C["PROPOSED: copy and canonicalise<br/>resolve expiry and admin conflicts"]
    %% Source: ../nostr-rust-forum/docs/adr/ADR-2014-member-lifecycle-one-source-of-truth.md:188
    D["PROPOSED: dual-write and reconcile<br/>then retire fallback and old readers"]
    A --> B --> C --> D
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust.rs:565
    E["Design check: past suspension timestamp<br/>does not deny under current predicate"]
    E -.-> C
```

The bounded route/migration inventory finds no proposed relay membership module, table or unified endpoint. Phase 1's shared whitelist SQL and DDL exist. Copying a past expiry to a past suspension while declaring the member active needs a defined denial-preserving rule before migration equivalence can be claimed.

## NF-25.7 Both admin allocation paths send zone keys

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/admin.rs:568
    ADD["Add User form captures AutoGrant, since 071e78c"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/admin.rs:608-613
    ADDOK["after add_to_whitelist_signer succeeds:<br/>after_allocation for the new pubkey"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/admin.rs:643
    EDIT["Members cohort editor captures AutoGrant"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/admin.rs:654-656
    EDITOK["after update_cohorts_signer succeeds and a cohort was added:<br/>after_allocation"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/auto_grant.rs:323-328
    NONE["capture is None with no encrypted zone<br/>or no zone-key store: nothing sent"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/auto_grant.rs:456
    ALLOC["after_allocation narrows the grant run to that one pubkey"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/auto_grant.rs:428-433
    RUN["explicit admin action: never-granted targets are approved,<br/>not held for the first-grant confirmation"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/auto_grant.rs:362
    ROSTER["grants are planned from a fresh whitelist roster read"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/auto_grant.rs:466-470
    FAIL["a failed grant is a toast: retry from Admin, Encryption"]
    ADD --> ADDOK --> ALLOC
    EDIT --> EDITOK --> ALLOC
    ADD -.-> NONE
    ALLOC --> RUN --> ROSTER --> FAIL
```

Before 071e78c only the cohort editor sent keys, so a member added through the form with an encrypted-zone cohort waited for an admin's manual sweep. Both paths now share one client-side grant run; the whitelist write and the key grant remain separate steps, and a grant failure does not undo the membership.
