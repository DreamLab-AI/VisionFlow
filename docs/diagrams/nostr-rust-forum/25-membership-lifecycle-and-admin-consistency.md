---
id: NF-25
title: Membership lifecycle, cohort grants and admin consistency
area: nostr-rust-forum
governing:
  - ../nostr-rust-forum/docs/adr/ADR-2014-member-lifecycle-one-source-of-truth.md
adrs: [ADR-2014]
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
verified_commit: {nostr-rust-forum: 6064622dbb1a7cc1ba2d072109f9e3e8a96157df}
---

## For developers

The dynamic admin split and invite revoke/cohort findings have a [scoped source-repair successor, NF-29](29-external-review-repairs.md#nf-292-device-lifecycle-and-dynamic-administration-settle-at-their-owners). The dynamic admin split and invite cohort findings are marked resolved at the re-verified stamp 6064622; other membership findings remain separate.

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
    B["DELETE whitelist row<br/>no removed-state tombstone"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/admin/mod.rs:506
    C["Primary pod deprovision<br/>failure logs a warning"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/admin/mod.rs:524
    D["Auth member cleanup<br/>failure logs a warning"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/admin/mod.rs:533
    E["UI reports Deleted<br/>refetch is best effort"]
    A --> B --> C --> D --> E
```

The storage fixture can grant a new row after deletion. It establishes missing tombstone enforcement, without demonstrating unauthorised re-admission or exercising pod deletion.

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
