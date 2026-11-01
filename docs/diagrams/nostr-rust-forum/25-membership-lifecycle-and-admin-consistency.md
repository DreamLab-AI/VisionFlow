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
verified_commit: {nostr-rust-forum: c39f41f104b114d6a3fa29013daaf6fa58b45080}
---

## For developers

[The membership audit](../../estate-review/2026-11-01-forum-membership-lifecycle.md) records 37 selected native cases and seven extracted-SQL scenarios. The pager uses the actual Rust implementation with synthetic pages. SQL cases exercise chosen schedules on SQLite, without running the async Worker handlers or live D1. Consolidation below is proposed, not deployed.

## For the business

Approving, demoting and removing a person should give an operator one intelligible result. Today the interface can derive Pending from an incomplete list, leave a second administrator grant intact, and report deletion despite incomplete cleanup. A unified lifecycle needs receipts that explain those outcomes.

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
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/invites.rs:261
    A["Invite reads cohorts: members"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-core/src/whitelist_sql.rs:73
    B["Concurrent shared SQL grant<br/>commits members + zoneB"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/invites.rs:268
    C["Invite merges its old snapshot<br/>with zoneA"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/invites.rs:271
    D["Invite replaces stored cohorts<br/>members + zoneA, zoneB lost"]
    A --> B --> C --> D
```

This is a selected SQLite schedule. The serial control keeps all three cohorts. Shared SQL itself preserves ordinary existing string grants, but added native cases show duplicate incoming values and a malformed legacy null array need stronger set validation.

## NF-25.3 A relay-only demotion leaves another source of authority

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/admin/mod.rs:439
    A["Members UI sends set-admin false<br/>to relay endpoint"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/whitelist.rs:450
    B["whitelist.is_admin becomes 0"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/auth.rs:194
    C["Relay dynamic admin query<br/>now returns false"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/admin.rs:90
    D["Auth members.is_admin remains 1<br/>auth resolver still returns true"]
    A --> B --> C
    B -.-> D
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/admin.rs:71
    E["Fixture assumes no static admin grant<br/>static authority is a separate source"]
    E -.-> C
    E -.-> D
```

The fixture executes the write and both store queries. Resolver decisions are traced from source. The auth resolver may still grant after a failed relay lookup; an error in one source is not a denial from the union.

## NF-25.4 Invite creation and revocation use different identifiers

```mermaid
flowchart LR
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/invites.rs:361
    A["Create invite<br/>ID: 8 characters<br/>code: 16 characters"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/admin/invites.rs:175
    B["Client sends code<br/>in revoke URL"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/invites.rs:733
    C["Handler looks up<br/>WHERE id = input"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/invites.rs:742
    D["No row: 404<br/>Invite not found"]
    A --> B --> C --> D
```

The SQL fixture misses by code and succeeds by ID. The client reports the error; this is a broken revocation contract, not a demonstrated silent success.

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
    %% Source: ../nostr-rust-forum/docs/adr/ADR-2014-member-lifecycle-one-source-of-truth.md:135
    A["PROPOSED: relay owns members<br/>auth retains real names and handles"]
    %% Source: ../nostr-rust-forum/docs/adr/ADR-2014-member-lifecycle-one-source-of-truth.md:146
    B["PROPOSED: one transition module<br/>state and audit in one batch"]
    %% Source: ../nostr-rust-forum/docs/adr/ADR-2014-member-lifecycle-one-source-of-truth.md:167
    C["PROPOSED: copy and canonicalise<br/>resolve expiry and admin conflicts"]
    %% Source: ../nostr-rust-forum/docs/adr/ADR-2014-member-lifecycle-one-source-of-truth.md:175
    D["PROPOSED: dual-write and reconcile<br/>then retire fallback and old readers"]
    A --> B --> C --> D
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust.rs:565
    E["Design check: past suspension timestamp<br/>does not deny under current predicate"]
    E -.-> C
```

The bounded route/migration inventory finds no proposed relay membership module, table or unified endpoint. Phase 1's shared whitelist SQL and DDL exist. Copying a past expiry to a past suspension while declaring the member active needs a defined denial-preserving rule before migration equivalence can be claimed.
