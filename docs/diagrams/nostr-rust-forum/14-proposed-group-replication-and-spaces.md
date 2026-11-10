---
id: NF-14
title: Proposed group authority, replication and spatial descriptors
area: nostr-rust-forum
governing:
  - ../nostr-rust-forum/docs/BASELINE-architecture.md
adrs: [ADR-2030, ADR-2031, ADR-2032]
sources:
  - ../nostr-rust-forum/docs/BASELINE-architecture.md
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/mesh.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/mod.rs
verified_commit: {nostr-rust-forum: 6064622dbb1a7cc1ba2d072109f9e3e8a96157df}
---

## For developers

All feature nodes below are proposed. The [section audit](../../estate-review/2026-11-01-forum-spaces-and-replication.md) records missing implementation, rollback and parser requirements; voice and its publisher remain explicitly deferred.

## For the business

The intended benefit is one membership decision governing a shared space across clients. Signed freshness, replay protection and confidentiality during outages and rollback must hold together for that benefit to be dependable.

## NF-14.1 Proposed group membership has one source per group

```mermaid
flowchart TD
    COHORT["PROPOSED: zone cohort authors membership<br/>BASELINE-architecture.md:305,306"] --> PROJECT["PROPOSED: route changes and project cohort<br/>BASELINE-architecture.md:306,307"]
    STREAM["PROPOSED: standalone home admission stream<br/>BASELINE-architecture.md:308"] --> SIGN["PROPOSED: home signs 39002<br/>BASELINE-architecture.md:309"]
    PROJECT --> SIGN
    LAG["PROPOSED: projection lag denies access<br/>BASELINE-architecture.md:307"] --> STOP["PROPOSED: stop signing stale projection<br/>BASELINE-architecture.md:326,327"]
    BRIDGE["PROPOSED: bridge reads under scoped grant<br/>separate from membership<br/>BASELINE-architecture.md:310"]
    SIGN --> BRIDGE
```

**What it shows.** Zone-backed and standalone groups have different membership sources; both produce a home-signed projection. **Why it is this way.** A second independently editable group roster would diverge from cohort authority. A read grant does not make its holder a member or writer.

## NF-14.2 Proposed mirrors need more than a valid signature

```mermaid
flowchart TD
    PIN["PROPOSED: operator-pinned home key<br/>BASELINE-architecture.md:313,314"] --> COPY["PROPOSED: store signed state verbatim<br/>BASELINE-architecture.md:317"]
    COPY --> ORDER["PROPOSED: revision order and per-kind watermark<br/>BASELINE-architecture.md:317,318"]
    ORDER --> FRESH["PROPOSED: lease freshness and 10-second rechecks<br/>BASELINE-architecture.md:320,321"]
    FRESH --> DENY["PROPOSED: stale means deny and close sessions<br/>BASELINE-architecture.md:320,322"]
    GAP["CURRENT: outbound connector remains a trait<br/>mesh.rs:143"]
    ALARM["CURRENT: alarm clears idle state<br/>relay_do/mod.rs:371,373,374"]
```

**What it shows.** Authority, order, freshness and ongoing enforcement are separate requirements. **Why it is this way.** The current connector and alarm do not implement this chain. The voice bound is separately disputed by the preceding issuer audit and is not a guarantee of this diagram.

## NF-14.3 Proposed spatial descriptors do not themselves grant access

```mermaid
flowchart TD
    META["PROPOSED: home-signed 39000 descriptor<br/>BASELINE-architecture.md:338"] --> REFS["PROPOSED: closed content-reference set<br/>BASELINE-architecture.md:339,340"]
    REFS --> PLACE["PROPOSED: consumer applies local placement<br/>BASELINE-architecture.md:341"]
    CLAIM["PROPOSED: 30078 claim by member<br/>home key in namespace<br/>BASELINE-architecture.md:343,344"] --> DISPLAY["PROPOSED: advisory display state only<br/>BASELINE-architecture.md:343"]
    VOICE["PROPOSED and DEFERRED: 30312 media publisher<br/>BASELINE-architecture.md:387"] --> URL["PROPOSED: configured SFU URL must match<br/>BASELINE-architecture.md:342"]
```

**What it shows.** Layout, advisory claims and the deferred call publisher serve different purposes. **Why it is this way.** None supplies membership authority. The audit's encoded-dot example requires canonical path validation before the eventual descriptor fetcher is accepted.
