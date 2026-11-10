---
id: DE-05
title: Migration evidence and package boundaries
area: dream-engine
governing:
  - ../dream-engine/docs/adr/ADR-DE-001-standalone-rust-engine.md
  - ../dream-engine/docs/adr/ADR-DE-002-licence-and-provenance.md
adrs: [dream-engine:ADR-DE-001, dream-engine:ADR-DE-002]
sources:
  - ../dream-engine/docs/guide/migrating-from-the-ts-fork.md
  - ../dream-engine/crates/dream-core/tests/ts_parity.rs
  - ../dream-engine/crates/dream-core/Cargo.toml
  - ../dream-engine/crates/dream-core/src/ledger/read.rs
  - ../dream-engine/crates/dream-gate/src/verdict.rs
  - ../dream-engine/Cargo.toml
  - ../dream-engine/NOTICE
  - ../dream-engine/deny.toml
  - ../dream-engine/.github/workflows/ci.yml
  - ../project/agentbox/services/dream-engine/Cargo.toml
  - ../project/agentbox/services/dream-engine/src/lib.rs
  - ../project/agentbox/services/dream-engine/tests/night_e2e.rs
verified_commit: {dream-engine: b40d50724bb6344dc00d57270f7166b138714406, agentbox: 9dfc27b16e82e9786535a55e790710493dd51fdf}
---

## For developers

Migration evidence has several scopes: fixture compatibility, actual old/new validator behaviour, isolated host orchestration and deployed operation. The [dated audit](../../estate-review/2026-11-01-dream-migration-and-provenance.md) checks the first three without claiming the fourth. Package metadata and cached archive checks bind the inspected consumer to four engine releases.

## For the business

Passing a compatibility suite does not prove that an existing installation can retire its old nightly driver. Package consumers also need the promised provenance information and an enforced boundary around components the owner intends never to publish. The current implementation supplies part of this evidence and leaves explicit migration and release-control gaps.

## DE-05.1 Fixture compatibility is narrower than a host migration

```mermaid
flowchart TB
    GUIDE["Guide points host retirement at parity checks<br/>migrating-from-the-ts-fork.md:43"] --> FIXTURE["Rust expectations derived from reading TypeScript<br/>TypeScript not executed by this suite<br/>ts_parity.rs:3"]
    FIXTURE --> INPUTS["Configuration, ledger and witness examples<br/>ts_parity.rs:32<br/>ts_parity.rs:175<br/>ts_parity.rs:335"]
    GUIDE --> HOST["Separate host-night tests<br/>night_e2e.rs:1"]
    HOST --> DOUBLES["Scripted model, recording forge,<br/>local evaluator and SSH/SCP shims<br/>night_e2e.rs:11"]
    DOUBLES --> OFF["Several night services disabled in fixture<br/>night_e2e.rs:108"]
    INPUTS --> TRIM["Rust trimming differs from JavaScript whitespace<br/>ledger/read.rs:215<br/>verdict.rs:394<br/>verdict.rs:444"]
```

**What it shows.** Compatibility fixtures and isolated host tests establish different facts. The audit runs both and separately executes the old validator, exposing U+FEFF and U+0085 differences. No live host migration is certified.

**Why it is this way.** The standalone repository supplies mechanisms and offline checks; the host owns scheduling and delivery. The guide's retirement condition therefore needs evidence beyond the named library suite. That suite is also excluded from the dream-core package (`../dream-engine/crates/dream-core/Cargo.toml:11`).

## DE-05.2 Dependency policy and publication policy have different controls

```mermaid
flowchart TB
    CORE["Five workspace crates, dual licence declaration<br/>../dream-engine/Cargo.toml:3<br/>../dream-engine/Cargo.toml:8"] --> ALLOW["Licence allowlist and named dependency bans<br/>deny.toml:10<br/>deny.toml:26"]
    ALLOW --> CI["CI invokes cargo-deny<br/>ci.yml:77"]
    NOTICE["Repository NOTICE records design lineage<br/>NOTICE:9"] --- CORE
    ADAPTER["Separate AGPL adapter with a package version<br/>services/dream-engine/Cargo.toml:4<br/>services/dream-engine/Cargo.toml:8"] --> CORE
    ADAPTER --> PINS["Exact core 0.4.0 and gate 0.3.0 dependencies<br/>services/dream-engine/Cargo.toml:26"]
    ADAPTER --> ENVELOPE["Envelope module remains in adapter<br/>services/dream-engine/src/lib.rs:54"]
```

**What it shows.** The dependency direction and licence policy exist, while the adapter manifest has no publish=false restriction. Offline package lists include its envelope module and omit root NOTICE from all five permissive packages. These are package-selection observations, not an upload or a legal compliance verdict.

**Why it is this way.** ADR-DE-002 separates reusable engine code from product-specific code and prohibits publishing the adapter. Source placement supports the first boundary; a written prohibition alone does not enforce the second. The audit records the missing guard and provenance evidence without changing release policy.
