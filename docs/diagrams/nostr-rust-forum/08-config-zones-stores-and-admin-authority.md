---
id: NF-08
title: Config projection, zones, the KV/D1/R2 store map and admin authority
area: nostr-rust-forum
governing:
  - ../nostr-rust-forum/docs/BASELINE-architecture.md
  - ../nostr-rust-forum/docs/IDENTITY-keys-and-trust.md
adrs: [ADR-2004, ADR-2006, ADR-2007]
sources:
  - ../nostr-rust-forum/forum.example.toml
  - ../nostr-rust-forum/SETUP.md
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/wrangler.toml
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/wrangler.toml
  - ../nostr-rust-forum/crates/nostr-bbs-pod-worker/wrangler.toml
  - ../nostr-rust-forum/crates/nostr-bbs-preview-worker/wrangler.toml
  - ../nostr-rust-forum/crates/nostr-bbs-search-worker/wrangler.toml
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/schema.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/admin.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/zone_approval.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/auth.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/lib.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/zone_config.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/whitelist.rs
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/admin_shared.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/migrations/0002_governance.sql
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/migrations/0005_governance_receipts.sql
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/migrations/002_mod_wot_invites_welcome.sql
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/migrations/0003_username_reservations.sql
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/devices.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/username.rs
verified_commit: d48a7a546
---

## NF-08.1 forum.toml — the single source of access truth and its projection

```mermaid
flowchart TB
    TOML["forum.toml (operator-owned)<br/>forum.example.toml:30 deployment<br/>forum.example.toml:38 webauthn<br/>forum.example.toml:46 pod<br/>forum.example.toml:56 relay<br/>forum.example.toml:66 admin"]
    TOML2["forum.toml continued<br/>forum.example.toml:76 branding<br/>forum.example.toml:105 zones (4 example blocks)<br/>forum.example.toml:147 trust :154 invites :161 moderation<br/>forum.example.toml:175 mesh :201 ratelimit :208 features"]
    TOML3["forum.toml continued<br/>forum.example.toml:221 nip05 :229 native_pod :241 provision<br/>forum.example.toml:251 export :261 git :272 governance<br/>forum.example.toml:287 payments :291 payments.token<br/>forum.example.toml:302 calendar :312 custody"]
    ZC["ZONE_CONFIG JSON (zones serialised)"]
    RELAYENV["relay-worker env<br/>nostr-bbs-relay-worker/wrangler.toml:18"]
    CLIENTENV["client window.__ENV__.ZONE_CONFIG"]
    GATE["relay gate - the real access boundary<br/>nostr-bbs-relay-worker/src/zone_config.rs:74"]
    TILES["client renders tiles from the same JSON"]

    TOML --> ZC
    TOML2 --> ZC
    TOML3 --> ZC
    ZC --> RELAYENV --> GATE
    ZC --> CLIENTENV --> TILES

    N1["INVARIANT: one JSON, two enforcement points - the tiles a member sees and the gate the relay<br/>enforces cannot describe two different models. Relay side parses at nostr-bbs-relay-worker/src/zone_config.rs:88"]
    N2["INVARIANT deny-by-default: an absent or malformed ZONE_CONFIG yields an EMPTY config,<br/>so every zone lookup misses and every gate denies nostr-bbs-relay-worker/src/zone_config.rs:88"]
    N3["forum.example.toml ships zeroed placeholder pubkeys that FAIL validation by design forum.example.toml:66-72"]
```

## NF-08.2 Zone visibility and the read/write gate lattice

```mermaid
flowchart LR
    subgraph vis["ZoneVisibility"]
        PUB["Public<br/>nostr-bbs-relay-worker/src/zone_config.rs:28"]
        LOCK["Locked (default)<br/>nostr-bbs-relay-worker/src/zone_config.rs:31"]
        HID["Hidden<br/>nostr-bbs-relay-worker/src/zone_config.rs:33"]
    end
    READ["cohorts_can_read<br/>nostr-bbs-relay-worker/src/zone_config.rs:120"]
    WRITE["cohorts_can_write<br/>nostr-bbs-relay-worker/src/zone_config.rs:136"]
    PUBR["is_public_read - Public AND required_cohorts empty<br/>nostr-bbs-relay-worker/src/zone_config.rs:102"]
    DEFS["defs_visible_to_nonmember - anything not Hidden<br/>nostr-bbs-relay-worker/src/zone_config.rs:111"]
    EFF["effective_write_cohorts = write_cohorts ?? required_cohorts<br/>nostr-bbs-relay-worker/src/zone_config.rs:56"]

    PUB --> PUBR --> READ
    LOCK --> DEFS
    PUB --> DEFS
    HID -->|"omitted entirely"| DEFS
    EFF --> WRITE

    N1["INVARIANT: an EMPTY effective write set denies every non-admin - writes are never anonymous<br/>nostr-bbs-relay-worker/src/zone_config.rs:141"]
    N2["Unknown zone id returns false on BOTH gates - no zone, no access<br/>read nostr-bbs-relay-worker/src/zone_config.rs:122, write nostr-bbs-relay-worker/src/zone_config.rs:138"]
    N3["A public zone with write_cohorts friends is openly readable and inner-circle writable<br/>asserted at nostr-bbs-relay-worker/src/zone_config.rs:170-173"]
```

## NF-08.3 Auto-approval of a new joiner — config-driven cohort grant

```mermaid
sequenceDiagram
    autonumber
    participant U as New user
    participant AW as auth-worker username::claim<br/>nostr-bbs-auth-worker/src/username.rs:210
    participant ZA as zone_approval::new_joiner_cohorts_json<br/>nostr-bbs-auth-worker/src/zone_approval.rs:31
    participant D1 as RELAY_DB whitelist<br/>nostr-bbs-relay-worker/src/whitelist.rs:315

    U->>AW: POST /api/username/claim (NIP-98 authed)
    AW->>ZA: ZONE_CONFIG string
    ZA->>ZA: base cohort vector starts as members zone_approval.rs:32
    ZA->>ZA: for each zone with auto_approve, add required_cohorts de-duplicated zone_approval.rs:36-41
    ZA-->>AW: JSON cohort array
    AW->>D1: whitelist row for the claimed pubkey

    Note over ZA: INVARIANT opt-in per zone, deny-by-default - absent, empty or malformed ZONE_CONFIG yields exactly the members cohort alone zone_approval.rs:45
    Note over ZA: Only auto_approve zones grant - family and business stay admin-gated zone_approval.rs:74-78
```

## NF-08.4 Persistent stores — which worker binds which D1, KV and R2

```mermaid
flowchart TB
    subgraph d1["D1 databases"]
        AUTHDB["nostr-bbs-auth<br/>auth-worker DB nostr-bbs-auth-worker/wrangler.toml:9"]
        RELDB["nostr-bbs-relay<br/>relay-worker DB nostr-bbs-relay-worker/wrangler.toml:58"]
    end
    subgraph kv["KV namespaces"]
        SESS["SESSIONS nostr-bbs-auth-worker/wrangler.toml:23"]
        PODMETA["POD_META nostr-bbs-pod-worker/wrangler.toml:16"]
        RL["RATE_LIMIT nostr-bbs-preview-worker/wrangler.toml:9"]
        SCFG["SEARCH_CONFIG nostr-bbs-search-worker/wrangler.toml:19"]
        KVALIAS["KV - same physical id as SESSIONS nostr-bbs-auth-worker/wrangler.toml:38"]
    end
    subgraph r2["R2 buckets"]
        PODS["nostr-bbs-pods nostr-bbs-pod-worker/wrangler.toml:10"]
        VECS["nostr-bbs-vectors nostr-bbs-search-worker/wrangler.toml:15"]
    end
    subgraph do["Durable Object"]
        RELAYDO["RELAY class NostrRelayDO<br/>nostr-bbs-relay-worker/wrangler.toml:74 sqlite class :78"]
    end

    AUTHW["auth-worker"] --> AUTHDB
    AUTHW -->|"RELAY_DB nostr-bbs-auth-worker/wrangler.toml:18"| RELDB
    AUTHW --> SESS
    AUTHW --> KVALIAS
    AUTHW -->|"POD_META backwards-compat reads nostr-bbs-auth-worker/wrangler.toml:30"| PODMETA
    AUTHW -->|"PODS nostr-bbs-auth-worker/wrangler.toml:42"| PODS
    RELAYW["relay-worker"] --> RELDB
    RELAYW -->|"REPLAY_DB points at nostr-bbs-auth nostr-bbs-relay-worker/wrangler.toml:69"| AUTHDB
    RELAYW --> RELAYDO
    PODW["pod-worker"] --> PODS
    PODW --> PODMETA
    PODW -->|"REPLAY_DB nostr-bbs-pod-worker/wrangler.toml:25"| AUTHDB
    SEARCHW["search-worker"] --> VECS
    SEARCHW --> SCFG
    SEARCHW -->|"REPLAY_DB nostr-bbs-search-worker/wrangler.toml:25"| AUTHDB
    PREVW["preview-worker"] --> RL

    N1["INVARIANT: NIP-98 replay lives in ONE database - every worker binds REPLAY_DB (or DB) to<br/>nostr-bbs-auth so cross-worker replay is detected nostr-bbs-relay-worker/wrangler.toml:63-70"]
    N2["ANOMALY O11 still live: KV and SESSIONS are the SAME physical namespace id<br/>nostr-bbs-auth-worker/wrangler.toml:24 and :39 - key-collision risk across the two logical uses"]
    N3["The auth-worker RELAY_DB database_id is a ZERO placeholder that every deployment must override<br/>nostr-bbs-auth-worker/wrangler.toml:20 - shipped unusable on purpose"]
```

## NF-08.5 Table map — who creates each table and who reads it

```mermaid
flowchart TB
    subgraph authd1["nostr-bbs-auth D1 - bootstrapped by nostr-bbs-auth-worker/src/schema.rs:19"]
        A1["challenges schema.rs:27<br/>webauthn_credentials schema.rs:32<br/>nip1984_reports schema.rs:50<br/>moderation_actions schema.rs:70"]
        A2["mod_reports schema.rs:82<br/>wot_entries schema.rs:103<br/>members schema.rs:116<br/>invitations schema.rs:123"]
        A3["invitation_redemptions schema.rs:136<br/>welcome_messages schema.rs:164<br/>instance_settings schema.rs:182<br/>username_reservations schema.rs:208"]
    end
    subgraph relayd1["nostr-bbs-relay D1 - bootstrapped by nostr-bbs-relay-worker/src/lib.rs:593"]
        R1["channel_zones lib.rs:622<br/>admin_log lib.rs:627<br/>settings lib.rs:638<br/>reports lib.rs:644<br/>hidden_events lib.rs:659"]
        R2["profiles lib.rs:681<br/>agent_registry lib.rs:695<br/>broker_cases lib.rs:705<br/>broker_decisions lib.rs:727"]
        R3["governance_receipts lib.rs:744<br/>broker_roles lib.rs:762<br/>pubkey_aliases lib.rs:775<br/>device_keys - created by the AUTH worker nostr-bbs-auth-worker/src/devices.rs:123"]
    end

    N1["INVARIANT: both bootstraps are idempotent and run on EVERY cold start, so a newly added table exists<br/>before any handler touches it - CREATE TABLE IF NOT EXISTS throughout schema.rs:27 and lib.rs:622"]
    N2["device_keys is the one cross-worker table: the AUTH worker creates and writes it into the RELAY's D1<br/>so the relay DO can read it at NIP-42 AUTH with no cross-worker call - see NF-02.7"]
    N3["Two tables are missing from BOTH bootstraps - see NF-08.6"]
```

Both workers bootstrap idempotently on every cold start — the auth worker at
`nostr-bbs-auth-worker/src/schema.rs:19`, the relay at `nostr-bbs-relay-worker/src/lib.rs:593`.

## NF-08.6 The two tables no code in this repo creates

```mermaid
flowchart TB
    SETUP["SETUP.md operator step 1"]
    EVENTS["events table<br/>SETUP.md:69 - created by wrangler d1 execute"]
    WL["whitelist table<br/>SETUP.md:82 - created by wrangler d1 execute"]
    READERS["Read on the hot path<br/>whitelist SELECT nostr-bbs-relay-worker/src/whitelist.rs:87<br/>whitelist INSERT nostr-bbs-relay-worker/src/whitelist.rs:315<br/>whitelist admin count nostr-bbs-relay-worker/src/whitelist.rs:427"]
    NOCREATE["No CREATE TABLE for events or whitelist exists in<br/>relay migrations 0001-0005 or in ensure_schema<br/>nostr-bbs-relay-worker/src/lib.rs:593"]

    SETUP --> EVENTS
    SETUP --> WL
    WL --> READERS
    NOCREATE -.-> READERS

    N1["DIVERGENCE: the relay's two most load-bearing tables (events, whitelist) are deployment-time<br/>artefacts of a SETUP.md copy-paste, not repo migrations. Every other table is idempotently created<br/>in code. A deployment that skips SETUP.md:65-87 boots and then fails every admission query."]
    N2["Governance tables are created TWICE - by migration 0002_governance.sql:5,17,39,52 and inline by<br/>the relay bootstrap relay lib.rs:695,705,727,762. Both use IF NOT EXISTS, so this is duplication, not drift.<br/>SETUP.md:97 states the same."]
```

## NF-08.7 Admin authority — three sources, one union, and where they disagree

```mermaid
flowchart TB
    CANON["Canonical algorithm and SQL constants<br/>nostr-bbs-core/src/admin_shared.rs:91 ADMIN_PUBKEYS_VAR<br/>nostr-bbs-core/src/admin_shared.rs:109 is_static_admin"]
    S1["1. Static set - ADMIN_PUBKEYS env var"]
    S2["2. RELAY_DB whitelist.is_admin<br/>nostr-bbs-core/src/admin_shared.rs:72"]
    S3["3. auth DB members.is_admin<br/>nostr-bbs-core/src/admin_shared.rs:76"]
    AUTHIS["auth-worker is_admin - union of all three<br/>nostr-bbs-auth-worker/src/admin.rs:57<br/>static nostr-bbs-auth-worker/src/admin.rs:71<br/>RELAY_DB nostr-bbs-auth-worker/src/admin.rs:76<br/>DB nostr-bbs-auth-worker/src/admin.rs:90"]
    RELIS["relay-worker query_is_admin<br/>nostr-bbs-relay-worker/src/auth.rs:179<br/>static nostr-bbs-relay-worker/src/auth.rs:183<br/>members nostr-bbs-relay-worker/src/auth.rs:192<br/>whitelist nostr-bbs-relay-worker/src/auth.rs:201"]

    CANON --> S1 & S2 & S3
    S1 & S2 & S3 --> AUTHIS
    S1 & S2 & S3 --> RELIS

    N1["INVARIANT: the pubkey is lower-cased before every lookup - a mixed-case NIP-98 pubkey would<br/>otherwise miss every store and be silently denied nostr-bbs-auth-worker/src/admin.rs:62"]
    N2["INVARIANT fail-closed: any D1 error or missing row returns false, never ambient authority<br/>nostr-bbs-auth-worker/src/admin.rs:103"]
    N3["DIVERGENCE: the relay reads members from its OWN D1 (nostr-bbs-relay) at<br/>nostr-bbs-relay-worker/src/auth.rs:192, but no migration and no bootstrap creates a members table there<br/>(relay lib.rs:593 creates 13 tables, none of them members). That branch is structurally dead;<br/>effective relay authority is ADMIN_PUBKEYS union whitelist.is_admin only."]
    N4["DOC-DRIFT: ADMIN_PUBKEYS is DECLARED in exactly one wrangler template - the search worker,<br/>nostr-bbs-search-worker/wrangler.toml:33 - yet is read by the auth worker (admin.rs:71) and the relay<br/>(auth.rs:183). The relay template asserts the opposite: there is no ADMIN_PUBKEYS reader in src/<br/>nostr-bbs-relay-worker/wrangler.toml:10-13. SETUP.md:119-124 never lists it either."]
    N5["Consequence of N4: a by-the-book deployment has NO static admin bootstrap on relay or auth,<br/>and the search worker ships a REAL non-placeholder pubkey as its default admin<br/>nostr-bbs-search-worker/wrangler.toml:33 - the only non-generic value in any template."]
```

## NF-08.8 Cross-worker feature gates that must be set in lockstep

```mermaid
flowchart LR
    DKA["auth-worker DEVICE_KEYS_ENABLED=false<br/>nostr-bbs-auth-worker/wrangler.toml:55"]
    DKR["relay-worker DEVICE_KEYS_ENABLED=false<br/>nostr-bbs-relay-worker/wrangler.toml:21"]
    DKC["client window.__ENV__ third setting<br/>SETUP.md:121"]
    GATEFN["auth gate reads its OWN binding<br/>nostr-bbs-auth-worker/src/devices.rs:100"]
    SHARED["shared parse rule only<br/>nostr-bbs-core feature_gate"]
    AM["AUTH_MODE=nip42 default<br/>nostr-bbs-relay-worker/wrangler.toml:31"]
    MESH["MESH_MODE=standalone<br/>nostr-bbs-relay-worker/wrangler.toml:52"]
    ESC["ESCALATION_DEFAULT_TIER=medium<br/>nostr-bbs-relay-worker/wrangler.toml:48"]

    DKA --> GATEFN --> SHARED
    DKR --> SHARED
    DKC --> SHARED

    N1["INVARIANT ADR-2004: each worker reads its own binding and decides ALONE - no cross-worker call.<br/>Only the parse rule is shared, so the two cannot drift on what true means<br/>nostr-bbs-auth-worker/src/devices.rs:100-107"]
    N2["INVARIANT: enables on the EXACT string true; unset, empty or anything else is off<br/>nostr-bbs-auth-worker/src/devices.rs:112, asserted nostr-bbs-auth-worker/src/devices.rs:974"]
    N3["AUTH_MODE: anything other than allowlist resolves to the secure nip42 default<br/>nostr-bbs-relay-worker/wrangler.toml:29-31 - see NF-03"]
    N4["ESCALATION_DEFAULT_TIER is a declared SCAFFOLD - the authoritative risk-tier schema is owned by<br/>agentbox, EXTERNAL: see AB-14 and AB-15; an unrecognised tier folds to medium<br/>nostr-bbs-relay-worker/wrangler.toml:41-48"]
```
