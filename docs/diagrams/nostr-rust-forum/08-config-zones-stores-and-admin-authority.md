---
id: NF-08
title: Config projection, zones, the KV/D1/R2 store map and admin authority
area: nostr-rust-forum
governing:
  - ../nostr-rust-forum/docs/BASELINE-architecture.md
  - ../nostr-rust-forum/docs/IDENTITY-keys-and-trust.md
adrs: [ADR-2004, ADR-2006, ADR-2007, ADR-2011, ADR-2012, ADR-2014, ADR-2022, ADR-2023, ADR-2029]
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
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/migrations/0008_whitelist.sql
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/migrations/0009_events.sql
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/tests/events_schema.rs
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/whitelist_sql.rs
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/admin_shared.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/migrations/0002_governance.sql
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/migrations/0005_governance_receipts.sql
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/migrations/002_mod_wot_invites_welcome.sql
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/migrations/0003_username_reservations.sql
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/devices.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/username.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/admins.rs
  - ../nostr-rust-forum/docs/adr/ADR-2012-d1-ledger-becomes-a-chain-view.md
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs
verified_commit: 1b27f79d08537b61b3d5a5d02cac9ee5999b79ed
---

## NF-08.1 forum.toml — the single source of access truth and its projection

```mermaid
flowchart TB
    TOML["forum.toml (operator-owned)<br/>forum.example.toml:49 deployment (profile line 52, ADR-2029 proposed)<br/>forum.example.toml:58 webauthn<br/>forum.example.toml:66 pod<br/>forum.example.toml:76 relay<br/>forum.example.toml:86 admin"]
    TOML2["forum.toml continued<br/>forum.example.toml:96 branding<br/>forum.example.toml:125 zones (4 example blocks)<br/>forum.example.toml:189 trust :196 invites :203 moderation<br/>forum.example.toml:181 [[ingress]] commented example, ADR-2022<br/>forum.example.toml:217 mesh :243 ratelimit :250 features"]
    TOML3["forum.toml continued<br/>forum.example.toml:264 nip05 :272 native_pod :284 provision<br/>forum.example.toml:294 export :304 git :315 governance<br/>forum.example.toml:330 payments :334 payments.token<br/>forum.example.toml:345 calendar :354 poker :377 custody"]
    ZC["ZONE_CONFIG JSON (zones serialised)"]
    RELAYENV["relay-worker env<br/>nostr-bbs-relay-worker/wrangler.toml:26"]
    CLIENTENV["client window.__ENV__.ZONE_CONFIG"]
    GATE["relay gate - the real access boundary<br/>nostr-bbs-relay-worker/src/zone_config.rs:84"]
    TILES["client renders tiles from the same JSON"]

    TOML --> ZC
    TOML2 --> ZC
    TOML3 --> ZC
    ZC --> RELAYENV --> GATE
    ZC --> CLIENTENV --> TILES

    N1["INVARIANT: one JSON, two enforcement points - the tiles a member sees and the gate the relay<br/>enforces cannot describe two different models. Relay side parses at nostr-bbs-relay-worker/src/zone_config.rs:108"]
    N2["INVARIANT deny-by-default: an absent or malformed ZONE_CONFIG yields an EMPTY config,<br/>so every zone lookup misses and every gate denies nostr-bbs-relay-worker/src/zone_config.rs:108"]
    N3["forum.example.toml ships zeroed placeholder pubkeys that FAIL validation by design forum.example.toml:86-91"]
    N4["OPEN: [deployment] profile is proposed (ADR-2029), only forum deploys until it is accepted and implemented<br/>forum.example.toml:47, forum.example.toml:52"]
    N5["[[ingress]] recipients are admitted without a whitelist row, role primary, copy or agent (ADR-2022, live)<br/>forum.example.toml:165-185"]
```

**Open:** `profile` in `[deployment]` selects a `"forum"` or `"room"` wire shape for the life of the deployment, but ADR-2029 is proposed with implementation none and the example file says only `"forum"` deploys until it is accepted (`../nostr-rust-forum/forum.example.toml:47`, `../nostr-rust-forum/docs/adr/ADR-2029-room-profile-deployment-type.md:5`); its dependencies ADR-2027 (accepted, partial, inactive) and ADR-2028 (proposed) are likewise not live.

## NF-08.2 Zone visibility and the read/write gate lattice

```mermaid
flowchart LR
    subgraph vis["ZoneVisibility"]
        PUB["Public<br/>nostr-bbs-relay-worker/src/zone_config.rs:28"]
        LOCK["Locked (default)<br/>nostr-bbs-relay-worker/src/zone_config.rs:31"]
        HID["Hidden<br/>nostr-bbs-relay-worker/src/zone_config.rs:33"]
    end
    READ["cohorts_can_read<br/>nostr-bbs-relay-worker/src/zone_config.rs:161"]
    WRITE["cohorts_can_write<br/>nostr-bbs-relay-worker/src/zone_config.rs:177"]
    PUBR["is_public_read - Public AND required_cohorts empty<br/>nostr-bbs-relay-worker/src/zone_config.rs:143"]
    DEFS["defs_visible_to_nonmember - anything not Hidden<br/>nostr-bbs-relay-worker/src/zone_config.rs:152"]
    EFF["effective_write_cohorts = write_cohorts ?? required_cohorts<br/>nostr-bbs-relay-worker/src/zone_config.rs:63"]
    ZENC["Zone.encrypted flag<br/>nostr-bbs-relay-worker/src/zone_config.rs:58"]
    MGATE["ENCRYPTION_ENABLED master gate<br/>field nostr-bbs-relay-worker/src/zone_config.rs:77 loaded :86"]
    ISENC["is_encrypted = gate AND zone.encrypted AND NOT is_public_read<br/>nostr-bbs-relay-worker/src/zone_config.rs:133"]

    PUB --> PUBR --> READ
    LOCK --> DEFS
    PUB --> DEFS
    HID -->|"omitted entirely"| DEFS
    EFF --> WRITE
    ZENC --> ISENC
    MGATE --> ISENC
    PUBR -->|"public zones never enforced, even if flagged"| ISENC

    N1["INVARIANT: an EMPTY effective write set denies every non-admin - writes are never anonymous<br/>nostr-bbs-relay-worker/src/zone_config.rs:182"]
    N2["Unknown zone id returns false on BOTH gates - no zone, no access<br/>read nostr-bbs-relay-worker/src/zone_config.rs:163, write nostr-bbs-relay-worker/src/zone_config.rs:179"]
    N3["A public zone with write_cohorts friends is openly readable and inner-circle writable<br/>asserted at nostr-bbs-relay-worker/src/zone_config.rs:342-345"]
    N4["DOC-DRIFT: ENCRYPTION_ENABLED is read from env at zone_config.rs:86 but declared in NO<br/>wrangler.toml [vars] block in any worker - unlike DEVICE_KEYS_ENABLED (see NF-08.8) it has no<br/>documented deployment default, so a deploy that forgets it silently leaves encryption OFF"]
    N5["see NF-08.9 for the shape check that enforces an encrypted zone once is_encrypted is true"]
```

## NF-08.3 Auto-approval of a new joiner — config-driven cohort grant

```mermaid
sequenceDiagram
    autonumber
    participant U as New user
    participant AW as auth-worker username::claim<br/>nostr-bbs-auth-worker/src/username.rs:210
    participant ZA as zone_approval::new_joiner_cohorts_json<br/>nostr-bbs-auth-worker/src/zone_approval.rs:31
    participant D1 as RELAY_DB whitelist INSERT<br/>nostr-bbs-auth-worker/src/username.rs:307

    U->>AW: POST /api/username/claim (NIP-98 authed)
    AW->>ZA: ZONE_CONFIG string
    ZA->>ZA: base cohort vector starts as members zone_approval.rs:32
    ZA->>ZA: for each zone with auto_approve, add required_cohorts de-duplicated zone_approval.rs:36-41
    ZA-->>AW: JSON cohort array
    AW->>D1: whitelist row for the claimed pubkey, ON CONFLICT DO NOTHING username.rs:308

    Note over ZA: INVARIANT opt-in per zone, deny-by-default - absent, empty or malformed ZONE_CONFIG yields exactly the members cohort alone zone_approval.rs:45
    Note over ZA: Only auto_approve zones grant - family and business stay admin-gated zone_approval.rs:74-78
```

## NF-08.4 Persistent stores — which worker binds which D1, KV and R2

```mermaid
flowchart TB
    subgraph d1["D1 databases"]
        AUTHDB["nostr-bbs-auth<br/>auth-worker DB nostr-bbs-auth-worker/wrangler.toml:9"]
        RELDB["nostr-bbs-relay<br/>relay-worker DB nostr-bbs-relay-worker/wrangler.toml:99"]
    end
    subgraph kv["KV namespaces"]
        SESS["SESSIONS nostr-bbs-auth-worker/wrangler.toml:27"]
        PODMETA["POD_META nostr-bbs-pod-worker/wrangler.toml:16"]
        RL["RATE_LIMIT nostr-bbs-preview-worker/wrangler.toml:9"]
        SCFG["SEARCH_CONFIG nostr-bbs-search-worker/wrangler.toml:19"]
        KVALIAS["KV - same physical id as SESSIONS nostr-bbs-auth-worker/wrangler.toml:43"]
    end
    subgraph r2["R2 buckets"]
        PODS["nostr-bbs-pods nostr-bbs-pod-worker/wrangler.toml:10"]
        VECS["nostr-bbs-vectors nostr-bbs-search-worker/wrangler.toml:15"]
    end
    subgraph do["Durable Object"]
        RELAYDO["RELAY class NostrRelayDO<br/>nostr-bbs-relay-worker/wrangler.toml:115 sqlite class :120"]
    end

    AUTHW["auth-worker"] --> AUTHDB
    AUTHW -->|"RELAY_DB nostr-bbs-auth-worker/wrangler.toml:18"| RELDB
    AUTHW --> SESS
    AUTHW --> KVALIAS
    AUTHW -->|"POD_META backwards-compat reads nostr-bbs-auth-worker/wrangler.toml:34"| PODMETA
    AUTHW -->|"PODS nostr-bbs-auth-worker/wrangler.toml:47"| PODS
    RELAYW["relay-worker"] --> RELDB
    RELAYW -->|"REPLAY_DB points at nostr-bbs-auth nostr-bbs-relay-worker/wrangler.toml:110"| AUTHDB
    RELAYW --> RELAYDO
    PODW["pod-worker"] --> PODS
    PODW --> PODMETA
    PODW -->|"REPLAY_DB nostr-bbs-pod-worker/wrangler.toml:25"| AUTHDB
    SEARCHW["search-worker"] --> VECS
    SEARCHW --> SCFG
    SEARCHW -->|"REPLAY_DB nostr-bbs-search-worker/wrangler.toml:25"| AUTHDB
    PREVW["preview-worker"] --> RL

    N1["INVARIANT: NIP-98 replay lives in ONE database - every worker binds REPLAY_DB (or DB) to<br/>nostr-bbs-auth so cross-worker replay is detected nostr-bbs-relay-worker/wrangler.toml:104-111"]
    N2["ANOMALY O11 now DOCUMENTED, not removed: KV and SESSIONS are still the SAME physical namespace id<br/>nostr-bbs-auth-worker/wrangler.toml:28 and :44; the template says the key prefixes are disjoint<br/>nostr-bbs-auth-worker/wrangler.toml:22-25 and a unit test pins the admin cache key outside the rl: prefix<br/>nostr-bbs-auth-worker/src/admins.rs:372"]
    N3["The auth-worker RELAY_DB database_id is a ZERO placeholder that every deployment must override<br/>nostr-bbs-auth-worker/wrangler.toml:20 - shipped unusable on purpose"]
    N4["PROPOSED, NOT BUILT: ADR-2012 demotes the pod-worker payment D1 from ledger to a height-stamped<br/>derived view over the sidestr chain, with a staleness bound that is an error<br/>ADR-2012-d1-ledger-becomes-a-chain-view.md:5-6 ADR-2012-d1-ledger-becomes-a-chain-view.md:37-41.<br/>Nothing in this store map changes until it is built - see NF-04.11"]
```

## NF-08.5 Table map — who creates each table and who reads it

```mermaid
flowchart TB
    subgraph authd1["nostr-bbs-auth D1 - bootstrapped by nostr-bbs-auth-worker/src/schema.rs:19"]
        A1["challenges schema.rs:27<br/>webauthn_credentials schema.rs:32<br/>nip1984_reports schema.rs:50<br/>moderation_actions schema.rs:70"]
        A2["mod_reports schema.rs:82<br/>wot_entries schema.rs:103<br/>members schema.rs:116<br/>invitations schema.rs:123"]
        A3["invitation_redemptions schema.rs:136<br/>welcome_messages schema.rs:164<br/>instance_settings schema.rs:182<br/>username_reservations schema.rs:208"]
    end
    subgraph relayd1["nostr-bbs-relay D1 - bootstrapped by nostr-bbs-relay-worker/src/lib.rs:811"]
        RE["events lib.rs:823 and channel_creators lib.rs:827 - EVENTS_CREATE_SQL lib.rs:717 and<br/>CHANNEL_CREATORS_CREATE_SQL lib.rs:732, created before every ALTER and index"]
        R0["whitelist lib.rs:841 - WHITELIST_CREATE_SQL from nostr-bbs-core/src/whitelist_sql.rs:42<br/>then the ALTER column list lib.rs:877-890"]
        R1["channel_zones lib.rs:927<br/>admin_log lib.rs:932<br/>settings lib.rs:943<br/>reports lib.rs:949<br/>hidden_events lib.rs:964<br/>moderation_actions lib.rs:973"]
        R2["profiles lib.rs:986<br/>agent_registry lib.rs:1000<br/>broker_cases lib.rs:741<br/>broker_decisions lib.rs:765"]
        R3["governance_receipts lib.rs:783<br/>broker_roles lib.rs:1009<br/>pubkey_aliases lib.rs:1022<br/>case_side_receipts lib.rs:1035<br/>case_delegations lib.rs:1047<br/>device_keys - created by the AUTH worker nostr-bbs-auth-worker/src/devices.rs:123"]
    end

    N1["INVARIANT: both bootstraps are idempotent, so a newly added table exists before any handler touches it.<br/>The relay runs ensure_schema ONCE PER ISOLATE, before the WebSocket branch, and latches SCHEMA_READY only<br/>when every CREATE succeeded, so a transient D1 failure is retried on the next request<br/>nostr-bbs-relay-worker/src/lib.rs:195, nostr-bbs-relay-worker/src/lib.rs:200, nostr-bbs-relay-worker/src/lib.rs:222-225<br/>CREATE TABLE IF NOT EXISTS throughout schema.rs:27 and lib.rs:927"]
    N2["device_keys is the one cross-worker table: the AUTH worker creates and writes it into the RELAY's D1<br/>so the relay DO can read it at NIP-42 AUTH with no cross-worker call - see NF-02.7"]
    N3["events and channel_creators are created by the relay bootstrap since 66ca8d8, and checked in as migration 0009 - see NF-08.6"]
    N4["INVARIANT: migration 0006 is MIRRORED into ensure_schema, the live schema path, so the<br/>application-receipt and delegation tables exist on a cold start without the migration runner<br/>lib.rs:1035 lib.rs:1047"]
```

Both workers bootstrap idempotently — the auth worker at
`nostr-bbs-auth-worker/src/schema.rs:19`, the relay at `nostr-bbs-relay-worker/src/lib.rs:811`, which now returns whether every CREATE succeeded and is latched once per isolate (`nostr-bbs-relay-worker/src/lib.rs:222-225`). Earlier the DO and `scheduled` paths never bootstrapped, so a WebSocket-only isolate could miss the schema; the call now precedes the WebSocket branch (`nostr-bbs-relay-worker/src/lib.rs:228-229`). When `PUSH_ENABLED` is true the bootstrap also creates the Web Push tables (ADR-2023), and a default deploy pays nothing (`nostr-bbs-relay-worker/src/lib.rs:849`).

## NF-08.6 Where the relay's load-bearing tables come from

```mermaid
flowchart TB
    SETUP["SETUP.md relay step: wrangler d1 execute --file migrations/0009_events.sql<br/>optional, the relay creates the tables itself SETUP.md:100"]
    MIGE["events and channel_creators, checked in<br/>migrations/0009_events.sql:23 migrations/0009_events.sql:44"]
    CONSTE["EVENTS_CREATE_SQL nostr-bbs-relay-worker/src/lib.rs:717<br/>CHANNEL_CREATORS_CREATE_SQL nostr-bbs-relay-worker/src/lib.rs:732"]
    TEST["a test holds the migration and the constants identical<br/>tests/events_schema.rs:66"]
    ENSE["ensure_schema runs both first, once per isolate<br/>nostr-bbs-relay-worker/src/lib.rs:823 nostr-bbs-relay-worker/src/lib.rs:827"]
    ALT["then ALTER TABLE events ADD COLUMN d_tag and received_at,<br/>for a table made from the older SETUP.md<br/>nostr-bbs-relay-worker/src/lib.rs:832 nostr-bbs-relay-worker/src/lib.rs:833"]
    MIG["whitelist table, checked in<br/>migrations/0008_whitelist.sql:30"]
    CORE["one DDL string, WHITELIST_CREATE_SQL<br/>nostr-bbs-core/src/whitelist_sql.rs:42"]
    ENS["ensure_schema runs it next, before the whitelist ALTER list<br/>nostr-bbs-relay-worker/src/lib.rs:841"]
    READERS["Read on the hot path<br/>whitelist SELECT nostr-bbs-relay-worker/src/whitelist.rs:135<br/>whitelist cohort grant nostr-bbs-relay-worker/src/whitelist.rs:373<br/>whitelist admin count nostr-bbs-relay-worker/src/whitelist.rs:482"]

    SETUP -.-> MIGE
    MIGE --> CONSTE
    TEST -.-> CONSTE
    CONSTE --> ENSE --> ALT --> ENS
    MIG --> CORE
    CORE --> ENS
    ENS --> READERS

    N1["INVARIANT (ADR-2014 phase 1, extended to events at 66ca8d8): each table's DDL is one string shared by the<br/>migration record and the live bootstrap, and each precedes the ALTERs and indexes that touch it, so a<br/>fresh D1 has a table to alter. events first nostr-bbs-relay-worker/src/lib.rs:823, whitelist next<br/>nostr-bbs-relay-worker/src/lib.rs:838-842, with migrations/0008_whitelist.sql:14-20"]
    N2["Closed at 66ca8d8: no migration created events, and the SETUP.md copy-paste omitted d_tag and received_at,<br/>which save_event inserts on every write, so a deploy that followed the guide failed every save<br/>migrations/0009_events.sql:3-6"]
    N3["Governance tables are created TWICE - by migration 0002_governance.sql:5,17,39,52 and inline by<br/>the relay bootstrap, agent_registry lib.rs:1000, broker_cases lib.rs:741, broker_decisions lib.rs:765,<br/>broker_roles lib.rs:1009. Both use IF NOT EXISTS, so this is duplication, not drift.<br/>SETUP.md:123 states the same."]
    N4["DEBT: the migration cannot repair an existing table - SQLite has no ADD COLUMN IF NOT EXISTS, so the<br/>0009 file adds no column and leaves the two ALTERs to its header comment or to ensure_schema, which<br/>discards their result (let _ =), so a real failure there is silent migrations/0009_events.sql:12-17 nostr-bbs-relay-worker/src/lib.rs:835"]
```

**What it shows.** Until 66ca8d8 no code in the repository created `events`: the
bootstrap only indexed it and the operator copied seven columns out of
`SETUP.md`. The relay now creates `events` and `channel_creators` first in
`ensure_schema` (`nostr-bbs-relay-worker/src/lib.rs:823`), from constants a
test holds identical to `migrations/0009_events.sql` (`tests/events_schema.rs:66`),
and then adds the two missing columns to an older table
(`nostr-bbs-relay-worker/src/lib.rs:832`). The whitelist follows the same
shape from 1a26e51 (`migrations/0008_whitelist.sql:30`).

**Why it is this way.** ADR-2014 phase 1 checked the whitelist in because
admission reads `expires_at`, which no DDL created at all
(`migrations/0008_whitelist.sql:5-8`); 66ca8d8 applies the same pattern to the
table every write depends on.

**Debt:** the migration is not a complete record of a deployed schema, because
the two column additions are a hand-applied comment in the file
(`migrations/0009_events.sql:12-17`) and a bootstrap loop that discards its
result (`nostr-bbs-relay-worker/src/lib.rs:835`).

## NF-08.7 Admin authority — three sources, two unions, and where they differ

```mermaid
flowchart TB
    CANON["Canonical algorithm and SQL constants<br/>nostr-bbs-core/src/admin_shared.rs:91 ADMIN_PUBKEYS_VAR<br/>nostr-bbs-core/src/admin_shared.rs:109 is_static_admin"]
    S1["1. Static set - ADMIN_PUBKEYS env var"]
    S2["2. RELAY_DB whitelist.is_admin<br/>nostr-bbs-core/src/admin_shared.rs:72"]
    S3["3. auth DB members.is_admin<br/>nostr-bbs-core/src/admin_shared.rs:76"]
    AUTHIS["auth-worker is_admin - union of all three<br/>nostr-bbs-auth-worker/src/admin.rs:57<br/>static nostr-bbs-auth-worker/src/admin.rs:71<br/>RELAY_DB nostr-bbs-auth-worker/src/admin.rs:76<br/>DB nostr-bbs-auth-worker/src/admin.rs:90"]
    RELIS["relay-worker query_is_admin<br/>nostr-bbs-relay-worker/src/auth.rs:181<br/>static nostr-bbs-relay-worker/src/auth.rs:185<br/>whitelist nostr-bbs-relay-worker/src/auth.rs:194"]

    CANON --> S1 & S2 & S3
    S1 & S2 & S3 --> AUTHIS
    S1 & S2 --> RELIS

    N1["INVARIANT: the pubkey is lower-cased before every lookup - a mixed-case NIP-98 pubkey would<br/>otherwise miss every store and be silently denied nostr-bbs-auth-worker/src/admin.rs:62"]
    N2["INVARIANT fail-closed: any D1 error or missing row returns false, never ambient authority<br/>nostr-bbs-auth-worker/src/admin.rs:103"]
    N3["CLOSED at 66ca8d8: the relay used to query members in its OWN D1 where no migration or bootstrap creates<br/>that table, so the branch failed on every call and the error was swallowed. The query is removed, and relay<br/>authority is ADMIN_PUBKEYS union whitelist.is_admin only nostr-bbs-relay-worker/src/auth.rs:185 nostr-bbs-relay-worker/src/auth.rs:194"]
    N3B["DRIFT: the doc comment on query_is_admin still names members - its Task #7 paragraph says the relay<br/>whitelist/members tables may carry no is_admin row, below a paragraph saying the relay no longer queries it<br/>nostr-bbs-relay-worker/src/auth.rs:174 nostr-bbs-relay-worker/src/auth.rs:164"]
    N3C["INVARIANT: the two workers differ by design - the auth worker unions three sources, the relay two,<br/>because members lives only in the auth D1 nostr-bbs-auth-worker/src/admin.rs:90 nostr-bbs-relay-worker/src/auth.rs:194"]
    N4["DOC-DRIFT: ADMIN_PUBKEYS is DECLARED in exactly one wrangler template - the search worker,<br/>nostr-bbs-search-worker/wrangler.toml:41 - yet is read by the auth worker (admin.rs:71) and the relay<br/>(auth.rs:185). The relay template asserts the opposite: there is no ADMIN_PUBKEYS reader in src/<br/>nostr-bbs-relay-worker/wrangler.toml:18-21. SETUP.md:147-151 never lists it either."]
    N5["Consequence of N4: a by-the-book deployment has NO static admin bootstrap on relay or auth,<br/>and the search worker ships a REAL non-placeholder pubkey as its default admin<br/>nostr-bbs-search-worker/wrangler.toml:41 - the only non-generic value in any template."]
```

## NF-08.8 Cross-worker feature gates that must be set in lockstep

```mermaid
flowchart LR
    DKA["auth-worker DEVICE_KEYS_ENABLED=false<br/>nostr-bbs-auth-worker/wrangler.toml:69"]
    DKR["relay-worker DEVICE_KEYS_ENABLED=false<br/>nostr-bbs-relay-worker/wrangler.toml:29"]
    DKC["client window.__ENV__ third setting<br/>SETUP.md:171-174"]
    GATEFN["auth gate reads its OWN binding<br/>nostr-bbs-auth-worker/src/devices.rs:100"]
    SHARED["shared parse rule only<br/>nostr-bbs-core feature_gate"]
    AM["AUTH_MODE=nip42 default<br/>nostr-bbs-relay-worker/wrangler.toml:39"]
    MESH["MESH_MODE=standalone<br/>nostr-bbs-relay-worker/wrangler.toml:93"]
    ESC["ESCALATION_DEFAULT_TIER=medium<br/>nostr-bbs-relay-worker/wrangler.toml:61"]
    CAL["CALIBRATION_SELECTION_KEY - a SECRET, deliberately NOT a plaintext var<br/>nostr-bbs-relay-worker/wrangler.toml:64-67"]

    DKA --> GATEFN --> SHARED
    DKR --> SHARED
    DKC --> SHARED

    N1["INVARIANT ADR-2004: each worker reads its own binding and decides ALONE - no cross-worker call.<br/>Only the parse rule is shared, so the two cannot drift on what true means<br/>nostr-bbs-auth-worker/src/devices.rs:100-107"]
    N2["INVARIANT: enables on the EXACT string true; unset, empty or anything else is off<br/>nostr-bbs-auth-worker/src/devices.rs:112, asserted nostr-bbs-auth-worker/src/devices.rs:974"]
    N3["AUTH_MODE: anything other than allowlist resolves to the secure nip42 default<br/>nostr-bbs-relay-worker/wrangler.toml:37-39 - see NF-03"]
    N4["ESCALATION_DEFAULT_TIER is a declared SCAFFOLD - the authoritative risk-tier schema is owned by<br/>agentbox, EXTERNAL: see AB-14 and AB-15; an unrecognised tier folds to medium<br/>nostr-bbs-relay-worker/wrangler.toml:54-61"]
    N5["INVARIANT: calibration sampling is HMAC over a key the agent cannot read. The request id is the<br/>31402 d tag, which the agent chooses, so the key secrecy is the only thing stopping it grinding tags<br/>until it finds one sampling never selects nostr-bbs-relay-worker/wrangler.toml:69-74 - see NF-12"]
    N6["Unset, the relay STILL samples and logs a warning on every projection - silently disabling<br/>oversight is treated as the worse failure nostr-bbs-relay-worker/wrangler.toml:76-79"]
    PUSH["relay-worker PUSH_ENABLED=false, ADR-2023 content-free Web Push, off by default<br/>nostr-bbs-relay-worker/wrangler.toml:86 with VAPID_PUBLIC_KEY nostr-bbs-relay-worker/wrangler.toml:87 and PUSH_MAX_FANOUT nostr-bbs-relay-worker/wrangler.toml:90"]
    RURL["RELAY_URL is blank by default, and required in nip42 mode: blank logs a console error on every AUTH<br/>nostr-bbs-relay-worker/wrangler.toml:48-51"]
    N8["All three templates now ship System One advisory flags ON (relay JEV_TRIAGE=1 nostr-bbs-relay-worker/wrangler.toml:11, search JEV_SEARCH_RERANK=1 nostr-bbs-search-worker/wrangler.toml:32, auth JEV_RATIONALE_CHECK=1 nostr-bbs-auth-worker/wrangler.toml:53) pointing at TypeSafe jev-latest (nostr-bbs-relay-worker/wrangler.toml:12-13); TYPESAFE_API_KEY is a Worker secret named at :14 and never a plaintext var, and the auth worker declares no endpoint, so it resolves to TypeSafe through the same secret and stays private only when JEV_RATIONALE_CHECK_LOCAL_ONLY=1 (nostr-bbs-auth-worker/wrangler.toml:54-58)"]
    N7["A second lockstep gate, ENCRYPTION_ENABLED, follows the same own-binding pattern across relay<br/>and clients but is undeclared in every wrangler.toml - see NF-08.2"]
```

## NF-08.9 Encrypted-zone write enforcement — shape check, not decryption

```mermaid
flowchart TB
    WRITE["kind-42 write into a zone<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1450"]
    LOOKUP["ZoneConfig.load(env).is_encrypted(zone)<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1457-1458"]
    TAGCHK["is_zone_ciphertext - zk tag: zone id, epoch >= 1, 64-hex zone pubkey<br/>nostr-bbs-relay-worker/src/zone_config.rs:211-217"]
    SHAPECHK["content shaped as NIP-44 v2 - base64, version 0x02, length 132..87472<br/>nostr-bbs-relay-worker/src/zone_config.rs:218-223"]
    REJECT["send_ok false - blocked: encrypted zone requires zone-key ciphertext<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1459-1464"]
    ACCEPT["event stored"]

    WRITE --> LOOKUP
    LOOKUP -->|"encrypted"| TAGCHK
    LOOKUP -->|"not encrypted"| ACCEPT
    TAGCHK -->|"tag ok"| SHAPECHK
    TAGCHK -->|"no zk tag or malformed"| REJECT
    SHAPECHK -->|"passes"| ACCEPT
    SHAPECHK -->|"fails"| REJECT

    N1["INVARIANT: shape only, no decryption - the relay holds no zone key and cannot tell a real<br/>ciphertext from random bytes of the right length; what it guarantees is that no plaintext,<br/>from any client or any author including admins, lands in an encrypted zone zone_config.rs:204-207"]
    N2["A sealed ADR-2017 envelope into an encrypted zone is exempt from this drift check but still<br/>subject to the admin-only sealed-write gate - nip_handlers.rs:1436-1447"]
```
