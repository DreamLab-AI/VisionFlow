---
id: NF-08
title: Config projection, zones, the KV/D1/R2 store map and admin authority
area: nostr-rust-forum
governing:
  - ../nostr-rust-forum/docs/BASELINE-architecture.md
  - ../nostr-rust-forum/docs/IDENTITY-keys-and-trust.md
adrs: [ADR-2004, ADR-2006, ADR-2007, ADR-2011, ADR-2012, ADR-2014, ADR-2016, ADR-2022, ADR-2023, ADR-2027, ADR-2028, ADR-2029, ADR-2033]
sources:
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/utils/devices.rs
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
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/migrations/0005_drop_nip1984_reports.sql
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/devices.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/username.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/admins.rs
  - ../nostr-rust-forum/docs/adr/ADR-2012-d1-ledger-becomes-a-chain-view.md
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/feature_gate.rs
  - ../nostr-rust-forum/docs/adr/ADR-2029-room-profile-deployment-type.md
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/moderation.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/user_admin.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/admin_events.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/recipient_keys.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/migrations/0013_recipient_keys_proof.sql
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/zone_groups.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/recipient_keys.rs
  - ../nostr-rust-forum/crates/nostr-bbs-pod-worker/src/lib.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/governance_api.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/invites.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/moderation.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/wot.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/welcome.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/webauthn.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/did.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/auth.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/storage.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/receipts.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/mod_cache.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/cron.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/agent_disclosure.rs
  - ../nostr-rust-forum/crates/nostr-bbs-search-worker/src/auth.rs
  - ../nostr-rust-forum/crates/nostr-bbs-pod-worker/src/auth.rs
  - ../nostr-rust-forum/crates/nostr-bbs-rate-limit/src/replay.rs
verified_commit: 8a0ca418b9f129b219431880d9fa5a81603428fc
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
    GATE["relay gate - the real access boundary<br/>nostr-bbs-relay-worker/src/zone_config.rs:89"]
    TILES["client renders tiles from the same JSON"]

    TOML --> ZC
    TOML2 --> ZC
    TOML3 --> ZC
    ZC --> RELAYENV --> GATE
    ZC --> CLIENTENV --> TILES

    N1["INVARIANT: one JSON, two enforcement points - the tiles a member sees and the gate the relay<br/>enforces cannot describe two different models. Relay side parses at nostr-bbs-relay-worker/src/zone_config.rs:113"]
    N2["INVARIANT deny-by-default: an absent or malformed ZONE_CONFIG yields an EMPTY config,<br/>so every zone lookup misses and every gate denies nostr-bbs-relay-worker/src/zone_config.rs:113"]
    N3["forum.example.toml ships zeroed placeholder pubkeys that FAIL validation by design forum.example.toml:86-91"]
    N4["OPEN: [deployment] profile is proposed (ADR-2029), only forum deploys until it is accepted and implemented<br/>forum.example.toml:47, forum.example.toml:52"]
    N5["[[ingress]] recipients are admitted without a whitelist row, role primary, copy or agent (ADR-2022, live)<br/>forum.example.toml:165-185"]
```

**Open:** `profile` in `[deployment]` selects a `"forum"` or `"room"` wire shape for the life of the deployment, but ADR-2029 is proposed with implementation none and the example file says only `"forum"` deploys until it is accepted (`../nostr-rust-forum/forum.example.toml:47`, `../nostr-rust-forum/docs/adr/ADR-2029-room-profile-deployment-type.md:5`); its dependencies ADR-2027 (accepted, partial, inactive) and ADR-2028 (proposed, header still `implementation_status: none`) are likewise not live. **Drift:** ADR-2028's header is behind the code: its zone group binding and epoch uniqueness (decisions 4, 6 and 8) are on main behind `ZONE_GROUPS_ENABLED`, off by default (`../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/zone_groups.rs:1-4`, NF-08.9).

## NF-08.2 Zone visibility and the read/write gate lattice

```mermaid
flowchart LR
    subgraph vis["ZoneVisibility"]
        PUB["Public<br/>nostr-bbs-relay-worker/src/zone_config.rs:28"]
        LOCK["Locked (default)<br/>nostr-bbs-relay-worker/src/zone_config.rs:31"]
        HID["Hidden<br/>nostr-bbs-relay-worker/src/zone_config.rs:33"]
    end
    READ["cohorts_can_read<br/>nostr-bbs-relay-worker/src/zone_config.rs:191"]
    WRITE["cohorts_can_write<br/>nostr-bbs-relay-worker/src/zone_config.rs:207"]
    PUBR["is_public_read - Public AND required_cohorts empty<br/>nostr-bbs-relay-worker/src/zone_config.rs:173"]
    DEFS["defs_visible_to_nonmember - anything not Hidden<br/>nostr-bbs-relay-worker/src/zone_config.rs:182"]
    EFF["effective_write_cohorts = write_cohorts ?? required_cohorts<br/>nostr-bbs-relay-worker/src/zone_config.rs:68"]
    ZENC["Zone.encrypted flag<br/>nostr-bbs-relay-worker/src/zone_config.rs:58"]
    MGATE["ENCRYPTION_ENABLED master gate<br/>field nostr-bbs-relay-worker/src/zone_config.rs:82 loaded :91"]
    ISENC["is_encrypted = gate AND zone.encrypted AND NOT is_public_read<br/>nostr-bbs-relay-worker/src/zone_config.rs:163"]

    PUB --> PUBR --> READ
    LOCK --> DEFS
    PUB --> DEFS
    HID -->|"omitted entirely"| DEFS
    EFF --> WRITE
    ZENC --> ISENC
    MGATE --> ISENC
    PUBR -->|"public zones never enforced, even if flagged"| ISENC

    N1["INVARIANT: an EMPTY effective write set denies every non-admin - writes are never anonymous<br/>nostr-bbs-relay-worker/src/zone_config.rs:212"]
    N2["Unknown zone id returns false on BOTH gates - no zone, no access<br/>read nostr-bbs-relay-worker/src/zone_config.rs:193, write nostr-bbs-relay-worker/src/zone_config.rs:209"]
    N3["A public zone with write_cohorts friends is openly readable and inner-circle writable<br/>asserted at nostr-bbs-relay-worker/src/zone_config.rs:397-400"]
    N4["DOC-DRIFT: ENCRYPTION_ENABLED is read from env at zone_config.rs:91 but declared in NO<br/>wrangler.toml [vars] block in any worker - unlike DEVICE_KEYS_ENABLED (see NF-08.8) it has no<br/>documented deployment default, so a deploy that forgets it silently leaves encryption OFF"]
    N5["see NF-08.9 for the shape check that enforces an encrypted zone once is_encrypted is true"]
```

## NF-08.3 Auto-approval of a new joiner — config-driven cohort grant

```mermaid
sequenceDiagram
    autonumber
    participant U as New user
    participant AW as auth-worker username::claim<br/>nostr-bbs-auth-worker/src/username.rs:210
    participant ZA as zone_approval::new_joiner_cohorts_json<br/>nostr-bbs-auth-worker/src/zone_approval.rs:30
    participant D1 as RELAY_DB whitelist INSERT<br/>nostr-bbs-auth-worker/src/username.rs:307

    U->>AW: POST /api/username/claim (NIP-98 authed)
    AW->>ZA: ZONE_CONFIG string
    ZA->>ZA: base cohort vector starts as members zone_approval.rs:31
    ZA->>ZA: for each zone with auto_approve, add required_cohorts de-duplicated zone_approval.rs:34-38
    ZA-->>AW: JSON cohort array
    AW->>D1: whitelist row for the claimed pubkey, ON CONFLICT DO NOTHING username.rs:308

    Note over ZA: INVARIANT opt-in per zone, deny-by-default - absent, empty or malformed ZONE_CONFIG yields exactly the members cohort alone zone_approval.rs:29, tested zone_approval.rs:53-60
    Note over ZA: Only auto_approve zones grant - family and business stay admin-gated zone_approval.rs:74-78
```

## NF-08.4 Persistent stores — which worker binds which D1, KV and R2

```mermaid
flowchart TB
    subgraph d1["D1 databases"]
        AUTHDB["nostr-bbs-auth<br/>auth-worker DB nostr-bbs-auth-worker/wrangler.toml:9"]
        RELDB["nostr-bbs-relay<br/>relay-worker DB nostr-bbs-relay-worker/wrangler.toml:101"]
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
        RELAYDO["RELAY class NostrRelayDO<br/>nostr-bbs-relay-worker/wrangler.toml:117 sqlite class :122"]
    end

    AUTHW["auth-worker"] --> AUTHDB
    AUTHW -->|"RELAY_DB nostr-bbs-auth-worker/wrangler.toml:18"| RELDB
    AUTHW --> SESS
    AUTHW --> KVALIAS
    AUTHW -->|"POD_META backwards-compat reads nostr-bbs-auth-worker/wrangler.toml:34"| PODMETA
    AUTHW -->|"PODS nostr-bbs-auth-worker/wrangler.toml:47"| PODS
    RELAYW["relay-worker"] --> RELDB
    RELAYW -->|"REPLAY_DB points at nostr-bbs-auth nostr-bbs-relay-worker/wrangler.toml:112"| AUTHDB
    RELAYW --> RELAYDO
    PODW["pod-worker"] --> PODS
    PODW --> PODMETA
    PODW -->|"REPLAY_DB nostr-bbs-pod-worker/wrangler.toml:25"| AUTHDB
    PODW -->|"RELAY_DB admin reads nostr-bbs-pod-worker/wrangler.toml:32, zero id :34"| RELDB
    SEARCHW["search-worker"] --> VECS
    SEARCHW --> SCFG
    SEARCHW -->|"REPLAY_DB nostr-bbs-search-worker/wrangler.toml:25"| AUTHDB
    PREVW["preview-worker"] --> RL

    N1["INVARIANT: NIP-98 replay lives in ONE database - relay, pod and search bind REPLAY_DB, and auth uses DB, all to<br/>nostr-bbs-auth so cross-worker replay is detected nostr-bbs-relay-worker/wrangler.toml:106-113<br/>auth DB is nostr-bbs-auth nostr-bbs-auth-worker/wrangler.toml:9-10. The preview worker binds no replay store."]
    N2["ANOMALY O11 now DOCUMENTED, not removed: KV and SESSIONS are still the SAME physical namespace id<br/>nostr-bbs-auth-worker/wrangler.toml:28 and :44; the template says the key prefixes are disjoint<br/>nostr-bbs-auth-worker/wrangler.toml:22-25 and a unit test pins the admin cache key outside the rl: prefix<br/>nostr-bbs-auth-worker/src/admins.rs:347"]
    N3["The auth-worker RELAY_DB database_id is a ZERO placeholder that every deployment must override<br/>nostr-bbs-auth-worker/wrangler.toml:20 - shipped unusable on purpose"]
    N5["DRIFT: the pod template's older comment says admin checks use members.is_admin or whitelist.is_admin<br/>through REPLAY_DB nostr-bbs-pod-worker/wrangler.toml:19-20, the newer one beside the RELAY_DB block says<br/>the relay whitelist only nostr-bbs-pod-worker/wrangler.toml:29-30, and the code reads RELAY_DB only<br/>nostr-bbs-pod-worker/src/lib.rs:2022 - see NF-08.7"]
    N4["PROPOSED AND PARKED: ADR-2012 demotes the pod-worker payment D1 from ledger to a height-stamped<br/>derived view over the sidestr chain, with a staleness bound that is an error<br/>ADR-2012-d1-ledger-becomes-a-chain-view.md:48-51. Status proposed, implementation partial, inactive<br/>ADR-2012-d1-ledger-becomes-a-chain-view.md:5-7, parked at P3 ADR-2012-d1-ledger-becomes-a-chain-view.md:165.<br/>The chain view is not built, so this store map is unchanged - see NF-04.11"]
```

## NF-08.5 Table map — who creates each table and who reads it

> **Superseded** by NF-08.15 at 94199e5 (frozen at 4a73a7a).

```mermaid
flowchart TB
    subgraph authd1["nostr-bbs-auth D1 - bootstrapped by nostr-bbs-auth-worker/src/schema.rs:19"]
        A1["challenges schema.rs:28<br/>webauthn_credentials schema.rs:33<br/>nip1984_reports schema.rs:51<br/>moderation_actions schema.rs:71"]
        A2["mod_reports schema.rs:83<br/>wot_entries schema.rs:104<br/>members schema.rs:117<br/>invitations schema.rs:124"]
        A3["invitation_redemptions schema.rs:137<br/>welcome_messages schema.rs:165<br/>instance_settings schema.rs:183<br/>username_reservations schema.rs:209"]
    end
    subgraph relayd1["nostr-bbs-relay D1 - bootstrapped by nostr-bbs-relay-worker/src/lib.rs:1059"]
        RE["events nostr-bbs-relay-worker/src/lib.rs:1071 and channel_creators nostr-bbs-relay-worker/src/lib.rs:1075 - EVENTS_CREATE_SQL nostr-bbs-relay-worker/src/lib.rs:720 and<br/>CHANNEL_CREATORS_CREATE_SQL nostr-bbs-relay-worker/src/lib.rs:735, created before every ALTER and index"]
        R0["whitelist nostr-bbs-relay-worker/src/lib.rs:1100 - WHITELIST_CREATE_SQL from nostr-bbs-core/src/whitelist_sql.rs:42<br/>then the ALTER column list nostr-bbs-relay-worker/src/lib.rs:1183-1196"]
        R1["channel_zones nostr-bbs-relay-worker/src/lib.rs:1233<br/>admin_log nostr-bbs-relay-worker/src/lib.rs:1238<br/>settings nostr-bbs-relay-worker/src/lib.rs:1249<br/>reports nostr-bbs-relay-worker/src/lib.rs:1255<br/>hidden_events nostr-bbs-relay-worker/src/lib.rs:1270<br/>moderation_actions nostr-bbs-relay-worker/src/lib.rs:1279"]
        R2["profiles nostr-bbs-relay-worker/src/lib.rs:1292<br/>agent_registry nostr-bbs-relay-worker/src/lib.rs:1306<br/>broker_cases nostr-bbs-relay-worker/src/lib.rs:989<br/>broker_decisions nostr-bbs-relay-worker/src/lib.rs:1013"]
        R3["governance_receipts nostr-bbs-relay-worker/src/lib.rs:1031<br/>broker_roles nostr-bbs-relay-worker/src/lib.rs:1315<br/>pubkey_aliases nostr-bbs-relay-worker/src/lib.rs:1328<br/>case_side_receipts nostr-bbs-relay-worker/src/lib.rs:1341<br/>case_delegations nostr-bbs-relay-worker/src/lib.rs:1353<br/>device_keys - created by the AUTH worker nostr-bbs-auth-worker/src/devices.rs:123"]
        R4["tombstone, key and marker tables - see NF-08.11"]
    end

    N1["INVARIANT: both bootstraps are idempotent, so a newly added table exists before any handler touches it.<br/>The relay runs ensure_schema ONCE PER ISOLATE, before the WebSocket branch, and latches SCHEMA_READY only<br/>when every CREATE succeeded, so a transient D1 failure is retried on the next request<br/>nostr-bbs-relay-worker/src/lib.rs:197, nostr-bbs-relay-worker/src/lib.rs:202, nostr-bbs-relay-worker/src/lib.rs:224-227<br/>CREATE TABLE IF NOT EXISTS throughout schema.rs:28 and nostr-bbs-relay-worker/src/lib.rs:1233"]
    N2["device_keys was the first cross-worker table: the AUTH worker creates and writes it into the RELAY's D1, so the relay DO<br/>resolves a device to its owner with no cross-worker call - on EVENT and REQ through effective_pubkey nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:3781,<br/>NOT at NIP-42 AUTH, which binds the raw signing key nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:2152 - see NF-08.12, NF-20.4 and NF-02.7"]
    N3["events and channel_creators are created by the relay bootstrap since 66ca8d8, and checked in as migration 0009 - see NF-08.6"]
    N4["INVARIANT: migration 0006 is MIRRORED into ensure_schema, the live schema path, so the<br/>application-receipt and delegation tables exist on a cold start without the migration runner<br/>nostr-bbs-relay-worker/src/lib.rs:1341 nostr-bbs-relay-worker/src/lib.rs:1353"]
```

Both workers bootstrap idempotently — the auth worker at
`nostr-bbs-auth-worker/src/schema.rs:19`, the relay at `nostr-bbs-relay-worker/src/lib.rs:1059`, which returns whether every load-bearing CREATE succeeded and is latched once per isolate (`nostr-bbs-relay-worker/src/lib.rs:224-227`). Earlier the DO and `scheduled` paths never bootstrapped, so a WebSocket-only isolate could miss the schema; the call now precedes the WebSocket branch (`nostr-bbs-relay-worker/src/lib.rs:230-231`). When `PUSH_ENABLED` is true the bootstrap also creates the Web Push tables (ADR-2023), and a default deploy pays nothing (`nostr-bbs-relay-worker/src/lib.rs:1106-1108`); `recipient_keys` and `zone_epoch_keys` follow the same flag-gated pattern (`nostr-bbs-relay-worker/src/lib.rs:1120-1158`). The NIP-09 tombstone tables are created on every deploy, and two one-time passes then backfill them from stored kind-5s and re-key plain replaceable addresses, each guarded by a `settings` marker (`nostr-bbs-relay-worker/src/lib.rs:1423-1424`).

## NF-08.6 Where the relay's load-bearing tables come from

```mermaid
flowchart TB
    SETUP["SETUP.md relay step: wrangler d1 execute --file migrations/0009_events.sql<br/>optional, the relay creates the tables itself SETUP.md:100"]
    MIGE["events and channel_creators, checked in<br/>migrations/0009_events.sql:23 migrations/0009_events.sql:44"]
    CONSTE["EVENTS_CREATE_SQL nostr-bbs-relay-worker/src/lib.rs:720<br/>CHANNEL_CREATORS_CREATE_SQL nostr-bbs-relay-worker/src/lib.rs:735"]
    TEST["a test holds the migration and the constants identical<br/>tests/events_schema.rs:66"]
    ENSE["ensure_schema runs both first, once per isolate<br/>nostr-bbs-relay-worker/src/lib.rs:1071 nostr-bbs-relay-worker/src/lib.rs:1075"]
    ALT["then ALTER TABLE events ADD COLUMN d_tag and received_at,<br/>for a table made from the older SETUP.md<br/>nostr-bbs-relay-worker/src/lib.rs:1091 nostr-bbs-relay-worker/src/lib.rs:1092"]
    MIG["whitelist table, checked in<br/>migrations/0008_whitelist.sql:30"]
    CORE["one DDL string, WHITELIST_CREATE_SQL<br/>nostr-bbs-core/src/whitelist_sql.rs:42"]
    ENS["ensure_schema runs it next, before the whitelist ALTER list<br/>nostr-bbs-relay-worker/src/lib.rs:1100"]
    READERS["Read on the hot path<br/>whitelist SELECT nostr-bbs-relay-worker/src/whitelist.rs:135<br/>whitelist cohort grant nostr-bbs-relay-worker/src/whitelist.rs:373<br/>whitelist admin count nostr-bbs-relay-worker/src/whitelist.rs:482"]

    SETUP -.-> MIGE
    MIGE --> CONSTE
    TEST -.-> CONSTE
    CONSTE --> ENSE --> ALT --> ENS
    MIG --> CORE
    CORE --> ENS
    ENS --> READERS

    N1["INVARIANT (ADR-2014 phase 1, extended to events at 66ca8d8): each table's DDL is one string shared by the<br/>migration record and the live bootstrap, and each precedes the ALTERs and indexes that touch it, so a<br/>fresh D1 has a table to alter. events first nostr-bbs-relay-worker/src/lib.rs:1071, whitelist next<br/>nostr-bbs-relay-worker/src/lib.rs:1097-1101, with migrations/0008_whitelist.sql:14-20"]
    N2["Closed at 66ca8d8: no migration created events, and the SETUP.md copy-paste omitted d_tag and received_at,<br/>which save_event inserts on every write, so a deploy that followed the guide failed every save<br/>migrations/0009_events.sql:3-6"]
    N3["Governance tables are created TWICE - by migration 0002_governance.sql:5,17,39,52 and inline by<br/>the relay bootstrap, agent_registry nostr-bbs-relay-worker/src/lib.rs:1306, broker_cases nostr-bbs-relay-worker/src/lib.rs:989, broker_decisions nostr-bbs-relay-worker/src/lib.rs:1013,<br/>broker_roles nostr-bbs-relay-worker/src/lib.rs:1315. Both use IF NOT EXISTS, so this is duplication, not drift.<br/>DRIFT: SETUP.md:123-125 states the duplication but credits the AUTH worker with the inline copy;<br/>the auth bootstrap nostr-bbs-auth-worker/src/schema.rs:19 creates no governance table"]
    N4["DEBT: the migration cannot repair an existing table - SQLite has no ADD COLUMN IF NOT EXISTS, so the<br/>0009 file adds no column and leaves the two ALTERs to its header comment or to ensure_schema, which<br/>discards their result (let _ =), so a real failure there is silent migrations/0009_events.sql:12-17 nostr-bbs-relay-worker/src/lib.rs:1094"]
    N5["TENSION with N1: the NIP-09 tombstone tables break the pattern - their DDL lives only in the bootstrap constants<br/>nostr-bbs-relay-worker/src/lib.rs:746 nostr-bbs-relay-worker/src/lib.rs:755 and no migration file records them,<br/>while recipient_keys and zone_epoch_keys keep it, each held identical to migrations 0012 and 0014 by a test<br/>nostr-bbs-relay-worker/src/recipient_keys.rs:29-31 nostr-bbs-relay-worker/src/zone_groups.rs:26-28"]
    N6["RESOLVED at 4a73a7a: the proof-column ALTER is now labelled migration 0013 nostr-bbs-relay-worker/src/lib.rs:1146 and<br/>nostr-bbs-relay-worker/src/recipient_keys.rs:45, matching the checked-in file 0013_recipient_keys_proof.sql:10"]
```

**What it shows.** Until 66ca8d8 no code in the repository created `events`: the
bootstrap only indexed it and the operator copied seven columns out of
`SETUP.md`. The relay now creates `events` and `channel_creators` first in
`ensure_schema` (`nostr-bbs-relay-worker/src/lib.rs:1071`), from constants a
test holds identical to `migrations/0009_events.sql` (`tests/events_schema.rs:66`),
and then adds the two missing columns to an older table
(`nostr-bbs-relay-worker/src/lib.rs:1091`). The whitelist follows the same
shape from 1a26e51 (`migrations/0008_whitelist.sql:30`).

**Why it is this way.** ADR-2014 phase 1 checked the whitelist in because
admission reads `expires_at`, which no DDL created at all
(`migrations/0008_whitelist.sql:5-8`); 66ca8d8 applies the same pattern to the
table every write depends on.

**Debt:** the migration is not a complete record of a deployed schema, because
the two column additions are a hand-applied comment in the file
(`migrations/0009_events.sql:12-17`) and a bootstrap loop that discards its
result (`nostr-bbs-relay-worker/src/lib.rs:1094`).

**Drift:** `SETUP.md:123-125` credits the auth worker with the inline governance tables; only the relay bootstrap creates them (N3).

## NF-08.7 Admin authority — three sources, four resolvers, and where they differ

```mermaid
flowchart TB
    CANON["Canonical algorithm and SQL constants<br/>nostr-bbs-core/src/admin_shared.rs:91 ADMIN_PUBKEYS_VAR<br/>nostr-bbs-core/src/admin_shared.rs:109 is_static_admin"]
    S1["1. Static set - ADMIN_PUBKEYS env var"]
    S2["2. RELAY_DB whitelist.is_admin<br/>nostr-bbs-core/src/admin_shared.rs:72"]
    S3["3. auth DB members.is_admin - still defined, no longer consulted by either resolver<br/>nostr-bbs-core/src/admin_shared.rs:76"]
    AUTHIS["auth-worker is_admin - union of static and RELAY_DB only<br/>nostr-bbs-auth-worker/src/admin.rs:53<br/>static nostr-bbs-auth-worker/src/admin.rs:63<br/>RELAY_DB nostr-bbs-auth-worker/src/admin.rs:72"]
    RELIS["relay-worker query_is_admin<br/>nostr-bbs-relay-worker/src/auth.rs:181<br/>static nostr-bbs-relay-worker/src/auth.rs:185<br/>whitelist nostr-bbs-relay-worker/src/auth.rs:194"]

    PODIS["pod-worker is_admin_user - RELAY_DB whitelist.is_admin ONLY, no static set<br/>nostr-bbs-pod-worker/src/lib.rs:2016, read nostr-bbs-pod-worker/src/lib.rs:2022"]
    CHK["GET check-whitelist isAdmin - whitelist row only<br/>nostr-bbs-relay-worker/src/whitelist.rs:135, reply nostr-bbs-relay-worker/src/whitelist.rs:180"]

    CANON --> S1 & S2 & S3
    S1 & S2 --> AUTHIS
    S1 & S2 --> RELIS
    SRCHIS["search-worker - static ADMIN_PUBKEYS ONLY, no D1 - see NF-07.2"]
    S2 --> PODIS
    S1 --> SRCHIS
    S2 --> CHK

    N1["Auth lower-cases at resolver entry nostr-bbs-auth-worker/src/admin.rs:58<br/>Relay query binds its supplied pubkey nostr-bbs-relay-worker/src/auth.rs:195"]
    N2["RESOLVED at nostr-rust-forum 6064622: the auth resolver no longer falls through to auth members<br/>False after the static set and RELAY_DB grant nothing nostr-bbs-auth-worker/src/admin.rs:85"]
    N3["CLOSED at 66ca8d8: the relay used to query members in its OWN D1 where no migration or bootstrap creates<br/>that table, so the branch failed on every call and the error was swallowed. The query is removed, and relay<br/>authority is ADMIN_PUBKEYS union whitelist.is_admin only nostr-bbs-relay-worker/src/auth.rs:185 nostr-bbs-relay-worker/src/auth.rs:194"]
    N3B["DRIFT: the doc comment on query_is_admin still names members - its Task #7 paragraph says the relay<br/>whitelist/members tables may carry no is_admin row, below a paragraph saying the relay no longer queries it<br/>nostr-bbs-relay-worker/src/auth.rs:174 nostr-bbs-relay-worker/src/auth.rs:164"]
    N3C["REVOCATION GAP: RESOLVED at nostr-rust-forum 6064622 - relay set-admin changes only whitelist nostr-bbs-relay-worker/src/whitelist.rs:450<br/>a retained auth members grant no longer confers authority nostr-bbs-auth-worker/src/admin.rs:72"]
    N6["TENSION, NF-10 O25 / KF-27: a static ADMIN_PUBKEYS admin is an admin to the relay, auth and search workers but not to the<br/>pod worker PODIS, a whitelist-only admin is not one to search, and check-whitelist reports isAdmin false for them, which is what the client reads to honour a<br/>restored sealed original's tombstone - see NF-08.10"]
    N4["DOC-DRIFT: ADMIN_PUBKEYS is DECLARED in exactly one wrangler template - the search worker,<br/>nostr-bbs-search-worker/wrangler.toml:41 - yet is read by the auth worker (admin.rs:63) and the relay<br/>(nostr-bbs-relay-worker/src/auth.rs:185). The relay template asserts the opposite: there is no ADMIN_PUBKEYS reader in src/<br/>nostr-bbs-relay-worker/wrangler.toml:18-21. SETUP.md:147-151 never lists it either."]
    N5["Consequence of N4: a by-the-book deployment has NO static admin bootstrap on relay or auth,<br/>and the search worker ships a REAL non-placeholder pubkey as its default admin<br/>nostr-bbs-search-worker/wrangler.toml:41 - the only non-generic value in any template."]
```

The [1 November membership review](../../estate-review/2026-11-01-forum-membership-lifecycle.md) corrects only the normalisation, error-fallback and revocation claims in NF-08.7. The [2 November review](../../estate-review/2026-11-02-forum-config-ci-invariants.md) re-read every citation in this topic, the template and bootstrap notes included, at the declared revision.

## NF-08.8 Cross-worker feature gates that must be set in lockstep

```mermaid
flowchart LR
    DKA["auth-worker DEVICE_KEYS_ENABLED=false<br/>nostr-bbs-auth-worker/wrangler.toml:69"]
    DKR["relay-worker DEVICE_KEYS_ENABLED=false<br/>nostr-bbs-relay-worker/wrangler.toml:29"]
    DKC["client window.__ENV__ third setting<br/>SETUP.md:175-178"]
    GATEFN["auth gate reads its OWN binding<br/>nostr-bbs-auth-worker/src/devices.rs:100"]
    SHARED["shared parse rule only<br/>nostr-bbs-core feature_gate"]
    AM["AUTH_MODE=nip42 default<br/>nostr-bbs-relay-worker/wrangler.toml:39"]
    MESH["MESH_MODE=standalone<br/>nostr-bbs-relay-worker/wrangler.toml:95"]
    ESC["ESCALATION_DEFAULT_TIER=medium<br/>nostr-bbs-relay-worker/wrangler.toml:63"]
    CAL["CALIBRATION_SELECTION_KEY - a SECRET, deliberately NOT a plaintext var<br/>nostr-bbs-relay-worker/wrangler.toml:66-69"]

    DKA --> GATEFN --> SHARED
    DKR --> SHARED
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/utils/devices.rs:111
    CLIENTPARSE["Client uses its own loose parser<br/>boolean true or normalised true / 1"]
    DKC --> CLIENTPARSE

    N1["INVARIANT ADR-2004: each worker reads its own binding and decides ALONE - no cross-worker call.<br/>Only the parse rule is shared, so the two cannot drift on what true means<br/>nostr-bbs-auth-worker/src/devices.rs:100-107, relay delegates at nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1003"]
    N2["INVARIANT: enables on the EXACT string true; unset, empty, padded, TRUE or 1 is off<br/>rule nostr-bbs-core/src/feature_gate.rs:65, asserted nostr-bbs-auth-worker/src/devices.rs:988"]
    N3["AUTH_MODE: anything other than allowlist resolves to the secure nip42 default<br/>nostr-bbs-relay-worker/wrangler.toml:37-39 - see NF-03"]
    N4["ESCALATION_DEFAULT_TIER is a declared SCAFFOLD - the authoritative risk-tier schema is owned by<br/>agentbox, EXTERNAL: see AB-14 and AB-15; an unrecognised tier folds to medium<br/>nostr-bbs-relay-worker/wrangler.toml:56-63"]
    N5["INVARIANT: calibration sampling is HMAC over a key the agent cannot read. The request id is the<br/>31402 d tag, which the agent chooses, so the key secrecy is the only thing stopping it grinding tags<br/>until it finds one sampling never selects nostr-bbs-relay-worker/wrangler.toml:71-76 - see NF-12"]
    N6["Unset, the relay STILL samples and logs a warning on every projection - silently disabling<br/>oversight is treated as the worse failure nostr-bbs-relay-worker/wrangler.toml:78-81"]
    PUSH["relay-worker PUSH_ENABLED=false, ADR-2023 content-free Web Push, off by default<br/>nostr-bbs-relay-worker/wrangler.toml:88 with VAPID_PUBLIC_KEY nostr-bbs-relay-worker/wrangler.toml:89 and PUSH_MAX_FANOUT nostr-bbs-relay-worker/wrangler.toml:92"]
    RURL["RELAY_URL is blank by default and must be set in nip42 mode: while it and RELAY_URL_ALIASES are blank the relay FAILS CLOSED,<br/>refusing every NIP-42 AUTH, so every write and every protected-kind read is refused, and logs an error on every AUTH<br/>nostr-bbs-relay-worker/wrangler.toml:42-49, RELAY_URL_ALIASES adds extra hostnames nostr-bbs-relay-worker/wrangler.toml:53 - see NF-12.<br/>DOC-DRIFT: that logged error still names only refused writes, not the refused protected-kind reads - see NF-03.2"]
    N8["All three templates now ship System One advisory flags ON (relay JEV_TRIAGE=1 nostr-bbs-relay-worker/wrangler.toml:11, search JEV_SEARCH_RERANK=1 nostr-bbs-search-worker/wrangler.toml:32, auth JEV_RATIONALE_CHECK=1 nostr-bbs-auth-worker/wrangler.toml:53) pointing at TypeSafe jev-latest (nostr-bbs-relay-worker/wrangler.toml:12-13); TYPESAFE_API_KEY is a Worker secret named at :14 and never a plaintext var, and the auth worker declares no endpoint, so it resolves to TypeSafe through the same secret and stays private only when JEV_RATIONALE_CHECK_LOCAL_ONLY=1 (nostr-bbs-auth-worker/wrangler.toml:54-58)"]
    N9["TENSION: the workers enable only on exact true nostr-bbs-core/src/feature_gate.rs:65, but the client lower-cases<br/>and also accepts 1 - an operator who sets 1 in all three places gets a client that registers device keys<br/>and workers that 404 the device routes. Client parser: CLIENTPARSE above"]
    CLIENTPARSE -.- N9
    RKA["auth-worker RECIPIENT_KEYS_ENABLED=false<br/>nostr-bbs-auth-worker/wrangler.toml:70, exact true nostr-bbs-auth-worker/src/recipient_keys.rs:72"]
    RKR["relay-worker RECIPIENT_KEYS_ENABLED and ZONE_GROUPS_ENABLED, exact true<br/>nostr-bbs-relay-worker/src/recipient_keys.rs:70 nostr-bbs-relay-worker/src/zone_groups.rs:47<br/>declared in NO relay template"]
    RKT["recipient_keys table: created by the RELAY only when its flag is on<br/>nostr-bbs-relay-worker/src/lib.rs:1129-1139, written by the AUTH routes via RELAY_DB<br/>nostr-bbs-auth-worker/src/recipient_keys.rs:571"]
    RKA --> RKT
    RKR --> RKT
    N10["TENSION: a third lockstep pair with no shared parse crate and no relay default - auth on with the relay off<br/>serves registration routes over a table the relay never created, and the relay gift-wrap gate stays on its old rule.<br/>Unlike DEVICE_KEYS_ENABLED N1, the relay template does not declare either flag"]
    RKT -.- N10
    N7["A second lockstep gate, ENCRYPTION_ENABLED, follows the same own-binding pattern across relay<br/>and clients but is undeclared in every wrangler.toml - see NF-08.2"]
```

**Tension:** the client accepts `1` for `DEVICE_KEYS_ENABLED` (`../nostr-rust-forum/crates/nostr-bbs-forum-client/src/utils/devices.rs:111`) while both workers need exact `true` (N9).

## NF-08.9 Encrypted-zone write enforcement — shape check, not decryption

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1480
    WRITE["kind-42 write into a zone"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1487-1488
    LOOKUP["ZoneConfig.load(env).is_encrypted(zone)"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/zone_config.rs:241-247
    TAGCHK["is_zone_ciphertext - zk tag: zone id, epoch >= 1, 64-hex zone pubkey"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/zone_config.rs:248-253
    SHAPECHK["content shaped as NIP-44 v2 - base64, version 0x02, length 132..87472"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1489-1494
    REJECT["send_ok false - blocked: encrypted zone requires zone-key ciphertext"]
    ACCEPT["Continue admission<br/>other event gates still apply"]

    WRITE --> LOOKUP
    LOOKUP -->|"encrypted"| TAGCHK
    LOOKUP -->|"not encrypted"| ACCEPT
    TAGCHK -->|"tag ok"| SHAPECHK
    TAGCHK -->|"no zk tag or malformed"| REJECT
    SHAPECHK -->|"passes"| ACCEPT
    SHAPECHK -->|"fails"| REJECT
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1576-1588
    ZG["ZONE_GROUPS_ENABLED exactly true, kind 42 or kanban bound to a zone:<br/>zone_group_gate, later in admission"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:2567
    HINT["zk fifth element must match the zone's group_pubkey from ZONE_CONFIG<br/>zone_config.rs:63 zone_config.rs:136"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:2578-2586
    UNIQ["first key stored for a zone and epoch wins in zone_epoch_keys,<br/>a different key is refused, a D1 failure refuses"]
    ACCEPT -.-> ZG --> HINT --> UNIQ

    N1["Shape only: the relay holds no zone key and cannot authenticate ciphertext<br/>Accepted byte shape does not guarantee encryption or content secrecy zone_config.rs:234-237"]
    N2["ADR-2017 exempts a valid sealed marker from the past timestamp bound only<br/>future timestamp limit remains<br/>admin and encrypted-zone gates still apply<br/>ciphertext shape checking is not bypassed<br/>nip_handlers.rs:1466-1494 nip_handlers.rs:1793"]
    N3["OPEN, ADR-2028: inside the bound zone a cohort member can still claim FUTURE epochs first with junk keys and<br/>block the owner's later rotations; the code says ZONE_GROUPS_ENABLED must stay off until an owner-only epoch<br/>binding lands nip_handlers.rs:1570-1575"]
```

## NF-08.10 Who writes and who clears the NIP-09 tombstone tables

```mermaid
flowchart TB
    subgraph writers["Writers - tombstone first, DELETE after, never one batch"]
        K5["member kind-5 below TL3 - member_deletion_gate first<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1320-1321, gate nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:3352"]
        K5ANY["admin or TL3+ kind-5 - can_delete_any skips the gate<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1638-1640, process_deletion nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:3513"]
        ENV["sealed original named by a moderator: envelope deleted only when its<br/>author is a CURRENT admin nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:3440-3450"]
        REP["POST /api/reports/resolve as delete, NIP-98 admin<br/>nostr-bbs-relay-worker/src/moderation.rs:240, tombstone nostr-bbs-relay-worker/src/moderation.rs:349"]
        PURGE["POST /api/admin/events/delete, ADR-2017 purge<br/>nostr-bbs-relay-worker/src/admin_events.rs:233, tombstone nostr-bbs-relay-worker/src/admin_events.rs:281"]
        USR["POST /api/admin/user/delete with delete_events<br/>nostr-bbs-relay-worker/src/user_admin.rs:154, tombstone nostr-bbs-relay-worker/src/user_admin.rs:225"]
    end
    TB["deleted_events and deleted_addresses<br/>nostr-bbs-relay-worker/src/lib.rs:746 nostr-bbs-relay-worker/src/lib.rs:755"]
    FN["tombstone_before_delete - returns whether every statement ran, callers never act on it<br/>nostr-bbs-relay-worker/src/lib.rs:1569"]
    ING["ingest refusal after every other gate: blocked, this event was deleted<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1607-1609"]
    RESET["POST /api/admin/reset-db, NIP-98 admin<br/>nostr-bbs-relay-worker/src/whitelist.rs:507, clears both tables nostr-bbs-relay-worker/src/whitelist.rs:524-525"]

    K5 --> FN
    K5ANY --> FN
    K5ANY --> ENV --> FN
    REP --> FN
    PURGE --> FN
    USR --> FN
    FN --> TB --> ING
    RESET -->|"wipes all, markers kept"| TB

    N1["OPEN, NF-10 O15 / KF-14: no route removes ONE tombstone - reset-db is the only way out, and it wipes the events,<br/>whitelist and every tombstone together nostr-bbs-relay-worker/src/whitelist.rs:521-525"]
    N2["OPEN, NF-10 O18 / KF-18: the envelope delete trusts current admin standing, not a migrator record -<br/>a migrator who has lost admin leaves the envelope in place nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:3400-3402"]
    N3["INVARIANT: every writer fails open - a tombstone error neither stops nor fails the deletion<br/>nostr-bbs-relay-worker/src/lib.rs:1564-1568, and the tables are never served to a client"]
    N4["A report resolved as hide records none: only delete tombstones nostr-bbs-relay-worker/src/moderation.rs:318"]
```

Admin authority for every writer above is `require_nip98_admin` (the relay resolver of NF-08.7), except the kind-5 path, which trusts the trust ladder: TL3+ deletes anyone's events without being an admin (`../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1638-1640`). The ingest refusal fails open when its lookup fails, and the purge of a user by `pubkey` leaves device-key and gift-wrap rows untombstoned (NF-10 O17 / KF-16, see NF-25.5).

**Open (NF-10 O13 / KF-17):** `process_deletion` writes an `(id, signer)` tombstone for any well-formed `e` target, stored or not (`../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:3536-3546`), and the member gate then accepts a no-row target as already deleted on exactly such a row (`../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:3058`), so a signer can re-name an id this relay never stored.

**Open (NF-10 O19 / KF-19):** a member cannot delete their own migrated post: the gate classifies a sealed original before anything else, and the original author is not recorded (`../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:2963-2965`, `:3057`). The purge's tombstones under the original author (`../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/admin_events.rs:281`) are therefore never consulted by it.

**Open (NF-10 O12 / KF-21):** the member gate answers up to 2,000 targets with one `json_each` statement (`../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/lib.rs:793-796`); its tests run on rusqlite, never on Cloudflare D1, so D1's behaviour for that statement is unproven.

## NF-08.11 Relay tables added for deletion and key registration — migration-less or flag-gated

```mermaid
flowchart TB
    ENS["relay ensure_schema<br/>nostr-bbs-relay-worker/src/lib.rs:1059"]
    R4["NIP-09 tombstones, ADR-2033: deleted_events nostr-bbs-relay-worker/src/lib.rs:746<br/>deleted_addresses nostr-bbs-relay-worker/src/lib.rs:755, created right after events<br/>nostr-bbs-relay-worker/src/lib.rs:1082-1084, no migration file"]
    R5["Flag-gated, absent on a default deploy: recipient_keys when RECIPIENT_KEYS_ENABLED or<br/>ZONE_GROUPS_ENABLED is exactly true nostr-bbs-relay-worker/src/lib.rs:1129-1139,<br/>zone_epoch_keys when ZONE_GROUPS_ENABLED is nostr-bbs-relay-worker/src/lib.rs:1156-1158"]
    R6["settings one-time markers nip09_tombstones_backfilled nostr-bbs-relay-worker/src/lib.rs:984<br/>and nip09_plain_addresses_rekeyed nostr-bbs-relay-worker/src/lib.rs:981, passes run last<br/>nostr-bbs-relay-worker/src/lib.rs:1423-1424"]
    ENS --> R4
    ENS --> R5
    ENS --> R6
    R4 -.-> R6
    N7["reset-db deletes both tombstone tables but no settings marker, so a reset database never re-runs<br/>the backfill nostr-bbs-relay-worker/src/whitelist.rs:524-525 - see NF-08.10"]
    N5["TENSION: a failed tombstone CREATE logs but does not clear ok, unlike events and whitelist, so the<br/>isolate still latches SCHEMA_READY and runs without the re-insert refusal until a new isolate retries<br/>nostr-bbs-relay-worker/src/lib.rs:1086-1088"]
    N6["recipient_keys is the second cross-worker table, the reverse of device_keys: the RELAY creates it and the<br/>AUTH worker writes it through RELAY_DB, creating nothing nostr-bbs-auth-worker/src/recipient_keys.rs:22-24 - see NF-08.8"]
```

## NF-08.12 Relay D1 tables - who writes and who reads each, across the two workers

```mermaid
flowchart TB
    WL["whitelist - W auth: /api/admins add and remove nostr-bbs-auth-worker/src/admins.rs:162 nostr-bbs-auth-worker/src/admins.rs:240, invite seed and merge<br/>nostr-bbs-auth-worker/src/invites.rs:238 nostr-bbs-auth-worker/src/invites.rs:263, username claim nostr-bbs-auth-worker/src/username.rs:307, agent provision nostr-bbs-auth-worker/src/governance_api.rs:466;<br/>relay: grant nostr-bbs-relay-worker/src/whitelist.rs:373, set-admin nostr-bbs-relay-worker/src/whitelist.rs:450, cohort edit nostr-bbs-relay-worker/src/whitelist.rs:593, delete nostr-bbs-relay-worker/src/user_admin.rs:199<br/>R relay admission nostr-bbs-relay-worker/src/relay_do/storage.rs:445, relay admin nostr-bbs-relay-worker/src/auth.rs:194, check-whitelist nostr-bbs-relay-worker/src/whitelist.rs:135,<br/>auth is_admin nostr-bbs-auth-worker/src/admin.rs:72, admin list nostr-bbs-auth-worker/src/admins.rs:75, device owner check nostr-bbs-auth-worker/src/devices.rs:281, pod admin nostr-bbs-pod-worker/src/lib.rs:2022"]
    AR["agent_registry - W auth register nostr-bbs-auth-worker/src/governance_api.rs:381, provision nostr-bbs-auth-worker/src/governance_api.rs:476, revoke nostr-bbs-auth-worker/src/governance_api.rs:532<br/>R auth list nostr-bbs-auth-worker/src/governance_api.rs:347 and is_registered_agent nostr-bbs-auth-worker/src/governance_api.rs:1102, relay nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:3801, relay disclosure nostr-bbs-relay-worker/src/agent_disclosure.rs:105"]
    BC["broker_cases - W relay: case upsert on kind-31402 nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1033 - projection of kind-31403 nostr-bbs-relay-worker/src/relay_do/receipts.rs:192, supersession nostr-bbs-relay-worker/src/relay_do/receipts.rs:244,<br/>appeal reopen nostr-bbs-relay-worker/src/relay_do/receipts.rs:283, cron ageing nostr-bbs-relay-worker/src/cron.rs:844<br/>R auth case list and read nostr-bbs-auth-worker/src/governance_api.rs:572 nostr-bbs-auth-worker/src/governance_api.rs:600, relay nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:3996, cron nostr-bbs-relay-worker/src/cron.rs:622"]
    BD["broker_decisions - W relay projection nostr-bbs-relay-worker/src/relay_do/receipts.rs:185, supersession nostr-bbs-relay-worker/src/relay_do/receipts.rs:234 nostr-bbs-relay-worker/src/relay_do/receipts.rs:242<br/>R auth GET /api/governance/decisions nostr-bbs-auth-worker/src/governance_api.rs:619 nostr-bbs-auth-worker/src/governance_api.rs:646, relay nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:4194"]
    BR["broker_roles - W auth grant nostr-bbs-auth-worker/src/governance_api.rs:693, revoke nostr-bbs-auth-worker/src/governance_api.rs:757<br/>R auth list nostr-bbs-auth-worker/src/governance_api.rs:724, relay is_reviewer nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:4013 and governance_rank nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:4330"]
    DK["device_keys - W auth register nostr-bbs-auth-worker/src/devices.rs:331, revoke nostr-bbs-auth-worker/src/devices.rs:526<br/>R auth list nostr-bbs-auth-worker/src/devices.rs:478; relay device_owner nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:3764 through effective_pubkey nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:3782<br/>on EVENT nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1162 and REQ viewer context nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:2277 - never at NIP-42 AUTH"]
    MA["moderation_actions, RELAY copy - W the admin kind-30910/30911 mirror nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:4855<br/>R the ban gate nostr-bbs-relay-worker/src/relay_do/mod_cache.rs:117"]
    RP["reports - W relay kind-1984 projection nostr-bbs-relay-worker/src/moderation.rs:75 from nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:4799"]
    WL --> AR --> BC --> BD
    BR --> DK --> MA --> RP
```

## NF-08.13 Auth D1 tables - who writes and who reads each

> **Superseded** by NF-08.14 at 94199e5 (frozen at 4a73a7a).

```mermaid
flowchart TB
    RPL["nip98_replay - all four NIP-98 verifiers, INSERT OR IGNORE nostr-bbs-rate-limit/src/replay.rs:13: auth via DB nostr-bbs-auth-worker/src/auth.rs:11,<br/>relay nostr-bbs-relay-worker/src/auth.rs:31, search nostr-bbs-search-worker/src/auth.rs:31 and pod nostr-bbs-pod-worker/src/auth.rs:10 via REPLAY_DB - one table, so replay is cross-worker"]
    MEM["members - W invite redemption nostr-bbs-auth-worker/src/invites.rs:672 nostr-bbs-auth-worker/src/invites.rs:878, /api/admins nostr-bbs-auth-worker/src/admins.rs:182 nostr-bbs-auth-worker/src/admins.rs:251,<br/>delete-member nostr-bbs-auth-worker/src/admins.rs:310. R the invite flow only nostr-bbs-auth-worker/src/invites.rs:134. NOT read by any admin resolver; the shared<br/>members SQL constants have no caller in the workspace nostr-bbs-core/src/admin_shared.rs:76 nostr-bbs-core/src/admin_shared.rs:82"]
    MOD["moderation_actions, AUTH copy - W /api/mod/ban and siblings nostr-bbs-auth-worker/src/moderation.rs:257, R /api/mod/actions nostr-bbs-auth-worker/src/moderation.rs:415.<br/>A different table from the relay copy the ban gate reads - NF-08.12"]
    MRP["mod_reports - W /api/mod/report nostr-bbs-auth-worker/src/moderation.rs:342, action nostr-bbs-auth-worker/src/moderation.rs:589<br/>R /api/mod/reports nostr-bbs-auth-worker/src/moderation.rs:501 nostr-bbs-auth-worker/src/moderation.rs:572"]
    WOT["wot_entries - W wot.rs:320 wot.rs:389, deletes wot.rs:312 wot.rs:401 - R wot.rs:165 wot.rs:462"]
    INST["instance_settings - W seed schema.rs:203, welcome welcome.rs:226 welcome.rs:279, wot wot.rs:220 wot.rs:331<br/>R welcome.rs:82 wot.rs:117 nostr-bbs-auth-worker/src/invites.rs:116"]
    USR["username_reservations - W claim username.rs:237, release username.rs:376, real name username.rs:435, dismiss username.rs:762,<br/>delete-member nostr-bbs-auth-worker/src/admins.rs:315 - R username.rs:191 username.rs:344 username.rs:466 username.rs:492"]
    NIP["nip1984_reports - created schema.rs:51 and read by GET /api/moderation/reports nostr-bbs-auth-worker/src/moderation.rs:651,<br/>but NO code writes it: the relay keeps kind-1984 in its own reports table - DRIFT against nostr-bbs-auth-worker/src/moderation.rs:631-632"]
    CH["challenges - W webauthn.rs:803 webauthn.rs:1203, deletes webauthn.rs:800 webauthn.rs:1094 webauthn.rs:1467 - R webauthn.rs:719 webauthn.rs:1350"]
    CRED["webauthn_credentials - W webauthn.rs:1081, counter webauthn.rs:1464<br/>R webauthn.rs:1042 webauthn.rs:1168 webauthn.rs:1273 webauthn.rs:1496 nostr-bbs-auth-worker/src/did.rs:49"]
    RPL --> MEM --> MOD --> MRP --> WOT
    INST --> USR --> NIP --> CH --> CRED
```

## NF-08.14 Auth D1 tables, after the dead report table was dropped - who writes and who reads each

```mermaid
flowchart TB
    RPL["nip98_replay - all four NIP-98 verifiers, INSERT OR IGNORE nostr-bbs-rate-limit/src/replay.rs:13: auth via DB nostr-bbs-auth-worker/src/auth.rs:11,<br/>relay nostr-bbs-relay-worker/src/auth.rs:31, search nostr-bbs-search-worker/src/auth.rs:31 and pod nostr-bbs-pod-worker/src/auth.rs:10 via REPLAY_DB - one table, so replay is cross-worker"]
    MEM["members - W invite redemption nostr-bbs-auth-worker/src/invites.rs:672 nostr-bbs-auth-worker/src/invites.rs:901, /api/admins nostr-bbs-auth-worker/src/admins.rs:182 nostr-bbs-auth-worker/src/admins.rs:251,<br/>delete-member nostr-bbs-auth-worker/src/admins.rs:310. R the invite flow only nostr-bbs-auth-worker/src/invites.rs:134. NOT read by any admin resolver; the shared<br/>members SQL constants have no caller in the workspace nostr-bbs-core/src/admin_shared.rs:76 nostr-bbs-core/src/admin_shared.rs:82"]
    MOD["moderation_actions, AUTH copy - W /api/mod/ban and siblings nostr-bbs-auth-worker/src/moderation.rs:257, R /api/mod/actions nostr-bbs-auth-worker/src/moderation.rs:415.<br/>A different table from the relay copy the ban gate reads - NF-08.12"]
    MRP["mod_reports - W /api/mod/report nostr-bbs-auth-worker/src/moderation.rs:342, action nostr-bbs-auth-worker/src/moderation.rs:589<br/>R /api/mod/reports nostr-bbs-auth-worker/src/moderation.rs:501 nostr-bbs-auth-worker/src/moderation.rs:572"]
    WOT["wot_entries - W wot.rs:320 wot.rs:389, deletes wot.rs:312 wot.rs:401 - R wot.rs:165 wot.rs:462"]
    INST["instance_settings - W seed schema.rs:183, welcome welcome.rs:226 welcome.rs:279, wot wot.rs:220 wot.rs:331<br/>R welcome.rs:82 wot.rs:117 nostr-bbs-auth-worker/src/invites.rs:116"]
    USR["username_reservations - W claim username.rs:237, release username.rs:376, real name username.rs:435, dismiss username.rs:762,<br/>delete-member nostr-bbs-auth-worker/src/admins.rs:315 - R username.rs:191 username.rs:344 username.rs:466 username.rs:492"]
    NIP["reports (the relay's D1) - W relay kind-1984 projection nostr-bbs-relay-worker/src/moderation.rs:75, R auth GET /api/moderation/reports through RELAY_DB nostr-bbs-auth-worker/src/moderation.rs:632 nostr-bbs-auth-worker/src/moderation.rs:675.<br/>The auth bootstrap no longer creates nip1984_reports nostr-bbs-auth-worker/src/schema.rs:46, and migration 0005 drops it nostr-bbs-auth-worker/migrations/0005_drop_nip1984_reports.sql:10"]
    CH["challenges - W webauthn.rs:803 webauthn.rs:1203, deletes webauthn.rs:800 webauthn.rs:1094 webauthn.rs:1467 - R webauthn.rs:719 webauthn.rs:1350"]
    CRED["webauthn_credentials - W webauthn.rs:1081, counter webauthn.rs:1464<br/>R webauthn.rs:1042 webauthn.rs:1168 webauthn.rs:1273 webauthn.rs:1496 nostr-bbs-auth-worker/src/did.rs:49"]
    RPL --> MEM --> MOD --> MRP --> WOT
    INST --> USR --> NIP --> CH --> CRED
```

## NF-08.15 Auth D1 bootstrap — the table list without the dead report table

```mermaid
flowchart TB
    subgraph authd1["nostr-bbs-auth D1 - bootstrapped by nostr-bbs-auth-worker/src/schema.rs:19"]
        A1["challenges schema.rs:28<br/>webauthn_credentials schema.rs:33<br/>moderation_actions schema.rs:51"]
        A2["mod_reports schema.rs:63<br/>wot_entries schema.rs:84<br/>members schema.rs:97<br/>invitations schema.rs:104"]
        A3["invitation_redemptions schema.rs:117<br/>welcome_messages schema.rs:145<br/>instance_settings schema.rs:163<br/>username_reservations schema.rs:189"]
    end
    GONE["nip1984_reports is not created: only a comment remains<br/>nostr-bbs-auth-worker/src/schema.rs:46<br/>and migration 0005 drops the table and its index<br/>nostr-bbs-auth-worker/migrations/0005_drop_nip1984_reports.sql:10"]
    RPT["kind-1984 reports live in the relay's reports table<br/>nostr-bbs-relay-worker/src/lib.rs:1255"]
    A1 --> A2 --> A3
    A3 -.-> GONE
    GONE -.-> RPT
    N["INVARIANT: the auth worker has two report tables of its own kind, mod_reports for kind-30913<br/>nostr-bbs-auth-worker/src/schema.rs:63, and reads the relay's reports through RELAY_DB, never a copy<br/>nostr-bbs-auth-worker/src/moderation.rs:675"]
    RPT -.-> N
```

**What it shows:** the auth worker's bootstrap now creates the tables above, one fewer than NF-08.5 listed. The relay D1 half of NF-08.5 is unchanged except for the `reports` line shown here.
**Why it is this way:** commit 94199e5 removed a table nothing wrote; NF-08.14 gives the per-table writers and readers. Read from source at 8a0ca41, nothing executed.

