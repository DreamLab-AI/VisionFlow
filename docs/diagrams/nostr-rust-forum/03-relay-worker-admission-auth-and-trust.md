---
id: NF-03
title: relay-worker — NIP-42 AUTH, the EVENT admission pipeline, trust ladder and federation
area: nostr-rust-forum
governing:
  - ../nostr-rust-forum/docs/BASELINE-architecture.md
  - ../nostr-rust-forum/docs/IDENTITY-keys-and-trust.md
adrs: [ADR-2004, ADR-2005, ADR-2006, ADR-2010, ADR-2011, ADR-2016, ADR-2017, ADR-2018, ADR-2022, ADR-2023]
sources:
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust_sweep.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/lib.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip42.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/storage.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/migrations/0009_events.sql
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/zone_config.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/wrangler.toml
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/governance.rs
  - ../nostr-rust-forum/README.md
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/ingress.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/mesh.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/push_fanout.rs
  - ../nostr-rust-forum/docs/adr/ADR-2010-durable-governance-outcome-receipts.md
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/Cargo.toml
verified_commit: 6064622dbb1a7cc1ba2d072109f9e3e8a96157df
---

## NF-03.1 Worker entry — three doors into the relay

```mermaid
flowchart TB
    REQ["HTTP or WebSocket request"]
    F["fetch<br/>nostr-bbs-relay-worker/src/lib.rs:209"]
    BOOT["ensure_schema + ensure_replay_schema, once per isolate: SCHEMA_READY latches only after a pass with no failed CREATE<br/>nostr-bbs-relay-worker/src/lib.rs:222 lib.rs:223 lib.rs:224 latch lib.rs:200-203"]
    WS["Upgrade: websocket to Durable Object RELAY<br/>nostr-bbs-relay-worker/src/lib.rs:228"]
    N11["Accept application/nostr+json to NIP-11 relay info<br/>nostr-bbs-relay-worker/src/lib.rs:237"]
    RT["route dispatches the HTTP admin/REST route table<br/>nostr-bbs-relay-worker/src/lib.rs:253 lib.rs:286"]
    CRON["scheduled every 5 min<br/>nostr-bbs-relay-worker/src/lib.rs:1146<br/>nostr-bbs-relay-worker/wrangler.toml:123"]

    REQ --> F --> BOOT
    BOOT --> WS
    BOOT --> N11
    BOOT --> RT

    N1["The DO is a SINGLETON: every socket goes to get_by_name main<br/>nostr-bbs-relay-worker/src/lib.rs:230"]
    N2["Error responses are JSON-shaped by hand rather than leaking the framework's Debug text<br/>nostr-bbs-relay-worker/src/lib.rs:256-267"]
    N3["The cron does NOT pass through the bootstrap: neither scheduled nor the Durable Object creates the schema,<br/>which is why fetch bootstraps BEFORE the WebSocket branch nostr-bbs-relay-worker/src/lib.rs:218-221; scheduled goes straight to the<br/>demotion sweep nostr-bbs-relay-worker/src/lib.rs:1157"]
```

## NF-03.2 NIP-42 AUTH — the verdict table

```mermaid
sequenceDiagram
    autonumber
    participant C as Client socket
    participant DO as NostrRelayDO
    participant EV as evaluate_auth_event<br/>nostr-bbs-relay-worker/src/relay_do/nip42.rs:176

    DO-->>C: AUTH challenge (per-session, unpredictable)
    C->>DO: ["AUTH", kind-22242 event]
    DO->>EV: event, expected_challenge, own_relay_url, now, max_skew
    EV->>EV: kind must be 22242 nip42.rs:183
    EV->>EV: verify_event_strict - id + Schnorr nip42.rs:187
    EV->>EV: challenge tag must equal THIS session's issued value nip42.rs:193
    EV->>EV: relay tag must name THIS relay, canonicalised nip42.rs:203
    EV->>EV: created_at within +/- 600 s nip42.rs:212 nip42.rs:35
    EV-->>DO: AuthVerdict::Ok(pubkey) nip42.rs:216

    Note over EV: A MISSING session challenge can never match, so it rejects - there is no unchallenged path nip42.rs:191-196
    Note over EV: own_relay_url unset or blank SKIPS the relay-tag check only. This fails OPEN on the defence-in-depth check and never on the challenge or signature, so the gate can roll out before RELAY_URL is configured nip42.rs:168-172 - RELAY_URL ships blank nostr-bbs-relay-worker/wrangler.toml:51, and handle_auth now logs a console error on every AUTH while it is blank nip42.rs:230 nip_handlers.rs:2054
    Note over EV: canonical_relay_url absorbs scheme, case and a trailing slash but keeps host identity nip42.rs:144, asserted nip42.rs:332-343
```

## NF-03.3 AUTH_MODE — the write gate and the protected-read set

```mermaid
stateDiagram-v2
    [*] --> Nip42: parse_auth_mode default<br/>nostr-bbs-relay-worker/src/relay_do/nip42.rs:123
    [*] --> Allowlist: AUTH_MODE == "allowlist" exactly<br/>nostr-bbs-relay-worker/src/relay_do/nip42.rs:125
    Nip42 --> WriteDenied: unauthenticated EVENT<br/>"auth-required: NIP-42 AUTH required to publish"<br/>nostr-bbs-relay-worker/src/relay_do/nip42.rs:248
    Nip42 --> ReadDenied: REQ/COUNT naming a protected kind<br/>nostr-bbs-relay-worker/src/relay_do/nip42.rs:261
    Allowlist --> WriteAllowed: legacy - allowlist alone gates<br/>nostr-bbs-relay-worker/src/relay_do/nip42.rs:249

    note right of Nip42
        Protected read set nip42.rs:44
        4 encrypted DM, 13 seal, 14 private DM,
        1059 gift wrap, 30910-30916 moderation
        The filter gate reads only the kinds a filter
        NAMES, so a kindless filter passes it,
        asserted nip42.rs:392-393. The PER-EVENT gate
        catches that case - see NF-03.14
    end note
    note right of Allowlist
        README.md:130 now agrees with the code: NIP-42 relay admission is "Implemented;
        example worker selects AUTH_MODE=nip42". The earlier "scaffolded" ledger row is gone.
        nip42 IS the default (nip42.rs:115), the template ships AUTH_MODE = "nip42"
        (nostr-bbs-relay-worker/wrangler.toml:39), and every EVENT passes write_auth_ok
        (nip_handlers.rs:1102) after the rate, shape, expiry and signature
        checks and before any admission lookup - see NF-03.4. A denial AFTER a completed AUTH is
        "restricted:", while legacy allowlist mode keeps "blocked:" (nip42.rs:282)
    end note
    note right of WriteDenied
        INVARIANT fail-closed parse: a typo, an empty value or any unrecognised
        string resolves to nip42, never to allowlist - asserted nip42.rs:322-327
    end note
```

## NF-03.4 EVENT admission pipeline — every gate, in order

```mermaid
sequenceDiagram
    autonumber
    participant C as Authenticated socket
    participant H as handle_event<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1049
    participant Z as zone_config<br/>nostr-bbs-relay-worker/src/zone_config.rs:208
    participant D1 as relay D1
    participant B as broadcast_event

    C->>H: ["EVENT", event]
    H->>H: 1 per-IP rate limit nip_handlers.rs:1057
    H->>H: 2 structural validation nip_handlers.rs:1063
    H->>H: 3 NIP-40 expiration tag in the past, or malformed nip_handlers.rs:1070
    H->>H: 4 verify_event_strict BEFORE any side effect nip_handlers.rs:1077
    H->>H: 5 NIP-42 write gate write_auth_ok nip_handlers.rs:1102
    H->>D1: 6 resolve the effective principal ONCE, never for a gift wrap (ADR-099 device key to owner) nip_handlers.rs:1121
    alt kind 1059 gift wrap
        H->>D1: 7a recipient from first p tag: a configured ingress recipient (ADR-2022) needs no row, any other must be whitelisted nip_handlers.rs:1139 nip_handlers.rs:1142
    else every other kind
        H->>D1: 7b is_whitelisted on the principal nip_handlers.rs:1161
    end
    H->>H: 8 mesh peer? federated_kinds allowlist nip_handlers.rs:1176
    H->>D1: 9 suspension and silence, checked for the device key AND its owner nip_handlers.rs:1189
    H->>D1: 10 ban/mute ingress gate, admins bypass nip_handlers.rs:1213 nip_handlers.rs:1217
    H->>D1: 11 trust-level gates for 40, 41, 1984 and 5 nip_handlers.rs:1228
    H->>H: 12 NIP-29 admin kinds need an h tag and an admin nip_handlers.rs:1299
    H->>D1: 13 governance kinds need agent_registry, except a 31403 (gated at step 14) and a kanban approval request nip_handlers.rs:1324 nip_handlers.rs:1325 nip_handlers.rs:1326
    H->>D1: 14 kind-31403 is admin, or a reviewer delegated to THIS case, then the FR2.2 rationale check nip_handlers.rs:1344 nip_handlers.rs:1354 nip_handlers.rs:1372
    H->>D1: 15 a superseding 31403 is authority-gated nip_handlers.rs:1390 nip_handlers.rs:1396
    H->>H: 16 an appeal-marked 31402 must name its case in a case tag distinct from d nip_handlers.rs:1407
    H->>Z: 17a ADR-2017 kind-42 sealed tag: admin-only AND an encrypted zone, else rejected nip_handlers.rs:1440 nip_handlers.rs:1445
    H->>D1: 17b kind-42 zone write gate nip_handlers.rs:1451 nip_handlers.rs:1452
    H->>Z: 17c ADR-2016 encrypted zone requires NIP-44 v2 shaped ciphertext nip_handlers.rs:1458 nip_handlers.rs:1464
    H->>D1: 18 NIP-52 RSVP and calendar/kanban write gates nip_handlers.rs:1490 nip_handlers.rs:1515
    H->>H: 19 NIP-16 treatment - Ephemeral is OK-then-broadcast, never saved nip_handlers.rs:1540
    H->>D1: 20 push_event_known, then save_event nip_handlers.rs:1549 nip_handlers.rs:1552
    H->>B: 21 broadcast + post-save effects, incl. ADR-2018 activity coalescing nip_handlers.rs:1554 nip_handlers.rs:1564

    Note over H: INVARIANT: the signature is verified BEFORE any side effect - admission state changes and activity tracking alike nip_handlers.rs:1075-1077
    Note over H: In nip42 mode gift wraps are NOT exempt from the write gate - the wrap is signed by an ephemeral key but is published over the sender's authenticated socket nip_handlers.rs:1087-1094
    Note over H: INVARIANT FR2.2 is enforced at the RELAY, not only in the UI - a 31403 is a signed event any client or script can publish straight here, so a rule that lives only in the forum UI is a suggestion nip_handlers.rs:1360-1366 governance.rs:758
    Note over H: A refusal never fills the rationale in - absence is refused, not papered over governance.rs:761-763
    Note over Z: INVARIANT ADR-2017: a sealed original is exempt from validate_event's timestamp-drift check on the tag alone, so this write gate is what actually enforces admin-only AND encrypted-zone-only for it nip_handlers.rs:1703-1711 nip_handlers.rs:1748-1760
```

## NF-03.5 Gift-wrap admission — recipient-keyed, never author-keyed

```mermaid
flowchart LR
    EV["kind-1059 event, ephemeral author"]
    EX["gift_wrap_recipient<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:138"]
    K["kind must be 1059<br/>nip_handlers.rs:139"]
    T["first non-empty p tag<br/>nip_handlers.rs:142"]
    ING["configured ingress recipient (INGRESS_RECIPIENTS binding): admitted with NO D1 row, so a reset-db cannot silence it<br/>nostr-bbs-relay-worker/src/ingress.rs:27 nip_handlers.rs:1139 nip_handlers.rs:1141"]
    WL["is_whitelisted(recipient)<br/>nip_handlers.rs:1142"]
    OK["accepted"]
    NO["blocked: gift-wrap recipient not whitelisted<br/>nip_handlers.rs:1150"]

    EV --> EX --> K --> T --> ING
    ING -->|"declared"| OK
    ING -->|"not declared"| WL
    WL -->|"member"| OK
    WL -->|"not a member, or no p tag"| NO

    N1["INVARIANT ADR-2005: the ephemeral author must NEVER be the admission principal - an author-keyed<br/>check would reject every gift wrap nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1128-1134"]
    N2["Gift wraps are DELIBERATELY excluded from ban gating: each is signed by a throwaway key so an<br/>author-keyed ban cannot bind. The recipient-whitelist gate is what bounds them instead<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:56-65, exclusion visible in nip_handlers.rs:69-75"]
    N4["INVARIANT ADR-2022: the ingress list only widens who a wrap may be ADDRESSED to - it grants no cohort, admin right or zone key, and it is consulted only inside the kind-1059 branch, so events an ingress key authors are checked against the whitelist as before<br/>nostr-bbs-relay-worker/src/ingress.rs:9-12"]
    N3["EXTERNAL: agentbox mirrors sessions as NIP-59 gift-wrapped self-DMs to a different relay - see AB-13"]
```

## NF-03.6 Ban gating — which kinds a banned author may not publish

```mermaid
flowchart TB
    GATE["is_ban_gated_kind<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:68"]
    LIT["1 note, 5 deletion, 7 reaction, 40 channel create,<br/>41 channel metadata, 42 message, 30023 long-form<br/>nip_handlers.rs:69"]
    REP["1984 NIP-56 report nip_handlers.rs:70"]
    CAL["calendar date/plain events, 31925 RSVP<br/>nip_handlers.rs:71 nip_handlers.rs:72 nip_handlers.rs:73"]
    KAN["kanban kinds + agent-intent<br/>nip_handlers.rs:74 nip_handlers.rs:75"]
    CALL["call site: admins bypass, then ModCache 60 s check<br/>nip_handlers.rs:1213"]

    GATE --> LIT & REP & CAL & KAN
    GATE --> CALL

    N1["WI-2 history: the gate previously fired only for kinds 1 and 42, so a banned user could still<br/>publish reactions, deletions, reports, long-form articles, channel create/metadata, calendar<br/>events/RSVPs, kanban and agent-intent nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:56-65"]
    N2["Admins bypass at the call site so they can publish warnings while themselves under moderation<br/>for something else nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1214"]
    N3["Pure predicate by design, so the gate's SCOPE is unit-testable without an Env<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:67"]
```

## NF-03.7 Trust-level gates on the admission path

```mermaid
flowchart TB
    ADMIN{"admin_cache.is_admin<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1226"}
    TL["get_trust_level<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1228"]
    K40["kind 40 channel creation<br/>TL2+ nip_handlers.rs:1231"]
    K41["kind 41 channel metadata<br/>TL2+ for own, TL3+ for any<br/>nip_handlers.rs:1247 nip_handlers.rs:1257"]
    K1984["kind 1984 report<br/>TL1+ nip_handlers.rs:1273"]
    K5["kind 5 deletion<br/>own always, others TL3+<br/>nip_handlers.rs:1284 nip_handlers.rs:1286"]
    IC["TL2 branch only: is_channel_creator, matched against the signing key AND its device owner<br/>nip_handlers.rs:1259 defined nip_handlers.rs:2536"]
    CC["1. channel_creators row for the channel id<br/>nip_handlers.rs:2548"]
    FB["2. fallback when no row: the kind-40 event itself<br/>nip_handlers.rs:2549"]
    ANS["first row found decides: creator_matches against the sanction subjects (key plus owner)<br/>nip_handlers.rs:2555"]
    SAVE["save_event, when a kind 40 was stored<br/>relay_do/storage.rs:159"]
    INS["INSERT INTO channel_creators, first write wins<br/>relay_do/storage.rs:162 relay_do/storage.rs:163"]
    DDL["table created by ensure_schema from CHANNEL_CREATORS_CREATE_SQL<br/>nostr-bbs-relay-worker/src/lib.rs:732 lib.rs:827<br/>and by migrations/0009_events.sql:44"]
    DEL["kind-5 process_deletion removes only the kind-40 events row<br/>nip_handlers.rs:2457"]

    ADMIN -->|"admin: skip the whole block"| SKIP["gates bypassed"]
    ADMIN -->|"non-admin"| TL --> K40 & K41 & K1984 & K5
    K41 --> IC --> CC -->|"no row"| FB
    CC --> ANS
    FB --> ANS
    K40 -.->|"accepted and stored"| SAVE --> INS --> DDL
    DEL -.->|"leaves untouched"| CC

    N1["kind 41 without an e tag is rejected as invalid before the trust check<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1243-1244"]
    N2["ANOMALY O5 CLOSED at 66ca8d8: the creator used to be re-read from the kind-40 event, so a kind-5<br/>deletion of it locked a TL2 author out of their own channel. The creator is now persisted at creation in<br/>channel_creators and read first, so the deletion no longer changes the answer<br/>nip_handlers.rs:2548 storage.rs:162"]
    N3["The kind-5 privilege is re-derived after save because trust_level is scoped to this non-admin<br/>block that admins skip entirely nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1571-1573"]
    N4["DEBT: the creator INSERT result is discarded (let _ =) and the code comment says failures never block<br/>storage, so a channel whose write failed has no row and is still on the deletion-exposed fallback<br/>relay_do/storage.rs:171 nip_handlers.rs:2549. Channels created before the table existed have no row either<br/>and are likewise exposed, nip_handlers.rs:2534-2535"]
    N5["INVARIANT: the answer comes from the first of the two lookups that returns a row, so once a<br/>channel_creators row exists the kind-40 event is never consulted again<br/>nip_handlers.rs:2555"]
```

## NF-03.8 The trust ladder — promotion, hysteresis, and what is never demoted

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust.rs:28
    TL0["TL0 Newcomer"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust.rs:183
    TL1["TL1 Member<br/>3 days, 10 reads, 1 post"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust.rs:174
    TL2["TL2 Regular<br/>14 days, 50 reads, 10 posts<br/>no moderation actions"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust.rs:33
    TL3["TL3 Trusted<br/>administrative grant only"]
    TL0 -->|earns TL1| TL1
    TL0 -->|earns TL2 directly| TL2
    TL1 -->|earns TL2| TL2
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust.rs:349
    G["Demotion requires inactivity<br/>and broken hysteresis band<br/>TL0, TL3 and admins excluded"]
    G -->|TL2 still earns TL1| TL1
    G -->|TL1 or TL2 no longer earns TL1| TL0
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust.rs:84
    H["Default band is 90 percent<br/>after about six months idle<br/>One transition need not mean one rung"]
    H -.-> G
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust.rs:238
    R["Promotion checks a snapshot<br/>but its unguarded write can overwrite<br/>a concurrent TL3 grant - NF-24.3"]
    TL3 -.-> R
```

## NF-03.9 Why a demotion did NOT happen — the named hold reasons

```mermaid
classDiagram
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust.rs:285
    class HoldReason {
        AdminGrantedLevel
        ExemptAdminRow
        AtFloor
        WithinActivityWindow
        ClearsHysteresis
    }
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust.rs:308
    class DemotionDecision {
        Hold with reason
        Demote from to
    }
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust.rs:326
    class decide_demotion {
        pure policy
        returns decision
    }
    decide_demotion --> DemotionDecision
    DemotionDecision --> HoldReason
    note for HoldReason "Named reasons exist in policy.<br/>Sweep retains only their total<br/>trust_sweep.rs:297."
    note for decide_demotion "Scheduled demotion only.<br/>Per-member entry point removed<br/>trust.rs:406-419."
    note for DemotionDecision "Conditional audit is repaired<br/>trust_sweep.rs:441.<br/>Capped fairness remains open<br/>See NF-24.2."
```

## NF-03.10 Post-save effects — activity coalescing, moderation mirror, governance projection

```mermaid
sequenceDiagram
    autonumber
    participant H as handle_event after save<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1552
    participant A as ActivityLedger<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:2238
    participant T as trust
    participant M as moderation_actions
    participant G as broker_cases / broker_decisions
    participant P as push_fanout<br/>nostr-bbs-relay-worker/src/relay_do/push_fanout.rs:47

    H->>H: OK then broadcast nip_handlers.rs:1553 nip_handlers.rs:1554
    H->>T: increment_posts_created for kinds 1, 7, 40, 42, 1984 nip_handlers.rs:1558
    H->>A: note_write_activity - ADR-2018 coalesced flush<br/>nip_handlers.rs:1564 nip_handlers.rs:2238
    Note over H,T: Save success also includes duplicate IDs - this counter is not exact-once<br/>nip_handlers.rs:1546-1564 — see NF-17.6
    alt kind 5
        H->>H: process_deletion, can_delete_any = admin or TL3+ nip_handlers.rs:1572
    end
    alt kind 1984
        H->>H: process_report, auto-hide check nip_handlers.rs:1579
    end
    alt kinds 30910 / 30911 / 30915 / 30916 from an ADMIN
        H->>M: mirror_moderation_action + mod_cache.invalidate nip_handlers.rs:1588
    end
    alt kind 31402
        H->>G: project_appeal if an appeal e-tag, else project_action_request nip_handlers.rs:1601 nip_handlers.rs:1610
    end
    alt kind 31403
        H->>G: project_supersession if a supersedes e-tag, else project_action_response nip_handlers.rs:1621 nip_handlers.rs:1634
    end

    H->>P: ADR-2023 content-free Web Push wake-up, last, only for kinds 40 / 42 / 1059 and not for a repeat submission nip_handlers.rs:1549 nip_handlers.rs:1655

    Note over P: ADR-2023 is staged: PUSH_ENABLED ships "false"<br/>nostr-bbs-relay-worker/wrangler.toml:86<br/>Disabled fan-out returns at once push_fanout.rs:48<br/>Candidates use the shared read gate push_fanout.rs:94
    Note over H,A: ADR-2010: the relay OK certifies storage only.<br/>Without projection-committed, the response is accepted<br/>but not applied - reported separately nip_handlers.rs:1630-1642
    Note over A,T: Reads and writes share this ledger.<br/>A later activity call triggers an elapsed-window flush.<br/>Pending-read threshold can flush sooner - see NF-17.4<br/>nip_handlers.rs:2228 nip_handlers.rs:2246-2248
    Note over M: Mirror requires an admin on this relay.<br/>A lifted ban must stop enforcement nip_handlers.rs:1588
    Note over G: External human-approval workflow<br/>spans the estate - see ES-05 and AB-14
```

**Tension (ADR-2010 vs code):** ADR-2010 records `decision_status: proposed`, `implementation_status: partial`, `activation_status: staged` (`../nostr-rust-forum/docs/adr/ADR-2010-durable-governance-outcome-receipts.md:5-7`), yet `handle_event` runs all three receipt-bearing projections (appeal, supersession, first-order response) unconditionally after every stored governance event (`nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1601`, `nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1621`, `nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1634`).

## NF-03.11 Structural validation limits

```mermaid
flowchart LR
    V["validate_event<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1662"]
    L1["id 64, pubkey 64, sig 128 hex chars<br/>nip_handlers.rs:1663"]
    L2["content cap - registration kinds 0 and 9024 get 8 KiB<br/>nip_handlers.rs:1667 nip_handlers.rs:1673 nip_handlers.rs:41"]
    L3["max 2000 tags nip_handlers.rs:1677 nip_handlers.rs:42"]
    L4["max 1024 bytes per tag value nip_handlers.rs:1682 nip_handlers.rs:43"]
    L5["created_at drift cap 7 days, sealed originals exempt from the PAST bound only (ADR-2017) nip_handlers.rs:1688 nip_handlers.rs:44 nip_handlers.rs:1728"]
    L6["max 20 subscriptions per socket nip_handlers.rs:45 nip_handlers.rs:1796"]

    V --> L1 & L2 & L3 & L4 & L5
    V -.-> L6
```

## NF-03.12 Zone enforcement on writes — sealed originals, encrypted-zone ciphertext, plain writes

```mermaid
flowchart TB
    K42["kind 42 channel message<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1414"]
    ETAG["e tag required, else invalid<br/>nip_handlers.rs:1416"]
    CZ["cached_channel_zone - memoised channel_zones lookup<br/>nip_handlers.rs:1435 nip_handlers.rs:2189"]
    SEALED{"has_sealed_tag<br/>nip_handlers.rs:1440"}
    SREJ["sealed_write_rejection: non-admin -> admin-only; zone not encrypted -> encrypted-zones-only (ADR-2017)<br/>nip_handlers.rs:1445 nip_handlers.rs:1748-1760"]
    UNSCOPED["no row: UNSCOPED, any whitelisted member may post"]
    WGATE["has_zone_write_access<br/>nip_handlers.rs:1451"]
    CIPHER["is_encrypted zone requires is_zone_ciphertext - zk tag + NIP-44 v2 shape (ADR-2016)<br/>nip_handlers.rs:1458 zone_config.rs:133 zone_config.rs:208"]
    CAL["31922 / 31923 / kanban: zone tag then write cohorts<br/>nip_handlers.rs:1515 nip_handlers.rs:1526"]
    RSVP["31925 RSVP: author's projection tier for the TARGET must be Full<br/>nip_handlers.rs:1490 nip_handlers.rs:1491 nip_handlers.rs:1505"]

    K42 --> ETAG --> CZ
    CZ --> SEALED
    SEALED -->|"sealed"| SREJ
    CZ -->|"present, not sealed or sealed+admin+encrypted"| WGATE
    WGATE --> CIPHER
    CAL --> WGATE

    N1["Finding-5 fix: an undeclared channel previously defaulted to a home zone that the shipped four-zone<br/>model does not define, so the write gate denied ALL non-admins - a permanent lockout of every<br/>non-admin-created channel, including the creator's own nip_handlers.rs:1425-1434"]
    N2["INVARIANT: channel_zones rows are written SOLELY by the admin-only /api/admin/channel-zone endpoint<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1419-1423"]
    N3["The RSVP target is resolved from D1, NEVER from an author-mirrored tag which would be spoofable;<br/>an unresolvable target denies for non-admins nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1480-1485, deny at nip_handlers.rs:1512"]
    N4["Writes route through write_cohorts ?? required_cohorts, so a public zone can be read-by-all yet<br/>write-restricted - see NF-08.2"]
    N5["ADR-2016: the relay holds no zone key; this checks payload SHAPE only<br/>It cannot authenticate ciphertext or guarantee content secrecy<br/>zone_config.rs:199-207"]
```

## NF-03.13 Mesh federation — configured, gated, and not actually wired

```mermaid
flowchart LR
    MODE["MESH_MODE = standalone<br/>nostr-bbs-relay-worker/wrangler.toml:93"]
    PEERS["MESH_PEER_RELAYS empty<br/>nostr-bbs-relay-worker/wrangler.toml:94"]
    KINDS["MESH_FEDERATED_KINDS - 14 kinds incl. 31400-31405<br/>nostr-bbs-relay-worker/wrangler.toml:95"]
    DIDS["MESH_ALLOWED_REMOTE_DIDS empty<br/>nostr-bbs-relay-worker/wrangler.toml:96"]
    GATE["is_mesh_peer AND NOT is_federated_kind_allowed to blocked<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1176"]

    MODE & PEERS & KINDS & DIDS --> GATE

    N1["A local client whose pubkey is not in the remote-DID list bypasses this check entirely<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1173-1175"]
    N2["README.md:131 now says a concrete transport and delegation validation exist and a live cross-node run is separate evidence, which agrees with the dependency nostr-bbs-relay-worker/Cargo.toml:27. The relay-worker side of the outbound socket is still only a trait: PeerConnector nostr-bbs-relay-worker/src/mesh.rs:143, documented as deferred wiring mesh.rs:25. With MESH_ALLOWED_REMOTE_DIDS empty the gate above is inert regardless."]
    N3["The federated-kind list is where the Agent Control Surface would cross a relay boundary -<br/>EXTERNAL: see NF-06 and, for the consuming side, VC-24 and AB-14"]
```

## NF-03.14 Protected reads — the filter gate, viewer-context caching, and the per-event gate

```mermaid
sequenceDiagram
    autonumber
    participant C as Client socket
    participant RQ as handle_req<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1770
    participant PG as protected_read_blocked<br/>nostr-bbs-relay-worker/src/relay_do/nip42.rs:261
    participant VC as resolve_viewer_context<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:2156
    participant AE as authorize_event<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:2262
    participant PP as protected_read_permitted<br/>nostr-bbs-relay-worker/src/relay_do/nip42.rs:85
    participant A as ActivityLedger<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:2228

    C->>RQ: REQ sub id, filters
    RQ->>PG: do these filters NAME a protected kind nip_handlers.rs:1826
    PG-->>C: CLOSED auth-required when they do nip_handlers.rs:1827
    RQ->>RQ: gate_kind_1059_filters rewrites the mandatory p tag nip_handlers.rs:1838
    RQ->>RQ: store the GATED filter, never the client's raw one nip_handlers.rs:1857
    RQ->>RQ: query_events against the gated filter nip_handlers.rs:1865
    RQ->>VC: ADR-2018: memoised cached_viewer_cohorts + admin check, once per REQ<br/>nip_handlers.rs:1873 nip_handlers.rs:2168
    RQ->>RQ: a kind-1059 wrap is delivered only to the recipient named in its p tag, checked per event before authorize_event gift_wrap_readable_by nip_handlers.rs:156 nip_handlers.rs:1880
    RQ->>AE: every returned event, one at a time, against the resolved<br/>ViewerContext nip_handlers.rs:1883
    AE->>PP: kind, author, p recipients, viewer, mode nip_handlers.rs:2280
    PP-->>AE: correspondence needs a party to it, moderation needs only auth nip42.rs:85
    AE-->>RQ: Withhold when not permitted nip_handlers.rs:2287
    RQ->>A: note_read_activity - only for delivered events, batched into the SAME<br/>coalesced flush as writes nip_handlers.rs:1914 nip_handlers.rs:2228

    Note over RQ: INVARIANT ordering is load-bearing - both gates run BEFORE the<br/>subscription is stored. The previous order stored the raw filter, so a refused<br/>historical read still delivered every later matching event live<br/>nip_handlers.rs:1826-1862
    Note over AE: INVARIANT the protected check is per EVENT, not per filter - a<br/>filter that omits kinds matches every kind while naming none and passed both<br/>filter gates nip_handlers.rs:2268-2272
    Note over VC: DRIFT resolved: has_zone_access (a per-event D1 read) was removed<br/>from trust.rs — zone_read_permitted decides purely from the cached<br/>ViewerContext instead trust.rs:640-642 nip_handlers.rs:2114
    Note over AE: Zone memo caches only positive bindings - an unbound channel still reads D1<br/>nip_handlers.rs:2189-2198 — see NF-17 for cost and failure boundaries
    Note over PP: The two halves of the protected set have different rules - a<br/>moderation event is readable by the membership, a DM only by a party to it<br/>nip42.rs:53-60
```
