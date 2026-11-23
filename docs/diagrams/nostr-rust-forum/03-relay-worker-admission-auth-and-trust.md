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
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/recipient_keys.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/auth.rs
verified_commit: 4a73a7a94bfb4d92fea03f04926cccd38c14e9af
---

## NF-03.1 Worker entry — three doors into the relay

```mermaid
flowchart TB
    REQ["HTTP or WebSocket request"]
    F["fetch<br/>nostr-bbs-relay-worker/src/lib.rs:211"]
    BOOT["ensure_schema + ensure_replay_schema, once per isolate: SCHEMA_READY latches only after a pass with no failed CREATE<br/>nostr-bbs-relay-worker/src/lib.rs:224 lib.rs:225 lib.rs:226 latch lib.rs:202-204"]
    WS["Upgrade: websocket to Durable Object RELAY<br/>nostr-bbs-relay-worker/src/lib.rs:230"]
    N11["Accept application/nostr+json to NIP-11 relay info<br/>nostr-bbs-relay-worker/src/lib.rs:239"]
    RT["route dispatches the HTTP admin/REST route table<br/>nostr-bbs-relay-worker/src/lib.rs:255 lib.rs:288"]
    CRON["scheduled every 5 min<br/>nostr-bbs-relay-worker/src/lib.rs:1622<br/>nostr-bbs-relay-worker/wrangler.toml:125"]

    REQ --> F --> BOOT
    BOOT --> WS
    BOOT --> N11
    BOOT --> RT

    N1["The DO is a SINGLETON: every socket goes to get_by_name main<br/>nostr-bbs-relay-worker/src/lib.rs:232"]
    N2["Error responses are JSON-shaped by hand rather than leaking the framework's Debug text<br/>nostr-bbs-relay-worker/src/lib.rs:258-269"]
    N3["The cron does NOT pass through the bootstrap: neither scheduled nor the Durable Object creates the schema,<br/>which is why fetch bootstraps BEFORE the WebSocket branch nostr-bbs-relay-worker/src/lib.rs:220-223; scheduled goes straight to the<br/>demotion sweep nostr-bbs-relay-worker/src/lib.rs:1633"]
```

## NF-03.2 NIP-42 AUTH — the verdict table

```mermaid
sequenceDiagram
    autonumber
    participant C as Client socket
    participant DO as NostrRelayDO, handle_auth<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:2124
    participant EV as evaluate_auth_event<br/>nostr-bbs-relay-worker/src/relay_do/nip42.rs:197

    DO-->>C: AUTH challenge at connect (per-session, unpredictable) - see NF-11.2
    C->>DO: ["AUTH", kind-22242 event]
    DO->>DO: allowed set = RELAY_URL plus RELAY_URL_ALIASES, canonicalised, explicit config only nip_handlers.rs:2131
    DO->>EV: event, expected_challenge, allowed_relay_urls, now, max_skew nip_handlers.rs:2140
    EV->>EV: kind must be 22242 nip42.rs:204
    EV->>EV: verify_event_strict - id + Schnorr nip42.rs:208
    EV->>EV: challenge tag must equal THIS session's issued value nip42.rs:214
    EV->>EV: empty allowed set REJECTS - invalid: relay url unconfigured nip42.rs:226
    EV->>EV: relay tag must match one allowed URL, canonicalised nip42.rs:229
    EV->>EV: created_at within +/- 600 s nip42.rs:239 nip42.rs:35
    EV-->>DO: AuthVerdict::Ok(pubkey) nip42.rs:243
    DO->>DO: mark session authed and persist it, then OK true nip_handlers.rs:2152 nip_handlers.rs:2156

    Note over EV: A MISSING session challenge can never match, so it rejects - there is no unchallenged path nip42.rs:212-217
    Note over EV: RESOLVED since 6064622 - the relay-tag check no longer fails OPEN. A blank RELAY_URL leaves the allowed set empty and every AUTH is refused, because a request-derived identity would let an attacker's Host poison the binding nip42.rs:220-227. The set is built by allowed_relay_urls nip42.rs:252
    Note over DO: DEBT the template still ships RELAY_URL and RELAY_URL_ALIASES blank nostr-bbs-relay-worker/wrangler.toml:49 wrangler.toml:53, so an unconfigured nip42 deployment refuses every AUTH and therefore every write. handle_auth logs a console error on each AUTH while it is blank nip_handlers.rs:2137 nip42.rs:284. The kit records it as open finding KF-29 at 4a73a7a - see NF-10.11
    Note over DO: DOC-DRIFT the runtime message lags the template and SETUP - the console error handle_auth emits still says only that every authenticated write is REFUSED nip42.rs:288, and the call-site comment says the same nip_handlers.rs:2135-2137, though the empty set refuses every AUTH and so every protected-kind read as well nip42.rs:226
    Note over EV: RESOLVED at 4a73a7a - the evaluate_auth_event doc comment now says the caller builds the set from RELAY_URL and RELAY_URL_ALIASES only, never from the request host nip42.rs:187-193, and the helper comment says its one caller is the per-AUTH log, nothing at startup or in NIP-11 nip42.rs:274-277 nip_handlers.rs:2137
    Note over EV: canonical_relay_url absorbs scheme, case and a trailing slash but keeps host identity nip42.rs:163, asserted nip42.rs:432-443
```

## NF-03.3 AUTH_MODE — the write gate and the protected-read set

```mermaid
stateDiagram-v2
    [*] --> Nip42: parse_auth_mode default<br/>nostr-bbs-relay-worker/src/relay_do/nip42.rs:142
    [*] --> Allowlist: AUTH_MODE == "allowlist" exactly<br/>nostr-bbs-relay-worker/src/relay_do/nip42.rs:144
    Nip42 --> WriteDenied: unauthenticated EVENT<br/>"auth-required: NIP-42 AUTH required to publish"<br/>nostr-bbs-relay-worker/src/relay_do/nip42.rs:302
    Nip42 --> ReadDenied: REQ/COUNT naming a protected kind<br/>nostr-bbs-relay-worker/src/relay_do/nip42.rs:315
    Allowlist --> WriteAllowed: legacy - allowlist alone gates<br/>nostr-bbs-relay-worker/src/relay_do/nip42.rs:303

    note right of Nip42
        Protected read set nip42.rs:44
        4 encrypted DM, 13 seal, 14 private DM,
        1059 gift wrap, 30910-30916 moderation
        The filter gate reads only the kinds a filter
        NAMES, so a kindless filter passes it,
        asserted nip42.rs:492-493. The PER-EVENT gate
        catches that case - see NF-03.14
    end note
    note right of Allowlist
        README.md:130 agrees with the code: NIP-42 relay admission is "Implemented;
        example worker selects AUTH_MODE=nip42".
        nip42 IS the default (nip42.rs:134), the template ships AUTH_MODE = "nip42"
        (nostr-bbs-relay-worker/wrangler.toml:39), and every EVENT passes write_auth_ok
        (nip_handlers.rs:1142) after the rate, shape, expiry and signature
        checks and before any admission lookup - see NF-03.4. A denial AFTER a completed AUTH is
        "restricted:", while legacy allowlist mode keeps "blocked:" (nip42.rs:336)
    end note
    note right of WriteDenied
        INVARIANT fail-closed parse: a typo, an empty value or any unrecognised
        string resolves to nip42, never to allowlist - asserted nip42.rs:422-427
    end note
```

## NF-03.4 EVENT admission pipeline — every gate, in order, up to the save

```mermaid
sequenceDiagram
    autonumber
    participant C as Authenticated socket
    participant H as handle_event<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1089
    participant Z as zone_config<br/>nostr-bbs-relay-worker/src/zone_config.rs:238
    participant D1 as relay D1

    C->>H: ["EVENT", event]
    H->>H: 1 per-IP rate limit nip_handlers.rs:1097
    H->>H: 2 structural validation nip_handlers.rs:1103
    H->>H: 3 NIP-40 expiration tag in the past, or malformed nip_handlers.rs:1110
    H->>H: 4 verify_event_strict BEFORE any side effect nip_handlers.rs:1117
    H->>H: 5 NIP-42 write gate write_auth_ok nip_handlers.rs:1142
    H->>D1: 6 resolve the effective principal ONCE, never for a gift wrap (ADR-099 device key to owner) nip_handlers.rs:1161
    alt kind 1059 gift wrap
        H->>D1: 7a gift_wrap_recipient_verdict on the first p tag - ingress, epoch key, inbox key owner or whitelisted recipient, see NF-03.5 nip_handlers.rs:1179 nip_handlers.rs:1182
    else every other kind
        H->>D1: 7b is_whitelisted on the principal nip_handlers.rs:1193
    end
    H->>H: 8 mesh peer? federated_kinds allowlist nip_handlers.rs:1208
    H->>D1: 9 suspension and silence, checked for the device key AND its owner nip_handlers.rs:1220 nip_handlers.rs:1221
    H->>D1: 10 ban/mute ingress gate, admins bypass nip_handlers.rs:1245 nip_handlers.rs:1249
    H->>D1: 11 non-admin trust-level gates for 40, 41 and 1984 nip_handlers.rs:1260
    H->>D1: 11b non-admin kind 5 below TL3 - member_deletion_gate, see NF-03.15 nip_handlers.rs:1320 nip_handlers.rs:1321
    H->>H: 12 NIP-29 admin kinds need an h tag and an admin nip_handlers.rs:1329
    H->>D1: 13 governance kinds need agent_registry, except a 31403 (gated at step 14) and a kanban approval request nip_handlers.rs:1354 nip_handlers.rs:1355 nip_handlers.rs:1356
    H->>D1: 14 kind-31403 is admin, or a reviewer delegated to THIS case, then the FR2.2 rationale check nip_handlers.rs:1374 nip_handlers.rs:1384 nip_handlers.rs:1402
    H->>D1: 15 a superseding 31403 is authority-gated nip_handlers.rs:1420 nip_handlers.rs:1426
    H->>H: 16 an appeal-marked 31402 must name its case in a case tag distinct from d nip_handlers.rs:1437
    H->>Z: 17a ADR-2017 kind-42 sealed tag: admin-only AND an encrypted zone, else rejected nip_handlers.rs:1470 nip_handlers.rs:1475
    H->>D1: 17b kind-42 zone write gate nip_handlers.rs:1481 nip_handlers.rs:1482
    H->>Z: 17c ADR-2016 encrypted zone requires NIP-44 v2 shaped ciphertext nip_handlers.rs:1488 nip_handlers.rs:1494
    H->>D1: 18 NIP-52 RSVP and calendar/kanban write gates nip_handlers.rs:1520 nip_handlers.rs:1545
    H->>D1: 18b ZONE_GROUPS_ENABLED only - zone_group_gate on the zk group hint and epoch uniqueness nip_handlers.rs:1576 nip_handlers.rs:1588
    H->>H: 19 NIP-16 treatment - Ephemeral is OK-then-broadcast, never saved nip_handlers.rs:1598
    H->>D1: 20 NIP-09 is_deleted - a tombstoned event is refused, see NF-03.16 nip_handlers.rs:1607 nip_handlers.rs:1608
    H->>D1: 21 push_event_known, then save_event nip_handlers.rs:1615 nip_handlers.rs:1618
    H->>H: 22 OK, broadcast and the post-save effects of NF-03.10 nip_handlers.rs:1619 nip_handlers.rs:1620

    Note over H: INVARIANT: the signature is verified BEFORE any side effect - admission state changes and activity tracking alike nip_handlers.rs:1115-1117
    Note over H: In nip42 mode gift wraps are NOT exempt from the write gate - the wrap is signed by an ephemeral key but is published over the sender's authenticated socket nip_handlers.rs:1127-1134
    Note over H: INVARIANT FR2.2 is enforced at the RELAY, not only in the UI - a 31403 is a signed event any client or script can publish straight here, so a rule that lives only in the forum UI is a suggestion nip_handlers.rs:1390-1396 governance.rs:807
    Note over H: A refusal never fills the rationale in - absence is refused, not papered over governance.rs:810-812
    Note over Z: INVARIANT ADR-2017: a sealed original is exempt from validate_event's timestamp-drift check on the tag alone, so this write gate is what actually enforces admin-only AND encrypted-zone-only for it nip_handlers.rs:1769-1777 nip_handlers.rs:1814-1826
    Note over H: DEBT the authors flag step 18b themselves - a cohort member can still first-write-win future epochs with junk keys, so ZONE_GROUPS_ENABLED must stay off until an owner-only epoch binding lands nip_handlers.rs:1570-1575
    Note over H: Step 12 still guards NIP-29 admin kinds with an acknowledged TODO nip_handlers.rs:1330-1332, while NIP-11 no longer advertises NIP-29 - see NF-11.14
```

## NF-03.5 Gift-wrap admission — recipient-keyed, never author-keyed

```mermaid
flowchart TB
    EV["kind-1059 event, ephemeral author"]
    EX["gift_wrap_recipient: first non-empty p tag<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:160 nip_handlers.rs:164"]
    ING["ingress = recipient declared in INGRESS_RECIPIENTS: needs no D1 row, so a reset-db cannot silence it<br/>nostr-bbs-relay-worker/src/ingress.rs:27 nip_handlers.rs:1179-1181"]
    V["gift_wrap_recipient_verdict<br/>nip_handlers.rs:2352"]
    NOP["no p tag: blocked, recipient not whitelisted<br/>nip_handlers.rs:2358-2359"]
    EPO{"ZONE_GROUPS_ENABLED and a registered unrevoked epoch key, not ingress<br/>nip_handlers.rs:2364"}
    RK{"RECIPIENT_KEYS_ENABLED exactly true<br/>nip_handlers.rs:2368 nip_handlers.rs:2389-2390"}
    OFF["flag off: ingress, or is_whitelisted(recipient), else<br/>blocked: gift-wrap recipient not whitelisted<br/>nip_handlers.rs:2369-2372 recipient_keys.rs:27"]
    ROUTE["flag on: route - ingress accepts, a revoked inbox key is refused as not eligible,<br/>a registered key is judged by its OWNER's membership, any other key must itself be a member<br/>recipient_keys.rs:94-102 nip_handlers.rs:2380-2386"]
    VER["verdict: owner not a member - not eligible, unregistered non-member - not registered<br/>recipient_keys.rs:106-113 recipient_keys.rs:23 recipient_keys.rs:25"]
    OK["accepted"]

    EV --> EX --> ING --> V
    V -->|"no recipient"| NOP
    V --> EPO
    EPO -->|"yes, no whitelist read"| OK
    EPO -->|"no"| RK
    RK -->|"off"| OFF
    RK -->|"on"| ROUTE --> VER
    OFF -->|"member or ingress"| OK
    VER -->|"member"| OK

    N1["INVARIANT ADR-2005: the ephemeral author must NEVER be the admission principal - an author-keyed<br/>check would reject every gift wrap nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1168-1174"]
    N2["Gift wraps are DELIBERATELY excluded from ban gating: each is signed by a throwaway key so an<br/>author-keyed ban cannot bind. The recipient gate is what bounds them instead<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:72-74, exclusion visible in nip_handlers.rs:77-86"]
    N4["INVARIANT ADR-2022: the ingress list only widens who a wrap may be ADDRESSED to - it grants no cohort, admin right or zone key, and it is consulted only inside the kind-1059 branch, so events an ingress key authors are checked against the whitelist as before<br/>nostr-bbs-relay-worker/src/ingress.rs:9-12"]
    N5["Neither RECIPIENT_KEYS_ENABLED nor ZONE_GROUPS_ENABLED is set in the template, and only the exact string true enables<br/>either, so the shipped path is the flag-off branch nip_handlers.rs:2389-2390 nip_handlers.rs:2462-2463"]
    N3["EXTERNAL: agentbox mirrors sessions as NIP-59 gift-wrapped self-DMs to a different relay - see AB-13"]
```

## NF-03.6 Ban gating — which kinds a banned author may not publish

```mermaid
flowchart TB
    GATE["is_ban_gated_kind<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:77"]
    LIT["1 note, 5 deletion, 7 reaction, 9 chat, 40 channel create,<br/>41 channel metadata, 42 message, 30023 long-form,<br/>34550 community definition<br/>nip_handlers.rs:78-80 nip_handlers.rs:91"]
    REP["1984 NIP-56 report nip_handlers.rs:81"]
    CAL["calendar date/plain events, 31925 RSVP<br/>nip_handlers.rs:82 nip_handlers.rs:83 nip_handlers.rs:84"]
    KAN["kanban kinds + agent-intent<br/>nip_handlers.rs:85 nip_handlers.rs:86"]
    CALL["call site: admins bypass, then ModCache 60 s check per sanction subject<br/>nip_handlers.rs:1245 nip_handlers.rs:1249"]

    GATE --> LIT & REP & CAL & KAN
    GATE --> CALL

    N1["WI-2 history: the gate previously fired only for kinds 1 and 42, so a banned user could still<br/>publish reactions, deletions, reports, long-form articles, channel create/metadata, calendar<br/>events/RSVPs, kanban and agent-intent nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:65-71"]
    N2["Admins bypass at the call site so they can publish warnings while themselves under moderation<br/>for something else nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1246"]
    N3["Pure predicate by design, so the gate's SCOPE is unit-testable without an Env<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:76"]
    N4["INVARIANT: kind 5 is ban-gated, so a banned or muted member cannot publish a deletion at all,<br/>whatever the member deletion gate of NF-03.15 would say nip_handlers.rs:80"]
    N5["DUPLICATE, harmless: a ban-gated kind asks admin_cache.is_admin twice for the same pubkey - the bypass nip_handlers.rs:1246<br/>and the trust gate nip_handlers.rs:1258. The second is a hit on the per-DO cache entry, fresh for 30 s<br/>nostr-bbs-relay-worker/src/auth.rs:46 nostr-bbs-relay-worker/src/auth.rs:110, so it costs no D1 read"]
```

## NF-03.7 Trust-level gates on the admission path

```mermaid
flowchart TB
    ADMIN{"admin_cache.is_admin<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1258"}
    TL["get_trust_level<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1260"]
    K40["kind 40 channel creation<br/>TL2+ nip_handlers.rs:1263"]
    K41["kind 41 channel metadata<br/>TL2+ for own, TL3+ for any<br/>nip_handlers.rs:1279 nip_handlers.rs:1289"]
    K1984["kind 1984 report<br/>TL1+ nip_handlers.rs:1305"]
    K5["kind 5 deletion below TL3: member_deletion_gate,<br/>TL3+ skips the lookup - see NF-03.15<br/>nip_handlers.rs:1320 nip_handlers.rs:1321"]
    IC["TL2 branch only: is_channel_creator, matched against the signing key AND its device owner<br/>nip_handlers.rs:1291 defined nip_handlers.rs:3703"]
    CC["1. channel_creators row for the channel id<br/>nip_handlers.rs:3715"]
    FB["2. fallback when no row: the kind-40 event itself<br/>nip_handlers.rs:3716"]
    ANS["first row found decides: creator_matches against the sanction subjects (key plus owner)<br/>nip_handlers.rs:3722"]
    SAVE["save_event, when a kind 40 was stored<br/>relay_do/storage.rs:232"]
    INS["INSERT INTO channel_creators, first write wins<br/>relay_do/storage.rs:235 relay_do/storage.rs:236"]
    DDL["table created by ensure_schema from CHANNEL_CREATORS_CREATE_SQL<br/>nostr-bbs-relay-worker/src/lib.rs:735 lib.rs:1075<br/>and by migrations/0009_events.sql:44"]
    DEL["kind-5 process_deletion removes only the kind-40 events row<br/>nip_handlers.rs:3512"]

    ADMIN -->|"admin: skip the whole block"| SKIP["gates bypassed"]
    ADMIN -->|"non-admin"| TL --> K40 & K41 & K1984 & K5
    K41 --> IC --> CC -->|"no row"| FB
    CC --> ANS
    FB --> ANS
    K40 -.->|"accepted and stored"| SAVE --> INS --> DDL
    DEL -.->|"leaves untouched"| CC

    N1["kind 41 without an e tag is rejected as invalid before the trust check<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1275-1276"]
    N2["ANOMALY O5 CLOSED at 66ca8d8: the creator used to be re-read from the kind-40 event, so a kind-5<br/>deletion of it locked a TL2 author out of their own channel. The creator is now persisted at creation in<br/>channel_creators and read first, so the deletion no longer changes the answer<br/>nip_handlers.rs:3715 storage.rs:235"]
    N3["The kind-5 privilege is re-derived after save because trust_level is scoped to this non-admin<br/>block that admins skip entirely nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1635-1639"]
    N4["DEBT: the creator INSERT result is discarded (let _ =) and the code comment says failures never block<br/>storage, so a channel whose write failed has no row and is still on the deletion-exposed fallback<br/>relay_do/storage.rs:244 nip_handlers.rs:3716. Channels created before the table existed have no row either<br/>and are likewise exposed, nip_handlers.rs:3701-3702"]
    N5["INVARIANT: the answer comes from the first of the two lookups that returns a row, so once a<br/>channel_creators row exists the kind-40 event is never consulted again<br/>nip_handlers.rs:3722"]
    N6["CHANGED since 6064622: the old rule here was own always, others TL3+. A member's kind 5 is now refused<br/>whole unless every e target is vouched for - the gate of NF-03.15 replaced the targets_others check"]
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

## NF-03.10 Post-save effects — activity coalescing, deletion, moderation mirror, governance projection

```mermaid
sequenceDiagram
    autonumber
    participant H as handle_event after save<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1618
    participant A as ActivityLedger<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:2602
    participant T as trust
    participant M as moderation_actions
    participant G as broker_cases / broker_decisions
    participant P as push_fanout<br/>nostr-bbs-relay-worker/src/relay_do/push_fanout.rs:47

    H->>H: OK then broadcast nip_handlers.rs:1619 nip_handlers.rs:1620
    H->>T: counts_as_post - kinds 1, 7, 9, 40, 42, 1984 - increments posts_created on the principal nip_handlers.rs:1624 nip_handlers.rs:96-97
    H->>A: note_write_activity - ADR-2018 coalesced flush<br/>nip_handlers.rs:1630 nip_handlers.rs:2602
    Note over H,T: Save success also includes duplicate IDs of regular kinds - this counter is not exact-once<br/>nip_handlers.rs:1612-1630 — see NF-17.6
    alt kind 5
        H->>H: process_deletion, can_delete_any = admin or TL3+, tombstones then deletes - see NF-11.18 nip_handlers.rs:1638 nip_handlers.rs:1640
    end
    alt kind 1984
        H->>H: process_report, auto-hide check nip_handlers.rs:1645
    end
    alt kinds 30910 / 30911 / 30915 / 30916 from an ADMIN
        H->>M: mirror_moderation_action + mod_cache.invalidate nip_handlers.rs:1654 nip_handlers.rs:1656
    end
    alt kind 31402
        H->>G: project_appeal if an appeal e-tag, else project_action_request nip_handlers.rs:1667 nip_handlers.rs:1676
    end
    alt kind 31403
        H->>G: project_supersession if a supersedes e-tag, else project_action_response nip_handlers.rs:1687 nip_handlers.rs:1700
    end

    H->>P: ADR-2023 content-free Web Push wake-up, last, only for kinds 40 / 42 / 1059 and not for a repeat submission nip_handlers.rs:1615 nip_handlers.rs:1721

    Note over P: ADR-2023 is staged: PUSH_ENABLED ships "false"<br/>nostr-bbs-relay-worker/wrangler.toml:88<br/>Disabled fan-out returns at once push_fanout.rs:48<br/>Candidates use the shared read gate push_fanout.rs:94
    Note over H,A: ADR-2010: the relay OK certifies storage only.<br/>Without projection-committed, the response is accepted<br/>but not applied - reported separately nip_handlers.rs:1696-1706
    Note over A,T: Reads and writes share this ledger.<br/>A later activity call triggers an elapsed-window flush.<br/>Pending-read threshold can flush sooner - see NF-17.4<br/>nip_handlers.rs:2592 nip_handlers.rs:2610-2612
    Note over M: Mirror requires an admin on this relay.<br/>A lifted ban must stop enforcement nip_handlers.rs:1653
    Note over G: External human-approval workflow<br/>spans the estate - see ES-05 and AB-14
    Note over G: RESOLVED: the 31403 body is no longer parsed separately by gate and projection -<br/>action and reasoning come from one pure helper nip_handlers.rs:254, and the projection is the<br/>receipt-bearing project_action_response nip_handlers.rs:4112, not re-audited line by line
```

**Tension (ADR-2010 vs code):** ADR-2010 records `decision_status: proposed`, `implementation_status: partial`, `activation_status: staged` (`../nostr-rust-forum/docs/adr/ADR-2010-durable-governance-outcome-receipts.md:5-7`), yet `handle_event` runs all three receipt-bearing projections (appeal, supersession, first-order response) unconditionally after every stored governance event (`nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1667`, `nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1687`, `nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1700`).

## NF-03.11 Structural validation limits and the subscription cap

```mermaid
flowchart LR
    V["validate_event<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1728"]
    L1["id 64, pubkey 64, sig 128 hex chars<br/>nip_handlers.rs:1729"]
    L2["content cap - registration kinds 0 and 9024 get 8 KiB<br/>nip_handlers.rs:1733 nip_handlers.rs:1739 nip_handlers.rs:41"]
    L3["max 2000 tags nip_handlers.rs:1743 nip_handlers.rs:42"]
    L4["max 1024 bytes per tag value nip_handlers.rs:1748 nip_handlers.rs:43"]
    L5["created_at drift cap 7 days, sealed originals exempt from the PAST bound only (ADR-2017) nip_handlers.rs:1754 nip_handlers.rs:44 nip_handlers.rs:1794"]
    L6["max 20 subscriptions per socket, decided by subscription_cap_reached on REQ<br/>nip_handlers.rs:45 nip_handlers.rs:52-53 nip_handlers.rs:1862"]
    L7["#91: a REQ reusing an id the session holds is a REPLACEMENT and is never refused - an AUTH replay<br/>re-sends every held id, and at a full session it used to be refused wholesale nip_handlers.rs:47-51"]

    V --> L1 & L2 & L3 & L4 & L5
    V -.-> L6 --> L7

    N1["OPEN (NF-10 O20, KF-22): the cap is read and the borrow released at nip_handlers.rs:1862, the subscription<br/>is inserted only at nip_handlers.rs:1929-1931, after the D1 awaits at nip_handlers.rs:1907 and nip_handlers.rs:1908.<br/>Two REQs with different ids on one session can both pass, so the cap is a resource bound, not a security boundary"]
```

## NF-03.12 Zone enforcement on writes — sealed originals, encrypted-zone ciphertext, plain writes

```mermaid
flowchart TB
    K42["kind 42 channel message<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1444"]
    ETAG["e tag required, else invalid<br/>nip_handlers.rs:1446"]
    CZ["cached_channel_zone - memoised channel_zones lookup<br/>nip_handlers.rs:1465 nip_handlers.rs:2311"]
    SEALED{"has_sealed_tag<br/>nip_handlers.rs:1470"}
    SREJ["sealed_write_rejection: non-admin -> admin-only; zone not encrypted -> encrypted-zones-only (ADR-2017)<br/>nip_handlers.rs:1475 nip_handlers.rs:1814-1826"]
    UNSCOPED["no row: UNSCOPED, any whitelisted member may post"]
    WGATE["has_zone_write_access<br/>nip_handlers.rs:1481"]
    CIPHER["is_encrypted zone requires is_zone_ciphertext - zk tag + NIP-44 v2 shape (ADR-2016)<br/>nip_handlers.rs:1488 zone_config.rs:163 zone_config.rs:238"]
    CAL["31922 / 31923 / kanban: zone tag then write cohorts<br/>nip_handlers.rs:1545 nip_handlers.rs:1556"]
    RSVP["31925 RSVP: author's projection tier for the TARGET must be Full<br/>nip_handlers.rs:1520 nip_handlers.rs:1521 nip_handlers.rs:1535"]

    K42 --> ETAG --> CZ
    CZ --> SEALED
    SEALED -->|"sealed"| SREJ
    CZ -->|"present, not sealed or sealed+admin+encrypted"| WGATE
    CZ -->|"absent"| UNSCOPED
    WGATE --> CIPHER
    CAL --> WGATE

    N1["Finding-5 fix: an undeclared channel previously defaulted to a home zone that the shipped four-zone<br/>model does not define, so the write gate denied ALL non-admins - a permanent lockout of every<br/>non-admin-created channel, including the creator's own nip_handlers.rs:1455-1464"]
    N2["INVARIANT: channel_zones rows are written SOLELY by the admin-only /api/admin/channel-zone endpoint<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1449-1453"]
    N3["The RSVP target is resolved from D1, NEVER from an author-mirrored tag which would be spoofable;<br/>an unresolvable target denies for non-admins nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1510-1515, deny at nip_handlers.rs:1542"]
    N4["Writes route through write_cohorts ?? required_cohorts, so a public zone can be read-by-all yet<br/>write-restricted - see NF-08.2"]
    N5["ADR-2016: the relay holds no zone key; this checks payload SHAPE only<br/>It cannot authenticate ciphertext or guarantee content secrecy<br/>zone_config.rs:229-237"]
```

## NF-03.13 Mesh federation — configured, gated, and not actually wired

```mermaid
flowchart LR
    MODE["MESH_MODE = standalone<br/>nostr-bbs-relay-worker/wrangler.toml:95"]
    PEERS["MESH_PEER_RELAYS empty<br/>nostr-bbs-relay-worker/wrangler.toml:96"]
    KINDS["MESH_FEDERATED_KINDS - 14 kinds incl. 31400-31405<br/>nostr-bbs-relay-worker/wrangler.toml:97"]
    DIDS["MESH_ALLOWED_REMOTE_DIDS empty<br/>nostr-bbs-relay-worker/wrangler.toml:98"]
    GATE["is_mesh_peer AND NOT is_federated_kind_allowed to blocked<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1208"]

    MODE & PEERS & KINDS & DIDS --> GATE

    N1["A local client whose pubkey is not in the remote-DID list bypasses this check entirely<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1205-1207"]
    N2["README.md:131 now says a concrete transport and delegation validation exist and a live cross-node run is separate evidence, which agrees with the dependency nostr-bbs-relay-worker/Cargo.toml:27. The relay-worker side of the outbound socket is still only a trait: PeerConnector nostr-bbs-relay-worker/src/mesh.rs:143, documented as deferred wiring mesh.rs:25. With MESH_ALLOWED_REMOTE_DIDS empty the gate above is inert regardless."]
    N3["The federated-kind list is where the Agent Control Surface would cross a relay boundary -<br/>EXTERNAL: see NF-06 and, for the consuming side, VC-24 and AB-14"]
```

## NF-03.14 Protected reads — the filter gate, the wrap read set, viewer-context caching, and the per-event gate

```mermaid
sequenceDiagram
    autonumber
    participant C as Client socket
    participant RQ as handle_req<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1836
    participant PG as protected_read_blocked<br/>nostr-bbs-relay-worker/src/relay_do/nip42.rs:315
    participant VC as resolve_viewer_context<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:2271
    participant AE as authorize_event<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:2626
    participant PP as protected_read_permitted_with_keys<br/>nostr-bbs-relay-worker/src/relay_do/nip42.rs:99
    participant A as ActivityLedger<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:2592

    C->>RQ: REQ sub id, filters
    RQ->>RQ: subscription cap, a reused id is exempt - see NF-03.11 nip_handlers.rs:1862
    RQ->>PG: do these filters NAME a protected kind nip_handlers.rs:1895
    PG-->>C: CLOSED auth-required when they do nip_handlers.rs:1896
    RQ->>RQ: wrap_read_set - own key plus registered inbox keys and readable epoch keys,<br/>None when both features are off nip_handlers.rs:1907 nip_handlers.rs:2531
    RQ->>RQ: gate_kind_1059_filters_with - no read set forces the p tag to the authed key,<br/>a read set keeps a client p tag or binds to the set nip_handlers.rs:1908 nip_handlers.rs:2246-2257
    RQ->>RQ: store the GATED filter, never the client's raw one nip_handlers.rs:1931
    RQ->>RQ: query_events against the gated filter nip_handlers.rs:1939
    RQ->>VC: ADR-2018: memoised cohorts + admin check, plus recipient and epoch keys, once per REQ<br/>nip_handlers.rs:1947 nip_handlers.rs:2283 nip_handlers.rs:2288-2293
    RQ->>RQ: a kind-1059 wrap is delivered only when its FIRST p tag is in the read set,<br/>or is the session key, checked per event before authorize_event<br/>nip_handlers.rs:1954-1959 nip_handlers.rs:178 nip_handlers.rs:191
    RQ->>AE: every returned event, one at a time, against the resolved<br/>ViewerContext nip_handlers.rs:1961
    AE->>PP: kind, author, p recipients, viewer, recipient keys, mode nip_handlers.rs:2644
    PP-->>AE: correspondence needs a party to it - for 1059 an inbox key counts -<br/>moderation needs only auth nip42.rs:99
    AE-->>RQ: Withhold when not permitted nip_handlers.rs:2652
    RQ->>A: note_read_activity - only for delivered events, batched into the SAME<br/>coalesced flush as writes nip_handlers.rs:1992 nip_handlers.rs:2592

    Note over RQ: INVARIANT ordering is load-bearing - both gates run BEFORE the<br/>subscription is stored. The previous order stored the raw filter, so a refused<br/>historical read still delivered every later matching event live<br/>nip_handlers.rs:1887-1894 nip_handlers.rs:1895-1936
    Note over RQ: With recipient keys on, the STORED filter keeps the client's p tag on purpose,<br/>so a key registered a moment later is not behind a dead subscription -<br/>isolation is the per-event check nip_handlers.rs:2224-2229
    Note over AE: INVARIANT the protected check is per EVENT, not per filter - a<br/>filter that omits kinds matches every kind while naming none and passed both<br/>filter gates nip_handlers.rs:2632-2636
    Note over VC: DRIFT resolved: has_zone_access (a per-event D1 read) was removed<br/>from trust.rs — zone_read_permitted decides purely from the cached<br/>ViewerContext instead trust.rs:640-642 nip_handlers.rs:2201
    Note over AE: Zone memo caches only positive bindings - an unbound channel still reads D1<br/>nip_handlers.rs:2311-2319 — see NF-17 for cost and failure boundaries
    Note over PP: The two halves of the protected set have different rules - a<br/>moderation event is readable by the membership, a DM only by a party to it<br/>nip42.rs:53-60
```

## NF-03.15 NIP-09 member deletion gate — a member's kind 5 is vouched for target by target (PR #94)

```mermaid
flowchart TB
    K5["kind 5 from a non-admin below TL3<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1320"]
    G["member_deletion_gate<br/>nip_handlers.rs:3351"]
    NOE{"any e targets?<br/>deletion_e_targets nip_handlers.rs:2980 nip_handlers.rs:3352-3356"}
    ACC0["accepted: an a-only kind 5 is scoped later by a_ref_target, see NF-11.18"]
    CAN{"all 64 lowercase hex?<br/>is_canonical_event_id nip_handlers.rs:2995 nip_handlers.rs:3358"}
    UNK["refused: deletion names an event this relay does not hold<br/>nip_handlers.rs:2972-2973"]
    Q["ONE D1 statement over json_each classifies every target that is not the signer's own stored row<br/>DELETION_TARGETS_CLASSIFY_SQL lib.rs:811-823, bound at nip_handlers.rs:3367-3369"]
    QF["refused: could not check deletion targets - FAIL CLOSED on a missing binding,<br/>serialisation, bind or query error nip_handlers.rs:3361-3377 nip_handlers.rs:2977"]
    CLS["classify_member_target per row nip_handlers.rs:3047"]
    FOR{"Foreign: another author's stored row<br/>nip_handlers.rs:3049"}
    WRAP{"kind 1059 addressed to one of the signer's recipient keys?<br/>keys fetched only then nip_handlers.rs:3081-3086 nip_handlers.rs:3381-3384"}
    OTH["refused: TL3+ required to delete others' events<br/>nip_handlers.rs:2960"]
    SEAL{"SealedOriginal: no row, a stored kind-42 sealed tag names it<br/>nip_handlers.rs:3056"}
    SR["refused: migrated (sealed) posts can only be removed by a moderator<br/>nip_handlers.rs:2965-2966"]
    TOMB{"AlreadyDeleted: no row, deleted_events holds id and signer<br/>nip_handlers.rs:3057"}
    OK["accepted"]

    K5 --> G --> NOE
    NOE -->|"none"| ACC0
    NOE -->|"some"| CAN
    CAN -->|"no"| UNK
    CAN -->|"yes"| Q
    Q -->|"error"| QF
    Q -->|"no rows: every target is the signer's own"| OK
    Q -->|"rows"| CLS --> FOR
    FOR -->|"foreign"| WRAP
    WRAP -->|"not theirs"| OTH
    WRAP -->|"theirs: target excused"| OK
    FOR -->|"no row"| SEAL
    SEAL -->|"yes"| SR
    SEAL -->|"no"| TOMB
    TOMB -->|"yes"| OK
    TOMB -->|"no"| UNK

    N1["Why whole-event refusal: every client folds every e tag of a stored kind 5, the event is signed and the relay<br/>cannot strip a tag, so one target the member cannot be shown to own refuses it all nip_handlers.rs:2940-2949"]
    N2["INVARIANT: the sealed flag is read BEFORE the tombstone flag, so a claim written before the gate existed<br/>cannot reopen a sealed original nip_handlers.rs:3042-3046 nip_handlers.rs:3056-3057. Refusals report in the order<br/>foreign, sealed, unknown nip_handlers.rs:3075-3096"]
    N3["INVARIANT: fail-closed is safe because the gate only runs for members - admins skip the whole non-admin<br/>block and TL3+ skips the lookup, so it can never block a moderator nip_handlers.rs:3347-3350 nip_handlers.rs:1315-1320"]
    N4["OPEN (NF-10 O12, KF-21): one statement may carry up to 2,000 ids MAX_TAG_COUNT nip_handlers.rs:42, travelling as one JSON<br/>value because D1 caps bound parameters near 100 lib.rs:793-797. Every test of it runs on in-memory rusqlite<br/>nip_handlers.rs:7700-7708 nip_handlers.rs:7758-7774, never on Cloudflare D1"]
    N5["OPEN (NF-10 O13, KF-17): a no-row target is accepted on an (id, signer) row in deleted_events. Pre-gate claims and the<br/>backfill from stored kind 5s lib.rs:915-925 hold such rows, and a moderator's process_deletion still writes one for any<br/>well-formed id nip_handlers.rs:3535-3546, so a holder can re-name an id never stored here and have it accepted and broadcast"]
    N6["OPEN (NF-10 O14, KF-20): a kind 5 with no e and no a target is accepted with nothing to act on nip_handlers.rs:3353-3356.<br/>An expiring kind 5 is later removed by the NIP-40 sweep, which writes no tombstone - see NF-11.19"]
```

## NF-03.16 NIP-09 ingest refusal — a deleted event stays deleted (PR #89)

```mermaid
flowchart TB
    GATES["every admission gate of NF-03.4 has passed"]
    EPH["Ephemeral 20000-29999: OK and broadcast, never stored, so never tombstone-checked<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1598-1601"]
    ISD["is_deleted<br/>nip_handlers.rs:1607 defined nip_handlers.rs:3476"]
    REG["regular kinds: deleted_events holds (id, author)<br/>IS_DELETED_ID_SQL lib.rs:777-778, bound nip_handlers.rs:3499"]
    ADR["replaceable and addressable kinds: the id tombstone, OR a deleted_addresses row for<br/>(kind, author, d_tag) with deleted_until at or after created_at<br/>IS_DELETED_ADDRESSABLE_SQL lib.rs:784-787, bound nip_handlers.rs:3486-3497"]
    REF["refused: blocked: this event was deleted<br/>nip_handlers.rs:1608"]
    SAVE["push_event_known, then save_event<br/>nip_handlers.rs:1615 nip_handlers.rs:1618"]
    TBL["tables deleted_events (id, pubkey) and deleted_addresses (kind, pubkey, d_tag)<br/>lib.rs:746 lib.rs:755, created by ensure_schema lib.rs:1083-1084"]

    GATES --> EPH
    GATES --> ISD
    ISD --> REG & ADR
    REG -->|"match"| REF
    ADR -->|"match"| REF
    REG -->|"no match"| SAVE
    ADR -->|"no match"| SAVE
    TBL -.-> ISD

    N1["INVARIANT: the check runs after every authorisation gate and before save_event nip_handlers.rs:1607 nip_handlers.rs:1618,<br/>so anyone still holding the signed copy cannot publish it again, whichever route removed it nip_handlers.rs:1604-1606"]
    N2["INVARIANT: an event id is a hash over its author, so a tombstone naming the wrong author blocks nothing<br/>lib.rs:741-745 - which is why the writers record the row's REAL author, see NF-11.18 and NF-11.19"]
    N3["Fail-open by design: a missing binding or a bind failure answers not deleted - the tombstone is a re-insert<br/>guard, not an availability gate nip_handlers.rs:3472-3478 nip_handlers.rs:3501-3502"]
    N4["Pre-existing deletions are carried over once per database by backfill_tombstones_once from stored kind 5s,<br/>own-deletion semantics only lib.rs:1484 lib.rs:915-918"]
```
