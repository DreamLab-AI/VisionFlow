---
id: NF-03
title: relay-worker — NIP-42 AUTH, the EVENT admission pipeline, trust ladder and federation
area: nostr-rust-forum
governing:
  - ../nostr-rust-forum/docs/BASELINE-architecture.md
  - ../nostr-rust-forum/docs/IDENTITY-keys-and-trust.md
adrs: [ADR-2004, ADR-2005, ADR-2006, ADR-2010, ADR-2011, ADR-2017, ADR-2018]
sources:
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/lib.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip42.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/zone_config.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/wrangler.toml
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/governance.rs
  - ../nostr-rust-forum/README.md
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/Cargo.toml
verified_commit: 72463fbde35ac4c68539b1f65a08ff03b9941201
---

## NF-03.1 Worker entry — three doors into the relay

```mermaid
flowchart TB
    REQ["HTTP or WebSocket request"]
    F["fetch<br/>nostr-bbs-relay-worker/src/lib.rs:192"]
    BOOT["ensure_schema + ensure_replay_schema<br/>nostr-bbs-relay-worker/src/lib.rs:195 lib.rs:196"]
    WS["Upgrade: websocket to Durable Object RELAY<br/>nostr-bbs-relay-worker/src/lib.rs:205"]
    N11["Accept application/nostr+json to NIP-11 relay info<br/>nostr-bbs-relay-worker/src/lib.rs:214"]
    RT["route dispatches the HTTP admin/REST route table<br/>nostr-bbs-relay-worker/src/lib.rs:230 lib.rs:263"]
    CRON["scheduled every 5 min<br/>nostr-bbs-relay-worker/src/lib.rs:1021<br/>nostr-bbs-relay-worker/wrangler.toml:98"]

    REQ --> F --> BOOT
    BOOT --> WS
    BOOT --> N11
    BOOT --> RT
    CRON -.-> BOOT

    N1["The DO is a SINGLETON: every socket goes to get_by_name main<br/>nostr-bbs-relay-worker/src/lib.rs:207"]
    N2["Error responses are JSON-shaped by hand rather than leaking the framework's Debug text<br/>nostr-bbs-relay-worker/src/lib.rs:234"]
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
    Note over EV: own_relay_url unset or blank SKIPS the relay-tag check only. This fails OPEN on the defence-in-depth check and never on the challenge or signature, so the gate can roll out before RELAY_URL is configured nip42.rs:168-172 - RELAY_URL ships blank nostr-bbs-relay-worker/wrangler.toml:38
    Note over EV: canonical_relay_url absorbs scheme, case and a trailing slash but keeps host identity nip42.rs:144, asserted nip42.rs:299-310
```

## NF-03.3 AUTH_MODE — the write gate and the protected-read set

```mermaid
stateDiagram-v2
    [*] --> Nip42: parse_auth_mode default<br/>nostr-bbs-relay-worker/src/relay_do/nip42.rs:123
    [*] --> Allowlist: AUTH_MODE == "allowlist" exactly<br/>nostr-bbs-relay-worker/src/relay_do/nip42.rs:125
    Nip42 --> WriteDenied: unauthenticated EVENT<br/>"auth-required: NIP-42 AUTH required to publish"<br/>nostr-bbs-relay-worker/src/relay_do/nip42.rs:228
    Nip42 --> ReadDenied: REQ/COUNT naming a protected kind<br/>nostr-bbs-relay-worker/src/relay_do/nip42.rs:241
    Allowlist --> WriteAllowed: legacy - allowlist alone gates<br/>nostr-bbs-relay-worker/src/relay_do/nip42.rs:229

    note right of Nip42
        Protected read set nip42.rs:44
        4 encrypted DM, 13 seal, 14 private DM,
        1059 gift wrap, 30910-30916 moderation
        The filter gate reads only the kinds a filter
        NAMES, so a kindless filter passes it,
        asserted nip42.rs:350-360. The PER-EVENT gate
        catches that case - see NF-03.14
    end note
    note right of Allowlist
        DOC-DRIFT README.md:441: the status ledger calls NIP-42 AUTH "scaffolded",
        says the relay "currently gates on a pubkey allowlist (auth_required: false)"
        and that challenge/response is "not yet the enforced admission path".
        The code says the opposite: nip42 IS the default (nip42.rs:115), the template
        ships AUTH_MODE = "nip42" (nostr-bbs-relay-worker/wrangler.toml:31), and
        every EVENT passes write_auth_ok (nip_handlers.rs:845) before any other check.
    end note
    note right of WriteDenied
        INVARIANT fail-closed parse: a typo, an empty value or any unrecognised
        string resolves to nip42, never to allowlist - asserted nip42.rs:289-294
    end note
```

## NF-03.4 EVENT admission pipeline — every gate, in order

```mermaid
sequenceDiagram
    autonumber
    participant C as Authenticated socket
    participant H as handle_event<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:792
    participant Z as zone_config<br/>nostr-bbs-relay-worker/src/zone_config.rs:202
    participant D1 as relay D1
    participant B as broadcast_event

    C->>H: ["EVENT", event]
    H->>H: 1 per-IP rate limit nip_handlers.rs:800
    H->>H: 2 structural validation nip_handlers.rs:806
    H->>H: 3 NIP-40 expiration tag in the past nip_handlers.rs:813
    H->>H: 4 verify_event_strict BEFORE any side effect nip_handlers.rs:820
    H->>H: 5 NIP-42 write gate write_auth_ok nip_handlers.rs:845
    alt kind 1059 gift wrap
        H->>D1: 6a recipient from first p tag must be whitelisted nip_handlers.rs:858 nip_handlers.rs:859
    else every other kind
        H->>D1: 6b effective_pubkey then is_whitelisted nip_handlers.rs:878 nip_handlers.rs:879
    end
    H->>H: 7 mesh peer? federated_kinds allowlist nip_handlers.rs:894
    H->>D1: 8 suspension and silence nip_handlers.rs:905
    H->>D1: 9 ban/mute ingress gate, admins bypass nip_handlers.rs:928
    H->>D1: 10 trust-level gates for 40, 41, 1984 and 5 nip_handlers.rs:940
    H->>H: 11 NIP-29 admin kinds need an h tag and an admin nip_handlers.rs:1013
    H->>D1: 12 governance kinds need agent_registry, except a kanban approval request nip_handlers.rs:1034 nip_handlers.rs:1036
    H->>H: 13 kind-31403 is admin, or a reviewer delegated to THIS case nip_handlers.rs:1054 nip_handlers.rs:1082
    H->>D1: 14 a superseding 31403 is authority-gated nip_handlers.rs:1100 nip_handlers.rs:1105
    H->>Z: 15a ADR-2017 kind-42 sealed tag: admin-only AND an encrypted zone, else rejected nip_handlers.rs:1138 nip_handlers.rs:1143
    H->>D1: 15b kind-42 zone write gate nip_handlers.rs:1148 nip_handlers.rs:1151
    H->>Z: 15c ADR-2018 encrypted zone requires NIP-44 v2 shaped ciphertext nip_handlers.rs:1157 nip_handlers.rs:1163
    H->>D1: 16 NIP-52 RSVP and calendar/kanban write gates nip_handlers.rs:1189 nip_handlers.rs:1214
    H->>H: 17 NIP-16 treatment - Ephemeral is OK-then-broadcast, never saved nip_handlers.rs:1239
    H->>D1: 18 save_event nip_handlers.rs:1246
    H->>B: 19 broadcast + post-save effects, incl. ADR-2018 activity coalescing nip_handlers.rs:1248 nip_handlers.rs:1259

    Note over H: INVARIANT: the signature is verified BEFORE any side effect - admission state changes and activity tracking alike nip_handlers.rs:818-820
    Note over H: In nip42 mode gift wraps are NOT exempt from the write gate - the wrap is signed by an ephemeral key but is published over the sender's authenticated socket nip_handlers.rs:830-837
    Note over H: INVARIANT FR2.2 is enforced at the RELAY, not only in the UI - a 31403 is a signed event any client or script can publish straight here, so a rule that lives only in the forum UI is a suggestion nip_handlers.rs:1070-1076 governance.rs:758
    Note over H: A refusal never fills the rationale in - absence is refused, not papered over governance.rs:761-763
    Note over Z: INVARIANT ADR-2017: a sealed original is exempt from validate_event's timestamp-drift check on the tag alone, so this write gate is what actually enforces admin-only AND encrypted-zone-only for it nip_handlers.rs:1372-1380 nip_handlers.rs:1417-1431
```

## NF-03.5 Gift-wrap admission — recipient-keyed, never author-keyed

```mermaid
flowchart LR
    EV["kind-1059 event, ephemeral author"]
    EX["gift_wrap_recipient<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:115"]
    K["kind must be 1059<br/>nip_handlers.rs:116"]
    T["first non-empty p tag<br/>nip_handlers.rs:119"]
    WL["is_whitelisted(recipient)<br/>nip_handlers.rs:859"]
    OK["accepted"]
    NO["blocked: gift-wrap recipient not whitelisted<br/>nip_handlers.rs:867"]

    EV --> EX --> K --> T --> WL
    WL -->|"member"| OK
    WL -->|"not a member, or no p tag"| NO

    N1["INVARIANT ADR-2005: the ephemeral author must NEVER be the admission principal - an author-keyed<br/>check would reject every gift wrap nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:850-856"]
    N2["Gift wraps are DELIBERATELY excluded from ban gating: each is signed by a throwaway key so an<br/>author-keyed ban cannot bind. The recipient-whitelist gate is what bounds them instead<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:56-65, exclusion visible in nip_handlers.rs:69-75"]
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
    CALL["call site: admins bypass, then ModCache 60 s check<br/>nip_handlers.rs:928"]

    GATE --> LIT & REP & CAL & KAN
    GATE --> CALL

    N1["WI-2 history: the gate previously fired only for kinds 1 and 42, so a banned user could still<br/>publish reactions, deletions, reports, long-form articles, channel create/metadata, calendar<br/>events/RSVPs, kanban and agent-intent nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:56-65"]
    N2["Admins bypass at the call site so they can publish warnings while themselves under moderation<br/>for something else nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:929"]
    N3["Pure predicate by design, so the gate's SCOPE is unit-testable without an Env<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:67"]
```

## NF-03.7 Trust-level gates on the admission path

```mermaid
flowchart TB
    ADMIN{"admin_cache.is_admin<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:938"}
    TL["get_trust_level<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:940"]
    K40["kind 40 channel creation<br/>TL2+ nip_handlers.rs:943"]
    K41["kind 41 channel metadata<br/>TL2+ for own, TL3+ for any<br/>nip_handlers.rs:959 nip_handlers.rs:969"]
    K1984["kind 1984 report<br/>TL1+ nip_handlers.rs:983"]
    K5["kind 5 deletion<br/>own always, others TL3+<br/>nip_handlers.rs:994 nip_handlers.rs:996"]

    ADMIN -->|"admin: skip the whole block"| SKIP["gates bypassed"]
    ADMIN -->|"non-admin"| TL --> K40 & K41 & K1984 & K5

    N1["kind 41 without an e tag is rejected as invalid before the trust check<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:955-956"]
    N2["ANOMALY O5 re-verified and STILL LIVE: is_channel_creator is looked up from the kind-40 event<br/>nip_handlers.rs:970, so deleting the kind-40 destroys the lookup and locks a TL2 author out of<br/>their own channel's metadata"]
    N3["The kind-5 privilege is re-derived after save because trust_level is scoped to this non-admin<br/>block that admins skip entirely nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1266-1269"]
```

## NF-03.8 The trust ladder — promotion, hysteresis, and what is never demoted

```mermaid
stateDiagram-v2
    [*] --> TL0: Newcomer, default on whitelist entry<br/>nostr-bbs-relay-worker/src/trust.rs:28
    TL0 --> TL1: 3 days, 10 reads, 1 post<br/>nostr-bbs-relay-worker/src/trust.rs:183
    TL1 --> TL2: 14 days, 50 reads, 10 posts, ZERO mod actions<br/>nostr-bbs-relay-worker/src/trust.rs:174
    TL2 --> TL3: admin grant only, never computed<br/>nostr-bbs-relay-worker/src/trust.rs:33
    TL2 --> TL1: band broken AND still earns TL1<br/>nostr-bbs-relay-worker/src/trust.rs:377
    TL2 --> TL0: band broken and TL1 not earned<br/>nostr-bbs-relay-worker/src/trust.rs:379
    TL1 --> TL0: band broken<br/>nostr-bbs-relay-worker/src/trust.rs:391

    note right of TL3
        INVARIANT ADR-2006: TL3 is an administrative grant and is never auto-demoted
        nostr-bbs-relay-worker/src/trust.rs:334
        check_promotion also refuses to touch a TL3 row nostr-bbs-relay-worker/src/trust.rs:221
    end note
    note right of TL0
        Guard order is deliberate: TL3, the TL0 floor and admin/exempt rows are excluded
        BEFORE any activity arithmetic, so an exempt row is never even measured against
        the band nostr-bbs-relay-worker/src/trust.rs:323-336
        Floor hold nostr-bbs-relay-worker/src/trust.rs:339, admin hold nostr-bbs-relay-worker/src/trust.rs:344
    end note
    note right of TL2
        Hysteresis 90 percent of the promotion threshold, plus a ~6-month inactivity gate
        thresholds nostr-bbs-relay-worker/src/trust.rs:84 nostr-bbs-relay-worker/src/trust.rs:85
        inactivity gate nostr-bbs-relay-worker/src/trust.rs:349, band nostr-bbs-relay-worker/src/trust.rs:359
        ADR-2006 permits TL2 to land DIRECTLY on TL0 - one committed transition per sweep,
        not one rung of the ladder nostr-bbs-relay-worker/src/trust.rs:303-305
    end note
```

## NF-03.9 Why a demotion did NOT happen — the named hold reasons

```mermaid
classDiagram
    class HoldReason {
        AdminGrantedLevel : trust.rs:287
        ExemptAdminRow : trust.rs:290
        AtFloor : trust.rs:292
        WithinActivityWindow : trust.rs:295
        ClearsHysteresis : trust.rs:297
    }
    class DemotionDecision {
        Hold(HoldReason) : trust.rs:309
        Demote from to : trust.rs:311
    }
    class decide_demotion {
        pure, no IO, no clock read : trust.rs:326
        debug_assert new_level lower : trust.rs:398
    }
    decide_demotion --> DemotionDecision
    DemotionDecision --> HoldReason

    note for HoldReason "ADR-2006 acceptance: every hold is an EXPLICIT NAMED policy outcome rather than a silent nothing-happened, so a sweep can report WHY a candidate survived nostr-bbs-relay-worker/src/trust.rs:278-282"
    note for decide_demotion "CURRENT SOURCE: decide_demotion is pure policy. There is no per-pubkey demotion entry point by decision (ADR-2006): only the scheduled trust_sweep path commits demotions. trust.rs:406-419 records why the seam was removed rather than just left uncalled."
    note for DemotionDecision "2026-09-07: OFFSET and ignored-write findings are historical. trust_sweep now uses a stable keyset, D1 batch and explicit outcomes. Remaining: its unconditional audit INSERT can commit when the optimistic UPDATE changes zero rows; the conflict is detected after the batch. See NF-11.12 for the mechanism and NF-10.10 for the residual case."
```

## NF-03.10 Post-save effects — activity coalescing, moderation mirror, governance projection

```mermaid
sequenceDiagram
    autonumber
    participant H as handle_event after save<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1246
    participant A as ActivityLedger<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1903
    participant T as trust
    participant M as moderation_actions
    participant G as broker_cases / broker_decisions

    H->>H: OK then broadcast nip_handlers.rs:1247 nip_handlers.rs:1248
    H->>T: increment_posts_created for kinds 1, 7, 40, 42, 1984 nip_handlers.rs:1253
    H->>A: note_write_activity - ADR-2018 coalesced flush, replaces a per-event update_last_active + check_promotion nip_handlers.rs:1259 nip_handlers.rs:1903
    alt kind 5
        H->>H: process_deletion, can_delete_any = admin or TL3+ nip_handlers.rs:1267
    end
    alt kind 1984
        H->>H: process_report, auto-hide check nip_handlers.rs:1275
    end
    alt kinds 30910 / 30911 / 30915 / 30916 from an ADMIN
        H->>M: mirror_moderation_action + mod_cache.invalidate nip_handlers.rs:1284
    end
    alt kind 31402
        H->>G: project_appeal if an appeal e-tag, else project_action_request nip_handlers.rs:1297 nip_handlers.rs:1299
    end
    alt kind 31403
        H->>G: project_supersession if a supersedes e-tag, else project_action_response nip_handlers.rs:1310 nip_handlers.rs:1316
    end

    Note over H: ADR-2010: the relay OK certified STORAGE ONLY.<br/>If the decision did not reach projection-committed the response is accepted<br/>but NOT applied, and the relay says so rather than letting its OK stand as<br/>the last word nip_handlers.rs:1312-1323
    Note over A: DRIFT resolved: the read path (NF-03.14) also flushes through this<br/>same ledger via note_read_activity, so a promotion check now runs at most once<br/>per pubkey per coalescing window across BOTH writes and reads<br/>nip_handlers.rs:1893 nip_handlers.rs:1911-1913
    Note over M: A moderation mirror is only respected when the signer is an admin ON THIS RELAY - a lifted ban must also stop being enforced nip_handlers.rs:1284
    Note over G: EXTERNAL: the human-approval loop these projections serve spans the estate - see ES-05 and AB-14
```

## NF-03.11 Structural validation limits

```mermaid
flowchart LR
    V["validate_event<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1331"]
    L1["id 64, pubkey 64, sig 128 hex chars<br/>nip_handlers.rs:1332"]
    L2["content cap - registration kinds 0 and 9024 get 8 KiB<br/>nip_handlers.rs:1336 nip_handlers.rs:1342 nip_handlers.rs:41"]
    L3["max 2000 tags nip_handlers.rs:1346 nip_handlers.rs:42"]
    L4["max 1024 bytes per tag value nip_handlers.rs:1351 nip_handlers.rs:43"]
    L5["created_at drift cap 7 days, sealed originals exempt (ADR-2017) nip_handlers.rs:1357 nip_handlers.rs:44 nip_handlers.rs:1397"]
    L6["max 20 subscriptions per socket nip_handlers.rs:45 nip_handlers.rs:1465"]

    V --> L1 & L2 & L3 & L4 & L5
    V -.-> L6
```

## NF-03.12 Zone enforcement on writes — sealed originals, encrypted-zone ciphertext, plain writes

```mermaid
flowchart TB
    K42["kind 42 channel message<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1112"]
    ETAG["e tag required, else invalid<br/>nip_handlers.rs:1114"]
    CZ["cached_channel_zone - memoised channel_zones lookup<br/>nip_handlers.rs:1133 nip_handlers.rs:1854"]
    SEALED{"has_sealed_tag<br/>nip_handlers.rs:1138"}
    SREJ["sealed_write_rejection: non-admin -> admin-only; zone not encrypted -> encrypted-zones-only (ADR-2017)<br/>nip_handlers.rs:1143 nip_handlers.rs:1417-1431"]
    UNSCOPED["no row: UNSCOPED, any whitelisted member may post"]
    WGATE["has_zone_write_access<br/>nip_handlers.rs:1149"]
    CIPHER["is_encrypted zone requires is_zone_ciphertext - zk tag + NIP-44 v2 shape (ADR-2018)<br/>nip_handlers.rs:1157 zone_config.rs:127 zone_config.rs:202"]
    CAL["31922 / 31923 / kanban: zone tag then write cohorts<br/>nip_handlers.rs:1214 nip_handlers.rs:1225"]
    RSVP["31925 RSVP: author's projection tier for the TARGET must be Full<br/>nip_handlers.rs:1189 nip_handlers.rs:1190 nip_handlers.rs:1204"]

    K42 --> ETAG --> CZ
    CZ --> SEALED
    SEALED -->|"sealed"| SREJ
    CZ -->|"present, not sealed or sealed+admin+encrypted"| WGATE
    WGATE --> CIPHER
    CAL --> WGATE

    N1["Finding-5 fix: an undeclared channel previously defaulted to a home zone that the shipped four-zone<br/>model does not define, so the write gate denied ALL non-admins - a permanent lockout of every<br/>non-admin-created channel, including the creator's own nip_handlers.rs:1123-1132"]
    N2["INVARIANT: channel_zones rows are written SOLELY by the admin-only /api/admin/channel-zone endpoint<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1117-1121"]
    N3["The RSVP target is resolved from D1, NEVER from an author-mirrored tag which would be spoofable;<br/>an unresolvable target denies for non-admins nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1179-1184, deny at nip_handlers.rs:1211"]
    N4["Writes route through write_cohorts ?? required_cohorts, so a public zone can be read-by-all yet<br/>write-restricted - see NF-08.2"]
    N5["INVARIANT ADR-2018: the relay holds no zone key, so is_zone_ciphertext is a SHAPE check only - it<br/>guarantees nothing readable is stored in an encrypted zone by ANY author, admins included<br/>zone_config.rs:193-201"]
```

## NF-03.13 Mesh federation — configured, gated, and not actually wired

```mermaid
flowchart LR
    MODE["MESH_MODE = standalone<br/>nostr-bbs-relay-worker/wrangler.toml:68"]
    PEERS["MESH_PEER_RELAYS empty<br/>nostr-bbs-relay-worker/wrangler.toml:69"]
    KINDS["MESH_FEDERATED_KINDS - 14 kinds incl. 31400-31405<br/>nostr-bbs-relay-worker/wrangler.toml:70"]
    DIDS["MESH_ALLOWED_REMOTE_DIDS empty<br/>nostr-bbs-relay-worker/wrangler.toml:71"]
    GATE["is_mesh_peer AND NOT is_federated_kind_allowed to blocked<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:894"]

    MODE & PEERS & KINDS & DIDS --> GATE

    N1["A local client whose pubkey is not in the remote-DID list bypasses this check entirely<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:891-893"]
    N2["DOC-DRIFT README.md:442: the row is right that federation is designed-not-shipped (no concrete<br/>MeshTransport impl ships) but wrong on one fact - it says nostr-bbs-mesh is NOT a dependency of the<br/>relay-worker, while nostr-bbs-relay-worker/Cargo.toml:26 declares it. With MESH_ALLOWED_REMOTE_DIDS<br/>empty the gate above is inert regardless."]
    N3["The federated-kind list is where the Agent Control Surface would cross a relay boundary -<br/>EXTERNAL: see NF-06 and, for the consuming side, VC-24 and AB-14"]
```

## NF-03.14 Protected reads — the filter gate, viewer-context caching, and the per-event gate

```mermaid
sequenceDiagram
    autonumber
    participant C as Client socket
    participant RQ as handle_req<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1439
    participant PG as protected_read_blocked<br/>nostr-bbs-relay-worker/src/relay_do/nip42.rs:241
    participant VC as resolve_viewer_context<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1821
    participant AE as authorize_event<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1927
    participant PP as protected_read_permitted<br/>nostr-bbs-relay-worker/src/relay_do/nip42.rs:85
    participant A as ActivityLedger<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1893

    C->>RQ: REQ sub id, filters
    RQ->>PG: do these filters NAME a protected kind nip_handlers.rs:1495
    PG-->>C: CLOSED auth-required when they do nip_handlers.rs:1496
    RQ->>RQ: gate_kind_1059_filters rewrites the mandatory p tag nip_handlers.rs:1507
    RQ->>RQ: store the GATED filter, never the client's raw one nip_handlers.rs:1526
    RQ->>RQ: query_events against the gated filter nip_handlers.rs:1534
    RQ->>VC: ADR-2018: memoised cached_viewer_cohorts + admin check, once per REQ<br/>nip_handlers.rs:1542 nip_handlers.rs:1833
    RQ->>AE: every returned event, one at a time, against the resolved<br/>ViewerContext nip_handlers.rs:1549
    AE->>PP: kind, author, p recipients, viewer, mode nip_handlers.rs:1945
    PP-->>AE: correspondence needs a party to it, moderation needs only auth nip42.rs:85
    AE-->>RQ: Withhold when not permitted nip_handlers.rs:1952
    RQ->>A: note_read_activity - only for delivered events, batched into the SAME<br/>coalesced flush as writes nip_handlers.rs:1580 nip_handlers.rs:1893

    Note over RQ: INVARIANT ordering is load-bearing - both gates run BEFORE the<br/>subscription is stored. The previous order stored the raw filter, so a refused<br/>historical read still delivered every later matching event live<br/>nip_handlers.rs:1495-1527
    Note over AE: INVARIANT the protected check is per EVENT, not per filter - a<br/>filter that omits kinds matches every kind while naming none and passed both<br/>filter gates nip_handlers.rs:1933-1937
    Note over VC: DRIFT resolved: has_zone_access (a per-event D1 read) was removed<br/>from trust.rs — zone_read_permitted decides purely from the cached<br/>ViewerContext instead trust.rs:640-642 nip_handlers.rs:1779
    Note over PP: The two halves of the protected set have different rules - a<br/>moderation event is readable by the membership, a DM only by a party to it<br/>nip42.rs:53-60
```
