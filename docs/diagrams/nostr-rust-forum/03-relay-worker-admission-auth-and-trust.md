---
id: NF-03
title: relay-worker — NIP-42 AUTH, the EVENT admission pipeline, trust ladder and federation
area: nostr-rust-forum
governing:
  - ../nostr-rust-forum/docs/BASELINE-architecture.md
  - ../nostr-rust-forum/docs/IDENTITY-keys-and-trust.md
adrs: [ADR-2004, ADR-2005, ADR-2006, ADR-2010]
sources:
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/lib.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip42.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/whitelist.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/auth.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/zone_config.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/wrangler.toml
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/governance.rs
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/moderation_events.rs
  - ../nostr-rust-forum/README.md
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/Cargo.toml
verified_commit: 380a595f150dd96bfe27ff278fff9ded1be7fbd0
---

## NF-03.1 Worker entry — three doors into the relay

```mermaid
flowchart TB
    REQ["HTTP or WebSocket request"]
    F["fetch<br/>nostr-bbs-relay-worker/src/lib.rs:158"]
    BOOT["ensure_schema + ensure_replay_schema<br/>nostr-bbs-relay-worker/src/lib.rs:160"]
    WS["Upgrade: websocket to Durable Object RELAY<br/>nostr-bbs-relay-worker/src/lib.rs:171"]
    N11["Accept application/nostr+json to NIP-11 relay info<br/>nostr-bbs-relay-worker/src/lib.rs:180"]
    RT["HTTP admin/REST route table<br/>nostr-bbs-relay-worker/src/lib.rs:226"]
    CRON["scheduled every 5 min<br/>nostr-bbs-relay-worker/src/lib.rs:853<br/>nostr-bbs-relay-worker/wrangler.toml:82"]

    REQ --> F --> BOOT
    BOOT --> WS
    BOOT --> N11
    BOOT --> RT
    CRON -.-> BOOT

    N1["The DO is a SINGLETON: every socket goes to get_by_name main<br/>nostr-bbs-relay-worker/src/lib.rs:172"]
    N2["Error responses are JSON-shaped by hand rather than leaking the framework's Debug text<br/>nostr-bbs-relay-worker/src/lib.rs:200"]
```

## NF-03.2 NIP-42 AUTH — the verdict table

```mermaid
sequenceDiagram
    autonumber
    participant C as Client socket
    participant DO as NostrRelayDO
    participant EV as evaluate_auth_event<br/>nostr-bbs-relay-worker/src/relay_do/nip42.rs:123

    DO-->>C: AUTH challenge (per-session, unpredictable)
    C->>DO: ["AUTH", kind-22242 event]
    DO->>EV: event, expected_challenge, own_relay_url, now, max_skew
    EV->>EV: kind must be 22242 nip42.rs:130
    EV->>EV: verify_event_strict - id + Schnorr nip42.rs:134
    EV->>EV: challenge tag must equal THIS session's issued value nip42.rs:140
    EV->>EV: relay tag must name THIS relay, canonicalised nip42.rs:150
    EV->>EV: created_at within +/- 600 s nip42.rs:159 nip42.rs:35
    EV-->>DO: AuthVerdict::Ok(pubkey) nip42.rs:163

    Note over EV: A MISSING session challenge can never match, so it rejects - there is no unchallenged path nip42.rs:138-143
    Note over EV: own_relay_url unset or blank SKIPS the relay-tag check only. This fails OPEN on the defence-in-depth check and never on the challenge or signature, so the gate can roll out before RELAY_URL is configured nip42.rs:115-119 - RELAY_URL ships blank nostr-bbs-relay-worker/wrangler.toml:38
    Note over EV: canonical_relay_url absorbs scheme, case and a trailing slash but keeps host identity nip42.rs:91, asserted nip42.rs:246-259
```

## NF-03.3 AUTH_MODE — the write gate and the protected-read set

```mermaid
stateDiagram-v2
    [*] --> Nip42: parse_auth_mode default<br/>nostr-bbs-relay-worker/src/relay_do/nip42.rs:70
    [*] --> Allowlist: AUTH_MODE == "allowlist" exactly<br/>nostr-bbs-relay-worker/src/relay_do/nip42.rs:72
    Nip42 --> WriteDenied: unauthenticated EVENT<br/>"auth-required: NIP-42 AUTH required to publish"<br/>nostr-bbs-relay-worker/src/relay_do/nip42.rs:175
    Nip42 --> ReadDenied: REQ/COUNT naming a protected kind<br/>nostr-bbs-relay-worker/src/relay_do/nip42.rs:188
    Allowlist --> WriteAllowed: legacy - allowlist alone gates<br/>nostr-bbs-relay-worker/src/relay_do/nip42.rs:176

    note right of Nip42
        Protected read set nip42.rs:44
        4 encrypted DM, 13 seal, 14 private DM,
        1059 gift wrap, 30910-30916 moderation
        INVARIANT: anything else - including a filter
        with no kinds constraint - stays open,
        asserted nip42.rs:297-308
    end note
    note right of Allowlist
        DOC-DRIFT README.md:383: the status ledger calls NIP-42 AUTH "scaffolded",
        says the relay "currently gates on a pubkey allowlist (auth_required: false)"
        and that challenge/response is "not yet the enforced admission path".
        The code says the opposite: nip42 IS the default (nip42.rs:62), the template
        ships AUTH_MODE = "nip42" (nostr-bbs-relay-worker/wrangler.toml:31), and
        every EVENT passes write_auth_ok (nip_handlers.rs:535) before any other check.
    end note
    note right of WriteDenied
        INVARIANT fail-closed parse: a typo, an empty value or any unrecognised
        string resolves to nip42, never to allowlist - asserted nip42.rs:236-242
    end note
```

## NF-03.4 EVENT admission pipeline — every gate, in order

```mermaid
sequenceDiagram
    autonumber
    participant C as Authenticated socket
    participant H as handle_event<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:479
    participant D1 as relay D1
    participant B as broadcast_event

    C->>H: ["EVENT", event]
    H->>H: 1 per-IP rate limit nip_handlers.rs:487
    H->>H: 2 structural validation nip_handlers.rs:493
    H->>H: 3 NIP-40 expiration tag in the past nip_handlers.rs:499
    H->>H: 4 verify_event_strict BEFORE any side effect nip_handlers.rs:510
    H->>H: 5 NIP-42 write gate write_auth_ok nip_handlers.rs:535
    alt kind 1059 gift wrap
        H->>D1: 6a recipient from first p tag must be whitelisted nip_handlers.rs:547 nip_handlers.rs:549
    else every other kind
        H->>D1: 6b effective_pubkey then is_whitelisted nip_handlers.rs:568 nip_handlers.rs:569
    end
    H->>H: 7 mesh peer? federated_kinds allowlist nip_handlers.rs:584
    H->>D1: 8 suspension and silence nip_handlers.rs:595
    H->>D1: 9 ban/mute ingress gate, admins bypass nip_handlers.rs:618
    H->>D1: 10 trust-level gates for 40, 41, 1984 and 5 nip_handlers.rs:628
    H->>H: 11 NIP-29 admin kinds need an h tag and an admin nip_handlers.rs:699
    H->>D1: 12 governance kinds need agent_registry nip_handlers.rs:724
    H->>H: 13 kind-31403 is admin-only nip_handlers.rs:740
    H->>D1: 14 a superseding 31403 is authority-gated nip_handlers.rs:757
    H->>D1: 15 kind-42 zone write gate nip_handlers.rs:770
    H->>D1: 16 NIP-52 RSVP and calendar/kanban write gates nip_handlers.rs:819 nip_handlers.rs:844
    H->>H: 17 NIP-16 treatment - Ephemeral is OK-then-broadcast, never saved nip_handlers.rs:867
    H->>D1: 18 save_event nip_handlers.rs:876
    H->>B: 19 broadcast + post-save effects nip_handlers.rs:878

    Note over H: INVARIANT: the signature is verified BEFORE any side effect - admission state changes and activity tracking alike nip_handlers.rs:508-510
    Note over H: In nip42 mode gift wraps are NOT exempt from the write gate - the wrap is signed by an ephemeral key but is published over the sender's authenticated socket nip_handlers.rs:520-527
```

## NF-03.5 Gift-wrap admission — recipient-keyed, never author-keyed

```mermaid
flowchart LR
    EV["kind-1059 event, ephemeral author"]
    EX["gift_wrap_recipient<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:114"]
    K["kind must be 1059<br/>nip_handlers.rs:115"]
    T["first non-empty p tag<br/>nip_handlers.rs:118"]
    WL["is_whitelisted(recipient)<br/>nip_handlers.rs:549"]
    OK["accepted"]
    NO["blocked: gift-wrap recipient not whitelisted<br/>nip_handlers.rs:557"]

    EV --> EX --> K --> T --> WL
    WL -->|"member"| OK
    WL -->|"not a member, or no p tag"| NO

    N1["INVARIANT ADR-2005: the ephemeral author must NEVER be the admission principal - an author-keyed<br/>check would reject every gift wrap nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:540-546"]
    N2["Gift wraps are DELIBERATELY excluded from ban gating: each is signed by a throwaway key so an<br/>author-keyed ban cannot bind. The recipient-whitelist gate is what bounds them instead<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:62-64, exclusion visible in nip_handlers.rs:68"]
    N3["EXTERNAL: agentbox mirrors sessions as NIP-59 gift-wrapped self-DMs to a different relay - see AB-13"]
```

## NF-03.6 Ban gating — which kinds a banned author may not publish

```mermaid
flowchart TB
    GATE["is_ban_gated_kind<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:67"]
    LIT["1 note, 5 deletion, 7 reaction, 40 channel create,<br/>41 channel metadata, 42 message, 30023 long-form<br/>nip_handlers.rs:68"]
    REP["1984 NIP-56 report nip_handlers.rs:69"]
    CAL["31922 / 31923 calendar, 31925 RSVP<br/>nip_handlers.rs:70 nip_handlers.rs:71 nip_handlers.rs:72"]
    KAN["kanban kinds nip_handlers.rs:73 + 38000 agent intent nip_handlers.rs:74"]
    CALL["call site: admins bypass, then ModCache 60 s check<br/>nip_handlers.rs:618"]

    GATE --> LIT & REP & CAL & KAN
    GATE --> CALL

    N1["WI-2 history: the gate previously fired only for kinds 1 and 42, so a banned user could still<br/>publish reactions, deletions, reports, long-form articles, channel create/metadata and calendar<br/>events nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:57-61"]
    N2["Admins bypass at the call site so they can publish warnings while themselves under moderation<br/>for something else nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:614"]
    N3["Pure predicate by design, so the gate's SCOPE is unit-testable without an Env<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:66"]
```

## NF-03.7 Trust-level gates on the admission path

```mermaid
flowchart TB
    ADMIN{"admin_cache.is_admin<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:628"}
    TL["get_trust_level<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:630"]
    K40["kind 40 channel creation<br/>TL2+ nip_handlers.rs:633"]
    K41["kind 41 channel metadata<br/>TL2+ for own, TL3+ for any<br/>nip_handlers.rs:649 nip_handlers.rs:659"]
    K1984["kind 1984 report<br/>TL1+ nip_handlers.rs:673"]
    K5["kind 5 deletion<br/>own always, others TL3+<br/>nip_handlers.rs:684 nip_handlers.rs:686"]

    ADMIN -->|"admin: skip the whole block"| SKIP["gates bypassed"]
    ADMIN -->|"non-admin"| TL --> K40 & K41 & K1984 & K5

    N1["kind 41 without an e tag is rejected as invalid before the trust check<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:645"]
    N2["ANOMALY O5 re-verified and STILL LIVE: is_channel_creator is looked up from the kind-40 event<br/>nip_handlers.rs:660, so deleting the kind-40 destroys the lookup and locks a TL2 author out of<br/>their own channel's metadata"]
    N3["The kind-5 privilege is re-derived after save because trust_level is scoped to this non-admin<br/>block that admins skip entirely nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:894-899"]
```

## NF-03.8 The trust ladder — promotion, hysteresis, and what is never demoted

```mermaid
stateDiagram-v2
    [*] --> TL0: Newcomer, default on whitelist entry<br/>nostr-bbs-relay-worker/src/trust.rs:28
    TL0 --> TL1: 3 days, 10 reads, 1 post<br/>nostr-bbs-relay-worker/src/trust.rs:184
    TL1 --> TL2: 14 days, 50 reads, 10 posts, ZERO mod actions<br/>nostr-bbs-relay-worker/src/trust.rs:175
    TL2 --> TL3: admin grant only, never computed<br/>nostr-bbs-relay-worker/src/trust.rs:33
    TL2 --> TL1: band broken AND still earns TL1<br/>nostr-bbs-relay-worker/src/trust.rs:376
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
        the band nostr-bbs-relay-worker/src/trust.rs:323-325
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
    note for decide_demotion "CURRENT SOURCE: decide_demotion is pure policy. The per-pubkey check_demotion entry point was removed; only the scheduled trust_sweep path commits demotions. trust.rs:406-417 corrects the stale comments above the policy function."
    note for DemotionDecision "2026-09-07: OFFSET and ignored-write findings are historical. trust_sweep now uses a stable keyset, D1 batch and explicit outcomes. Remaining: its unconditional audit INSERT can commit when the optimistic UPDATE changes zero rows; the conflict is detected after the batch. See NF-11.12 for the mechanism and NF-10.10 for the residual case."
```

## NF-03.10 Post-save effects — activity, moderation mirror, governance projection

```mermaid
sequenceDiagram
    autonumber
    participant H as handle_event after save<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:876
    participant T as trust
    participant M as moderation_actions
    participant G as broker_cases / broker_decisions

    H->>H: OK then broadcast nip_handlers.rs:877 nip_handlers.rs:878
    H->>T: increment_posts_created for kinds 1, 7, 40, 42, 1984 nip_handlers.rs:883
    H->>T: update_last_active nip_handlers.rs:886
    H->>T: check_promotion nip_handlers.rs:889
    alt kind 5
        H->>H: process_deletion, can_delete_any = admin or TL3+ nip_handlers.rs:896
    end
    alt kind 1984
        H->>H: process_report, auto-hide check nip_handlers.rs:904
    end
    alt kinds 30910 / 30911 / 30915 / 30916 from an ADMIN
        H->>M: mirror_moderation_action + mod_cache.invalidate nip_handlers.rs:913
    end
    alt kind 31402
        H->>G: project_appeal if an appeal e-tag, else project_action_request nip_handlers.rs:925
    end
    alt kind 31403
        H->>G: project_supersession if a supersedes e-tag, else project_action_response nip_handlers.rs:938
    end

    Note over H: ADR-2010: the relay OK certified STORAGE ONLY. If the decision did not reach projection-committed the response is accepted but NOT applied, and the relay says so rather than letting its OK stand as the last word nip_handlers.rs:942-952
    Note over M: A moderation mirror is only respected when the signer is an admin ON THIS RELAY - a lifted ban must also stop being enforced nip_handlers.rs:908-913
    Note over G: EXTERNAL: the human-approval loop these projections serve spans the estate - see ES-05 and AB-14
```

## NF-03.11 Structural validation limits

```mermaid
flowchart LR
    V["validate_event<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:961"]
    L1["id 64, pubkey 64, sig 128 hex chars<br/>nip_handlers.rs:962"]
    L2["content cap - registration kinds 0 and 9024 get 8 KiB<br/>nip_handlers.rs:966 nip_handlers.rs:972 nip_handlers.rs:40"]
    L3["max 2000 tags nip_handlers.rs:976 nip_handlers.rs:41"]
    L4["max 1024 bytes per tag value nip_handlers.rs:981 nip_handlers.rs:42"]
    L5["created_at drift cap 7 days nip_handlers.rs:988 nip_handlers.rs:43"]
    L6["max 20 subscriptions per socket nip_handlers.rs:44"]

    V --> L1 & L2 & L3 & L4 & L5
    V -.-> L6
```

## NF-03.12 Zone enforcement on writes

```mermaid
flowchart TB
    K42["kind 42 channel message<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:770"]
    ETAG["e tag required, else invalid<br/>nip_handlers.rs:771"]
    CZ["get_channel_zone - a channel_zones row<br/>nip_handlers.rs:791"]
    UNSCOPED["no row: UNSCOPED, any whitelisted member may post"]
    WGATE["has_zone_write_access<br/>nip_handlers.rs:792"]
    CAL["31922 / 31923 / kanban: zone tag then write cohorts<br/>nip_handlers.rs:844 nip_handlers.rs:855"]
    RSVP["31925 RSVP: author's projection tier for the TARGET must be Full<br/>nip_handlers.rs:819 nip_handlers.rs:827 nip_handlers.rs:834"]

    K42 --> ETAG --> CZ
    CZ -->|"absent"| UNSCOPED
    CZ -->|"present"| WGATE
    CAL --> WGATE

    N1["Finding-5 fix: an undeclared channel previously defaulted to a home zone that the shipped four-zone<br/>model does not define, so the write gate denied ALL non-admins - a permanent lockout of every<br/>non-admin-created channel, including the creator's own nip_handlers.rs:781-787"]
    N2["INVARIANT: channel_zones rows are written SOLELY by the admin-only /api/admin/channel-zone endpoint<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:775-777"]
    N3["The RSVP target is resolved from D1, NEVER from an author-mirrored tag which would be spoofable;<br/>an unresolvable target denies for non-admins nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:809-814, deny at nip_handlers.rs:838"]
    N4["Writes route through write_cohorts ?? required_cohorts, so a public zone can be read-by-all yet<br/>write-restricted - see NF-08.2"]
```

## NF-03.13 Mesh federation — configured, gated, and not actually wired

```mermaid
flowchart LR
    MODE["MESH_MODE = standalone<br/>nostr-bbs-relay-worker/wrangler.toml:52"]
    PEERS["MESH_PEER_RELAYS empty<br/>nostr-bbs-relay-worker/wrangler.toml:53"]
    KINDS["MESH_FEDERATED_KINDS - 14 kinds incl. 31400-31405<br/>nostr-bbs-relay-worker/wrangler.toml:54"]
    DIDS["MESH_ALLOWED_REMOTE_DIDS empty<br/>nostr-bbs-relay-worker/wrangler.toml:55"]
    GATE["is_mesh_peer AND NOT is_federated_kind_allowed to blocked<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:584"]

    MODE & PEERS & KINDS & DIDS --> GATE

    N1["A local client whose pubkey is not in the remote-DID list bypasses this check entirely<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:582-583"]
    N2["DOC-DRIFT README.md:384: the row is right that federation is designed-not-shipped (no concrete<br/>MeshTransport impl ships) but wrong on one fact - it says nostr-bbs-mesh is NOT a dependency of the<br/>relay-worker, while nostr-bbs-relay-worker/Cargo.toml:26 declares it. With MESH_ALLOWED_REMOTE_DIDS<br/>empty the gate above is inert regardless."]
    N3["The federated-kind list is where the Agent Control Surface would cross a relay boundary -<br/>EXTERNAL: see NF-06 and, for the consuming side, VC-24 and AB-14"]
```
