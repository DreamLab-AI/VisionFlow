---
id: NF-05
title: The two Leptos clients — boot, identity, transport, onboarding, messaging and the retro BBS
area: nostr-rust-forum
governing:
  - ../nostr-rust-forum/docs/BASELINE-architecture.md
  - ../nostr-rust-forum/docs/IDENTITY-keys-and-trust.md
adrs: [ADR-2008, ADR-2023, ADR-2025]
sources:
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/keys.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/main.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/app.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/relay.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/relay/receipt.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/utils/lookup.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/stores/tombstones.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/stores/deletion_sync.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/stores/signer_admin.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/whitelist.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/auth.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/auth/mod.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/auth/passkey.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/webauthn.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/auth/nip98.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/auth/nip07.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/auth/session.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/auth/webauthn.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/stores/channels.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/stores/zones.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/stores/profile_cache.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/stores/notifications.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/components/global_search.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/dm/mod.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/utils/push.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/wrangler.toml
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/components/push_settings.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/components/onboarding_modal.rs
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/gift_wrap.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/utils/search_client.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/signup.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/settings.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/thread.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/channel.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/message_jump.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/events.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/pod_browser.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/components/recovery_sheet.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/components/git_panel.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/components/agent_badge.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/admin/user_table.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/utils/relay_url.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/wallet/mod.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/mod.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/store.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/sw.js
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/Trunk.toml
  - ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/app.rs
  - ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/signer.rs
  - ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/menu.rs
  - ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/chrome.rs
  - ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/pwa.rs
  - ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/dm.rs
  - ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/config.rs
  - ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/identity.rs
  - ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/ascii_img.rs
  - ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/upload.rs
  - ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/passkey.rs
  - ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/relay.rs
  - ../nostr-rust-forum/docs/diagrams/00-anomaly-register.md
  - ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/screens.rs
  - ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/theme.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/category.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/components/mobile_bottom_nav.rs
  - ../nostr-rust-forum/docs/adr/ADR-2027-nip17-overlay-for-nostr-chat-interop.md
  - ../nostr-rust-forum/docs/adr/ADR-2028-zones-adopt-nip171-key-plane.md
  - ../nostr-rust-forum/docs/adr/ADR-2023-content-free-web-push-for-pwa-alerts.md
verified_commit: af4f0d56f735ca29dddf9be2549d5c0e9f987891
---

## NF-05.1 Forum client boot and the FORUM_BASE discipline

```mermaid
sequenceDiagram
    autonumber
    participant B as Browser
    participant M as main.rs<br/>nostr-bbs-forum-client/src/main.rs:18
    participant SW as Service worker
    participant A as App<br/>nostr-bbs-forum-client/src/app.rs:312
    participant DB as IndexedDB

    B->>M: load the WASM bundle
    M->>M: mount_to_body(App) main.rs:24
    M->>SW: register_service_worker main.rs:47
    M->>SW: sw url and scope built from FORUM_BASE main.rs:55 main.rs:56
    M->>SW: update_via_cache = None, so sw.js is always revalidated main.rs:64
    M->>DB: evict cached messages older than 30 days main.rs:195
    A->>A: provide_auth nostr-bbs-forum-client/src/app.rs:319, provide_zone_access nostr-bbs-forum-client/src/app.rs:320,<br/>provide_render_tier nostr-bbs-forum-client/src/app.rs:321
    A->>A: provide_toasts nostr-bbs-forum-client/src/app.rs:324, wallet::provide_wallet<br/>nostr-bbs-forum-client/src/app.rs:340, provide_profile_cache nostr-bbs-forum-client/src/app.rs:331
    A->>A: start_admin_alerts nostr-bbs-forum-client/src/app.rs:349, provide_agent_disclosure<br/>nostr-bbs-forum-client/src/app.rs:440
    A->>A: RelayConnection::new nostr-bbs-forum-client/src/app.rs:448, provide_channel_store<br/>nostr-bbs-forum-client/src/app.rs:450, zone_crypto::store::provide_zone_key_store<br/>nostr-bbs-forum-client/src/app.rs:453

    Note over A: wallet::provide_wallet creates one store per configured compiled-in chain (ADR-2015 amended by ADR-2021),<br/>inert unless SIDESTR_WALLET is on - nostr-bbs-forum-client/src/app.rs:337-340, wallet/mod.rs:58<br/>wallet/mod.rs:239
    Note over A: INVARIANT ADR-090 - FORUM_BASE is applied in exactly TWO places: the const consumed by Router base= at<br/>nostr-bbs-forum-client/src/app.rs:44 and :929
    Note over A: and base_href() for every link at nostr-bbs-forum-client/src/app.rs:53, used e.g. at :936. A third application<br/>re-introduces the double-prefix / deep-route-404 bug class.
    Note over M: main.rs:55-57 re-read the same option_env! for the service-worker SCOPE - a deliberate, documented mirror of<br/>app::FORUM_BASE, not a third application of the prefix
    Note over A: current_app_path strips the prefix back off a browser path so the router sees an unprefixed route<br/>nostr-bbs-forum-client/src/app.rs:82
```

**Debt:** the mobile bottom nav strips the prefix with its own `option_env!("FORUM_BASE")` read (`../nostr-rust-forum/crates/nostr-bbs-forum-client/src/components/mobile_bottom_nav.rs:30`) instead of calling `current_app_path` (`../nostr-rust-forum/crates/nostr-bbs-forum-client/src/app.rs:82`). This is not a third *application* of the prefix, so the ADR-090 invariant above holds. It is a second hand-written copy of the stripping rule, though, and it can drift from `utils::paths::app_path`. Closing it means routing the nav through `current_app_path`.


## NF-05.2 Route table

```mermaid
flowchart LR
    subgraph pubroutes["Public"]
        R1["/ HomeOrForums nostr-bbs-forum-client/src/app.rs:947"]
        R2["/about nostr-bbs-forum-client/src/app.rs:948 | /login nostr-bbs-forum-client/src/app.rs:949 | /signup nostr-bbs-forum-client/src/app.rs:954"]
        R3["/connect magic link - must NOT be auth-gated nostr-bbs-forum-client/src/app.rs:953"]
        R4["/glossary nostr-bbs-forum-client/src/app.rs:958 | /go/:event_id MessageJumpPage nostr-bbs-forum-client/src/app.rs:960 | /join/:code nostr-bbs-forum-client/src/app.rs:965"]
    end
    subgraph authed["Auth-gated"]
        R5["/setup nostr-bbs-forum-client/src/app.rs:967 | /forums nostr-bbs-forum-client/src/app.rs:981 | /settings nostr-bbs-forum-client/src/app.rs:991"]
        R6["/chat/:channel_id nostr-bbs-forum-client/src/app.rs:978 | /dm nostr-bbs-forum-client/src/app.rs:979 | /dm/:pubkey nostr-bbs-forum-client/src/app.rs:980"]
        R7["/forums/:category/board nostr-bbs-forum-client/src/app.rs:986 | /:section nostr-bbs-forum-client/src/app.rs:987 | /:topic nostr-bbs-forum-client/src/app.rs:988"]
        R8["/events nostr-bbs-forum-client/src/app.rs:989 | /profile/:pubkey nostr-bbs-forum-client/src/app.rs:990 | /pod nostr-bbs-forum-client/src/app.rs:1000"]
        R14["/wallet AuthGatedWallet - member wallets, ADR-2015 nostr-bbs-forum-client/src/app.rs:1001"]
    end
    subgraph gov["Governance"]
        R9["/governance member read-only nostr-bbs-forum-client/src/app.rs:998"]
        R10["/governance/admin admin write nostr-bbs-forum-client/src/app.rs:999"]
        R11["/admin - its own internal gate nostr-bbs-forum-client/src/app.rs:992"]
    end
    subgraph zonealias["Zone-slug aliases, declared LAST"]
        R12["/:category nostr-bbs-forum-client/src/app.rs:1020 | /:category/board nostr-bbs-forum-client/src/app.rs:1022"]
        R13["/:category/:section nostr-bbs-forum-client/src/app.rs:1023 | /:category/:section/:topic nostr-bbs-forum-client/src/app.rs:1024"]
    end

    ROUTER["Router base=FORUM_BASE nostr-bbs-forum-client/src/app.rs:929"] --> pubroutes & authed & gov & zonealias

    N1["/chat with no channel redirects to /forums - a legacy path kept alive nostr-bbs-forum-client/src/app.rs:977"]
    N2["The zone-slug aliases are declared last so the static routes out-score them; a slug is only valid<br/>if it resolves in the live ZONE_CONFIG nostr-bbs-forum-client/src/pages/category.rs:103"]
    N3["The member and admin governance views are separate ROUTES bound to separate components -<br/>see NF-06.10"]
    N4["/go/:event_id resolves a bare event id to its topic and lands the reader on it - see NF-05.15 for<br/>the deep-link walk and NF-05.6 for the tombstone/zone-decrypt gate it shares with normal ingest"]
```

## NF-05.3 Passkey identity — register, log in, derive

```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant AS as AuthStore<br/>nostr-bbs-forum-client/src/auth/mod.rs:105
    participant PK as auth::passkey<br/>nostr-bbs-forum-client/src/auth/passkey.rs:115
    participant AW as auth-worker
    participant WA as WebAuthn authenticator

    U->>AS: register
    AS->>PK: register_passkey auth/mod.rs:299
    PK->>AW: POST /auth/register/options auth/passkey.rs:115
    PK->>WA: create() ceremony
    WA-->>PK: credential + PRF output
    PK->>PK: extract PRF from creation auth/webauthn.rs:123
    PK->>PK: derive_from_prf(prf, prfSalt) auth/passkey.rs:153
    PK->>AW: POST /auth/register/verify, NIP-98 signed with the derived key auth/passkey.rs:181-182
    Note over PK,AW: RESOLVED at nostr-rust-forum 6064622 - the client now sends the NIP-98 header the worker requires<br/>See NF-23.1 for the earlier unsigned request
    U->>AS: log in
    AS->>PK: authenticate_passkey auth/mod.rs:328
    PK->>AW: POST /auth/login/options auth/passkey.rs:247
    PK->>PK: extract PRF from assertion auth/webauthn.rs:169
    PK->>PK: derive using returned salt auth/passkey.rs:274
    PK->>AW: verify call, itself NIP-98 authenticated auth/passkey.rs:300
    AS->>AS: PrfSigner built from the derived key auth/mod.rs:770

    Note over AS,PK: Private-key copy is held in StoredValue auth/mod.rs:107<br/>get_privkey_bytes returns a Zeroizing copy auth/mod.rs:232
    Note over AS,PK: pagehide skips bfcache and clears only privkey<br/>Other signer copies are not cleared<br/>auth/session.rs:428 auth/session.rs:442
    Note over PK,AW: Server checks are in NF-02.3 and NF-02.4
```

[NF-23 and its report](../../estate-review/2026-11-01-forum-passkey-contracts.md) qualify this sequence. At the report's stamp the worker stored a different PRF salt from creation; that is resolved at 6064622, where the worker stores the salt it issued (`../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/webauthn.rs:1077`). Creation requires PRF evaluation immediately, and the hybrid guard tries an attestation-only method on an assertion. Real authenticator and cross-browser acceptance remain open.

## NF-05.4 Three identity backends behind one signing seam

```mermaid
classDiagram
    class AuthStore {
        sign_event sync : auth/mod.rs:202
        get_privkey_bytes Zeroizing : auth/mod.rs:232
        login_with_local_key : auth/mod.rs:438
        logout clears storage : auth/mod.rs:676 auth/mod.rs:695
    }
    class PrfSigner {
        passkey-derived key : auth/mod.rs:714
    }
    class Nip07Signer {
        window.nostr extension : auth/nip07.rs:165
        getPublicKey : auth/nip07.rs:85
        signEvent : auth/nip07.rs:120
    }
    class LocalKey {
        pasted nsec or hex : auth/mod.rs:438
    }
    class Nip98Builder {
        kind 27235 : auth/nip98.rs:27
        u tag : auth/nip98.rs:158
        method tag : auth/nip98.rs:159
        payload sha256 : auth/nip98.rs:179
    }
    AuthStore --> PrfSigner
    AuthStore --> Nip07Signer
    AuthStore --> LocalKey
    AuthStore --> Nip98Builder

    note for Nip98Builder "create_nip98_token_with_signer routes through the Signer TRAIT, so an extension-backed session builds a valid NIP-98 header without the client ever holding a key auth/nip98.rs:149"
    note for AuthStore "Session key persistence is remember-me dependent: localStorage auth/session.rs:156 or sessionStorage auth/session.rs:163; both cleared on logout auth/session.rs:187"
    note for Nip07Signer "sign_event_async is the async seam an extension needs; the sync path is key-backed only auth/mod.rs:669"
```

## NF-05.5 Relay transport — frames in and out

```mermaid
sequenceDiagram
    autonumber
    participant C as RelayConnection<br/>nostr-bbs-forum-client/src/relay.rs:262
    participant R as relay-worker

    C->>R: connect nostr-bbs-forum-client/src/relay.rs:444
    R-->>C: open: new conn_gen, auth flags reset nostr-bbs-forum-client/src/relay.rs:497
    C->>R: queued non-REQ frames flushed, every subscription REQ replayed and counted anonymous nostr-bbs-forum-client/src/relay.rs:514 nostr-bbs-forum-client/src/relay.rs:526
    C->>C: authenticated cleared BEFORE state = Connected nostr-bbs-forum-client/src/relay.rs:553
    C->>R: REQ subscription, counted in its ledger only when the socket is open nostr-bbs-forum-client/src/relay.rs:644 nostr-bbs-forum-client/src/relay.rs:681
    R-->>C: EVENT nostr-bbs-forum-client/src/relay.rs:1238
    C->>C: verify_event_strict BEFORE dispatch nostr-bbs-forum-client/src/relay.rs:1262
    R-->>C: EOSE, credited only under the current wire id nostr-bbs-forum-client/src/relay.rs:1290 nostr-bbs-forum-client/src/relay.rs:1303
    C->>R: EVENT publish, held while the socket is open but AUTH unresolved nostr-bbs-forum-client/src/relay.rs:754 nostr-bbs-forum-client/src/relay.rs:780
    R-->>C: OK, an AUTH reply is routed to on_auth_ok, see NF-05.18 nostr-bbs-forum-client/src/relay.rs:1384 nostr-bbs-forum-client/src/relay.rs:1395
    R-->>C: AUTH challenge nostr-bbs-forum-client/src/relay.rs:1424
    C->>R: signed kind-22242 AUTH response, recorded as pending_auth nostr-bbs-forum-client/src/relay.rs:1438 nostr-bbs-forum-client/src/relay.rs:1459
    C->>R: CLOSE nostr-bbs-forum-client/src/relay.rs:738
    C->>R: COUNT (NIP-45), kept pending and re-sent once AUTH is accepted nostr-bbs-forum-client/src/relay.rs:720 nostr-bbs-forum-client/src/relay.rs:945 nostr-bbs-forum-client/src/relay.rs:1174
    R-->>C: COUNT answer nostr-bbs-forum-client/src/relay.rs:1335

    Note over C: INVARIANT: the client re-verifies every inbound event's id and Schnorr signature - it does not trust the relay nostr-bbs-forum-client/src/relay.rs:1262
    Note over C: publish_with_ack registers a callback for the relay's OK, then sends or holds the frame like publish nostr-bbs-forum-client/src/relay.rs:795
    Note over C: Every send is paced to at most 8 frames a second, below the relay's per-IP cap nostr-bbs-forum-client/src/relay.rs:46 nostr-bbs-forum-client/src/relay.rs:286
    Note over C: OPEN O22 / KF-24: a post made while the socket is CLOSED goes to pending_messages nostr-bbs-forum-client/src/relay.rs:824<br/>and is flushed on reopen BEFORE AUTH nostr-bbs-forum-client/src/relay.rs:514, where the relay refuses it auth-required
```

## NF-05.6 Derived, never accumulated — the counting invariant

```mermaid
flowchart TB
    EVS["channel_messages: HashMap of channel to Vec of events<br/>channels.rs:77"]
    TOMB["tombstones are (id, deleter), seeded from the persisted sealed and held sets<br/>channels.rs:119 channels.rs:135, tombstones.rs:67"]
    GATE1["admits_message gate BEFORE decrypt<br/>channels.rs:580"]
    DEC["zone_crypto::store::prepare_incoming - ciphertext to plaintext or placeholder<br/>channels.rs:615"]
    GATE2["admits_message gate AGAIN AFTER decrypt - a restored sealed original<br/>carries the ORIGINAL id, which a kind-5 may already have tombstoned<br/>channels.rs:622 ADR-2017"]
    INS["insert_message helper: dedup by id, then sort by created_at<br/>channels.rs:807 channels.rs:629"]
    CNT["count_for = Vec::len on every read<br/>channels.rs:305 channels.rs:307"]
    UNREAD["unread_count is a Memo filtering !read<br/>notifications.rs:271"]
    HON["honoured per target: ordinary post, every deleter counts. Restored sealed original,<br/>only its inner author, a relay-vouched admin, or the reader's own delete<br/>tombstones.rs:114 tombstones.rs:131, signer_admin.rs:108 signer_admin.rs:127"]

    EVS --> GATE1 --> DEC --> GATE2 --> INS --> CNT
    TOMB --> HON
    HON --> GATE1
    HON --> GATE2
    EVS --> UNREAD

    N1["INVARIANT ADR-091: post counts are NEVER stored as an independent field - a mutable counter drifts<br/>against deletions and replaceable events channels.rs:68"]
    N2["admits_message is the single admission predicate for the broad sync, ensure_subscribed and the deep-link replay<br/>channels.rs:267 channels.rs:720 channels.rs:737. insert_message is the only path that ADDS to channel_messages.<br/>The other mutations only remove or rewrite: apply_deletion folds a kind-5 per deleter channels.rs:194 tombstones.rs:164,<br/>remove_message for the reader's own delete channels.rs:243, the empty-EOSE reset channels.rs:478, and redecrypt,<br/>which re-admits a restored original zone_crypto/store.rs:463"]
    N3["The same discipline is applied to notifications - unread is derived, not incremented<br/>notifications.rs:271. init_sync opens NO subscription of its own,<br/>and is idempotent behind a synced flag notifications.rs:483"]
    N4["OPEN O25 / KF-27: the client cannot identify TL3 moderators tombstones.rs:17-19, and the isAdmin it asks for<br/>signer_admin.rs:108 is the whitelist row's is_admin alone whitelist.rs:135 whitelist.rs:180, while the relay's own<br/>admin test also honours static ADMIN_PUBKEYS auth.rs:181. O13 / KF-17, relay half per the kit register, not re-read<br/>here: an ordinary post honours every deleter tombstones.rs:7-10, so a claim the relay accepts on an id it never stored hides it"]
    N5["Kit #81: the sync filters are split so kind-5s cannot crowd out channels or other zones - channels 40 and 41,<br/>deletions, reactions and one kind-42 filter per zone plus a catch-all, each with an explicit limit 1000<br/>channels.rs:1006 channels.rs:1021 channels.rs:1042 channels.rs:963. The broad message sync starts only once the<br/>channel list has an authenticated answer or its deadline passed nostr-bbs-forum-client/src/app.rs:903 lookup.rs:140"]
```

## NF-05.7 Config projection into the browser

```mermaid
flowchart TB
    ENV["window.__ENV__ injected by the deployment"]
    ZC["ZONE_CONFIG read<br/>nostr-bbs-forum-client/src/stores/zones.rs:182 zones.rs:186"]
    LOAD["load_zones parse<br/>nostr-bbs-forum-client/src/stores/zones.rs:160"]
    FB["fallback_zones when absent<br/>nostr-bbs-forum-client/src/stores/zones.rs:329"]
    URLS["utils/relay_url.rs window_env reader<br/>nostr-bbs-forum-client/src/utils/relay_url.rs:151"]
    R["relay_url utils/relay_url.rs:8 | auth_api_base :36 | pod_api_base :69 | brand_label :98 | bbs_enabled :108"]
    COH["admin cohort editor is ZONE_CONFIG-driven, not a hardcoded list<br/>nostr-bbs-forum-client/src/admin/user_table.rs:19-20"]

    ENV --> ZC --> LOAD --> FB
    ENV --> URLS --> R
    ZC --> COH

    N1["The client renders what the config describes; the relay is the real boundary enforcing the SAME<br/>JSON - see NF-08.1 and NF-08.2"]
    N2["ANOMALY O8 re-verified and STILL LIVE: relay.rs reads window.__ENV__ inline<br/>nostr-bbs-forum-client/src/relay.rs:1537 instead of calling the shared window_env helper, and the two<br/>resolvers' hardcoded fallbacks DIVERGE - nostr-bbs-forum-client/src/relay.rs:30 versus utils/relay_url.rs:13"]
    N3["ANOMALY O7 CLOSED: settings no longer hardcodes example.test - it imports the onboarding modal's NIP05_HOST, which is baked from NOSTR_BBS_NIP05_DOMAIN, so the two cannot drift<br/>nostr-bbs-forum-client/src/pages/settings.rs:35, nostr-bbs-forum-client/src/components/onboarding_modal.rs:59"]
```

## NF-05.8 Signup — generated key, recovery sheet and asynchronous provisioning

```mermaid
sequenceDiagram
    autonumber
    participant U as New user
    participant S as SignupPage
    participant A as Local auth
    participant W as Auth and pod workers
    participant R as RecoverySheet
    U->>S: enter display name
    S->>A: register_with_generated_key signup.rs:259
    A-->>S: generated private hex held in signal signup.rs:261
    S->>W: async handle claim signup.rs:274
    S->>W: then attempt pod provisioning signup.rs:290
    Note over S,W: Failures log deferred work signup.rs:283 signup.rs:292<br/>Profile phase follows signup.rs:297
    S->>R: backup phase uses one recovery sheet signup.rs:641
    Note over S,R: Rust QR and master-key connect fragment<br/>recovery_sheet.rs:290 recovery_sheet.rs:294
    U->>R: print/download attempt and acknowledgement
    R-->>S: positive-only ready callback recovery_sheet.rs:303 signup.rs:647
    S->>A: finish confirms backup and applies remember preference signup.rs:309
    Note over R,S: Print failure and unticked acknowledgement do not reset parent readiness<br/>Advanced override is intentional signup.rs:674
```

[The recovery review](../../estate-review/2026-11-01-forum-recovery-and-connect.md) and NF-22 distinguish attempted output, local import and relay acceptance. This signup path generates a local key; it is not the passkey ceremony in NF-05.3. The downloaded HTML interpolates unescaped display text and differs from the screen's optional device/sweep sections. Fragment-only navigation excludes the key from the HTTP request, but does not prove WASM-only custody or successful history clearing.

**Invariant:** Settings never publishes a kind-0 over a profile it has not loaded: the save merges only the fields it manages into the member's hydrated kind-0, keeps every other field, and refuses while the existing event is still being fetched, because kind-0 is replaceable and an absent field is lost for good (`../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/settings.rs:1850`, `../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/settings.rs:1907`). SVG avatars are refused, since the pod serves stored bytes with the declared type (`../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/settings.rs:1988`).

## NF-05.9 Direct messages — two wraps per message, and why history exists

```mermaid
sequenceDiagram
    autonumber
    participant A as send_message<br/>nostr-bbs-forum-client/src/dm/mod.rs:485
    participant PAIR as gift_wrap_pair_with_signer<br/>nostr-bbs-core/src/gift_wrap.rs:769
    participant R as relay
    participant B as process_gift_wrap_event<br/>nostr-bbs-forum-client/src/dm/mod.rs:1223

    A->>A: optimistic bubble applied synchronously, publish on a spawned task dm/mod.rs:478-479
    A->>A: kind-10050 DM relay list published once per pubkey before the first wrap dm/mod.rs:611
    A->>PAIR: one rumor, sealed to the recipient and to the sender dm/mod.rs:655
    PAIR-->>A: (to_recipient, to_self) dm/mod.rs:659
    A->>A: re-key the optimistic message to the SELF copy's id so the relay echo dedups dm/mod.rs:660-668
    A->>R: publish BOTH wraps, own copy FIRST so sent history survives a failed recipient publish dm/mod.rs:693 dm/mod.rs:695 dm/mod.rs:696
    B->>R: REQ gift wraps by my own p tag, windowed by since dm/mod.rs:908 dm/mod.rs:983
    B->>R: REQ legacy kind-4, unwindowed dm/mod.rs:984
    R-->>B: wraps
    B->>B: unwrap_gift_with_signer dm/mod.rs:1229
    B->>B: InvalidKind {KIND_ZONE_KEY_GRANT} or a poker rumor is not a DM - not an error, the zone-key store or the table handles it dm/mod.rs:1236-1238
    B->>B: a failure is surfaced through state.error, not swallowed dm/mod.rs:1249-1257

    Note over A: INVARIANT a single wrap is write-only - encrypted to the recipient, authored by a throwaway key, p-tagged to the recipient alone, so the sender can neither decrypt nor find it, and sent history kept vanishing dm/mod.rs:626-630
    Note over B: ADR-2016: a zone-key grant rides the SAME gift-wrap transport (kind 1059) as a DM. Core refuses the grant's rumor kind before unwrap can succeed, so dispatch never needs a kind check of its own - dm/mod.rs:1231-1235, zone_crypto/mod.rs:93
    Note over B: The gift-wrap half CANNOT be narrowed to one partner relay-side, so it fetches the whole wrap inbox and selects the conversation locally after unwrapping - the price NIP-59 sender anonymity charges dm/mod.rs:965-969
    Note over B: The realtime window is widened by GIFT_WRAP_LOOKBACK_SECS because NIP-59 randomises the outer created_at into the PAST - a since equals now subscription would skip a message sent this second dm/mod.rs:997 dm/mod.rs:976-984
    Note over B: Tier 0 (ADR-2023): a layout-level incoming watcher subscribes to the account's own gift-wrap inbox, unwraps each wrap not yet announced and raises a DM notification, because the per-page DMStore only subscribes while a DM screen is open dm/mod.rs:1116 dm/mod.rs:1144
    Note over B: Read marks are kept per account in localStorage under nbbs:dm-read-marks and message timestamps are clamped to now, so a future-dated rumor cannot pin a conversation dm/mod.rs:1461 dm/mod.rs:1469 dm/mod.rs:1983
    Note over B: Event-id dedup makes the widened overlap free dm/mod.rs:350-351
    Note over B: ANOMALY O6 CORRECTED: the filed defect was a silent no-op NIP-07 DM subscription. The subscription is registered unconditionally dm/mod.rs:359. The NIP-07 limitation is per-event, and it IS surfaced dm/mod.rs:1249-1257
```

**Open:** ADR-2027 (NIP-17 overlay) is accepted, partly implemented and inactive, and ADR-2028 (NIP-171 key plane, superseding ADR-2026) is proposed and inactive (`../nostr-rust-forum/docs/adr/ADR-2027-nip17-overlay-for-nostr-chat-interop.md:6`, `../nostr-rust-forum/docs/adr/ADR-2028-zones-adopt-nip171-key-plane.md:5`, `:7`). Both implementations reached main with the upgrade-branch merge a80c572 and stay dormant: the extended DM path (reply and imeta rumors, multi-relay delivery) runs only with `RECIPIENT_KEYS_ENABLED` (`../nostr-rust-forum/crates/nostr-bbs-forum-client/src/dm/mod.rs:560`, `:1637-1638`), and the owner activation workflows are still missing ([NF-28](28-nip171-owner-recovery-and-ticket-contracts.md)), so neither is drawn as live here.


## NF-05.10 Feature surfaces the forum client calls out to

```mermaid
flowchart LR
    THREAD["thread reply - kind 42 e-tagging the topic root<br/>nostr-bbs-forum-client/src/pages/thread.rs:1023<br/>edit/delete via kind 5 thread.rs:1091"]
    CHAN["channel message - kind 42<br/>nostr-bbs-forum-client/src/pages/channel.rs:835"]
    EVENTS["calendar - kind 31923 events, 31925 RSVPs<br/>nostr-bbs-forum-client/src/pages/events.rs:137 events.rs:195"]
    SEARCH["global search - kind-40 channel names over the relay<br/>global_search.rs:467 plus NIP-50 text search over kind-42<br/>global_search.rs:725-760, worker as semantic layer/fallback<br/>global_search.rs:360. A hit links to /go/:event_id, not /chat/:channel - see NF-05.2 and NF-05.15"]
    BADGE["agent disclosure over public HTTP GET<br/>nostr-bbs-forum-client/src/components/agent_badge.rs:317"]
    GIT["pod git control panel - _git/status :213 stage :272 unstage :310<br/>discard :359 commit :385 diff :434 log :466<br/>nostr-bbs-forum-client/src/components/git_panel.rs:213"]
    POD["pod browser - {POD_API}/pods/{pubkey} and /HEAD<br/>nostr-bbs-forum-client/src/pages/pod_browser.rs:513 pod_browser.rs:582<br/>OPEN O26 / KF-28: mesh and pod cards print the raw probe error under Details<br/>pod_browser.rs:1045 pod_browser.rs:1058 pod_browser.rs:1181"]
    ADMINT["admin actions - POST /api/admin/suspend :269 and /api/admin/silence :701, NIP-98 signed<br/>nostr-bbs-forum-client/src/admin/user_table.rs:269"]

    N1["DOC-DRIFT consumer-surface-map: the map lists the git panel as calling /.well-known/apps, but the<br/>client fetches and PUTs /apps/manifest.json instead - git_panel.rs:893 and git_panel.rs:946; no<br/>.well-known path exists anywhere in the crate"]
    N2["Calendar: only 31923 (time-based) and 31925 (RSVP) are used client-side. Kind 31922 (date-based),<br/>which the relay ban-gates and zone-gates, has no client surface - see NF-03.6 and NF-03.12"]
    N3["DRIFT resolved: the worker's index was fed only by an admin-gated ingest call, so a member's own<br/>post was unfindable by non-admins. The relay now answers text search directly (NIP-50, matched<br/>case-insensitively against every post's content) and is authoritative; the worker is consulted only<br/>in semantic mode or when the relay returns nothing global_search.rs:336-360"]
    N4["A zone-encrypted post is never a search result either way: the relay match is on ciphertext, which<br/>relay_text_search discards client-side global_search.rs:735-742, and the semantic index is fed only<br/>public content - see NF-07.4"]
```

## NF-05.11 Retro BBS client — session adoption and key custody

Custody and boot-routing correction: [current ADR-2008 review](../../estate-review/2026-11-01-forum-bbs-signer.md). That review was narrow; the whole topic was re-read and restamped at nostr-rust-forum c39f41f on 2026-11-02 ([audit](../../estate-review/2026-11-02-forum-clients-agents-search.md)).

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/app.rs:26
    A["Boot adopts forum session"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/signer.rs:332
    B["Valid local key wins<br/>else stored NIP-07 session with provider"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/app.rs:45
    C["Initial screen: Boards if PWA and pinned<br/>else MainMenu if signed in, else Landing"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/app.rs:134
    D["PWA with no adopted signer<br/>starts background unwrap"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/pwa.rs:233
    E["Valid baked key installs signer<br/>otherwise route to Rebind"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/signer.rs:126
    F["Local and baked keys live in PrfSigner<br/>behind shared Rc handles"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-core/src/keys.rs:40
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/signer.rs:106
    G["SecretKey zeroizes at final drop<br/>retained handles can still sign after logout"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/signer.rs:229
    H["Logout clears stored handles<br/>forum storage and socket remain; see NF-19"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/relay.rs:79
    I["ADR-2016: mask kind-42 posts with zk tag<br/>BBS reader directs users to forum"]
    A --> B --> C
    B --> D --> E
    B --> F --> G --> H
    E --> F
    C -. reader .-> I
```

## NF-05.12 BBS screen machine and its own surfaces

```mermaid
flowchart TB
    ENUM["Screen enum<br/>nostr-bbs-bbs-client/src/menu.rs:14<br/>Rebind :28 MainMenu :31 Agents :33 Chat :37"]
    GO["BbsState::go - the single transition point<br/>nostr-bbs-bbs-client/src/chrome.rs:63 chrome.rs:65"]
    KEYS["from_menu_key numeric entry menu.rs:87<br/>parse_command slash entry menu.rs:186<br/>aliases table menu.rs:138<br/>menu_order ten screens menu.rs:56"]
    VIEW["ScreenView dispatch<br/>nostr-bbs-bbs-client/src/screens.rs:124"]
    PWAENTER["enter_pwa jumps straight to Boards<br/>nostr-bbs-bbs-client/src/chrome.rs:82"]
    ASCII["ASCII images are rendered SERVER-side by the preview worker<br/>nostr-bbs-bbs-client/src/ascii_img.rs:203"]
    UP["uploads PUT to the viewer's own pod, NIP-98 authed<br/>nostr-bbs-bbs-client/src/upload.rs:244 upload.rs:255"]
    IDENT["Identity::derive - DID, WebID and pod URLs, fail-closed<br/>nostr-bbs-bbs-client/src/identity.rs:28"]
    DMB["DM filters kinds [4, 1059] for my p tag and my own sent copies<br/>nostr-bbs-bbs-client/src/dm.rs:463 dm.rs:464, a send wraps a NIP-17 pair<br/>and publishes the self-copy first dm.rs:371 dm.rs:377"]

    ENUM --> GO --> VIEW
    KEYS --> GO
    PWAENTER --> GO
    VIEW --> ASCII & UP & DMB
    IDENT --> UP

    N1["Theme cycling is in-memory only - no persistence chrome.rs:97, palette parse<br/>nostr-bbs-bbs-client/src/theme.rs:23"]
    N2["Config is read from window.__ENV__ exactly as the forum client does<br/>nostr-bbs-bbs-client/src/config.rs:267, relay URL config.rs:148, pwa_mode from ?pwa=1 config.rs:186,<br/>plus an ENCRYPTION_ENABLED gate config.rs:68 config.rs:162"]
    N3["The BBS derives its passkey key through the SAME core helper as the forum client<br/>nostr-bbs-bbs-client/src/passkey.rs:179, with the PRF buffer zeroized after use nostr-bbs-bbs-client/src/passkey.rs:181"]
    N4["ADR-2016: board_composer refuses to render a reply box on an encrypted board - board_is_encrypted<br/>gates on the ENCRYPTION_ENABLED config flag and the board's zone, and the reader is pointed at the<br/>forum instead nostr-bbs-bbs-client/src/screens.rs:906-918, nostr-bbs-bbs-client/src/relay.rs:88-98"]
```

## NF-05.13 Notification ingress — why a reply addressed to you is not burned

```mermaid
stateDiagram-v2
    [*] --> Classify: kind-42 post arrives<br/>nostr-bbs-forum-client/src/stores/notifications.rs:1148
    Classify --> OwnPost: the viewer wrote it<br/>notifications.rs:1158
    Classify --> BeforeBaseline: older than the persisted first-sync floor<br/>notifications.rs:1162
    Classify --> DirectedAtMe: a p tag names me, so the read-position gate is skipped<br/>notifications.rs:1164
    Classify --> AlreadyRead: at or before the channel read position<br/>notifications.rs:1187
    DirectedAtMe --> Notify
    Classify --> Notify: genuine unseen activity<br/>notifications.rs:1189
    Notify --> [*]
    OwnPost --> [*]
    BeforeBaseline --> [*]
    AlreadyRead --> [*]

    note right of AlreadyRead
        INVARIANT an id is burned into the persisted dedup set only when its
        verdict is permanent (Notify, OwnPost, BeforeBaseline) AND, for a post
        that would notify, the notification level allows it. AlreadyRead is
        reversible, because read positions are written by render-time effects
        and can be wrong, and a level loosened later can still surface the
        post notifications.rs:1134 notifications.rs:1264
        The set is a capacity-bounded FIFO of 2000 ids restored per account
        notifications.rs:171 notifications.rs:202 notifications.rs:540-543
    end note
    note right of DirectedAtMe
        The read position is per CHANNEL, but a channel holds many topics
        and is stamped to its newest message by a render-time effect, so
        opening a section's title list marked every reply in every topic
        read having shown the reader nothing but titles
        notifications.rs:1164-1182
    end note
    note left of Notify
        Kit #92: a post from an encrypted zone notifies as
        New encrypted message in its channel, with no text
        and no author key notifications.rs:106-108
        notifications.rs:723 notifications.rs:743
    end note
    note right of Classify
        Order is deliberate - authorship, then the sync floor, then the read
        position - so the most DURABLE reason wins and a post is not filed
        under a reversible verdict when an irreversible one applies
        notifications.rs:660-679
    end note
```

## NF-05.14 Search from the client — relay text search first, the worker as semantic layer

```mermaid
sequenceDiagram
    autonumber
    participant U as Member
    participant GS as GlobalSearch<br/>nostr-bbs-forum-client/src/components/global_search.rs:223
    participant R as relay
    participant SC as search_api_base<br/>nostr-bbs-forum-client/src/utils/search_client.rs:109
    participant SW as search-worker

    U->>GS: type a query, debounced
    GS->>R: kind-40 channel names over the relay global_search.rs:467
    GS->>R: relay_text_search - REQ kind-42, NIP-50 search field global_search.rs:744-753
    R-->>GS: matching posts, up to RELAY_TEXT_LIMIT, 1.2s wait then unsubscribe global_search.rs:744-760
    GS->>GS: zone-encrypted hits (zk tag) are discarded client-side global_search.rs:733-742

    alt semantic mode OR relay returned nothing
        GS->>SC: resolve the worker base at RUNTIME search_client.rs:109
        SC-->>GS: one base for both entry points, window.__ENV__ first then a compile-time fallback search_client.rs:73-88
        GS->>SW: POST /search global_search.rs:811
        SW-->>GS: only labels the worker considers public - see NF-07.4
    end

    Note over GS: DRIFT resolved: the worker's index is fed only by an admin-gated /ingest call, so a<br/>non-admin's own post was unfindable through it alone. The relay holds every post and matches<br/>content directly, so it is now the authoritative text path global_search.rs:336-345
    Note over SC: resolve_search_base tries runtime, then compile-time, then a working fallback<br/>search_client.rs:73-81, search_api_base is resolved fresh on every call, never cached, precisely<br/>so a runtime window.__ENV__ value wins over anything baked in at build time search_client.rs:109-110
    Note over SW: The worker request body's result-count field is k - serde discards any other key<br/>(a plausible-looking limit is silently dropped) search_client.rs:119-126
```

## NF-05.15 Message deep-link — /go/:event_id resolves a bare id to a reader position

```mermaid
sequenceDiagram
    autonumber
    participant U as Member
    participant MJ as MessageJumpPage<br/>nostr-bbs-forum-client/src/pages/message_jump.rs:102
    participant R as relay
    participant CS as ChannelStore
    participant TP as ThreadPage<br/>nostr-bbs-forum-client/src/pages/thread.rs:377

    U->>MJ: navigate to /go/:event_id (search hit, or any shared link)
    MJ->>R: subscribe ids=[event_id] message_jump.rs:147-153
    R-->>MJ: the target event
    MJ->>CS: ensure_subscribed(&relay, channel_of(event)) message_jump.rs:177
    MJ->>MJ: focus = edit_target_of(event) else event.id - an edit renders as its original message_jump.rs:180
    MJ->>MJ: topic_root_of walks the reply chain (e-tag reply/root markers) up to 64 hops message_jump.rs:85-98 message_jump.rs:189
    MJ->>MJ: resolve the topic's zone via section_to_zone message_jump.rs:194-202
    MJ->>TP: navigate to /forums/:zone/:section/:topic?focus=:id, replace=true message_jump.rs:203-208

    TP->>TP: landing_target - focus post if loaded, else the newest reply thread.rs:267-283
    TP->>TP: scroll_into_view + flash, re-applied while replies stream in for LANDING_SETTLE_MS=2500ms thread.rs:207 thread.rs:689 thread.rs:298

    Note over MJ: If an ancestor is still streaming in, topic_root_of returns None and the Effect simply<br/>re-runs on the next store update - no retry loop, no timer message_jump.rs:85-98 message_jump.rs:189
    Note over MJ: RESOLVE_TIMEOUT_MS=6000: if the topic can never be resolved (unscoped channel, withheld<br/>parent, relay timeout) the page falls back to the single-note view /view/:id instead of hanging message_jump.rs:32 message_jump.rs:157-162
    Note over TP: The 2.5s settle window exists because a reply can still be en route when the page first<br/>paints, re-landing during that window is what keeps the reader ON the target post thread.rs:206-207
```

## NF-05.16 Zone encryption depends on resolved policy and locally held keys

[The zone-key audit](../../estate-review/2026-11-01-forum-zone-key-distribution.md) separates tested crypto/planning behaviour from browser-only lifecycle findings. NF-27 expands rotation, delivery and recovery.

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/app.rs:798
    G["Gate on, relay authenticated and this identity's keys loaded<br/>start member grant sync once per sign-in"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/mod.rs:752
    V["Verify wrap and seal, unwrap private rumor<br/>admin authority and key-pair validation"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/store.rs:799
    K["Persist as this identity's row<br/>a different key for the slot is refused, see NF-27.6"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/store.rs:1244
    W["Resolve channel zone before writing<br/>unknown mapping passes plaintext through"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/store.rs:1248
    E["Mapped encrypted zone uses the send key: highest held,<br/>never one pending delivery, no key means refuse and retain draft"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/stores/channels.rs:580
    T["Incoming event passes tombstone check<br/>then decrypt or keep placeholder"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/stores/channels.rs:622
    I["Check restored original ID against tombstones<br/>insert with ID deduplication"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/store.rs:462
    R["Later key arrival reopens cached events<br/>ReadCache applies deletion predicate and deduplication"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/app.rs:799
    O["Gate off stops grant sync<br/>already held keys can still decrypt"]
    G --> V --> K --> R
    W --> E
    T --> I
    K --> T
    O --> T
```

## NF-05.17 Web Push in the client — default off, staged, content-free

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/utils/push.rs:245
    CFG["GET /api/push/config - the client learns whether the relay has push on, so no rebuild is needed"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/components/push_settings.rs:27
    CARD["Settings selects Push, LocalOnly Browser alerts, or Hidden<br/>server push being off does not hide local alerts"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/utils/push.rs:262
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/utils/push.rs:302
    SUB["enable subscribes this device and registers it with a NIP-98 signed POST /api/push/subscribe<br/> "]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/utils/push.rs:418
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/utils/push.rs:429
    PREFS["server-side prefs: replies and DMs on, topics and posts off, mutes, quiet hours<br/> "]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/sw.js:335
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/sw.js:241
    SWPUSH["service worker push handler: the push is EMPTY, so it drains counters from the relay<br/> "]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/utils/push.rs:163
    LOCAL["Tier 0 local alert only while the page is hidden, permission granted and server push inactive"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/utils/push.rs:369
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/utils/push.rs:373
    OUT["logout attempts token deletion and browser cleanup<br/>registration never writes the localStorage credential copy it reads<br/> "]

    CFG --> CARD
    CARD -->|Push mode| SUB --> SWPUSH
    CARD -->|registered Push mode| PREFS
    LOCAL -. "showNotification directly" .-> SHOWN
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/utils/push.rs:447
    SHOWN["Generic system notification through worker registration"]
    OUT -.-> SUB

    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/sw.js:241
    N1["The empty push body carries no message text, author key or channel ID<br/>the service worker fetches generic counters; see NF-18 for metadata and receipt limits<br/> ADR-2023 decision 1 and 2"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/wrangler.toml:86
    N2["OPEN: ADR-2023 is accepted and complete but activation is staged - the relay ships PUSH_ENABLED false<br/> - see NF-08.8"]
```

**What it shows.** How the browser side of ADR-2023 works: the client asks the relay whether push is on, subscribes the device with a signed request, and from then on an empty push wakes the service worker, which drains content-free counters. Where server push is off, Tier 0 raises generic system notifications from the page. Logout cleanup is best effort, and its missing credential copy is documented in [the push audit](../../estate-review/2026-11-01-forum-web-push.md).

**Why it is this way.** Push payloads transit Google, Apple and Mozilla, and the service worker has no access to the page’s localStorage or sessionStorage, so ADR-2023 chose payload-less push with a bearer-token drain instead of RFC 8291 encryption (`../nostr-rust-forum/docs/adr/ADR-2023-content-free-web-push-for-pwa-alerts.md:34-35`, `../nostr-rust-forum/docs/adr/ADR-2023-content-free-web-push-for-pwa-alerts.md:38-39`).

**Open:** ADR-2023 records `activation_status: staged` (no live Cloudflare deploy, push-service round trip or iOS device run was verified), and the shipped relay template leaves `PUSH_ENABLED = "false"` (`../nostr-rust-forum/docs/adr/ADR-2023-content-free-web-push-for-pwa-alerts.md:7`, `../nostr-rust-forum/crates/nostr-bbs-relay-worker/wrangler.toml:86`).

## NF-05.18 NIP-42 AUTH acceptance, held writes and the per-subscription REQ ledger (kit #91)

```mermaid
sequenceDiagram
    autonumber
    participant C as RelayConnection<br/>nostr-bbs-forum-client/src/relay.rs:262
    participant L as ReqLedger<br/>nostr-bbs-forum-client/src/relay/receipt.rs:71
    participant R as relay-worker

    C->>R: AUTH frame, pending_auth recorded, answer timer armed nostr-bbs-forum-client/src/relay.rs:1459 nostr-bbs-forum-client/src/relay.rs:1471
    R-->>C: OK for the AUTH event id nostr-bbs-forum-client/src/relay.rs:1395
    C->>C: classify_auth_ok decides Accepted, Retry or Rejected receipt.rs:272 nostr-bbs-forum-client/src/relay.rs:960
    alt Accepted
        C->>L: replay_after_auth: every REQ counted as authenticated nostr-bbs-forum-client/src/relay.rs:1161 nostr-bbs-forum-client/src/relay.rs:1166
        C->>R: every REQ and pending COUNT re-sent, held writes flushed, paced nostr-bbs-forum-client/src/relay.rs:1174 nostr-bbs-forum-client/src/relay.rs:1199
        C->>C: authenticated set true only now nostr-bbs-forum-client/src/relay.rs:1210
    else rate-limited, below AUTH_RETRY_LIMIT 3
        C->>R: same AUTH frame re-sent after 1 s times attempts nostr-bbs-forum-client/src/relay.rs:978 receipt.rs:42
    else rejected, unsigned, or stalled
        C->>C: refuse_auth: auth_note set, held writes released so each OK false reaches its callback nostr-bbs-forum-client/src/relay.rs:1018 nostr-bbs-forum-client/src/relay.rs:1006
    end
    R-->>C: EOSE under the current wire id nostr-bbs-forum-client/src/relay.rs:1303
    C->>L: answered is authed only when every REQ sent was answered and the last went out after AUTH receipt.rs:109 nostr-bbs-forum-client/src/relay.rs:1305
    C->>C: on_receipt(authed) feeds the page receipts in NF-05.20 nostr-bbs-forum-client/src/relay.rs:1315
    C->>L: no answer within 8 s: on_watch says Settled, Resend or GiveUp after 2 re-sends receipt.rs:131 receipt.rs:50 receipt.rs:55
    C->>R: Resend: CLOSE the old wire id, REQ under a fresh one nostr-bbs-forum-client/src/relay.rs:1133 nostr-bbs-forum-client/src/relay.rs:1142

    Note over L: INVARIANT: an answer is never credited to a REQ sent after it - a late EOSE under a retired id is discarded receipt.rs:31-33 receipt.rs:184
    Note over C: Stall check: 60 s without a signature or 10 s without an answer releases held writes, timers carry conn_gen receipt.rs:60 receipt.rs:67 nostr-bbs-forum-client/src/relay.rs:1046
```

**What it shows.** Before kit #91 the `authenticated` signal meant only "AUTH frame sent", so a REQ answered anonymously could be taken for the member view. Now the relay's `OK true` is the trigger, and the ledger tells the page whether an EOSE is the authenticated answer.

## NF-05.19 Incremental kind-5 deletion sync (kit #93)

```mermaid
flowchart TB
    LOAD["Per-scope localStorage record: since marker, sealed and held tombstones<br/>deletion_sync.rs:614 deletion_sync.rs:166"]
    REQ["kind-5 filter with since = marker, limit 1000, beside the channel filters<br/>channels.rs:974 channels.rs:525 channels.rs:526"]
    OBS["Each kind-5: observe for the marker, then apply_deletion per deleter<br/>channels.rs:385 channels.rs:388 deletion_sync.rs:508"]
    EOSE["Channel EOSE: on_main_eose decides<br/>channels.rs:511 deletion_sync.rs:570"]
    PAGE["Marked sync filled the page: fetch until = oldest delivered, inclusive<br/>deletion_sync.rs:402 channels.rs:987"]
    NEXT["Each page: Done, Next or GiveUp, at most 20 pages<br/>deletion_sync.rs:424 deletion_sync.rs:123"]
    DONE["Settled: marker = min(newest delivered, now) minus 14 days, never backwards<br/>deletion_sync.rs:266 deletion_sync.rs:119"]
    GIVE["GiveUp: old marker stays<br/>OPEN O23 / KF-25: no resume cursor, the next session re-fetches the same pages<br/>deletion_sync.rs:30-34"]
    SAVE["persist the record<br/>deletion_sync.rs:628"]

    LOAD --> REQ --> OBS --> EOSE
    EOSE -->|full marked page| PAGE --> NEXT
    NEXT -->|Next| PAGE
    EOSE -->|short or unmarked| DONE
    NEXT -->|Done| DONE
    NEXT -->|GiveUp| GIVE
    DONE --> SAVE
    GIVE --> SAVE

    N1["INVARIANT: a kind-5 the relay accepts after a marker-advancing sync is at or above that marker,<br/>because the relay refuses created_at more than 7 days from its clock deletion_sync.rs:54-58 deletion_sync.rs:110"]
    N2["Only tombstones that hide a restored sealed original persist, the relay hard-deletes ordinary targets.<br/>An unmarked first sync takes the newest 1000 and never pages deletion_sync.rs:25-28"]
```

## NF-05.20 Page lookup states: loading, found, not found, timed out (kit #91)

```mermaid
flowchart TB
    URL(["deep link /forums/zone/section/topic"])
    GATED["zone_is_gated: a non-public zone needs an authenticated receipt<br/>lookup.rs:123 lookup.rs:62"]
    SEC{"thread page: channel found, or channel_receipt settles?<br/>thread.rs:640-645, the zone page applies<br/>the same rule to its sections category.rs:148"}
    TOP{"topic root found among the channel's kind-42 by id?<br/>thread.rs:485 thread.rs:502"}
    HIST["history receipt from ensure_subscribed<br/>channels.rs:761 thread.rs:651"]
    DEC["Lookup.decide: presence, then receipt, then 15 s deadline re-armed per target<br/>lookup.rs:91 lookup.rs:37 thread.rs:637"]
    LOADING(["Loading topic<br/>thread.rs:1179"])
    NF(["Topic Not Found<br/>thread.rs:1184"])
    TO(["Topic Unavailable, not claimed absent<br/>thread.rs:1188"])
    FOUND(["render the topic"])

    URL --> GATED --> SEC
    SEC -->|found| TOP
    TOP --> DEC
    HIST --> DEC
    DEC -->|found| FOUND
    DEC -->|receipt settles| NF
    DEC -->|deadline passed| TO
    DEC -->|neither| LOADING

    GAP["OPEN O21 / KF-23: no input says zone keys have not arrived thread.rs:638-655.<br/>Read from code for a sealed migrated original only: it sits in an envelope with another id<br/>until redecrypt restores it under its own id store.rs:447-449 store.rs:463, so the receipt can settle<br/>first and Not Found renders. An ordinary encrypted root keeps its id as a placeholder store.rs:409-411.<br/>Seen live 2026-11-22 for about 12 s to 29 s, then self-corrected, mechanism not reproduced"]
    GAP -.-> NF
```

**What it shows.** Kit #91 replaced "not found as soon as loading clears" with a decision that waits for the subscription that would carry the topic. The O21 window remains open in code at this stamp; kit PR #95 is not merged and nothing here describes a fix.
