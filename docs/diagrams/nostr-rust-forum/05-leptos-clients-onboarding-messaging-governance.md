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
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/section.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/grant_backlog.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/utils/pod_client.rs
  - ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/screens.rs
  - ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/theme.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/category.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/components/mobile_bottom_nav.rs
  - ../nostr-rust-forum/docs/adr/ADR-2027-nip17-overlay-for-nostr-chat-interop.md
  - ../nostr-rust-forum/docs/adr/ADR-2028-zones-adopt-nip171-key-plane.md
  - ../nostr-rust-forum/docs/adr/ADR-2023-content-free-web-push-for-pwa-alerts.md
verified_commit: 4a73a7a94bfb4d92fea03f04926cccd38c14e9af
---
Split 2026-11-23: NF-05.9, NF-05.10, NF-05.13 to NF-05.15 and NF-05.18 to NF-05.20 moved to [NF-30](30-client-messaging-search-notifications-and-sync.md) as NF-30.1 to NF-30.8; NF-05.11, NF-05.12, NF-05.16 and NF-05.17 moved to [NF-31](31-retro-bbs-client-zone-encryption-and-web-push.md) as NF-31.1 to NF-31.4. The old IDs resolve through the aliases declared there.

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
    N4["/go/:event_id resolves a bare event id to its topic and lands the reader on it - see NF-30.5 for<br/>the deep-link walk and NF-05.6 for the tombstone/zone-decrypt gate it shares with normal ingest"]
    N5["DEBT: three auth gates are written out by hand - AuthGatedChannel nostr-bbs-forum-client/src/app.rs:1471, AuthGatedDmList nostr-bbs-forum-client/src/app.rs:1498,<br/>AuthGatedDmChat nostr-bbs-forum-client/src/app.rs:1525 - each the same body as the auth_gated! macro nostr-bbs-forum-client/src/app.rs:1566-1594 that the other<br/>thirteen gated pages use nostr-bbs-forum-client/src/app.rs:1596-1608"]
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
    participant C as RelayConnection<br/>nostr-bbs-forum-client/src/relay.rs:266
    participant R as relay-worker

    C->>R: connect nostr-bbs-forum-client/src/relay.rs:453
    R-->>C: open: new conn_gen, auth flags reset nostr-bbs-forum-client/src/relay.rs:506
    C->>R: queued non-REQ frames flushed, every subscription REQ replayed and counted anonymous nostr-bbs-forum-client/src/relay.rs:525 nostr-bbs-forum-client/src/relay.rs:537
    C->>C: authenticated cleared BEFORE state = Connected nostr-bbs-forum-client/src/relay.rs:564
    C->>R: REQ subscription, counted in its ledger only when the socket is open nostr-bbs-forum-client/src/relay.rs:655 nostr-bbs-forum-client/src/relay.rs:692
    R-->>C: EVENT nostr-bbs-forum-client/src/relay.rs:1318
    C->>C: verify_event_strict BEFORE dispatch nostr-bbs-forum-client/src/relay.rs:1342
    R-->>C: EOSE, credited only under the current wire id nostr-bbs-forum-client/src/relay.rs:1370 nostr-bbs-forum-client/src/relay.rs:1383
    C->>R: EVENT publish, held while the socket is open but AUTH unresolved nostr-bbs-forum-client/src/relay.rs:770 nostr-bbs-forum-client/src/relay.rs:796
    R-->>C: OK, an AUTH reply is routed to on_auth_ok, see NF-30.6 nostr-bbs-forum-client/src/relay.rs:1464 nostr-bbs-forum-client/src/relay.rs:1475
    R-->>C: AUTH challenge nostr-bbs-forum-client/src/relay.rs:1504
    C->>R: signed kind-22242 AUTH response, recorded as pending_auth nostr-bbs-forum-client/src/relay.rs:1518 nostr-bbs-forum-client/src/relay.rs:1539
    C->>R: CLOSE nostr-bbs-forum-client/src/relay.rs:749
    C->>R: COUNT (NIP-45), kept pending and re-sent once AUTH is accepted nostr-bbs-forum-client/src/relay.rs:731 nostr-bbs-forum-client/src/relay.rs:961 nostr-bbs-forum-client/src/relay.rs:1254
    R-->>C: COUNT answer nostr-bbs-forum-client/src/relay.rs:1415

    Note over C: INVARIANT: the client re-verifies every inbound event's id and Schnorr signature - it does not trust the relay nostr-bbs-forum-client/src/relay.rs:1342
    Note over C: publish_with_ack registers a callback for the relay's OK, then sends or holds the frame like publish nostr-bbs-forum-client/src/relay.rs:811
    Note over C: Every send is paced to at most 8 frames a second, below the relay's per-IP cap nostr-bbs-forum-client/src/relay.rs:46 nostr-bbs-forum-client/src/relay.rs:290
    Note over C: OPEN O22 / KF-24: a post made while the socket is CLOSED goes to pending_messages nostr-bbs-forum-client/src/relay.rs:840<br/>and is flushed on reopen BEFORE AUTH nostr-bbs-forum-client/src/relay.rs:525, where the relay refuses it auth-required
```

## NF-05.6 Derived, never accumulated — the counting invariant

```mermaid
flowchart TB
    EVS["channel_messages: HashMap of channel to Vec of events<br/>channels.rs:77"]
    TOMB["tombstones are (id, deleter), seeded from the persisted sealed and held sets<br/>channels.rs:119 channels.rs:135, tombstones.rs:69"]
    GATE1["admits_message gate BEFORE decrypt<br/>channels.rs:580"]
    DEC["zone_crypto::store::prepare_incoming - ciphertext to plaintext or placeholder<br/>channels.rs:615"]
    GATE2["admits_message gate AGAIN AFTER decrypt - a restored sealed original<br/>carries the ORIGINAL id, which a kind-5 may already have tombstoned<br/>channels.rs:622 ADR-2017"]
    INS["insert_message helper: dedup by id, then sort by created_at<br/>channels.rs:807 channels.rs:629"]
    CNT["count_for = Vec::len on every read<br/>channels.rs:305 channels.rs:307"]
    UNREAD["unread_count is a Memo filtering !read<br/>notifications.rs:271"]
    HON["honoured per target: ordinary post, every deleter counts. Restored sealed original,<br/>only its inner author, a relay-vouched admin, or the reader's own delete<br/>tombstones.rs:116 tombstones.rs:133, signer_admin.rs:108 signer_admin.rs:127"]

    EVS --> GATE1 --> DEC --> GATE2 --> INS --> CNT
    TOMB --> HON
    HON --> GATE1
    HON --> GATE2
    EVS --> UNREAD

    N1["INVARIANT ADR-091: post counts are NEVER stored as an independent field - a mutable counter drifts<br/>against deletions and replaceable events channels.rs:68"]
    N2["admits_message is the single admission predicate for the broad sync, ensure_subscribed and the deep-link replay<br/>channels.rs:267 channels.rs:720 channels.rs:737. insert_message is the only path that ADDS to channel_messages.<br/>The other mutations only remove or rewrite: apply_deletion folds a kind-5 per deleter channels.rs:194 tombstones.rs:166,<br/>remove_message for the reader's own delete channels.rs:243, the empty-EOSE reset channels.rs:478, and redecrypt,<br/>which re-admits a restored original zone_crypto/store.rs:502"]
    N3["The same discipline is applied to notifications - unread is derived, not incremented<br/>notifications.rs:271. init_sync opens NO subscription of its own,<br/>and is idempotent behind a synced flag notifications.rs:483"]
    N4["OPEN O25 / KF-27: the client cannot identify TL3 moderators tombstones.rs:19-21, and the isAdmin it asks for<br/>signer_admin.rs:108 is the whitelist row's is_admin alone whitelist.rs:135 whitelist.rs:180, while the relay's own<br/>admin test also honours static ADMIN_PUBKEYS auth.rs:181. O13 / KF-17, relay half in NF-10.14, not re-read<br/>here: an ordinary post honours every deleter tombstones.rs:7-10, so a claim the relay accepts on an id it never stored hides it"]
    N5["Kit #81: the sync filters are split so kind-5s cannot crowd out channels or other zones - channels 40 and 41,<br/>deletions, reactions and one kind-42 filter per zone plus a catch-all, each with an explicit limit 1000<br/>channels.rs:1006 channels.rs:1021 channels.rs:1042 channels.rs:963. The broad message sync starts only once the<br/>channel list has an authenticated answer or its deadline passed nostr-bbs-forum-client/src/app.rs:903 lookup.rs:260,<br/>because start_msg_sync builds its per-zone filters from the channels known at that instant and the pre-AUTH answer<br/>omits Hidden-zone channels nostr-bbs-forum-client/src/app.rs:887-892; the deadline is armed at relay connect nostr-bbs-forum-client/src/app.rs:896-897"]
    N6["DEBT, two kind-42 subscriptions write one store: the broad start_msg_sync channels.rs:560 and the per-channel<br/>ensure_subscribed channels.rs:693 both pass admits_message and insert_message, which dedups by id channels.rs:807-809.<br/>The broad one is per-zone #e filters plus a catch-all with no #e message_sync_filters channels.rs:1042 channels.rs:1076,<br/>so a deep-linked channel's history can arrive twice - overlapping authority and redundant relay traffic, not a count error"]
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
    N2["ANOMALY O8 re-verified and STILL LIVE: relay.rs reads window.__ENV__ inline<br/>nostr-bbs-forum-client/src/relay.rs:1617 instead of calling the shared window_env helper, and the two<br/>resolvers' hardcoded fallbacks DIVERGE - nostr-bbs-forum-client/src/relay.rs:30 versus utils/relay_url.rs:13"]
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
    Note over S,W: DEBT O8: provision_pod has three near-identical copies - signup.rs:176-204, settings.rs:2024-2052<br/>and provision_pod_signer pod_client.rs:49-78. Settings provisions again on a 403 or 404 settings.rs:658-659<br/>after upload_to_pod_signer has already provisioned and retried on either pod_client.rs:38-40
    Note over S,W: RESOLVED F-10: settings imports only the username helpers from onboarding_modal settings.rs:18-21,<br/>and the OnboardingModal component is recorded as removed dead code onboarding_modal.rs:187-193
```

[The recovery review](../../estate-review/2026-11-01-forum-recovery-and-connect.md) and NF-22 distinguish attempted output, local import and relay acceptance. This signup path generates a local key; it is not the passkey ceremony in NF-05.3. The downloaded HTML interpolates unescaped display text and differs from the screen's optional device/sweep sections. Fragment-only navigation excludes the key from the HTTP request, but does not prove WASM-only custody or successful history clearing.

**Invariant:** Settings never publishes a kind-0 over a profile it has not loaded: the save merges only the fields it manages into the member's hydrated kind-0, keeps every other field, and refuses while the existing event is still being fetched, because kind-0 is replaceable and an absent field is lost for good (`../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/settings.rs:1850`, `../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/settings.rs:1907`). SVG avatars are refused, since the pod serves stored bytes with the declared type (`../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/settings.rs:1988`).
