---
id: NF-05
title: The two Leptos clients — boot, identity, transport, onboarding, messaging and the retro BBS
area: nostr-rust-forum
governing:
  - ../nostr-rust-forum/docs/BASELINE-architecture.md
  - ../nostr-rust-forum/docs/IDENTITY-keys-and-trust.md
adrs: [ADR-2008]
sources:
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/main.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/app.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/relay.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/auth/mod.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/auth/passkey.rs
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
verified_commit: 72463fbde35ac4c68539b1f65a08ff03b9941201
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
    A->>A: provide_auth nostr-bbs-forum-client/src/app.rs:319, provide_zone_access nostr-bbs-forum-client/src/app.rs:320, provide_render_tier nostr-bbs-forum-client/src/app.rs:321
    A->>A: provide_toasts nostr-bbs-forum-client/src/app.rs:324, wallet::provide_wallet nostr-bbs-forum-client/src/app.rs:338, provide_profile_cache nostr-bbs-forum-client/src/app.rs:329
    A->>A: start_admin_alerts nostr-bbs-forum-client/src/app.rs:347, provide_agent_disclosure nostr-bbs-forum-client/src/app.rs:438
    A->>A: RelayConnection::new nostr-bbs-forum-client/src/app.rs:446, provide_channel_store nostr-bbs-forum-client/src/app.rs:448, zone_crypto::store::provide_zone_key_store nostr-bbs-forum-client/src/app.rs:451

    Note over A: wallet::provide_wallet (ADR-2015) provides member-wallet context on sidestr:dreamlab, inert unless the SIDESTR_WALLET deployment gate is on - nostr-bbs-forum-client/src/app.rs:335-338, wallet/mod.rs:58 wallet/mod.rs:239
    Note over A: INVARIANT ADR-090 - FORUM_BASE is applied in exactly TWO places: the const consumed by Router base= at nostr-bbs-forum-client/src/app.rs:44 and :902
    Note over A: and base_href() for every link at nostr-bbs-forum-client/src/app.rs:53, used e.g. at :909. A third application re-introduces the double-prefix / deep-route-404 bug class.
    Note over M: main.rs:55-57 re-read the same option_env! for the service-worker SCOPE - a deliberate, documented mirror of app::FORUM_BASE, not a third application of the prefix
    Note over A: current_app_path strips the prefix back off a browser path so the router sees an unprefixed route nostr-bbs-forum-client/src/app.rs:82
```

## NF-05.2 Route table

```mermaid
flowchart TB
    subgraph pubroutes["Public"]
        R1["/ HomeOrForums nostr-bbs-forum-client/src/app.rs:920"]
        R2["/about nostr-bbs-forum-client/src/app.rs:921 | /login nostr-bbs-forum-client/src/app.rs:922 | /signup nostr-bbs-forum-client/src/app.rs:927"]
        R3["/connect magic link - must NOT be auth-gated nostr-bbs-forum-client/src/app.rs:926"]
        R4["/glossary nostr-bbs-forum-client/src/app.rs:931 | /go/:event_id MessageJumpPage nostr-bbs-forum-client/src/app.rs:933 | /join/:code nostr-bbs-forum-client/src/app.rs:938"]
    end
    subgraph authed["Auth-gated"]
        R5["/setup nostr-bbs-forum-client/src/app.rs:940 | /forums nostr-bbs-forum-client/src/app.rs:954 | /settings nostr-bbs-forum-client/src/app.rs:964"]
        R6["/chat/:channel_id nostr-bbs-forum-client/src/app.rs:951 | /dm nostr-bbs-forum-client/src/app.rs:952 | /dm/:pubkey nostr-bbs-forum-client/src/app.rs:953"]
        R7["/forums/:category/board nostr-bbs-forum-client/src/app.rs:959 | /:section nostr-bbs-forum-client/src/app.rs:960 | /:topic nostr-bbs-forum-client/src/app.rs:961"]
        R8["/events nostr-bbs-forum-client/src/app.rs:962 | /profile/:pubkey nostr-bbs-forum-client/src/app.rs:963 | /pod nostr-bbs-forum-client/src/app.rs:973"]
        R14["/wallet AuthGatedWallet - member wallets, ADR-2015 nostr-bbs-forum-client/src/app.rs:974"]
    end
    subgraph gov["Governance"]
        R9["/governance member read-only nostr-bbs-forum-client/src/app.rs:971"]
        R10["/governance/admin admin write nostr-bbs-forum-client/src/app.rs:972"]
        R11["/admin - its own internal gate nostr-bbs-forum-client/src/app.rs:965"]
    end
    subgraph zonealias["Zone-slug aliases, declared LAST"]
        R12["/:category nostr-bbs-forum-client/src/app.rs:993 | /:category/board nostr-bbs-forum-client/src/app.rs:995"]
        R13["/:category/:section nostr-bbs-forum-client/src/app.rs:996 | /:category/:section/:topic nostr-bbs-forum-client/src/app.rs:997"]
    end

    ROUTER["Router base=FORUM_BASE nostr-bbs-forum-client/src/app.rs:902"] --> pubroutes & authed & gov & zonealias

    N1["/chat with no channel redirects to /forums - a legacy path kept alive nostr-bbs-forum-client/src/app.rs:950"]
    N2["The zone-slug aliases are declared last so the static routes out-score them; a slug is only valid<br/>if it resolves in the live ZONE_CONFIG nostr-bbs-forum-client/src/pages/category.rs:102"]
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
    PK->>AW: POST /auth/register/verify auth/passkey.rs:169
    U->>AS: log in
    AS->>PK: authenticate_passkey auth/mod.rs:328
    PK->>AW: POST /auth/login/options auth/passkey.rs:237
    PK->>PK: extract PRF from assertion auth/webauthn.rs:169
    PK->>PK: re-derive the SAME keypair auth/passkey.rs:264
    PK->>AW: verify call, itself NIP-98 authenticated auth/passkey.rs:290
    AS->>AS: PrfSigner built from the derived key auth/mod.rs:707

    Note over AS: The private key never leaves the device - it is held out of the reactive signal graph in a StoredValue auth/mod.rs:107 and handed out only as a Zeroizing buffer auth/mod.rs:232
    Note over AS: A pagehide listener zeroizes on navigation away auth/mod.rs:824
    Note over PK: EXTERNAL: the server half of this ceremony is NF-02.3 and NF-02.4
```

## NF-05.4 Three identity backends behind one signing seam

```mermaid
classDiagram
    class AuthStore {
        sign_event sync : auth/mod.rs:202
        get_privkey_bytes Zeroizing : auth/mod.rs:232
        login_with_local_key : auth/mod.rs:438
        logout clears storage : auth/mod.rs:676 auth/mod.rs:688
    }
    class PrfSigner {
        passkey-derived key : auth/mod.rs:707
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

    note for Nip98Builder "create_nip98_token_with_signer routes through the Signer TRAIT, so an extension-backed session builds a valid NIP-98 header without the client ever holding a key auth/nip98.rs:148"
    note for AuthStore "Session key persistence is remember-me dependent: localStorage auth/session.rs:156 or sessionStorage auth/session.rs:163; both cleared on logout auth/session.rs:187"
    note for Nip07Signer "sign_event_async is the async seam an extension needs; the sync path is key-backed only auth/mod.rs:669"
```

## NF-05.5 Relay transport — frames in and out

```mermaid
sequenceDiagram
    autonumber
    participant C as RelayConnection<br/>nostr-bbs-forum-client/src/relay.rs:188
    participant R as relay-worker

    C->>R: connect nostr-bbs-forum-client/src/relay.rs:348
    R-->>C: open, state = Connected nostr-bbs-forum-client/src/relay.rs:397
    C->>R: REQ subscription nostr-bbs-forum-client/src/relay.rs:513 nostr-bbs-forum-client/src/relay.rs:534
    R-->>C: EVENT nostr-bbs-forum-client/src/relay.rs:700
    C->>C: verify_event_strict BEFORE dispatch nostr-bbs-forum-client/src/relay.rs:720
    R-->>C: EOSE nostr-bbs-forum-client/src/relay.rs:748
    C->>R: EVENT publish nostr-bbs-forum-client/src/relay.rs:563
    R-->>C: OK nostr-bbs-forum-client/src/relay.rs:800
    R-->>C: AUTH challenge nostr-bbs-forum-client/src/relay.rs:823
    C->>R: signed kind-22242 AUTH response nostr-bbs-forum-client/src/relay.rs:837
    C->>R: CLOSE nostr-bbs-forum-client/src/relay.rs:549

    Note over C: INVARIANT: the client re-verifies every inbound event's id and Schnorr signature - it does not trust the relay nostr-bbs-forum-client/src/relay.rs:720
    Note over C: publish_with_ack awaits the relay's OK rather than fire-and-forget nostr-bbs-forum-client/src/relay.rs:602
    Note over C: The client half of NIP-42 is fully wired: it signs the challenge on demand - see NF-03.2 and NF-03.3
```

## NF-05.6 Derived, never accumulated — the counting invariant

```mermaid
flowchart TB
    EVS["channel_messages: HashMap of channel to Vec of events<br/>channels.rs:71"]
    TOMB["tombstones: HashSet of deleted ids (kind-5, NIP-09)<br/>channels.rs:93 channels.rs:98"]
    GATE1["admits_message gate BEFORE decrypt<br/>channels.rs:405"]
    DEC["zone_crypto::store::prepare_incoming - ciphertext to plaintext or placeholder<br/>channels.rs:440"]
    GATE2["admits_message gate AGAIN AFTER decrypt - a restored sealed original<br/>carries the ORIGINAL id, which a kind-5 may already have tombstoned<br/>channels.rs:446 ADR-2017"]
    INS["insert_message helper: dedup by id, then sort by created_at<br/>channels.rs:611"]
    CNT["count_for = Vec::len on every read<br/>channels.rs:159 channels.rs:161"]
    UNREAD["unread_count is a Memo filtering !read<br/>notifications.rs:159"]

    EVS --> GATE1 --> DEC --> GATE2 --> INS --> CNT
    TOMB --> GATE1
    TOMB --> GATE2
    EVS --> UNREAD

    N1["INVARIANT ADR-091: post counts are NEVER stored as an independent field - a mutable counter drifts<br/>against deletions and replaceable events channels.rs:62"]
    N2["admits_message is the tombstone-set inverse - insert_message is the ONLY writer of channel_messages,<br/>so dedup-by-id plus the double tombstone gate is what makes len() correct even when a deletion and<br/>its target race channels.rs:609 channels.rs:828"]
    N3["The same discipline is applied to notifications - unread is derived, not incremented<br/>notifications.rs:185. init_sync opens NO subscription of its own,<br/>and is idempotent behind a synced flag notifications.rs:324"]
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
    N2["ANOMALY O8 re-verified and STILL LIVE: relay.rs reads window.__ENV__ inline<br/>nostr-bbs-forum-client/src/relay.rs:974 instead of calling the shared window_env helper, and the two<br/>resolvers' hardcoded fallbacks DIVERGE - nostr-bbs-forum-client/src/relay.rs:22 versus utils/relay_url.rs:9"]
    N3["ANOMALY O7 re-verified, narrower than filed: NIP05_USERNAME_HOST is hardcoded example.test at<br/>nostr-bbs-forum-client/src/pages/settings.rs:32 and used at settings.rs:334, but signup.rs does NOT<br/>read it at all - so it is a settings-only display fallback, not a split env read"]
```

## NF-05.8 Signup — key, recovery sheet, pod

```mermaid
sequenceDiagram
    autonumber
    participant U as New user
    participant S as SignupPage<br/>nostr-bbs-forum-client/src/pages/signup.rs:207
    participant RS as RecoverySheet<br/>nostr-bbs-forum-client/src/components/recovery_sheet.rs:294
    participant POD as pod-worker

    U->>S: create account (passkey ceremony, see NF-05.3)
    S->>POD: eager provision_pod during creation signup.rs:290
    S->>POD: POST {POD_API}/pods/{pubkey}/.provision, NIP-98 authed signup.rs:136
    S->>RS: render the printable recovery sheet signup.rs:641
    RS->>RS: nsec QR generated in-WASM recovery_sheet.rs:66 recovery_sheet.rs:294
    RS->>RS: /connect magic link with the nsec in the URL FRAGMENT recovery_sheet.rs:290

    Note over RS: The sheet is 100 percent client-side - the nsec travels in the fragment (#k=), which browsers do not send to the server
    Note over S: ANOMALY O8 re-verified: a SECOND provision_pod lives in settings.rs:1752, triggered when an avatar upload 404s settings.rs:585 - two independent implementations of the same call
    Note over POD: EXTERNAL: the pod side of provisioning is NF-04 - the pod server itself is solid-pod-rs (SP-*)
```

## NF-05.9 Direct messages — two wraps per message, and why history exists

```mermaid
sequenceDiagram
    autonumber
    participant A as send_message<br/>nostr-bbs-forum-client/src/dm/mod.rs:355
    participant PAIR as gift_wrap_pair_with_signer<br/>nostr-bbs-core/src/gift_wrap.rs:615
    participant R as relay
    participant B as process_gift_wrap_event<br/>nostr-bbs-forum-client/src/dm/mod.rs:812

    A->>A: optimistic bubble applied synchronously, publish on a spawned task dm/mod.rs:351-352
    A->>PAIR: one rumor, sealed to the recipient and to the sender dm/mod.rs:466
    PAIR-->>A: (to_recipient, to_self) dm/mod.rs:466
    A->>A: re-key the optimistic message to the SELF copy's id so the relay echo dedups dm/mod.rs:468-475
    A->>R: publish BOTH wraps dm/mod.rs:483 dm/mod.rs:484
    B->>R: REQ gift wraps by my own p tag, windowed by since dm/mod.rs:579 dm/mod.rs:654
    B->>R: REQ legacy kind-4, unwindowed dm/mod.rs:655
    R-->>B: wraps
    B->>B: unwrap_gift_with_signer dm/mod.rs:818
    B->>B: InvalidKind {KIND_ZONE_KEY_GRANT} is not a DM - not an error, the zone-key store handles it dm/mod.rs:825-827
    B->>B: a failure is surfaced through state.error, not swallowed dm/mod.rs:836-844

    Note over A: INVARIANT a single wrap is write-only - encrypted to the recipient, authored by a throwaway key, p-tagged to the recipient alone, so the sender can neither decrypt nor find it, and sent history kept vanishing dm/mod.rs:443-447
    Note over B: ADR-2016: a zone-key grant rides the SAME gift-wrap transport (kind 1059) as a DM. Core refuses the grant's rumor kind before unwrap can succeed, so dispatch never needs a kind check of its own - dm/mod.rs:820-824, zone_crypto/mod.rs:87
    Note over B: The gift-wrap half CANNOT be narrowed to one partner relay-side, so it fetches the whole wrap inbox and selects the conversation locally after unwrapping - the price NIP-59 sender anonymity charges dm/mod.rs:636-640
    Note over B: The realtime window is widened by GIFT_WRAP_LOOKBACK_SECS because NIP-59 randomises the outer created_at into the PAST - a since equals now subscription would skip a message sent this second dm/mod.rs:668 dm/mod.rs:647-655
    Note over B: Event-id dedup makes the widened overlap free dm/mod.rs:244-245
    Note over B: ANOMALY O6 CORRECTED: the filed defect was a silent no-op NIP-07 DM subscription. The subscription is registered unconditionally dm/mod.rs:253. The NIP-07 limitation is per-event, and it IS surfaced dm/mod.rs:836-844
```

## NF-05.10 Feature surfaces the forum client calls out to

```mermaid
flowchart LR
    THREAD["thread reply - kind 42 e-tagging the topic root<br/>nostr-bbs-forum-client/src/pages/thread.rs:1006<br/>edit/delete via kind 5 thread.rs:1074"]
    CHAN["channel message - kind 42<br/>nostr-bbs-forum-client/src/pages/channel.rs:835"]
    EVENTS["calendar - kind 31923 events, 31925 RSVPs<br/>nostr-bbs-forum-client/src/pages/events.rs:137 events.rs:195"]
    SEARCH["global search - kind-40 channel names over the relay<br/>global_search.rs:467 plus NIP-50 text search over kind-42<br/>global_search.rs:725-760, worker as semantic layer/fallback<br/>global_search.rs:360. A hit links to /go/:event_id, not /chat/:channel - see NF-05.2 and NF-05.15"]
    BADGE["agent disclosure over public HTTP GET<br/>nostr-bbs-forum-client/src/components/agent_badge.rs:317"]
    GIT["pod git control panel - _git/status :213 stage :272 unstage :310<br/>discard :359 commit :385 diff :434 log :466<br/>nostr-bbs-forum-client/src/components/git_panel.rs:213"]
    POD["pod browser - {POD_API}/pods/{pubkey} and /HEAD<br/>nostr-bbs-forum-client/src/pages/pod_browser.rs:513 pod_browser.rs:582"]
    ADMINT["admin actions - POST /api/admin/suspend :269 and /api/admin/silence :701, NIP-98 signed<br/>nostr-bbs-forum-client/src/admin/user_table.rs:266"]

    N1["DOC-DRIFT consumer-surface-map: the map lists the git panel as calling /.well-known/apps, but the<br/>client fetches and PUTs /apps/manifest.json instead - git_panel.rs:893 and git_panel.rs:946; no<br/>.well-known path exists anywhere in the crate"]
    N2["Calendar: only 31923 (time-based) and 31925 (RSVP) are used client-side. Kind 31922 (date-based),<br/>which the relay ban-gates and zone-gates, has no client surface - see NF-03.6 and NF-03.12"]
    N3["DRIFT resolved: the worker's index was fed only by an admin-gated ingest call, so a member's own<br/>post was unfindable by non-admins. The relay now answers text search directly (NIP-50, matched<br/>case-insensitively against every post's content) and is authoritative; the worker is consulted only<br/>in semantic mode or when the relay returns nothing global_search.rs:336-360"]
    N4["A zone-encrypted post is never a search result either way: the relay match is on ciphertext, which<br/>relay_text_search discards client-side global_search.rs:735-742, and the semantic index is fed only<br/>public content - see NF-07.4"]
```

## NF-05.11 Retro BBS client — session adoption and key custody

```mermaid
stateDiagram-v2
    [*] --> Boot: bbs-client app.rs
    Boot --> Adopt: adopt_forum_session on boot<br/>nostr-bbs-bbs-client/src/app.rs:26
    Adopt --> MainMenu: has_key, session adopted<br/>nostr-bbs-bbs-client/src/app.rs:48
    Adopt --> Landing: no key
    Boot --> PwaUnwrap: pwa AND not has_key<br/>nostr-bbs-bbs-client/src/app.rs:132
    PwaUnwrap --> MainMenu: adopt_baked_or_rebind<br/>nostr-bbs-bbs-client/src/pwa.rs:228
    Landing --> Settings: sign-in CTA<br/>nostr-bbs-bbs-client/src/screens.rs:219

    note right of Adopt
        choose_adoption is a PURE decision: LocalKey, Nip07 or None
        nostr-bbs-bbs-client/src/signer.rs:327
        A valid local key WINS over a NIP-07 extension
        nostr-bbs-bbs-client/src/signer.rs:332
    end note
    note right of PwaUnwrap
        Zone-bound one-shot PWA: bake_local binds the baked key to a zone
        nostr-bbs-bbs-client/src/pwa.rs:185 and persists a BootProfile
        nostr-bbs-bbs-client/src/pwa.rs:219 nostr-bbs-bbs-client/src/pwa.rs:466
        The nostr-bbs-bbs-client/src/app.rs:132 guard is what makes it ONE-SHOT
    end note
    note right of MainMenu
        ADR-2008 divergence, re-verified and REFINED: BbsSigner does not hold a bare
        SecretKey field - it holds an Rc dyn Signer handle
        nostr-bbs-bbs-client/src/signer.rs:67. A SecretKey is constructed transiently
        nostr-bbs-bbs-client/src/signer.rs:209, the stack buffer is zeroized
        nostr-bbs-bbs-client/src/signer.rs:210, and the key is moved into a PrfSigner
        nostr-bbs-bbs-client/src/signer.rs:126. Zeroize-on-drop is delegated to
        nostr_bbs_core::SecretKey nostr-bbs-bbs-client/src/signer.rs:34 - there is no
        Drop impl on BbsSigner itself. The ADR-105 sign()-seam divergence stands; the
        custody is one layer better than the anomaly register describes.
    end note
    note right of MainMenu
        ADR-2016: every ingested kind-42 is masked before it reaches a
        bucket - ingest calls mask_encrypted, which replaces a
        zone-encrypted post's content with ENCRYPTED_PLACEHOLDER
        nostr-bbs-bbs-client/src/relay.rs:44-45 nostr-bbs-bbs-client/src/relay.rs:79. This client holds no
        zone keys, so the reader is pointed at the forum instead of
        shown ciphertext.
    end note
```

## NF-05.12 BBS screen machine and its own surfaces

```mermaid
flowchart TB
    ENUM["Screen enum<br/>nostr-bbs-bbs-client/src/menu.rs:14<br/>Rebind :29 MainMenu :32 Agents :34 Chat :38"]
    GO["BbsState::go - the single transition point<br/>nostr-bbs-bbs-client/src/chrome.rs:63 chrome.rs:65"]
    KEYS["from_menu_key numeric entry menu.rs:87<br/>parse_command slash entry menu.rs:186<br/>aliases table menu.rs:138<br/>menu_order ten screens menu.rs:56"]
    VIEW["ScreenView dispatch<br/>nostr-bbs-bbs-client/src/screens.rs:124"]
    PWAENTER["enter_pwa jumps straight to Boards<br/>nostr-bbs-bbs-client/src/chrome.rs:82"]
    ASCII["ASCII images are rendered SERVER-side by the preview worker<br/>nostr-bbs-bbs-client/src/ascii_img.rs:203"]
    UP["uploads PUT to the viewer's own pod, NIP-98 authed<br/>nostr-bbs-bbs-client/src/upload.rs:244 upload.rs:255"]
    IDENT["Identity::derive - DID, WebID and pod URLs, fail-closed<br/>nostr-bbs-bbs-client/src/identity.rs:28"]
    DMB["DM filter kinds [4, 1059] scoped to my p tag<br/>nostr-bbs-bbs-client/src/dm.rs:388, wrap delegated to core dm.rs:264"]

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
    [*] --> Classify: kind-42 post arrives<br/>nostr-bbs-forum-client/src/stores/notifications.rs:882
    Classify --> OwnPost: the viewer wrote it<br/>notifications.rs:892
    Classify --> BeforeBaseline: older than the persisted first-sync floor<br/>notifications.rs:896
    Classify --> DirectedAtMe: a p tag names me, so the read-position gate is skipped<br/>notifications.rs:898
    Classify --> AlreadyRead: at or before the channel read position<br/>notifications.rs:921
    DirectedAtMe --> Notify
    Classify --> Notify: genuine unseen activity<br/>notifications.rs:923
    Notify --> [*]
    OwnPost --> [*]
    BeforeBaseline --> [*]
    AlreadyRead --> [*]

    note right of AlreadyRead
        INVARIANT only a PERMANENT verdict is burned into the persisted
        dedup set notifications.rs:517. AlreadyRead is reversible, because
        read positions are written by render-time effects and can be wrong,
        so a cheap re-check on a later pass is worth more than a permanent
        veto notifications.rs:515-529
    end note
    note right of DirectedAtMe
        The read position is per CHANNEL, but a channel holds many topics
        and is stamped to its newest message by a render-time effect, so
        opening a section's title list marked every reply in every topic
        read having shown the reader nothing but titles
        notifications.rs:898-916
    end note
    note right of Classify
        Order is deliberate - authorship, then the sync floor, then the read
        position - so the most DURABLE reason wins and a post is not filed
        under a reversible verdict when an irreversible one applies
        notifications.rs:495-514
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
    participant TP as ThreadPage<br/>nostr-bbs-forum-client/src/pages/thread.rs:376

    U->>MJ: navigate to /go/:event_id (search hit, or any shared link)
    MJ->>R: subscribe ids=[event_id] message_jump.rs:147-153
    R-->>MJ: the target event
    MJ->>CS: ensure_subscribed(&relay, channel_of(event)) message_jump.rs:177
    MJ->>MJ: focus = edit_target_of(event) else event.id - an edit renders as its original message_jump.rs:180
    MJ->>MJ: topic_root_of walks the reply chain (e-tag reply/root markers) up to 64 hops message_jump.rs:85-98 message_jump.rs:189
    MJ->>MJ: resolve the topic's zone via section_to_zone message_jump.rs:194-202
    MJ->>TP: navigate to /forums/:zone/:section/:topic?focus=:id, replace=true message_jump.rs:203-208

    TP->>TP: landing_target - focus post if loaded, else the newest reply thread.rs:266-282
    TP->>TP: scroll_into_view + flash, re-applied while replies stream in for LANDING_SETTLE_MS=2500ms thread.rs:206 thread.rs:667 thread.rs:297

    Note over MJ: If an ancestor is still streaming in, topic_root_of returns None and the Effect simply<br/>re-runs on the next store update - no retry loop, no timer message_jump.rs:85-98 message_jump.rs:189
    Note over MJ: RESOLVE_TIMEOUT_MS=6000: if the topic can never be resolved (unscoped channel, withheld<br/>parent, relay timeout) the page falls back to the single-note view /view/:id instead of hanging message_jump.rs:32 message_jump.rs:157-162
    Note over TP: The 2.5s settle window exists because a reply can still be en route when the page first<br/>paints, re-landing during that window is what keeps the reader ON the target post thread.rs:205-206
```

## NF-05.16 Zone end-to-end encryption — grant, encrypt, decrypt, redecrypt, and why deletion still works

```mermaid
flowchart TB
    GRANT["Grant sync Effect: once NIP-42 authed, start_grant_sync pulls zone-key grants<br/>out of the member's own gift wraps nostr-bbs-forum-client/src/app.rs:795-811, store.rs:217"]
    DISP["A grant rides the SAME kind-1059 transport as a DM; core rejects its rumor<br/>kind before DM unwrap succeeds, so dm/mod.rs never special-cases it<br/>dm/mod.rs:825-827 zone_crypto/mod.rs:87 - see NF-05.9"]
    KEYSTORE["ZoneKeyStore.insert - new key kept, then redecrypt() runs over every<br/>cached message in that zone store.rs:105-121"]
    WRITE["ZoneWriter::prepare on publish: encrypt when the zone is encrypted,<br/>refuse with no key, pass through a plaintext zone store.rs:480-491"]
    INGEST["prepare_incoming on every inbound kind-42: decrypt to plaintext,<br/>or a missing-key placeholder under the SAME event id store.rs:130-144 store.rs:408-414"]
    TOMB1["admits_message gate BEFORE prepare_incoming - a kind-5 that arrived<br/>first still suppresses the still-encrypted event channels.rs:405"]
    TOMB2["admits_message gate AFTER prepare_incoming - a restored sealed original<br/>carries its ORIGINAL id, which is what a kind-5 tombstoned (ADR-2017)<br/>channels.rs:446"]
    REDECRYPT["redecrypt re-runs decryption in place over channel_messages once a key<br/>arrives; a sealed placeholder that now opens is replaced by the real<br/>event, same id, so no re-fetch and no re-dedup is needed store.rs:151-181"]

    GRANT --> DISP --> KEYSTORE --> REDECRYPT
    WRITE --> INGEST
    INGEST --> TOMB2
    TOMB1 --> INGEST
    KEYSTORE --> INGEST

    N1["INVARIANT: a message is ALWAYS admitted through the SAME tombstone gate whether it is plaintext,<br/>freshly decrypted, or still a placeholder - encryption changes what the reader sees, never whether<br/>a deletion applies to it channels.rs:405 channels.rs:446 ADR-2017"]
    N2["The retro BBS client never holds a zone key at all: it masks ciphertext instead of decrypting it<br/>nostr-bbs-bbs-client/src/relay.rs:79 - see NF-05.11, and refuses to compose into an encrypted board<br/>nostr-bbs-bbs-client/src/screens.rs:906-918 - see NF-05.12"]
    N3["Dormant with the deployment gate off: keys already held keep decrypting, but no gift wrap is<br/>re-opened (which would prompt a NIP-07 signer for every DM) until encryption_enabled() is true<br/>nostr-bbs-forum-client/src/app.rs:796-800 zone_crypto/mod.rs:99"]
```
