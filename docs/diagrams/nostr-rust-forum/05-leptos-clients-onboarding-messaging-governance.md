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
  - ../nostr-rust-forum/SETUP.md
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/dm/mod.rs
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/gift_wrap.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/utils/search_client.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/signup.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/settings.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/thread.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/channel.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/events.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/pod_browser.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/components/recovery_sheet.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/components/git_panel.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/components/agent_badge.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/admin/user_table.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/utils/relay_url.rs
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
  - ../nostr-rust-forum/docs/diagrams/00-anomaly-register.md
  - ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/screens.rs
  - ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/theme.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/category.rs
verified_commit: 2f90c1916
---

## NF-05.1 Forum client boot and the FORUM_BASE discipline

```mermaid
sequenceDiagram
    autonumber
    participant B as Browser
    participant M as main.rs<br/>nostr-bbs-forum-client/src/main.rs:21
    participant SW as Service worker
    participant A as App<br/>nostr-bbs-forum-client/src/app.rs:299
    participant DB as IndexedDB

    B->>M: load the WASM bundle
    M->>M: mount_to_body(App) main.rs:21
    M->>SW: register_service_worker main.rs:34
    M->>SW: sw url and scope built from FORUM_BASE main.rs:53 main.rs:54
    M->>SW: update_via_cache = None, so sw.js is always revalidated main.rs:61
    M->>DB: evict cached messages older than 30 days main.rs:146
    A->>A: provide_auth nostr-bbs-forum-client/src/app.rs:306, provide_zone_access nostr-bbs-forum-client/src/app.rs:307, provide_profile_cache nostr-bbs-forum-client/src/app.rs:316
    A->>A: start_admin_alerts nostr-bbs-forum-client/src/app.rs:330, provide_agent_disclosure nostr-bbs-forum-client/src/app.rs:421
    A->>A: one app-wide RelayConnection nostr-bbs-forum-client/src/app.rs:429

    Note over A: INVARIANT ADR-090 - FORUM_BASE is applied in exactly TWO places: the const consumed by Router base= at nostr-bbs-forum-client/src/app.rs:45 and :832
    Note over A: and base_href() for every link at nostr-bbs-forum-client/src/app.rs:54, used e.g. at :1070. A third application re-introduces the double-prefix / deep-route-404 bug class.
    Note over M: main.rs:52 re-reads the same option_env! for the service-worker SCOPE - a deliberate, documented mirror of app::FORUM_BASE, not a third application of the prefix
    Note over A: current_app_path strips the prefix back off a browser path so the router sees an unprefixed route nostr-bbs-forum-client/src/app.rs:83
```

## NF-05.2 Route table

```mermaid
flowchart TB
    subgraph pubroutes["Public"]
        R1["/ HomeOrForums nostr-bbs-forum-client/src/app.rs:850"]
        R2["/about nostr-bbs-forum-client/src/app.rs:851 | /login nostr-bbs-forum-client/src/app.rs:852 | /signup nostr-bbs-forum-client/src/app.rs:857"]
        R3["/connect magic link - must NOT be auth-gated nostr-bbs-forum-client/src/app.rs:856"]
        R4["/glossary nostr-bbs-forum-client/src/app.rs:861 | /join/:code nostr-bbs-forum-client/src/app.rs:867"]
    end
    subgraph authed["Auth-gated"]
        R5["/setup nostr-bbs-forum-client/src/app.rs:869 | /forums nostr-bbs-forum-client/src/app.rs:883 | /settings nostr-bbs-forum-client/src/app.rs:893"]
        R6["/chat/:channel_id nostr-bbs-forum-client/src/app.rs:880 | /dm nostr-bbs-forum-client/src/app.rs:881 | /dm/:pubkey nostr-bbs-forum-client/src/app.rs:882"]
        R7["/forums/:category/board nostr-bbs-forum-client/src/app.rs:888 | /:section nostr-bbs-forum-client/src/app.rs:889 | /:topic nostr-bbs-forum-client/src/app.rs:890"]
        R8["/events nostr-bbs-forum-client/src/app.rs:891 | /profile/:pubkey nostr-bbs-forum-client/src/app.rs:892 | /pod nostr-bbs-forum-client/src/app.rs:902"]
    end
    subgraph gov["Governance"]
        R9["/governance member read-only nostr-bbs-forum-client/src/app.rs:900"]
        R10["/governance/admin admin write nostr-bbs-forum-client/src/app.rs:901"]
        R11["/admin - its own internal gate nostr-bbs-forum-client/src/app.rs:894"]
    end
    subgraph zonealias["Zone-slug aliases, declared LAST"]
        R12["/:category nostr-bbs-forum-client/src/app.rs:918 | /:category/board nostr-bbs-forum-client/src/app.rs:920"]
        R13["/:category/:section nostr-bbs-forum-client/src/app.rs:921 | /:category/:section/:topic nostr-bbs-forum-client/src/app.rs:922"]
    end

    ROUTER["Router base=FORUM_BASE nostr-bbs-forum-client/src/app.rs:832"] --> pubroutes & authed & gov & zonealias

    N1["/chat with no channel redirects to /forums - a legacy path kept alive nostr-bbs-forum-client/src/app.rs:879"]
    N2["The zone-slug aliases are declared last so the static routes out-score them; a slug is only valid<br/>if it resolves in the live ZONE_CONFIG nostr-bbs-forum-client/src/pages/category.rs:102"]
    N3["The member and admin governance views are separate ROUTES bound to separate components -<br/>see NF-06.10"]
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
    participant C as RelayConnection<br/>nostr-bbs-forum-client/src/relay.rs:182
    participant R as relay-worker

    C->>R: connect relay.rs:342
    R-->>C: open, state = Connected relay.rs:391
    C->>R: REQ subscription relay.rs:507 relay.rs:529
    R-->>C: EVENT relay.rs:694
    C->>C: verify_event_strict BEFORE dispatch relay.rs:714
    R-->>C: EOSE relay.rs:742
    C->>R: EVENT publish relay.rs:557
    R-->>C: OK relay.rs:794
    R-->>C: AUTH challenge relay.rs:817
    C->>R: signed kind-22242 AUTH response relay.rs:840
    C->>R: CLOSE relay.rs:549

    Note over C: INVARIANT: the client re-verifies every inbound event's id and Schnorr signature - it does not trust the relay relay.rs:714
    Note over C: publish_with_ack awaits the relay's OK rather than fire-and-forget relay.rs:596
    Note over C: The client half of NIP-42 is fully wired: it signs the challenge on demand - see NF-03.2 and NF-03.3
```

## NF-05.6 Derived, never accumulated — the counting invariant

```mermaid
flowchart LR
    EVS["channel_messages: HashMap of channel to Vec of events<br/>nostr-bbs-forum-client/src/stores/channels.rs:71"]
    DEDUP["push only if no event with this id is present<br/>nostr-bbs-forum-client/src/stores/channels.rs:444"]
    CNT["count_for = Vec::len on every read<br/>nostr-bbs-forum-client/src/stores/channels.rs:159 channels.rs:161"]
    UNREAD["unread_count is a Memo filtering !read<br/>nostr-bbs-forum-client/src/stores/notifications.rs:161"]

    EVS --> DEDUP --> CNT
    EVS --> UNREAD

    N1["INVARIANT ADR-091: post counts are NEVER stored as an independent field - a mutable counter drifts<br/>against deletions and replaceable events nostr-bbs-forum-client/src/stores/channels.rs:62"]
    N2["The event id is the ONLY counter; dedup on insert is what makes len() correct<br/>nostr-bbs-forum-client/src/stores/channels.rs:438"]
    N3["The same discipline is applied to notifications - unread is derived, not incremented<br/>nostr-bbs-forum-client/src/stores/notifications.rs:161. init_sync opens NO subscription of its own,<br/>and is idempotent behind a synced flag notifications.rs:276"]
```

## NF-05.7 Config projection into the browser

```mermaid
flowchart TB
    ENV["window.__ENV__ injected by the deployment"]
    ZC["ZONE_CONFIG read<br/>nostr-bbs-forum-client/src/stores/zones.rs:177 zones.rs:181"]
    LOAD["load_zones parse<br/>nostr-bbs-forum-client/src/stores/zones.rs:155"]
    FB["fallback_zones when absent<br/>nostr-bbs-forum-client/src/stores/zones.rs:324"]
    URLS["utils/relay_url.rs window_env reader<br/>nostr-bbs-forum-client/src/utils/relay_url.rs:131"]
    R["relay_url utils/relay_url.rs:8 | auth_api_base :36 | pod_api_base :69 | brand_label :98 | bbs_enabled :108"]
    COH["admin cohort editor is ZONE_CONFIG-driven, not a hardcoded list<br/>nostr-bbs-forum-client/src/admin/user_table.rs:19"]

    ENV --> ZC --> LOAD --> FB
    ENV --> URLS --> R
    ZC --> COH

    N1["The client renders what the config describes; the relay is the real boundary enforcing the SAME<br/>JSON - see NF-08.1 and NF-08.2"]
    N2["ANOMALY O8 re-verified and STILL LIVE: relay.rs reads window.__ENV__ inline<br/>nostr-bbs-forum-client/src/relay.rs:968 instead of calling the shared window_env helper, and the two<br/>resolvers' hardcoded fallbacks DIVERGE - relay.rs:22 versus utils/relay_url.rs:13"]
    N3["ANOMALY O7 re-verified, narrower than filed: NIP05_USERNAME_HOST is hardcoded example.test at<br/>nostr-bbs-forum-client/src/pages/settings.rs:32 and used at settings.rs:334, but signup.rs does NOT<br/>read it at all - so it is a settings-only display fallback, not a split env read"]
```

## NF-05.8 Signup — key, recovery sheet, pod

```mermaid
sequenceDiagram
    autonumber
    participant U as New user
    participant S as SignupPage<br/>nostr-bbs-forum-client/src/pages/signup.rs:288
    participant RS as RecoverySheet<br/>nostr-bbs-forum-client/src/components/recovery_sheet.rs:294
    participant POD as pod-worker

    U->>S: create account (passkey ceremony, see NF-05.3)
    S->>POD: eager provision_pod during creation signup.rs:288
    S->>POD: POST {POD_API}/pods/{pubkey}/.provision, NIP-98 authed signup.rs:180
    S->>RS: render the printable recovery sheet signup.rs:644
    RS->>RS: nsec QR generated in-WASM recovery_sheet.rs:66 recovery_sheet.rs:294
    RS->>RS: /connect magic link with the nsec in the URL FRAGMENT recovery_sheet.rs:290

    Note over RS: The sheet is 100 percent client-side - the nsec travels in the fragment (#k=), which browsers do not send to the server
    Note over S: ANOMALY O8 re-verified: a SECOND provision_pod lives in settings.rs:1752, triggered when an avatar upload 404s settings.rs:592 - two independent implementations of the same call
    Note over POD: EXTERNAL: the pod side of provisioning is NF-04 - the pod server itself is solid-pod-rs (SP-*)
```

## NF-05.9 Direct messages — two wraps per message, and why history exists

```mermaid
sequenceDiagram
    autonumber
    participant A as send_message<br/>nostr-bbs-forum-client/src/dm/mod.rs:355
    participant PAIR as gift_wrap_pair_with_signer<br/>nostr-bbs-core/src/gift_wrap.rs:539
    participant R as relay
    participant B as process_gift_wrap_event<br/>nostr-bbs-forum-client/src/dm/mod.rs:781

    A->>A: optimistic bubble applied synchronously, publish on a spawned task dm/mod.rs:351-352
    A->>PAIR: one rumor, sealed to the recipient and to the sender dm/mod.rs:444
    PAIR-->>A: (to_recipient, to_self) dm/mod.rs:445
    A->>A: re-key the optimistic message to the SELF copy's id so the relay echo dedups dm/mod.rs:449-452
    A->>R: publish BOTH wraps dm/mod.rs:461 dm/mod.rs:462
    B->>R: REQ gift wraps by my own p tag, windowed by since dm/mod.rs:558 dm/mod.rs:632
    B->>R: REQ legacy kind-4, unwindowed dm/mod.rs:634
    R-->>B: wraps
    B->>B: unwrap_gift_with_signer dm/mod.rs:787
    B->>B: a failure is surfaced through state.error, not swallowed dm/mod.rs:801

    Note over A: INVARIANT a single wrap is write-only - encrypted to the recipient, authored by a throwaway key, p-tagged to the recipient alone, so the sender can neither decrypt nor find it, and sent history kept vanishing dm/mod.rs:440-443
    Note over B: The gift-wrap half CANNOT be narrowed to one partner relay-side, so it fetches the whole wrap inbox and selects the conversation locally after unwrapping - the price NIP-59 sender anonymity charges dm/mod.rs:615-619
    Note over B: The realtime window is widened by GIFT_WRAP_LOOKBACK_SECS because NIP-59 randomises the outer created_at into the PAST - a since equals now subscription would skip a message sent this second dm/mod.rs:646 dm/mod.rs:628-630
    Note over B: Event-id dedup makes the widened overlap free dm/mod.rs:244-245
    Note over B: ANOMALY O6 CORRECTED: the filed defect was a silent no-op NIP-07 DM subscription. The subscription is registered unconditionally dm/mod.rs:253. The NIP-07 limitation is per-event, and it IS surfaced dm/mod.rs:801
```

## NF-05.10 Feature surfaces the forum client calls out to

```mermaid
flowchart LR
    THREAD["thread reply - kind 42 e-tagging the topic root<br/>nostr-bbs-forum-client/src/pages/thread.rs:637<br/>edit/delete via kind 5 thread.rs:821"]
    CHAN["channel message - kind 42<br/>nostr-bbs-forum-client/src/pages/channel.rs:803"]
    EVENTS["calendar - kind 31923 events, 31925 RSVPs<br/>nostr-bbs-forum-client/src/pages/events.rs:129 events.rs:187"]
    SEARCH["global search - kind-40 channel names over the relay<br/>global_search.rs:426 plus semantic search over the worker<br/>global_search.rs:317, one runtime-resolved base<br/>nostr-bbs-forum-client/src/utils/search_client.rs:109"]
    BADGE["agent disclosure over public HTTP GET<br/>nostr-bbs-forum-client/src/components/agent_badge.rs:317"]
    GIT["pod git control panel - _git/status :213 stage :272 unstage :310<br/>discard :359 commit :385 diff :434 log :466<br/>nostr-bbs-forum-client/src/components/git_panel.rs:213"]
    POD["pod browser - {POD_API}/pods/{pubkey} and /HEAD<br/>nostr-bbs-forum-client/src/pages/pod_browser.rs:513 pod_browser.rs:586"]
    ADMINT["admin actions - POST /api/admin/suspend :266 and /api/admin/silence :678, NIP-98 signed<br/>nostr-bbs-forum-client/src/admin/user_table.rs:266"]

    N1["DOC-DRIFT consumer-surface-map: the map lists the git panel as calling /.well-known/apps, but the<br/>client fetches and PUTs /apps/manifest.json instead - git_panel.rs:893 and git_panel.rs:946; no<br/>.well-known path exists anywhere in the crate"]
    N2["Calendar: only 31923 (time-based) and 31925 (RSVP) are used client-side. Kind 31922 (date-based),<br/>which the relay ban-gates and zone-gates, has no client surface - see NF-03.6 and NF-03.12"]
    N3["Semantic search goes to the search worker over HTTP, not to the relay - see NF-07"]
    N4["Both search entry points now resolve ONE runtime base through search_api_base rather than two<br/>disagreeing compile-time constants nostr-bbs-forum-client/src/utils/search_client.rs:109,<br/>and the search worker fails CLOSED, so a message in an openly readable channel must be ingested with<br/>public true or it is indexed and permanently unfindable - see NF-07.4"]
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
    N2["Config is read from window.__ENV__ exactly as the forum client does<br/>nostr-bbs-bbs-client/src/config.rs:265, relay URL config.rs:144, pwa_mode from ?pwa=1 config.rs:180"]
    N3["The BBS derives its passkey key through the SAME core helper as the forum client<br/>nostr-bbs-bbs-client/src/passkey.rs:179, with the PRF buffer zeroized after use nostr-bbs-bbs-client/src/passkey.rs:181"]
```

## NF-05.13 Notification ingress — why a reply addressed to you is not burned

```mermaid
stateDiagram-v2
    [*] --> Classify: kind-42 post arrives<br/>nostr-bbs-forum-client/src/stores/notifications.rs:821
    Classify --> OwnPost: the viewer wrote it<br/>notifications.rs:831
    Classify --> BeforeBaseline: older than the persisted first-sync floor<br/>notifications.rs:835
    Classify --> DirectedAtMe: a p tag names me, so the read-position gate is skipped<br/>notifications.rs:837
    Classify --> AlreadyRead: at or before the channel read position<br/>notifications.rs:802
    DirectedAtMe --> Notify
    Classify --> Notify: genuine unseen activity<br/>notifications.rs:793
    Notify --> [*]
    OwnPost --> [*]
    BeforeBaseline --> [*]
    AlreadyRead --> [*]

    note right of AlreadyRead
        INVARIANT only a PERMANENT verdict is burned into the persisted
        dedup set notifications.rs:807. AlreadyRead is reversible, because
        read positions are written by render-time effects and can be wrong,
        so a cheap re-check on a later pass is worth more than a permanent
        veto notifications.rs:799-802
    end note
    note right of DirectedAtMe
        The read position is per CHANNEL, but a channel holds many topics
        and is stamped to its newest message by a render-time effect, so
        opening a section's title list marked every reply in every topic
        read having shown the reader nothing but titles
        notifications.rs:840-846
    end note
    note right of Classify
        Order is deliberate - authorship, then the sync floor, then the read
        position - so the most DURABLE reason wins and a post is not filed
        under a reversible verdict when an irreversible one applies
        notifications.rs:817-820
    end note
```

## NF-05.14 Search from the client — one base, two paths

```mermaid
sequenceDiagram
    autonumber
    participant U as Member
    participant GS as global_search<br/>nostr-bbs-forum-client/src/components/global_search.rs:317
    participant SC as search_api_base<br/>nostr-bbs-forum-client/src/utils/search_client.rs:109
    participant SW as search-worker
    participant R as relay

    U->>GS: type a query, debounced
    GS->>R: kind-40 channel names over the relay global_search.rs:426
    GS->>SC: resolve the worker base at RUNTIME search_client.rs:109
    SC-->>GS: one base for both entry points
    GS->>SW: POST /search global_search.rs:709
    SW-->>GS: only labels the worker considers public - see NF-07.4

    Note over SC: A compile-time URL bakes one endpoint into the artefact, so the base is read from window.__ENV__ first and a blank value is treated as absent SETUP.md:196-197
    Note over SW: The result count field is k - serde discards any other key, so a plausible-looking limit is ignored and the default of 10 applies SETUP.md:218-221
```
