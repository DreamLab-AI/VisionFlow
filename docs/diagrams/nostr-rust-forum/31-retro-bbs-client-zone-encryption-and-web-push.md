---
id: NF-31
title: The retro BBS client, zone encryption and Web Push in the client
area: nostr-rust-forum
governing:
  - ../nostr-rust-forum/docs/BASELINE-architecture.md
  - ../nostr-rust-forum/docs/IDENTITY-keys-and-trust.md
adrs: [ADR-2008, ADR-2023]
sources:
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/keys.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/app.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/relay.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/stores/channels.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/utils/push.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/wrangler.toml
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/components/push_settings.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/settings.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/mod.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/store.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/sw.js
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
  - ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/screens.rs
  - ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/theme.rs
  - ../nostr-rust-forum/docs/adr/ADR-2023-content-free-web-push-for-pwa-alerts.md
verified_commit: de31aaa6e6c6b672cc310a2659cf33fd4887602c
aliases:
  NF-05.11: NF-31.1
  NF-05.12: NF-31.2
  NF-05.16: NF-31.3
  NF-05.17: NF-31.4
---
Split from NF-05 on 2026-11-23: the retro BBS client and the client-side key and push policy (formerly NF-05.11, NF-05.12, NF-05.16 and NF-05.17). Boot, identity, transport, config and signup stay in NF-05. Old panel IDs resolve through the `aliases:` map above; the split kept NF-05's stamp without a re-read; the panels were re-verified at de31aaa in pass C (2026-11-24).

## NF-31.1 Retro BBS client — session adoption and key custody

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

## NF-31.2 BBS screen machine and its own surfaces

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

## NF-31.3 Zone encryption depends on resolved policy and locally held keys

[The zone-key audit](../../estate-review/2026-11-01-forum-zone-key-distribution.md) separates tested crypto/planning behaviour from browser-only lifecycle findings. NF-27 expands rotation, delivery and recovery.

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/app.rs:798
    G["Gate on, relay authenticated and this identity's keys loaded<br/>start member grant sync once per sign-in"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/mod.rs:762
    V["Verify wrap and seal, unwrap private rumor<br/>admin authority and key-pair validation"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/store.rs:1313
    K["Persist as this identity's row<br/>a different key for the slot is refused, see NF-27.6"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/store.rs:1825
    W["Resolve channel zone before writing<br/>unknown mapping passes plaintext through"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/store.rs:1829
    E["Mapped encrypted zone uses the send key: highest held,<br/>never one pending delivery, no key means refuse and retain draft"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/stores/channels.rs:580
    T["Incoming event passes tombstone check<br/>then decrypt or keep placeholder"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/stores/channels.rs:622
    I["Check restored original ID against tombstones<br/>insert with ID deduplication"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/store.rs:591
    R["Later key arrival reopens cached events<br/>ReadCache applies deletion predicate and deduplication"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/app.rs:799
    O["Gate off stops grant sync<br/>already held keys can still decrypt"]
    G --> V --> K --> R
    W --> E
    T --> I
    K --> T
    O --> T
```

## NF-31.4 Web Push in the client — default off, staged, content-free

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
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/wrangler.toml:88
    N2["OPEN: ADR-2023 is accepted and complete but activation is staged - the relay ships PUSH_ENABLED false<br/> - see NF-08.8"]
```

**What it shows.** How the browser side of ADR-2023 works: the client asks the relay whether push is on, subscribes the device with a signed request, and from then on an empty push wakes the service worker, which drains content-free counters. Where server push is off, Tier 0 raises generic system notifications from the page. Logout cleanup is best effort, and its missing credential copy is documented in [the push audit](../../estate-review/2026-11-01-forum-web-push.md).

**Why it is this way.** Push payloads transit Google, Apple and Mozilla, and the service worker has no access to the page’s localStorage or sessionStorage, so ADR-2023 chose payload-less push with a bearer-token drain instead of RFC 8291 encryption (`../nostr-rust-forum/docs/adr/ADR-2023-content-free-web-push-for-pwa-alerts.md:34-35`, `../nostr-rust-forum/docs/adr/ADR-2023-content-free-web-push-for-pwa-alerts.md:38-39`).

**Open:** ADR-2023 records `activation_status: staged` (no live Cloudflare deploy, push-service round trip or iOS device run was verified), and the shipped relay template leaves `PUSH_ENABLED = "false"` (`../nostr-rust-forum/docs/adr/ADR-2023-content-free-web-push-for-pwa-alerts.md:7`, `../nostr-rust-forum/crates/nostr-bbs-relay-worker/wrangler.toml:88`).
