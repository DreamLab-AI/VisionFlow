---
id: NF-30
title: Forum client messaging, feature surfaces, notification ingress, search, deep links, AUTH and deletion sync
area: nostr-rust-forum
governing:
  - ../nostr-rust-forum/docs/BASELINE-architecture.md
  - ../nostr-rust-forum/docs/IDENTITY-keys-and-trust.md
adrs: [ADR-2023, ADR-2034]
sources:
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/relay.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/relay/receipt.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/utils/lookup.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/stores/deletion_sync.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/stores/channels.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/stores/notifications.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/components/global_search.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/dm/mod.rs
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/gift_wrap.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/utils/search_client.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/thread.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/channel.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/message_jump.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/events.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/pod_browser.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/components/git_panel.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/components/agent_badge.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/admin/user_table.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/mod.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/store.rs
  - ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/relay.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/section.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/grant_backlog.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/grant_paging.rs
  - ../nostr-rust-forum/docs/security/known-findings.md
  - ../nostr-rust-forum/docs/adr/ADR-2034-absence-is-reported-only-after-an-authenticated-answer.md
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/category.rs
  - ../nostr-rust-forum/docs/adr/ADR-2027-nip17-overlay-for-nostr-chat-interop.md
  - ../nostr-rust-forum/docs/adr/ADR-2028-zones-adopt-nip171-key-plane.md
verified_commit: de31aaa6e6c6b672cc310a2659cf33fd4887602c
aliases:
  NF-05.9: NF-30.1
  NF-05.10: NF-30.2
  NF-05.13: NF-30.3
  NF-05.14: NF-30.4
  NF-05.15: NF-30.5
  NF-05.18: NF-30.6
  NF-05.19: NF-30.7
  NF-05.20: NF-30.8
---
Split from NF-05 on 2026-11-23: the client's messaging and relay-facing behaviour (formerly NF-05.9, NF-05.10, NF-05.13 to NF-05.15 and NF-05.18 to NF-05.20). Boot, identity, transport, config and signup stay in NF-05. Old panel IDs resolve through the `aliases:` map above; the stamp is NF-05's, and the panels were not re-read.

## NF-30.1 Direct messages — two wraps per message, and why history exists

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
    Note over B: ADR-2016: a zone-key grant rides the SAME gift-wrap transport (kind 1059) as a DM. Core refuses the grant's rumor kind before unwrap can succeed, so dispatch never needs a kind check of its own - dm/mod.rs:1231-1235, zone_crypto/mod.rs:100
    Note over B: The gift-wrap half CANNOT be narrowed to one partner relay-side, so it fetches the whole wrap inbox and selects the conversation locally after unwrapping - the price NIP-59 sender anonymity charges dm/mod.rs:965-969
    Note over B: The realtime window is widened by GIFT_WRAP_LOOKBACK_SECS because NIP-59 randomises the outer created_at into the PAST - a since equals now subscription would skip a message sent this second dm/mod.rs:997 dm/mod.rs:976-984
    Note over B: Tier 0 (ADR-2023): a layout-level incoming watcher subscribes to the account's own gift-wrap inbox, unwraps each wrap not yet announced and raises a DM notification, because the per-page DMStore only subscribes while a DM screen is open dm/mod.rs:1116 dm/mod.rs:1144
    Note over B: Read marks are kept per account in localStorage under nbbs:dm-read-marks and message timestamps are clamped to now, so a future-dated rumor cannot pin a conversation dm/mod.rs:1461 dm/mod.rs:1469 dm/mod.rs:1983
    Note over B: Event-id dedup makes the widened overlap free dm/mod.rs:350-351
    Note over B: ANOMALY O6 CORRECTED: the filed defect was a silent no-op NIP-07 DM subscription. The subscription is registered unconditionally dm/mod.rs:359. The NIP-07 limitation is per-event, and it IS surfaced dm/mod.rs:1249-1257
```

**Open:** ADR-2027 (NIP-17 overlay) is accepted, partly implemented and inactive, and ADR-2028 (NIP-171 key plane, superseding ADR-2026) is proposed and inactive (`../nostr-rust-forum/docs/adr/ADR-2027-nip17-overlay-for-nostr-chat-interop.md:6`, `../nostr-rust-forum/docs/adr/ADR-2028-zones-adopt-nip171-key-plane.md:5`, `:7`). Both implementations reached main with the upgrade-branch merge a80c572 and stay dormant: the extended DM path (reply and imeta rumors, multi-relay delivery) runs only with `RECIPIENT_KEYS_ENABLED` (`../nostr-rust-forum/crates/nostr-bbs-forum-client/src/dm/mod.rs:560`, `:1637-1638`), and the owner activation workflows are still missing ([NF-28](28-nip171-owner-recovery-and-ticket-contracts.md)), so neither is drawn as live here.


## NF-30.2 Feature surfaces the forum client calls out to

```mermaid
flowchart LR
    THREAD["thread reply - kind 42 e-tagging the topic root<br/>nostr-bbs-forum-client/src/pages/thread.rs:1043<br/>edit/delete via kind 5 thread.rs:1111"]
    CHAN["channel message - kind 42<br/>nostr-bbs-forum-client/src/pages/channel.rs:835"]
    EVENTS["calendar - kind 31923 events, 31925 RSVPs<br/>nostr-bbs-forum-client/src/pages/events.rs:137 events.rs:195"]
    SEARCH["global search - kind-40 channel names over the relay<br/>global_search.rs:467 plus NIP-50 text search over kind-42<br/>global_search.rs:725-760, worker as semantic layer/fallback<br/>global_search.rs:360. A hit links to /go/:event_id, not /chat/:channel - see NF-05.2 and NF-30.5"]
    BADGE["agent disclosure over public HTTP GET<br/>nostr-bbs-forum-client/src/components/agent_badge.rs:317"]
    GIT["pod git control panel - _git/status :213 stage :272 unstage :310<br/>discard :359 commit :385 diff :434 log :466<br/>nostr-bbs-forum-client/src/components/git_panel.rs:213"]
    POD["pod browser - {POD_API}/pods/{pubkey} and /HEAD<br/>nostr-bbs-forum-client/src/pages/pod_browser.rs:513 pod_browser.rs:582<br/>OPEN O26 / KF-28: mesh and pod cards print the raw probe error under Details<br/>pod_browser.rs:1045 pod_browser.rs:1058 pod_browser.rs:1181"]
    ADMINT["admin actions - POST /api/admin/suspend :269 and /api/admin/silence :701, NIP-98 signed<br/>nostr-bbs-forum-client/src/admin/user_table.rs:269"]

    N1["DOC-DRIFT consumer-surface-map: the map lists the git panel as calling /.well-known/apps, but the<br/>client fetches and PUTs /apps/manifest.json instead - git_panel.rs:893 and git_panel.rs:946; no<br/>.well-known path exists anywhere in the crate"]
    N2["Calendar: only 31923 (time-based) and 31925 (RSVP) are used client-side. Kind 31922 (date-based),<br/>which the relay ban-gates and zone-gates, has no client surface - see NF-03.6 and NF-03.12"]
    N3["DRIFT resolved: the worker's index was fed only by an admin-gated ingest call, so a member's own<br/>post was unfindable by non-admins. The relay now answers text search directly (NIP-50, matched<br/>case-insensitively against every post's content) and is authoritative; the worker is consulted only<br/>in semantic mode or when the relay returns nothing global_search.rs:336-360"]
    N4["A zone-encrypted post is never a search result either way: the relay match is on ciphertext, which<br/>relay_text_search discards client-side global_search.rs:735-742, and the semantic index is fed only<br/>public content - see NF-07.4"]
```

## NF-30.3 Notification ingress — why a reply addressed to you is not burned

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

## NF-30.4 Search from the client — relay text search first, the worker as semantic layer

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

## NF-30.5 Message deep-link — /go/:event_id resolves a bare id to a reader position

```mermaid
sequenceDiagram
    autonumber
    participant U as Member
    participant MJ as MessageJumpPage<br/>nostr-bbs-forum-client/src/pages/message_jump.rs:102
    participant R as relay
    participant CS as ChannelStore
    participant TP as ThreadPage<br/>nostr-bbs-forum-client/src/pages/thread.rs:379

    U->>MJ: navigate to /go/:event_id (search hit, or any shared link)
    MJ->>R: subscribe ids=[event_id] message_jump.rs:147-153
    R-->>MJ: the target event
    MJ->>CS: ensure_subscribed(&relay, channel_of(event)) message_jump.rs:177
    MJ->>MJ: focus = edit_target_of(event) else event.id - an edit renders as its original message_jump.rs:180
    MJ->>MJ: topic_root_of walks the reply chain (e-tag reply/root markers) up to 64 hops message_jump.rs:85-98 message_jump.rs:189
    MJ->>MJ: resolve the topic's zone via section_to_zone message_jump.rs:194-202
    MJ->>TP: navigate to /forums/:zone/:section/:topic?focus=:id, replace=true message_jump.rs:203-208

    TP->>TP: landing_target - focus post if loaded, else the newest reply thread.rs:269-285
    TP->>TP: scroll_into_view + flash, re-applied while replies stream in for LANDING_SETTLE_MS=2500ms thread.rs:209 thread.rs:709 thread.rs:300

    Note over MJ: If an ancestor is still streaming in, topic_root_of returns None and the Effect simply<br/>re-runs on the next store update - no retry loop, no timer message_jump.rs:85-98 message_jump.rs:189
    Note over MJ: RESOLVE_TIMEOUT_MS=6000: if the topic can never be resolved (unscoped channel, withheld<br/>parent, relay timeout) the page falls back to the single-note view /view/:id instead of hanging message_jump.rs:32 message_jump.rs:157-162
    Note over TP: The 2.5s settle window exists because a reply can still be en route when the page first<br/>paints, re-landing during that window is what keeps the reader ON the target post thread.rs:208-209
```

## NF-30.6 NIP-42 AUTH acceptance, held writes and the per-subscription REQ ledger (kit #91)

```mermaid
sequenceDiagram
    autonumber
    participant C as RelayConnection<br/>nostr-bbs-forum-client/src/relay.rs:266
    participant L as ReqLedger<br/>nostr-bbs-forum-client/src/relay/receipt.rs:71
    participant R as relay-worker

    C->>R: AUTH frame, pending_auth recorded, answer timer armed nostr-bbs-forum-client/src/relay.rs:1539 nostr-bbs-forum-client/src/relay.rs:1551
    R-->>C: OK for the AUTH event id nostr-bbs-forum-client/src/relay.rs:1475
    C->>C: classify_auth_ok decides Accepted, Retry or Rejected receipt.rs:318 nostr-bbs-forum-client/src/relay.rs:976
    alt Accepted
        C->>L: replay_after_auth: every REQ counted as authenticated nostr-bbs-forum-client/src/relay.rs:1229 nostr-bbs-forum-client/src/relay.rs:1234
        C->>R: every REQ and pending COUNT re-sent, held writes flushed, paced nostr-bbs-forum-client/src/relay.rs:1254 nostr-bbs-forum-client/src/relay.rs:1279
        C->>C: authenticated set true only now nostr-bbs-forum-client/src/relay.rs:1290
    else rate-limited, below AUTH_RETRY_LIMIT 3
        C->>R: same AUTH frame re-sent after 1 s times attempts nostr-bbs-forum-client/src/relay.rs:994 receipt.rs:42
    else rejected, unsigned, or stalled
        C->>C: refuse_auth: auth_note set, held writes released so each OK false reaches its callback nostr-bbs-forum-client/src/relay.rs:1034 nostr-bbs-forum-client/src/relay.rs:1022
    end
    R-->>C: EOSE under the current wire id nostr-bbs-forum-client/src/relay.rs:1383
    C->>L: answered is authed only when every REQ sent was answered and the last went out after AUTH receipt.rs:109 nostr-bbs-forum-client/src/relay.rs:1385
    C->>C: on_receipt(authed) feeds the page receipts in NF-30.8 nostr-bbs-forum-client/src/relay.rs:1395
    C->>C: kit #98: the answer clock starts when the REQ frame leaves the paced queue, not when it is queued nostr-bbs-forum-client/src/relay.rs:318 receipt.rs:262
    C->>L: timer fires: still queued or under 8 s on the wire, re-check later receipt.rs:245 nostr-bbs-forum-client/src/relay.rs:1159 - no longer the latest REQ, nothing receipt.rs:121
    C->>L: 8 s on the wire unanswered: on_watch says Settled, Resend or GiveUp after 2 re-sends receipt.rs:138 receipt.rs:50 receipt.rs:55
    C->>R: Resend: CLOSE the old wire id, REQ under a fresh one nostr-bbs-forum-client/src/relay.rs:1185 nostr-bbs-forum-client/src/relay.rs:1209

    Note over L: INVARIANT: an answer is never credited to a REQ sent after it - a late EOSE under a retired id is discarded receipt.rs:31-33 receipt.rs:191
    Note over C: Stall check: 60 s without a signature or 10 s without an answer releases held writes, timers carry conn_gen receipt.rs:60 receipt.rs:67 nostr-bbs-forum-client/src/relay.rs:1062
```

**What it shows.** Before kit #91 the `authenticated` signal meant only "AUTH frame sent", so a REQ answered anonymously could be taken for the member view. Now the relay's `OK true` is the trigger, and the ledger tells the page whether an EOSE is the authenticated answer.

## NF-30.7 Incremental kind-5 deletion sync (kit #93)

```mermaid
flowchart TB
    LOAD["Per-scope localStorage record: since marker, sealed and held tombstones<br/>deletion_sync.rs:627 deletion_sync.rs:166"]
    REQ["kind-5 filter with since = marker, limit 1000, beside the channel filters<br/>channels.rs:974 channels.rs:525 channels.rs:526"]
    OBS["Each kind-5: observe for the marker, then apply_deletion per deleter<br/>channels.rs:385 channels.rs:388 deletion_sync.rs:521"]
    EOSE["Channel EOSE: on_main_eose decides<br/>channels.rs:511 deletion_sync.rs:583"]
    PAGE["Marked sync filled the page: fetch until = oldest delivered, inclusive<br/>deletion_sync.rs:402 channels.rs:987"]
    NEXT["Each page: Done, Next or GiveUp, at most 20 pages<br/>deletion_sync.rs:424 deletion_sync.rs:123"]
    DONE["Settled: marker = min(newest delivered, now) minus 14 days, never backwards<br/>deletion_sync.rs:266 deletion_sync.rs:119"]
    GIVE["GiveUp: old marker stays<br/>OPEN O23 / KF-25: no resume cursor, the next session re-fetches the same pages<br/>deletion_sync.rs:30-34"]
    SAVE["persist the record<br/>deletion_sync.rs:641"]

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

## NF-30.8 Page lookup states: loading, keys loading, found, not found, timed out (kit #91, kit #98)
> **Superseded** by NF-30.9 at de31aaa (frozen at 4a73a7a).

```mermaid
flowchart TB
    URL(["deep link /forums/zone/section/topic"])
    GATED["zone_is_gated: a non-public zone needs an authenticated receipt<br/>lookup.rs:234 lookup.rs:63"]
    SEC{"thread page: channel found, or channel_receipt settles?<br/>thread.rs:661-665, the zone page applies<br/>the same rule to its sections category.rs:148"}
    TOP{"topic root found among the channel's kind-42 by id?<br/>thread.rs:487 thread.rs:504"}
    HIST["history receipt from ensure_subscribed<br/>channels.rs:761 thread.rs:670"]
    KEYS["key_state_memo: encrypted zone and a zone-key store, phase from key_phase_for<br/>thread.rs:645-656 lookup.rs:195 zone_crypto/store.rs:173"]
    BACK["GrantBacklog: settled once every grant subscription reached EOSE and no wrap is in flight, latched<br/>stalled after 20 s without progress or 120 s from the start of the sync<br/>grant_backlog.rs:153-159 grant_backlog.rs:35 grant_backlog.rs:42 grant_backlog.rs:164-176"]
    DEC["Lookup.decide_with_keys: presence, then relay answer, then keys, 15 s deadline re-armed per target<br/>lookup.rs:134-151 lookup.rs:38 thread.rs:639 thread.rs:668"]
    LOADING(["Loading topic, or Fetching your keys for this zone<br/>thread.rs:1199-1203"])
    NF(["Topic Not Found<br/>thread.rs:1208"])
    TO(["Topic Unavailable: no relay answer, or keys too slow<br/>thread.rs:1212 thread.rs:1215-1217"])
    FOUND(["render the topic"])

    URL --> GATED --> SEC
    SEC -->|found| TOP
    TOP --> DEC
    HIST --> DEC
    BACK --> KEYS --> DEC
    DEC -->|found| FOUND
    DEC -->|answered, keys settled or not needed| NF
    DEC -->|no answer by the deadline, or keys stalled| TO
    DEC -->|no answer yet, or keys pending or progressing| LOADING

    CLOSED["CLOSED O21 / KF-23 by kit #98 = fd31193: an answered relay no longer means Not Found while keys arrive lookup.rs:146-149.<br/>Why keys matter: a sealed original sits under its envelope id until a key opens it zone_crypto/store.rs:449-451,<br/>and redecrypt restores it in place when a key lands zone_crypto/store.rs:486-488. The section page applies the same rule to its topic list<br/>section.rs:318 section.rs:334 section.rs:575-579 section.rs:603-606. Operator observation 2026-11-22, same cold device:<br/>Fetching your keys from 9.2 s, the topic at 28.4 s, no Topic Not Found; before the fix Not Found showed from 12.2 s to 28.6 s."]
    OPEN["STILL OPEN: keys still take about 28 s on a cold device - the code records 28.6 s to the first usable key grant_backlog.rs:38-39.<br/>The grant filter asks for at most 1000 wraps, newest first, shared with DMs zone_crypto/store.rs:1138-1148 channels.rs:963, so an account<br/>with more wraps can miss its oldest grants; paging them is kit PR #95, a draft. A member who holds no key settles too lookup.rs:161-163,<br/>so a sealed-original topic they cannot open is Not Found once keys settle lookup.rs:147."]
    CLOSED -.-> DEC
    OPEN -.-> NF
```

**What it shows.** Kit #91 made a deep-linked page wait for the subscription that would carry the topic; kit #98 (fd31193) adds the member's zone-key backlog as an input, so an encrypted zone shows "Fetching your keys" rather than "Topic Not Found" until the grant backlog settles or stalls.

## NF-30.9 Page lookup states after the grant backlog pages (kit #91, #98, #95)

```mermaid
flowchart TB
    URL(["deep link /forums/zone/section/topic"])
    GATED["zone_is_gated: a non-public zone needs an authenticated receipt<br/>lookup.rs:234 lookup.rs:63"]
    SEC{"thread page: channel found, or channel_receipt settles?<br/>thread.rs:661-665, the zone page applies<br/>the same rule to its sections category.rs:148"}
    TOP{"topic root found among the channel's kind-42 by id?<br/>thread.rs:487 thread.rs:504"}
    HIST["history receipt from ensure_subscribed<br/>channels.rs:761 thread.rs:670"]
    KEYS["key_state_memo: encrypted zone and a zone-key store, phase from key_phase_for<br/>thread.rs:645-656 lookup.rs:195 zone_crypto/store.rs:255"]
    BACK["GrantBacklog: settled once every grant scope paged to its end, no wrap is in flight or left to retry, and no scope gave up; latched<br/>stalled after 20 s without progress, 120 s from the later of the sync start and the last scope delivery, or at once after a give-up<br/>grant_backlog.rs:324-333 grant_backlog.rs:40 grant_backlog.rs:49 grant_backlog.rs:339-353"]
    DEC["Lookup.decide_with_keys: presence, then relay answer, then keys, 15 s deadline re-armed per target<br/>lookup.rs:134-151 lookup.rs:38 thread.rs:639 thread.rs:668"]
    LOADING(["Loading topic, or Fetching your keys for this zone<br/>thread.rs:1199-1203"])
    NF(["Topic Not Found<br/>thread.rs:1208"])
    TO(["Topic Unavailable: no relay answer, or keys too slow<br/>thread.rs:1212 thread.rs:1215-1217"])
    FOUND(["render the topic"])

    URL --> GATED --> SEC
    SEC -->|found| TOP
    TOP --> DEC
    HIST --> DEC
    BACK --> KEYS --> DEC
    DEC -->|found| FOUND
    DEC -->|answered, keys settled or not needed| NF
    DEC -->|no answer by the deadline, or keys stalled| TO
    DEC -->|no answer yet, or keys pending or progressing| LOADING

    KEYWAIT["Since kit #98 = fd31193: an answered relay no longer means Not Found while keys arrive lookup.rs:146-149.<br/>Why keys matter: a sealed original sits under its envelope id until a key opens it zone_crypto/store.rs:539-541,<br/>and redecrypt restores it in place when a key lands zone_crypto/store.rs:576-578. The section page applies the same rule to its topic list<br/>section.rs:318 section.rs:334 section.rs:575-579 section.rs:603-606. KF-23 reads Closed in the table docs/security/known-findings.md:91."]
    PAGED["Kit PR #95: each grant page still asks for at most 1000 wraps, newest first, shared with DMs zone_crypto/store.rs:1637-1643 channels.rs:963,<br/>but every scope now pages to its end, page 0 plus at most 50 grant_paging.rs:108 (NF-27.8). A grant left to retry keeps the backlog open<br/>grant_backlog.rs:333, so the page shows the keys copy, never Not Found. A member who holds no key settles too lookup.rs:161-163,<br/>so a sealed-original topic they cannot open is Not Found once keys settle lookup.rs:147."]
    LIMITS["OPEN KF-31 a failed relay query reads as empty, KF-32 the 51,000-wrap bound, KF-33 junk wraps sharing one timestamp<br/>docs/security/known-findings.md:108-110 - each ends in the keys-timed-out copy, never Not Found"]
    KEYWAIT -.-> DEC
    PAGED -.-> NF
    LIMITS -.-> TO
```

**What it shows.** The decision is unchanged since kit #98 (`../nostr-rust-forum/crates/nostr-bbs-forum-client/src/utils/lookup.rs:134-151`); kit PR #95 changed what it waits on, so the backlog is complete past the row cap and a retryable wrap holds it (`../nostr-rust-forum/docs/adr/ADR-2034-absence-is-reported-only-after-an-authenticated-answer.md:75-80`). Live, 2026-11-24, after the website repinned to de31aaa: a fresh device on a direct Family topic link showed "Fetching your keys" at 4.5 to 5.4 s and the decrypted topic at 11.7 to 13.0 s over three runs, never "Topic Not Found" (9.2 s and 28.4 s on 2026-11-22); an operator observation, not a code citation.
