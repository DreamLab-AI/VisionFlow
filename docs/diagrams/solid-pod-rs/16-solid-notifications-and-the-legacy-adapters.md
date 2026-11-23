---
id: SP-16
title: Solid Notifications — channels, pump, webhooks, the legacy adapters and the server mounts
area: solid-pod-rs
governing: [../solid-pod-rs/README.md, ../solid-pod-rs/crates/solid-pod-rs/docs/explanation/ecosystem-integration.md]
adrs: [ADR-2005, ADR-2006]
sources:
  - ../solid-pod-rs/crates/solid-pod-rs-nostr/src/ws.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/notifications/mod.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/notifications/signing.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/notifications/legacy.rs
  - ../solid-pod-rs/crates/solid-pod-rs-forge/src/lib.rs
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/lib.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/handlers/legacy_notifications.rs
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/notifications_ws.rs
verified_commit: {solid-pod-rs: a28a74832dd0a6e47a531e34e4adc46a3946dfd3, agentbox: 59b1aca1cc3541b767872ecbc161c89e65d6ebca}
aliases:
  SP-08.1: SP-16.1
  SP-08.2: SP-16.2
  SP-08.3: SP-16.3
  SP-08.4: SP-16.4
  SP-08.17: SP-16.5
  SP-08.19: SP-16.6
  SP-08.23: SP-16.7
---
Split from SP-08 on 2026-11-23: the Solid Notifications surface (formerly SP-08.1 to SP-08.4, SP-08.17, SP-08.19 and SP-08.23). ActivityPub and the relay stay in SP-08. Old panel IDs resolve through the `aliases:` map above; the stamp is SP-08's, and the panels were not re-read.

## SP-16.1 Solid Notifications 0.2 — the three channels

```mermaid
flowchart TD
    SE["StorageEvent from the backend watcher — see SP-06.4"]
    CN["ChangeNotification::from_storage_event<br/>solid-pod-rs/src/notifications/mod.rs:89"]
    WS["WebSocketChannelManager<br/>solid-pod-rs/src/notifications/mod.rs:243"]
    WH["WebhookChannelManager<br/>solid-pod-rs/src/notifications/mod.rs:383"]
    LEG["LegacyNotificationChannel — solid-0.1<br/>solid-pod-rs/src/notifications/legacy.rs:221"]
    DISC["discovery_document<br/>solid-pod-rs/src/notifications/mod.rs:793"]
    SUB["Subscription<br/>solid-pod-rs/src/notifications/mod.rs:62"]
    CT["ChannelType<br/>solid-pod-rs/src/notifications/mod.rs:55"]

    SE --> CN
    CN --> WS
    CN --> WH
    SE --> LEG
    DISC --> CT --> SUB

    N["DEFAULT_MAX_SUBSCRIPTIONS 10 000 caps only InMemoryNotifications<br/>(solid-pod-rs/src/notifications/mod.rs:125), whose publish is a no-op<br/>(solid-pod-rs/src/notifications/mod.rs:229). The two channel managers that deliver<br/>keep plain unbounded maps (solid-pod-rs/src/notifications/mod.rs:244 and :385)."]
    SUB -.-> N
    N2["Updates-Via names wss://base/.notifications only while the hub is mounted<br/>(solid-pod-rs-server/src/lib.rs:1792, :1793), on GET and HEAD<br/>(solid-pod-rs-server/src/lib.rs:1974) and OPTIONS (solid-pod-rs-server/src/lib.rs:3219).<br/>The hub is on by default (solid-pod-rs-server/src/lib.rs:733) and serves the legacy<br/>session, not these managers: SP-16.6. See SP-03.11."]
    WS -.-> N2
```

**What it shows.** The core crate defines two Notifications 0.2 channel managers, the legacy `solid-0.1` adapter and the discovery document. They are library types: `solid-pod-rs-server` instantiates none of them; its `/.notifications` route runs the legacy session instead (SP-16.6).

**Why it is this way.** The crate owns protocol and leaves routing to the embedder (the same line SP-16.5 draws). Since SP-PAR-279 the reference server advertises only the channel it serves.

**Tension:** RESOLVED at a28a748 (SP-PAR-279). `/.notifications` is now a mounted `solid-0.1` WebSocket (`solid-pod-rs-server/src/lib.rs:6527`), and `Updates-Via` is sent only when it is mounted (`solid-pod-rs-server/src/lib.rs:1793`). No route serves the Notifications 0.2 managers.

**Drift:** the subscription cap note in earlier readings implied the delivering channels were bounded. Only `InMemoryNotifications` is capped, and it delivers nothing.

## SP-16.2 WebSocket notification pump

```mermaid
sequenceDiagram
    autonumber
    participant C as subscriber
    participant M as WebSocketChannelManager<br/>solid-pod-rs/src/notifications/mod.rs:243
    participant P as pump_from_storage<br/>solid-pod-rs/src/notifications/mod.rs:316
    participant S as Storage::watch
    participant B as broadcast channel, capacity 1024<br/>solid-pod-rs/src/notifications/mod.rs:257

    C->>M: subscribe(topic, base_url)<br/>solid-pod-rs/src/notifications/mod.rs:278
    M-->>C: Subscription id
    C->>M: stream()<br/>solid-pod-rs/src/notifications/mod.rs:303
    M-->>C: broadcast::Receiver
    S->>P: StorageEvent
    P->>B: ChangeNotification, every event, no topic filter<br/>solid-pod-rs/src/notifications/mod.rs:323
    B-->>C: delivered to every live receiver
    C->>M: unsubscribe(id)<br/>solid-pod-rs/src/notifications/mod.rs:296
    Note over M: with_heartbeat sets the keepalive interval<br/>solid-pod-rs/src/notifications/mod.rs:266
    Note over B: A broadcast channel drops messages for a slow receiver rather than growing<br/>without bound — a stalled subscriber degrades its own stream, not the pod.
```

**Drift:** the doc comment on `pump_from_storage` says each event goes to clients "whose subscription topic covers the event path" (`solid-pod-rs/src/notifications/mod.rs:314`); the body sends every event to the one broadcast channel (`solid-pod-rs/src/notifications/mod.rs:323`). Topic filtering and any WAC check are left to the transport that reads the receiver.

## SP-16.3 Webhook delivery — signing, retry and the circuit breaker

```mermaid
stateDiagram-v2
    [*] --> Closed
    Closed --> Delivering: deliver_one<br/>solid-pod-rs/src/notifications/mod.rs:600
    Delivering --> Closed: 2xx
    Delivering --> Backoff: failure or Retry-After
    Backoff --> Delivering: compute_backoff with jitter<br/>solid-pod-rs/src/notifications/mod.rs:533
    Backoff --> Failed: max_attempts reached<br/>solid-pod-rs/src/notifications/mod.rs:447
    Failed --> Open: consecutive_failures crosses the threshold<br/>solid-pod-rs/src/notifications/mod.rs:463
    Open --> Closed: reset_circuit<br/>solid-pod-rs/src/notifications/mod.rs:484

    note right of Delivering
      Each request is signed per RFC 9421 when a SignerConfig is wired
      (solid-pod-rs/src/notifications/mod.rs:441): sign_request
      (solid-pod-rs/src/notifications/signing.rs:154) over the COVERED_COMPONENTS
      list (solid-pod-rs/src/notifications/signing.rs:31), with a
      Content-Digest header (solid-pod-rs/src/notifications/signing.rs:207).
      verify_signed_request (solid-pod-rs/src/notifications/signing.rs:313) is the
      receiving half.
    end note
    note right of Backoff
      Backoff is capped by with_max_backoff
      (solid-pod-rs/src/notifications/mod.rs:456) and jittered within the top
      20 percent of the cap (solid-pod-rs/src/notifications/mod.rs:823).
    end note
    note right of Open
      An open breaker returns before any send
      (solid-pod-rs/src/notifications/mod.rs:602), so no delivery can close it;
      only reset_circuit does. The counter is one per manager
      (solid-pod-rs/src/notifications/mod.rs:398), shared by every subscription.
    end note
```

**Tension:** the breaker is one counter per manager, not per target. Any 2xx or 410 from any subscriber's target resets it (`solid-pod-rs/src/notifications/mod.rs:618`, `solid-pod-rs/src/notifications/mod.rs:624`), so one dead endpoint among healthy ones rarely trips it. It opens after ten consecutive exhausted deliveries across the whole manager with no success in between (a correlated outage, or a dead sole subscriber). Once open it is permanent until an operator calls `reset_circuit`, because the open check at `solid-pod-rs/src/notifications/mod.rs:602` runs before any send, and it then blocks every subscriber. A per-URL breaker, or a half-open timer, would bound both effects.

**Drift:** the struct comment says the circuit stays open "until a successful delivery resets the counter" (`solid-pod-rs/src/notifications/mod.rs:380`); while it is open no delivery is attempted, so that cannot happen. `with_max_attempts` says "Default 5" (`solid-pod-rs/src/notifications/mod.rs:446`); the constructor sets `max_retries: 3`, four attempts (`solid-pod-rs/src/notifications/mod.rs:421`).

## SP-16.4 The two legacy `solid-0.1` adapters

```mermaid
sequenceDiagram
    autonumber
    participant C as SolidOS data browser
    participant CH as LegacyNotificationChannel — the driver's adapter<br/>solid-pod-rs/src/notifications/legacy.rs:221
    participant AZ as SubscriptionAuthorizer<br/>solid-pod-rs/src/notifications/legacy.rs:152
    participant S as LegacyWebSocketSession — second adapter<br/>solid-pod-rs/src/notifications/legacy.rs:729
    participant W as LegacyWacRead<br/>solid-pod-rs/src/notifications/legacy.rs:666

    C->>CH: sub uri, parsed by parse_subscribe<br/>solid-pod-rs/src/notifications/legacy.rs:447
    CH->>AZ: subscribe checks URL cap, origin, per-connection cap, then the authoriser<br/>solid-pod-rs/src/notifications/legacy.rs:339
    alt denied
        AZ-->>CH: DenyReason<br/>solid-pod-rs/src/notifications/legacy.rs:131
        CH-->>C: err line<br/>solid-pod-rs/src/notifications/legacy.rs:462
    else granted
        CH-->>C: ack line<br/>solid-pod-rs/src/notifications/legacy.rs:457
    end
    Note over CH: On a storage event the channel tests matches_subscription<br/>(solid-pod-rs/src/notifications/legacy.rs:393) and emits to_legacy_line<br/>(solid-pod-rs/src/notifications/legacy.rs:436). No WAC re-check at fan-out.
    C->>S: sub uri, handle_message<br/>solid-pod-rs/src/notifications/legacy.rs:785
    S->>W: can_read at subscribe<br/>solid-pod-rs/src/notifications/legacy.rs:825
    W-->>S: allowed or forbidden
    S->>W: can_read again on every change<br/>solid-pod-rs/src/notifications/legacy.rs:886
    S-->>C: pub uri, or err uri forbidden after revocation

    Note over AZ: INVARIANT: a channel subscription is authorised, and the default authoriser is<br/>DenyAllAuthorizer (solid-pod-rs/src/notifications/legacy.rs:255): a channel nobody<br/>configured rejects every sub. AllowAllAuthorizer (:164) and DenyAllAuthorizer (:178)<br/>are the two poles.
    Note over CH: Bounds: MAX_SUBSCRIPTIONS_PER_CONNECTION 100<br/>(solid-pod-rs/src/notifications/legacy.rs:75) and MAX_URL_LENGTH 2048<br/>(solid-pod-rs/src/notifications/legacy.rs:78). PROTOCOL_GREETING<br/>(solid-pod-rs/src/notifications/legacy.rs:88) is sent by the driver, SP-16.5.
```

**What it shows.** The module holds two independent adapters. `LegacyNotificationChannel` authorises once, at `sub`, through a synchronous `SubscriptionAuthorizer`. `LegacyWebSocketSession` checks an asynchronous `LegacyWacRead` at `sub` and again on every change, and tells a revoked reader `forbidden`.

**Tension:** the library driver (SP-16.5) still wraps the channel, not the session, so a consumer that wires the driver never re-checks access after `ack`. The reference server no longer does: its `/.notifications` route builds `LegacyWebSocketSession` (`solid-pod-rs-server/src/notifications_ws.rs:331`) and re-checks Read on every change (SP-16.6). The driver path remains open.

## SP-16.5 The legacy `solid-0.1` driver — the crate mounts nothing

```mermaid
sequenceDiagram
    autonumber
    participant T as the CONSUMER's transport<br/>actix-ws, axum ws, tungstenite
    participant D as LegacyWsDriver<br/>solid-pod-rs/src/handlers/legacy_notifications.rs:91
    participant R as run_loop<br/>solid-pod-rs/src/handlers/legacy_notifications.rs:167
    participant H as handle_line<br/>solid-pod-rs/src/handlers/legacy_notifications.rs:241
    participant CH as LegacyNotificationChannel — see SP-16.4

    T->>T: perform the HTTP to WebSocket upgrade ITSELF<br/>solid-pod-rs/src/handlers/legacy_notifications.rs:9
    T->>D: new(storage event Receiver)<br/>solid-pod-rs/src/handlers/legacy_notifications.rs:99
    D->>R: drive three inputs — storage events, inbound text, heartbeat
    R-->>T: OutboundFrame::Text with the protocol greeting, FIRST<br/>solid-pod-rs/src/handlers/legacy_notifications.rs:174
    T->>R: an inbound text frame
    R->>H: parse the line
    H->>CH: sub / unsub, authorised at sub only
    CH-->>H: ack or err, and an unknown opcode closes the socket<br/>solid-pod-rs/src/handlers/legacy_notifications.rs:264
    H-->>R: Vec of OutboundFrame<br/>solid-pod-rs/src/handlers/legacy_notifications.rs:78
    CH-->>R: a storage event<br/>solid-pod-rs/src/handlers/legacy_notifications.rs:204
    R->>R: pub line if matches_subscription<br/>solid-pod-rs/src/handlers/legacy_notifications.rs:211
    R-->>T: frames on an mpsc channel the transport writes

    Note over D: INVARIANT: this crate does NOT mount itself at /ws/solid-0.1<br/>(solid-pod-rs/src/handlers/legacy_notifications.rs:6). The consumer owns the<br/>upgrade and the route — the recommended path is only a recommendation<br/>(solid-pod-rs/src/handlers/legacy_notifications.rs:18). That is the same<br/>ownership line the ecosystem doc draws — this crate owns protocol, not routing.
    Note over R: The outbound mpsc is capacity-bounded (default 256,<br/>solid-pod-rs/src/handlers/legacy_notifications.rs:102) and overridable<br/>(:108). Only pub frames use try_send and drop under back-pressure — lossy by<br/>design, matching JSS. Ack, err and heartbeat frames wait.
    Note over CH: This is the DRIVER half — notifications/legacy.rs is the protocol half<br/>(SP-16.4). Neither is reachable without the default-off legacy-notifications<br/>feature — see SP-01.6.
```

**Open:** `matches_subscription` compares the stored subscription string with the storage event's path as-is (`solid-pod-rs/src/notifications/legacy.rs:393`, called at `solid-pod-rs/src/handlers/legacy_notifications.rs:211`). A client subscribes with an absolute URL, and the origin check expects one; whether a `pub` line is ever emitted therefore depends on the form of path the consumer's storage events carry. The driver does not normalise either side. Only a running consumer can show which.

## SP-16.6 `/.notifications` in the reference server — Read checked per subscription and per delivery

```mermaid
sequenceDiagram
    autonumber
    participant C as client, browser or agent
    participant H as handle_notifications<br/>solid-pod-rs-server/src/notifications_ws.rs:291
    participant HUB as NotifyHub, one storage watcher<br/>solid-pod-rs-server/src/notifications_ws.rs:90
    participant SES as LegacyWebSocketSession<br/>solid-pod-rs/src/notifications/legacy.rs:729
    participant CK as run_checker, WAC Read<br/>solid-pod-rs-server/src/notifications_ws.rs:266

    C->>H: GET /.notifications upgrade, the dotfile guard admits GET only<br/>solid-pod-rs-server/src/lib.rs:5004
    H->>H: an Origin outside a non-empty allowed_origins is 403 (:301)
    H->>HUB: admit, 1024 sockets and 16 per peer address, else 503 (:306)
    H->>HUB: ensure_started watches the storage root once (:309)
    H->>H: anonymous unless a NIP-98 header verifies (:315)
    C->>SES: sub url, handle_message<br/>solid-pod-rs/src/notifications/legacy.rs:785
    SES->>CK: can_read at subscribe
    CK->>CK: local_path, same origin, no dot segments or encoded separators (:244)
    CK->>CK: enforce_read_inner with charge false, a paid read is refused (:275)
    HUB->>SES: storage event as origin plus path (:373)
    SES->>CK: can_read on the changed resource<br/>solid-pod-rs/src/notifications/legacy.rs:861
    SES-->>C: only a pub frame leaves, the err forbidden frame is dropped<br/>solid-pod-rs-server/src/notifications_ws.rs:378
    Note over HUB: Pinged every 30 s, closed after 90 s silent so the slot returns<br/>(solid-pod-rs-server/src/notifications_ws.rs:393). Lagged events are skipped.
    Note over H: INVARIANT: a subscriber never learns that a resource it cannot read<br/>changed. Mounted by default, --no-notifications clears it (SP-16.1).
```

The route wires the session that re-checks access, not the library driver of SP-16.5. With `allowed_origins` empty any browser origin may connect, which is safe only while NIP-98 is the sole credential (`solid-pod-rs-server/src/notifications_ws.rs:44`).

## SP-16.7 The change-notification envelope

```mermaid
classDiagram
    class StorageEvent {
        Created, Updated or Deleted, each carrying a path  solid-pod-rs/src/notifications/mod.rs:91-93
    }
    class ChangeNotification {
        +context, serialised as @context  solid-pod-rs/src/notifications/mod.rs:78
        +id  solid-pod-rs/src/notifications/mod.rs:80
        +kind, serialised as type  solid-pod-rs/src/notifications/mod.rs:81
        +object  solid-pod-rs/src/notifications/mod.rs:83
        +published  solid-pod-rs/src/notifications/mod.rs:84
        +from_storage_event(event, pod_base)  solid-pod-rs/src/notifications/mod.rs:89
    }
    class as_ns {
        CONTEXT activitystreams  solid-pod-rs/src/notifications/mod.rs:46
        CREATE, UPDATE, DELETE  solid-pod-rs/src/notifications/mod.rs:47
    }
    class WebSocketChannelManager {
        +stream, a broadcast Receiver of ChangeNotification  solid-pod-rs/src/notifications/mod.rs:303
    }
    class WebhookChannelManager {
        -send_once, POST of serde_json bytes  solid-pod-rs/src/notifications/mod.rs:549-554
    }
    StorageEvent ..> ChangeNotification : mapped by from_storage_event
    as_ns ..> ChangeNotification : supplies context and type
    ChangeNotification ..> WebSocketChannelManager : broadcast as the typed value
    ChangeNotification ..> WebhookChannelManager : serialised per delivery
    note for ChangeNotification "Created maps to Create, Updated to Update, Deleted to Delete\n(solid-pod-rs/src/notifications/mod.rs:91).\nobject is pod_base without its trailing slash plus the path (:95),\nid is urn:uuid with a fresh v4 (:98), published is RFC 3339 now (:100).\nOne id per event: the webhook retry loop resends the same note (:611),\nso a receiver can de-duplicate retries by id."
    note for WebhookChannelManager "Content-Type application/ld+json\n(solid-pod-rs/src/notifications/mod.rs:561); the same bytes are what\nwebhook-signing signs when a signer is configured (SP-16.3)."
```

- **What it shows:** the one payload both Notifications 0.2 managers carry, an Activity Streams 2.0 activity built from a storage event. No actor, no target and no state accompany it: a subscriber learns which resource changed and how, then reads it.
- **Why it is this way:** the shape follows the Solid Notifications 0.2 examples (`solid-pod-rs/src/notifications/mod.rs:13`). As SP-16.1 records, the reference server mounts neither manager, so this envelope reaches only an embedder that wires them.
