---
id: SP-08
title: Federation and outward surfaces — ActivityPub, NIP-01 relay, Solid Notifications, the forge, MCP
area: solid-pod-rs
governing: [../solid-pod-rs/README.md, ../solid-pod-rs/crates/solid-pod-rs/docs/explanation/ecosystem-integration.md]
adrs: [ADR-2005, ADR-2006, ADR-2009, ADR-2010]
sources:
  - ../solid-pod-rs/crates/solid-pod-rs-activitypub/src/actor.rs
  - ../solid-pod-rs/crates/solid-pod-rs-activitypub/src/inbox.rs
  - ../solid-pod-rs/crates/solid-pod-rs-activitypub/src/outbox.rs
  - ../solid-pod-rs/crates/solid-pod-rs-activitypub/src/delivery.rs
  - ../solid-pod-rs/crates/solid-pod-rs-activitypub/src/http_sig.rs
  - ../solid-pod-rs/crates/solid-pod-rs-activitypub/src/store.rs
  - ../solid-pod-rs/crates/solid-pod-rs-activitypub/src/ssrf.rs
  - ../solid-pod-rs/crates/solid-pod-rs-activitypub/src/error.rs
  - ../solid-pod-rs/crates/solid-pod-rs-nostr/src/relay.rs
  - ../solid-pod-rs/crates/solid-pod-rs-nostr/src/ws.rs
  - ../solid-pod-rs/crates/solid-pod-rs-nostr/src/typestate.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/notifications/mod.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/notifications/signing.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/notifications/legacy.rs
  - ../solid-pod-rs/crates/solid-pod-rs-forge/src/lib.rs
  - ../solid-pod-rs/crates/solid-pod-rs-forge/src/router.rs
  - ../solid-pod-rs/crates/solid-pod-rs-forge/src/auth.rs
  - ../solid-pod-rs/crates/solid-pod-rs-forge/src/token.rs
  - ../solid-pod-rs/crates/solid-pod-rs-forge/src/ownership.rs
  - ../solid-pod-rs/crates/solid-pod-rs-forge/src/hosted.rs
  - ../solid-pod-rs/crates/solid-pod-rs-forge/src/bodies.rs
  - ../solid-pod-rs/crates/solid-pod-rs-forge/src/spine/mod.rs
  - ../solid-pod-rs/crates/solid-pod-rs-forge/src/spine/issues.rs
  - ../solid-pod-rs/crates/solid-pod-rs-forge/src/repo/browse.rs
  - ../solid-pod-rs/crates/solid-pod-rs-forge/src/repo/mod.rs
  - ../solid-pod-rs/crates/solid-pod-rs-forge/src/html/views.rs
  - ../solid-pod-rs/crates/solid-pod-rs-forge/src/request.rs
  - ../solid-pod-rs/crates/solid-pod-rs-forge/src/config.rs
  - ../solid-pod-rs/crates/solid-pod-rs-forge/src/api.rs
  - ../solid-pod-rs/crates/solid-pod-rs-forge/src/api_issues.rs
  - ../solid-pod-rs/crates/solid-pod-rs-forge/src/labels.rs
  - ../solid-pod-rs/crates/solid-pod-rs-forge/src/pulls.rs
  - ../solid-pod-rs/crates/solid-pod-rs-forge/src/trail.rs
  - ../solid-pod-rs/crates/solid-pod-rs/docs/adr/ADR-2009-forge-plugins-comparator-and-default-off-mirror.md
  - ../solid-pod-rs/crates/solid-pod-rs/docs/adr/ADR-2010-forge-and-mirror-are-on-in-shipped-builds.md
  - ../solid-pod-rs/crates/solid-pod-rs/docs/explanation/security-model.md
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/main.rs
  - ../project/agentbox/lib/solid-pod-rs.nix
  - ../solid-pod-rs/crates/solid-pod-rs-server/Cargo.toml
  - ../solid-pod-rs/crates/solid-pod-rs-forge/Cargo.toml
  - ../solid-pod-rs/crates/solid-pod-rs-forge/README.md
  - ../solid-pod-rs/crates/solid-pod-rs-forge/src/web_edit.rs
  - ../solid-pod-rs/crates/solid-pod-rs-forge/src/mirror.rs
  - ../solid-pod-rs/crates/solid-pod-rs-forge/src/publish.rs
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/mcp/mod.rs
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/mcp/tools.rs
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/mcp/skills.rs
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/lib.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/handlers/legacy_notifications.rs
verified_commit: {solid-pod-rs: 27e18995fc20790fa956acb41f7d8420544a13ec, agentbox: c57d9c0508f4044915f9aecea4b9aa5629cc6a35}
---

## SP-08.1 Solid Notifications 0.2 — the three channels

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
    N2["Updates-Via advertises wss://base/.notifications (solid-pod-rs-server/src/lib.rs:1417)<br/>on GET, HEAD and OPTIONS responses only (solid-pod-rs-server/src/lib.rs:1575,<br/>solid-pod-rs-server/src/lib.rs:2571). The reference server mounts nothing there: the<br/>path falls to the LDP catch-all (solid-pod-rs-server/src/lib.rs:5781). See SP-03.11."]
    WS -.-> N2
```

**What it shows.** The core crate defines two Notifications 0.2 channel managers, the legacy `solid-0.1` adapter and the discovery document. They are library types: none of them is instantiated by `solid-pod-rs-server`.

**Why it is this way.** The crate owns protocol and leaves routing to the embedder (the same line SP-08.17 draws). The consequence is that the reference server advertises a channel it does not serve.

**Tension:** `set_updates_via` points every GET, HEAD and OPTIONS response at `/.notifications` on the WebSocket scheme, but no route in the server builds a `WebSocketChannelManager` or upgrades that path, so a client that follows the header reaches the LDP GET handler (`solid-pod-rs-server/src/lib.rs:5781`). Either the header should be dropped from the reference server or a channel mounted behind it.

**Drift:** the subscription cap note in earlier readings implied the delivering channels were bounded. Only `InMemoryNotifications` is capped, and it delivers nothing.

## SP-08.2 WebSocket notification pump

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

## SP-08.3 Webhook delivery — signing, retry and the circuit breaker

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

## SP-08.4 The two legacy `solid-0.1` adapters

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
    Note over CH: Bounds: MAX_SUBSCRIPTIONS_PER_CONNECTION 100<br/>(solid-pod-rs/src/notifications/legacy.rs:75) and MAX_URL_LENGTH 2048<br/>(solid-pod-rs/src/notifications/legacy.rs:78). PROTOCOL_GREETING<br/>(solid-pod-rs/src/notifications/legacy.rs:88) is sent by the driver, SP-08.17.
```

**What it shows.** The module holds two independent adapters. `LegacyNotificationChannel` authorises once, at `sub`, through a synchronous `SubscriptionAuthorizer`. `LegacyWebSocketSession` checks an asynchronous `LegacyWacRead` at `sub` and again on every change, and tells a revoked reader `forbidden`.

**Tension:** the transport driver (SP-08.17) wraps the channel, not the session, so the path a consumer actually wires never re-checks access after `ack`. A reader whose grant is revoked keeps receiving `pub` lines for that subtree until it disconnects. Only the integration tests construct `LegacyWebSocketSession`. Either the driver should use the session, or the channel should re-check on fan-out.

## SP-08.5 ActivityPub — signature verification and the inbox

```mermaid
sequenceDiagram
    autonumber
    participant R as remote instance
    participant E as embedder's HTTP layer
    participant V as verify_request_signature<br/>solid-pod-rs-activitypub/src/http_sig.rs:375
    participant K as HttpActorKeyResolver<br/>solid-pod-rs-activitypub/src/http_sig.rs:96
    participant SS as SSRF guard<br/>solid-pod-rs-activitypub/src/ssrf.rs:47
    participant I as handle_inbox<br/>solid-pod-rs-activitypub/src/inbox.rs:50
    participant S as Store<br/>solid-pod-rs-activitypub/src/store.rs:30

    R->>E: POST an activity to the inbox
    E->>V: verify the HTTP Signature
    V->>V: parse_signature_header<br/>solid-pod-rs-activitypub/src/http_sig.rs:181
    V->>V: validate_covered_components<br/>solid-pod-rs-activitypub/src/http_sig.rs:303
    V->>V: check_date_freshness — the replay window<br/>solid-pod-rs-activitypub/src/http_sig.rs:339
    V->>V: digest_header must match the body<br/>solid-pod-rs-activitypub/src/http_sig.rs:325
    V->>K: resolve the actor's public key
    K->>SS: resolve_ssrf_safe before fetching<br/>solid-pod-rs-activitypub/src/ssrf.rs:66
    K-->>V: VerifiedActor<br/>solid-pod-rs-activitypub/src/http_sig.rs:76
    V-->>E: verified, or SigError<br/>solid-pod-rs-activitypub/src/error.rs:12
    E->>I: activity plus the VerifiedActor
    I->>I: claimed actor must equal the verified actor<br/>solid-pod-rs-activitypub/src/inbox.rs:65
    I->>S: record_inbox — dedup by activity id<br/>solid-pod-rs-activitypub/src/store.rs:249
    alt the activity is a Follow
        I->>S: add_follower<br/>solid-pod-rs-activitypub/src/store.rs:133
        I->>I: build_accept<br/>solid-pod-rs-activitypub/src/inbox.rs:158
    end
    I-->>E: InboxOutcome<br/>solid-pod-rs-activitypub/src/inbox.rs:21

    Note over V: The covered-component list is validated rather than trusted: request-target,<br/>host and date always, digest for POST — accepting whichever headers the sender<br/>chose to sign is the classic HTTP-Signature bypass.
    Note over K: The actor-key fetch is SSRF-guarded because the key URL comes from the<br/>UNTRUSTED signature header. is_private_ip<br/>(solid-pod-rs-activitypub/src/ssrf.rs:33) is the classifier.
    Note over I: with_also_known_as (solid-pod-rs-activitypub/src/actor.rs:182) is the bridge<br/>that lets one ActivityPub actor declare the same subject as a WebID and a<br/>did:nostr — see SP-05.12 and ES-04. negotiate_actor_format<br/>(solid-pod-rs-activitypub/src/actor.rs:157) drives conneg between the AS2<br/>ActorFormat::ActivityJson and the Solid/LDP profile (solid-pod-rs-activitypub/src/actor.rs:139).
```

**What it shows.** Verification and inbox handling are separate calls. `handle_inbox` does not verify; it requires a `VerifiedActor`, binds the activity's `actor` to it, then records and dispatches.

**Invariant:** nothing is stored before the claimed actor matches the verified actor (`solid-pod-rs-activitypub/src/inbox.rs:65` precedes `solid-pod-rs-activitypub/src/inbox.rs:72`).

**Tension:** "verify before store" is a caller contract, not a type guarantee. `VerifiedActor` has public fields (`solid-pod-rs-activitypub/src/http_sig.rs:76`) and the doc invites soft-logged callers to hand one through (`solid-pod-rs-activitypub/src/inbox.rs:48`). The reference server mounts no inbox at all: with `activitypub` it serves only the actor document at `/profile/card.jsonld` (`solid-pod-rs-server/src/lib.rs:1515`).

## SP-08.6 ActivityPub — outbox and delivery

```mermaid
sequenceDiagram
    autonumber
    participant L as local actor
    participant O as handle_outbox_post<br/>solid-pod-rs-activitypub/src/outbox.rs:143
    participant W as wrap_note_in_create<br/>solid-pod-rs-activitypub/src/outbox.rs:100
    participant HO as handle_outbox<br/>solid-pod-rs-activitypub/src/outbox.rs:33
    participant S as Store
    participant D as DeliveryWorker::drain_once<br/>solid-pod-rs-activitypub/src/delivery.rs:105
    participant R as remote inbox

    L->>O: POST an object to the outbox
    O->>W: a bare Note is wrapped in a Create
    O->>HO: the activity
    HO->>S: record_outbox<br/>solid-pod-rs-activitypub/src/store.rs:299
    HO->>S: follower_inboxes for fan-out, or targetInbox for a Follow<br/>solid-pod-rs-activitypub/src/store.rs:180
    HO->>S: enqueue_delivery, one row per inbox<br/>solid-pod-rs-activitypub/src/outbox.rs:83
    loop DeliveryWorker::run tick<br/>solid-pod-rs-activitypub/src/delivery.rs:248
        D->>S: next_due_delivery<br/>solid-pod-rs-activitypub/src/store.rs:358
        D->>D: resolve_ssrf_safe and pin the address, no redirects<br/>solid-pod-rs-activitypub/src/delivery.rs:124
        D->>D: sign_request<br/>solid-pod-rs-activitypub/src/http_sig.rs:436
        D->>R: POST the activity
        alt delivered
            D->>S: drop_delivery<br/>solid-pod-rs-activitypub/src/store.rs:377
            D->>S: mark_outbox_state<br/>solid-pod-rs-activitypub/src/store.rs:323
        else 4xx other than 408 and 429
            D->>S: drop and mark failed<br/>solid-pod-rs-activitypub/src/delivery.rs:175
        else other failure
            D->>S: reschedule_delivery with backoff<br/>solid-pod-rs-activitypub/src/store.rs:385
        end
    end
    Note over D: DeliveryOutcome (solid-pod-rs-activitypub/src/delivery.rs:34) and<br/>DeliveryConfig (:49) make the retry policy explicit. enqueue_to_inboxes<br/>(solid-pod-rs-activitypub/src/delivery.rs:232) is a separate helper, not on this path.
```

**Drift:** `handle_outbox` says "Ensure actor field is present and matches" (`solid-pod-rs-activitypub/src/outbox.rs:57`) but only fills a missing `actor`; a caller-supplied foreign `actor` is kept and then signed with the pod's key. The caller is the local owner, so this is a correctness gap rather than an external bypass.

## SP-08.8 The embedded NIP-01 relay

```mermaid
flowchart TD
    WS["serve_relay_ws / serve_relay_ws_stream<br/>solid-pod-rs-nostr/src/ws.rs:136"]
    LIM["RelayLimits<br/>solid-pod-rs-nostr/src/ws.rs:32"]
    DISP["dispatch_message_with_limits<br/>solid-pod-rs-nostr/src/ws.rs:164"]
    EV["handle_event<br/>solid-pod-rs-nostr/src/ws.rs:194"]
    UV["UncheckedEvent::verify then ingest_verified<br/>solid-pod-rs-nostr/src/ws.rs:222<br/>solid-pod-rs-nostr/src/ws.rs:225"]
    REQ["handle_req -> Relay::history<br/>solid-pod-rs-nostr/src/ws.rs:233"]
    CLS["handle_close<br/>solid-pod-rs-nostr/src/ws.rs:290"]
    ING["Relay::ingest — the untyped entry, verifies itself<br/>solid-pod-rs-nostr/src/relay.rs:416"]
    VER["Event::verify — Schnorr plus canonical id<br/>solid-pod-rs-nostr/src/relay.rs:63"]
    STORE["EventStore trait, InMemoryEventStore<br/>solid-pod-rs-nostr/src/relay.rs:225"]
    BC["Relay::broadcast to live subscribers<br/>solid-pod-rs-nostr/src/relay.rs:391"]
    FIL["Filter::matches<br/>solid-pod-rs-nostr/src/relay.rs:150"]
    INFO["RelayInfo::jss_compatible — NIP-11<br/>solid-pod-rs-nostr/src/relay.rs:319"]

    WS --> LIM --> DISP
    DISP --> EV --> UV --> VER
    UV --> STORE
    UV --> BC
    ING --> VER
    ING --> STORE
    DISP --> REQ --> FIL
    DISP --> CLS
    INFO --> WS

    N["Kind classes decide retention: is_replaceable<br/>(solid-pod-rs-nostr/src/relay.rs:206), is_ephemeral (:210) and<br/>is_parameterised_replaceable (:214) — an ephemeral event is broadcast but never<br/>stored, so the store cannot be filled with transient traffic."]
    STORE -.-> N
    N2["canonical_id (solid-pod-rs-nostr/src/relay.rs:48) recomputes the NIP-01 id from<br/>content — the same never-trust-the-claimed-id rule NIP-98 applies in SP-05.2."]
    VER -.-> N2
    N3["notice (solid-pod-rs-nostr/src/ws.rs:298) and ok_frame (solid-pod-rs-nostr/src/ws.rs:302)<br/>are the two NIP-01 reply shapes; InMemoryEventStore is capped at construction,<br/>oldest evicted first (solid-pod-rs-nostr/src/relay.rs:251)."]
    DISP -.-> N3
```

**What it shows.** The WebSocket path takes the typestate route (SP-08.9); `Relay::ingest` is a second, untyped entry that verifies internally. Both reach the same NIP-16 retention rules. The reference server mounts no relay endpoint; this is a library surface for embedders.

## SP-08.9 The relay typestate — verification on the typed path

```mermaid
stateDiagram-v2
    [*] --> Unchecked: UncheckedEvent.new<br/>solid-pod-rs-nostr/src/typestate.rs:138
    Unchecked --> Verified: verify()<br/>solid-pod-rs-nostr/src/typestate.rs:187
    Unchecked --> Rejected: VerifyError<br/>solid-pod-rs-nostr/src/typestate.rs:80
    Unchecked --> Escaped: into_inner_unchecked — named escape<br/>solid-pod-rs-nostr/src/typestate.rs:197
    Verified --> Ingested: ingest_verified<br/>solid-pod-rs-nostr/src/typestate.rs:339
    Rejected --> [*]
    Ingested --> [*]

    note right of Verified
      VerifiedEvent (solid-pod-rs-nostr/src/typestate.rs:235) is the only type
      ingest_verified accepts, so forgetting to verify on that path is a COMPILE
      error. Accessors like d_tag (:281) and get_tag (:287) hang off the
      verified type.
    end note
    note right of Escaped
      into_inner_unchecked yields a raw Event and is named so an audit can grep
      for it. It is not the only way round verification: Relay::store is public
      (solid-pod-rs-nostr/src/relay.rs:378) and EventStore::put takes any Event
      (solid-pod-rs-nostr/src/relay.rs:227).
    end note
```

**Tension:** the panel's earlier title, "verification cannot be skipped", held only for `ingest_verified`. Any holder of a `Relay` can call `relay.store().put(event)` and persist an unverified event that later `REQ` history returns. Making `store()` crate-private, or giving it a read-only view, would make the typestate claim true for the whole relay.

## SP-08.10 The forge — request dispatch

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant SRV as handle_forge<br/>solid-pod-rs-server/src/lib.rs:5420
    participant RA as ForgeService::resolve_agent<br/>solid-pod-rs-forge/src/lib.rs:369
    participant A as auth::resolve_agent<br/>solid-pod-rs-forge/src/auth.rs:29
    participant F as ForgeService::handle<br/>solid-pod-rs-forge/src/lib.rs:419
    participant R as parse_route<br/>solid-pod-rs-forge/src/router.rs:398
    participant H as the matching h_* handler<br/>solid-pod-rs-forge/src/lib.rs:518

    C->>SRV: any method under /forge
    SRV->>SRV: build ForgeRequest<br/>solid-pod-rs-forge/src/request.rs:23
    SRV->>RA: resolve the caller first<br/>solid-pod-rs-server/src/lib.rs:5471
    RA->>A: with the host pinned to base_url when configured
    A->>A: 1. forge push token, Bearer f1.*<br/>solid-pod-rs-forge/src/auth.rs:42
    A->>A: 2. NIP-98, body-bound when a body exists, replay-checked<br/>solid-pod-rs-forge/src/auth.rs:58<br/>solid-pod-rs-forge/src/auth.rs:59
    A-->>SRV: ForgeAgent<br/>solid-pod-rs-forge/src/ownership.rs:16
    SRV->>F: request and agent
    F->>F: strip_prefix, percent_guard, parse_route<br/>solid-pod-rs-forge/src/router.rs:377<br/>solid-pod-rs-forge/src/lib.rs:449
    F->>R: parse
    R-->>F: Route<br/>solid-pod-rs-forge/src/router.rs:12
    F->>F: private_gate when private_repos is on<br/>solid-pod-rs-forge/src/lib.rs:454
    F->>H: dispatch
    H-->>F: ForgeResponse<br/>solid-pod-rs-forge/src/request.rs:85
    F-->>C: finish_page adds CSP and nosniff to HTML<br/>solid-pod-rs-forge/src/lib.rs:474

    Note over A: A missing or invalid credential resolves to ForgeAgent::Anonymous rather than<br/>an error (solid-pod-rs-forge/src/auth.rs:36, solid-pod-rs-forge/src/auth.rs:46,<br/>solid-pod-rs-forge/src/auth.rs:68). Handlers then apply their own owner checks.
    Note over H: Every repository reader resolves through resolve_repo<br/>(solid-pod-rs-forge/src/lib.rs:621) or gated_git_dir (solid-pod-rs-forge/src/lib.rs:752),<br/>which 503 an unmigrated mirror (solid-pod-rs-forge/src/lib.rs:651) — see SP-14.4.
    Note over R: GitSmart is rebuilt, not forwarded verbatim: only owner/name[.git]/suffix is<br/>accepted (solid-pod-rs-forge/src/lib.rs:798) and the CGI always gets<br/>owner/name.git plus the suffix (solid-pod-rs-forge/src/lib.rs:803). See SP-06.12.
```

**Drift:** `auth.rs` numbers NIP-98 as "Scheme 3" and says scheme 2, a pod session, "is injected by the server" (`solid-pod-rs-forge/src/auth.rs:50`). The reference server never injects one: it always calls `resolve_agent`, and its own comment says a pod session "would be" injected (`solid-pod-rs-server/src/lib.rs:5470`). In this server a forge caller is a token holder, a NIP-98 signer or anonymous.

## SP-08.11 The forge push token

```mermaid
flowchart LR
    KEY["load_or_create_token_key — 32 bytes in .forge-token-key<br/>solid-pod-rs-forge/src/lib.rs:186"]
    MINT["token::mint(key, agent, iat, ttl)<br/>solid-pod-rs-forge/src/token.rs:66"]
    VER["token::verify(key, token, now), constant-time<br/>solid-pod-rs-forge/src/token.rs:87"]
    MAC["mac_tag — HMAC-SHA256 over the payload<br/>solid-pod-rs-forge/src/token.rs:126"]
    V1["TOKEN_VERSION 1, the f1. prefix<br/>solid-pod-rs-forge/src/token.rs:25"]
    ERR["TokenError<br/>solid-pod-rs-forge/src/token.rs:45"]
    API["POST /api/token — Route::ApiToken<br/>solid-pod-rs-forge/src/lib.rs:577"]

    KEY --> MINT --> MAC
    V1 --> MINT
    API --> MINT
    MAC --> VER --> ERR

    N["DIVERGENCE: this is a bespoke HMAC token format rather than an established one.<br/>It exists for the podless did:nostr identity — a git client that cannot mint a<br/>NIP-98 event per request holds a short-lived bearer instead. The version prefix<br/>at least makes a future format migration detectable."]
    MINT -.-> N
    N2["The key is generated on first use and persisted under the plugin dir, so it<br/>survives a restart but is not shared across deployments."]
    KEY -.-> N2
```

**Tension:** a token is "short-lived" only per token. `h_api_token` mints for any non-anonymous agent (`solid-pod-rs-forge/src/lib.rs:1465`), and a valid `f1.` bearer resolves to that agent, so a holder can renew indefinitely. There is no revocation short of replacing the key file. Requiring NIP-98 (not a bearer) to mint would bound a stolen token's life.

**Debt:** the key is two v4 UUIDs (`solid-pod-rs-forge/src/lib.rs:196`), 244 bits from the OS RNG rather than a direct 256-bit fill, and it is written (`solid-pod-rs-forge/src/lib.rs:198`) before being set to `0600` (`solid-pod-rs-forge/src/lib.rs:202`), so it exists briefly with the process umask's mode. The MAC itself uses the RustCrypto `hmac` crate.

## SP-08.12 The forge ownership model

```mermaid
classDiagram
    class ForgeAgent {
        <<enum>>
        solid-pod-rs-forge/src/ownership.rs:16
        +owner()  solid-pod-rs-forge/src/ownership.rs:37
        +author_id()  solid-pod-rs-forge/src/ownership.rs:49
        +is_nostr()  solid-pod-rs-forge/src/ownership.rs:60
        +can_write_namespace(owner_segment)  solid-pod-rs-forge/src/ownership.rs:68
    }
    class OwnerKind {
        <<enum>>
        solid-pod-rs-forge/src/ownership.rs:78
        +classify_owner(segment)  solid-pod-rs-forge/src/ownership.rs:88
    }
    class Names {
        +valid_name_segment  solid-pod-rs-forge/src/ownership.rs:101
        +strip_git_suffix  solid-pod-rs-forge/src/router.rs:595
    }
    class ForgeConfig {
        +normalized_prefix  solid-pod-rs-forge/src/config.rs:180
        +chain_allowed  solid-pod-rs-forge/src/config.rs:192
    }
    ForgeAgent ..> OwnerKind
    OwnerKind ..> Names
    ForgeConfig ..> ForgeAgent
    note for ForgeAgent "Ownership is namespace equality: an agent owns exactly the first path\nsegment equal to its owner(). There is no forge ACL. The rule is applied\nroute by route, not through one predicate."
```

**What it shows.** Owner identity is one segment: a pod username or a 64-hex nostr key. Write rights follow from equality with that segment.

**Drift:** `can_write_namespace` is not the single predicate for mutating routes. It guards smart-HTTP push (`solid-pod-rs-forge/src/lib.rs:814`) and trail ownership (`solid-pod-rs-forge/src/trail.rs:921`). Other routes compare `agent.owner()` inline (`solid-pod-rs-forge/src/api.rs:515`, `solid-pod-rs-forge/src/labels.rs:296`) and also let the author of an issue or pull request act on it in someone else's namespace (`solid-pod-rs-forge/src/api_issues.rs:165`, `solid-pod-rs-forge/src/pulls.rs:765`). Two further rules sit beside ownership: `open_edit` bypasses it for edits (SP-08.18), and any write into a mirror's served store is refused with 409 (`solid-pod-rs-forge/src/lib.rs:733`).

## SP-08.13 Forge repository browsing and the issue spine

```mermaid
flowchart TD
    BR["repo::browse — git plumbing<br/>solid-pod-rs-forge/src/repo/browse.rs:49"]
    VR["valid_rev<br/>solid-pod-rs-forge/src/repo/browse.rs:111"]
    VP["valid_repo_path<br/>solid-pod-rs-forge/src/repo/browse.rs:125"]
    LT["list_tree -> parse_ls_tree<br/>solid-pod-rs-forge/src/repo/browse.rs:168"]
    RB["read_blob<br/>solid-pod-rs-forge/src/repo/browse.rs:229"]
    CL["commit_log / commit_patch<br/>solid-pod-rs-forge/src/repo/browse.rs:265<br/>solid-pod-rs-forge/src/repo/browse.rs:326"]
    SP["SpineStore trait, FsSpineStore<br/>solid-pod-rs-forge/src/spine/mod.rs:24"]
    AW["spine atomic_write<br/>solid-pod-rs-forge/src/spine/mod.rs:115"]
    IX["IssueIndex allocate / by_state / set_state<br/>solid-pod-rs-forge/src/spine/issues.rs:106"]
    LD["load_issue_index / save_issue_index<br/>solid-pod-rs-forge/src/spine/issues.rs:142"]
    TP["ThreadPointer<br/>solid-pod-rs-forge/src/spine/issues.rs:44"]
    RT["render_thread — resolve bodies from pods or the hosted store<br/>solid-pod-rs-forge/src/bodies.rs:189"]
    HS["HostedStore for podless authors<br/>solid-pod-rs-forge/src/hosted.rs:23"]
    VIEW["html::views — the rendered pages<br/>solid-pod-rs-forge/src/html/views.rs:10"]

    VR --> BR
    VP --> BR
    BR --> LT
    BR --> RB
    BR --> CL
    SP --> AW
    SP --> LD --> IX --> TP --> RT --> HS
    LT --> VIEW
    RB --> VIEW
    CL --> VIEW
    RT --> VIEW

    N["INVARIANT: valid_rev and valid_repo_path constrain every caller-supplied ref<br/>and path before it becomes a git argv element — the forge shells out, so an<br/>unvalidated rev is an argument-injection vector. list_tree, read_blob and<br/>commit_log call them first."]
    VR -.-> N
    N2["An issue THREAD is a pointer into a pod (or the hosted store), not a copy — the<br/>forge stores the index and the pod stores the content, so the author keeps<br/>custody of their own words. BodyOutcome (solid-pod-rs-forge/src/bodies.rs:105)<br/>is how an unresolvable body renders without breaking the page."]
    TP -.-> N2
    N3["esc (solid-pod-rs-forge/src/request.rs:181), sanitize_filename (:200) and<br/>looks_textual (:217) are the escaping and content-type helpers the views use;<br/>the views module calls its first tier Phase 0 (solid-pod-rs-forge/src/html/views.rs:1)."]
    VIEW -.-> N3
    N4["A full object id that does not resolve is 404 in commit_log, not an empty list<br/>(solid-pod-rs-forge/src/repo/browse.rs:298): a mirror commit that was never<br/>published answers like any other missing object (SP-14.4)."]
    CL -.-> N4
```

## SP-08.14 The pod as an MCP tool surface

```mermaid
sequenceDiagram
    autonumber
    participant A as agent
    participant H as handle_mcp<br/>solid-pod-rs-server/src/mcp/mod.rs:159
    participant D as dispatch<br/>solid-pod-rs-server/src/mcp/mod.rs:97
    participant T as call_tool<br/>solid-pod-rs-server/src/mcp/tools.rs:1129
    participant AU as authorise — the REST enforcement path<br/>solid-pod-rs-server/src/mcp/tools.rs:144
    participant W as wac_check — metadata tools<br/>solid-pod-rs-server/src/mcp/tools.rs:105
    participant S as Storage

    A->>H: POST /mcp with a JSON-RPC message
    H->>H: NIP-98 identity from the request<br/>solid-pod-rs-server/src/mcp/mod.rs:179
    H->>H: origin_of fills the McpCtx<br/>solid-pod-rs-server/src/mcp/mod.rs:154
    H->>H: a subscribe call upgrades to SSE here<br/>solid-pod-rs-server/src/mcp/mod.rs:198
    H->>D: other messages
    D->>D: is_allowed_method gate<br/>solid-pod-rs-server/src/mcp/mod.rs:83<br/>solid-pod-rs-server/src/mcp/mod.rs:101
    D->>T: tools/call
    alt resource read or write, skill read
        T->>AU: enforce_read_ctx or enforce_write_ctx
    else list, head, ACL
        T->>W: sidecar paths elevated to Control<br/>solid-pod-rs-server/src/mcp/tools.rs:111
    end
    AU->>S: resolve and evaluate the ACL
    T-->>D: tool_json / tool_text / tool_error<br/>solid-pod-rs-server/src/mcp/mod.rs:70
    D-->>H: rpc_result or rpc_error<br/>solid-pod-rs-server/src/mcp/mod.rs:56
    H-->>A: JSON-RPC response

    Note over H: PROTOCOL_VERSION 2025-03-26<br/>solid-pod-rs-server/src/mcp/mod.rs:25
    Note over T: The endpoint is off by default (SP-02.3) and that default is the documented<br/>mitigation: the security model still says keep MCP disabled<br/>(../solid-pod-rs/crates/solid-pod-rs/docs/explanation/security-model.md:108).
```

**Drift:** the security model's warning (`../solid-pod-rs/crates/solid-pod-rs/docs/explanation/security-model.md:108`, AUD-009) describes an anonymous read path and sidecar operations without Control. At the pinned revision the source has the fixes that audit asked for: resource and skill reads go through `authorise` (`solid-pod-rs-server/src/mcp/tools.rs:144`), and the metadata gate elevates sidecars to Control (`solid-pod-rs-server/src/mcp/tools.rs:111`). An earlier reading attributed the warning to the README, which no longer mentions MCP.

**Open:** whether AUD-009 is closed needs the audit's own regression tests (anonymous read, Write without Control) run against this revision. This trace shows the gate is wired, not that every tool reaches it.

## SP-08.15 The MCP tool catalogue

```mermaid
flowchart LR
    subgraph RES["Resource tools"]
        LR2["list_resources<br/>solid-pod-rs-server/src/mcp/tools.rs:185"]
        RR["read_resource<br/>solid-pod-rs-server/src/mcp/tools.rs:367"]
        WR["write_resource<br/>solid-pod-rs-server/src/mcp/tools.rs:400"]
        CR["create_resource<br/>solid-pod-rs-server/src/mcp/tools.rs:439"]
        DR["delete_resource<br/>solid-pod-rs-server/src/mcp/tools.rs:520"]
        HR["head_resource<br/>solid-pod-rs-server/src/mcp/tools.rs:548"]
    end
    subgraph ACL["ACL tools"]
        RA["read_acl<br/>solid-pod-rs-server/src/mcp/tools.rs:743"]
        WA["write_acl with its own lockout guard<br/>solid-pod-rs-server/src/mcp/tools.rs:851"]
        BA["build_acl_jsonld<br/>solid-pod-rs-server/src/mcp/tools.rs:792"]
    end
    subgraph SK["Skill and doc tools"]
        LS["list_skills<br/>solid-pod-rs-server/src/mcp/tools.rs:576"]
        GS["get_skill<br/>solid-pod-rs-server/src/mcp/tools.rs:596"]
        PS["get_pod_skill<br/>solid-pod-rs-server/src/mcp/tools.rs:624"]
        LD["list_docs / read_docs from the embedded DOCS_DIR<br/>solid-pod-rs-server/src/mcp/tools.rs:660"]
        PI["pod_info<br/>solid-pod-rs-server/src/mcp/tools.rs:692"]
    end
    subgraph JV["Jev tool, feature jev only"]
        PO["propose_operation — proposes, never executes<br/>solid-pod-rs-server/src/mcp/tools.rs:289"]
        JR["jev_rerank of list_resources<br/>solid-pod-rs-server/src/mcp/tools.rs:262"]
    end
    subgraph FED["Federation tool"]
        CRP["call_remote_pod, SSRF-checked target<br/>solid-pod-rs-server/src/mcp/tools.rs:974<br/>solid-pod-rs-server/src/mcp/tools.rs:1002"]
        FG["federation_gate_path<br/>solid-pod-rs-server/src/mcp/tools.rs:949"]
        FA["is_forwardable_auth_header allowlist<br/>solid-pod-rs-server/src/mcp/tools.rs:47"]
    end
    SUB["handle_subscribe — SSE change stream<br/>solid-pod-rs-server/src/mcp/tools.rs:1269"]

    RES --> SUB
    ACL --> SUB
    FED --> FG
    FED --> FA

    N["INVARIANT: call_remote_pod forwards only an ALLOWLISTED set of auth headers<br/>(case-insensitively) so an agent cannot smuggle an arbitrary credential through<br/>the pod — pinned by two tests<br/>(solid-pod-rs-server/src/mcp/tools.rs:1364 and :1373)."]
    FA -.-> N
    N2["path_matches_scope (solid-pod-rs-server/src/mcp/tools.rs:1245) bounds an SSE<br/>subscription to the scope the agent asked for, and each event is WAC-checked<br/>before it is framed by sse_event (:1256)."]
    SUB -.-> N2
    N3["Skills are discovered from CONVENTIONAL pod paths<br/>(solid-pod-rs-server/src/mcp/skills.rs:18, with :19 and :20 for apps and bots), so a pod publishes agent<br/>instructions as ordinary WAC-governed resources rather than as server config.<br/>discover_skills (:109) and read_skill (:152) are the two entry points."]
    SK -.-> N3
    N4["propose_operation is listed and dispatched only when the jev feature is built<br/>(solid-pod-rs-server/Cargo.toml:184) AND the process env enables it<br/>(solid-pod-rs-server/src/mcp/tools.rs:1147). It checks WAC read on the container,<br/>drops unreadable children before any Jev call, and proposes an LDP operation<br/>without executing it. jev_rerank only ever sees children the caller may read."]
    JV -.-> N4
```

## SP-08.18 The forge has outgrown the request-dispatch picture

```mermaid
flowchart TD
    R["Route enum — the first reading plus the routes below<br/>solid-pod-rs-forge/src/router.rs:12"]
    subgraph ALWAYS["Compiled without a feature"]
        FK["ApiFork<br/>solid-pod-rs-forge/src/router.rs:193"]
        CP["ApiCompare<br/>solid-pod-rs-forge/src/router.rs:200"]
        PL["ApiPulls<br/>solid-pod-rs-forge/src/router.rs:209"]
        XL["Xlogin — vendored login widget<br/>solid-pod-rs-forge/src/router.rs:237"]
        ED["ApiEdit — commit one file over HTTP<br/>solid-pod-rs-forge/src/router.rs:279"]
        PV["ApiPreview<br/>solid-pod-rs-forge/src/router.rs:287"]
        RS["RepoSearch<br/>solid-pod-rs-forge/src/router.rs:255"]
    end
    subgraph ANN["Feature announce"]
        NO["ApiNostr — kind 30617 and 30618 for a repo<br/>solid-pod-rs-forge/src/router.rs:169"]
        AN["ApiAnnounce — owner-triggered relay publish<br/>solid-pod-rs-forge/src/router.rs:177"]
        PUB["publish_event, ws and wss relays only<br/>solid-pod-rs-forge/src/publish.rs:298"]
    end
    subgraph MIR["Feature mirror, inert unless configured"]
        AM["ApiMirror<br/>solid-pod-rs-forge/src/router.rs:150"]
        PM["plan_mirror — fetch only when the followed key's state changes<br/>solid-pod-rs-forge/src/mirror.rs:633"]
    end
    subgraph ANC["Feature anchoring"]
        BT["BlocktrailsJson<br/>solid-pod-rs-forge/src/router.rs:101"]
        TR["trail module<br/>solid-pod-rs-forge/src/lib.rs:90"]
    end

    R --> ALWAYS
    R --> ANN
    R --> MIR
    R --> ANC
    NO --> PUB
    AN --> PUB
    AM --> PM

    N["INVARIANT: with open_edit false (the default) an edit needs an owner credential;<br/>with it true no authorisation check runs at all and any caller may edit any repo,<br/>so the service logs a warning when it is on<br/>(default false at solid-pod-rs-forge/src/config.rs:131, warning at solid-pod-rs-forge/src/lib.rs:299, gate at solid-pod-rs-forge/src/web_edit.rs:226)."]
    ED -.-> N
    N2["The module gates are in solid-pod-rs-forge/src/lib.rs:66 to :74 and the feature<br/>declarations in solid-pod-rs-forge/Cargo.toml:57, :61 and :65. Route variants<br/>always parse; the feature decides what the handler does. SP-08.10 to SP-08.13<br/>describe the dispatch, ownership, token and browse paths; they do not enumerate<br/>every route."]
    R -.-> N2
```

**What it shows.** The forge's route table includes fork, compare, pull-request, search, web-edit and preview routes that compile unconditionally, plus NIP-34 announcement (feature `announce`), relay mirroring (feature `mirror`) and repo-held blocktrails (feature `anchoring`). Since SP-GIT-02 the `migrate` module (`solid-pod-rs-forge/src/lib.rs:69`) and the reader gate are compiled without the `mirror` feature, so a plugin directory that was mirrored once stays gated in any build (SP-14.4).

**Why it is this way.** Each optional tier is a feature gate so a plain forge adds neither relay sockets nor the Bitcoin stack. The README states announcement publication is opt-in: with no `announce_relays` nothing is sent (`../solid-pod-rs/crates/solid-pod-rs-forge/README.md:24-33`).

**Drift (ADR-2009 vs code):** ADR-2009 says the mirror is default off and "nothing in `solid-pod-rs-server` starts it" (`../solid-pod-rs/crates/solid-pod-rs/docs/adr/ADR-2009-forge-plugins-comparator-and-default-off-mirror.md:34`); the server calls `start_forge` at boot when built with `forge` (`../solid-pod-rs/crates/solid-pod-rs-server/src/main.rs:459-460`) and reads `FORGE_MIRROR_RELAYS` (`../solid-pod-rs/crates/solid-pod-rs-server/src/lib.rs:5328`). ADR-2010 supersedes the default-off clause and defines "on" as compiled in and started at boot, acting on nothing until relays and follow entries are configured (`../solid-pod-rs/crates/solid-pod-rs/docs/adr/ADR-2010-forge-and-mirror-are-on-in-shipped-builds.md:41`).

**Drift (ADR text vs the SP-GIT-02 repair):** both ADRs gained a review qualification dated 2026-11-01 (`../solid-pod-rs/crates/solid-pod-rs/docs/adr/ADR-2009-forge-plugins-comparator-and-default-off-mirror.md:71`, `../solid-pod-rs/crates/solid-pod-rs/docs/adr/ADR-2010-forge-and-mirror-are-on-in-shipped-builds.md:72`) that still treats hidden refs as the mechanism and complete object withholding as open (`../solid-pod-rs/crates/solid-pod-rs/docs/adr/ADR-2010-forge-and-mirror-are-on-in-shipped-builds.md:79`). At this revision the boundary is store isolation plus the reader gate (SP-14.3, SP-14.4), and neither ADR says so.

**Tension (ADR-2010 vs agentbox):** ADR-2010 reads accepted / complete / staged (`../solid-pod-rs/crates/solid-pod-rs/docs/adr/ADR-2010-forge-and-mirror-are-on-in-shipped-builds.md:5-7`), and agentbox's pin enables `forge-full` (`../project/agentbox/lib/solid-pod-rs.nix:18`). That pin is `76ab7aab` (`../project/agentbox/lib/solid-pod-rs.nix:58`), which predates both the hidden-tip repair and the SP-GIT-02 store isolation. A shipped agentbox build therefore carries the pre-repair mirror; it stays inert only while no follow entries are configured. ADR-2009 moved the same way, accepted / complete / staged (`../solid-pod-rs/crates/solid-pod-rs/docs/adr/ADR-2009-forge-plugins-comparator-and-default-off-mirror.md:5-7`).

**Open:** the forge README (`../solid-pod-rs/crates/solid-pod-rs-forge/README.md:33`) records the web-edit tier as "not included" for announce triggering, while `ApiEdit` is routed (`solid-pod-rs-forge/src/router.rs:279`); nothing states whether an edit announces.

## SP-08.16 Estate consumers of these surfaces

```mermaid
flowchart TD
    POD["solid-pod-rs — this repo"]
    VC["EXTERNAL: VisionClaw embeds the pod crates in its actix runtime<br/>(src/handlers/solid_proxy_handler.rs) and uses LDP, WAC, NIP-98,<br/>WebID and did:nostr resolution. See VC-26 and ES-08.4."]
    AB["EXTERNAL: agentbox forwards agent state through a pod-bridge<br/>adapter and can emit Solid Notifications 0.2 events. See AB-13 and ES-08."]
    NF["EXTERNAL: nostr-rust-forum's nostr-bbs-pod-worker compiles the<br/>core feature to wasm32 on Cloudflare Workers for pod-backed<br/>thread storage. See the nostr-rust-forum area."]
    DW["EXTERNAL: dreamlab-ai-website consumes the pod TRANSITIVELY through<br/>the forum kit and has no direct dependency. See the<br/>dreamlab-ai-website area."]

    POD --> VC
    POD --> AB
    POD --> NF
    NF --> DW

    N["The boundary contract is the crate's public API surface, not its internals:<br/>Storage, ldp helpers, wac::evaluate_access* and AclResolver,<br/>auth::nip98::verify / verify_at, oidc::*, the notification channel managers,<br/>interop::did_nostr::* and PodError. Anything else may change between minor<br/>versions."]
    POD -.-> N
    N2["Three integration patterns: full server embedding (VisionClaw, agentbox),<br/>core-only wasm embedding (the forum worker) and transitive consumption<br/>(the website). Pin matrix in SP-09.7."]
    POD -.-> N2
```

## SP-08.17 The legacy `solid-0.1` driver — the crate mounts nothing

```mermaid
sequenceDiagram
    autonumber
    participant T as the CONSUMER's transport<br/>actix-ws, axum ws, tungstenite
    participant D as LegacyWsDriver<br/>solid-pod-rs/src/handlers/legacy_notifications.rs:91
    participant R as run_loop<br/>solid-pod-rs/src/handlers/legacy_notifications.rs:167
    participant H as handle_line<br/>solid-pod-rs/src/handlers/legacy_notifications.rs:241
    participant CH as LegacyNotificationChannel — see SP-08.4

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
    Note over CH: This is the DRIVER half — notifications/legacy.rs is the protocol half<br/>(SP-08.4). Neither is reachable without the default-off legacy-notifications<br/>feature — see SP-01.6.
```

**Open:** `matches_subscription` compares the stored subscription string with the storage event's path as-is (`solid-pod-rs/src/notifications/legacy.rs:393`, called at `solid-pod-rs/src/handlers/legacy_notifications.rs:211`). A client subscribes with an absolute URL, and the origin check expects one; whether a `pub` line is ever emitted therefore depends on the form of path the consumer's storage events carry. The driver does not normalise either side. Only a running consumer can show which.
