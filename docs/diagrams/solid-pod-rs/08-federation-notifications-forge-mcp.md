---
id: SP-08
title: Federation and outward surfaces — ActivityPub, NIP-01 relay, Solid Notifications, the forge, MCP
area: solid-pod-rs
governing: [../solid-pod-rs/README.md, ../solid-pod-rs/crates/solid-pod-rs/docs/explanation/ecosystem-integration.md]
adrs: [ADR-2005, ADR-2006]
sources:
  - ../solid-pod-rs/crates/solid-pod-rs-activitypub/src/actor.rs
  - ../solid-pod-rs/crates/solid-pod-rs-activitypub/src/inbox.rs
  - ../solid-pod-rs/crates/solid-pod-rs-activitypub/src/outbox.rs
  - ../solid-pod-rs/crates/solid-pod-rs-activitypub/src/delivery.rs
  - ../solid-pod-rs/crates/solid-pod-rs-activitypub/src/http_sig.rs
  - ../solid-pod-rs/crates/solid-pod-rs-activitypub/src/store.rs
  - ../solid-pod-rs/crates/solid-pod-rs-activitypub/src/ssrf.rs
  - ../solid-pod-rs/crates/solid-pod-rs-activitypub/src/discovery.rs
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
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/mcp/mod.rs
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/mcp/tools.rs
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/mcp/skills.rs
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/lib.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/handlers/legacy_notifications.rs
verified_commit: 1d9da5270
---

## SP-08.1 Solid Notifications 0.2 — the three channels

```mermaid
flowchart TD
    SE["StorageEvent from the backend watcher — see SP-06.5"]
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

    N["InMemoryNotifications caps subscriptions at DEFAULT_MAX_SUBSCRIPTIONS 10 000<br/>(solid-pod-rs/src/notifications/mod.rs:125) — an unbounded subscription table is<br/>a memory-exhaustion vector on a public pod."]
    SUB -.-> N
    N2["The Updates-via response header advertises the WebSocket endpoint on every LDP<br/>response (solid-pod-rs-server/src/lib.rs:1141) — see SP-03.11."]
    WS -.-> N2
```

## SP-08.2 WebSocket notification pump

```mermaid
sequenceDiagram
    autonumber
    participant C as subscriber
    participant M as WebSocketChannelManager<br/>solid-pod-rs/src/notifications/mod.rs:243
    participant P as pump_from_storage<br/>solid-pod-rs/src/notifications/mod.rs:316
    participant S as Storage::watch
    participant B as broadcast channel

    C->>M: subscribe(topic, base_url)<br/>solid-pod-rs/src/notifications/mod.rs:278
    M-->>C: Subscription id
    C->>M: stream()<br/>solid-pod-rs/src/notifications/mod.rs:303
    M-->>C: broadcast::Receiver
    S->>P: StorageEvent
    P->>B: ChangeNotification
    B-->>C: delivered to every live receiver
    C->>M: unsubscribe(id)<br/>solid-pod-rs/src/notifications/mod.rs:296
    Note over M: with_heartbeat sets the keepalive interval<br/>solid-pod-rs/src/notifications/mod.rs:266
    Note over B: A broadcast channel drops messages for a slow receiver rather than growing<br/>without bound — a stalled subscriber degrades its own stream, not the pod.
```

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
      (solid-pod-rs/src/notifications/mod.rs:456) and jittered
      (solid-pod-rs/src/notifications/mod.rs:823), so a fleet of pods retrying the
      same dead endpoint does not synchronise into a thundering herd.
    end note
```

## SP-08.4 The legacy `solid-0.1` WebSocket adapter

```mermaid
sequenceDiagram
    autonumber
    participant C as SolidOS data browser
    participant S as LegacyWebSocketSession<br/>solid-pod-rs/src/notifications/legacy.rs:729
    participant W as LegacyWacRead<br/>solid-pod-rs/src/notifications/legacy.rs:666
    participant CH as LegacyNotificationChannel<br/>solid-pod-rs/src/notifications/legacy.rs:221

    C->>S: connect
    S-->>C: PROTOCOL_GREETING "protocol solid-0.1"<br/>solid-pod-rs/src/notifications/legacy.rs:88
    C->>S: sub <resource-uri>
    S->>S: parse_subscribe<br/>solid-pod-rs/src/notifications/legacy.rs:447
    S->>W: WAC read check for the subscriber's WebID
    alt denied
        W-->>S: DenyReason<br/>solid-pod-rs/src/notifications/legacy.rs:131
        S-->>C: err line<br/>solid-pod-rs/src/notifications/legacy.rs:462
    else granted
        S-->>C: ack line<br/>solid-pod-rs/src/notifications/legacy.rs:457
    end
    CH->>S: StorageEvent
    S->>S: matches_subscription?<br/>solid-pod-rs/src/notifications/legacy.rs:393
    S-->>C: pub <uri> via to_legacy_line<br/>solid-pod-rs/src/notifications/legacy.rs:436

    Note over S: INVARIANT: a subscription is WAC-checked. Without SubscriptionAuthorizer<br/>(solid-pod-rs/src/notifications/legacy.rs:152) a subscriber would learn that a<br/>private resource CHANGED even when it may not read it — a metadata leak.<br/>AllowAllAuthorizer (:164) and DenyAllAuthorizer (:178) are the two poles.
    Note over S: Bounds: MAX_SUBSCRIPTIONS_PER_CONNECTION 100<br/>(solid-pod-rs/src/notifications/legacy.rs:75) and MAX_URL_LENGTH 2048<br/>(solid-pod-rs/src/notifications/legacy.rs:78).
```

## SP-08.5 ActivityPub — actor and inbox

```mermaid
sequenceDiagram
    autonumber
    participant R as remote instance
    participant I as handle_inbox<br/>solid-pod-rs-activitypub/src/inbox.rs:50
    participant V as verify_request_signature<br/>solid-pod-rs-activitypub/src/http_sig.rs:375
    participant K as HttpActorKeyResolver<br/>solid-pod-rs-activitypub/src/http_sig.rs:96
    participant SS as SSRF guard<br/>solid-pod-rs-activitypub/src/ssrf.rs:47
    participant S as Store<br/>solid-pod-rs-activitypub/src/store.rs:30

    R->>I: POST an activity to the inbox
    I->>V: verify the HTTP Signature
    V->>V: parse_signature_header<br/>solid-pod-rs-activitypub/src/http_sig.rs:181
    V->>V: validate_covered_components<br/>solid-pod-rs-activitypub/src/http_sig.rs:303
    V->>V: check_date_freshness — the replay window<br/>solid-pod-rs-activitypub/src/http_sig.rs:339
    V->>V: digest_header must match the body<br/>solid-pod-rs-activitypub/src/http_sig.rs:325
    V->>K: resolve the actor's public key
    K->>SS: resolve_ssrf_safe before fetching<br/>solid-pod-rs-activitypub/src/ssrf.rs:66
    K-->>V: VerifiedActor<br/>solid-pod-rs-activitypub/src/http_sig.rs:76
    V-->>I: verified, or SigError<br/>solid-pod-rs-activitypub/src/error.rs:12
    I->>S: record_inbox — dedup by activity id<br/>solid-pod-rs-activitypub/src/store.rs:249
    alt the activity is a Follow
        I->>I: build_accept<br/>solid-pod-rs-activitypub/src/inbox.rs:158
        I->>S: add_follower<br/>solid-pod-rs-activitypub/src/store.rs:133
    end
    I-->>R: InboxOutcome<br/>solid-pod-rs-activitypub/src/inbox.rs:21

    Note over V: INVARIANT: signature verification runs BEFORE anything is stored, and the<br/>covered-component list is validated rather than trusted — accepting whichever<br/>headers the sender chose to sign is the classic HTTP-Signature bypass.
    Note over K: The actor-key fetch is SSRF-guarded because the key URL comes from the<br/>UNTRUSTED signature header. is_private_ip<br/>(solid-pod-rs-activitypub/src/ssrf.rs:33) is the classifier.
```

## SP-08.6 ActivityPub — outbox and delivery

```mermaid
sequenceDiagram
    autonumber
    participant L as local actor
    participant O as handle_outbox_post<br/>solid-pod-rs-activitypub/src/outbox.rs:143
    participant W as wrap_note_in_create<br/>solid-pod-rs-activitypub/src/outbox.rs:100
    participant S as Store
    participant Q as enqueue_to_inboxes<br/>solid-pod-rs-activitypub/src/delivery.rs:232
    participant D as DeliveryWorker::drain_once<br/>solid-pod-rs-activitypub/src/delivery.rs:105
    participant R as remote inbox

    L->>O: POST an object to the outbox
    O->>W: a bare Note is wrapped in a Create
    O->>S: record_outbox<br/>solid-pod-rs-activitypub/src/store.rs:299
    O->>S: follower_inboxes for fan-out<br/>solid-pod-rs-activitypub/src/store.rs:180
    O->>Q: one queue row per recipient inbox
    Q->>S: enqueue_delivery<br/>solid-pod-rs-activitypub/src/store.rs:341
    loop DeliveryWorker::run tick<br/>solid-pod-rs-activitypub/src/delivery.rs:248
        D->>S: next_due_delivery<br/>solid-pod-rs-activitypub/src/store.rs:358
        D->>D: sign_request<br/>solid-pod-rs-activitypub/src/http_sig.rs:436
        D->>R: POST the activity
        alt delivered
            D->>S: drop_delivery<br/>solid-pod-rs-activitypub/src/store.rs:377
            D->>S: mark_outbox_state<br/>solid-pod-rs-activitypub/src/store.rs:323
        else failed
            D->>S: reschedule_delivery with backoff<br/>solid-pod-rs-activitypub/src/store.rs:385
        end
    end
    Note over D: DeliveryOutcome (solid-pod-rs-activitypub/src/delivery.rs:34) and<br/>DeliveryConfig (:49) make the retry policy explicit rather than implicit in a<br/>loop. Delivery is also SSRF-guarded — a follower's inbox URL is remote input.
```

## SP-08.7 ActivityPub — actor documents, caching and NodeInfo

```mermaid
classDiagram
    class Actor {
        solid-pod-rs-activitypub/src/actor.rs:43
        +PublicKey  solid-pod-rs-activitypub/src/actor.rs:21
        +Endpoints  solid-pod-rs-activitypub/src/actor.rs:32
    }
    class Render {
        +generate_actor_keypair  solid-pod-rs-activitypub/src/actor.rs:74
        +render_actor  solid-pod-rs-activitypub/src/actor.rs:97
        +negotiate_actor_format  solid-pod-rs-activitypub/src/actor.rs:157
        +with_also_known_as  solid-pod-rs-activitypub/src/actor.rs:182
    }
    class ActorCache {
        +cache_actor  solid-pod-rs-activitypub/src/store.rs:414
        +get_cached_actor  solid-pod-rs-activitypub/src/store.rs:431
        +is_actor_cache_fresh  solid-pod-rs-activitypub/src/store.rs:454
    }
    class NodeInfo {
        +nodeinfo_wellknown  solid-pod-rs-activitypub/src/discovery.rs:17
        +nodeinfo_2_1  solid-pod-rs-activitypub/src/discovery.rs:33
    }
    Actor <.. Render
    Actor <.. ActorCache
    Render ..> NodeInfo
    note for Render "with_also_known_as is the bridge that lets one ActivityPub actor declare the\nsame subject as a WebID and a did:nostr — the estate's one-identity claim\nreaches the fediverse through this field. See SP-05.12 and ES-04.\nActorFormat (solid-pod-rs-activitypub/src/actor.rs:139) drives conneg between\nthe AS2 and JSON-LD representations."
```

## SP-08.8 The embedded NIP-01 relay

```mermaid
flowchart TD
    WS["serve_relay_ws / serve_relay_ws_stream<br/>solid-pod-rs-nostr/src/ws.rs:136"]
    LIM["RelayLimits<br/>solid-pod-rs-nostr/src/ws.rs:32"]
    DISP["dispatch_message_with_limits<br/>solid-pod-rs-nostr/src/ws.rs:164"]
    EV["handle_event -> Relay::ingest<br/>solid-pod-rs-nostr/src/ws.rs:194"]
    REQ["handle_req -> Relay::history<br/>solid-pod-rs-nostr/src/ws.rs:233"]
    CLS["handle_close<br/>solid-pod-rs-nostr/src/ws.rs:290"]
    ING["Relay::ingest<br/>solid-pod-rs-nostr/src/relay.rs:416"]
    VER["Event::verify — Schnorr plus canonical id<br/>solid-pod-rs-nostr/src/relay.rs:63"]
    STORE["EventStore trait, InMemoryEventStore<br/>solid-pod-rs-nostr/src/relay.rs:225"]
    BC["Relay::broadcast to live subscribers<br/>solid-pod-rs-nostr/src/relay.rs:391"]
    FIL["Filter::matches<br/>solid-pod-rs-nostr/src/relay.rs:150"]
    INFO["RelayInfo::jss_compatible — NIP-11<br/>solid-pod-rs-nostr/src/relay.rs:319"]

    WS --> LIM --> DISP
    DISP --> EV --> ING --> VER
    ING --> STORE
    ING --> BC
    DISP --> REQ --> FIL
    DISP --> CLS
    INFO --> WS

    N["Kind classes decide retention: is_replaceable<br/>(solid-pod-rs-nostr/src/relay.rs:206), is_ephemeral (:210) and<br/>is_parameterised_replaceable (:214) — an ephemeral event is broadcast but never<br/>stored, so the store cannot be filled with transient traffic."]
    STORE -.-> N
    N2["canonical_id (solid-pod-rs-nostr/src/relay.rs:48) recomputes the NIP-01 id from<br/>content — the same never-trust-the-claimed-id rule NIP-98 applies in SP-05.2."]
    VER -.-> N2
    N3["ok_frame and notice (solid-pod-rs-nostr/src/ws.rs:298) are the two NIP-01 reply<br/>shapes; InMemoryEventStore is capped at construction<br/>(solid-pod-rs-nostr/src/relay.rs:251)."]
    DISP -.-> N3
```

## SP-08.9 The relay typestate — verification cannot be skipped

```mermaid
stateDiagram-v2
    [*] --> Unchecked: UncheckedEvent.new<br/>solid-pod-rs-nostr/src/typestate.rs:138
    Unchecked --> Verified: verify()<br/>solid-pod-rs-nostr/src/typestate.rs:187
    Unchecked --> Rejected: VerifyError<br/>solid-pod-rs-nostr/src/typestate.rs:80
    Unchecked --> Escaped: into_inner_unchecked — the ONE audited escape<br/>solid-pod-rs-nostr/src/typestate.rs:197
    Verified --> Ingested: ingest_verified<br/>solid-pod-rs-nostr/src/typestate.rs:339
    Rejected --> [*]
    Ingested --> [*]

    note right of Verified
      VerifiedEvent (solid-pod-rs-nostr/src/typestate.rs:235) is the only type
      ingest_verified accepts, so forgot to verify becomes a COMPILE error
      rather than a runtime hole. Accessors like d_tag (:281) and get_tag (:287)
      hang off the verified type.
    end note
    note right of Escaped
      into_inner_unchecked is deliberately named so an audit can grep for every
      place verification was bypassed — the escape hatch is visible, not silent.
    end note
```

## SP-08.10 The forge — request dispatch

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant SRV as handle_forge<br/>solid-pod-rs-server/src/lib.rs:4466
    participant F as ForgeService::handle<br/>solid-pod-rs-forge/src/lib.rs:219
    participant A as resolve_agent<br/>solid-pod-rs-forge/src/auth.rs:29
    participant R as parse_route<br/>solid-pod-rs-forge/src/router.rs:172
    participant H as the matching h_* handler<br/>solid-pod-rs-forge/src/lib.rs:261

    C->>SRV: any method under /forge
    SRV->>F: ForgeRequest<br/>solid-pod-rs-forge/src/request.rs:23
    F->>A: three auth schemes, in order
    A->>A: 1. forge push token, Bearer f1.*<br/>solid-pod-rs-forge/src/auth.rs:42
    A->>A: 2. pod session, injected by the server<br/>solid-pod-rs-forge/src/auth.rs:50
    A->>A: 3. NIP-98, body-bound when a body exists<br/>solid-pod-rs-forge/src/auth.rs:58
    A-->>F: ForgeAgent<br/>solid-pod-rs-forge/src/ownership.rs:16
    F->>R: strip_prefix then parse<br/>solid-pod-rs-forge/src/router.rs:151
    R-->>F: Route<br/>solid-pod-rs-forge/src/router.rs:12
    F->>H: dispatch
    H-->>C: ForgeResponse<br/>solid-pod-rs-forge/src/request.rs:85

    Note over A: An unrecognised or invalid credential resolves to ForgeAgent::Anonymous<br/>rather than an error (solid-pod-rs-forge/src/auth.rs:36) — the route handlers<br/>then apply their own ownership checks, fail-closed.
    Note over R: GitSmart routes forward the prefix-stripped path VERBATIM to the git CGI, so<br/>the forge reuses the same smart-HTTP service as a pod repo — see SP-06.12.
```

## SP-08.11 The forge push token

```mermaid
flowchart LR
    KEY["load_or_create_token_key — 32 bytes under the plugin dir<br/>solid-pod-rs-forge/src/lib.rs:97"]
    MINT["token::mint(key, agent, iat, ttl)<br/>solid-pod-rs-forge/src/token.rs:66"]
    VER["token::verify(key, token, now)<br/>solid-pod-rs-forge/src/token.rs:87"]
    MAC["mac_tag — HMAC over the payload<br/>solid-pod-rs-forge/src/token.rs:126"]
    V1["TOKEN_VERSION 1, the f1. prefix<br/>solid-pod-rs-forge/src/token.rs:25"]
    ERR["TokenError<br/>solid-pod-rs-forge/src/token.rs:45"]
    API["POST /api/token — Route::ApiToken<br/>solid-pod-rs-forge/src/lib.rs:303"]

    KEY --> MINT --> MAC
    V1 --> MINT
    API --> MINT
    MAC --> VER --> ERR

    N["DIVERGENCE: this is a bespoke HMAC token format rather than an established one.<br/>It exists for the Tier-2.5 podless did:nostr identity — a git client that cannot<br/>mint a NIP-98 event per request holds a short-lived bearer instead. The version<br/>prefix at least makes a future format migration detectable."]
    MINT -.-> N
    N2["The key is generated on first use and persisted under the plugin dir, so it<br/>survives a restart but is not shared across deployments."]
    KEY -.-> N2
```

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
        +strip_git_suffix  solid-pod-rs-forge/src/router.rs:299
    }
    class ForgeConfig {
        +normalized_prefix  solid-pod-rs-forge/src/config.rs:61
        +chain_allowed  solid-pod-rs-forge/src/config.rs:73
    }
    ForgeAgent ..> OwnerKind
    OwnerKind ..> Names
    ForgeConfig ..> ForgeAgent
    note for ForgeAgent "can_write_namespace is the single authorisation predicate for every mutating\nforge route — a namespace is owned by exactly one agent, so the forge does not\nneed a full ACL evaluation of its own."
```

## SP-08.13 Forge repository browsing and the issue spine

```mermaid
flowchart TD
    BR["repo::browse — git plumbing<br/>solid-pod-rs-forge/src/repo/browse.rs:50"]
    VR["valid_rev<br/>solid-pod-rs-forge/src/repo/browse.rs:104"]
    VP["valid_repo_path<br/>solid-pod-rs-forge/src/repo/browse.rs:118"]
    LT["list_tree -> parse_ls_tree<br/>solid-pod-rs-forge/src/repo/browse.rs:161"]
    RB["read_blob<br/>solid-pod-rs-forge/src/repo/browse.rs:222"]
    CL["commit_log / commit_patch<br/>solid-pod-rs-forge/src/repo/browse.rs:258"]
    SP["SpineStore trait, FsSpineStore<br/>solid-pod-rs-forge/src/spine/mod.rs:24"]
    AW["spine atomic_write<br/>solid-pod-rs-forge/src/spine/mod.rs:115"]
    IX["IssueIndex allocate / by_state / set_state<br/>solid-pod-rs-forge/src/spine/issues.rs:101"]
    LD["load_issue_index / save_issue_index<br/>solid-pod-rs-forge/src/spine/issues.rs:137"]
    TP["ThreadPointer<br/>solid-pod-rs-forge/src/spine/issues.rs:44"]
    RT["render_thread — resolve bodies from pods or the hosted store<br/>solid-pod-rs-forge/src/bodies.rs:165"]
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

    N["INVARIANT: valid_rev and valid_repo_path constrain every caller-supplied ref<br/>and path before it becomes a git argv element — the forge shells out, so an<br/>unvalidated rev is an argument-injection vector."]
    VR -.-> N
    N2["An issue THREAD is a pointer into a pod (or the hosted store), not a copy — the<br/>forge stores the index and the pod stores the content, so the author keeps<br/>custody of their own words. BodyOutcome (solid-pod-rs-forge/src/bodies.rs:105)<br/>is how an unresolvable body renders without breaking the page."]
    TP -.-> N2
    N3["esc (solid-pod-rs-forge/src/request.rs:181), sanitize_filename (:200) and<br/>looks_textual (:217) are the XSS-safe content-type spine the CHANGELOG calls<br/>Phase 0."]
    VIEW -.-> N3
```

## SP-08.14 The pod as an MCP tool surface

```mermaid
sequenceDiagram
    autonumber
    participant A as agent
    participant H as handle_mcp<br/>solid-pod-rs-server/src/mcp/mod.rs:160
    participant D as dispatch<br/>solid-pod-rs-server/src/mcp/mod.rs:97
    participant T as call_tool<br/>solid-pod-rs-server/src/mcp/tools.rs:828
    participant W as wac_check<br/>solid-pod-rs-server/src/mcp/tools.rs:83
    participant S as Storage

    A->>H: POST /mcp with a JSON-RPC message
    H->>H: origin_of builds the McpCtx<br/>solid-pod-rs-server/src/mcp/mod.rs:154
    H->>D: is_allowed_method gate<br/>solid-pod-rs-server/src/mcp/mod.rs:83
    D->>T: tools/call
    T->>W: WAC check per tool, per path
    W->>S: resolve and evaluate the ACL
    W-->>T: allow or deny
    T-->>D: tool_json / tool_text / tool_error<br/>solid-pod-rs-server/src/mcp/mod.rs:70
    D-->>H: rpc_result or rpc_error<br/>solid-pod-rs-server/src/mcp/mod.rs:56
    H-->>A: JSON-RPC response

    Note over H: PROTOCOL_VERSION 2025-03-26<br/>solid-pod-rs-server/src/mcp/mod.rs:25
    Note over T: DIVERGENCE: the README's status section records "anonymous MCP reads / WAC<br/>sidecar bypass" as a reproduced CRITICAL finding and says to keep MCP disabled.<br/>The endpoint is off by default (SP-02.3) — that default is the mitigation.
```

## SP-08.15 The MCP tool catalogue

```mermaid
flowchart LR
    subgraph RES["Resource tools"]
        LR2["list_resources<br/>solid-pod-rs-server/src/mcp/tools.rs:139"]
        RR["read_resource<br/>solid-pod-rs-server/src/mcp/tools.rs:175"]
        WR["write_resource<br/>solid-pod-rs-server/src/mcp/tools.rs:204"]
        CR["create_resource<br/>solid-pod-rs-server/src/mcp/tools.rs:231"]
        DR["delete_resource<br/>solid-pod-rs-server/src/mcp/tools.rs:278"]
        HR["head_resource<br/>solid-pod-rs-server/src/mcp/tools.rs:295"]
    end
    subgraph ACL["ACL tools"]
        RA["read_acl<br/>solid-pod-rs-server/src/mcp/tools.rs:479"]
        WA["write_acl with its own lockout guard<br/>solid-pod-rs-server/src/mcp/tools.rs:583"]
        BA["build_acl_jsonld<br/>solid-pod-rs-server/src/mcp/tools.rs:524"]
    end
    subgraph SK["Skill and doc tools"]
        LS["list_skills<br/>solid-pod-rs-server/src/mcp/tools.rs:319"]
        GS["get_skill<br/>solid-pod-rs-server/src/mcp/tools.rs:339"]
        PS["get_pod_skill<br/>solid-pod-rs-server/src/mcp/tools.rs:361"]
        LD["list_docs / read_docs from the embedded DOCS_DIR<br/>solid-pod-rs-server/src/mcp/tools.rs:396"]
        PI["pod_info<br/>solid-pod-rs-server/src/mcp/tools.rs:428"]
    end
    subgraph FED["Federation tool"]
        CRP["call_remote_pod<br/>solid-pod-rs-server/src/mcp/tools.rs:677"]
        FG["federation_gate_path<br/>solid-pod-rs-server/src/mcp/tools.rs:669"]
        FA["is_forwardable_auth_header allowlist<br/>solid-pod-rs-server/src/mcp/tools.rs:45"]
    end
    SUB["handle_subscribe — SSE change stream<br/>solid-pod-rs-server/src/mcp/tools.rs:944"]

    RES --> SUB
    ACL --> SUB
    FED --> FG
    FED --> FA

    N["INVARIANT: call_remote_pod forwards only an ALLOWLISTED set of auth headers<br/>(case-insensitively) so an agent cannot smuggle an arbitrary credential through<br/>the pod — pinned by two tests<br/>(solid-pod-rs-server/src/mcp/tools.rs:1015 and :1024)."]
    FA -.-> N
    N2["path_matches_scope (solid-pod-rs-server/src/mcp/tools.rs:920) bounds an SSE<br/>subscription to the scope the agent asked for; sse_event (:931) frames it."]
    SUB -.-> N2
    N3["Skills are discovered from CONVENTIONAL pod paths<br/>(solid-pod-rs-server/src/mcp/skills.rs:24), so a pod publishes agent<br/>instructions as ordinary WAC-governed resources rather than as server config.<br/>discover_skills (:109) and read_skill (:152) are the two entry points."]
    SK -.-> N3
```

## SP-08.16 Estate consumers of these surfaces

```mermaid
flowchart TD
    POD["solid-pod-rs — this repo"]
    VC["EXTERNAL: VisionClaw embeds the pod in its actix runtime<br/>(solid_pod_handler) and uses LDP, WAC, NIP-98, WebID and<br/>did:nostr resolution. See VC-26."]
    AB["EXTERNAL: agentbox forwards agent state through a pod-bridge<br/>adapter and streams agent events over the Solid Notifications<br/>WebSocket channel. See AB-13 and ES-08."]
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
    H->>CH: sub / unsub, WAC-checked
    CH-->>H: ack, err or pub
    H-->>R: Vec of OutboundFrame<br/>solid-pod-rs/src/handlers/legacy_notifications.rs:78
    R-->>T: frames on an mpsc channel the transport writes

    Note over D: INVARIANT: this crate does NOT mount itself at /ws/solid-0.1<br/>(solid-pod-rs/src/handlers/legacy_notifications.rs:6). The consumer owns the<br/>upgrade and the route — the recommended path is only a recommendation<br/>(solid-pod-rs/src/handlers/legacy_notifications.rs:18). That is the same<br/>ownership line the ecosystem doc draws — this crate owns protocol, not routing.
    Note over R: The outbound mpsc is capacity-bounded (default 256,<br/>solid-pod-rs/src/handlers/legacy_notifications.rs:102) and overridable<br/>(:108). Lower caps drop frames earlier under back-pressure — lossy by design,<br/>matching JSS, so a slow client degrades its own stream and not the pod.
    Note over CH: This is the DRIVER half — notifications/legacy.rs is the protocol half<br/>(SP-08.4). Neither is reachable without the default-off legacy-notifications<br/>feature — see SP-01.6.
```
