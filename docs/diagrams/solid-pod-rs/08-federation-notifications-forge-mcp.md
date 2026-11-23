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
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/notifications_ws.rs
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/nostr_mount.rs
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/activitypub.rs
  - ../solid-pod-rs/crates/solid-pod-rs-forge/src/operator_relays.rs
verified_commit: {solid-pod-rs: a28a74832dd0a6e47a531e34e4adc46a3946dfd3, agentbox: 59b1aca1cc3541b767872ecbc161c89e65d6ebca}
---
Split 2026-11-23: SP-08.1 to SP-08.4, SP-08.17, SP-08.19 and SP-08.23 moved to [SP-16](16-solid-notifications-and-the-legacy-adapters.md) as SP-16.1 to SP-16.7; SP-08.10 to SP-08.15, SP-08.18 and SP-08.22 moved to [SP-17](17-forge-and-mcp-tool-surface.md) as SP-17.1 to SP-17.8. The old IDs resolve through the aliases declared there.

## SP-08.5 ActivityPub — signature verification and the inbox

```mermaid
sequenceDiagram
    autonumber
    participant R as remote instance
    participant E as embedder's HTTP layer
    participant V as verify_request_signature<br/>solid-pod-rs-activitypub/src/http_sig.rs:614
    participant K as HttpActorKeyResolver<br/>solid-pod-rs-activitypub/src/http_sig.rs:165
    participant SS as SsrfGuard<br/>solid-pod-rs-activitypub/src/ssrf.rs:43
    participant I as handle_inbox_with<br/>solid-pod-rs-activitypub/src/inbox.rs:129
    participant S as Store<br/>solid-pod-rs-activitypub/src/store.rs:30

    R->>E: POST an activity to the inbox
    E->>V: verify the HTTP Signature
    V->>V: parse_signature_header<br/>solid-pod-rs-activitypub/src/http_sig.rs:420
    V->>V: validate_covered_components<br/>solid-pod-rs-activitypub/src/http_sig.rs:542
    V->>V: check_date_freshness — the replay window<br/>solid-pod-rs-activitypub/src/http_sig.rs:578
    V->>V: digest_header must match the body<br/>solid-pod-rs-activitypub/src/http_sig.rs:564
    V->>K: resolve the actor's public key
    K->>SS: resolve_ssrf_safe_with before fetching, then a cache of keys<br/>solid-pod-rs-activitypub/src/ssrf.rs:126<br/>solid-pod-rs-activitypub/src/http_sig.rs:335
    K-->>V: VerifiedActor<br/>solid-pod-rs-activitypub/src/http_sig.rs:76
    V-->>E: verified, or SigError<br/>solid-pod-rs-activitypub/src/error.rs:12
    E->>I: activity plus the VerifiedActor
    I->>I: claimed actor must equal the verified actor<br/>solid-pod-rs-activitypub/src/inbox.rs:145
    I->>S: record_inbox_from — dedup by id, keyed to the sender<br/>solid-pod-rs-activitypub/src/store.rs:383
    alt the activity is a Follow
        I->>S: add_follower<br/>solid-pod-rs-activitypub/src/store.rs:233
        I->>I: build_accept<br/>solid-pod-rs-activitypub/src/inbox.rs:294
    end
    I-->>E: InboxOutcome<br/>solid-pod-rs-activitypub/src/inbox.rs:79

    Note over V: The covered-component list is validated rather than trusted: request-target,<br/>host and date always, digest for POST — accepting whichever headers the sender<br/>chose to sign is the classic HTTP-Signature bypass.
    Note over K: The actor-key fetch is SSRF-guarded because the key URL comes from the<br/>UNTRUSTED signature header. is_private_ip<br/>(solid-pod-rs-activitypub/src/ssrf.rs:182) is the classifier.
    Note over I: with_also_known_as (solid-pod-rs-activitypub/src/actor.rs:182) is the bridge<br/>that lets one ActivityPub actor declare the same subject as a WebID and a<br/>did:nostr — see SP-05.12 and ES-04. negotiate_actor_format<br/>(solid-pod-rs-activitypub/src/actor.rs:157) drives conneg between the AS2<br/>ActorFormat::ActivityJson and the Solid/LDP profile (solid-pod-rs-activitypub/src/actor.rs:139).
```

**What it shows.** Verification and inbox handling are separate calls. `handle_inbox` does not verify; it requires a `VerifiedActor`, binds the activity's `actor` to it, then records and dispatches.

**Invariant:** nothing is stored before the claimed actor matches the verified actor and every Follow bound passes (`solid-pod-rs-activitypub/src/inbox.rs:145` precedes `solid-pod-rs-activitypub/src/inbox.rs:204`).

**Tension:** "verify before store" is a caller contract, not a type guarantee. `VerifiedActor` has public fields (`solid-pod-rs-activitypub/src/http_sig.rs:76`) and the doc invites soft-logged callers to hand one through (`solid-pod-rs-activitypub/src/inbox.rs:106`). The reference server meets it: its inbox, mounted only with `--ap-federation`, verifies before it calls the library (`solid-pod-rs-server/src/activitypub.rs:603`, then `:613`). See SP-08.20.

## SP-08.6 ActivityPub — outbox and delivery

```mermaid
sequenceDiagram
    autonumber
    participant L as local actor
    participant O as handle_outbox_post<br/>solid-pod-rs-activitypub/src/outbox.rs:166
    participant W as wrap_note_in_create<br/>solid-pod-rs-activitypub/src/outbox.rs:114
    participant HO as handle_outbox<br/>solid-pod-rs-activitypub/src/outbox.rs:33
    participant S as Store
    participant D as DeliveryWorker::drain_once<br/>solid-pod-rs-activitypub/src/delivery.rs:162
    participant R as remote inbox

    L->>O: POST an object to the outbox
    O->>W: a bare Note is wrapped in a Create
    O->>HO: the activity
    HO->>S: record_outbox<br/>solid-pod-rs-activitypub/src/store.rs:435
    HO->>S: follower_inboxes for fan-out, or targetInbox for a Follow<br/>solid-pod-rs-activitypub/src/store.rs:281
    HO->>S: enqueue_delivery, one row per inbox<br/>solid-pod-rs-activitypub/src/outbox.rs:97
    loop DeliveryWorker::run tick<br/>solid-pod-rs-activitypub/src/delivery.rs:419
        D->>S: next_due_delivery<br/>solid-pod-rs-activitypub/src/store.rs:529
        D->>D: resolve_ssrf_safe_with, pin the address, no proxy, no redirects<br/>solid-pod-rs-activitypub/src/delivery.rs:289
        D->>D: sign_request<br/>solid-pod-rs-activitypub/src/http_sig.rs:691
        D->>R: POST the activity
        alt delivered
            D->>S: drop_delivery<br/>solid-pod-rs-activitypub/src/store.rs:592
            D->>S: mark_outbox_state<br/>solid-pod-rs-activitypub/src/store.rs:459
        else 4xx other than 408 and 429
            D->>S: drop and mark failed<br/>solid-pod-rs-activitypub/src/delivery.rs:342
        else other failure
            D->>S: reschedule_delivery with backoff<br/>solid-pod-rs-activitypub/src/store.rs:600
        end
    end
    Note over D: DeliveryOutcome (solid-pod-rs-activitypub/src/delivery.rs:54) and<br/>DeliveryConfig (:69) make the retry policy explicit. enqueue_to_inboxes<br/>(solid-pod-rs-activitypub/src/delivery.rs:399) is a separate helper, not on this path.
```

**Drift:** RESOLVED at a28a748. `handle_outbox` now always replaces `actor` with the pod's own (`solid-pod-rs-activitypub/src/outbox.rs:51`) and replaces an `id` outside the actor's origin (`solid-pod-rs-activitypub/src/outbox.rs:60`).

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

**What it shows.** The WebSocket path takes the typestate route (SP-08.9); `Relay::ingest` is a second, untyped entry that verifies internally. Both reach the same NIP-16 retention rules. The reference server mounts the typed path behind `--nostr` (SP-08.21).

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

## SP-08.20 ActivityPub federation mounted on the server — `--ap-federation`

```mermaid
flowchart TD
    FLAG["--ap-federation needs --activitypub and --public-base-url<br/>solid-pod-rs-server/src/main.rs:635<br/>solid-pod-rs-server/src/main.rs:672"]
    START["Federation::start_with_limits, SsrfGuard::strict,<br/>delivery worker spawned<br/>solid-pod-rs-server/src/main.rs:653<br/>solid-pod-rs-server/src/activitypub.rs:405"]
    ROUTES["POST /inbox, the actor's inbox and outbox, followers, following,<br/>registered ahead of the pod handlers<br/>solid-pod-rs-server/src/lib.rs:6562"]
    subgraph IN["Inbox"]
        HOST["Host must be the configured authority, else 421<br/>solid-pod-rs-server/src/activitypub.rs:582"]
        BODY["body read within 5 s and 256 KiB before a slot is taken<br/>solid-pod-rs-server/src/activitypub.rs:589"]
        SLOT["16 inbox slots, else 503 with Retry-After<br/>solid-pod-rs-server/src/activitypub.rs:593"]
        SIG["verify_request_signature, key fetch guarded and cached<br/>solid-pod-rs-server/src/activitypub.rs:603"]
        BIND["actor equals signer, id on the signer's origin,<br/>a follower's inbox on its own origin<br/>solid-pod-rs-activitypub/src/inbox.rs:145<br/>solid-pod-rs-activitypub/src/inbox.rs:158<br/>solid-pod-rs-activitypub/src/inbox.rs:180"]
        CAPS["5000 followers, 50 per host by default, else 429<br/>solid-pod-rs-activitypub/src/inbox.rs:186<br/>solid-pod-rs-activitypub/src/inbox.rs:192<br/>solid-pod-rs-server/src/main.rs:647"]
        TRIM["200 rows per sender, then 10000 in all,<br/>evicting from the heaviest sender<br/>solid-pod-rs-activitypub/src/inbox.rs:210<br/>solid-pod-rs-activitypub/src/inbox.rs:212<br/>solid-pod-rs-activitypub/src/store.rs:766"]
        ACC["Accept queued only under the delivery-queue bound<br/>solid-pod-rs-server/src/activitypub.rs:626"]
    end
    subgraph OUT["Outbox and delivery"]
        OWN["POST outbox: NIP-98 bound to the body, Control on /<br/>solid-pod-rs-server/src/activitypub.rs:684"]
        TGT["a Follow's target inbox is fetched, never read from the request<br/>solid-pod-rs-server/src/activitypub.rs:702<br/>solid-pod-rs-server/src/activitypub.rs:716"]
        RUN["8 deliveries at once, 2 per host, a breaker per host<br/>solid-pod-rs-activitypub/src/delivery.rs:33<br/>solid-pod-rs-activitypub/src/delivery.rs:37<br/>solid-pod-rs-activitypub/src/delivery.rs:202"]
        HP["the signed Host keeps a non-default port<br/>solid-pod-rs-activitypub/src/http_sig.rs:700"]
    end
    FLAG --> START --> ROUTES
    ROUTES --> HOST --> BODY --> SLOT --> SIG --> BIND --> CAPS --> TRIM --> ACC
    ROUTES --> OWN --> TGT --> RUN --> HP
    ACC --> RUN
    N["Keys and the store file are written 0600 through a fresh temp file<br/>(solid-pod-rs-server/src/activitypub.rs:275). A slow or hostile host holds<br/>at most its own two delivery slots and never delays another host's rows."]
    RUN -.-> N
```

Every bound above is a refusal, not a queue: past it the remote is told 429, 503 or 421 and the store is untouched.

## SP-08.21 The relay mounted on the server — `--nostr`

```mermaid
flowchart TD
    FLAG["--nostr, mounted at /relay, keeping 1000 events by default<br/>solid-pod-rs-server/src/main.rs:463<br/>solid-pod-rs-server/src/nostr_mount.rs:49<br/>solid-pod-rs-server/src/nostr_mount.rs:51"]
    VAL["validate_path: whole plain segments, no dot, not the root<br/>solid-pod-rs-server/src/nostr_mount.rs:114"]
    ROUTE["exactly P and P/info, routed ahead of the pod handlers<br/>solid-pod-rs-server/src/lib.rs:6538"]
    HR["handle_relay: upgrade, NIP-11 on Accept, else 404<br/>solid-pod-rs-server/src/nostr_mount.rs:169"]
    UP["upgrade: at most 256 sockets, else 503<br/>solid-pod-rs-server/src/nostr_mount.rs:215"]
    DISP["dispatch_message_with_limits, the typed path of SP-08.8<br/>solid-pod-rs-server/src/nostr_mount.rs:234"]
    STORE["InMemoryEventStore bounded by --nostr-max-events<br/>solid-pod-rs-server/src/nostr_mount.rs:83"]
    FLAG --> VAL --> ROUTE --> HR --> UP --> DISP --> STORE
    N["The relay is open by design: anyone may publish and subscribe. Its two paths<br/>never reach WAC, while /relayx and /relay/other stay WAC-governed pod paths;<br/>a pod resource stored at P is shadowed while the relay is mounted."]
    ROUTE -.-> N
```
