---
id: CS-03
title: Server listeners, route surface and XR presence
area: codescape
governing:
  - ../codescape/docs/BASELINE-codescape.md
  - ../codescape/AGENTS.md
  - ../codescape/docs/adr/ADR-0001-extraction-from-visionclaw.md
adrs: [codescape:ADR-0001, visionclaw:ADR-2143]
sources:
  - ../codescape/docs/BASELINE-codescape.md
  - ../codescape/AGENTS.md
  - ../codescape/docs/adr/ADR-0001-extraction-from-visionclaw.md
  - ../codescape/crates/codescape-server/src/main.rs
  - ../codescape/crates/codescape-server/src/lib.rs
  - ../codescape/crates/codescape-server/src/http.rs
  - ../codescape/crates/codescape-server/src/auth.rs
  - ../codescape/crates/codescape-server/src/config.rs
  - ../codescape/crates/codescape-server/src/ingest_api.rs
  - ../codescape/crates/codescape-server/src/routes/mod.rs
  - ../codescape/crates/codescape-server/src/routes/presence.rs
  - ../codescape/crates/codescape-server/src/xr_presence.rs
verified_commit: 446f8e386d7fc31e4cf28db25bf976c636b149ce
---

## CS-03.1 Boot order and the two listeners

```mermaid
flowchart TD
    ENV["check_environment before argument parsing<br/>main.rs:48"]
    EXIT2["RUVECTOR_PG_CONNINFO set means exit code 2<br/>main.rs:49,50<br/>config.rs:15,311"]
    VAL["Config validate, failure is exit code 2<br/>main.rs:53,55"]
    TOK["ingest token file must be readable and non-empty, else boot fails<br/>main.rs:90,91,94"]
    REPOS["each repo root must be a directory, RepoEngine opened and threads spawned<br/>main.rs:97,121,122"]
    MAIN["main listener with auth_gate, api_rate_gate and isolation headers<br/>main.rs:149,156,157,158"]
    INGEST["ingest listener only when a token exists, two workers<br/>main.rs:175,176,185"]
    OFF["no token: ingest listener disabled<br/>main.rs:191,192"]
    FLOOR["floor hold cap ticks every second<br/>main.rs:241,242,245"]
    STOP["SIGTERM or interrupt stops ingest then main<br/>main.rs:263,268,271"]
    ENV -->|set| EXIT2
    ENV -->|unset| VAL --> TOK --> REPOS --> MAIN
    TOK --> INGEST
    TOK --> OFF
    MAIN --> FLOOR --> STOP
    INGEST --> STOP
    TENS["TENSION: OpenAccess is the only verifier today, the main port is open until P6 NIP-98<br/>auth.rs:118,120,126"]
    MAIN --- TENS
    NOTE["see VC-39.6 for forwarded-header trust and VC-43.1 for presence admission"]
    TENS --- NOTE
```

**What it shows.** The memory-store refusal runs before anything else, then the token decides whether the second listener exists at all. Both listeners share one shutdown flag (`main.rs:116,267`).

## CS-03.2 Route surface and the routes that must stay 404

```mermaid
flowchart LR
    REQ["request on the main listener"]
    API["registered routes health, snapshot, node, search, query, physics, settlement, ontology-refs<br/>routes/mod.rs:442,443,444,445,446,447,448,449"]
    WS["websocket routes positions, events, wss, voice, presence<br/>routes/mod.rs:450,451,452,453,454"]
    NS["NOT_SERVED memory-cloud, analytics, graph fold, canary answer 404 with tail match<br/>http.rs:123,124,125,126,127,133,134"]
    SPA["static bundle with SPA fallback, backend paths never fall back<br/>http.rs:139,140,141,142,143"]
    ING["ingest app serves one POST route, everything else 404<br/>ingest_api.rs:198,199,201"]
    REQ --> API
    REQ --> WS
    REQ --> NS
    REQ --> SPA
    NS -.->|"registered before the static fallback"| SPA
    INV["INVARIANT: knowledge routes return 404 and stay that way<br/>AGENTS.md:78"]
    NS --- INV
    EXT["EXTERNAL: the XR client probes these routes at boot, which is why they answer 404 not 501<br/>http.rs:122"]
    NS --- EXT
    DOC["DOC-DRIFT: crate docs call the ingest listener the presence-ingest app, it ingests agent activity<br/>lib.rs:17<br/>ingest_api.rs:178,180"]
    ING --- DOC
```

**What it shows.** The knowledge-work routes are explicit 404s ahead of the SPA fallback, so a client cannot mistake them for a page. Hook callers can only appear as agents (`ingest_api.rs:178,180`).

## CS-03.3 One XR presence session

```mermaid
sequenceDiagram
    autonumber
    participant XR as XR client
    participant SES as presence_session<br/>routes/presence.rs:112
    participant HUB as PresenceHub<br/>xr_presence.rs:602
    participant REPO as RepoEngine actors
    SES->>XR: challenge nonce (presence.rs:120,127)
    XR->>SES: auth with did, room_id, metadata, 10 s deadline (presence.rs:34,92,99)
    SES->>SES: room_of bounds the room id, Identity::resolve labels the DID (presence.rs:133,136)
    SES->>REPO: register_actor as a human, failure is tolerated (presence.rs:143,146)
    SES->>HUB: join room, name, model_uri (presence.rs:150)
    HUB-->>SES: Membership with the joined roster (presence.rs:155)
    SES->>REPO: one focus presence on no node (presence.rs:41,159)
    loop until close, shutdown or heartbeat loss
        XR->>SES: binary 0x43 pose, decoded then rate limited (presence.rs:176,178)
        SES->>HUB: member.relay pose (presence.rs:181)
        HUB-->>SES: member.rx control then pose frames (presence.rs:198,199,200)
    end
    SES->>HUB: drop member announces avatar_left (presence.rs:213)
    Note over SES: poses touch last_seen once a second and re-announce every 30 s (presence.rs:36,38)
    Note over SES: DIVERGENCE: the signature is accepted unverified until P6 (presence.rs:98)
    Note over HUB: see VC-43.1
```

**What it shows.** The session is a thin shell: all room state lives in the hub, and the desktop sees XR members as ordinary human actors. Poses never heat code nodes (`routes/presence.rs:12`).

## CS-03.4 Replacement and relay fencing after the 2026-11-03 repair

```mermaid
sequenceDiagram
    autonumber
    participant OLD as Old Membership<br/>xr_presence.rs:720
    participant HUB as PresenceHub.join<br/>xr_presence.rs:613
    participant NEW as Replacement Membership<br/>xr_presence.rs:681
    participant OBS as Observer
    HUB->>HUB: same avatar_id found, swap_remove old entry (xr_presence.rs:625,626)
    HUB-->>OLD: try_send Close then cancel signal (xr_presence.rs:627,630)
    HUB-->>OBS: avatar_left for the old local_id (xr_presence.rs:631)
    HUB->>NEW: fresh local_id, announce avatar_joined, push entry (xr_presence.rs:639,664,668)
    OLD->>HUB: relay with the held handle (xr_presence.rs:739)
    HUB->>HUB: pose room and avatar must match the handle (xr_presence.rs:740)
    HUB->>HUB: local_id must still be a current room member, else reach 0 (xr_presence.rs:747,750)
    Note over OLD,HUB: INVARIANT: a replaced handle cannot inject poses, its drop is a no-op (xr_presence.rs:1013,1018)
    Note over HUB: TENSION: avatar_id comes from the claimed DID and the signature is unchecked, so a client claiming another DID evicts that member (xr_presence.rs:20,21,250,625)
    Note over OLD: see VC-43.2 for the pre-repair behaviour at 58108c0
```

**What it shows.** Relay now checks membership, not just room and avatar, which closes the stale-handle gap recorded at `ADR-0001-extraction-from-visionclaw.md:44,45`. Eviction by DID claim remains because identity is open.

## CS-03.5 Two queues per member and the roster-overflow close

```mermaid
flowchart TD
    SEND["Inner::send_all to every member except the sender<br/>xr_presence.rs:573"]
    KIND{"message kind"}
    POSE["binary pose: try_send on a 64-deep queue, a full queue drops it<br/>xr_presence.rs:73,577"]
    CTRL["text roster event: try_send on the control queue sized 2 x members<br/>xr_presence.rs:578,666"]
    FULL["control queue full: set the cancel flag, count as not reached<br/>xr_presence.rs:580,581,582"]
    RECV["MemberInbox::recv: cancel first, then control, then poses<br/>xr_presence.rs:532,538,539,540,541,542"]
    CLOSE["Close roster overflow, reconnect for a fresh roster<br/>xr_presence.rs:534,535"]
    SOCK["session closes with a policy code<br/>routes/presence.rs:201,203"]
    SEND --> KIND
    KIND -->|binary| POSE
    KIND -->|text| CTRL
    CTRL -->|full| FULL --> RECV
    POSE --> RECV
    CTRL --> RECV
    RECV -->|cancelled| CLOSE --> SOCK
    INV["INVARIANT: poses cannot displace roster transitions, an unrecoverable roster ends the session<br/>xr_presence.rs:1025,1074,1082,1090"]
    RECV --- INV
    DEBT["DEBT: a replaced socket is told roster overflow or roster changed, never replaced by a newer connection, because cancel is checked first<br/>xr_presence.rs:627,630,532,534,540"]
    CLOSE --- DEBT
    NOTE["see VC-43.3 for the single shared queue this replaced"]
    POSE --- NOTE
```

**What it shows.** Lossy poses and reliable roster events no longer share a bound, and exhausting the reliable one disconnects rather than diverges. The test accepts any Close, so the wrong reason text is not pinned (`xr_presence.rs:1001`).

## CS-03.6 How the presence gaps were found, repaired and left documented

```mermaid
flowchart TD
    G1["2026-11-02 review at 58108c0: relay did not check membership, one queue carried poses and roster<br/>ADR-0001-extraction-from-visionclaw.md:38,40,43"]
    NEED["closure needs a replacement-handle test and a saturated-queue test<br/>ADR-0001-extraction-from-visionclaw.md:46,47"]
    FIX["2026-11-03 d2c3c3b fences relay and splits control from poses<br/>ADR-0001-extraction-from-visionclaw.md:21,22,23,24,25"]
    TESTS["regressions cover stale handles, pose saturation and control saturation<br/>ADR-0001-extraction-from-visionclaw.md:32,33,34"]
    BASE["BASELINE records the repair, no deployment claim<br/>BASELINE-codescape.md:3,4,5,6,7"]
    G1 --> NEED --> FIX --> TESTS
    FIX --> BASE
    D1["DOC-DRIFT: VC-43.2 and VC-43.3 describe the pre-repair behaviour as current<br/>xr_presence.rs:747,578"]
    D2["DOC-DRIFT: open item says presence carries no avatars between clients, the hub relays poses<br/>BASELINE-codescape.md:120<br/>xr_presence.rs:739,748"]
    D3["OPEN: identity is unverified until P6, so DID claims are labels<br/>BASELINE-codescape.md:124<br/>xr_presence.rs:20,21"]
    FIX --- D1
    BASE --- D2
    BASE --- D3
    NOTE["see VC-43.2 and VC-43.3"]
    D1 --- NOTE
```

**What it shows.** The repair is source and test closure only; the ADR states no service was deployed (`ADR-0001-extraction-from-visionclaw.md:34`). Two older documents still describe the earlier behaviour.
