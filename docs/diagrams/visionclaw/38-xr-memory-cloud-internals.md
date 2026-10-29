---
id: VC-38
title: XR memory cloud internals - snapshot, route, query, beat and semantic encoding
area: visionclaw
governing:
  - ../project/docs/XR-client.md
adrs: [ADR-2107, ADR-2133, ADR-2134, ADR-2135, ADR-2136]
sources:
  - ../project/docs/XR-client.md
  - ../project/xr-client/rust/src/lib.rs
  - ../project/xr-client/rust/src/memory_cloud.rs
  - ../project/xr-client/rust/src/cloud_frame.rs
  - ../project/xr-client/rust/src/memory_route.rs
  - ../project/xr-client/rust/src/memory_query.rs
  - ../project/xr-client/rust/src/beat.rs
  - ../project/xr-client/rust/src/semantic.rs
  - ../project/xr-client/rust/src/pulse.rs
  - ../project/xr-client/rust/src/binary_protocol.rs
  - ../project/xr-client/rust/src/render_store.rs
  - ../project/xr-client/rust/src/frame_budget.rs
  - ../project/xr-client/scripts/graph_scene.gd
  - ../project/xr-client/scripts/memory_cloud_layer.gd
  - ../project/xr-client/scripts/memory_search.gd
  - ../project/xr-client/scripts/beat_pulse.gd
  - ../project/xr-client/scripts/memory_bursts.gd
  - ../project/crates/visionclaw-tri-layout/src/lib.rs
  - ../project/src/handlers/socket_flow_handler/session_relay.rs
  - ../project/src/handlers/socket_flow_handler/message_routing.rs
verified_commit: {visionclaw: e74bcddc772d47a45b1971156a2bab540eaa027c}
---

## For developers

VC-36.25 drew the seams of the desktop-parity layers and VC-36.27 listed the gdext classes; this topic opens the six Rust modules behind them. `memory_cloud.rs`, `cloud_frame.rs`, `memory_route.rs`, `memory_query.rs`, `beat.rs` and `semantic.rs` are, except `memory_query.rs` which is a headset-side design, ports of the desktop explorer's TypeScript (`xr-client/rust/src/memory_cloud.rs:1-11`, `xr-client/rust/src/cloud_frame.rs:1-6`, `xr-client/rust/src/beat.rs:1-3`, `xr-client/rust/src/semantic.rs:1-3`), each split into a plain-Rust core that `cargo test` covers headless and a thin `RefCounted` adapter that GDScript composes. The headset never runs HNSW and never fetches the vectors blob: it draws a server-built snapshot, a route the desktop relays, and a sidecar top-k line it builds from its own query (`xr-client/rust/src/memory_query.rs:3-9`).

Threading is simpler than it looks. None of the six modules holds a `Mutex`, `Arc` or `RefCell`, and every adapter method runs on the Godot calling thread with `&mut self`. The single concurrent thing is the mic tempo worker that `pulse.rs` spawns per analysis and reads back over an `mpsc` channel (xr-client/rust/src/pulse.rs:142). Clocks are explicit parameters: wall-clock epoch milliseconds for the beat and the route's `sentAt`, per-frame `dt` for animation, and no monotonic clock anywhere in the beat path (xr-client/rust/src/pulse.rs:27).

Every number that crosses a boundary is clamped where it enters. The snapshot is bounded at 50 000 rows (xr-client/rust/src/memory_cloud.rs:47), a route at 64 path rows, 64 sidecar rows and 120 query characters (xr-client/rust/src/memory_route.rs:148), a headset query at 2 000 characters and k 50 (xr-client/rust/src/memory_query.rs:46), a `memory_flash` batch at 64 events (xr-client/rust/src/semantic.rs:341). VC-38.1 is the ownership map, VC-38.2 to VC-38.6 the cloud and its frame, VC-38.7 to VC-38.11 the relayed route, VC-38.12 and VC-38.13 the headset query, VC-38.14 to VC-38.17 the beat and VC-38.18 and VC-38.19 the burst and beam encoding.

## For the business

The memory cloud is the room-sized picture of what the estate remembers: every stored memory is a point, grouped and coloured by where it came from and how old it is. A person in the headset can ask a question in plain words and see the line of related memories light up, with the best answer ringed. The same cloud on the desktop and in the headset tell the same story because the headset reuses the desktop's colours, framing and scoring rather than inventing its own.

Two things make it feel alive. A shared beat, taken from the desktop, a tap, or the room's own music if the wearer opts in, makes the graph and the cloud pulse together. And a small burst flares on the memory touched whenever an agent stores, retrieves, searches or deletes, in a colour that says which verb it was. Nothing the microphone hears is kept or sent; it is analysed in memory and dropped.

The honest limits are in the register below: the headset has not yet been used to judge brightness, legibility or stereo agreement of these layers, and a few tuning constants are copied by hand between the headset and its server.

## VC-38.1 Ownership map - who owns state, who runs on which thread

```mermaid
flowchart TB
    subgraph gd["GDScript owners on the scene-tree thread"]
        GS["_on_graph_text routes by message type<br/>xr-client/scripts/graph_scene.gd:3029"]
        LAYER["MemoryCloudLayer owns the cloud, the frame and the route objects<br/>xr-client/scripts/memory_cloud_layer.gd:132"]
        SEARCH["MemorySearch owns the query object and one HTTP request in flight<br/>xr-client/scripts/memory_search.gd:58"]
        BEATGD["BeatPulse script wraps the gdext BeatPulse and pings every 2 s<br/>xr-client/scripts/beat_pulse.gd:32"]
    end
    subgraph cls["gdext adapters, RefCounted, no_init, created through create()"]
        MC["MemoryCloud<br/>xr-client/rust/src/memory_cloud.rs:883"]
        CF["CloudFrame<br/>xr-client/rust/src/cloud_frame.rs:276"]
        MR["MemoryRoute<br/>xr-client/rust/src/memory_route.rs:1575"]
        MQ["MemoryQuery<br/>xr-client/rust/src/memory_query.rs:466"]
        BP["BeatPulse<br/>xr-client/rust/src/pulse.rs:41"]
        FC["MemoryFlashCodec, static parse<br/>xr-client/rust/src/pulse.rs:208"]
    end
    subgraph core["Plain Rust cores, headless-testable, no locks"]
        CS["CloudState: snapshot, index maps, colours, drawn set<br/>xr-client/rust/src/memory_cloud.rs:767"]
        PG["PlacementGlide: bounds and glide<br/>xr-client/rust/src/cloud_frame.rs:193"]
        RG["RouteGate and ActiveRoute: ordering, sampling, clock<br/>xr-client/rust/src/memory_route.rs:588 xr-client/rust/src/memory_route.rs:1408"]
        QA["parse_query_response and sidecar_route<br/>xr-client/rust/src/memory_query.rs:218 xr-client/rust/src/memory_query.rs:261"]
        BS["BeatSync and MicBeat: arbitration and mic window<br/>xr-client/rust/src/beat.rs:397 xr-client/rust/src/beat.rs:854"]
        SE["semantic: burst profile, hue shift, beam shape<br/>xr-client/rust/src/semantic.rs:91"]
    end
    WORKER["mic tempo worker, the only extra thread, one per analysis<br/>result returns over mpsc, detached<br/>xr-client/rust/src/pulse.rs:142 xr-client/rust/src/pulse.rs:141"]
    GS --> LAYER
    GS --> BEATGD
    LAYER --> MC
    LAYER --> CF
    LAYER --> MR
    SEARCH --> MQ
    BEATGD --> BP
    BEATGD --> FC
    MC --> CS
    CF --> PG
    MR --> RG
    MQ --> QA
    BP --> BS
    FC --> SE
    BS --> WORKER
    INV["INVARIANT: no Mutex, Arc or RefCell in the six modules, every adapter method borrows self on the calling thread<br/>the lone channel is xr-client/rust/src/pulse.rs:19"]
    WORKER --> INV
```

**What it shows.** Three tiers. GDScript scripts own the lifecycle and call `create()`; six gdext adapters convert Godot types; plain-Rust cores hold the decisions. The mic worker is the only thing off the main thread. **Why it is this way.** The crate header makes GDScript compose the scene and Rust own every threshold (xr-client/rust/src/lib.rs:21); `pulse.rs` states that all decision logic lives in the pure modules so it is unit-tested headless (xr-client/rust/src/pulse.rs:10). Recorded under ADR-2134 for the beat and ADR-2133 for the cloud.

## VC-38.2 Snapshot load, from HTTP completion to a MultiMesh buffer

```mermaid
sequenceDiagram
    autonumber
    participant L as MemoryCloudLayer<br/>xr-client/scripts/memory_cloud_layer.gd:336
    participant M as MemoryCloud adapter<br/>xr-client/rust/src/memory_cloud.rs:883
    participant S as CloudState load<br/>xr-client/rust/src/memory_cloud.rs:791
    participant P as CloudSnapshot parse<br/>xr-client/rust/src/memory_cloud.rs:184
    participant F as CloudFrame<br/>xr-client/rust/src/cloud_frame.rs:276

    L->>M: classify_status(transport_ok, code) xr-client/rust/src/memory_cloud.rs:1132
    alt Ready 2xx
        L->>M: load_bytes(body) xr-client/rust/src/memory_cloud.rs:904
        M->>S: load(bytes)
        S->>P: parse(bytes) xr-client/rust/src/memory_cloud.rs:792
        Note over P: JSON error, version other than 1, blank snapshotId,<br/>count above 50000, then positions 3n and metadata n<br/>xr-client/rust/src/memory_cloud.rs:188 xr-client/rust/src/memory_cloud.rs:193 xr-client/rust/src/memory_cloud.rs:196
        P-->>S: positions with non-finite values zeroed, metadata with empty defaults xr-client/rust/src/memory_cloud.rs:206
        S->>S: IndexMaps::build, build_colours, keep.clear, refresh_drawn xr-client/rust/src/memory_cloud.rs:794
        S-->>M: Ok(rows) or Err with the previous snapshot kept and last_error set xr-client/rust/src/memory_cloud.rs:804
        M-->>L: row count or minus one, minus one sets state failed and keeps the old cloud xr-client/scripts/memory_cloud_layer.gd:344
        L->>F: set_cloud_positions(positions) - robust bounds, placed false, next step snaps xr-client/rust/src/cloud_frame.rs:293
        L->>M: build_buffer(sprite, opacity, dim) when dirty xr-client/rust/src/memory_cloud.rs:1018
        M-->>L: PackedFloat32Array, stride 20
    else Forbidden 401 or 403
        L-->>L: hide points, clear route, state forbidden xr-client/scripts/memory_cloud_layer.gd:365
    else Unavailable 503
        L->>M: parse_retry_after then unavailable_delay, retry at the returned delay xr-client/rust/src/memory_cloud.rs:1150
    else Stale 409 or Failed
        L-->>L: one immediate reload on 409, then failed, VC-38.5 xr-client/scripts/memory_cloud_layer.gd:374
    end
```

**What it shows.** One load, end to end. A failed parse never clobbers the snapshot on screen: the previous one stays and `last_error` is set. **Why it is this way.** The load policy ports `memoryCloudStore.ts` (xr-client/rust/src/memory_cloud.rs:10); the positions-only snapshot is the ADR-2133 sidecar contract.

**Invariant:** `count` is judged before the array shapes, so an oversize snapshot reports `TooLarge` rather than a shape error (xr-client/rust/src/memory_cloud.rs:193 then xr-client/rust/src/memory_cloud.rs:196).

## VC-38.3 Snapshot wire shape and the stride-20 instance buffer

```mermaid
classDiagram
    class WireSnapshot {
        version u32 must be 1
        snapshotId String non-blank
        generatedAt optional default 0
        count usize at most 50000
        positions f32 flat 3 per row cloud-local
        metadata array of WireMeta
        namespaces sorted by the server
        sourceTypes sorted by the server
        def : xr-client/rust/src/memory_cloud.rs:125
    }
    class WireMeta {
        id key namespace sourceType all optional
        updatedAt optional non-finite becomes 0
        def : xr-client/rust/src/memory_cloud.rs:110
    }
    class CloudSnapshot {
        snapshot_id generated_at count
        positions metadata namespaces source_types
        def : xr-client/rust/src/memory_cloud.rs:152
    }
    class CloudError {
        Json Version EmptyId Shape TooLarge
        def : xr-client/rust/src/memory_cloud.rs:164
    }
    class InstanceBuffer {
        CLOUD_STRIDE 20 floats per drawn row
        floats 0 to 11 row-major 3x4 transform uniform scale sprite
        floats 12 to 15 rgb times k and alpha opacity
        floats 16 to 19 NEUTRAL_EMPHASIS 0 0 0 1
        k is 1 or one minus dim for rows not in keep
        a row with no position is skipped
        def : xr-client/rust/src/memory_cloud.rs:67 build : xr-client/rust/src/memory_cloud.rs:475
    }
    class ColourMode {
        Namespace 0 SourceType 1 Age 2
        palette of 16 cycled, unlisted category takes slot 0
        age ramp over three stops
        def : xr-client/rust/src/memory_cloud.rs:246
    }
    WireSnapshot "1" --> "n" WireMeta
    WireSnapshot ..> CloudSnapshot : parse validates and converts
    CloudSnapshot ..> CloudError : parse may fail
    CloudSnapshot ..> InstanceBuffer : drawn rows packed
    ColourMode ..> InstanceBuffer : per row rgb
```

**What it shows.** The only JSON the cloud reads, and the only buffer it writes. Unknown fields such as `dim`, `strata` or `vectorsUrl` are ignored because the wire struct does not deny them. **Why it is this way.** Positions and metadata only, no vectors, because the headset never searches (xr-client/rust/src/memory_cloud.rs:5).

**Drift:** the doc comment on `build_buffer` still says "Stride-16 instance buffer" (xr-client/rust/src/memory_cloud.rs:1016) while the constant and the test are stride 20 (xr-client/rust/src/memory_cloud.rs:67).

**Drift:** `NEUTRAL_EMPHASIS` is documented as the value `memory_flash` bursts "write it per instance" (xr-client/rust/src/memory_cloud.rs:72), but Rust always writes the neutral value; the emphasis is applied in GDScript on a duplicate of the buffer (xr-client/scripts/memory_cloud_layer.gd:871).

## VC-38.4 Which rows are drawn, where a flash lands, what the pointer picks

```mermaid
flowchart TB
    subgraph lod["select_drawn, deterministic level of detail"]
        L0{"rows at most cap?"}
        L1["draw every row<br/>xr-client/rust/src/memory_cloud.rs:391"]
        L2["pinned rows first, even past the cap<br/>route and sidecar rows must land on visible points<br/>xr-client/rust/src/memory_cloud.rs:399"]
        L3["budget is cap minus pinned, split per namespace<br/>floor share, one row minimum while budget lasts, then largest remainder<br/>xr-client/rust/src/memory_cloud.rs:402 xr-client/rust/src/memory_cloud.rs:431 xr-client/rust/src/memory_cloud.rs:437"]
        L4["each namespace takes evenly spaced rows, centred in each stride<br/>output sorted ascending so binary_search finds the instance<br/>xr-client/rust/src/memory_cloud.rs:461 xr-client/rust/src/memory_cloud.rs:467"]
    end
    subgraph fl["resolve_flash, never a random point"]
        F0{"key present?"}
        F1["try namespace:key then bare key, every matching row, FlashMatch Key<br/>xr-client/rust/src/memory_cloud.rs:595 xr-client/rust/src/memory_cloud.rs:599"]
        F2["else up to NAMESPACE_PICKS 3 distinct rows of the namespace<br/>xorshift64 seeded per call, FlashMatch Namespace<br/>xr-client/rust/src/memory_cloud.rs:608 xr-client/rust/src/memory_cloud.rs:87"]
        F3["else empty, FlashMatch None<br/>xr-client/rust/src/memory_cloud.rs:622"]
    end
    subgraph pk["pick_ray, angular nearest"]
        K1["direction must be finite and longer than 1e-9<br/>xr-client/rust/src/memory_cloud.rs:638"]
        K2["only drawn rows, in front of the origin, t above 1e-4<br/>xr-client/rust/src/memory_cloud.rs:650"]
        K3["smallest perpendicular over range ratio within tan of max_angle<br/>xr-client/rust/src/memory_cloud.rs:654 xr-client/rust/src/memory_cloud.rs:655"]
    end
    L0 -->|yes| L1
    L0 -->|no| L2 --> L3 --> L4
    L4 --> F0
    F0 -->|yes| F1
    F0 -->|no| F2
    F1 -.->|key present but unmatched| F2
    F2 -->|namespace unknown| F3
    F1 --> K1
    NOTE7["flash targets come from all sampled rows, not only the drawn subset, an undrawn row maps to no instance<br/>xr-client/rust/src/memory_cloud.rs:864"]
    F2 --> NOTE7
    K1 --> K2 --> K3
```

**What it shows.** The three pure functions that turn a snapshot into something on screen: which rows are sprites, which rows a burst targets, which row the wand is pointing at. **Why it is this way.** Ports of `resolveFlashTargets` and the desktop quota rules (xr-client/rust/src/memory_cloud.rs:582); deterministic selection so the same snapshot and cap always draw the same points.

**Invariant:** the drawn set is sorted ascending and unique, so `instance_of_row` can binary-search it (xr-client/rust/src/memory_cloud.rs:467, xr-client/rust/src/memory_cloud.rs:869).

## VC-38.5 Load policy, retry and the single stale reload

```mermaid
flowchart TB
    S0["classify_status(transport_ok, code)<br/>xr-client/rust/src/memory_cloud.rs:677"]
    S1["Ready 200 to 299: load, streak 0, stale flag cleared, buffer dirty<br/>xr-client/scripts/memory_cloud_layer.gd:349"]
    S2["Forbidden 401 or 403: hide points, clear the route, quiet state not a toast<br/>xr-client/scripts/memory_cloud_layer.gd:362"]
    S3["Unavailable 503: Retry-After seconds or HTTP-date, else 5 s doubling, clamped 1 s to 60 s<br/>xr-client/rust/src/memory_cloud.rs:731 xr-client/rust/src/memory_cloud.rs:748"]
    S4["Stale 409: reload once at once, a second 409 fails with snapshot rebuilt twice<br/>xr-client/scripts/memory_cloud_layer.gd:374 xr-client/scripts/memory_cloud_layer.gd:378"]
    S5["Failed: transport failure or any other status, reported with the cause<br/>xr-client/scripts/memory_cloud_layer.gd:381"]
    S6["Ready, Forbidden, a failed parse, a second 409 and Failed re-offer a pending route, a 503 and the first 409 do not<br/>xr-client/scripts/memory_cloud_layer.gd:533"]
    S0 --> S1
    S0 --> S2
    S0 --> S3
    S0 --> S4
    S0 --> S5
    S1 --> S6
    S2 --> S6
    S4 --> S6
    S5 --> S6
    N1["INVARIANT: the backoff streak grows only on a 503 without Retry-After, shift capped at 20, saturating multiply<br/>xr-client/rust/src/memory_cloud.rs:749 xr-client/rust/src/memory_cloud.rs:751"]
    N2["a past HTTP-date yields 0 then the 1 s floor, anything that is neither digits nor a parseable date yields None<br/>xr-client/rust/src/memory_cloud.rs:742 xr-client/rust/src/memory_cloud.rs:739"]
    S3 --> N1
    S3 --> N2
```

**What it shows.** The whole status policy in one place. Only a 503 retries on a timer; only a 409 retries immediately, and only once. **Why it is this way.** Ported from `memoryCloudStore.ts`: the cloud is a power-user surface, so a 401 or 403 is silence rather than an error toast (xr-client/scripts/memory_cloud_layer.gd:362).

## VC-38.6 Placing the cloud: robust bounds, the triangle's memory vertex, the glide

```mermaid
flowchart TB
    subgraph bounds["robust_bounds, percentile box"]
        B1["drop points with any non-finite coordinate<br/>xr-client/rust/src/cloud_frame.rs:76"]
        B2["per axis the 5th and 95th percentile by select_nth, centre is the midpoint<br/>xr-client/rust/src/cloud_frame.rs:86 xr-client/rust/src/cloud_frame.rs:94 xr-client/rust/src/cloud_frame.rs:96"]
        B3["radius is half the diagonal of the percentile box, floor 1.0<br/>xr-client/rust/src/cloud_frame.rs:101 xr-client/rust/src/cloud_frame.rs:47"]
    end
    subgraph graph["graph side"]
        G1["graph_bounds_for folds every node into its own graph frame first<br/>so the separated triangle does not inflate the radius<br/>xr-client/rust/src/cloud_frame.rs:166 xr-client/rust/src/cloud_frame.rs:171"]
        G2["BinaryProtocolClient.graph_robust_bounds reads it from the render store, GDScript samples it once a second<br/>xr-client/rust/src/binary_protocol.rs:1936 xr-client/rust/src/render_store.rs:2122 xr-client/rust/src/cloud_frame.rs:49"]
    end
    subgraph place["cloud_placement"]
        P1["with both bounds: scale is MEMORY_BODY_SCALE 10 times k over 5 times graph radius over cloud radius<br/>base is the graph centre<br/>xr-client/rust/src/cloud_frame.rs:146"]
        P2["without: scale is 10 times k, base is the origin<br/>xr-client/rust/src/cloud_frame.rs:149"]
        P3["position is base plus TriangleFrame::separated().memory_centre(graph radius or 152, scale times cloud radius)<br/>offset is minus the cloud centre, applied to the inner node at once<br/>xr-client/rust/src/cloud_frame.rs:151 crates/visionclaw-tri-layout/src/lib.rs:84 crates/visionclaw-tri-layout/src/lib.rs:110"]
    end
    subgraph glide["PlacementGlide.step"]
        D1["factor is dt over 0.8 s capped at 1, the first placement and reduced motion snap with factor 1<br/>xr-client/rust/src/cloud_frame.rs:251 xr-client/rust/src/cloud_frame.rs:248"]
        D2["a new snapshot resets placed so it snaps, graph updates glide<br/>xr-client/rust/src/cloud_frame.rs:235"]
        D3["step returns position, scale, offset and the sprite size from the desktop point size and fov<br/>xr-client/rust/src/cloud_frame.rs:186 xr-client/rust/src/cloud_frame.rs:246"]
    end
    B1 --> B2 --> B3
    G1 --> G2
    B3 --> P1
    G2 --> P1
    P1 --> P3
    P2 --> P3
    P3 --> D1 --> D2 --> D3
```

**What it shows.** How the cloud ends up one memory-body width from the graph, scaled to match its size, and how a graph that grows or shrinks moves it smoothly instead of popping. **Why it is this way.** A port of `robustBounds.ts`, `cloudFrame.ts` and the placement loop of `EmbeddingCloudLayer.tsx` (xr-client/rust/src/cloud_frame.rs:3), with the placement delegated to the shared triangle crate so the room and the desktop agree where the memory vertex is (ADR-2135, VC-36.26).

**Invariant:** the graph bounds are accepted only as exactly four finite floats with a positive radius, anything else clears them (xr-client/rust/src/cloud_frame.rs:311).

**Drift:** `CloudFrame.step` hard-wires the desktop point size and field of view for the sprite (xr-client/rust/src/cloud_frame.rs:339), and `MemoryCloud.sprite_local_size` ignores placement scale altogether and passes the desktop constant (xr-client/rust/src/memory_cloud.rs:1023); the per-frame `sprite` from `step` is what the layer applies (xr-client/scripts/memory_cloud_layer.gd:752).
## VC-38.7 The memoryRoute frame - wire shape, parsed frame, relay limits

```mermaid
classDiagram
    class MemoryRouteWire {
        type must be memoryRoute
        snapshotId required non-blank
        seq optional finite and non-negative else 0
        sentAt optional finite else 0 milliseconds
        path required at most 64 rows each an integer 0 to u32 max
        sidecar optional bad entries dropped at most 64
        query optional first 120 characters
        sidecarTotal and sidecarAgree optional validated pair
        serverTime stamped by the relay and ignored here
        def : xr-client/rust/src/memory_route.rs:317 limits : xr-client/rust/src/memory_route.rs:148
    }
    class RouteFrame {
        snapshot_id seq sent_at path sidecar query stats
        source Relay or SidecarTopK
        origin optional query point, only a headset query sets it
        hit_points and hit_sampled, only a headset query sets them
        anchors counts origin plus hit points or path rows
        def : xr-client/rust/src/memory_route.rs:338 anchors : xr-client/rust/src/memory_route.rs:369
    }
    class SidecarStats {
        total in_sample agree
        valid only when in_sample at most total and agree at most in_sample
        a bad pair drops the stats, never the frame
        def : xr-client/rust/src/memory_route.rs:408 from_wire : xr-client/rust/src/memory_route.rs:416
    }
    class RouteError {
        Json WrongType EmptySnapshot BadIndex TooLong
        def : xr-client/rust/src/memory_route.rs:494
    }
    class RelayValidator {
        server side, same limits, stricter on seq and sentAt and a non-array sidecar
        frame at most 16 KiB, relayed to the same user's sessions at most 4 Hz
        def : src/handlers/socket_flow_handler/session_relay.rs:164 src/handlers/socket_flow_handler/session_relay.rs:32
    }
    MemoryRouteWire ..> RouteFrame : parse_route converts
    MemoryRouteWire ..> RouteError : parse_route may reject
    RouteFrame o-- SidecarStats
    RelayValidator ..> MemoryRouteWire : emits the canonical frame
```

**What it shows.** What the headset accepts from the desktop's relayed frame and what it makes of it. An empty `path` is the clear signal; a bad `sidecar` entry is dropped, a bad `path` entry rejects the whole frame. **Why it is this way.** The relay and the headset parser share one set of limits, but they differ at the edges: the relay rejects a negative `seq` (src/handlers/socket_flow_handler/session_relay.rs:195) and a non-array `sidecar` (src/handlers/socket_flow_handler/session_relay.rs:188) while accepting `null`; the client defaults a bad `seq` to 0 (xr-client/rust/src/memory_route.rs:554) and rejects any non-array `sidecar`, `null` included, as a JSON error (xr-client/rust/src/memory_route.rs:325).

**Drift:** `docs/XR-client.md` describes the frame as `{type, snapshotId, seq, sentAt, path[], sidecar[], query}` (docs/XR-client.md:443) while the code reads two further fields, `sidecarTotal` and `sidecarAgree` (xr-client/rust/src/memory_route.rs:209).

## VC-38.8 RouteGate - ordering, reload and the five verdicts

```mermaid
flowchart TB
    IN["RouteFrame offered with loaded snapshot id and row count<br/>xr-client/rust/src/memory_route.rs:606"]
    Q1{"strictly newer than the last applied by sentAt then seq?"}
    ST["Stale<br/>xr-client/rust/src/memory_route.rs:613"]
    Q2{"path and hit points both empty?"}
    CL["Clear, last recorded, pending dropped<br/>xr-client/rust/src/memory_route.rs:621"]
    Q3{"names the loaded snapshot?"}
    Q4{"already reloaded for this id?"}
    DR1["Drop<br/>xr-client/rust/src/memory_route.rs:625"]
    RL["Reload, frame held as pending, id remembered<br/>xr-client/rust/src/memory_route.rs:629"]
    Q5{"fewer than 2 anchors, a non-finite point or a path row outside the snapshot?"}
    DR2["Drop<br/>xr-client/rust/src/memory_route.rs:633"]
    AP["Apply, last recorded, pending dropped<br/>xr-client/rust/src/memory_route.rs:641"]
    AR["after_reload re-offers the pending frame once the cloud has reloaded<br/>a second mismatch drops it<br/>xr-client/rust/src/memory_route.rs:646"]
    IN --> Q1
    Q1 -->|no| ST
    Q1 -->|yes| Q2
    Q2 -->|yes| CL
    Q2 -->|no| Q3
    Q3 -->|no| Q4
    Q4 -->|yes| DR1
    Q4 -->|no| RL
    Q3 -->|yes| Q5
    Q5 -->|yes| DR2
    Q5 -->|no| AP
    RL --> AR
    AR --> IN
```

**What it shows.** The whole decision, in the order the code takes it. `last` moves only on a Clear or an Apply, never on a Drop, Reload or Stale, so a bad frame cannot lock out a good one with the same stamp. **Why it is this way.** Ordering is `(sentAt, seq)` so a desktop page reload, which restarts `seq` at 0 with a later `sentAt`, is accepted while a re-delivered older frame is not (xr-client/rust/src/memory_route.rs:585). The Reload verdict hands the cloud one chance to catch up with a snapshot the desktop already holds.

**Invariant:** `RouteFrame.path` rows are range-checked against the loaded row count before an Apply (xr-client/rust/src/memory_route.rs:633), hit points are checked only for finiteness.

`reloaded_for` is set at the first Reload and cleared on Apply and on Clear (xr-client/rust/src/memory_route.rs:620, xr-client/rust/src/memory_route.rs:640), so a later mismatch is a new episode and reloads again, while a second mismatch for the same id inside one episode is still dropped instead of reloading in a loop (VisionClaw `6ca5ff4b2`, 2026-10-18, estate register ER-26).

**Open:** `sent_at` is Unix milliseconds on both paths, the relay's `sentAt` passed through unrebased and the headset's own `Time.get_unix_time_from_system()` (xr-client/scripts/memory_cloud_layer.gd:491), so a headset clock ahead of the desktop's makes desktop frames `Stale` until the desktop catches up and a clock behind does the reverse; nothing states the unit or bounds the skew.

## VC-38.9 From a text frame to per-frame buffers

```mermaid
sequenceDiagram
    autonumber
    participant GS as _on_graph_text<br/>xr-client/scripts/graph_scene.gd:3029
    participant L as apply_route_json<br/>xr-client/scripts/memory_cloud_layer.gd:473
    participant MR as offer_json<br/>xr-client/rust/src/memory_route.rs:1664
    participant G as RouteGate offer<br/>xr-client/rust/src/memory_route.rs:606
    participant AR as ActiveRoute set<br/>xr-client/rust/src/memory_route.rs:1455
    participant T as tick<br/>xr-client/rust/src/memory_route.rs:1830

    GS->>L: memoryRoute frame, the layer gates it xr-client/scripts/graph_scene.gd:3049
    L->>MR: offer_json(json, loaded snapshot id, positions) xr-client/scripts/memory_cloud_layer.gd:476
    MR->>MR: parse_route, a failure returns error and sets last_error xr-client/rust/src/memory_route.rs:516
    MR->>G: offer(frame, loaded, rows)
    G-->>MR: Apply Clear Stale Reload or Drop
    MR->>AR: on Apply, set(frame, positions) xr-client/rust/src/memory_route.rs:2011
    AR->>AR: sample_frame, restart the trace and cue clocks, unless it is a repeat of the same relayed route xr-client/rust/src/memory_route.rs:1459
    AR-->>MR: true for a new route, false for a refresh
    MR-->>L: verdict string apply, refresh, clear, stale, reload or drop xr-client/rust/src/memory_route.rs:2008
    alt apply
        L->>L: build the ArrayMesh from mesh_arrays, set_keep(keep_rows), tick(0), emit route_changed xr-client/scripts/memory_cloud_layer.gd:569
    else refresh
        L->>L: set_keep and stats only, the trace is not replayed xr-client/scripts/memory_cloud_layer.gd:539
    else reload
        L->>L: reload the cloud, then after_reload re-offers xr-client/scripts/memory_cloud_layer.gd:533
    end
    loop every frame while the route is active
        L->>T: tick(delta, glow, beat phase, beat pulse, reduced motion) xr-client/scripts/memory_cloud_layer.gd:630
        T-->>L: head_u, comet_u, tail_u, glow, path_t, answer, walk, walk window, walk width, walk glow, cue xr-client/rust/src/memory_route.rs:1852
        L->>L: bead_buffer and ring_buffer written to two MultiMeshes xr-client/scripts/memory_cloud_layer.gd:640
    end
```

**What it shows.** The text frame's whole life: gate, sample, mesh once, then two small buffers and a handful of shader parameters per frame. Geometry is built once per route; the animation is shader parameters plus instance buffers, never a rebuilt mesh. **Why it is this way.** Glow is additive geometry only, no post-process (XR-client Invariant 2, xr-client/rust/src/memory_route.rs:20), and the mesh stores each vertex's route parameter so the shader can reveal the trace without a per-frame rebuild (xr-client/rust/src/memory_route.rs:14).

**Drift:** the doc comment on `tick` lists the Dictionary as "head_u, tail_u, glow, path_t, answer, comet (bool)" (xr-client/rust/src/memory_route.rs:1828) while the code sets `comet_u` as a float and four `walk_*` keys plus `cue` (xr-client/rust/src/memory_route.rs:1852).

**Invariant:** the headset draws the four line layers of the tube at 0.1 of the desktop thickness and brightness and the comet tail at the walk factors 1.0 and 0.5, a named divergence from the desktop (operator decision of 2026-10-08) applied on top of the desktop-look mesh and pinned by a drift test on the desktop constants (xr-client/rust/src/memory_route.rs:74, xr-client/rust/src/memory_route.rs:875, xr-client/rust/src/memory_route.rs:894).

## VC-38.10 Route animation - what animate computes from elapsed time

```mermaid
flowchart TB
    EL["el seconds since the route arrived, advanced by dt clamped to 0..0.1 s<br/>xr-client/rust/src/memory_route.rs:1536"]
    RM{"reduced motion?"}
    R1["path_t 1, no comet, no pulse, no beat, walk 0<br/>xr-client/rust/src/memory_route.rs:1071 xr-client/rust/src/memory_route.rs:1067"]
    T1["tracing: path_t is el over 2.8 s, head is eased in-out times the last sample, comet at the head<br/>xr-client/rust/src/memory_route.rs:51 xr-client/rust/src/memory_route.rs:1074"]
    T2["converged: comet loops every COMET_LOOP_S at size 0.7, answer ring appears, pulse runs<br/>from the beat phase when locked, else every 0.9 s<br/>xr-client/rust/src/memory_route.rs:1087 xr-client/rust/src/memory_route.rs:1091 xr-client/rust/src/memory_route.rs:54"]
    W1["walk: 1 while tracing, then decays to 0 over 0.8 s after PATH_DUR<br/>xr-client/rust/src/memory_route.rs:1102 xr-client/rust/src/memory_route.rs:92"]
    G1["glow times 1 plus 0.2 times the beat pulse when a beat is locked<br/>xr-client/rust/src/memory_route.rs:1078"]
    CU["framing cue, a separate clock: lives 4.5 s, alpha is the lesser of a 0.25 s fade-in and a 1 s fade-out<br/>reduced motion keeps the fade and drops the march and the growth<br/>xr-client/rust/src/memory_route.rs:1134 xr-client/rust/src/memory_route.rs:185"]
    EL --> RM
    RM -->|yes| R1
    RM -->|no| T1 --> T2
    T1 --> W1
    T2 --> W1
    T1 --> G1
    EL --> CU
    INV["INVARIANT: no RNG, no wall clock, no thread in the animation, a function of dt, glow, the beat inputs and reduced motion only<br/>xr-client/rust/src/memory_route.rs:1059 xr-client/rust/src/memory_route.rs:1830"]
    G1 --> INV
```

**What it shows.** Every animated quantity of the route and the clock it hangs on. A new route restarts both clocks; a repeat of the same relayed route does not, so the trace is not replayed every time the desktop re-sends for late joiners. **Why it is this way.** Reduced motion shows the converged route with no moving part at all (ADR-2107, xr-client/rust/src/memory_route.rs:1057), which is also the default in the route object, the beat script and the layer (xr-client/rust/src/memory_route.rs:1596, xr-client/scripts/beat_pulse.gd:43, xr-client/scripts/memory_cloud_layer.gd:72).

## VC-38.11 Route geometry and the triangle budget

```mermaid
classDiagram
    class TubeMesh {
        five layers sheath outer, sheath inner, gradient body, core, comet tail
        RADIAL 6 vertices per ring
        per vertex centre in VERTEX, ring direction in NORMAL
        route parameter in UV x, layer radius in UV2 x
        one ArrayMesh surface, 13 slot array, unused slots nil
        def : xr-client/rust/src/memory_route.rs:810 xr-client/rust/src/memory_route.rs:136 xr-client/rust/src/memory_route.rs:229
    }
    class RouteSamples {
        pts knots answer
        quadratic Bezier bowed out of the cloud, 16 samples per hop at most
        a 12-hop route keeps 10, total at most ROUTE_RING_CAP 121
        def : xr-client/rust/src/memory_route.rs:697 xr-client/rust/src/memory_route.rs:142
    }
    class InstanceBuffers {
        STRIDE 16 floats, 3x4 transform then rgba
        beads 2 per knot plus 2 plus 12 guide dots
        rings 3 plus sidecar marks plus ghosts
        def : xr-client/rust/src/memory_route.rs:153 xr-client/rust/src/memory_route.rs:976 xr-client/rust/src/memory_route.rs:982
    }
    class Budget {
        tube triangles 5 times samples minus 1 times 6 times 2
        beads one triangle each, rings two
        ring_cap_for_budget picks the largest cap that fits, else one sample per hop
        def : xr-client/rust/src/memory_route.rs:969 xr-client/rust/src/memory_route.rs:1014
    }
    class RenderPriority {
        route 10, held 15, overlay 20, compile-time ordered
        route draws without depth test so opaque nodes cannot hide it
        def : xr-client/rust/src/memory_route.rs:163
    }
    RouteSamples ..> TubeMesh : build_route_mesh
    RouteSamples ..> InstanceBuffers : bead_buffer and ring_buffer
    Budget ..> RouteSamples : set_ring_cap resamples
    RenderPriority ..> TubeMesh
```

**What it shows.** What a route costs and how it is held under the frame budget. The frame-budget allocator computes `route_ring_cap` (xr-client/rust/src/frame_budget.rs:171), the layer passes it to `set_ring_cap` (xr-client/scripts/memory_cloud_layer.gd:933) and the route resamples and rebuilds its mesh when the cap changes. **Why it is this way.** Beads, halos and the comet are one-triangle camera-facing discs, the cloud sprite's shape, so a route stays cheap in stereo (xr-client/rust/src/memory_route.rs:146); the budget closure is `frame_budget`'s (xr-client/rust/src/frame_budget.rs:5).

**Debt:** `route_worst_triangles` is public with no caller outside tests (xr-client/rust/src/memory_route.rs:1024); the two `_desktop_look` functions are not dead, they are the production base layers wrapped by `build_route_mesh` and `bead_buffer` (xr-client/rust/src/memory_route.rs:876, xr-client/rust/src/memory_route.rs:1225).
## VC-38.12 Headset query - from the Memory Search page to a drawn route

```mermaid
sequenceDiagram
    autonumber
    participant MS as MemorySearch _run<br/>xr-client/scripts/memory_search.gd:183
    participant MQ as MemoryQuery adapter<br/>xr-client/rust/src/memory_query.rs:466
    participant SV as server /api/memory-cloud/query
    participant L as MemoryCloudLayer<br/>xr-client/scripts/memory_cloud_layer.gd:488
    participant MR as MemoryRoute offer_query_response<br/>xr-client/rust/src/memory_route.rs:1608

    MS->>MQ: request_body(text, k 50, namespace) xr-client/rust/src/memory_query.rs:540
    MQ-->>MS: JSON body or empty, trimmed, k clamped 1 to 50, text 1 to 2000 characters xr-client/rust/src/memory_query.rs:313
    MS->>SV: POST with the NIP-98 header, 15 s timeout, one request in flight xr-client/scripts/memory_search.gd:211
    SV-->>MS: snapshotId, embedModel, query text and position, sidecar results, method, tookMs
    MS->>MQ: ingest_response(body) xr-client/rust/src/memory_query.rs:559
    MQ-->>MS: caption and at most 8 hit rows with rank, key, namespace, score, row or minus one, position xr-client/rust/src/memory_query.rs:53
    MS->>L: apply_query_response(body) xr-client/scripts/memory_search.gd:233
    L->>MR: offer_query_response(body, loaded id, positions, now in Unix ms) xr-client/scripts/memory_cloud_layer.gd:492
    MR->>MR: parse_query_response a second time, then local_seq plus 1 xr-client/rust/src/memory_route.rs:1622
    MR->>MR: sidecar_route builds a RouteFrame with source SidecarTopK xr-client/rust/src/memory_query.rs:261
    MR->>MR: the same RouteGate and apply_decision as a relayed frame xr-client/rust/src/memory_route.rs:1625
    Note over MR: sent_at is the headset's own wall clock, so a later desktop relay replaces it only while the two clocks agree, sentAt is not rebased<br/>xr-client/rust/src/memory_route.rs:1605
```

**What it shows.** The headset's own search. The same JSON is parsed twice on purpose, once for the HUD list and once for the route, so neither depends on the other's state. **Why it is this way.** The headset never runs HNSW: the sidecar answers, and what is drawn is "query point to sidecar top-k in rank order", captioned as such so it is never passed off as a search traversal (xr-client/rust/src/memory_query.rs:19), ADR-2136.

**Drift:** `K := 50` in `memory_search.gd` is a hand copy of `HEADSET_K` (xr-client/scripts/memory_search.gd:25, xr-client/rust/src/memory_query.rs:51); the script's comment names `HEADSET_K` as the value it mirrors, and nothing enforces it.

**Debt:** `MemoryQuery.answer` is stored at `ingest_response` and never read back (xr-client/rust/src/memory_query.rs:582); `last_error` is not cleared by a later success.

## VC-38.13 sidecar_route - the decision that makes a route out of ranked hits

```mermaid
flowchart TB
    A0["QueryAnswer: hits in rank order, optional query position<br/>sampled_rows are the hits whose sampleIndex is a valid row<br/>xr-client/rust/src/memory_query.rs:154 xr-client/rust/src/memory_query.rs:177"]
    A1{"query position present?"}
    P1["path is the sampled rows in rank order, needs one<br/>xr-client/rust/src/memory_query.rs:269"]
    P2["path is the sampled rows reversed so the top hit is last, needs two<br/>xr-client/rust/src/memory_query.rs:270"]
    P3["too few points: empty path, which clears the previous route unless placed hits exist<br/>xr-client/rust/src/memory_query.rs:271"]
    T1["over 64 rows: with a position keep the head, without keep the tail, the answer end either way<br/>xr-client/rust/src/memory_query.rs:276 xr-client/rust/src/memory_query.rs:278"]
    H1["with a query position only, any placed hit overrides: every hit with a position draws, path keeps only the sampled rows lit<br/>xr-client/rust/src/memory_query.rs:281"]
    O1["origin is the query position, only when something draws<br/>xr-client/rust/src/memory_query.rs:288"]
    S1["sidecar marks: sampled rows, at most 64, stats total hits, in sample, agree<br/>xr-client/rust/src/memory_query.rs:289 xr-client/rust/src/memory_query.rs:291"]
    F1["RouteFrame: source SidecarTopK, query text cut to 120 characters, hit_points and hit_sampled from placed<br/>xr-client/rust/src/memory_query.rs:304"]
    A0 --> A1
    A1 -->|yes| P1
    A1 -->|no| P2
    P1 --> T1
    P2 --> T1
    A1 -->|too few| P3
    T1 --> H1 --> O1 --> S1 --> F1
    PARSE["parse_query_response: hits cut to 50, rank is index plus 1, non-finite score 0, a bad row makes the hit unsampled not the response invalid<br/>xr-client/rust/src/memory_query.rs:218 xr-client/rust/src/memory_query.rs:228 xr-client/rust/src/memory_query.rs:211"]
    PARSE --> A0
```

**What it shows.** How ranked hits become geometry. With the query's own point the line runs from it through the hits; without it the hits alone make a line, top hit last, so the answer ring still lands on the best match. **Why it is this way.** ADR-2136 added the query point to the response so the headset can draw from the question rather than from an arbitrary first hit (xr-client/rust/src/memory_query.rs:11).

**Invariant:** a hit's row is accepted only if it is a non-negative integer at most `u32::MAX`; an out-of-range `sampleIndex` leaves the response valid and the hit unsampled (xr-client/rust/src/memory_query.rs:211).

## VC-38.14 Beat clock frames - ping, pong and the relayed clock

```mermaid
sequenceDiagram
    autonumber
    participant GS as _on_graph_text<br/>xr-client/scripts/graph_scene.gd:3029
    participant BG as BeatPulse script on_text<br/>xr-client/scripts/beat_pulse.gd:95
    participant BP as BeatPulse handle_text<br/>xr-client/rust/src/pulse.rs:65
    participant BS as BeatSync and ClockOffset<br/>xr-client/rust/src/beat.rs:397
    participant SV as server relay

    BG->>BP: make_ping every 2 s xr-client/scripts/beat_pulse.gd:32
    BP-->>BG: ping JSON stamped with local epoch ms xr-client/rust/src/beat.rs:265
    BG->>SV: send_text(ping) on the graph socket xr-client/rust/src/binary_protocol.rs:1174
    SV-->>GS: pong echoing the timestamp with serverTime
    GS->>BG: _beat.on_text(json, type) after the settings router xr-client/scripts/graph_scene.gd:3039
    BG->>BP: handle_text(json) for beatClock and pong only xr-client/scripts/beat_pulse.gd:99
    alt beatClock frame
        BP->>BS: parse_beat_clock_frame then on_remote(frame, now) xr-client/rust/src/beat.rs:236 xr-client/rust/src/beat.rs:433
        Note over BS: bpm 40 to 220, phaseAt at least 0, confidence 0 to 1, source off, file, tap or spotify, mic is refused<br/>xr-client/rust/src/beat.rs:245 xr-client/rust/src/beat.rs:33
        BS->>BS: first frame with serverTime and no offset calls seed_provisional, a one-way estimate kept OUT of the sample window, no rtt xr-client/rust/src/beat.rs:435, xr-client/rust/src/beat.rs:333
    else pong frame
        BP->>BS: parse_pong then offset.add_sample(sent, server, now), which also clears any provisional seed xr-client/rust/src/beat.rs:274 xr-client/rust/src/beat.rs:308
        Note over BS: rtt must be finite, non-negative and at most 2000 ms<br/>offset is the sample with the smallest rtt of the last 8, server minus the trip midpoint<br/>xr-client/rust/src/beat.rs:313 xr-client/rust/src/beat.rs:326
    end
    BP-->>BG: the frame type consumed, or an empty string so the scene routes it on
```

**What it shows.** Two text frames are consumed by the beat module and the rest fall through. The offset estimator is the NTP minimum-delay filter: the least-delayed of the last eight round trips alone sets the clock. **Why it is this way.** ADR-2134 relays the desktop explorer's beat clock to the same user's sessions at up to 4 Hz (src/handlers/socket_flow_handler/message_routing.rs:64); the headset keeps its own offset because its wall clock and the server's differ.

The provisional seed from the first relayed frame is a separate one-way offset, not a window sample: `seed_provisional` is refused once any round trip has landed, the first real pong discards it, and `rtt_ms` stays unset until then (xr-client/rust/src/beat.rs:333, xr-client/rust/src/beat.rs:348). Stored in the window with `rtt 0` it would have won the minimum-delay filter over every real ping for eight pongs, roughly sixteen seconds (VisionClaw `6ca5ff4b2`, estate register ER-26).

## VC-38.15 BeatSync arbitration - which clock drives the pulse

```mermaid
flowchart TB
    ST["active(local_ms) picks one clock and converts its phase to local time<br/>xr-client/rust/src/beat.rs:504"]
    M{"mic listening and locked?"}
    MIC["Mic clock, phase already local<br/>xr-client/rust/src/beat.rs:507"]
    T{"a tap clock exists and the last tap is not older than the last change of the relayed clock?"}
    TAP["Tap clock<br/>xr-client/rust/src/beat.rs:510"]
    R{"relayed clock fresh: source not off and received within 6500 ms?"}
    REM["Remote clock, phase minus the ping offset when one exists and phaseAt is positive<br/>xr-client/rust/src/beat.rs:498 xr-client/rust/src/beat.rs:368 xr-client/rust/src/beat.rs:517"]
    TL["last tap clock, even stale<br/>xr-client/rust/src/beat.rs:528"]
    NONE["None, OFF_BEAT, pulse 0"]
    ST --> M
    M -->|yes| MIC
    M -->|no| T
    T -->|yes| TAP
    T -->|no| R
    R -->|yes| REM
    R -->|no| TL
    TL -->|no tap clock| NONE
    OUT["beat_at evaluates the clock: pulse is exp of minus 6 times beat phase, bar pulse only on every fourth beat<br/>xr-client/rust/src/beat.rs:191 xr-client/rust/src/beat.rs:204"]
    MIC --> OUT
    TAP --> OUT
    REM --> OUT
    TL --> OUT
    RED["pulse_intensity is exactly 0 under reduced motion<br/>xr-client/rust/src/beat.rs:543"]
    OUT --> RED
```

**What it shows.** The precedence: mic, then tap, then the desktop's clock, then a stale tap clock, then nothing. A tap wins until the desktop's clock actually changes (a different source, a different bpm, or a phase shifted by more than half a millisecond); a heartbeat that repeats the same clock does not count as a change. **Why it is this way.** A listening mic that has locked is the strongest evidence of what the room is playing; a tap is a person's deliberate act; the desktop is the default. A relayed clock with source `off` is stored but never fresh.

**Invariant:** a tap sequence resets after 2 500 ms, keeps nine timestamps, yields no tempo under three taps, and clamps the reading to 40..220 bpm (xr-client/rust/src/beat.rs:74, xr-client/rust/src/beat.rs:75, xr-client/rust/src/beat.rs:76).

**Open:** on the first one or two taps ever, the tap clock falls back to the default 120 bpm at confidence 0 (xr-client/rust/src/beat.rs:466); after an earlier reading it keeps that bpm and confidence with the phase reset (xr-client/rust/src/beat.rs:471), and a tap clock keeps winning arbitration until the desktop's clock changes; nothing records whether that is intended.

## VC-38.16 Mic tempo state - off, listening, searching, locked

```mermaid
stateDiagram-v2
    [*] --> Off
    Off --> Listening: set_mic_enabled true
    Listening --> Searching: first tempo estimate lands
    Searching --> Locked: two consecutive estimates at confidence 0.6 or more within 2 bpm
    Locked --> Searching: no confirming estimate for 8 seconds
    Listening --> Off: set_mic_enabled false
    Searching --> Off: set_mic_enabled false
    Locked --> Off: set_mic_enabled false
    note right of Locked
        phase_at is window start plus offset plus latency, local epoch ms,
        latency_ms is never set outside beat.rs
    end note
```

**What it shows.** The four states the HUD can report for the opt-in mic; `Locked` needs two agreeing confident estimates and lapses eight seconds after the last confirmation. **Why it is this way.** A single confident estimate could be a coincidence in a noisy room, so the lock requires the previous estimate to agree (xr-client/rust/src/beat.rs:820, xr-client/rust/src/beat.rs:975).

## VC-38.17 Mic analysis path - window, worker thread, lock

```mermaid
sequenceDiagram
    autonumber
    participant GD as beat_pulse.gd _process<br/>xr-client/scripts/beat_pulse.gd:238
    participant BP as BeatPulse process<br/>xr-client/rust/src/pulse.rs:130
    participant MB as MicBeat<br/>xr-client/rust/src/beat.rs:854
    participant W as worker thread<br/>xr-client/rust/src/pulse.rs:142

    GD->>BP: push_mic_frames(stereo frames, mix_rate) xr-client/rust/src/pulse.rs:115
    BP->>MB: mono is half of left plus right, push(mono, rate, now) xr-client/rust/src/pulse.rs:122
    MB->>MB: box-decimate to about 11025 Hz, rolling window of 8 s, in memory only xr-client/rust/src/beat.rs:822 xr-client/rust/src/beat.rs:558
    GD->>BP: process() every frame xr-client/scripts/beat_pulse.gd:250
    BP->>BP: collect a finished analysis with try_recv, apply_estimate xr-client/rust/src/pulse.rs:133
    alt nothing pending and the window holds 4 s and 2 s have passed
        BP->>MB: take_snapshot xr-client/rust/src/beat.rs:957
        BP->>W: spawn, move the samples in xr-client/rust/src/pulse.rs:141
        W-->>BP: analyse_window result over mpsc, detached, a dropped receiver fails the send silently xr-client/rust/src/beat.rs:1033 xr-client/rust/src/pulse.rs:143
    end
    BP->>MB: lock(now), then sync.set_mic_clock xr-client/rust/src/pulse.rs:148
    Note over MB,W: onset envelope of 256-sample frames hopping 128, positive flux, local-mean removal<br/>then autocorrelation with a log-normal prior at 118 bpm, comb fit and phase refinement<br/>70 to 180 bpm xr-client/rust/src/beat.rs:573 xr-client/rust/src/beat.rs:665 xr-client/rust/src/beat.rs:795
```

**What it shows.** The privacy-relevant half of the beat (the state machine is VC-38.16). Samples live in one rolling window of at most eight seconds and one analysis snapshot; turning the mic off clears the window, the accumulator, the estimate and the lock at once (xr-client/rust/src/beat.rs:896), and drops the pending receiver (xr-client/rust/src/pulse.rs:104). **Why it is this way.** Mic samples exist only in memory and are never written or sent (`xr-client/rust/src/pulse.rs:12-13`); analysis runs off the main thread because the autocorrelation search is the one expensive step.

**Invariant:** `MicBeat::push` is a no-op when disabled, empty, or given a NaN or non-positive rate (xr-client/rust/src/beat.rs:921).

**Debt:** the worker is detached with no handle and no cancellation, so a disabled mic costs one finished analysis before its result is discarded (xr-client/rust/src/pulse.rs:142); `decimate` is public with only test callers while `push` re-implements it inline (xr-client/rust/src/beat.rs:800, xr-client/rust/src/beat.rs:933); `last_pong_ms` is written and never read (xr-client/rust/src/pulse.rs:74).
## VC-38.18 memory_flash - from a text frame to a burst on the right point

```mermaid
sequenceDiagram
    autonumber
    participant BG as BeatPulse script on_memory_flash<br/>xr-client/scripts/beat_pulse.gd:107
    participant FC as MemoryFlashCodec parse<br/>xr-client/rust/src/pulse.rs:218
    participant SM as semantic parse_memory_flash<br/>xr-client/rust/src/semantic.rs:358
    participant BC as BinaryProtocolClient<br/>xr-client/rust/src/binary_protocol.rs:1494
    participant CL as MemoryCloudLayer<br/>xr-client/scripts/memory_cloud_layer.gd:430
    participant MB as MemoryBursts<br/>xr-client/scripts/memory_bursts.gd:121

    BG->>FC: parse(json) xr-client/scripts/beat_pulse.gd:108
    FC->>SM: parse_memory_flash(text) xr-client/rust/src/pulse.rs:220
    Note over SM: type must be memory_flash, data is one event or an array<br/>a non-object event is dropped, an event with empty key and empty namespace is dropped<br/>a string over 512 bytes becomes empty, at most 64 events kept after filtering<br/>xr-client/rust/src/semantic.rs:371 xr-client/rust/src/semantic.rs:392 xr-client/rust/src/semantic.rs:345
    SM-->>FC: key, namespace, action, timestamp, optional agentId
    FC->>FC: profile by action, hue-shifted colour by namespace xr-client/rust/src/pulse.rs:221
    FC-->>BG: descriptors with key, namespace, action, color, max_scale, duration, implode, rings, agent_id xr-client/rust/src/pulse.rs:234
    BG->>BC: record_memory_flash(agent_id) for every descriptor, whether or not bursts draw, minus one pulls every agent xr-client/scripts/beat_pulse.gd:113
    alt bursts off
        BG-->>BG: return xr-client/scripts/beat_pulse.gd:116
    else cloud shown and loaded
        BG->>CL: resolve_flash(key, namespace) xr-client/scripts/beat_pulse.gd:132
        CL-->>BG: rows by key, else up to 3 of the namespace, else none, an unmatched flash draws nothing
        BG->>MB: spawn_rows(rows, descriptor), restyles existing sprites, no new geometry xr-client/scripts/memory_bursts.gd:121
    else no cloud on screen
        BG->>MB: spawn at ambient_position(key, namespace, centre), a stable stand-in point xr-client/scripts/memory_bursts.gd:207
    end
```

**What it shows.** One flash, end to end. The colour and size say which verb it was (store green and wide, delete red and imploding, search cyan with three rings); the position says which memory. **Why it is this way.** The tables are hand copies of the desktop's `semanticEncoding.ts`; three tests guard drift, one parsing the TypeScript source, one comparing a parity fixture and one parsing the beam shader's colour uniforms (xr-client/rust/src/semantic.rs:12).

**Invariant:** the agent drift credit is applied before the bursts-enabled check, so a muted burst layer still moves the agent toward the memory vertex of the separated layout (xr-client/scripts/beat_pulse.gd:113, ADR-2135).

**Drift:** the doc comment on `MemoryFlashCodec::parse` lists the descriptor as `{key, namespace, action, color, max_scale, duration, implode, rings}` (xr-client/rust/src/pulse.rs:216) while the code also sets `agent_id` (xr-client/rust/src/pulse.rs:234).

**Debt:** `MemoryFlash.timestamp` is parsed and carried but never read downstream (xr-client/rust/src/semantic.rs:333); the string cap measures bytes, so a long non-ASCII key is refused sooner than 512 characters (xr-client/rust/src/semantic.rs:343).

## VC-38.19 semantic encoding tables - burst profile, hue shift, beam shape

```mermaid
flowchart TB
    subgraph verbs["MEMORY_ACTIONS, case-insensitive, unknown or empty falls back to access"]
        V0["store 0x39ff14, scale 4.6, 1.6 s, expand, 2 rings"]
        V1["retrieve 0x4fc3f7, scale 3.4, 1.8 s, expand, 1 ring"]
        V2["search 0x00fff7, scale 5.8, 2.2 s, expand, 3 rings"]
        V3["list 0xffd54f, scale 3.0, 1.4 s, expand, 1 ring"]
        V4["delete 0xff4444, scale 4.0, 1.4 s, implode, 1 ring"]
        V5["access 0x9ad6ff, scale 3.2, 1.6 s, expand, 1 ring"]
    end
    SRC["profile table xr-client/rust/src/semantic.rs:42 lookup xr-client/rust/src/semantic.rs:91"]
    HUE["namespace hue shift: hash of the UTF-16 code units, h times 31 plus unit, wrapping u32<br/>norm is h mod 1000 over 1000, shift is between minus 0.06 and plus 0.06 turns, empty namespace 0<br/>xr-client/rust/src/semantic.rs:118 xr-client/rust/src/semantic.rs:121"]
    COL["semantic_burst_color: sRGB to linear to HSL, add the shift to hue, back<br/>a zero shift returns the base colour unchanged<br/>xr-client/rust/src/semantic.rs:218"]
    subgraph beams["agent action beams, wire action type 0 to 5"]
        B0["Query r 0.5 to 0.5, 8 segments"]
        B1["Update r 1.0 to 1.0, 10"]
        B2["Create r 1.8 to 0.4, 12"]
        B3["Delete r 0.3 to 1.6, 10"]
        B4["Link r 1.3 to 1.3, 12"]
        B5["Transform r 0.9 to 0.9, 16"]
    end
    BEAM["beam_custom_rgb: code is the action type or 6 for unknown, then the two radii<br/>INSTANCE_CUSTOM alpha stays the agent status so the stride stays 16<br/>xr-client/rust/src/semantic.rs:314 consumed xr-client/rust/src/render_store.rs:2037"]
    verbs --> SRC --> HUE --> COL
    beams --> BEAM
    N1["Unknown shape falls back to Query, unknown colour to white<br/>xr-client/rust/src/semantic.rs:300 xr-client/rust/src/semantic.rs:249"]
    BEAM --> N1
```

**What it shows.** Every burst constant, the beam radii and the beam colour order (blue, yellow, green, red, purple, cyan, then white for unknown, xr-client/rust/src/semantic.rs:245). The verb picks colour, size, duration, direction and ring count; the namespace nudges the hue by a deterministic amount so two namespaces doing the same verb are told apart without losing the verb's colour. **Why it is this way.** The hash is the desktop's `charCodeAt` hash, taken over UTF-16 code units so non-ASCII namespaces match the desktop bit for bit (xr-client/rust/src/semantic.rs:105).

**Debt:** `AGENT_ACTION_NAMES`, `agent_action_color` and `MEMORY_ACTIONS` outside the module have no callers; the GPU counterpart of the beam colours is the shader's uniforms, not Rust (xr-client/rust/src/semantic.rs:241, xr-client/rust/src/semantic.rs:304).
