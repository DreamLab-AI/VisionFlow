---
id: VC-36
title: Godot + gdext OpenXR immersive client
area: visionclaw
governing:
  - ../project/docs/XR-client.md
  - ../project/docs/BASELINE-architecture.md
adrs: [ADR-2032, ADR-2033, ADR-2034, ADR-2035, ADR-2036, ADR-2039, ADR-2076, ADR-2079, ADR-2107, ADR-2108, ADR-2109, ADR-2133, ADR-2134, ADR-2135, ADR-2141, ADR-2142, ADR-2143, ADR-2186, ADR-2188, ADR-2189]
sources:
  - ../project/docs/XR-client.md
  - ../project/docs/BASELINE-architecture.md
  - ../project/xr-client/project.godot
  - ../project/xr-client/scripts/xr_boot.gd
  - ../project/xr-client/scripts/graph_scene.gd
  - ../project/xr-client/scripts/hud.gd
  - ../project/xr-client/scripts/hud_ia.gd
  - ../project/xr-client/scripts/label_budget.gd
  - ../project/xr-client/tests/unit/test_label_budget.gd
  - ../project/xr-client/scripts/xr_prefs.gd
  - ../project/xr-client/scripts/spatial_volume.gd
  - ../project/xr-client/scripts/highlight_writer.gd
  - ../project/xr-client/scripts/space_portals.gd
  - ../project/docs/adr/ADR-2188-the-xr-client-adopts-the-overview-first-contract-without-bloom.md
  - ../project/docs/adr/ADR-2189-desktop-and-xr-menus-share-one-seven-section-information-architecture.md
  - ../project/xr-client/data/ia.json
  - ../project/xr-client/tests/unit/test_hud_ia.gd
  - ../project/xr-client/openxr_action_map.tres
  - ../project/xr-client/rust/src/lib.rs
  - ../project/xr-client/rust/src/binary_protocol.rs
  - ../project/xr-client/rust/src/render_store.rs
  - ../project/xr-client/rust/src/transport.rs
  - ../project/xr-client/rust/src/presence.rs
  - ../project/xr-client/rust/src/signer.rs
  - ../project/xr-client/rust/src/webrtc_audio.rs
  - ../project/xr-client/rust/perf-regression/src/main.rs
  - ../project/xr-client/rust/Cargo.toml
  - ../project/src/actors/gpu/force_compute_actor.rs
  - ../project/src/handlers/layout_handler.rs
  - ../project/src/utils/auth.rs
  - ../project/src/main.rs
  - ../project/docker-compose.unified.yml
  - ../project/src/actors/presence_actor.rs
  - ../project/xr-client/rust/src/avatar_state.rs
  - ../project/xr-client/rust/src/gaze.rs
  - ../project/xr-client/rust/src/interaction.rs
  - ../project/xr-client/rust/src/lod.rs
  - ../project/xr-client/rust/src/proxemics.rs
  - ../project/xr-client/rust/src/selection.rs
  - ../project/xr-client/scenes/GraphScene.tscn
  - ../project/xr-client/materials/node_halo.gdshader
  - ../project/xr-client/scripts/spatial_environment.gd
  - ../project/xr-client/scripts/xr_theme.gd
  - ../project/xr-client/scripts/agent_avatar.gd
  - ../project/xr-client/materials/spatial_floor.gdshader
  - ../project/xr-client/tests/spatial_visual_fixture.gd
  - ../project/xr-client/tests/visual/hud_gallery.gd
  - ../project/xr-client/scripts/agent_choreography.gd
  - ../project/xr-client/scripts/agent_role.gd
  - ../project/xr-client/scripts/agent_effects.gd
  - ../project/xr-client/tests/unit/test_agent_choreography.gd
  - ../project/xr-client/tests/unit/test_write_denied_notice.gd
  - ../project/docs/adr/ADR-2109-xr-agent-embodiment-single-pose-owner-and-demo-via-ingest.md
  - ../project/xr-client/rust/src/render_store_pack.rs
  - ../project/xr-client/rust/src/frame_budget.rs
  - ../project/xr-client/rust/src/settings_sync.rs
  - ../project/xr-client/rust/src/hulls.rs
  - ../project/xr-client/rust/src/memory_cloud.rs
  - ../project/xr-client/rust/src/memory_route.rs
  - ../project/xr-client/rust/src/memory_query.rs
  - ../project/xr-client/rust/src/cloud_frame.rs
  - ../project/xr-client/rust/src/pulse.rs
  - ../project/xr-client/scripts/node_lod.gd
  - ../project/xr-client/scripts/frame_budget_pass.gd
  - ../project/xr-client/scripts/graph_parity.gd
  - ../project/xr-client/scripts/beat_pulse.gd
  - ../project/xr-client/scripts/memory_cloud_layer.gd
  - ../project/xr-client/scripts/memory_search.gd
  - ../project/src/actors/gpu/display_projection.rs
  - ../project/crates/visionclaw-tri-layout/src/lib.rs
  - ../project/crates/visionclaw-tri-layout/src/drift.rs
  - ../project/src/handlers/socket_flow_handler/filter_auth.rs
  - ../project/src/settings/auth_extractor.rs
  - ../project/src/handlers/memory_cloud_handler.rs
  - ../project/src/handlers/speech_socket_handler.rs
  - ../project/docs/adr/ADR-2036-defensive-openxr-boot.md
  - ../project/xr-client/scripts/backend_connection.gd
  - ../project/xr-client/scripts/connection_volume_source.gd
  - ../project/src/handlers/socket_flow_handler/http_handler.rs
  - ../project/src/handlers/socket_flow_handler/position_updates.rs
  - ../project/nginx.dev.conf
  - ../project/src/handlers/api_handler/graph/mod.rs
  - ../project/src/handlers/graph_access.rs
  - ../project/src/services/ontology_visibility.rs
  - ../project/src/settings/models.rs
  - ../project/src/actors/client_coordinator_actor.rs
  - ../project/client/src/features/graph/explore/nodeFilter.ts
  - ../project/xr-client/scripts/backend_caps.gd
verified_commit: {visionclaw: 010e1109bd78f253f0d82382bb94908b5023b1e9}
---

## For developers

Restamped at 86cdd76: the 2026-10-30 note that the spatial-volume merge 772039bd1 (ADR-2158, ADR-2159, ADR-2164) had not been re-read is closed: VC-36.3, VC-36.4, VC-36.7, VC-36.19, VC-36.22 and VC-36.24 are redrawn, and VC-36.23 has its gate line corrected, against the per-source `BackendConnection`, the per-volume `ConnectionVolumeSource` packing and the edge-run choreography at this stamp. The connections, volumes, portals, grabs and link layer themselves are drawn in VC-40, VC-44, VC-45, VC-46, VC-47 and VC-53, not here.

The [visual-budget and engine-gate audit](51-xr-visual-budget-and-engine-gates.md) qualifies the later extras allowance and 4.8-beta2 target; older engine and allocator descriptions below retain their historical pins.

Restamped at a950f4d80 (2026-11-06): only docs moved. The ADR-2188 headset receipt (`docs/adr/ADR-2188-the-xr-client-adopts-the-overview-first-contract-without-bloom.md:52`, `docs/XR-client.md:8`) resolves the live-run Open, ADR-2189's amendment resolves its acceptance-boundary Drift, and the XR-client.md changelog entry moves later citations down one line.

Restamped at 809a859aa (2026-11-06): the XR client's own code did not change. VC-36.10 drops the retired "nginx does not proxy presence" note. VC-36.18 now draws the caller-view gate on the pattern query. The new VC-36.31 shows what the headset's filter and query panel receive after the server visibility sweep, and why the ontology deny-list never reaches it. The `filter_auth.rs`, `force_compute_actor.rs`, `ia.json` and BASELINE citations moved with their symbols.

Restamped at ec785afa9 (2026-11-06): VC-36.3 is redrawn for the signed WebSocket upgrade of `6b811f7a1`, VC-36.8 for the ADR-2189 seven-section HUD, and VC-36.29 and VC-36.30 are new for the ADR-2188 label budget and the headset preferences with Fit. The ADR-2188 overview tier, edge limit and highlight mask are drawn in [VC-51](51-xr-visual-budget-and-engine-gates.md). Evidence at this stamp is headless GUT and Rust units; shader compilation and headset legibility were not checked from the container.

## For the business

The headset is a second way into the same knowledge graph: a person in a VIVE or Quest sees the graph, the memory cloud and the agents at room scale, and can grab and pin nodes when the server lets them write. Since 6 November its menu uses the same seven sections as the desktop, labels go to the most important nearby nodes, and its own settings survive a restart. What is still unproven is the experience in the headset itself: legibility, comfort and the 90 fps floor are measured by a person wearing it, not by the tests here.

## VC-36.1 Boot — OpenXR init, capability probe, deferred scene swap (ADR-2036)

```mermaid
sequenceDiagram
    autonumber
    participant G as Godot runtime<br/>xr-client/project.godot:15
    participant B as xr_boot.gd _ready<br/>xr-client/scripts/xr_boot.gd:9
    participant XS as XRServer<br/>godot builtin
    participant XI as OpenXR XRInterface<br/>vendor addon
    participant P as _probe_capabilities<br/>xr-client/scripts/xr_boot.gd:25
    participant T as _transition_to_graph_scene<br/>xr-client/scripts/xr_boot.gd:53
    participant E as _show_error ErrorOverlay<br/>xr-client/scripts/xr_boot.gd:64

    Note over G: run/main_scene=res://scenes/XRBoot.tscn<br/>project.godot:15 - openxr/enabled=true project.godot:64
    G->>B: _ready()
    B->>XS: find_interface("OpenXR")
    alt interface null
        B->>E: "OpenXR runtime not present."
        Note right of E: push_error + Label3D<br/>xr_boot.gd:64-67 - both error branches return from _ready,<br/>no use_xr, probe or scene swap xr_boot.gd:13 xr_boot.gd:17
    else present but not initialized
        B->>XI: initialize()
        alt returns false
            B->>E: "OpenXR initialise() returned false."
        end
    end
    B->>G: get_viewport().use_xr = true
    B->>P: _probe_capabilities(xr_interface)
    P->>XI: has_method("get_capabilities")
    opt method absent
        P-->>P: warn "capability queries unavailable"
        Note right of P: hand_tracking_available stays false so the<br/>controller-fallback warning also fires xr_boot.gd:32-38
    end
    P->>XI: is_passthrough_supported()
    opt false
        P-->>P: warn "proceeding without passthrough overlay"
    end
    P->>XI: get_capabilities()
    Note over P: XRInterface.XR_HAND_TRACKING = 16 (bit 4)<br/>caps & 16 - xr_boot.gd:34-36
    opt hand tracking bit clear
        P-->>P: warn "falling back to controller input"
    end
    P->>XI: has_method guard, then is_eye_gaze_interaction_supported() xr_boot.gd:47-48
    Note over P,XI: ADR-2036 INVARIANT: eye gaze is only QUERIED.<br/>The action map carries no eye-gaze profile since ER-13, so even a supporting device runs head gaze (Quest 3 = false anyway).<br/>xr_boot.gd:40-50
    opt unsupported or method absent
        P-->>P: warn "head-gaze primary (extension left disabled)"
    end
    P-->>B: _warnings
    opt warnings non-empty
        B->>G: push_warning(joined warnings)
    end
    B->>T: _transition_to_graph_scene()
    T->>T: load("res://scenes/GraphScene.tscn")
    alt load null
        T->>E: "GraphScene.tscn missing."
    else
        T->>G: change_scene_to_packed.call_deferred(graph_scene)
        Note over T,G: ADR-2036 INVARIANT: DEFERRED. A synchronous swap during<br/>_ready trips "Parent node is busy adding/removing children"<br/>while the OpenXR vendor addon is still adding XR nodes<br/>(Godot upstream hazard #113717). xr_boot.gd:58-61
    end
```

## VC-36.2 Renderer and XR project configuration (ADR-2032, forced Compatibility)

```mermaid
flowchart TB
    subgraph cfg["xr-client/project.godot"]
        A["config/features = PackedStringArray('4.3','Forward Mobile')<br/>project.godot:16"]
        R["renderer/rendering_method = 'gl_compatibility'<br/>project.godot:56"]
        RM["renderer/rendering_method.mobile = 'mobile'<br/>project.godot:57"]
        M["anti_aliasing/quality/msaa_3d = 0<br/>project.godot:59"]
        H["viewport/hdr_2d = false<br/>project.godot:60"]
        PH["common/physics_ticks_per_second = 90<br/>project.godot:51"]
        MS["common/max_physics_steps_per_frame = 8<br/>project.godot:52"]
        AL["[autoload] section is EMPTY<br/>project.godot:24"]
    end
    subgraph xr["[xr] block"]
        X1["openxr/enabled = true<br/>project.godot:64"]
        X2["openxr/default_action_map = res://openxr_action_map.tres<br/>project.godot:65"]
        X3["openxr/form_factor = 0 (HMD)<br/>project.godot:66"]
        X4["openxr/view_configuration = 1 (stereo)<br/>project.godot:67"]
        X5["openxr/reference_space = 2 (stage)<br/>project.godot:68"]
        X6["openxr/environment_blend_mode = 0 (opaque)<br/>project.godot:69"]
        X7["openxr/foveation_level = 0 - foveation_dynamic = false<br/>project.godot:70-71"]
        X8["openxr/submit_depth_buffer = true<br/>project.godot:72"]
        X9["openxr/startup_alert = false<br/>project.godot:73"]
    end
    subgraph inv["ADR-2032 consequences that travel with Compatibility"]
        C1["Glow/bloom OFF in WorldEnvironment<br/>glow blanks the second eye under Compat multiview"]
        C2["NVIDIA 580 open driver pinned - 610 fails the GL multiview second eye"]
        C3["Native X11 only - Wayland never brought up"]
        C4["Linear tonemap"]
    end
    cfg ~~~ xr
    xr ~~~ inv
    R --> C1
    R --> C2
    R --> C3
    R --> C4
    A -. "DOC-DRIFT" .-> D1
    D1["OPEN by design: project.godot:16 declares Godot 4.3 / Forward Mobile.<br/>The only build that has ever rendered on a headset is 4.6.1-stable on<br/>Compatibility. The file header now says so in text (project.godot:3-6) and<br/>leaves the array alone: config/features is EDITOR-MANAGED metadata, and its<br/>renderer tag is the Quest renderer decision (audit WP0, Invariant 1), so a hand<br/>edit would pre-empt it. docs/XR-client.md:974 and README:15-18 say the same.<br/>Header corrected 2026-10-07, assessed 2026-09-05 at ADR-2079 scope review"]
    AUD["[audio] driver/enable_input = true<br/>project.godot:22 - the engine may open an input device;<br/>capture itself is opt-in per session, docs/XR-client.md:1250"]
    RM -. "unexercised" .-> D2
    D2["DIVERGENCE: the .mobile override targets the Quest 3 APK,<br/>which is UNBUILT - no Android NDK is provisioned.<br/>90fps was measured only on VIVE Pro + dual RTX 6000.<br/>docs/BASELINE-architecture.md:279-280"]
    RM -. "TENSION" .-> D3
    D3["TENSION: project.godot:2 header still calls the Quest 3 APK the ship<br/>target, while xr-client/scripts/graph_scene.gd:41-42 reworded Quest 3<br/>standalone as a PLANNED target and names the Vive/SteamVR rig as the<br/>currently deployed one. project.godot:2, graph_scene.gd:41-42"]
```

## VC-36.3 Transport — signed /wss upgrade, topology request, position acknowledgements

```mermaid
sequenceDiagram
    autonumber
    participant GS as _connect_from_env<br/>xr-client/scripts/graph_scene.gd:2941
    participant CT as connect_to_server<br/>xr-client/scripts/graph_scene.gd:2978
    participant BC as BackendConnection, one per source<br/>xr-client/scripts/backend_connection.gd:59
    participant BP as BinaryProtocolClient (gdext)<br/>xr-client/rust/src/binary_protocol.rs:1191
    participant TR as spawn_graph_stream<br/>xr-client/rust/src/transport.rs:121
    participant SG as NostrSigner<br/>xr-client/rust/src/signer.rs:26
    participant SV as VisionClaw server /wss

    Note over GS,BC: _ready builds TWO connections, visionclaw with XR_NOSTR_SECRET and codescape with<br/>no secret, each with its own BinaryProtocolClient graph_scene.gd:1406-1416 (ADR-2159).<br/>GRAPH_STREAM_PATH="/wss" graph_scene.gd:88, PRESENCE_PATH="/ws/presence" graph_scene.gd:89
    GS->>GS: _vc_base() = XR_BACKEND_WS (default ws://localhost:4000), trailing "/" stripped graph_scene.gd:2958-2959
    GS->>CT: connect_to_server(base+"/wss", base+"/ws/presence", XR_ROOM_URN, XR_DISPLAY_NAME, XR_NOSTR_SECRET)
    Note over GS,CT: graph_scene.gd:2942-2949. The graph URL argument is only logged now: each<br/>connection owns its own URL (graph_scene.gd:2985-2987, backend_connection.gd:99-100).<br/>RESOLVED ADR-2076: no token argument - NIP-98 is the only graph-socket credential.<br/>RESOLVED ADR-2058 (see VC-32.1): the server's ?token= fallback is compiled out of release.<br/>DRIFT BASELINE-architecture.md still lists it as open (docs/BASELINE-architecture.md:406)
    CT->>BC: _connect_graph opens every connection graph_scene.gd:3009-3014
    BC->>BC: open() resets this connection's backoff, then _dial backend_connection.gd:108-112
    BC->>BP: close() then connect_to_url(graph_ws_url, secret), generation += 1<br/>backend_connection.gd:350-357
    BP->>BP: clear the store, swap in a fresh inbox, re-arm the freshness gate<br/>binary_protocol.rs:1315-1333
    BP->>TR: spawn_graph_stream(url, nostr_secret_hex, inbox) binary_protocol.rs:1334-1338
    TR->>TR: graph_upgrade_request builds the HTTP upgrade: same-authority Origin,<br/>Sec-WebSocket-Protocol "visionclaw" alone, or plus a signed NIP-98 "nostr." carrier when a secret is set<br/>transport.rs:49, transport.rs:68-70, transport.rs:71-82, transport.rs:84-87
    opt nostr_secret_hex non-empty (the codescape connection never has one)
        TR->>SG: nip98_http_authorization(signed URL, "GET") signer.rs:138
        SG-->>TR: Nostr header, re-encoded URL-safe as the carrier transport.rs:75-81
    end
    alt upgrade request cannot be built (bad scheme or malformed key)
        TR-->>BP: Disconnected, no connect attempted transport.rs:139-143
    end
    TR->>SV: connect_async_with_config(request, ws_config(), false) transport.rs:147
    Note over TR,SV: the server verifies the carrier before upgrading and answers 401 to a bad or replayed signature,<br/>then echoes the visionclaw subprotocol<br/>src/handlers/socket_flow_handler/http_handler.rs:151-153, src/handlers/socket_flow_handler/http_handler.rs:161-167, src/handlers/socket_flow_handler/http_handler.rs:448
    alt connect fails, including a refused signature
        TR-->>BP: Disconnected pushed into the inbox transport.rs:184 transport.rs:186
        Note right of BP: reconnect scheduled by the connection - see VC-36.4
    else connected
        TR->>SV: Text {"type":"requestInitialData"}
        Note over TR: GRAPH_REQUEST_INITIAL transport.rs:46
        Note over TR,SV: there is no position subscription, the server streams positions to every registered session and waits for the acknowledgements of this client (ADR-2141), the old GRAPH_SUBSCRIBE text frame is removed
        Note over TR: a failed send logs and falls through to Disconnected,<br/>Connected is never pushed transport.rs:155 transport.rs:186
        TR->>BP: Connected pushed only after the topology request is sent transport.rs:161
        BP-->>BC: poll emits connection_changed true binary_protocol.rs:1365-1367
        BC-->>GS: connected_changed(conn_id, true), backoff reset backend_connection.gd:378-383
        opt server selected NO subprotocol and a secret is set (the older first-message contract)
            TR->>SG: nip98_authenticate_json(url, "GET") signer.rs:126
            TR->>SV: Text authenticate transport.rs:165-171
            Note over TR,SV: never taken against VisionClaw, which always echoes visionclaw,<br/>and never by the codescape connection, which has no secret graph_scene.gd:1409
        end
        loop select over the socket and the outbound channel
            SV-->>TR: Binary becomes Frame transport.rs:199
            SV-->>TR: Text classified, frames of 64 KiB or more on the blocking pool<br/>transport.rs:204, transport.rs:239, transport.rs:249-253
            SV-->>TR: Ping answered with Pong transport.rs:207
            BP-->>TR: Outbound::Text (drag, pin, topology refetch) or Outbound::Binary (the 0x34 position acknowledgement) from the outbound channel transport.rs:91-98, transport.rs:217-223
            TR->>BP: push into inbox VecDeque transport.rs:266
        end
    end
    loop every physics tick
        GS->>BC: conn.poll(delta) for each connection, which calls client.poll() graph_scene.gd:3063-3064
        Note over BC: backend_connection.gd:134-136 - drain inbox, decode, emit signals
    end
```

**Invariant:** `Connected` reaches GDScript only after the topology request is on the wire, so a half-open socket never counts as live (`xr-client/rust/src/transport.rs:157-161`, `xr-client/scripts/graph_scene.gd:4632-4633`).
**Invariant:** wire ids are valid within one connection generation only; `_dial`, `close` and every topology load bump it (`xr-client/scripts/backend_connection.gd:17-19`, `xr-client/scripts/backend_connection.gd:360-362`), and the client swaps its inbox on reconnect so nothing from the old socket surfaces (`xr-client/rust/src/binary_protocol.rs:1321-1328`).
**Open (risk):** a refused upgrade signature (`src/handlers/socket_flow_handler/http_handler.rs:161-167`) reaches the headset only as a logged connect failure (`xr-client/rust/src/transport.rs:184`) followed by backoff retries, with no HUD notice naming the credential, unlike the REST write path of VC-36.15.
**Debt (loose end):** the first-message `authenticate` fallback (`xr-client/rust/src/transport.rs:162-180`) is justified by Codescape, but the Codescape connection is always built with an empty secret (`xr-client/scripts/graph_scene.gd:1409`) and VisionClaw always selects a subprotocol (`src/handlers/socket_flow_handler/http_handler.rs:10`), so no current connection takes it.

## VC-36.4 Per-socket independent reconnect backoff

```mermaid
stateDiagram-v2
    [*] --> Disconnected
    Disconnected --> Connecting: open() or retry timer fires
    Connecting --> Backoff: connect error or failed topology-request send, both surface as Disconnected
    Connecting --> Subscribed: requestInitialData sent, only then connection_changed true
    Subscribed --> Backoff: socket closed or recv error, surfaced as Disconnected
    Backoff --> Connecting: timer expires
    Backoff --> Backoff: presence only, kicked from presence calls the scheduler directly
    Subscribed --> Closed: close() - no reconnect
    note right of Subscribed
        The client has no Authenticated state.
        A configured identity signs the HTTP upgrade itself,
        so a refused signature never connects, transport.rs:49-89
    end note
    note right of Backoff
        GRAPH, per BackendConnection: backoff_delay backend_connection.gd:83-84
        min(RECONNECT_BASE_DELAY_SEC * 2^(max(attempts,1)-1), RECONNECT_MAX_DELAY_SEC)
        base 2.0 s backend_connection.gd:26, cap 60.0 s backend_connection.gd:27
        scheduled on a drop only while open backend_connection.gd:387-388, :372-375
        timer counted down in poll backend_connection.gd:137-142
        attempts reset on connected true backend_connection.gd:379-381
        PRESENCE, on the scene: _backoff_delay graph_scene.gd:4602-4606
        same 2.0 s / 60.0 s constants graph_scene.gd:37, :38
        presence timer graph_scene.gd:4611, kicked entry graph_scene.gd:5001
        INVARIANT every socket backs off INDEPENDENTLY: one
        BackendConnection per source plus the presence socket
        graph_scene.gd:893-895, backend_connection.gd:8-10
    end note
```

**Debt:** the presence socket keeps its own copy of the backoff (`xr-client/scripts/graph_scene.gd:4602-4613`) beside `BackendConnection.backoff_delay` (`xr-client/scripts/backend_connection.gd:83-84`). The two formulas differ only at `attempts` 0, which the presence path never passes.
**Drift:** the comment in `_on_client_connection_changed` says a drag-end pin persists across a reconnect (`xr-client/scripts/backend_connection.gd:385-386`), but a drag end records no pin since ADR-2172 F3 (`xr-client/scripts/backend_connection.gd:305-307`).

## VC-36.5 V3 / V5 wire records decoded by the gdext crate

```mermaid
classDiagram
    class FrameHeader {
        +u8 version_byte
        +PROTOCOL_V3 0x03 : binary_protocol.rs:25
        +PROTOCOL_V5 0x05 : binary_protocol.rs:29
        +HEADER_BYTES 1 : binary_protocol.rs:33
        +V5_SEQ_BYTES 8 : binary_protocol.rs:30
        +MSG_AGENT_ACTION 0x23 sibling frame, peeled off before the position decoder : binary_protocol.rs:998
    }
    class V5Envelope {
        +u8 tag_0x05 offset 0
        +u64 broadcast_seq offset 1..9 LE
        +bytes v3_records offset 9..
    }
    class NodeRecord52 {
        +u32 raw_id offset 0 : parse_node_record binary_protocol.rs:947
        +f32 position_x offset 4
        +f32 position_y offset 8
        +f32 position_z offset 12
        +f32 velocity_x offset 16
        +f32 velocity_y offset 20
        +f32 velocity_z offset 24
        +f32 sssp_distance offset 28
        +i32 sssp_parent offset 32
        +u32 cluster_id offset 36
        +f32 anomaly offset 40
        +u32 community_id offset 44
        +f32 centrality offset 48
        +NODE_RECORD_BYTES 52 : binary_protocol.rs:32
    }
    class NodeIdFlags {
        +NODE_ID_MASK 0x03FF_FFFF : binary_protocol.rs:41
        +AGENT_NODE_FLAG 0x8000_0000 : binary_protocol.rs:42
        +KNOWLEDGE_NODE_FLAG 0x4000_0000 : binary_protocol.rs:43
        +ONTOLOGY_TYPE_MASK 0x1C00_0000 : binary_protocol.rs:44
        +ONTOLOGY_CLASS_FLAG 0x0400_0000 : binary_protocol.rs:45
        +ONTOLOGY_INDIVIDUAL_FLAG 0x0800_0000 : binary_protocol.rs:46
        +ONTOLOGY_PROPERTY_FLAG 0x1000_0000 : binary_protocol.rs:47
    }
    class VisualsKey {
        +u32 community_id
        +u8 centrality_bucket 1/64ths
        +u8 anomaly_bucket 1/16ths
        +of(NodeUpdate) : binary_protocol.rs:131
    }
    V5Envelope "1" o-- "N" NodeRecord52
    NodeRecord52 --> NodeIdFlags
    NodeRecord52 --> VisualsKey : quantised - signal only on change
    FrameHeader <|-- V5Envelope
```

**Debt (loose end):** VisionClaw now writes only the defaults `(INFINITY, -1)` into slot 28 (`docs/BASELINE-architecture.md:252-255`). The headset decodes that slot (`xr-client/rust/src/binary_protocol.rs:962`), sanitises +Inf to 0.0 (`xr-client/rust/src/binary_protocol.rs:931-932`), and no other module reads it, so on the VisionClaw volume the eight SSSP bytes per record carry nothing.

## VC-36.6 Frame ingest — drain, version dispatch, decode and sanitise, then freshness admission

```mermaid
sequenceDiagram
    autonumber
    participant SV as server /wss
    participant IX as transport inbox VecDeque<br/>xr-client/rust/src/transport.rs:124
    participant PL as BinaryProtocolClient::poll<br/>xr-client/rust/src/binary_protocol.rs:1356
    participant EF as emit_frame<br/>xr-client/rust/src/binary_protocol.rs:2707
    participant AD as FreshnessGate::admit_and_ack<br/>xr-client/rust/src/binary_protocol.rs:892
    participant DP as decode_position_frame_with_sequence<br/>xr-client/rust/src/binary_protocol.rs:646
    participant PR as parse_node_record<br/>xr-client/rust/src/binary_protocol.rs:947
    participant SN as sanitize_node_update<br/>xr-client/rust/src/binary_protocol.rs:910
    participant FG as FreshnessGate::admit<br/>xr-client/rust/src/binary_protocol.rs:831
    participant RS as RenderStore<br/>xr-client/rust/src/render_store.rs:524
    participant GD as graph_scene.gd

    SV-->>IX: Binary frame pushed as Frame<br/>transport.rs:199
    GD->>PL: poll() once per physics tick, through each BackendConnection<br/>graph_scene.gd:3063-3064
    PL->>IX: drain the whole inbox into a Vec<br/>binary_protocol.rs:1361
    PL->>EF: Frame(bytes) goes to emit_frame<br/>binary_protocol.rs:1380
    alt first byte is 0x23
        EF-->>PL: MSG_AGENT_ACTION to the agent registry, early return, see VC-36.9<br/>binary_protocol.rs:2712 binary_protocol.rs:2743
    else position frame
        EF->>AD: admit_and_ack(bytes, now_ms) which calls admit_frame<br/>binary_protocol.rs:2752, binary_protocol.rs:876
        AD->>DP: decode first, gate second<br/>binary_protocol.rs:880
        alt version 0x05
            DP-->>AD: read u64 broadcast_seq at 1..9<br/>binary_protocol.rs:661 binary_protocol.rs:671
        else version 0x03
            DP-->>AD: no sequence, None<br/>binary_protocol.rs:660
        else other
            DP-->>EF: DecodeError::BadVersion<br/>binary_protocol.rs:673
            Note right of EF: error! and return, same arm for TooShort and Misaligned<br/>binary_protocol.rs:2760-2762<br/>the quiet warn-and-ignore path is ingest_frame, tests only<br/>binary_protocol.rs:976
        end
        loop each 52-byte record
            DP->>PR: parse_node_record(chunk)<br/>binary_protocol.rs:686
            PR-->>DP: NodeUpdate
            DP->>SN: sanitize_node_update(u)
            alt non-finite position or velocity, or any position component above WORLD_LIMIT_M
                SN-->>DP: None, record dropped<br/>binary_protocol.rs:911-922
                Note right of SN: WORLD_LIMIT_M = 10_000.0 binary_protocol.rs:38<br/>velocity is checked for finiteness only
            else survives
                SN-->>DP: Some(u), non-finite anomaly, centrality, sssp_distance zeroed<br/>binary_protocol.rs:925-927
            end
        end
        DP-->>AD: Option seq and updates
        AD->>FG: admit(sequence)<br/>binary_protocol.rs:881
        alt no sequence (V3)
            FG-->>AD: AcceptUnsequenced, watermark untouched<br/>binary_protocol.rs:832-837
        else awaiting_resync armed by a Disconnect (any sequenced frame, every frame is a full snapshot)
            FG-->>AD: AcceptResync, watermark re-based even backwards<br/>binary_protocol.rs:840-849
        else seq above watermark
            FG-->>AD: Accept<br/>binary_protocol.rs:854
        else seq equal to watermark
            FG-->>AD: RejectDuplicate<br/>binary_protocol.rs:859
        else seq below watermark
            FG-->>AD: RejectStale<br/>binary_protocol.rs:863
        end
        AD-->>EF: Some(updates) only when is_accepted<br/>binary_protocol.rs:882
        Note right of EF: a rejected frame was already decoded and sanitised, then dropped with warn! and NOT acknowledged, so the server rebuilds the chunk after its ack timeout<br/>binary_protocol.rs:2754-2758<br/>FrameKind and RejectAwaitingResync are removed, there are no deltas (ADR-2139)
        EF->>EF: ack_sequence(verdict) names a sequence only for Accept or AcceptResync, encode_broadcast_ack builds [0x34][u64 seq][u32 nodes][u64 ms]<br/>binary_protocol.rs:716, binary_protocol.rs:703
        EF->>RS: upsert, set_cluster, set_node_kind per update<br/>binary_protocol.rs:2769-2781
        RS-->>GD: node_visuals_updated only when VisualsKey changes or the node is new<br/>binary_protocol.rs:2785
        EF->>SV: after the records are applied, Outbound::Binary(ack) tells the server, which then builds this client's next chunk (ADR-2141)<br/>binary_protocol.rs:2798-2799
        Note over RS,GD: Per-frame signal traffic stays near zero<br/>once communities stabilise. binary_protocol.rs:124-130
    end
    Note over PL,FG: On Disconnected, poll arms FreshnessGate::reconnect<br/>so the next sequenced frame re-baselines<br/>binary_protocol.rs:1373 binary_protocol.rs:826.<br/>connect_to_url arms it too, because an aborted pump never pushes Disconnected<br/>binary_protocol.rs:1329-1332
```

**Debt:** `ingest_frame` is a second decode-and-sink path with a quiet `BadVersion` arm that production never calls, only tests (`xr-client/rust/src/binary_protocol.rs:976`, `xr-client/rust/src/binary_protocol.rs:3250`). The live path is `emit_frame` (`xr-client/rust/src/binary_protocol.rs:2707`).
**Invariant:** a frame is fully decoded and sanitised before the freshness gate judges it, so a stale frame costs a full decode and is then dropped (`xr-client/rust/src/binary_protocol.rs:880-881`).

## VC-36.7 RenderStore instance packing and the stride-16 invariant (ADR-2034)

```mermaid
sequenceDiagram
    autonumber
    participant GD as graph_scene.gd _physics_process<br/>xr-client/scripts/graph_scene.gd:3055
    participant PV as _pack_volumes<br/>xr-client/scripts/graph_scene.gd:1764
    participant CV as ConnectionVolumeSource.pack<br/>xr-client/scripts/connection_volume_source.gd:147
    participant UB as _update_beam_multimesh<br/>xr-client/scripts/graph_scene.gd:3625
    participant RS as build_edge_buffer<br/>xr-client/rust/src/render_store.rs:2087
    participant SC as edge_style_code<br/>xr-client/rust/src/render_store.rs:132
    participant MM as each volume's own Edges and Ribbons MultiMesh<br/>use_custom_data = true

    Note over RS: NODE_STRIDE = 20 (12 transform + 4 colour + 4 custom) render_store.rs:108<br/>EDGE_STRIDE = 12 (semantic-plane edges, uniform tint) render_store.rs:120<br/>EDGE_STRIDE_TYPED = 16 (12 transform + 4 INSTANCE_CUSTOM) render_store.rs:124
    GD->>PV: _pack_volumes() every physics frame graph_scene.gd:3110
    loop each volume, overlay last graph_scene.gd:1771-1776
        alt volume hidden or its connection offline
            PV->>CV: clear_instances(), nothing packed graph_scene.gd:1782-1786
        else
            PV->>CV: pack(eye, node and edge ratios, this volume's near caps, near radii)<br/>graph_scene.gd:1787-1790
            CV->>CV: set_pack_slot(mask) so two volumes on one client keep separate plans<br/>connection_volume_source.gd:153-154, :78-80
            CV->>RS: build_node_buffer_lod, then build_edge_buffer_lod(content_pairs, ...)<br/>connection_volume_source.gd:158-166
            Note over CV,RS: content_pairs is this volume's own edges, both ends in its class mask,<br/>heaviest first, re-read every 0.25 s or on a new generation connection_volume_source.gd:124-139
        end
    end
    loop each drawn edge
        RS->>SC: edge_style_code(edge_type)
        SC-->>RS: 2 subclass/taxonomy - 1 typed - 0 untyped
        Note right of SC: local-name match after last / # or :<br/>case-insensitive render_store.rs:126-134
        RS->>RS: pack 12 transform floats + custom[r,g,b,a=style]
    end
    RS-->>CV: near cylinders, far tier read back via ribbon_edge_buffer()<br/>binary_protocol.rs:2065 binary_protocol.rs:2145
    CV->>MM: NodeLod.assign_stride(edges, ebuf, EDGE_STRIDE) and the same for ribbons<br/>connection_volume_source.gd:167-168
    Note over CV,MM: ADR-2034 INVARIANT: divide by 16, never 12. EDGE_STRIDE = 16 here<br/>(connection_volume_source.gd:31-32) and assign_stride sizes instance_count from<br/>buf.size() / stride before the single buffer assignment (xr-client/scripts/node_lod.gd:65-75).<br/>A /12 divisor mis-sizes instance_count, set_buffer rejects EVERY frame and all edges<br/>vanish (regression 63d9bb9b8)
    Note over GD,PV: Both node and edge packs run on EVERY physics frame, no longer on alternating<br/>frames at 45 Hz, because the Rust pack plans make the pair cost about 1 ms<br/>at 13k nodes and 20k edges graph_scene.gd:3106-3111, render_store_pack.rs:1-12.<br/>The scene's own NodesMulti and EdgesMulti are only templates now and stay empty<br/>graph_scene.gd:1455-1460
    GD->>UB: _update_beam_multimesh() graph_scene.gd:3119
    alt agent_multi, its multimesh or _binary_client missing, or no build_beam_buffer
        UB-->>GD: return (BREAK) graph_scene.gd:3626-3629
    end
    UB->>RS: build_beam_buffer(EDGE_WORLD_RADIUS / (BEAM_MESH_RADIUS * _graph_scale)) graph_scene.gd:3632-3633
    RS->>RS: source = the published embodiment anchor, and nothing else
    Note over RS: ADR-2109 INVARIANT: the beam starts at the BODY the user sees.<br/>build_beam_buffer reads only agent_anchors: an agent with no anchor has<br/>NO beam, and an agent id is never looked up as a graph node.<br/>Anchors are visual-only, positions and physics untouched.<br/>render_store.rs:2169-2173, :2196-2198, set_agent_anchors render_store.rs:1156
    RS-->>UB: agent-to-target beams, same stride 16, custom r = action code, g/b = taper, a = status code<br/>render_store.rs:2164-2165
    UB->>UB: count = buf.size() / 16 graph_scene.gd:3635
    UB->>MM: GraphRoot/AgentMulti instance_count + buffer graph_scene.gd:3636-3639
    Note over UB: ADR-2034 server-which/client-where: the server owns WHICH node<br/>an agent works on plus status. The capsule room-position and the<br/>work-beam geometry are client concerns. Beam count = live<br/>working/blocked agents (tens), so this runs every frame.<br/>Beams use only the VisionClaw client (_binary_client), never Codescape.
```

## VC-36.8 HUD, shared instrument theme, the seven ADR-2189 sections with a rail, and press-mode controls

```mermaid
sequenceDiagram
    participant W as Controller ray
    participant HUD as _build_ui<br/>xr-client/scripts/hud.gd:411
    participant IA as HudIa.load_doc<br/>xr-client/scripts/hud_ia.gd:88
    participant Theme as XRTheme.create<br/>xr-client/scripts/xr_theme.gd:29
    participant Press as _press_fire<br/>xr-client/scripts/hud.gd:501
    participant Env as SpatialEnvironment._bind_hud<br/>xr-client/scripts/spatial_environment.gd:181
    HUD->>Theme: create opaque surfaces, typography and focus outlines
    HUD->>IA: _build_tab_bar reads res://data/ia.json<br/>hud.gd:506, hud.gd:510, hud_ia.gd:19
    alt missing, malformed, or its section ids are not the seven in order
        IA-->>HUD: ok false with FALLBACK_SECTIONS and FALLBACK_SCOPES<br/>hud_ia.gd:26-39, hud_ia.gd:90-96, hud_ia.gd:110-112
        Note over HUD: only push_warning, which a headset wearer never sees<br/>hud.gd:511-512
    else the seven sections
        IA-->>HUD: id, label, hotkey, subsection names and scope texts<br/>hud_ia.gd:97-118, ia.json:12, ia.json:1872
    end
    HUD->>HUD: one tab per section, then Fit, Legend and Explore<br/>hud.gd:517, hud.gd:528-545
    Note over HUD: Agents and Governance is shortened to Agents on its button hud.gd:521.<br/>Fit emits fit_view (VC-36.30), Legend opens the key page, Explore the query page<br/>hud.gd:539-544
    HUD->>HUD: build all thirteen pages in TAB_ORDER, each beside the rail<br/>hud.gd:272, hud.gd:612-620
    Note over HUD: XR_PAGES places every page under one section with its rail label hud_ia.gd:44-72,<br/>XR_ONLY names the headset-only subsections hud_ia.gd:75-83
    W->>HUD: press a section tab
    HUD->>HUD: _show_section reopens its last page, else its first live page<br/>hud.gd:1949-1955
    HUD->>HUD: _refresh_rail labels at most RAIL_MAX 3 pooled buttons,<br/>hidden when the section has one page<br/>hud.gd:282, hud.gd:1967-1977
    HUD->>HUD: Layout and Shapes pages carry ia.json's shared scope text<br/>hud.gd:550-551, hud.gd:562
    HUD->>HUD: _build_key_page mirrors the live palette<br/>xr-client/scripts/hud.gd:1445
    Note over HUD: DEBT: the Key swatch constants are duplicated from their<br/>sources by design, the same posture as SWARM_STATUS_COLORS,<br/>so a palette change must be made twice. hud.gd:289, hud.gd:1480
    HUD->>HUD: Explore holds Graph Query and Memory Search with an on-screen keyboard<br/>xr-client/scripts/hud.gd:1048 xr-client/scripts/hud.gd:1081 xr-client/scripts/hud.gd:1126 xr-client/scripts/hud.gd:1167
    HUD->>HUD: Encode holds colour mode and label steps, Look the edge share and Reduce motion,<br/>System Headset holds dwell, haptics and Low-cost visuals<br/>hud.gd:698, hud.gd:703, hud.gd:729, hud.gd:739, hud.gd:743, hud.gd:757, hud.gd:765, hud.gd:769
    HUD->>HUD: the beat row sits on System Connection, the header EdgeBadge reads Edges N of M drawn (VC-51.7)<br/>hud.gd:1565, hud.gd:1649, hud.gd:466, hud.gd:2549
    Note over HUD: A MIC badge shows while the microphone is on, and the HUD viewport attaches render-on-demand and batching<br/>xr-client/scripts/hud.gd:460 xr-client/scripts/hud.gd:382 xr-client/scripts/hud.gd:383
    HUD->>Press: create buttons and CheckButtons with press firing
    Note over Press: ADR-2033 press behaviour retained. Fresh headset jitter acceptance remains open.
    Env->>HUD: deferred bind after world-space reparent and synchronise preferences
    W->>HUD: press comfort toggle
    HUD->>Env: control_pressed visual_motion or visual_quality<br/>xr-client/scripts/hud.gd:102
    Env->>Env: apply scene-local reversible rendering budget<br/>xr-client/scripts/spatial_environment.gd:128
    Env->>HUD: set_visual_comfort without re-emitting<br/>xr-client/scripts/hud.gd:2711
    HUD->>HUD: flash_notice writes the bottom strip on EVERY page<br/>xr-client/scripts/hud.gd:2096, rendered at hud.gd:2433
    Note over HUD: ADR-2108: a rejected server write used to reach only push_warning,<br/>invisible in the headset. The strip is now the in-HMD channel for it.
    Note over HUD: Overflow guard and overlay shield remain explicit<br/>xr-client/scripts/hud.gd:2013<br/>xr-client/scripts/hud.gd:2001
    Note over HUD: see VC-40.1 - set_backend_capabilities greys every server-bound control,<br/>hides a page the backend cannot serve and a section with no page left
```

**What it shows.** Since 2026-11-06 the tab bar is the desktop's seven sections from `xr-client/data/ia.json` plus Fit, Legend and Explore, and a rail picks a section's pages, so every control is two levels deep. **Why it is this way.** ADR-2189 gives desktop and headset one information architecture, two levels deep (`docs/adr/ADR-2189-desktop-and-xr-menus-share-one-seven-section-information-architecture.md:39`, `docs/adr/ADR-2189-desktop-and-xr-menus-share-one-seven-section-information-architecture.md:108-109`); the headset takes the vocabulary from the generated file and places its own pages under it (`xr-client/scripts/hud_ia.gd:3-17`), with `test_hud_ia.gd` checking every name (`xr-client/tests/unit/test_hud_ia.gd:68`).
**Debt (parallel):** the built-in seven sections and scope texts (`xr-client/scripts/hud_ia.gd:26-39`) duplicate the committed `xr-client/data/ia.json`, itself a byte copy of `client/src/features/control-center/registry/ia.json` (`xr-client/scripts/hud_ia.gd:6-8`); `xr-client/tests/unit/test_hud_ia.gd:38` pins the fallback to the file.
**Open (risk):** a malformed or six-section `ia.json` silently drops the headset to the built-in menu: the only signal is `push_warning` (`xr-client/scripts/hud.gd:511-512`), the channel ADR-2108 already found invisible in a headset, not the notice strip (`xr-client/scripts/hud.gd:2096`).
Resolved at a950f4d80: ADR-2189's acceptance boundary now says the XR HUD is built and reads `xr-client/data/ia.json` (`docs/adr/ADR-2189-desktop-and-xr-menus-share-one-seven-section-information-architecture.md:18`), and its 2026-11-06 amendment withdraws the earlier "not built" text (`docs/adr/ADR-2189-desktop-and-xr-menus-share-one-seven-section-information-architecture.md:361`), matching the tab bar, rail and header buttons (`xr-client/scripts/hud.gd:506-545`).
**Tension (ADR-2189 vs code):** ADR-2189 gives XR "the same five dials as ± steppers on Layout and Look" (`docs/adr/ADR-2189-desktop-and-xr-menus-share-one-seven-section-information-architecture.md:148`), but the headset has only Spread and the label budget, with no Motion, Glow or Atmosphere stepper (`docs/XR-client.md:1125-1127`).
**Debt (loose end):** the rail is a pool of `RAIL_MAX` 3 buttons (`xr-client/scripts/hud.gd:282`) and `_refresh_rail` labels only that many (`xr-client/scripts/hud.gd:1969-1970`); Layout and System already fill it (`xr-client/scripts/hud_ia.gd:56-60`, `xr-client/scripts/hud_ia.gd:67-71`), so a fourth page in either would be unreachable from the rail without any error.

## VC-36.9 Agent co-presence and the 0x23 work-beam data plane

```mermaid
sequenceDiagram
    autonumber
    participant SV as server broadcast_to_all<br/>src/utils/binary_protocol.rs MessageType::AgentAction
    participant BP as BinaryProtocolClient::poll<br/>xr-client/rust/src/binary_protocol.rs:1356
    participant DA as decode agent-action batch<br/>xr-client/rust/src/binary_protocol.rs:1105
    participant AS as avatar_state AgentAvatarNode<br/>xr-client/rust/src/avatar_state.rs:455
    participant RS as RenderStore beam buffer<br/>xr-client/rust/src/render_store.rs:524
    participant HD as _build_swarm_page<br/>xr-client/scripts/hud.gd:1369

    Note over SV,BP: MSG_AGENT_ACTION = 0x23 binary_protocol.rs:998.<br/>Fanned to every /wss client on the SAME binary path as<br/>position frames - separated from 0x03/0x05 by the leading byte.
    SV-->>BP: [0x23][u16 count]([u16 ev_len][ev_len bytes])*
    BP->>DA: decode_agent_actions(frame)
    alt leading byte is not 0x23
        DA-->>BP: None
    else
        loop count events
            DA->>DA: source u32 | target u32 | action u8 | ts u32 | task line
            Note right of DA: build_agent_action_frame mirrors the server<br/>encode_agent_actions binary_protocol.rs:2868
        end
        DA-->>BP: Vec<AgentAction>
    end
    BP->>AS: update per-agent activity + gaze attention
    BP->>BP: record last_agent_action instant binary_protocol.rs:2715
    Note over BP: last_agent_action_age_ms returns -1 if none has arrived<br/>binary_protocol.rs:1799 - P1 liveness probe
    AS->>RS: agent status -> beam target + status code
    RS-->>RS: beam re-routes to the fold representative when the<br/>real target is folded away (render_store.rs:3127 test)
    BP->>HD: swarm roster update
    HD->>HD: SWARM_STATUS_COLORS {0 idle slate, 1 working green,<br/>2 blocked amber-red, 3 done cyan-white} hud.gd:289
    Note over HD: Mirrors render_store::agent_status_color - ADR-140 Pillar 3
    Note over AS,RS: INVARIANT: evidence only moves forward. record_agent_action rejects an<br/>action no newer than the record's evidence_ts (render_store.rs:954) and<br/>set_agent_state_at does the same (render_store.rs:1035), both via ts_is_newer<br/>(render_store.rs:512). Stale hits increment agent_actions_stale /<br/>agent_states_stale and expire_stale_agents (render_store.rs:1057) ages records out.<br/>RESOLVED: the XR-client.md paragraph that claimed actions were stored without<br/>freshness checks has been superseded in the governing doc
    RS->>RS: agent_hover_offset(target, agent_id, HOVER_RADIUS)
    Note over RS: Hover motion IS implemented. Golden-angle walk 2.3999632 rad keyed by<br/>agent id fans multiple agents around one node instead of stacking them,<br/>lifted by HOVER_LIFT. HOVER_RADIUS = 1.5 render_store.rs:441, :451-461
    Note over RS: Per-node target priority in hunt render_store.rs:1877-1887 - a grabbed node is<br/>pinned, an ACTIVE AGENT hovers at its target (local hover point, not a<br/>server position), a member folding IN chases its representative, everything<br/>else eases to self.targets. DIVERGENCE agent endpoints use LOCAL positions<br/>directly while beam targets are fold-remapped and drawn-gated - the closeout<br/>asks for explicit state precedence, expiry and visible stale/error handling.<br/>docs/XR-client.md 'Estate closeout qualification 2026-09-04'
    Note over RS,HD: ADR-2109 changed WHERE a beam starts, not what feeds the registry.<br/>The registry is now also the source of embodiment - see VC-36.21.<br/>graph_scene.gd:4231
```

## VC-36.10 Presence socket — challenge/auth/joined handshake and 0x43 pose traffic

```mermaid
sequenceDiagram
    autonumber
    participant GS as graph_scene.gd connect_to_server<br/>xr-client/scripts/graph_scene.gd:2978
    participant PC as PresenceClient::handshake<br/>xr-client/rust/src/presence.rs:307
    participant SG as NostrSigner<br/>xr-client/rust/src/signer.rs:35
    participant SV as server /ws/presence<br/>src/actors/presence_actor.rs:380
    participant PN as PresenceClientNode (gdext)<br/>xr-client/rust/src/presence.rs:448
    participant AV as Avatar.tscn / AgentAvatar.tscn

    Note over GS,SV: graph and presence dial the backend on port 4000 directly. nginx on port 3001<br/>also upgrades /ws/presence since 2026-11-06 but is the only Codescape route,<br/>so the live launch sets XR_CODESCAPE_WS alone. docs/XR-client.md:304<br/>nginx.dev.conf:56 nginx.dev.conf:168. RETIRED: the earlier 'nginx does not proxy presence' note
    GS->>PC: _connect_presence: close(), then join(presence_url, room_urn, display_name, secret)<br/>graph_scene.gd:3046-3052, client created at graph_scene.gd:1233
    SV-->>PC: {"type":"challenge","nonce":<64hex>,"ts":<u64>}
    alt first frame is not a challenge
        PC-->>GS: Err("expected challenge, got {other}") presence.rs:313-318
    else
        PC->>SG: did()
        PC->>SG: sign_challenge(nonce_bytes, ts)
        Note over SG: schnorr(SHA256(nonce || ts.to_le_bytes()))<br/>BIP-340 via NostrSigner presence.rs:13-14
        SG-->>PC: signature 128hex
        PC->>SV: {"type":"auth","did":...,"signature":...,"room_id":<urn>,...}
        SV-->>PC: {"type":"joined","room_id":<urn>,"avatar_id":<urn>, roster}
        loop while joined
            PC->>SV: binary 0x43 self pose
            SV-->>PC: [u8 0x43][u64 broadcast_seq][u32 room_id][u16 user_count] sibling poses
            Note over PC: SiblingBatch presence.rs:135, wire layout presence.rs:150
            SV-->>PC: text avatar_joined / avatar_left
            SV-->>PC: 0x44 agent co-presence {state, gaze, attention}
            Note over SV,PC: codec in visionclaw_xr_presence::agent_presence,<br/>driven by avatar_state - xr-client/rust/src/lib.rs:5-8
        end
        PC->>PN: expose to GDScript
        PN->>AV: instance / update sibling avatars
    end
    Note over PC: XR_NOSTR_SECRET is REQUIRED in practice: NostrAuth.create()<br/>returns an ephemeral signer when empty (graph_scene.gd:1241-1246),<br/>but an ephemeral key cannot satisfy the presence challenge<br/>or gate drag/pin. docs/XR-client.md INVARIANT 6
```

## VC-36.11 HTTP writes — NIP-98 per-request header and the dev-bearer fallback

```mermaid
sequenceDiagram
    autonumber
    participant HD as hud.gd control_pressed
    participant GS as graph_scene.gd handler
    participant HB as _http_base, then BackendConnection.http_base<br/>xr-client/scripts/graph_scene.gd:2972
    participant AH as _auth_headers<br/>xr-client/scripts/graph_scene.gd:2860
    participant NA as NostrAuth (gdext)<br/>xr-client/rust/src/signer.rs:208
    participant SV as VisionClaw REST

    HD->>GS: control_pressed(action)
    GS->>HB: _http_base()
    Note over HB: every VisionClaw request uses the visionclaw connection's base graph_scene.gd:2972-2975,<br/>xr-client/scripts/backend_connection.gd:174-186. Codescape serves none of these routes
    alt XR_BACKEND_HTTP set
        HB-->>GS: with_backend_prefix(override) - trailing "/" stripped backend_connection.gd:175-177, :75-79
    else wss:// prefix
        HB-->>GS: "https://" + ws.substr(6)
    else ws:// prefix
        HB-->>GS: "http://" + ws.substr(5)
    else neither
        HB-->>GS: ws unchanged
    end
    GS->>AH: _auth_headers(url, method)
    alt _nostr_auth non-null and _nostr_secret_present and has nip98_header
        AH->>NA: nip98_header(url, method)
        NA->>NA: nip98_http_authorization signer.rs:138
        NA-->>AH: "Nostr <b64 kind-27235>"
        AH-->>GS: Authorization: Nostr <b64>
        Note over AH,NA: INVARIANT the signed URL must be the EXACT request URL<br/>including query, or the server tag check fails.<br/>graph_scene.gd:2854-2855, docs/XR-client.md INVARIANT 6
    else no real secret
        AH-->>GS: Authorization: PHYSICS_BEARER
        AH-->>GS: X-Nostr-Pubkey: <pubkey_hex>
        Note over AH: DIVERGENCE the legacy dev bearer path still exists and is<br/>still constructed by the client. The server refuses it for any<br/>non-loopback peer, so the HP headset never passes it.<br/>_nostr_secret_present gates it graph_scene.gd:808, :1245, :2440.<br/>docs/XR-client.md:1034 (Dev bearer fallback)
    end
    AH-->>GS: Content-Type: application/json
    GS->>SV: POST /api/settings/physics/reset-layout graph_scene.gd:2877
    GS->>SV: PUT /api/settings/physics?graph=knowledge graph_scene.gd:2921
    GS->>SV: GET /api/graph/fold?level=<n> graph_scene.gd:2560
    GS->>SV: GET /api/graph/node/<id>/relations graph_scene.gd:5141
    GS->>SV: POST /api/canary/observe/<CANARY_M4_RAY> graph_scene.gd:4533
    Note over GS,SV: hud.gd's intervention decide POST uses the same signing path<br/>via hud.configure_intervention(_http_base(), _nostr_auth) graph_scene.gd:2201
    Note over GS,SV: INVARIANT REPEL_K_MAX is 500 and must not exceed the server validator bound (0..500):<br/>a Spread press that clamped above it was rejected with HTTP 400 and the whole PUT discarded.<br/>The bound lives server-side, so the two constants move together graph_scene.gd:1042-1046, clamp at graph_scene.gd:2816
    SV-->>GS: 401 or 403
    GS->>GS: _describe_write_failure names status, credential path and remedy<br/>xr-client/scripts/graph_scene.gd:2901
    GS->>HD: hud.flash_notice(_last_write_error) graph_scene.gd:2846
    Note over GS,HD: ADR-2108: the HP is never a loopback peer, so before this landed<br/>every server-routed HUD write 401d into an invisible push_warning.<br/>GUT cover: test_write_denied_notice.gd:16, :30, :38
```

## VC-36.12 Constrained layouts and the DAG-rank label accept (ADR-2035)

```mermaid
sequenceDiagram
    autonumber
    participant HD as hud.gd Layout tab
    participant GS as graph_scene.gd
    participant PM as _post_layout_mode<br/>xr-client/scripts/graph_scene.gd:2751
    participant PR as _post_radial<br/>xr-client/scripts/graph_scene.gd:2771
    participant LH as layout_handler.rs<br/>src/handlers/layout_handler.rs:45
    participant FC as force_compute_actor.rs<br/>src/actors/gpu/force_compute_actor.rs:443
    participant DR as compute_dag_ranks<br/>src/actors/gpu/force_compute_actor.rs:462

    Note over GS: LAYOUT_MODES = [forceDirected, hierarchical, radial, spectral,<br/>temporal, clustered] graph_scene.gd:960. DRIFT the server no longer advertises the<br/>same list: LIVE_LAYOUT_MODES is forceDirected, hierarchical, radial, dagLeftRight,<br/>typeClustering, spectral, temporal (layout_handler.rs:23-31), served by GET at :33-42.<br/>clustered is accepted but not advertised (:20-22, parse at :64-71), and the headset's<br/>cycle can never reach dagLeftRight or typeClustering
    HD->>GS: control_pressed("layout_cycle")
    GS->>GS: next_idx = (_layout_mode_idx + 1) % LAYOUT_MODES.size() graph_scene.gd:2743
    GS->>PM: _post_layout_mode(LAYOUT_MODES[next_idx])
    PM->>LH: POST /api/layout/mode + NIP-98 header
    HD->>GS: control_pressed("radial:<dagRank|typeTier|ego>")
    GS->>PR: _post_radial(mode)
    PR->>LH: POST /api/layout/radial (server layout_handler.rs:168-199)
    HD->>GS: Hierarchy toggle
    GS->>LH: PUT dagBiasK = 0.6 on / 0.0 off graph_scene.gd:2685
    HD->>GS: Shells +/- nudge
    GS->>LH: PUT dagLevelDistance graph_scene.gd:2697
    LH->>FC: SetRadialLayout{DagRank}
    loop each edge
        FC->>FC: hierarchy_pairs - filter edge.asserts_subsumption
        alt owl_property_iri provenance is rdfs:subClassOf
            FC-->>DR: (parent_idx, child_idx) pair
            Note right of FC: ADR-2035 AMENDED 2026-10-02 (N-14): ranks key on<br/>subClassOf PROVENANCE (Edge.asserts_subsumption),<br/>never on the bare "hierarchical" force label.<br/>force_compute_actor.rs:436-442, :443-460
        else equivalentClass, sameAs, subPropertyOf, instance-of,<br/>domain membership, or no provenance at all
            FC-->>DR: skipped - edge not on the GPU is skipped too
            Note right of FC: the ingest folds equivalence, sameAs, sub-property<br/>and instance-of into the same "hierarchical" label and<br/>domain membership has its own label - ranking any of<br/>them would fabricate layers. force_compute_actor.rs:436-441
        end
    end
    DR->>DR: ranks initialised to -1.0 for every node
    alt num_nodes == 0 or hierarchy_edges empty
        DR-->>LH: all nodes stay -1.0 (unranked)
        Note right of DR: Before the accept landed (73540faa0) EVERY node stayed<br/>unranked, so Radial: DAG and the Hierarchy toggle were<br/>silently inert - reported in-headset as the Radial Shells<br/>buttons "appearing disconnected". force_compute_actor.rs:462-466
    else
        DR-->>LH: ranks assigned
    end
    Note over FC: RESOLVED DOC-DRIFT: docs/XR-client.md 0.1.3 (2026-10-02) now<br/>documents the provenance keying, and the corrected doc-comment<br/>at force_compute_actor.rs:436-442 is committed at this revision -<br/>the old excludes-vs-accepts contradiction is gone from both.
    Note over DR: RESOLVED ADR-2079, SUPERSEDED by the amendment: the closeout's<br/>predicate test is replaced by only_subsumption_provenance_ranks<br/>at force_compute_actor.rs:5369 - bare "hierarchical" with no<br/>subClassOf IRI is now REJECTED, and equivalence, sub-property,<br/>sameAs and instance-of foldings stay out of the rank space.
    Note over FC,DR: RESOLVED DIVERGENCE: the old lossy-accept cost is gone - a<br/>domain-membership edge under "hierarchical" no longer ranks and<br/>can no longer shortcut a subclass chain (BFS lift), asserted by<br/>membership_edges_neither_rank_nor_shortcut_a_subclass_chain at<br/>force_compute_actor.rs:5449 and a_domain_root_never_ranks_below_its_own_members at force_compute_actor.rs:5261.
```

## VC-36.13 gdext class surface (GDScript composes, Rust owns the wire)

```mermaid
classDiagram
    class VisionclawXrExtension {
        <<gdextension>>
        on_level_init(InitLevel) : xr-client/rust/src/lib.rs:72
        init_tracing at InitLevel::Scene : xr-client/rust/src/lib.rs:82
    }
    class BinaryProtocolClient {
        base RefCounted : binary_protocol.rs:1191
        poll()
        build_edge_buffer(pairs, radius)
        build_beam_buffer(radius)
        last_agent_action_age_ms() : binary_protocol.rs:1797
    }
    class PresenceClientNode {
        base RefCounted : presence.rs:448
        handshake challenge-auth-joined
        0x43 pose - 0x44 agent co-presence
    }
    class NostrAuth {
        base RefCounted : signer.rs:208
        from_secret_hex(hex) : signer.rs:35
        generate() ephemeral : signer.rs:55
        nip98_authenticate_json(url, method) : signer.rs:126
        nip98_http_authorization(url, method) : signer.rs:138
    }
    class XrInteraction {
        base RefCounted : interaction.rs:122
        hand ray cast + pinch detection
        HandRay, TargetCandidate, RaycastHit : interaction.rs:28
    }
    class GazeTracker {
        base RefCounted : gaze.rs:215
        GazeResolver, OneEuroFilter : gaze.rs:115
        degrades eye-gaze to head when unsupported
    }
    class SelectionArbiterNode {
        base RefCounted : selection.rs:436
        three-resolver arbiter selection.rs:240
        DwellCharger, SelectionConfig : selection.rs:125
    }
    class ProxemicsSolver {
        base RefCounted : proxemics.rs:203
        Hall's-zones arc solver
    }
    class AgentAvatarNode {
        base RefCounted : avatar_state.rs:455
        per-agent activity + gaze attention
    }
    class LodPolicy {
        base RefCounted : lod.rs:379
        distance-bucket LOD
    }
    class SpatialVoiceRouter {
        base RefCounted : webrtc_audio.rs:141
        SpatialVoiceRouterCore : webrtc_audio.rs:43
        ListenerTransform, VoiceTrackState : webrtc_audio.rs:30
    }
    class RenderStore {
        plain struct : render_store.rs:524
        NODE_STRIDE 20 - EDGE_STRIDE 12 - EDGE_STRIDE_TYPED 16
    }
    VisionclawXrExtension --> BinaryProtocolClient
    VisionclawXrExtension --> PresenceClientNode
    VisionclawXrExtension --> NostrAuth
    VisionclawXrExtension --> XrInteraction
    VisionclawXrExtension --> GazeTracker
    VisionclawXrExtension --> SelectionArbiterNode
    VisionclawXrExtension --> ProxemicsSolver
    VisionclawXrExtension --> AgentAvatarNode
    VisionclawXrExtension --> LodPolicy
    VisionclawXrExtension --> SpatialVoiceRouter
    BinaryProtocolClient --> RenderStore
    PresenceClientNode --> NostrAuth
    SpatialVoiceRouter --> AgentAvatarNode
```

## VC-36.14 OpenXR action map and input bindings

```mermaid
flowchart LR
    subgraph acts["Actions - openxr_action_map.tres"]
        A1["trigger / trigger_click / trigger_touch<br/>openxr_action_map.tres:4,9,15 - /user/hand/left + right"]
        A2["grip / grip_click / grip_force<br/>openxr_action_map.tres:21,26,32"]
        A3["primary + primary_click + primary_touch<br/>openxr_action_map.tres:37,43,49"]
        A4["secondary + secondary_click + secondary_touch<br/>openxr_action_map.tres:55,61,67"]
        A5["menu_button openxr_action_map.tres:73 - select_button openxr_action_map.tres:79"]
        A6["ax_button + ax_touch + by_button + by_touch (A/X and B/Y)<br/>openxr_action_map.tres:85,91,97,103"]
        A7["poses: default_pose, aim, grip_pose, palm_pose openxr_action_map.tres:109,115,121,127 - haptic openxr_action_map.tres:133<br/>23 actions in all, sub-resources at openxr_action_map.tres:3-132"]
    end
    subgraph prof["Interaction profiles bound in the map"]
        P1["khr/simple_controller openxr_action_map.tres:200"]
        P2["htc/vive_controller openxr_action_map.tres:316 - the validated device"]
        P3["microsoft/motion_controller openxr_action_map.tres:440"]
        P4["oculus/touch_controller openxr_action_map.tres:588"]
        P5["bytedance/pico4_controller openxr_action_map.tres:740"]
        P6["valve/index_controller openxr_action_map.tres:920"]
        P7["hp/mixed_reality_controller openxr_action_map.tres:1036"]
        P8["samsung/odyssey_controller openxr_action_map.tres:1160"]
        P9["htc/vive_cosmos_controller openxr_action_map.tres:1284"]
        P10["htc/vive_focus3_controller openxr_action_map.tres:1424"]
        P11["huawei/controller openxr_action_map.tres:1516"]
        P12["htc/vive_tracker_htcx openxr_action_map.tres:1648"]
        P14["ext/hand_interaction_ext openxr_action_map.tres:1732"]
    end
    acts --> prof
    prof -.-> P13X
    prof --> XI["project.godot:65<br/>openxr/default_action_map"]
    P13X["REMOVED (ER-13, VisionClaw 2bf892c3f): the ext/eye_gaze_interaction profile, its gaze-pose binding and /user/eyes_ext in the default pose top-level paths are gone from the map.<br/>Eye gaze is now unbound on every headset and the gaze source always falls back to head gaze, because an interaction profile cannot be added to a live action map after the session starts.<br/>docs/adr/ADR-2036-defensive-openxr-boot.md:101-107, xr_boot.gd:40-50, gaze.rs:138"]
    XI --> CTL["XRController3D left and right read the actions<br/>xr-client/scripts/graph_scene.gd:1197 xr-client/scripts/graph_scene.gd:1198"]
    CTL -. "GDScript pushes values, no action-map read in Rust" .-> XR["XrInteraction (RefCounted) hand ray + pinch<br/>xr-client/rust/src/interaction.rs:122 xr-client/scripts/graph_scene.gd:1234"]
    CTL -.-> GZ["GazeTracker (RefCounted) head/eye ray<br/>xr-client/rust/src/gaze.rs:215 xr-client/scripts/graph_scene.gd:1239"]
    CTL -.-> SA["SelectionArbiterNode (RefCounted)<br/>xr-client/rust/src/selection.rs:436 xr-client/scripts/graph_scene.gd:1238"]
```

## VC-36.15 VISIONCLAW_DEV_MODE LAN bypass from the headset's perspective (ADR-2039, ADR-2108)

```mermaid
sequenceDiagram
    autonumber
    participant HP as HP-Desktop headset client<br/>godot --path xr-client
    participant CO as dev service environment<br/>docker-compose.unified.yml:116
    participant MN as main (dev banner)<br/>project/src/main.rs:294
    participant RH as enforce_release_env_hygiene<br/>project/src/main.rs:118
    participant AU as verify_access<br/>src/utils/auth.rs:191
    participant WA as handle_authenticate<br/>src/handlers/socket_flow_handler/filter_auth.rs:54
    participant SV as VisionClaw handlers

    rect rgb(240, 232, 210)
        Note over CO: VISIONCLAW_DEV_MODE: "${VISIONCLAW_DEV_MODE:-1}"<br/>docker-compose.unified.yml:116 - DEV SERVICE ONLY.<br/>ADR-2108 flipped the default from :-0 to :-1 on 2026-09-08, so<br/>launch.sh up dev boots an ARMED backend and the operator opts OUT<br/>with VISIONCLAW_DEV_MODE=0. Still not in the *common-environment<br/>anchor and not in the production service, so the default never<br/>reaches a release deploy. docker-compose.unified.yml:109-112
    end
    CO->>RH: process env, release build
    alt the var is merely PRESENT
        RH-->>RH: REFUSE TO BOOT, exit code 2 (ADR-06 D11) project/src/main.rs:136 project/src/main.rs:161
        Note right of RH: project/src/main.rs:109 - a release binary physically cannot<br/>honour SETTINGS_AUTH_BYPASS / VISIONCLAW_DEV_MODE.<br/>Caveat: both services load env_file .env, so the var must<br/>never sit in a .env shared with prod - sharper now that the<br/>compose default supplies it unasked.<br/>docker-compose.unified.yml:113-115
    end
    CO->>MN: process env, dev or dev-auth build (the hygiene hook is an empty stub here, project/src/main.rs:169)
    MN->>AU: dev_full_bypass_active() src/utils/auth.rs:116
    alt value in {1, true, TRUE with surrounding space}
        AU-->>MN: true src/utils/auth.rs:117-121
        MN-->>MN: warn banner "VISIONCLAW_DEV_MODE=1 - LAN-LOCAL AUTH BYPASS ACTIVE"<br/>project/src/main.rs:296
        Note over MN: Every request granted as DEV_MODE_PUBKEY<br/>= "dev-mode-local-admin" src/utils/auth.rs:96, project/src/main.rs:299
    else 0, yes, unset
        AU-->>MN: false
    end
    Note over AU: Release builds compile the function to a stub returning false<br/>src/utils/auth.rs:127-130
    HP->>SV: GET /wss upgrade, signed NIP-98 carrier only when XR_NOSTR_SECRET is set transport.rs:71-82
    SV-->>HP: echoes visionclaw, so the headset sends no authenticate frame transport.rs:165-166
    alt XR_NOSTR_SECRET set
        SV-->>HP: carrier verified, the WS session takes the signer's pubkey and power-user flag<br/>src/handlers/socket_flow_handler/http_handler.rs:161-162, src/handlers/socket_flow_handler/http_handler.rs:412-414
    else secret empty
        SV-->>HP: anonymous WS session in every mode
        Note over HP,SV: OPEN WS drag is dropped without a reply<br/>src/handlers/socket_flow_handler/position_updates.rs:199-201
    end
    Note over HP,WA: DRIFT the dev-mode WS grant runs on an authenticate frame src/handlers/socket_flow_handler/filter_auth.rs:72<br/>that the comment says the Godot client always sends, src/handlers/socket_flow_handler/filter_auth.rs:64,<br/>yet against VisionClaw the headset never sends one
    alt dev mode on
        HP->>AU: REST write such as PUT /api/settings/physics, the single REST chokepoint
        AU-->>HP: Ok(DEV_MODE_PUBKEY) src/utils/auth.rs:211-218
        Note over AU: LAN-local FULL bypass, peer-agnostic: no NIP-98,<br/>no token, no peer check. src/utils/auth.rs:203-218
        SV-->>HP: drag / pin / physics writes all accepted
        Note over HP,SV: This is what makes a 100%-local headset over the 25G rail<br/>friction-free for REST writes: XR_NOSTR_SECRET can be empty and the<br/>ephemeral signer still works. graph_scene.gd:1241-1246
    else dev mode off
        AU->>AU: require NIP-98 / bearer / peer check
        alt no or insufficient credential
            AU-->>HP: 401 or 403 - drag/pin refused, read stream still served
            Note over HP: ADR-2108 INVARIANT: the refusal is now VISIBLE in the headset.<br/>_describe_write_failure names the status, the credential path and<br/>the remedy (a separate message covers an unreachable backend), and<br/>hud.flash_notice puts it on the bottom strip of every tab.<br/>graph_scene.gd:2901 graph_scene.gd:2905 graph_scene.gd:2846 hud.gd:2096
        end
    end
    Note over SV: Other dev-bypass sites: REST extractor src/settings/auth_extractor.rs:25,<br/>memory cloud src/handlers/memory_cloud_handler.rs:66, speech socket src/handlers/speech_socket_handler.rs:165
```

**Open (risk):** with dev mode on and `XR_NOSTR_SECRET` empty, the headset's REST writes pass the bypass (`src/utils/auth.rs:211-218`) but its graph socket stays anonymous (`xr-client/rust/src/transport.rs:71-72`, `xr-client/rust/src/transport.rs:165-166`), so every node drag is dropped server-side without a reply (`src/handlers/socket_flow_handler/position_updates.rs:199-201`) and the headset shows nothing; the secret condition predates `6b811f7a1`, so this is not new.

## VC-36.16 Deploy ceremony and the perf-regression gate

```mermaid
sequenceDiagram
    autonumber
    participant OP as Operator (from machinelearn)
    participant HP as HP-Desktop via ssh john@10.10.10.1
    participant SVR as SteamVR + OpenXR runtime
    participant GD as godot --path xr-client
    participant CT as cargo test -p visionclaw-xr-gdext
    participant PR as xr-perf-regression<br/>xr-client/rust/perf-regression/src/main.rs:216

    Note over OP,HP: The verified path onto a headset is NOT the APK - it is the<br/>Compatibility renderer on native X11 driving SteamVR.<br/>docs/XR-client.md 'Deploy ceremony'
    OP->>HP: ssh (25G rail 10.10.10.0/30)
    OP->>SVR: verify SteamVR running, VIVE Pro tracked,<br/>~/.config/openxr/1/active_runtime.json -> SteamVR
    OP->>HP: kill previous godot process
    Note over OP,HP: SEPARATE kill then launch over TWO ssh calls -<br/>a single chained call races the compositor.
    OP->>HP: launch call carrying XAUTHORITY
    Note right of HP: The process has no inherited session, so without<br/>XAUTHORITY Godot cannot open the X11 display for<br/>the SteamVR compositor.
    HP->>GD: XR_BACKEND_WS=ws to the host LAN address (port 4000) XR_NOSTR_SECRET=<hex><br/>godot --path xr-client --rendering-driver opengl3<br/>--display-driver x11 res://scenes/XRBoot.tscn
    GD->>SVR: stereo submission via Compatibility (OpenGL 3)
    Note over GD,SVR: ADR-2032 INVARIANT: only OpenGL3 Compat submits BOTH eyes on<br/>SteamVR + Linux + NVIDIA. Vulkan/Forward+ renders one eye or<br/>fails to composite. Re-test before any renderer change.
    par change process
        OP->>CT: cargo test -p visionclaw-xr-gdext
        Note right of CT: 479 headless tests (361 library + 118 integration) as of 2026-10-07 per<br/>docs/XR-client.md:1257 - no headset/Godot/network. Godot-facing runtime<br/>classes are excluded by cfg(test) (xr-client/rust/src/lib.rs:64-71), so no scene/shader test runs.<br/>The count is the governing document's, not re-run for this stamp: the topic quoted 141 here and 218<br/>at VC-36.18 before the xr-graph and xr-cloud merges.
    and perf gate
        OP->>PR: --current run.json --baseline crates/visionclaw-xr-presence/benches/baseline.json
        Note right of PR: Flags --current --baseline --bench-name --update-baseline<br/>perf-regression/src/main.rs:45-49.<br/>Exit 0 no regression, 1 regression beyond budget, 2 bad input.<br/>Own workspace member so it does not drag in gdext<br/>(xr-client/rust/Cargo.toml:8)
        alt exit 1
            PR-->>OP: regression - markdown report, do not merge
        else exit 0
            PR-->>OP: clean
        end
    end
    Note over OP: Debug library: the headset run loads target/debug, so xr-client/rust/Cargo.toml sets [profile.dev] opt-level = 2.<br/>Measured on HP 2026-10-07: the 13 164-node pack took 18-20 ms unoptimised against 1.8 ms optimised<br/>xr-client/rust/Cargo.toml:74-75. GUT is vendored 9.6.1, 182 pass on HP docs/XR-client.md:1263.
    Note over OP: GUT scene tests xr-client/tests/unit/: test_scene_load, test_hud_tabs,<br/>test_hud_intervention, test_swarm_tab, test_query_builder,<br/>test_graph_agents, test_agent_avatar, test_agent_beam,<br/>test_did_badge, test_xr_config - runner xr-client/tests/run_gut.gd
    Note over OP: ADR-2109 added five and three remain: test_agent_choreography.gd:31 (materialise then travel),<br/>:55 (head turns never move a working agent), :69 (explicit done starts the edge run at<br/>once and stays selectable). Its two test_agent_demo_director cases were deleted<br/>with the client-side director (ADR-2142). ADR-2108 added test_write_denied_notice.gd:16
```

## VC-36.18 Query builder execute path — implemented, acceptance open

```mermaid
sequenceDiagram
    autonumber
    participant HD as _build_query_page<br/>xr-client/scripts/hud.gd:1048
    participant QB as query_builder.gd<br/>xr-client/scripts/query_builder.gd
    participant GS as graph_scene.gd
    participant SV as query_pattern, POST /api/graph/query/pattern<br/>src/handlers/api_handler/graph/mod.rs:1658
    participant PM as plane_manager.gd<br/>xr-client/scripts/plane_manager.gd

    HD->>HD: Execute button built via _press_fire hud.gd:501
    HD->>GS: control_pressed("query_execute")
    GS->>QB: read EXECUTE_ENABLED
    alt EXECUTE_ENABLED false
        QB-->>GS: no-op (BREAK)
    else true
        GS->>SV: POST /api/graph/query/pattern + NIP-98 headers (_auth_headers)
        SV->>SV: fetch_graph_snapshot then caller_view - the pattern runs over the caller's view only<br/>src/handlers/api_handler/graph/mod.rs:1666 src/handlers/api_handler/graph/mod.rs:1172 src/handlers/graph_access.rs:23
        alt 2xx
            SV-->>GS: result set
            GS->>PM: build result planes
            Note over PM: Semantic-plane edges use EDGE_STRIDE = 12<br/>(uniform tint, no per-instance channel)<br/>render_store.rs:120
        else 4xx/5xx
            SV-->>GS: error
            Note right of GS: DIVERGENCE: server correctness and the user-visible<br/>denied/error states are UNVERIFIED. Query Execute is<br/>implemented but runtime acceptance remains open.<br/>docs/XR-client.md:1023 (Query Execute is implemented)
        end
    end
    HD->>HD: Clear button built by _action_btn hud.gd:1779
    Note over HD,PM: DIVERGENCE: no headset/scene/shader test covers this path.<br/>The 218 library tests the 2026-09-04 closeout counted (docs/XR-client.md:1272), 361 at docs/XR-client.md:1257 now,<br/>exclude Godot-facing runtime<br/>classes by cfg(test) and source and helper results do not certify<br/>Godot execution. docs/XR-client.md closeout 2026-09-04
```

## VC-36.19 Co-proximate label fade — second MultiMesh, not a transparent gem.tres

```mermaid
sequenceDiagram
    autonumber
    participant GS as _update_proximity_labels<br/>xr-client/scripts/graph_scene.gd:1883
    participant SL as _set_labelled_nodes<br/>xr-client/scripts/graph_scene.gd:2156
    participant BP as set_labelled<br/>xr-client/rust/src/binary_protocol.rs:2007
    participant RS as RenderStore.set_labelled<br/>xr-client/rust/src/render_store.rs:1343
    participant UM as ConnectionVolumeSource.pack<br/>xr-client/scripts/connection_volume_source.gd:147
    participant NM as NodesMulti (opaque, gem.tres)
    participant NF as each volume's Faded MultiMesh, copied from NodesFadedMulti (gem_faded.tres)<br/>xr-client/scenes/GraphScene.tscn:123

    loop label cadence (~4 Hz)
        GS->>GS: shown = LabelBudget.select over pinned and ranked ids (VC-36.29) graph_scene.gd:1931
        loop each pool slot
            GS->>GS: a = distance-fade alpha for this label
            alt a > LABEL_VISIBLE_MIN_A (0.05), and pinned or within SPHERE_FADE_M 0.5 m graph_scene.gd:884, :1981, label_budget.gd:33
                GS->>SL: labelled.append(node_id)
            end
        end
        GS->>SL: _set_labelled_nodes(labelled)
        SL->>BP: set_labelled(ids) binary_protocol.rs:2007
        BP->>RS: store.set_labelled(&v) render_store.rs:1343
        RS->>RS: labelled.clear() then extend(ids)<br/>label_alpha.entry(id).or_insert(1.0)
    end
    Note over RS: Rather than make gem.tres transparent (all ~2.5k instances in the<br/>transparent queue, no depth write, broken overlap ordering), a labelled<br/>node is routed to a SECOND MultiMesh (NodesFadedMulti) instead.<br/>render_store.rs:113, :117, commit f508937dd
    loop every physics frame, once per visible volume (graph_scene.gd:3110, :1790)
        UM->>RS: build_node_buffer_lod(content_ids, ...) connection_volume_source.gd:158 - the tiered form of build_node_buffer render_store.rs:1930
        RS->>RS: pack_nodes runs step_label_fades() first render_store_pack.rs:129 render_store_pack.rs:130<br/>labelled id: alpha -= LABEL_FADE_STEP (0.1), floor LABEL_FADE_ALPHA (0.3)<br/>dropped id: alpha += LABEL_FADE_STEP, ceil 1.0, pruned once >= 1.0 render_store.rs:1365
        alt cached plan valid for this epoch, id list and sizes
            RS->>RS: pack_nodes_from_plan rewrites transforms only, faded entries take label_alpha render_store_pack.rs:146 render_store_pack.rs:156
        else rebuild
            loop each drawn id (emit_node render_store.rs:2005)
                alt label_alpha has an entry for id
                    RS->>RS: col[3] = alpha, append to faded_buf (NOT buf) render_store.rs:2070
                else
                    RS->>RS: append to buf (opaque, alpha 1.0)
                end
            end
        end
        RS-->>UM: buf (opaque) via build_node_buffer<br/>faded_buf via faded_node_buffer() render_store.rs:1360
        UM->>NM: NodeLod.assign(nodes, nbuf), instance_count = buf.size()/20<br/>connection_volume_source.gd:160, xr-client/scripts/node_lod.gd:53-62
        UM->>NF: NodeLod.assign(faded, faded_node_buffer()), size()/20<br/>connection_volume_source.gd:162-163
    end
    Note over RS: DRIFT: the doc comment on LABEL_FADE_STEP still says build_node_buffer runs at about 45 Hz on the desktop-Vive path,<br/>so 0.1 per build is a 0.15 s ramp. The scene now builds every physics frame, so the ramp is about 0.1 s at 90 Hz<br/>render_store.rs:114-116, graph_scene.gd:3106-3111
    Note over RS,UM: TENSION the fade state is per client, not per pack slot (render_store.rs:1365-1383), and two visible<br/>volumes share the VisionClaw client: knowledge and ontology (graph_scene.gd:94-97). Each pack steps every fade<br/>once (render_store_pack.rs:272, :130), so with both volumes up a fade advances twice per frame, about 0.05 s at 90 Hz.<br/>Source tracing only: the rate on a headset is not measured
    Note over RS,NF: INVARIANT a fading/labelled node still enters drawn +<br/>render_ids/render_positions in emit_node (render_store.rs:2080-2082)<br/>even though it left the opaque buffer, so edges still attach to it and<br/>the interaction ray still hits it.
    Note over NF: node_halo.gdshader multiplies rim ALPHA by COLOR.a<br/>(node_halo.gdshader:61, :93) so the halo dims with the sphere too. Halo quads are a third MultiMesh,<br/>the volume's Halos, filled from the same buffers connection_volume_source.gd:164.<br/>set_labelled reaches only the VisionClaw client (graph_scene.gd:2156-2159), so Codescape nodes never fade
    Note right of RS: Tests - fade-in to 0.3 and stay hittable<br/>labelled_node_moves_to_faded_buffer_and_eases_to_label_alpha<br/>render_store.rs:3940. Fade-out back to opaque<br/>unlabelled_node_fades_back_then_returns_to_opaque_buffer<br/>render_store.rs:3969. Replace-set and clear resets fades<br/>set_labelled_replaces_the_set_and_clear_resets_fades render_store.rs:4002
```

## VC-36.20 Spatial visual experience and its validation boundary (ADR-2107)

```mermaid
flowchart TB
    Scene["Production GraphScene<br/>procedural sky, shadow-free cool key and warm fill<br/>xr-client/scenes/GraphScene.tscn"]
    Env["SpatialEnvironment ready<br/>duplicate scene materials; create grid and one pooled focus marker<br/>xr-client/scripts/spatial_environment.gd:29"]
    Grid["Stationary metre grid<br/>transparent, no depth write; never an opaque floor<br/>xr-client/materials/spatial_floor.gdshader:4"]
    Focus["Current target or grab at actual world depth<br/>hover target expires after 350 ms, an active grab persists until release<br/>marker scale follows the node size factor, floored at 0.5<br/>xr-client/scripts/spatial_environment.gd:109 xr-client/scripts/spatial_environment.gd:122"]
    Comfort["Reduced motion ON by default<br/>Low cost removes grid, halo and MSAA and drops edge base alpha to 0.09<br/>reduced motion stops edge and halo pulses<br/>xr-client/scripts/spatial_environment.gd:128 xr-client/scripts/spatial_environment.gd:160 xr-client/scripts/spatial_environment.gd:161"]
    MSAA["INVARIANT: MSAA stays off whenever the viewport is an XR viewport.<br/>Under Compat the OpenXR multiview swapchain cannot be multisampled,<br/>so forcing it leaves both eye framebuffers incomplete and the headset<br/>goes black. viewport.use_xr is the gate.<br/>xr-client/scripts/spatial_environment.gd:140"]
    Agents["Agent motion respects the same preference: flash skipped, core bob stilled, no hover, fade-relocate replaces travel<br/>status colour and badge retained; alpha is the choreography's<br/>xr-client/scripts/agent_avatar.gd:220 xr-client/scripts/agent_avatar.gd:366 xr-client/scripts/agent_choreography.gd:367 xr-client/scripts/agent_choreography.gd:413 alpha xr-client/scripts/agent_avatar.gd:169"]
    Fixture["Offline production-material fixture<br/>48 opaque + 1 faded node, 60 edges, 4 relation styles<br/>xr-client/tests/spatial_visual_fixture.gd:7"]
    Gallery["Every tab in TAB_ORDER, ten with Forces and Shapes, and the radial gallery<br/>xr-client/tests/visual/hud_gallery.gd:11"]
    Limits["Observed desktop GL renders and GUT assertions<br/>No fresh headset, live authenticated graph or Quest frame-budget certification"]
    Scene --> Env
    Env --> Grid
    Env --> Focus
    Env --> Comfort
    Comfort --> MSAA
    MSAA --> Agents
    Scene --> Fixture
    Fixture --> Limits
    Gallery --> Limits
```

## VC-36.21 Registry to embodiment, the ~4 Hz reconcile (ADR-2109)

```mermaid
sequenceDiagram
    autonumber
    participant PR as _process label cadence<br/>xr-client/scripts/graph_scene.gd:3140
    participant RC as _reconcile_embodiment<br/>xr-client/scripts/graph_scene.gd:4231
    participant BP as BinaryProtocolClient registry<br/>xr-client/rust/src/binary_protocol.rs:1191
    participant SP as _spawn_work_agent<br/>xr-client/scripts/graph_scene.gd:4290
    participant CH as AgentChoreography.add_agent<br/>xr-client/scripts/agent_choreography.gd:81
    participant RL as AgentRole.infer<br/>xr-client/scripts/agent_role.gd:62
    participant DS as _despawn_work_agent<br/>xr-client/scripts/graph_scene.gd:4328

    PR->>RC: every LABEL_UPDATE_SEC (~4 Hz), alongside the proximity labels
    RC->>BP: agent_ids()
    loop each id in the Rust registry
        alt not yet embodied
            RC->>SP: _spawn_work_agent(id, "agent_<id>")
            SP->>SP: instantiate AgentAvatar under AgentsRoot/AgentSpawner<br/>xr-client/scripts/graph_scene.gd:1195
            Note over SP: ADR-2109 INVARIANT: embodiments hang off the UNIT-SCALE<br/>AgentsRoot,<br/>never under GraphRoot. Under GraphRoot they inherited the<br/>fit scale and<br/>a 0.15 m orb rendered as about 5 mm. graph_scene.gd:1194,<br/>:1196
            SP->>RL: infer(display_name, "")
            RL-->>SP: architect, analyst, coder, reviewer, tester, optimizer or<br/>generic<br/>xr-client/scripts/agent_role.gd:18
            SP->>SP: place at _drift_rest_slot(wire_id, cursor), alpha 0<br/>xr-client/scripts/graph_scene.gd:4315
            SP->>CH: add_agent(sid, rim, 0.0, rim)
        end
        RC->>BP: agent_status, agent_target_node, agent_task
        RC->>CH: update_registry(sid, status, has_target, target_id,<br/>target_world)<br/>xr-client/scripts/agent_choreography.gd:131
        Note over RC,CH: to_global converts the streamed server position to world<br/>metres<br/>graph_scene.gd:4205. Server owns WHICH node, status and<br/>task, the<br/>client owns WHERE in the room (ADR-140 motion-authority<br/>split).
        opt role still generic
            RC->>RL: infer(display_name, task) - a live agent may reveal its role in the task line
        end
    end
    loop each embodied id no longer in the registry
        RC->>DS: _despawn_work_agent(id) - remove record, clear its work ring, free the node
    end
    RC->>RC: _apply_drift_rest_slots() - step_agent_drift, then<br/>set_rim_slot(sid, centroid plus drift)<br/>graph_scene.gd:4269 graph_scene.gd:4277
    Note over RC: ADR-2135 INVARIANT: the layout is always separated, so a<br/>work agent rests at the triangle centroid<br/>(server origin) plus its activity drift on a 0.18 m ring by<br/>slot index, and a new agent materialises there.<br/>The old front-facing rim arc around the fitted bounds is<br/>deleted. The scene still only chooses the slot,<br/>agent_choreography.gd stays the sole pose writer<br/>graph_scene.gd:4282 graph_scene.gd:4128,<br/>binary_protocol.rs:1837
    Note over RC: DEBT: the rim vocabulary outlived the geometry - RIM_SLOTS,<br/>_rim_slot_cursor, the rim_index meta and<br/>AgentChoreography.set_rim_slot all now name the drift rest<br/>slot graph_scene.gd:4124 graph_scene.gd:4137<br/>graph_scene.gd:4313 agent_choreography.gd:124
    Note over RC: DEBT: the reconcile walks the whole registry every tick and<br/>rebuilds<br/>nothing incrementally. That is cheap at swarm scale (tens)<br/>and would not<br/>be at graph scale. graph_scene.gd:4231
```

## VC-36.22 The single pose owner, one embodiment's phase machine

```mermaid
stateDiagram-v2
    [*] --> Materialise
    Materialise --> Edge: settled and status already done
    Materialise --> Travel: settled and live with a target
    Materialise --> Rest: settled but idle or no target
    Travel --> Arrive: move finished, arrive event
    Arrive --> Work: ARRIVE_SEC elapsed
    Work --> Travel: registry target changed, hand-off
    Work --> Edge: status becomes done
    Arrive --> Edge: status becomes done
    Travel --> Edge: done transition from the registry
    Rest --> Edge: done transition from the registry
    Work --> Rest: target lost or status idle
    Arrive --> Rest: target lost or status idle
    Edge --> Travel: named again, live with a target
    Edge --> Rest: no longer done and not live
    Rest --> Travel: live again with a target
    note right of Work
        agent_choreography.gd is the SOLE writer of a work-layer
        embodiment's position and alpha. tick agent_choreography.gd:149
        pose agent_choreography.gd:155, applied graph_scene.gd:4403
        phases PH_MATERIALISE to PH_REST agent_choreography.gd:30-35
        transitions _after_settled :232-238, _check_work_transitions :246-255,
        done from update_registry :144-145, edge run exit :307-313, rest exit :258-261
        INVARIANT the proxemics arc places CONVERSATION-layer avatars only
        graph_scene.gd:4077, graph_scene.gd:4375
        INVARIANT nudge targets, idle radial drift and random graph points
        are deleted and must not return, ADR-2109 Consequences
        ADR-2135 activity drift moves only the REST SLOT, never a pose:
        graph_scene.gd:4277, Invariant 8 docs/XR-client.md:1244
    end note
    note left of Edge
        DONE RULE shared with the desktop (ADR-2135, ADR-2109):
        r0 + (edge - r0) * easeOutCubic(elapsed / 30 s) from the centre
        EDGE_SECONDS 30.0 agent_choreography.gd:50, run :306-323
        stops at the scene edge or the head-exclusion sphere, whichever is nearer :318
        holds when the edge is unknown agent_choreography.gd:314-316
        TRAVEL_SPEED 0.32 m per second agent_choreography.gd:45
        SLOT_DISTANCE 0.32 m off the node agent_choreography.gd:56
        HEAD_EXCLUSION 1.0 m agent_choreography.gd:59
        ALPHA_FADED 0.3 for Edge and Rest, agents stay selectable
        agent_choreography.gd:60, :296, :346
        Rest relaxes onto the drift rest slot :352-358
        Reduced motion replaces travel with fade, relocate, fade
        agent_choreography.gd:63, :382-396
        The 30 s evidence TTL is the other exit from working
        render_store.rs:1057
    end note
```

**Drift:** the earlier Complete and Park phases, `COMPLETE_SEC`, `ALPHA_PARKED` and the `park` event are gone at this stamp. A done agent now runs to the scene edge (`xr-client/scripts/agent_choreography.gd:282-298`) and a lapsed one rests on its slot (`xr-client/scripts/agent_choreography.gd:342-347`). The cross-client constant is pinned by `tests/unit/test_agent_edge_run.gd`, per the header at `xr-client/scripts/agent_choreography.gd:20-28`.

## VC-36.23 Demo mode is one server-side producer of 0x23 frames that the headset only controls and mirrors (ADR-2109 D5, ADR-2142)

```mermaid
sequenceDiagram
    autonumber
    participant HD as Swarm tab Start Agent Demo<br/>xr-client/scripts/hud.gd:1375
    participant GP as GraphParity<br/>xr-client/scripts/graph_parity.gd:134
    participant GS as _post_demo<br/>xr-client/scripts/graph_scene.gd:5681
    participant SV as server agents demo<br/>POST and GET /api/agents/demo
    participant BP as BinaryProtocolClient<br/>xr-client/rust/src/binary_protocol.rs:1918
    participant ST as RenderStore agent registry<br/>xr-client/rust/src/render_store.rs:939

    HD->>GP: control_pressed("toggle_demo") graph_parity.gd:134
    GP->>GS: _post_demo(not _demo_running) graph_parity.gd:137-138
    GS->>GS: _gate_for_volume(_focused_volume, BackendCaps.DEMO) refuses it on a backend without the<br/>route graph_scene.gd:5684, gate at graph_scene.gd:2246
    GS->>SV: POST {running: bool} with a NIP-98 header graph_scene.gd:5689
    Note over SV: the demo is produced ONCE, on the server: six agents on the<br/>reserved ids 0x80000000 | 0xD001..0xD006 send ordinary 0x23<br/>frames to EVERY client,<br/>so the headset and the browser see the same agents working<br/>the same nodes. The former client-side AgentDemoDirector<br/>GDScript and its tests are deleted.<br/>docs/adr/ADR-2109-xr-agent-embodiment-single-pose-owner-and-demo-via-ingest.md:146-153
    SV-->>BP: 0x23 frames over /wss, ingested like any agent action (the<br/>SAME door a live swarm uses)
    BP->>ST: record_agent_action
    GS->>GP: _on_demo_completed, apply_demo_body(body) on a 2xx graph_scene.gd:5697
    alt server refused or unreachable
        GS-->>HD: flash_notice with the server error, for example an empty<br/>graph graph_scene.gd:5703-5709
    end
    loop every DEMO_POLL_SEC 4.0
        GP->>SV: GET /api/agents/demo graph_parity.gd:404, graph_parity.gd:421
        SV-->>GP: running true or false, parse_demo_status graph_parity.gd:366
        GP->>GP: apply_demo_body adopts the SERVER state, so a demo started<br/>from the browser reads as running here graph_parity.gd:386
    end
    opt the demo has just STOPPED, from either client
        GP->>BP: retire_agents(DEMO_AGENT_IDS) for exactly the six reserved<br/>ids graph_parity.gd:393-397, binary_protocol.rs:1918
        BP->>ST: remove record and anchor outright render_store.rs:1215
        GP->>GS: _reconcile_embodiment() at once, real agents untouched graph_parity.gd:398-399
    end
    Note over GS,ST: ADR-2109 INVARIANT: no fake position frames, no demo branch<br/>in rendering or protocol. The scene only names the six<br/>agents through demo_role_name (graph_scene.gd:4184)<br/>and the synthetic agents carry NO demo marker in frames,<br/>roster rows or payloads, provenance is the reserved id range<br/>alone.<br/>A demo regression is therefore a production regression.
    Note over BP: server_clock_ms is no longer a producer dependency on the<br/>headset because timestamps come from the server, it is still<br/>exposed binary_protocol.rs:1935
```

**What it shows.** The headset holds only the HUD button and a mirror of the server's demo state. The producer, the six reserved agent ids and the timestamps all live on the server, so both clients see one demo.

**Why it is this way.** The demo used to exist once per client and each copy fed only its own screen. ADR-2142 made it one server-side producer and the 2026-10-19 amendment to ADR-2109 records that D5 ("demo via ingest" from a client director) no longer matches the code while D1-D4 and D6 hold.

## VC-36.24 What an embodied agent is made of, and the effects that follow it

```mermaid
flowchart TB
    subgraph body["Body - one AgentAvatar under AgentsRoot"]
        CORE["Faceted core, CORE_RADIUS 0.14 m<br/>xr-client/scripts/agent_avatar.gd:52"]
        FRAME["Procedural role frame, six shapes plus generic hoop<br/>frame_mesh caches per role<br/>xr-client/scripts/agent_role.gd:80"]
        PTR["Pointer cone, POINTER_LENGTH 0.09 m, aimed at the target<br/>xr-client/scripts/agent_avatar.gd:53, set_aim agent_avatar.gd:267"]
        BADGE["World-size badge, two-letter role code plus name<br/>set_work_identity xr-client/scripts/agent_avatar.gd:240"]
        CAP["Task caption, shown only while announcing or selected<br/>set_caption_visible xr-client/scripts/agent_avatar.gd:227"]
    end
    subgraph fx["Work cues - AgentEffects under AgentEffectsRoot"]
        RING["Pulsing ring on the node being worked<br/>set_ring xr-client/scripts/agent_effects.gd:104"]
        BEADS["Hand-off packet beads on a straight line between the previous and the new target node<br/>send_packets xr-client/scripts/agent_effects.gd:86 xr-client/scripts/agent_choreography.gd:268-271"]
        FLASH["600 ms arrival flash on the body<br/>flash xr-client/scripts/agent_avatar.gd:219"]
        BURST["Completion burst<br/>burst xr-client/scripts/agent_effects.gd:118"]
        BEAM["Work beam on GraphRoot/AgentMulti, origin at the anchor, built each frame from the registry<br/>working or blocked agents only, and only when the target is in the drawn set<br/>xr-client/rust/src/render_store.rs:2177 xr-client/rust/src/render_store.rs:2189-2192 xr-client/rust/src/render_store.rs:2196-2198"]
        POSE["Per-frame pose from the single pose owner: ring only while working with a target<br/>xr-client/scripts/graph_scene.gd:4418-4421"]
    end
    EV["_choreo.take_events drains depart, arrive, handoff and complete - there is no park event now<br/>xr-client/scripts/agent_choreography.gd:174 emitted at xr-client/scripts/agent_choreography.gd:298 xr-client/scripts/agent_choreography.gd:269 xr-client/scripts/agent_choreography.gd:279 xr-client/scripts/agent_choreography.gd:408<br/>the scene consumes only complete, handoff and arrive, depart is ignored<br/>xr-client/scripts/graph_scene.gd:4435-4449"]
    RM["Reduced motion default ON: beads, flash and hover suppressed<br/>xr-client/scripts/graph_scene.gd:4345"]
    TOG["Agents layer toggle hides AgentsRoot, AgentEffectsRoot and the Rust agent-type MultiMesh via set_type_visible 2,<br/>then forces a draw-list rebuild, which also drops beams to hidden targets<br/>choreography, registry, handles and physics keep running<br/>xr-client/scripts/graph_scene.gd:2483 xr-client/scripts/graph_scene.gd:2490 xr-client/scripts/graph_scene.gd:2493"]
    CORE --> FRAME --> PTR --> BADGE --> CAP
    CAP --> EV
    EV -->|"handoff"| BEADS
    EV -->|"arrive"| FLASH
    EV -->|"complete"| BURST
    CAP --> POSE --> RING
    CAP --> BEAM
    BURST --> RM
    RING --> RM
    BEAM --> RM --> TOG
    TOG --> INV["INVARIANT: anchors are visual only. set_agent_anchors never<br/>touches node positions or physics. An agent with no anchor has NO beam:<br/>the old fallback to a streamed node position is removed, and an agent id is never read as a node id.<br/>xr-client/rust/src/render_store.rs:1156 xr-client/rust/src/render_store.rs:2169-2173 xr-client/rust/src/render_store.rs:2196-2198"]
```

**Drift:** the comments at `xr-client/scripts/graph_scene.gd:4440` ("travels the real edge") and `xr-client/scripts/agent_effects.gd:84` ("along a real edge") describe an edge route, but the hand-off event carries only two world points (`xr-client/scripts/agent_choreography.gd:268-271`) and `send_packets` interpolates between them (`xr-client/scripts/agent_effects.gd:93-97`).

## VC-36.25 Desktop-parity layers added after the af3dff3 stamp (WP1-WP8, ADR-2047, ADR-2133, ADR-2134)

```mermaid
flowchart TB
    LIB["gdext crate grows by fourteen modules: attention, beat, cloud_frame, domain_palette, frame_budget,<br/>hulls, memory_cloud, memory_query, memory_route, perf_fixture, semantic, settings_sync, thread_cpu,<br/>and pulse (cfg not test) xr-client/rust/src/lib.rs:27-60.<br/>Since that count, link_index and link_packer joined for the ADR-2164 link layer (xr-client/rust/src/lib.rs:38-39, see VC-45)"]
    WIRE["INVARIANT: no new wire version. Records stay V3 0x03 and V5 0x05, 52 bytes<br/>binary_protocol.rs:25 binary_protocol.rs:29 binary_protocol.rs:32.<br/>The new data arrives as /wss TEXT frames and as REST reads"]

    subgraph text["Text frames routed in _on_graph_text"]
        T1["settingsUpdated, filter_update_success, graphUpdated<br/>graph_parity.gd:24, routed at graph_scene.gd:4553"]
        T2["beatClock, pong, memory_flash<br/>beat_pulse.gd:95, routed at graph_scene.gd:4557"]
        T3["memoryRoute, relayed per session<br/>memory_cloud_layer.gd:473, routed at graph_scene.gd:4567"]
    end

    subgraph rust["Rust side behind BinaryProtocolClient"]
        S1["handle_settings_text applies a nodeFilter to the store, ADR-2047 receive rules<br/>binary_protocol.rs:1579 settings_sync.rs:242; never writes back to the server"]
        S2["graphUpdated coalesced into requestInitialData by RefetchGate<br/>binary_protocol.rs:1621 settings_sync.rs:310"]
        S3["domain colour mode and cluster hulls<br/>binary_protocol.rs:1537 binary_protocol.rs:1712 hulls.rs:461"]
        S4["memory_flash credits the memory vertex of the agent drift field<br/>binary_protocol.rs:1819"]
    end

    subgraph pack["Per-frame pack, one pass shared by every layer"]
        P1["pack plans: transforms only while the visual epoch, id list and sizes hold<br/>render_store_pack.rs:129 render_store_pack.rs:146"]
        P2["tiers: NEAR_CAP gem nodes within NEAR_RADIUS_M, the rest 2-triangle impostors<br/>node_lod.gd:26 node_lod.gd:28 render_store_pack.rs:262 lod.rs:266"]
        P3["edges: NEAR_EDGE_CAP cylinders, the rest far ribbons<br/>node_lod.gd:15 node_lod.gd:17 render_store_pack.rs:363"]
        P4["FrameBudget turns drawn instance counts and the measured frame total into caps<br/>graph_scene.gd:3578 frame_budget_pass.gd:111, now also taking the per-volume demand frame_budget.rs:341"]
    end

    subgraph scene["Scene wiring in _ready"]
        W1["MemoryCloudLayer and MemorySearch under GraphRoot<br/>graph_scene.gd:1316 memory_cloud_layer.gd:42 memory_search.gd:21"]
        W2["GraphParity: colour, filter, hulls<br/>graph_scene.gd:1382 graph_parity.gd:170"]
        W3["impostor and ribbon MultiMeshes, then BeatPulse<br/>graph_scene.gd:1392 graph_scene.gd:1394 graph_scene.gd:1399"]
    end

    LIB --> WIRE
    WIRE --> T1 & T2 & T3
    T1 --> S1 --> S2
    T1 --> S3
    T2 --> S4
    S3 --> P1 --> P2 --> P3 --> P4
    P4 -->|"caps"| W2
    T3 --> W1
    T2 --> W3

    N1["INVARIANT: microphone capture is opt-in per session, shown by the MIC badge, analysed in memory only<br/>and never recorded or transmitted docs/XR-client.md:1250. The beat clock pings every 2 s beat_pulse.gd:32"]
    N2["OPEN: the memory cloud, its relayed route and the beat-clock relay have been exercised by GUT and desktop<br/>screenshots, not in a headset - stereo agreement, sprite legibility and route brightness are unmeasured<br/>docs/XR-client.md:1052"]
    N3["This topic catalogues the seams only. The internals of memory_cloud.rs, memory_route.rs, memory_query.rs,<br/>cloud_frame.rs, beat.rs and semantic.rs, about 9 300 lines at this stamp, are drawn in VC-38<br/>xr-client/rust/src/memory_route.rs:1575 xr-client/rust/src/memory_cloud.rs:883"]
    P4 --> N1
    S4 --> N2
    LIB --> N3
```

**What it shows.** The client's second life since af3dff3: a desktop-parity layer
(domain colour, ADR-2047 settings and filter sync, cluster hulls), a memory
cloud with a relayed query route, a beat clock with memory bursts, and a node
and edge LOD that keeps the triangle budget. None of it changes the graph wire.

**Why it is this way.** The benchmark scene failed its triangle budget at
baseline, 576 000 triangles for 1 000 gem nodes and 1.04 million at 13k nodes
once edges were drawn, so halo quads and node and edge LOD were added; every
benchmark run now stays under 100 000 triangles with at most six draw calls
(`docs/XR-client.md:983`, design at `docs/XR-client.md:669`).

## VC-36.26 The separated triangle, from the server projection to an agent's rest slot (ADR-2135)

```mermaid
flowchart TB
    TRI["visionclaw-tri-layout is the single definition of the geometry:<br/>knowledge, ontology and memory on the vertices of a ground-plane triangle, agents at the centroid<br/>crates/visionclaw-tri-layout/src/lib.rs:1 crates/visionclaw-tri-layout/src/lib.rs:106 (SEPARATION)"]
    SRV["server display projection, always on since the operator decision of 2026-10-08<br/>src/actors/gpu/display_projection.rs:110 src/actors/gpu/display_projection.rs:190"]
    FCA["ForceComputeActor calls project_display on every broadcast<br/>src/actors/gpu/force_compute_actor.rs:23 src/actors/gpu/force_compute_actor.rs:2419"]
    DRIFT["each action credits the vertex of the body it touched, credit halves over time and the agent falls back<br/>to the centroid, the offset recomputed from the current frame<br/>crates/visionclaw-tri-layout/src/drift.rs:1 crates/visionclaw-tri-layout/src/drift.rs:66 crates/visionclaw-tri-layout/src/drift.rs:133"]
    CLI["xr-client links the same crate<br/>xr-client/rust/Cargo.toml:24"]
    BPC["BinaryProtocolClient.agent_drift_offset, fed by applied 0x23 actions and memory flashes<br/>xr-client/rust/src/binary_protocol.rs:1837"]
    SLOT["_apply_drift_rest_slots moves the slot, _drift_rest_slot adds the ring spread<br/>xr-client/scripts/graph_scene.gd:4269 xr-client/scripts/graph_scene.gd:4282"]
    CH["AgentChoreography.set_rim_slot, still the only pose writer<br/>xr-client/scripts/agent_choreography.gd:124"]

    TRI --> SRV --> FCA
    TRI --> DRIFT
    TRI --> CLI --> BPC
    DRIFT --> BPC --> SLOT --> CH

    N1["Replaces the dual-disc projection that lived in force_compute_actor.rs: GraphPopulation, the centroid helper, the<br/>per-node projection and DISC_RIM_RADIUS moved to display_projection.rs, and dual-disc is now an option on top of the<br/>triangle, adding the rim clamp src/actors/gpu/display_projection.rs:30 src/actors/gpu/display_projection.rs:70"]
    N2["INVARIANT: ADR-2109 deleted the idle radial drift as a pose writer and ADR-2135 reintroduces a drift only as the rest slot<br/>a pose parks at. ADR-2109 records that its decision holds because the choreography still writes every pose<br/>docs/adr/ADR-2109-xr-agent-embodiment-single-pose-owner-and-demo-via-ingest.md:33 docs/adr/ADR-2109-xr-agent-embodiment-single-pose-owner-and-demo-via-ingest.md:136"]
    CH --> N2
    SRV --> N1
```

**What it shows.** One geometry crate feeds the server's display projection and
the headset's agent rest positions, so the room and the desktop agree where the
knowledge graph, the ontology and the memory cloud sit.

**Why it is this way.** The operator fixed the layout as always separated on
2026-10-08 (`src/actors/gpu/force_compute_actor.rs:2261-2266`, ADR-2135), so the
earlier front-arc rim slots lost their reason to exist.

## VC-36.27 gdext classes added after the af3dff3 stamp

```mermaid
classDiagram
    class BeatPulse {
        base RefCounted : pulse.rs:41
        beat clock, memory bursts, attention heat glue
    }
    class MemoryFlashCodec {
        base RefCounted : pulse.rs:208
    }
    class MemoryCloud {
        base RefCounted : memory_cloud.rs:883
    }
    class MemoryRoute {
        base RefCounted : memory_route.rs:1575
    }
    class MemoryQuery {
        base RefCounted : memory_query.rs:466
    }
    class CloudFrame {
        base RefCounted : cloud_frame.rs:276
    }
    class FrameBudget {
        base RefCounted : frame_budget.rs:754
        allocate : frame_budget.rs:341
    }
    FrameBudget ..> MemoryCloud : takes the sprite cap and triangle constants
    FrameBudget ..> MemoryRoute : takes the route ring cap
    CloudFrame ..> MemoryCloud : takes the desktop sprite constants
```

**What it shows.** The seven Godot-facing classes the crate gained since the stamp,
each a plain RefCounted handle that GDScript composes; the wire and the render store
stay the single owners of state, as in VC-36.13. The three dependencies between them
are the shared constants drawn below; the ownership, data flow and wire formats behind
each class are VC-38.1 to VC-38.19.

**Why it is this way.** The crate's own header names `semantic`, `attention` and `beat` as
ports of the desktop logic and `pulse` as their Godot glue
(`xr-client/rust/src/lib.rs:17-18`); `pulse` is the module `cargo test` excludes by
`cfg(not(test))` (`xr-client/rust/src/lib.rs:48-49`).


## VC-36.28 Server-state HUD layers added after the a053adc stamp (Forces, Shapes and cycles, physics readout, agent offsets, chase time)

```mermaid
flowchart TB
    HUD["HUD pages built in TAB_ORDER, reached through the seven sections (VC-36.8)<br/>xr-client/scripts/hud.gd:272"]
    GP["GraphParity polls and writes against the server<br/>xr-client/scripts/graph_parity.gd:63"]
    HUD --> F["Forces page (Layout, Semantic forces): one group of semantic-force rows at a time, prev / next / toggle<br/>hud.gd:921, hud.gd:973, graph_parity.gd:140-150"]
    HUD --> S["Shapes page (Layout, Shape) and Cycles toggle<br/>hud.gd:994, hud.gd:2158"]
    HUD --> AO["Agent offsets switch (per viewer, default on)<br/>hud.gd:2698, binary_protocol.rs:1557, render_store.rs:717"]
    F --> PUT["writes through the shared field document to the physics PUT, via _put_physics_body, gated by BackendCaps.PHYSICS_WRITE<br/>graph_parity.gd:352-358"]
    S --> CY["GET /api/analytics/cycles, committed only on a 2xx<br/>graph_scene.gd:2622, graph_scene.gd:2640"]
    CY --> OV["set_cycle_overlay(ids): cycle members glow, every other node dims to 0.22<br/>binary_protocol.rs:2550, render_store.rs:705, render_store.rs:708, render_store.rs:1323"]
    GP --> ST["GET /api/graph/settlement: settling, settled, paused, GPU not reporting or unavailable<br/>graph_parity.gd:59-63"]
    ST --> HUD
    CH["Chase time constant CHASE_MS 120: a node covers 63 percent of the way to its server target in that time at any frame rate<br/>graph_scene.gd:3567, graph_scene.gd:3571"]
    CH -. "pinned equal to the desktop responseMs by a cross-client test" .-> DESK["desktop tweening.responseMs<br/>client/src/features/settings/config/settings.ts"]
    ACK["Why 120: the acknowledged stream (ADR-2141) updates a node about 16 times a second, so the old fixed per-frame ease was slower than it looked"]
    ACK -.-> CH
    CAP["after ADR-2143 A6 every server-bound control above is greyed or hidden when the backend is Codescape: see VC-40"]
    HUD -.-> CAP
```

**What it shows.** The HUD gained three server-driven surfaces since the stamp: a Forces tab that edits the semantic-force parameters, a Shapes tab with a directed-cycle overlay from the server's analysis, and a physics-state readout. Two client-only switches sit beside them, the agent-offsets toggle and the chase time.

**Why it is this way.** The server owns every number the HUD shows. GraphParity reads the physics settings back at boot so the HUD never displays values the GPU is not running (`xr-client/scripts/graph_scene.gd` comments at the physics seed), and the cycle overlay is committed only on a successful response so the button never claims a state the renderer is not showing.

## VC-36.29 Labels are a priority-ranked budget of 8 plus 4 pinned, sized by draw calls (ADR-2188 D5, ADR-2186)

```mermaid
flowchart TD
    POOL["pool of 12 anchors built once, Title plus Detail Label3D each<br/>graph_scene.gd:1850,1851; label_budget.gd:23,24,25"] --> TICK
    TICK["_update_proximity_labels at LABEL_UPDATE_SEC 0.25 s<br/>graph_scene.gd:875,1883"] --> NEAR["nodes_near the eye within 1.5 m of world reach, at most 64<br/>graph_scene.gd:1900,1901; label_budget.gd:27,29"]
    NEAR --> KEEP["keep only overlay-volume nodes in front of the eye and give each a grid cell<br/>graph_scene.gd:1902,1910,1912,1915"]
    KEEP --> PIN["pinned: held, selected, hovered within 250 ms<br/>graph_scene.gd:1918,1920,1922; highlight_writer.gd:25,102"]
    KEEP --> RANK["query-marked ids first, then Rust rank_labels<br/>graph_scene.gd:1925,1927,1928"]
    RANK --> RUST["rank_label_candidates drops unlabelled and greyed nodes,<br/>sorts by centrality, degree, then id<br/>binary_protocol.rs:2490; render_store.rs:1533,1543,1546,1551"]
    PIN --> SEL["select: up to 4 pinned, then ranked until the budget,<br/>skipping a taken grid cell<br/>label_budget.gd:45,51,61,67"]
    RUST --> SEL
    SEL --> DRAW["pinned labels get the detail line, ranked ones a title only<br/>graph_scene.gd:1964,1965,1971"]
    DRAW --> FADE["unpinned labels fade from 1.0 m to 1.5 m<br/>graph_scene.gd:1974,1975; label_budget.gd:117,118"]
    STEP["Encode Labels steps the budget by 2 within 0 to 8, persisted<br/>graph_scene.gd:1995,1996,1997; label_budget.gd:90,94,95"] --> SEL
    WHY["draw calls: every visible Label3D is one call, the scene uses 31 to 34 of 50,<br/>so 8 titles plus 4 pinned titles with details is 16, down from 24<br/>label_budget.gd:16,17,18,19"] -.-> POOL
```

**What it shows.** Labels go to the most central nearby nodes rather than the nearest, with the held, selected and hovered nodes always labelled, and a coarse angular grid stops two titles sharing a cell. **Why it is this way.** ADR-2186 replaced proximity labels with a ranked budget on the desktop; ADR-2188 D5 set the headset default at 8 plus 4 (`docs/adr/ADR-2188-the-xr-client-adopts-the-overview-first-contract-without-bloom.md:35-36`) so that labels stay inside about 16 of PRD-008's 50 draw calls (`xr-client/scripts/label_budget.gd:12-21`), pinned by `xr-client/tests/unit/test_label_budget.gd:11-31` and `xr-client/tests/unit/test_label_budget.gd:107-114`.
**Open:** labels are taken only from the overlay (knowledge) volume, because the anchors sit in GraphRoot's frame (`xr-client/scripts/graph_scene.gd:1902-1911`, `xr-client/scripts/graph_scene.gd:102`), so the ontology and Codescape volumes carry no ranked labels at all; neither ADR-2188 D5 nor the doc says whether that is intended.

## VC-36.30 Headset preferences in xr_prefs.cfg, and Fit undoing the two-hand turn and scale (ADR-2189)

```mermaid
flowchart TD
    FILE["user://xr_prefs.cfg, section xr; a GUT run writes a test file instead<br/>xr_prefs.gd:17,19,20,44,45,47"] --> KEYS
    KEYS["view keys: label budget and text scale, dwell, haptics,<br/>reduced motion, low-cost visuals, VisionClaw colour mode<br/>xr_prefs.gd:156,157,159,160,163,164,165"]
    KEYS --> LIVE["inert under GUT unless a test sets path_override<br/>xr_prefs.gd:171,172,177,185"]
    BOOT["_ready applies the stored choices<br/>graph_scene.gd:1379,2062,2063,2064,2065,2066"] --> KEYS
    ENV["comfort: stored flag, overridden by XR_REDUCED_MOTION or XR_VISUAL_QUALITY<br/>spatial_environment.gd:79,80,81,84,85"] --> KEYS
    PRESS["HUD press: steps write back at once<br/>graph_scene.gd:1991,1997,2000,2004,2007"] --> KEYS
    COL["colour mode stored only while the focused volume is VisionClaw<br/>graph_scene.gd:2075,2076,2080,2082"] --> KEYS
    FIT["Fit header button emits fit_view<br/>hud.gd:528,540"] --> FV["fit_focused_volume<br/>graph_scene.gd:1993,1994,2033"]
    FV --> UNDO["clear the manual and two-hand state, arrangement back to its home transform<br/>graph_scene.gd:2034,2035,2036,2038,1427"]
    UNDO --> REFIT["refit the focused volume, else the knowledge volume; physics untouched<br/>graph_scene.gd:2039,2041,2043; spatial_volume.gd:156,157"]
    NOT["INVARIANT: layout mode is never persisted, because it POSTs<br/>the shared /api/layout/mode and a boot replay would move every viewer<br/>XR-client.md:1129,1130,1131"]
    RS["render scale left unwired until measured on a headset<br/>XR-client.md:1134,1135,1136"]
```

**What it shows.** Every headset-local control the ADR-2189 audit added survives a restart in one per-device file, while Fit is a pure view reset: the arrangement returns home and the focused volume refits, and the physics layout is not touched. **Why it is this way.** These are viewer settings, not shared state, so they belong to the headset (`docs/XR-client.md:1107-1120`); layout mode is shared through `/api/layout/mode`, so re-applying it at boot would move every viewer's graph (`docs/XR-client.md:1129-1132`).
**Open:** Fit resets the arrangement to its home transform without consulting the portal state (`xr-client/scripts/graph_scene.gd:2033-2043`), while entering a space snapshots and locks that arrangement (`xr-client/scripts/space_portals.gd:399`, `xr-client/scripts/space_portals.gd:411`); the Fit test covers only the ring view (`xr-client/tests/unit/test_hud_ia.gd:257`).
**Debt (loose end):** render scale, one of the desktop's five `xr.*` settings the headset never read, is deliberately left unwired because it acts on the 90 fps budget (`docs/XR-client.md:1133-1136`); no control or preference key exists for it (`xr-client/scripts/xr_prefs.gd:155-165`).

## VC-36.31 What the headset's filter and query panel receive from the server at 809a859aa

```mermaid
flowchart TB
    subgraph query["HUD Explore query panel"]
        Q1["preview and Execute both POST /api/graph/query/pattern with NIP-98 headers<br/>xr-client/scripts/graph_scene.gd:5564 xr-client/scripts/graph_scene.gd:5631"]
        Q2["query_pattern takes the caller as OptionalAuth<br/>src/handlers/api_handler/graph/mod.rs:1658-1661"]
        Q3["fetch_graph_view applies caller_view before matching<br/>src/handlers/api_handler/graph/mod.rs:1155 src/handlers/api_handler/graph/mod.rs:1172"]
        Q4["caller_view is visible_subgraph for the caller's pubkey, anonymous sees public only<br/>src/handlers/graph_access.rs:23 src/handlers/graph_access.rs:36-38"]
        Q1 --> Q2 --> Q3 --> Q4
    end
    subgraph onto["Ontology deny-list, not on the headset's path"]
        O1["every ontology read surface scrubs the caller's HiddenClasses<br/>src/services/ontology_visibility.rs:4-8"]
        O2["the headset calls graph, layout, settings, memory-cloud, analytics, agents, splats,<br/>canary and broker-decide routes, never /api/ontology - so O1 never runs for it<br/>xr-client/scripts/backend_caps.gd:30"]
        O1 -.-> O2
    end
    subgraph filt["Node filter: three default copies, one fixture"]
        F1["server NodeFilterSettings default<br/>src/settings/models.rs:192-208"]
        F2["filter_update_from_wire: an omitted key takes that default<br/>src/handlers/socket_flow_handler/filter_auth.rs:12 src/handlers/socket_flow_handler/filter_auth.rs:17"]
        F3["the reply goes through visible_subgraph for the sender, then its own filter<br/>src/actors/client_coordinator_actor.rs:58 src/actors/client_coordinator_actor.rs:1747"]
        F4["XR NodeFilter default, a third hand copy<br/>xr-client/rust/src/settings_sync.rs:44"]
        F7["INVARIANT: the XR default is asserted equal to the fixture default case<br/>xr-client/rust/src/settings_sync.rs:653-654"]
        F5["desktop NODE_FILTER_DEFAULTS<br/>client/src/features/graph/explore/nodeFilter.ts:57"]
        F6["INVARIANT: server wire defaults pinned to the fixture's shipped-default case<br/>src/handlers/socket_flow_handler/filter_auth.rs:588"]
        F1 --> F2 --> F3
        F2 --> F6
        F4 --> F5
        F4 --> F7
    end
    subgraph ack["filter_update_success on the headset"]
        A1["parsed as FilterAck and surfaced as filter_ack<br/>xr-client/rust/src/settings_sync.rs:251 xr-client/rust/src/binary_protocol.rs:1597"]
        A2["DEBT: the headset never sends filter_update, so the server only<br/>ever sends this ack to the sender and the branch is unreachable<br/>src/handlers/socket_flow_handler/filter_auth.rs:324"]
        A1 --> A2
    end
    query --> onto
    filt --> ack
```

**What it shows.** Every server read the headset's Explore panel makes now runs over the caller's own view. The ontology per-caller deny-list never applies, because no headset script calls an ontology route. The node-filter defaults live in three hand copies, all pinned by one fixture. **Why it is this way.** The 2026-11-06 visibility sweep (`ab05ee9d6`) gave every graph read one caller gate. `923feef34` and `f86704227` put the filter reply through the same gate and gave the frame's omitted keys the shipped defaults.
**Debt (parallel):** the node-filter defaults are written by hand three times: server (`src/settings/models.rs:192-208`), desktop (`client/src/features/graph/explore/nodeFilter.ts:57-68`) and headset (`xr-client/rust/src/settings_sync.rs:44-56`). The copies agree, and the shared fixture's default case pins the server wire copy (`src/handlers/socket_flow_handler/filter_auth.rs:588-591`) and the XR copy (`xr-client/rust/src/settings_sync.rs:653-654`). A new filter key still has to be added in three places.
**Debt (loose end):** `filter_update_success` is in the headset's handled set (`xr-client/scripts/graph_parity.gd:24`), but the server sends it to a `/wss` client after its own recompute, on its `filter_update` and on each graph change (`refresh_all_filters`, `src/actors/client_coordinator_actor.rs:134`) (`src/actors/client_coordinator_actor.rs:113-121`, `:1759-1766`; `src/handlers/socket_flow_handler/filter_auth.rs:410-413`), and the headset never sends one.
**Open (risk):** the pattern query sees the public nodes plus those owned by the signing pubkey (`src/handlers/graph_access.rs:36-41`), and power-user status plays no part. A headset with an ephemeral signer, or one granted `DEV_MODE_PUBKEY` under dev mode (`src/settings/auth_extractor.rs:25-30`), therefore counts over the public graph only. A desktop signed in as the owner counts over more, and the query page does not say which view answered.
Resolved at a950f4d80: the headset receipt is now on record. On 2026-11-06, on the HP at main 809a859aa, the OpenXR session was focused and ran at 88-90 fps with all three volumes (knowledge 84 nodes, ontology 9,604, Codescape 4,817), with zero script, parse, shader, heartbeat or reconnect errors over 15 minutes (`docs/adr/ADR-2188-the-xr-client-adopts-the-overview-first-contract-without-bloom.md:52`, `docs/XR-client.md:8`). ADR-2188 activation is now live (`docs/adr/ADR-2188-the-xr-client-adopts-the-overview-first-contract-without-bloom.md:7`).

**Open:** the receipt measures frame rate and errors only. Visual legibility of the rail and pages in the headset is still a human check (`docs/adr/ADR-2188-the-xr-client-adopts-the-overview-first-contract-without-bloom.md:52`).
