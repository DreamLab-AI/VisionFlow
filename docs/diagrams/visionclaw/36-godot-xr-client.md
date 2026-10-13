---
id: VC-36
title: Godot + gdext OpenXR immersive client
area: visionclaw
governing:
  - ../project/docs/XR-client.md
  - ../project/docs/BASELINE-architecture.md
adrs: [ADR-2032, ADR-2033, ADR-2034, ADR-2035, ADR-2036, ADR-2039, ADR-2076, ADR-2079, ADR-2107, ADR-2108, ADR-2109, ADR-2133, ADR-2134, ADR-2135]
sources:
  - ../project/docs/XR-client.md
  - ../project/docs/BASELINE-architecture.md
  - ../project/xr-client/project.godot
  - ../project/xr-client/scripts/xr_boot.gd
  - ../project/xr-client/scripts/graph_scene.gd
  - ../project/xr-client/scripts/hud.gd
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
  - ../project/xr-client/tests/unit
  - ../project/src/actors/gpu/force_compute_actor.rs
  - ../project/src/handlers/layout_handler.rs
  - ../project/src/utils/auth.rs
  - ../project/src/main.rs
  - ../project/docker-compose.unified.yml
  - ../project/crates/visionclaw-xr-presence/benches/baseline.json
  - ../project/src/actors/presence_actor.rs
  - ../project/xr-client/rust/src/avatar_state.rs
  - ../project/xr-client/rust/src/gaze.rs
  - ../project/xr-client/rust/src/interaction.rs
  - ../project/xr-client/rust/src/lod.rs
  - ../project/xr-client/rust/src/proxemics.rs
  - ../project/xr-client/rust/src/selection.rs
  - ../project/xr-client/scripts/plane_manager.gd
  - ../project/xr-client/scripts/query_builder.gd
  - ../project/xr-client/tests/run_gut.gd
  - ../project/xr-client/scenes/GraphScene.tscn
  - ../project/xr-client/materials/node_halo.gdshader
  - ../project/xr-client/scripts/spatial_environment.gd
  - ../project/xr-client/scripts/xr_theme.gd
  - ../project/xr-client/scripts/radial_menu.gd
  - ../project/xr-client/scripts/dwell_reticle.gd
  - ../project/xr-client/scripts/agent_avatar.gd
  - ../project/xr-client/materials/spatial_floor.gdshader
  - ../project/xr-client/materials/edge_flow.gdshader
  - ../project/xr-client/tests/spatial_visual_fixture.gd
  - ../project/xr-client/tests/visual/hud_gallery.gd
  - ../project/xr-client/scripts/agent_choreography.gd
  - ../project/xr-client/scripts/agent_demo_director.gd
  - ../project/xr-client/scripts/agent_role.gd
  - ../project/xr-client/scripts/agent_effects.gd
  - ../project/xr-client/tests/unit/test_agent_choreography.gd
  - ../project/xr-client/tests/unit/test_agent_demo_director.gd
  - ../project/xr-client/tests/unit/test_write_denied_notice.gd
  - ../project/docs/adr/ADR-2108-dev-profile-arms-visionclaw-dev-mode-by-default.md
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
  - ../project/xr-client/rust/src/beat.rs
  - ../project/xr-client/rust/src/semantic.rs
  - ../project/src/handlers/socket_flow_handler/filter_auth.rs
  - ../project/src/settings/auth_extractor.rs
  - ../project/src/handlers/memory_cloud_handler.rs
  - ../project/src/handlers/speech_socket_handler.rs
  - ../project/docs/adr/ADR-2036-defensive-openxr-boot.md
verified_commit: {visionclaw: a053adcef72f3262f57a4b32d902475a1bdc9048}
---

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
    Note over P,XI: ADR-2036 INVARIANT: eye gaze is only QUERIED.<br/>Blind-binding XR_EXT_eye_gaze_interaction in the action map<br/>trips the action-map error on devices that lack it (Quest 3 = false).<br/>xr_boot.gd:40-50
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
    R --> C1
    R --> C2
    R --> C3
    R --> C4
    A -. "DOC-DRIFT" .-> D1
    D1["OPEN by design: project.godot:16 declares Godot 4.3 / Forward Mobile.<br/>The only build that has ever rendered on a headset is 4.6.1-stable on<br/>Compatibility. The file header now says so in text (project.godot:3-6) and<br/>leaves the array alone: config/features is EDITOR-MANAGED metadata, and its<br/>renderer tag is the Quest renderer decision (audit WP0, Invariant 1), so a hand<br/>edit would pre-empt it. docs/XR-client.md:781 and README:15-18 say the same.<br/>Header corrected 2026-10-07, assessed 2026-09-05 at ADR-2079 scope review"]
    AUD["[audio] driver/enable_input = true<br/>project.godot:22 - the engine may open an input device;<br/>capture itself is opt-in per session, docs/XR-client.md:884"]
    RM -. "unexercised" .-> D2
    D2["DIVERGENCE: the .mobile override targets the Quest 3 APK,<br/>which is UNBUILT - no Android NDK is provisioned.<br/>90fps was measured only on VIVE Pro + dual RTX 6000.<br/>docs/BASELINE-architecture.md:209-210"]
    RM -. "TENSION" .-> D3
    D3["TENSION: project.godot:2 header still calls the Quest 3 APK the ship<br/>target, while xr-client/scripts/graph_scene.gd:25-26 reworded Quest 3<br/>standalone as a PLANNED target and names the Vive/SteamVR rig as the<br/>currently deployed one. project.godot:2, graph_scene.gd:25-26"]
```

## VC-36.3 Transport — /wss graph socket connect, subscribe, NIP-98 authenticate

```mermaid
sequenceDiagram
    autonumber
    participant GS as _connect_from_env<br/>xr-client/scripts/graph_scene.gd:1293
    participant CT as connect_to_server<br/>xr-client/scripts/graph_scene.gd:1323
    participant BP as BinaryProtocolClient (gdext)<br/>xr-client/rust/src/binary_protocol.rs:937
    participant TR as spawn_graph_stream<br/>xr-client/rust/src/transport.rs:68
    participant SG as NostrSigner<br/>xr-client/rust/src/signer.rs:25
    participant SV as VisionClaw server /wss

    Note over GS: XR_BACKEND_WS default ws://localhost:4000<br/>GRAPH_STREAM_PATH="/wss" graph_scene.gd:72<br/>PRESENCE_PATH="/ws/presence" graph_scene.gd:73
    GS->>GS: _env_or("XR_BACKEND_WS", DEFAULT_BACKEND_WS).rstrip("/")
    GS->>CT: connect_to_server(base+"/wss", base+"/ws/presence", XR_ROOM_URN, XR_DISPLAY_NAME, XR_NOSTR_SECRET)
    Note over GS,CT: RESOLVED ADR-2076: no token argument. with_token, the token<br/>parameters of spawn_graph_stream / graph_pump / connect_to_url,<br/>and XR_GRAPH_TOKEN are deleted. Query-token auth is gone from<br/>this client - NIP-98 is the only graph-socket credential. Also RESOLVED<br/>ADR-2058 (2026-09-05, see VC-32.1): the server's own ?token= query fallback is<br/>now compiled out of release entirely and survives only dev-auth-gated with a<br/>SECURITY: warning - BASELINE-architecture.md's "Known divergences" bullet<br/>(:248) recording it as still-open was not updated when ADR-2058 landed.
    CT->>BP: connect_to_url(url, nostr_secret_hex) binary_protocol.rs:1049
    BP->>TR: spawn_graph_stream(url, nostr_secret_hex, inbox)
    TR->>SV: connect_async_with_config(url, ws_config(), false)
    alt connect fails
        TR-->>BP: Disconnected pushed into the inbox transport.rs:125 transport.rs:127
        Note right of BP: reconnect scheduled - see VC-36.4
    else connected
        TR->>SV: Text {"type":"requestInitialData"}
        Note over TR: GRAPH_REQUEST_INITIAL transport.rs:44
        TR->>SV: Text GRAPH_SUBSCRIBE (binary position updates)
        Note over TR: GRAPH_SUBSCRIBE transport.rs:45<br/>a failed send of either frame logs and falls through to Disconnected,<br/>Connected is never pushed transport.rs:98 transport.rs:127
        TR->>BP: Connected pushed only after both control frames are sent transport.rs:104
        BP-->>GS: poll emits connection_changed true binary_protocol.rs:1081
        opt nostr_secret_hex non-empty
            TR->>SG: nip98_authenticate_json(full, "GET") signer.rs:116
            SG-->>TR: signed kind-27235 JSON
            TR->>SV: Text authenticate
            Note over TR,SV: Gates server-authoritative node drag/pin.<br/>transport.rs:108-115 - docs/XR-client.md 'Identity and signing'
        end
        loop select over the socket and the outbound channel
            SV-->>TR: Binary becomes Frame
            SV-->>TR: Text classified by classify_graph_text transport.rs:145
            SV-->>TR: Ping answered with Pong transport.rs:148
            BP-->>TR: drag and pin text from the outbound channel transport.rs:158
            TR->>BP: push into inbox VecDeque transport.rs:173
        end
    end
    loop every physics tick
        GS->>BP: poll() - drain inbox, decode, emit signals graph_scene.gd:1400 graph_scene.gd:1404
    end
```

**Invariant:** `Connected` reaches GDScript only after both control frames are on the wire, so a half-open socket never counts as live (`xr-client/rust/src/transport.rs:100-104`, `xr-client/scripts/graph_scene.gd:2858-2859`).
**Open:** the client sends the NIP-98 `authenticate` message and never reads an acknowledgement, so a rejected signature leaves the session anonymous with no signal (`xr-client/rust/src/transport.rs:113`); the visible symptom is the write-denied notice of VC-36.15.

## VC-36.4 Per-socket independent reconnect backoff

```mermaid
stateDiagram-v2
    [*] --> Disconnected
    Disconnected --> Connecting: connect_to_server / retry timer fires
    Connecting --> Backoff: connect error or failed control-frame send, both surface as Disconnected
    Connecting --> Subscribed: requestInitialData and subscribe sent, only then connection_changed true
    Subscribed --> Backoff: socket closed or recv error, surfaced as Disconnected
    Backoff --> Connecting: timer expires
    Backoff --> Backoff: presence only, kicked from presence calls the scheduler directly
    note right of Subscribed
        The client has no Authenticated state.
        NIP-98 authenticate is sent after Connected
        and its acceptance is never read back, transport.rs:108-115
    end note
    note right of Backoff
        _backoff_delay(attempts) graph_scene.gd:2821-2823
        min(RECONNECT_BASE_DELAY_SEC * 2^(attempts-1), RECONNECT_MAX_DELAY_SEC)
        base 2.0s graph_scene.gd:21, cap 60.0s graph_scene.gd:22
        graph socket timer graph_scene.gd:2830, attempts reset on every connected true graph_scene.gd:2865
        presence socket timer graph_scene.gd:2837, kicked entry graph_scene.gd:3188
        INVARIANT the two sockets back off INDEPENDENTLY
        graph_scene.gd:165, :167, :2828, :2835
    end note
```

## VC-36.5 V3 / V5 wire records decoded by the gdext crate

```mermaid
classDiagram
    class FrameHeader {
        +u8 version_byte
        +PROTOCOL_V3 0x03 : binary_protocol.rs:25
        +PROTOCOL_V5 0x05 : binary_protocol.rs:29
        +HEADER_BYTES 1 : binary_protocol.rs:33
        +V5_SEQ_BYTES 8 : binary_protocol.rs:30
        +MSG_AGENT_ACTION 0x23 sibling frame, peeled off before the position decoder : binary_protocol.rs:801
    }
    class V5Envelope {
        +u8 tag_0x05 offset 0
        +u64 broadcast_seq offset 1..9 LE
        +bytes v3_records offset 9..
    }
    class NodeRecord52 {
        +u32 raw_id offset 0 : parse_node_record binary_protocol.rs:750
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

## VC-36.6 Frame ingest — drain, version dispatch, decode and sanitise, then freshness admission

```mermaid
sequenceDiagram
    autonumber
    participant SV as server /wss
    participant IX as transport inbox VecDeque<br/>xr-client/rust/src/transport.rs:71
    participant PL as BinaryProtocolClient::poll<br/>xr-client/rust/src/binary_protocol.rs:1072
    participant EF as emit_frame<br/>xr-client/rust/src/binary_protocol.rs:2140
    participant AD as FreshnessGate::admit_frame<br/>xr-client/rust/src/binary_protocol.rs:694
    participant DP as decode_position_frame_with_sequence<br/>xr-client/rust/src/binary_protocol.rs:458
    participant PR as parse_node_record<br/>xr-client/rust/src/binary_protocol.rs:750
    participant SN as sanitize_node_update<br/>xr-client/rust/src/binary_protocol.rs:713
    participant FG as FreshnessGate::admit<br/>xr-client/rust/src/binary_protocol.rs:638
    participant RS as RenderStore<br/>xr-client/rust/src/render_store.rs:492
    participant GD as graph_scene.gd

    SV-->>IX: Binary frame pushed as Frame<br/>transport.rs:140
    GD->>PL: poll() once per physics tick<br/>graph_scene.gd:1400 graph_scene.gd:1404
    PL->>IX: drain the whole inbox into a Vec<br/>binary_protocol.rs:1077
    PL->>EF: Frame(bytes) goes to emit_frame<br/>binary_protocol.rs:1096
    alt first byte is 0x23
        EF-->>PL: MSG_AGENT_ACTION to the agent registry, early return, see VC-36.9<br/>binary_protocol.rs:2145 binary_protocol.rs:2189
    else position frame
        EF->>AD: admit_frame(bytes, FrameKind::Full)<br/>binary_protocol.rs:2194
        AD->>DP: decode first, gate second<br/>binary_protocol.rs:699
        alt version 0x05
            DP-->>AD: read u64 broadcast_seq at 1..9<br/>binary_protocol.rs:473 binary_protocol.rs:483
        else version 0x03
            DP-->>AD: no sequence, None<br/>binary_protocol.rs:472
        else other
            DP-->>EF: DecodeError::BadVersion<br/>binary_protocol.rs:485
            Note right of EF: error! and return, same arm for TooShort and Misaligned<br/>binary_protocol.rs:2200-2202<br/>the quiet warn-and-ignore path is ingest_frame, tests only<br/>binary_protocol.rs:779
        end
        loop each 52-byte record
            DP->>PR: parse_node_record(chunk)<br/>binary_protocol.rs:498
            PR-->>DP: NodeUpdate
            DP->>SN: sanitize_node_update(u)
            alt non-finite position or velocity, or any position component above WORLD_LIMIT_M
                SN-->>DP: None, record dropped<br/>binary_protocol.rs:714-725
                Note right of SN: WORLD_LIMIT_M = 10_000.0 binary_protocol.rs:38<br/>velocity is checked for finiteness only
            else survives
                SN-->>DP: Some(u), non-finite anomaly, centrality, sssp_distance zeroed<br/>binary_protocol.rs:728-736
            end
        end
        DP-->>AD: Option seq and updates
        AD->>FG: admit(sequence, kind)<br/>binary_protocol.rs:700
        alt no sequence (V3)
            FG-->>AD: AcceptUnsequenced, watermark untouched<br/>binary_protocol.rs:639-644
        else awaiting_resync armed by a Disconnect, kind Full
            FG-->>AD: AcceptResync, watermark re-based even backwards<br/>binary_protocol.rs:647-658
        else seq above watermark
            FG-->>AD: Accept<br/>binary_protocol.rs:672
        else seq equal to watermark
            FG-->>AD: RejectDuplicate<br/>binary_protocol.rs:677
        else seq below watermark
            FG-->>AD: RejectStale<br/>binary_protocol.rs:681
        end
        AD-->>EF: Some(updates) only when is_accepted<br/>binary_protocol.rs:701
        Note right of EF: a rejected frame was already decoded and sanitised, then dropped with warn!<br/>binary_protocol.rs:2196-2198<br/>emit_frame always passes Full, so RejectAwaitingResync is unreachable here<br/>binary_protocol.rs:2194 binary_protocol.rs:659
        EF->>RS: upsert, set_cluster, set_node_kind per update<br/>binary_protocol.rs:2209-2219
        RS-->>GD: node_visuals_updated only when VisualsKey changes or the node is new<br/>binary_protocol.rs:2223
        Note over RS,GD: Per-frame signal traffic stays near zero<br/>once communities stabilise. binary_protocol.rs:124-130
    end
    Note over PL,FG: On Disconnected, poll arms FreshnessGate::reconnect<br/>so the next Full frame re-baselines<br/>binary_protocol.rs:1089 binary_protocol.rs:632
```

**Debt:** `ingest_frame` is a second decode-and-sink path with a quiet `BadVersion` arm that production never calls, only tests (`xr-client/rust/src/binary_protocol.rs:779`, `xr-client/rust/src/binary_protocol.rs:2557`). The live path is `emit_frame` (`xr-client/rust/src/binary_protocol.rs:2140`).
**Invariant:** a frame is fully decoded and sanitised before the freshness gate judges it, so a stale frame costs a full decode and is then dropped (`xr-client/rust/src/binary_protocol.rs:699-700`).

## VC-36.7 RenderStore instance packing and the stride-16 invariant (ADR-2034)

```mermaid
sequenceDiagram
    autonumber
    participant GD as graph_scene.gd _process
    participant UE as _update_edge_multimesh<br/>xr-client/scripts/graph_scene.gd:2025
    participant UB as _update_beam_multimesh<br/>xr-client/scripts/graph_scene.gd:2059
    participant RS as build_edge_buffer<br/>xr-client/rust/src/render_store.rs:1830
    participant SC as edge_style_code<br/>xr-client/rust/src/render_store.rs:127
    participant MM as MultiMesh GraphRoot/EdgesMulti<br/>use_custom_data = true

    Note over RS: NODE_STRIDE = 20 (12 transform + 4 colour + 4 custom) render_store.rs:103<br/>EDGE_STRIDE = 12 (semantic-plane edges, uniform tint) render_store.rs:115<br/>EDGE_STRIDE_TYPED = 16 (12 transform + 4 INSTANCE_CUSTOM) render_store.rs:119
    GD->>UE: _update_edge_multimesh()
    alt edges_multi or multimesh or _binary_client null
        UE-->>GD: return (BREAK)
    end
    UE->>UE: er = EDGE_WORLD_RADIUS / (EDGE_MESH_RADIUS * _graph_scale)
    alt ribbons exist and the gdext build has build_edge_buffer_lod
        UE->>RS: build_edge_buffer_lod(_edge_pairs, er, eye, cylinder cap, near radius) graph_scene.gd:2036
        RS-->>UE: near-tier cylinders, far tier read back via ribbon_edge_buffer()<br/>binary_protocol.rs:1693 binary_protocol.rs:1773
        UE->>UE: far ribbons are a second MultiMesh, stride 16 too graph_scene.gd:2039
    else older gdext build or no LOD nodes
        UE->>RS: build_edge_buffer(_edge_pairs, er) graph_scene.gd:2041
    end
    loop each drawn edge
        RS->>SC: edge_style_code(edge_type)
        SC-->>RS: 2 subclass/taxonomy - 1 typed - 0 untyped
        Note right of SC: local-name match after last / # or :<br/>case-insensitive render_store.rs:121-129
        RS->>RS: pack 12 transform floats + custom[r,g,b,a=style]
    end
    RS-->>UE: PackedFloat32Array (16 floats/instance)
    Note over GD,UE: Both node and edge packs run on EVERY physics frame, no longer on alternating<br/>frames at 45 Hz, because the Rust pack plans make the pair cost about 1 ms<br/>at 13k nodes and 20k edges graph_scene.gd:1426-1432, render_store_pack.rs:1-12
    UE->>UE: count = buf.size() / 16
    Note over UE,MM: ADR-2034 INVARIANT: divide by 16, never 12.<br/>A /12 divisor mis-sizes instance_count, set_buffer rejects<br/>EVERY frame and all edges vanish (regression 63d9bb9b8).<br/>graph_scene.gd:2044-2046
    opt mm.instance_count != count
        UE->>MM: instance_count = count
    end
    opt count > 0
        UE->>MM: buffer = buf (single assignment, no per-instance GDScript)
    end
    GD->>UB: _update_beam_multimesh()
    alt _binary_client lacks build_beam_buffer
        UB-->>GD: return (BREAK - older gdext build)
    end
    UB->>RS: build_beam_buffer(EDGE_WORLD_RADIUS / (BEAM_MESH_RADIUS * _graph_scale))
    RS->>RS: source = published embodiment anchor, else the streamed agent node
    Note over RS: ADR-2109 INVARIANT: the beam starts at the BODY the user sees.<br/>build_beam_buffer prefers agent_anchors over the streamed node<br/>position, so a synthetic agent with no node still beams.<br/>Anchors are visual-only, positions and physics untouched.<br/>render_store.rs:1939, set_agent_anchors render_store.rs:1025
    RS-->>UB: agent-to-target beams, same stride 16, custom r = action code, g/b = taper, a = status code<br/>render_store.rs:1906-1907
    UB->>UB: count = buf.size() / 16
    UB->>MM: GraphRoot/AgentMulti instance_count + buffer
    Note over UB: ADR-2034 server-which/client-where: the server owns WHICH node<br/>an agent works on plus status. The capsule room-position and the<br/>work-beam geometry are client concerns. Beam count = live<br/>working/blocked agents (tens), so this runs every frame.
```

## VC-36.8 HUD, shared instrument theme, eight tabs and press-mode controls

```mermaid
sequenceDiagram
    participant W as Controller ray
    participant HUD as _build_ui<br/>xr-client/scripts/hud.gd:337
    participant Theme as XRTheme.create<br/>xr-client/scripts/xr_theme.gd:29
    participant Press as _press_fire<br/>xr-client/scripts/hud.gd:422
    participant Env as SpatialEnvironment._bind_hud<br/>xr-client/scripts/spatial_environment.gd:150
    HUD->>Theme: create opaque surfaces, typography and focus outlines
    HUD->>HUD: _build_tab_bar over TAB_ORDER<br/>xr-client/scripts/hud.gd:427
    Note over HUD: TAB_ORDER = graph, layout, query, pins, swarm, key, session, help<br/>hud.gd:222. The colour Key tab was added 2026-09-08 (a37433f5d),<br/>so this HUD now has EIGHT pages, not seven.
    HUD->>HUD: _build_key_page mirrors the live palette<br/>xr-client/scripts/hud.gd:992
    Note over HUD: DEBT: the Key swatch constants are duplicated from their<br/>sources by design, the same posture as SWARM_STATUS_COLORS,<br/>so a palette change must be made twice. hud.gd:230, hud.gd:1027
    HUD->>HUD: Query tab holds TWO modes since the stamp - Graph Query and Memory Search with an on-screen keyboard<br/>xr-client/scripts/hud.gd:603 xr-client/scripts/hud.gd:636 xr-client/scripts/hud.gd:677 xr-client/scripts/hud.gd:718
    HUD->>HUD: Graph page gains colour-mode and hull buttons, the beat row sits under it<br/>xr-client/scripts/hud.gd:1485 xr-client/scripts/hud.gd:1153
    Note over HUD: A MIC badge shows while the microphone is on, and the HUD viewport attaches render-on-demand and batching<br/>xr-client/scripts/hud.gd:386 xr-client/scripts/hud.gd:308 xr-client/scripts/hud.gd:309
    HUD->>Press: create buttons and CheckButtons with press firing
    Note over Press: ADR-2033 press behaviour retained. Fresh headset jitter acceptance remains open.
    HUD->>HUD: Help includes reduced-motion and low-cost controls<br/>xr-client/scripts/hud.gd:1193
    Env->>HUD: deferred bind after world-space reparent and synchronise preferences
    W->>HUD: press comfort toggle
    HUD->>Env: control_pressed visual_motion or visual_quality<br/>xr-client/scripts/hud.gd:96
    Env->>Env: apply scene-local reversible rendering budget<br/>xr-client/scripts/spatial_environment.gd:105
    Env->>HUD: set_visual_comfort without re-emitting<br/>xr-client/scripts/hud.gd:1950
    HUD->>HUD: flash_notice writes the bottom strip on EVERY tab<br/>xr-client/scripts/hud.gd:1436, rendered at hud.gd:1750
    Note over HUD: ADR-2108: a rejected server write used to reach only push_warning,<br/>invisible in the headset. The strip is now the in-HMD channel for it.
    Note over HUD: Overflow guard and overlay shield remain explicit<br/>xr-client/scripts/hud.gd:1412<br/>xr-client/scripts/hud.gd:1400
```

## VC-36.9 Agent co-presence and the 0x23 work-beam data plane

```mermaid
sequenceDiagram
    autonumber
    participant SV as server broadcast_to_all<br/>src/utils/binary_protocol.rs MessageType::AgentAction
    participant BP as BinaryProtocolClient::poll<br/>xr-client/rust/src/binary_protocol.rs:1072
    participant DA as decode agent-action batch<br/>xr-client/rust/src/binary_protocol.rs:854
    participant AS as avatar_state AgentAvatarNode<br/>xr-client/rust/src/avatar_state.rs:455
    participant RS as RenderStore beam buffer<br/>xr-client/rust/src/render_store.rs:492
    participant HD as _build_swarm_page<br/>xr-client/scripts/hud.gd:920

    Note over SV,BP: MSG_AGENT_ACTION = 0x23 binary_protocol.rs:801.<br/>Fanned to every /wss client on the SAME binary path as<br/>position frames - separated from 0x03/0x05 by the leading byte.
    SV-->>BP: [0x23][u16 count]([u16 ev_len][ev_len bytes])*
    BP->>DA: decode_agent_actions(frame)
    alt leading byte is not 0x23
        DA-->>BP: None
    else
        loop count events
            DA->>DA: source u32 | target u32 | action u8 | ts u32 | task line
            Note right of DA: build_agent_action_frame mirrors the server<br/>encode_agent_actions binary_protocol.rs:2270
        end
        DA-->>BP: Vec<AgentAction>
    end
    BP->>AS: update per-agent activity + gaze attention
    BP->>BP: record last_agent_action instant binary_protocol.rs:2148
    Note over BP: last_agent_action_age_ms returns -1 if none has arrived<br/>binary_protocol.rs:1448 - P1 liveness probe
    AS->>RS: agent status -> beam target + status code
    RS-->>RS: beam re-routes to the fold representative when the<br/>real target is folded away (render_store.rs:2615 test)
    BP->>HD: swarm roster update
    HD->>HD: SWARM_STATUS_COLORS {0 idle slate, 1 working green,<br/>2 blocked amber-red, 3 done cyan-white} hud.gd:230
    Note over HD: Mirrors render_store::agent_status_color - ADR-140 Pillar 3
    Note over AS,RS: INVARIANT: evidence only moves forward. record_agent_action rejects an<br/>action no newer than the record's evidence_ts (render_store.rs:823) and<br/>set_agent_state_at does the same (render_store.rs:904), both via ts_is_newer<br/>(render_store.rs:480). Stale hits increment agent_actions_stale /<br/>agent_states_stale and expire_stale_agents (render_store.rs:926) ages records out.<br/>RESOLVED: the XR-client.md paragraph that claimed actions were stored without<br/>freshness checks has been superseded in the governing doc
    RS->>RS: agent_hover_offset(target, agent_id, HOVER_RADIUS)
    Note over RS: Hover motion IS implemented. Golden-angle walk 2.3999632 rad keyed by<br/>agent id fans multiple agents around one node instead of stacking them,<br/>lifted by HOVER_LIFT. HOVER_RADIUS = 1.5 render_store.rs:412, :422-432
    Note over RS: Per-node target priority in hunt render_store.rs:1639-1649 - a grabbed node is<br/>pinned, an ACTIVE AGENT hovers at its target (local hover point, not a<br/>server position), a member folding IN chases its representative, everything<br/>else eases to self.targets. DIVERGENCE agent endpoints use LOCAL positions<br/>directly while beam targets are fold-remapped and drawn-gated - the closeout<br/>asks for explicit state precedence, expiry and visible stale/error handling.<br/>docs/XR-client.md 'Estate closeout qualification 2026-09-04'
    Note over RS,HD: ADR-2109 changed WHERE a beam starts, not what feeds the registry.<br/>The registry is now also the source of embodiment - see VC-36.21.<br/>graph_scene.gd:2452
```

## VC-36.10 Presence socket — challenge/auth/joined handshake and 0x43 pose traffic

```mermaid
sequenceDiagram
    autonumber
    participant GS as graph_scene.gd connect_to_server<br/>xr-client/scripts/graph_scene.gd:1323
    participant PC as PresenceClient::handshake<br/>xr-client/rust/src/presence.rs:307
    participant SG as NostrSigner<br/>xr-client/rust/src/signer.rs:34
    participant SV as server /ws/presence<br/>src/actors/presence_actor.rs:380
    participant PN as PresenceClientNode (gdext)<br/>xr-client/rust/src/presence.rs:448
    participant AV as Avatar.tscn / AgentAvatar.tscn

    Note over GS,SV: nginx :3001 does NOT proxy /ws/presence - the client points<br/>at the LAN backend directly. docs/XR-client.md 'Transport'
    GS->>PC: connect(base + "/ws/presence", room_urn, display_name)
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
    Note over PC: XR_NOSTR_SECRET is REQUIRED in practice: NostrAuth.create()<br/>returns an ephemeral signer when empty (graph_scene.gd:455-457),<br/>but an ephemeral key cannot satisfy the presence challenge<br/>or gate drag/pin. docs/XR-client.md INVARIANT 6
```

## VC-36.11 HTTP writes — NIP-98 per-request header and the dev-bearer fallback

```mermaid
sequenceDiagram
    autonumber
    participant HD as hud.gd control_pressed
    participant GS as graph_scene.gd handler
    participant HB as _http_base<br/>xr-client/scripts/graph_scene.gd:1311
    participant AH as _auth_headers<br/>xr-client/scripts/graph_scene.gd:1212
    participant NA as NostrAuth (gdext)<br/>xr-client/rust/src/signer.rs:198
    participant SV as VisionClaw REST

    HD->>GS: control_pressed(action)
    GS->>HB: _http_base()
    alt XR_BACKEND_HTTP set
        HB-->>GS: override.rstrip("/")
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
        NA->>NA: nip98_http_authorization signer.rs:128
        NA-->>AH: "Nostr <b64 kind-27235>"
        AH-->>GS: Authorization: Nostr <b64>
        Note over AH,NA: INVARIANT the signed URL must be the EXACT request URL<br/>including query, or the server tag check fails.<br/>graph_scene.gd:1206-1207, docs/XR-client.md INVARIANT 6
    else no real secret
        AH-->>GS: Authorization: PHYSICS_BEARER
        AH-->>GS: X-Nostr-Pubkey: <pubkey_hex>
        Note over AH: DIVERGENCE the legacy dev bearer path still exists and is<br/>still constructed by the client. The server refuses it for any<br/>non-loopback peer, so the HP headset never passes it.<br/>_nostr_secret_present gates it graph_scene.gd:94, :470, :1214.<br/>docs/XR-client.md:838 (Dev bearer fallback)
    end
    AH-->>GS: Content-Type: application/json
    GS->>SV: POST /api/settings/physics/reset-layout graph_scene.gd:1229
    GS->>SV: PUT /api/settings/physics?graph=knowledge graph_scene.gd:1273
    GS->>SV: GET /api/graph/fold?level=<n> graph_scene.gd:976
    GS->>SV: GET /api/graph/node/<id>/relations graph_scene.gd:3326
    GS->>SV: POST /api/canary/observe/<CANARY_M4_RAY> graph_scene.gd:2747
    Note over GS,SV: hud.gd's intervention decide POST uses the same signing path<br/>via hud.configure_intervention(_http_base(), _nostr_auth) graph_scene.gd:784
    Note over GS,SV: INVARIANT REPEL_K_MAX is 500 and must not exceed the server validator bound (0..500):<br/>a Spread press that clamped above it was rejected with HTTP 400 and the whole PUT discarded.<br/>The bound lives server-side, so the two constants move together graph_scene.gd:299-303, clamp at graph_scene.gd:1168
    SV-->>GS: 401 or 403
    GS->>GS: _describe_write_failure names status, credential path and remedy<br/>xr-client/scripts/graph_scene.gd:1253
    GS->>HD: hud.flash_notice(_last_write_error) graph_scene.gd:1198
    Note over GS,HD: ADR-2108: the HP is never a loopback peer, so before this landed<br/>every server-routed HUD write 401d into an invisible push_warning.<br/>GUT cover: test_write_denied_notice.gd:16, :30, :38
```

## VC-36.12 Constrained layouts and the DAG-rank label accept (ADR-2035)

```mermaid
sequenceDiagram
    autonumber
    participant HD as hud.gd Layout tab
    participant GS as graph_scene.gd
    participant PM as _post_layout_mode<br/>xr-client/scripts/graph_scene.gd:1110
    participant PR as _post_radial<br/>xr-client/scripts/graph_scene.gd:1130
    participant LH as layout_handler.rs<br/>src/handlers/layout_handler.rs:10
    participant FC as force_compute_actor.rs<br/>src/actors/gpu/force_compute_actor.rs:445
    participant DR as compute_dag_ranks<br/>src/actors/gpu/force_compute_actor.rs:464

    Note over GS: LAYOUT_MODES = [forceDirected, hierarchical, radial, spectral,<br/>temporal, clustered] graph_scene.gd:222 - server enumerates the<br/>same list at layout_handler.rs:10
    HD->>GS: control_pressed("layout_cycle")
    GS->>GS: next_idx = (_layout_mode_idx + 1) % LAYOUT_MODES.size() graph_scene.gd:1102
    GS->>PM: _post_layout_mode(LAYOUT_MODES[next_idx])
    PM->>LH: POST /api/layout/mode + NIP-98 header
    HD->>GS: control_pressed("radial:<dagRank|typeTier|ego>")
    GS->>PR: _post_radial(mode)
    PR->>LH: POST /api/layout/radial (server layout_handler.rs:140-171)
    HD->>GS: Hierarchy toggle
    GS->>LH: PUT dagBiasK = 0.6 on / 0.0 off graph_scene.gd:1044
    HD->>GS: Shells +/- nudge
    GS->>LH: PUT dagLevelDistance graph_scene.gd:1056
    LH->>FC: SetRadialLayout{DagRank}
    loop each edge
        FC->>FC: hierarchy_pairs - filter edge.asserts_subsumption
        alt owl_property_iri provenance is rdfs:subClassOf
            FC-->>DR: (parent_idx, child_idx) pair
            Note right of FC: ADR-2035 AMENDED 2026-10-02 (N-14): ranks key on<br/>subClassOf PROVENANCE (Edge.asserts_subsumption),<br/>never on the bare "hierarchical" force label.<br/>force_compute_actor.rs:438-444, :445-463
        else equivalentClass, sameAs, subPropertyOf, instance-of,<br/>domain membership, or no provenance at all
            FC-->>DR: skipped - edge not on the GPU is skipped too
            Note right of FC: the ingest folds equivalence, sameAs, sub-property<br/>and instance-of into the same "hierarchical" label and<br/>domain membership has its own label - ranking any of<br/>them would fabricate layers. force_compute_actor.rs:438-443
        end
    end
    DR->>DR: ranks initialised to -1.0 for every node
    alt num_nodes == 0 or hierarchy_edges empty
        DR-->>LH: all nodes stay -1.0 (unranked)
        Note right of DR: Before the accept landed (73540faa0) EVERY node stayed<br/>unranked, so Radial: DAG and the Hierarchy toggle were<br/>silently inert - reported in-headset as the Radial Shells<br/>buttons "appearing disconnected". force_compute_actor.rs:464-468
    else
        DR-->>LH: ranks assigned
    end
    Note over FC: RESOLVED DOC-DRIFT: docs/XR-client.md 0.1.3 (2026-10-02) now<br/>documents the provenance keying, and the corrected doc-comment<br/>at force_compute_actor.rs:438-444 is committed at this revision -<br/>the old excludes-vs-accepts contradiction is gone from both.
    Note over DR: RESOLVED ADR-2079, SUPERSEDED by the amendment: the closeout's<br/>predicate test is replaced by only_subsumption_provenance_ranks<br/>at force_compute_actor.rs:4481 - bare "hierarchical" with no<br/>subClassOf IRI is now REJECTED, and equivalence, sub-property,<br/>sameAs and instance-of foldings stay out of the rank space.
    Note over FC,DR: RESOLVED DIVERGENCE: the old lossy-accept cost is gone - a<br/>domain-membership edge under "hierarchical" no longer ranks and<br/>can no longer shortcut a subclass chain (BFS lift), asserted by<br/>membership_edges_neither_rank_nor_shortcut_a_subclass_chain at<br/>force_compute_actor.rs:4561 and a_domain_root_never_ranks_below_its_own_members at :4373.
```

## VC-36.13 gdext class surface (GDScript composes, Rust owns the wire)

```mermaid
classDiagram
    class VisionclawXrExtension {
        <<gdextension>>
        on_level_init(InitLevel) : xr-client/rust/src/lib.rs:63
        init_tracing at InitLevel::Scene : xr-client/rust/src/lib.rs:73
    }
    class BinaryProtocolClient {
        base RefCounted : binary_protocol.rs:937
        poll()
        build_edge_buffer(pairs, radius)
        build_beam_buffer(radius)
        last_agent_action_age_ms() : binary_protocol.rs:1446
    }
    class PresenceClientNode {
        base RefCounted : presence.rs:448
        handshake challenge-auth-joined
        0x43 pose - 0x44 agent co-presence
    }
    class NostrAuth {
        base RefCounted : signer.rs:198
        from_secret_hex(hex) : signer.rs:34
        generate() ephemeral : signer.rs:54
        nip98_authenticate_json(url, method) : signer.rs:116
        nip98_http_authorization(url, method) : signer.rs:128
    }
    class XrInteraction {
        base RefCounted : interaction.rs:122
        hand ray cast + pinch detection
        HandRay:24 TargetCandidate:32 RaycastHit:38
    }
    class GazeTracker {
        base RefCounted : gaze.rs:215
        GazeResolver:111 OneEuroFilter:45
        degrades eye-gaze to head when unsupported
    }
    class SelectionArbiterNode {
        base RefCounted : selection.rs:421
        three-resolver arbiter selection.rs:240
        DwellCharger:121 SelectionConfig:81
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
        SpatialVoiceRouterCore:39
        ListenerTransform:26 VoiceTrackState:33
    }
    class RenderStore {
        plain struct : render_store.rs:492
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
        P13["ext/eye_gaze_interaction openxr_action_map.tres:1656"]
        P14["ext/hand_interaction_ext openxr_action_map.tres:1740"]
    end
    acts --> prof
    prof --> XI["project.godot:65<br/>openxr/default_action_map"]
    P13 -. "BOUND" .-> EG["TENSION: ADR-2036 says the extension is never blind-bound (docs/adr/ADR-2036-defensive-openxr-boot.md:31), yet the map binds default_pose to<br/>/user/eyes_ext/input/gaze_ext/pose in the eye-gaze profile, loaded unconditionally.<br/>xr_boot only QUERIES support and the gaze source degrades to head at runtime.<br/>openxr_action_map.tres:1652-1656 project.godot:65 xr_boot.gd:40-50 gaze.rs:138"]
    XI --> CTL["XRController3D left and right read the actions<br/>xr-client/scripts/graph_scene.gd:429 xr-client/scripts/graph_scene.gd:430"]
    CTL -. "GDScript pushes values, no action-map read in Rust" .-> XR["XrInteraction (RefCounted) hand ray + pinch<br/>xr-client/rust/src/interaction.rs:122 xr-client/scripts/graph_scene.gd:459"]
    CTL -.-> GZ["GazeTracker (RefCounted) head/eye ray<br/>xr-client/rust/src/gaze.rs:215 xr-client/scripts/graph_scene.gd:464"]
    CTL -.-> SA["SelectionArbiterNode (RefCounted)<br/>xr-client/rust/src/selection.rs:421 xr-client/scripts/graph_scene.gd:463"]
```

## VC-36.15 VISIONCLAW_DEV_MODE LAN bypass from the headset's perspective (ADR-2039, ADR-2108)

```mermaid
sequenceDiagram
    autonumber
    participant HP as HP-Desktop headset client<br/>godot --path xr-client
    participant CO as dev service environment<br/>docker-compose.unified.yml:95
    participant MN as main (dev banner)<br/>project/src/main.rs:294
    participant RH as enforce_release_env_hygiene<br/>project/src/main.rs:118
    participant AU as verify_access<br/>src/utils/auth.rs:191
    participant WA as handle_authenticate<br/>src/handlers/socket_flow_handler/filter_auth.rs:10
    participant SV as VisionClaw handlers

    rect rgb(240, 232, 210)
        Note over CO: VISIONCLAW_DEV_MODE: "${VISIONCLAW_DEV_MODE:-1}"<br/>docker-compose.unified.yml:95 - DEV SERVICE ONLY.<br/>ADR-2108 flipped the default from :-0 to :-1 on 2026-09-08, so<br/>launch.sh up dev boots an ARMED backend and the operator opts OUT<br/>with VISIONCLAW_DEV_MODE=0. Still not in the *common-environment<br/>anchor and not in the production service, so the default never<br/>reaches a release deploy. docker-compose.unified.yml:88-91
    end
    CO->>RH: process env, release build
    alt the var is merely PRESENT
        RH-->>RH: REFUSE TO BOOT, exit code 2 (ADR-06 D11) project/src/main.rs:136 project/src/main.rs:161
        Note right of RH: project/src/main.rs:109 - a release binary physically cannot<br/>honour SETTINGS_AUTH_BYPASS / VISIONCLAW_DEV_MODE.<br/>Caveat: both services load env_file .env, so the var must<br/>never sit in a .env shared with prod - sharper now that the<br/>compose default supplies it unasked.<br/>docker-compose.unified.yml:92-94
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
    HP->>SV: GET /wss upgrade, no credential yet
    HP->>WA: authenticate frame (NIP-98 signed when XR_NOSTR_SECRET is set) transport.rs:113
    alt dev mode on
        WA-->>HP: authenticate_success as DEV_MODE_PUBKEY, power user, granted on the frame not at connect<br/>src/handlers/socket_flow_handler/filter_auth.rs:28 src/handlers/socket_flow_handler/filter_auth.rs:43
        HP->>AU: REST write such as PUT /api/settings/physics, the single REST chokepoint
        AU-->>HP: Ok(DEV_MODE_PUBKEY) src/utils/auth.rs:211-218
        Note over AU: LAN-local FULL bypass, peer-agnostic: no NIP-98,<br/>no token, no peer check. src/utils/auth.rs:203-218
        SV-->>HP: drag / pin / physics writes all accepted
        Note over HP,SV: This is what makes a 100%-local headset over the 25G rail<br/>friction-free: XR_NOSTR_SECRET can be empty and the<br/>ephemeral signer still works. graph_scene.gd:466-471
    else dev mode off
        AU->>AU: require NIP-98 / bearer / peer check
        alt no or insufficient credential
            AU-->>HP: 401 or 403 - drag/pin refused, read stream still served
            Note over HP: ADR-2108 INVARIANT: the refusal is now VISIBLE in the headset.<br/>_describe_write_failure names the status, the credential path and<br/>the remedy (a separate message covers an unreachable backend), and<br/>hud.flash_notice puts it on the bottom strip of every tab.<br/>graph_scene.gd:1253 graph_scene.gd:1257 graph_scene.gd:1198 hud.gd:1436
        end
    end
    Note over SV: Other dev-bypass sites: REST extractor src/settings/auth_extractor.rs:25,<br/>memory cloud src/handlers/memory_cloud_handler.rs:66, speech socket src/handlers/speech_socket_handler.rs:165
```

## VC-36.16 Deploy ceremony and the perf-regression gate

```mermaid
sequenceDiagram
    autonumber
    participant OP as Operator (from machinelearn)
    participant HP as HP-Desktop via ssh john@10.10.10.1
    participant SVR as SteamVR + OpenXR runtime
    participant GD as godot --path xr-client
    participant CT as cargo test -p visionclaw-xr-gdext
    participant PR as xr-perf-regression<br/>xr-client/rust/perf-regression/src/main.rs:32

    Note over OP,HP: The verified path onto a headset is NOT the APK - it is the<br/>Compatibility renderer on native X11 driving SteamVR.<br/>docs/XR-client.md 'Deploy ceremony'
    OP->>HP: ssh (25G rail 10.10.10.0/30)
    OP->>SVR: verify SteamVR running, VIVE Pro tracked,<br/>~/.config/openxr/1/active_runtime.json -> SteamVR
    OP->>HP: kill previous godot process
    Note over OP,HP: SEPARATE kill then launch over TWO ssh calls -<br/>a single chained call races the compositor.
    OP->>HP: launch call carrying XAUTHORITY
    Note right of HP: The process has no inherited session, so without<br/>XAUTHORITY Godot cannot open the X11 display for<br/>the SteamVR compositor.
    HP->>GD: XR_BACKEND_WS=ws://192.168.2.132 (port 4000) XR_NOSTR_SECRET=<hex><br/>godot --path xr-client --rendering-driver opengl3<br/>--display-driver x11 res://scenes/XRBoot.tscn
    GD->>SVR: stereo submission via Compatibility (OpenGL 3)
    Note over GD,SVR: ADR-2032 INVARIANT: only OpenGL3 Compat submits BOTH eyes on<br/>SteamVR + Linux + NVIDIA. Vulkan/Forward+ renders one eye or<br/>fails to composite. Re-test before any renderer change.
    par change process
        OP->>CT: cargo test -p visionclaw-xr-gdext
        Note right of CT: 479 headless tests (361 library + 118 integration) as of 2026-10-07 per<br/>docs/XR-client.md:891 - no headset/Godot/network. Godot-facing runtime<br/>classes are excluded by cfg(test) (xr-client/rust/src/lib.rs:55-62), so no scene/shader test runs.<br/>The count is the governing document's, not re-run for this stamp: the topic quoted 141 here and 218<br/>at VC-36.18 before the xr-graph and xr-cloud merges.
    and perf gate
        OP->>PR: --current run.json --baseline crates/visionclaw-xr-presence/benches/baseline.json
        Note right of PR: Flags --current --baseline --bench-name --update-baseline<br/>perf-regression/src/main.rs:45-49.<br/>Exit 0 no regression, 1 regression beyond budget, 2 bad input.<br/>Own workspace member so it does not drag in gdext<br/>(xr-client/rust/Cargo.toml:8)
        alt exit 1
            PR-->>OP: regression - markdown report, do not merge
        else exit 0
            PR-->>OP: clean
        end
    end
    Note over OP: Debug library: the headset run loads target/debug, so xr-client/rust/Cargo.toml sets [profile.dev] opt-level = 2.<br/>Measured on HP 2026-10-07: the 13 164-node pack took 18-20 ms unoptimised against 1.8 ms optimised<br/>xr-client/rust/Cargo.toml:65-66. GUT is vendored 9.6.1, 182 pass on HP docs/XR-client.md:897.
    Note over OP: GUT scene tests xr-client/tests/unit/: test_scene_load, test_hud_tabs,<br/>test_hud_intervention, test_swarm_tab, test_query_builder,<br/>test_graph_agents, test_agent_avatar, test_agent_beam,<br/>test_did_badge, test_xr_config - runner xr-client/tests/run_gut.gd
    Note over OP: ADR-2109 added five: test_agent_choreography.gd:30 (materialise then travel),<br/>:54 (head turns never move a working agent), :68 (explicit done parks at<br/>0.3 alpha and stays selectable), test_agent_demo_director.gd:75 (byte-exact<br/>0x23 layout), :105 (real targets, keep-alive, Stop retires exactly the demo<br/>ids). ADR-2108 added test_write_denied_notice.gd:16
```

## VC-36.18 Query builder execute path — implemented, acceptance open

```mermaid
sequenceDiagram
    autonumber
    participant HD as _build_query_page<br/>xr-client/scripts/hud.gd:603
    participant QB as query_builder.gd<br/>xr-client/scripts/query_builder.gd
    participant GS as graph_scene.gd
    participant SV as server /api/graph/query/pattern
    participant PM as plane_manager.gd<br/>xr-client/scripts/plane_manager.gd

    HD->>HD: Execute button built via _press_fire hud.gd:422
    HD->>GS: control_pressed("query_execute")
    GS->>QB: read EXECUTE_ENABLED
    alt EXECUTE_ENABLED false
        QB-->>GS: no-op (BREAK)
    else true
        GS->>SV: POST /api/graph/query/pattern + NIP-98 headers (_auth_headers)
        alt 2xx
            SV-->>GS: result set
            GS->>PM: build result planes
            Note over PM: Semantic-plane edges use EDGE_STRIDE = 12<br/>(uniform tint, no per-instance channel)<br/>render_store.rs:115
        else 4xx/5xx
            SV-->>GS: error
            Note right of GS: DIVERGENCE: server correctness and the user-visible<br/>denied/error states are UNVERIFIED. Query Execute is<br/>implemented but runtime acceptance remains open.<br/>docs/XR-client.md:827 (Query Execute is implemented)
        end
    end
    HD->>HD: Clear button built by _action_btn hud.gd:1289
    Note over HD,PM: DIVERGENCE: no headset/scene/shader test covers this path.<br/>The 218 library tests the 2026-09-04 closeout counted (docs/XR-client.md:906), 361 at docs/XR-client.md:891 now,<br/>exclude Godot-facing runtime<br/>classes by cfg(test) and source and helper results do not certify<br/>Godot execution. docs/XR-client.md closeout 2026-09-04
```

## VC-36.19 Co-proximate label fade — second MultiMesh, not a transparent gem.tres

```mermaid
sequenceDiagram
    autonumber
    participant GS as _update_proximity_labels<br/>xr-client/scripts/graph_scene.gd:638
    participant SL as _set_labelled_nodes<br/>xr-client/scripts/graph_scene.gd:739
    participant BP as set_labelled<br/>xr-client/rust/src/binary_protocol.rs:1635
    participant RS as RenderStore.set_labelled<br/>xr-client/rust/src/render_store.rs:1152
    participant UM as _update_multimesh<br/>xr-client/scripts/graph_scene.gd:1953
    participant NM as NodesMulti (opaque, gem.tres)
    participant NF as NodesFadedMulti (transparent, gem_faded.tres)<br/>xr-client/scenes/GraphScene.tscn:122

    loop label cadence (~4 Hz)
        GS->>GS: shown = ids whose proximity/grab label text != ""
        loop each pool slot
            GS->>GS: a = distance-fade alpha for this label
            alt a > LABEL_VISIBLE_MIN_A (0.05) graph_scene.gd:151, :724
                GS->>SL: labelled.append(node_id)
            end
        end
        GS->>SL: _set_labelled_nodes(labelled)
        SL->>BP: set_labelled(ids) binary_protocol.rs:1635
        BP->>RS: store.set_labelled(&v) render_store.rs:1152
        RS->>RS: labelled.clear() then extend(ids)<br/>label_alpha.entry(id).or_insert(1.0)
    end
    Note over RS: Rather than make gem.tres transparent (all ~2.5k instances in the<br/>transparent queue, no depth write, broken overlap ordering), a labelled<br/>node is routed to a SECOND MultiMesh (NodesFadedMulti) instead.<br/>render_store.rs:108, :112, commit f508937dd
    loop every physics frame (graph_scene.gd:1430)
        UM->>RS: build_node_buffer(ids, ...) render_store.rs:1692 - or build_node_buffer_lod when impostors exist graph_scene.gd:1967
        RS->>RS: pack_nodes runs step_label_fades() first render_store_pack.rs:125 render_store_pack.rs:126<br/>labelled id: alpha -= LABEL_FADE_STEP (0.1), floor LABEL_FADE_ALPHA (0.3)<br/>dropped id: alpha += LABEL_FADE_STEP, ceil 1.0, pruned once >= 1.0 render_store.rs:1174
        alt cached plan valid for this epoch, id list and sizes
            RS->>RS: pack_nodes_from_plan rewrites transforms only, faded entries take label_alpha render_store_pack.rs:142 render_store_pack.rs:152
        else rebuild
            loop each drawn id (emit_node render_store.rs:1766)
                alt label_alpha has an entry for id
                    RS->>RS: col[3] = alpha, append to faded_buf (NOT buf) render_store.rs:1813
                else
                    RS->>RS: append to buf (opaque, alpha 1.0)
                end
            end
        end
        RS-->>UM: buf (opaque) via build_node_buffer<br/>faded_buf via faded_node_buffer() render_store.rs:1169
        UM->>NM: buffer = buf, instance_count = buf.size()/20<br/>graph_scene.gd:1972-1975
        UM->>NF: buffer = faded_buf, instance_count = faded_buf.size()/20<br/>graph_scene.gd:1984-1988
    end
    Note over RS: DRIFT: the doc comment on LABEL_FADE_STEP still says build_node_buffer runs at about 45 Hz on the desktop-Vive path,<br/>so 0.1 per build is a 0.15 s ramp. The scene now builds every physics frame, so the ramp is about 0.1 s at 90 Hz<br/>render_store.rs:109-111, graph_scene.gd:1426-1432
    Note over RS,NF: INVARIANT a fading/labelled node still enters drawn +<br/>render_ids/render_positions in emit_node (render_store.rs:1823-1825)<br/>even though it left the opaque buffer, so edges still attach to it and<br/>the interaction ray still hits it.
    Note over NF: node_halo.gdshader multiplies rim ALPHA by COLOR.a<br/>(node_halo.gdshader:61, :93) so the halo dims with the sphere too. Halo quads are a third MultiMesh,<br/>NodesHaloMulti, filled from the same buffers graph_scene.gd:1993.
    Note right of RS: Tests - fade-in to 0.3 and stay hittable<br/>labelled_node_moves_to_faded_buffer_and_eases_to_label_alpha<br/>render_store.rs:3307. Fade-out back to opaque<br/>unlabelled_node_fades_back_then_returns_to_opaque_buffer<br/>render_store.rs:3336. Replace-set and clear resets fades<br/>set_labelled_replaces_the_set_and_clear_resets_fades render_store.rs:3369
```

## VC-36.20 Spatial visual experience and its validation boundary (ADR-2107)

```mermaid
flowchart TB
    Scene["Production GraphScene<br/>procedural sky, shadow-free cool key and warm fill<br/>xr-client/scenes/GraphScene.tscn"]
    Env["SpatialEnvironment ready<br/>duplicate scene materials; create grid and one pooled focus marker<br/>xr-client/scripts/spatial_environment.gd:25"]
    Grid["Stationary metre grid<br/>transparent, no depth write; never an opaque floor<br/>xr-client/materials/spatial_floor.gdshader:4"]
    Focus["Current target or grab at actual world depth<br/>hover target expires after 350 ms, an active grab persists until release<br/>marker scale follows the node size factor, floored at 0.5<br/>xr-client/scripts/spatial_environment.gd:89 xr-client/scripts/spatial_environment.gd:99"]
    Comfort["Reduced motion ON by default<br/>Low cost removes grid, halo and MSAA and drops edge base alpha to 0.09<br/>reduced motion stops edge and halo pulses<br/>xr-client/scripts/spatial_environment.gd:105 xr-client/scripts/spatial_environment.gd:137 xr-client/scripts/spatial_environment.gd:138"]
    MSAA["INVARIANT: MSAA stays off whenever the viewport is an XR viewport.<br/>Under Compat the OpenXR multiview swapchain cannot be multisampled,<br/>so forcing it leaves both eye framebuffers incomplete and the headset<br/>goes black. viewport.use_xr is the gate.<br/>xr-client/scripts/spatial_environment.gd:117"]
    Agents["Agent motion respects the same preference: flash skipped, core bob stilled, no hover, fade-relocate replaces travel<br/>status colour and badge retained; alpha is the choreography's<br/>xr-client/scripts/agent_avatar.gd:220 xr-client/scripts/agent_avatar.gd:366 xr-client/scripts/agent_choreography.gd:266 xr-client/scripts/agent_choreography.gd:315 alpha xr-client/scripts/agent_avatar.gd:169"]
    Fixture["Offline production-material fixture<br/>48 opaque + 1 faded node, 60 edges, 4 relation styles<br/>xr-client/tests/spatial_visual_fixture.gd:7"]
    Gallery["Every tab in TAB_ORDER, eight with the Key tab, and the radial gallery<br/>xr-client/tests/visual/hud_gallery.gd:11"]
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
    participant PR as _process label cadence<br/>xr-client/scripts/graph_scene.gd:1451
    participant RC as _reconcile_embodiment<br/>xr-client/scripts/graph_scene.gd:2452
    participant BP as BinaryProtocolClient registry<br/>xr-client/rust/src/binary_protocol.rs:937
    participant SP as _spawn_work_agent<br/>xr-client/scripts/graph_scene.gd:2511
    participant CH as AgentChoreography.add_agent<br/>xr-client/scripts/agent_choreography.gd:65
    participant RL as AgentRole.infer<br/>xr-client/scripts/agent_role.gd:62
    participant DS as _despawn_work_agent<br/>xr-client/scripts/graph_scene.gd:2549

    PR->>RC: every LABEL_UPDATE_SEC (~4 Hz), alongside the proximity labels
    RC->>BP: agent_ids()
    loop each id in the Rust registry
        alt not yet embodied
            RC->>SP: _spawn_work_agent(id, "agent_<id>")
            SP->>SP: instantiate AgentAvatar under AgentsRoot/AgentSpawner<br/>xr-client/scripts/graph_scene.gd:427
            Note over SP: ADR-2109 INVARIANT: embodiments hang off the UNIT-SCALE AgentsRoot,<br/>never under GraphRoot. Under GraphRoot they inherited the fit scale and<br/>a 0.15 m orb rendered as about 5 mm. graph_scene.gd:426, :428
            SP->>RL: infer(display_name, "")
            RL-->>SP: architect, analyst, coder, reviewer, tester, optimizer or generic<br/>xr-client/scripts/agent_role.gd:18
            SP->>SP: place at _drift_rest_slot(wire_id, cursor), alpha 0<br/>xr-client/scripts/graph_scene.gd:2536
            SP->>CH: add_agent(sid, rim, 0.0, rim)
        end
        RC->>BP: agent_status, agent_target_node, agent_task
        RC->>CH: update_registry(sid, status, has_target, target_id, target_world)<br/>xr-client/scripts/agent_choreography.gd:108
        Note over RC,CH: to_global converts the streamed server position to world metres<br/>graph_scene.gd:2441. Server owns WHICH node, status and task, the<br/>client owns WHERE in the room (ADR-140 motion-authority split).
        opt role still generic
            RC->>RL: infer(display_name, task) - a live agent may reveal its role in the task line
        end
    end
    loop each embodied id no longer in the registry
        RC->>DS: _despawn_work_agent(id) - remove record, clear its work ring, free the node
    end
    RC->>RC: _apply_drift_rest_slots() - step_agent_drift, then set_rim_slot(sid, centroid plus drift)<br/>graph_scene.gd:2490 graph_scene.gd:2498
    Note over RC: ADR-2135 INVARIANT: the layout is always separated, so a work agent rests at the triangle centroid<br/>(server origin) plus its activity drift on a 0.18 m ring by slot index, and a new agent materialises there.<br/>The old front-facing rim arc around the fitted bounds is deleted. The scene still only chooses the slot,<br/>agent_choreography.gd stays the sole pose writer graph_scene.gd:2503 graph_scene.gd:2378, binary_protocol.rs:1486
    Note over RC: DEBT: the rim vocabulary outlived the geometry - RIM_SLOTS, _rim_slot_cursor, the rim_index meta and<br/>AgentChoreography.set_rim_slot all now name the drift rest slot graph_scene.gd:2374 graph_scene.gd:2386 graph_scene.gd:2534 agent_choreography.gd:101
    Note over RC: DEBT: the reconcile walks the whole registry every tick and rebuilds<br/>nothing incrementally. That is cheap at swarm scale (tens) and would not<br/>be at graph scale. graph_scene.gd:2452
```

## VC-36.22 The single pose owner, one embodiment's phase machine

```mermaid
stateDiagram-v2
    [*] --> Materialise
    Materialise --> Travel: settled and live with a target
    Materialise --> Rest: settled but idle or no target
    Travel --> Arrive: move finished
    Arrive --> Work: ARRIVE_SEC elapsed
    Work --> Travel: registry target changed, hand-off
    Work --> Complete: status becomes done
    Work --> Park: target lost or status idle
    Arrive --> Complete: status becomes done
    Complete --> Park: COMPLETE_SEC elapsed
    Park --> Travel: live again with a target
    Park --> Rest: move finished
    Rest --> Travel: live again with a target
    note right of Work
        agent_choreography.gd is the SOLE writer of a work-layer
        embodiment's position and alpha. tick agent_choreography.gd:120
        pose agent_choreography.gd:126, applied graph_scene.gd:2619
        INVARIANT the proxemics arc places CONVERSATION-layer avatars only
        graph_scene.gd:2326, graph_scene.gd:2592
        INVARIANT nudge targets, idle radial drift and random graph points
        are deleted and must not return, ADR-2109 Consequences
        ADR-2135 activity drift moves only the REST SLOT, never a pose:
        graph_scene.gd:2498, Invariant 8 docs/XR-client.md:878
    end note
    note left of Park
        TRAVEL_SPEED 0.32 m per second agent_choreography.gd:36
        SLOT_DISTANCE 0.32 m off the node agent_choreography.gd:42
        HEAD_EXCLUSION 1.0 m agent_choreography.gd:45
        ALPHA_PARKED 0.3, parked agents stay selectable
        agent_choreography.gd:46
        Reduced motion replaces travel with fade, relocate, fade
        agent_choreography.gd:49
        Explicit done is the completion beat, the 30 s evidence TTL
        is the only other exit from working
        render_store.rs:926
    end note
```

## VC-36.23 Demo mode is a producer of synthetic 0x23 frames through the real door (ADR-2109)

```mermaid
sequenceDiagram
    autonumber
    participant HD as Swarm tab Start Agent Demo<br/>xr-client/scripts/hud.gd:926
    participant GS as _toggle_demo<br/>xr-client/scripts/graph_scene.gd:3856
    participant DD as AgentDemoDirector.start<br/>xr-client/scripts/agent_demo_director.gd:104
    participant EN as encode_action_frame<br/>xr-client/scripts/agent_demo_director.gd:325
    participant BP as BinaryProtocolClient.ingest<br/>xr-client/rust/src/binary_protocol.rs:1431
    participant ST as RenderStore agent registry<br/>xr-client/rust/src/render_store.rs:808

    HD->>GS: control_pressed("toggle_demo") graph_scene.gd:878
    GS->>DD: start(_binary_client, _server_to_world)
    Note over GS,DD: GraphScene _physics_process calls _demo.tick(delta) every frame and drives every phase below<br/>xr-client/scripts/graph_scene.gd:1438 xr-client/scripts/agent_demo_director.gd:144
    DD->>DD: _build_adjacency from get_edges, _pick_candidates, _separated_starts<br/>xr-client/scripts/agent_demo_director.gd:224
    alt no candidate nodes
        DD-->>GS: false when no candidate or the client lacks ingest, HUD flashes "Demo needs a loaded graph"<br/>xr-client/scripts/agent_demo_director.gd:107 xr-client/scripts/agent_demo_director.gd:114 graph_scene.gd:3862
    else running
        loop per role, staggered STAGGER_SEC
            DD->>EN: {source: 0x80000000 | 0xD001..0xD006, target: a REAL node id,<br/>action, ts, duration_ms, payload {"intent": caption}}
            Note over EN: [0x23][u16 count]([u16 len][u32 src][u32 tgt][u8 action]<br/>[u32 ts][u16 dur][payload])* - byte-for-byte the server's own<br/>encode_agent_actions layout. agent_demo_director.gd:325
            EN-->>DD: PackedByteArray
            DD->>BP: ingest(frame)
            BP->>ST: record_agent_action - the SAME door a live swarm uses
            DD->>DD: keep alive every KEEPALIVE_SEC 5.0 in the WORK and FOLLOW phases only, none while resting<br/>xr-client/scripts/agent_demo_director.gd:30 xr-client/scripts/agent_demo_director.gd:153-157 xr-client/scripts/agent_demo_director.gd:165
        end
        DD->>BP: server_clock_ms() stamps ts from the ADR-2034 clock anchor once a 0x23 action has anchored it<br/>xr-client/rust/src/binary_protocol.rs:1578
        Note over DD,BP: INVARIANT demo evidence orders and expires alongside live evidence<br/>because it rides the same clock. agent_demo_director.gd:211<br/>OPEN: before any action has anchored the clock server_clock_ms is -1 and the first frames fall back<br/>to the local tick clock, an unrelated time base. agent_demo_director.gd:215-216 binary_protocol.rs:1573
        DD->>BP: apply_agent_state(wire, "done", caption) on completion<br/>xr-client/rust/src/binary_protocol.rs:1591
    end
    HD->>GS: press again
    GS->>DD: stop()
    Note over GS,HD: after start or stop the toggle updates the HUD button and reconciles embodiment at once<br/>xr-client/scripts/graph_scene.gd:3863-3866
    DD->>BP: retire_agents(demo_wire_ids()) binary_protocol.rs:1567
    BP->>ST: remove record and anchor outright render_store.rs:1049
    Note over DD,ST: ADR-2109 INVARIANT: no fake position frames, no demo branch in rendering or protocol<br/>(the scene only ticks and toggles the director and resolves role names through<br/>display_name_for graph_scene.gd:2431), and the synthetic agents carry NO demo marker in names,<br/>frames, roster rows or payloads. Provenance is the reserved id<br/>range 0xD001-0xD0FF alone. agent_demo_director.gd:25, agent_demo_director.gd:87
    Note over DD,ST: A demo regression is therefore a production regression: the demo<br/>exercises exactly the live path. Cover: byte-exact layout<br/>test_agent_demo_director.gd:75, Stop retires exactly the demo ids<br/>test_agent_demo_director.gd:105
```

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
        BEADS["Hand-off packet beads on a straight line between the previous and the new target node<br/>send_packets xr-client/scripts/agent_effects.gd:86 xr-client/scripts/agent_choreography.gd:236-239"]
        FLASH["600 ms arrival flash on the body<br/>flash xr-client/scripts/agent_avatar.gd:219"]
        BURST["Completion burst<br/>burst xr-client/scripts/agent_effects.gd:118"]
        BEAM["Work beam on GraphRoot/AgentMulti, origin at the anchor, built each frame from the registry<br/>working or blocked agents only, and only when the target is in the drawn set<br/>xr-client/rust/src/render_store.rs:1920 xr-client/rust/src/render_store.rs:1933 xr-client/rust/src/render_store.rs:1939"]
        POSE["Per-frame pose from the single pose owner: ring only while working with a target<br/>xr-client/scripts/graph_scene.gd:2634-2637"]
    end
    EV["_choreo.take_events drains depart, arrive, handoff, complete, park<br/>xr-client/scripts/agent_choreography.gd:145 emitted at xr-client/scripts/agent_choreography.gd:216 xr-client/scripts/agent_choreography.gd:237 xr-client/scripts/agent_choreography.gd:247 xr-client/scripts/agent_choreography.gd:257 xr-client/scripts/agent_choreography.gd:310<br/>the scene consumes only complete, handoff and arrive, depart and park are ignored<br/>xr-client/scripts/graph_scene.gd:2651-2665"]
    RM["Reduced motion default ON: beads, flash and hover suppressed<br/>xr-client/scripts/graph_scene.gd:2566"]
    TOG["Agents layer toggle hides AgentsRoot, AgentEffectsRoot and the Rust agent-type MultiMesh via set_type_visible 2,<br/>then forces a draw-list rebuild, which also drops beams to hidden targets<br/>choreography, registry, handles and physics keep running<br/>xr-client/scripts/graph_scene.gd:912 xr-client/scripts/graph_scene.gd:919 xr-client/scripts/graph_scene.gd:922"]
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
    TOG --> INV["INVARIANT: anchors are visual only. set_agent_anchors never<br/>touches node positions or physics, and an agent with no anchor<br/>falls back to its streamed node position, and its beam is skipped if it has neither.<br/>xr-client/rust/src/render_store.rs:1025 xr-client/rust/src/render_store.rs:1939-1943"]
```

**Drift:** the comments at `xr-client/scripts/graph_scene.gd:2656` ("travels the real edge") and `xr-client/scripts/agent_effects.gd:84` ("along a real edge") describe an edge route, but the hand-off event carries only two world points (`xr-client/scripts/agent_choreography.gd:236-239`) and `send_packets` interpolates between them (`xr-client/scripts/agent_effects.gd:93-97`).

## VC-36.25 Desktop-parity layers added after the af3dff3 stamp (WP1-WP8, ADR-2047, ADR-2133, ADR-2134)

```mermaid
flowchart TB
    LIB["gdext crate grows by fourteen modules: attention, beat, cloud_frame, domain_palette, frame_budget,<br/>hulls, memory_cloud, memory_query, memory_route, perf_fixture, semantic, settings_sync, thread_cpu,<br/>and pulse (cfg not test) xr-client/rust/src/lib.rs:24-51"]
    WIRE["INVARIANT: no new wire version. Records stay V3 0x03 and V5 0x05, 52 bytes<br/>binary_protocol.rs:25 binary_protocol.rs:29 binary_protocol.rs:32.<br/>The new data arrives as /wss TEXT frames and as REST reads"]

    subgraph text["Text frames routed in _on_graph_text"]
        T1["settingsUpdated, filter_update_success, graphUpdated<br/>graph_parity.gd:23, routed at graph_scene.gd:2767"]
        T2["beatClock, pong, memory_flash<br/>beat_pulse.gd:95, routed at graph_scene.gd:2771"]
        T3["memoryRoute, relayed per session<br/>memory_cloud_layer.gd:473, routed at graph_scene.gd:2781"]
    end

    subgraph rust["Rust side behind BinaryProtocolClient"]
        S1["handle_settings_text applies a nodeFilter to the store, ADR-2047 receive rules<br/>binary_protocol.rs:1272 settings_sync.rs:238; never writes back to the server"]
        S2["graphUpdated coalesced into requestInitialData by RefetchGate<br/>binary_protocol.rs:1314 settings_sync.rs:306"]
        S3["domain colour mode and cluster hulls<br/>binary_protocol.rs:1249 binary_protocol.rs:1361 hulls.rs:461"]
        S4["memory_flash credits the memory vertex of the agent drift field<br/>binary_protocol.rs:1468"]
    end

    subgraph pack["Per-frame pack, one pass shared by every layer"]
        P1["pack plans: transforms only while the visual epoch, id list and sizes hold<br/>render_store_pack.rs:125 render_store_pack.rs:142"]
        P2["tiers: NEAR_CAP gem nodes within NEAR_RADIUS_M, the rest 2-triangle impostors<br/>node_lod.gd:26 node_lod.gd:28 render_store_pack.rs:245 lod.rs:266"]
        P3["edges: NEAR_EDGE_CAP cylinders, the rest far ribbons<br/>node_lod.gd:15 node_lod.gd:17 render_store_pack.rs:346"]
        P4["FrameBudget turns drawn instance counts and the measured frame total into caps<br/>graph_scene.gd:1999 frame_budget_pass.gd:50 frame_budget.rs:148"]
    end

    subgraph scene["Scene wiring in _ready"]
        W1["MemoryCloudLayer and MemorySearch under GraphRoot<br/>graph_scene.gd:542 memory_cloud_layer.gd:42 memory_search.gd:21"]
        W2["GraphParity: colour, filter, hulls<br/>graph_scene.gd:592 graph_parity.gd:96"]
        W3["impostor and ribbon MultiMeshes, then BeatPulse<br/>graph_scene.gd:599 graph_scene.gd:601 graph_scene.gd:603"]
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

    N1["INVARIANT: microphone capture is opt-in per session, shown by the MIC badge, analysed in memory only<br/>and never recorded or transmitted docs/XR-client.md:884. The beat clock pings every 2 s beat_pulse.gd:32"]
    N2["OPEN: the memory cloud, its relayed route and the beat-clock relay have been exercised by GUT and desktop<br/>screenshots, not in a headset - stereo agreement, sprite legibility and route brightness are unmeasured<br/>docs/XR-client.md:856"]
    N3["This topic catalogues the seams only. The internals of memory_cloud.rs, memory_route.rs, memory_query.rs,<br/>cloud_frame.rs, beat.rs and semantic.rs, about 9 300 lines at this stamp, are drawn in VC-38<br/>xr-client/rust/src/memory_route.rs:1566 xr-client/rust/src/memory_cloud.rs:883"]
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
(`docs/XR-client.md:790`, design at `docs/XR-client.md:599`).

## VC-36.26 The separated triangle, from the server projection to an agent's rest slot (ADR-2135)

```mermaid
flowchart TB
    TRI["visionclaw-tri-layout is the single definition of the geometry:<br/>knowledge, ontology and memory on the vertices of a ground-plane triangle, agents at the centroid<br/>crates/visionclaw-tri-layout/src/lib.rs:1 crates/visionclaw-tri-layout/src/lib.rs:106 (SEPARATION)"]
    SRV["server display projection, always on since the operator decision of 2026-10-08<br/>src/actors/gpu/display_projection.rs:110 src/actors/gpu/display_projection.rs:190"]
    FCA["ForceComputeActor calls project_display on every broadcast<br/>src/actors/gpu/force_compute_actor.rs:24 src/actors/gpu/force_compute_actor.rs:2198"]
    DRIFT["each action credits the vertex of the body it touched, credit halves over time and the agent falls back<br/>to the centroid, the offset recomputed from the current frame<br/>crates/visionclaw-tri-layout/src/drift.rs:1 crates/visionclaw-tri-layout/src/drift.rs:66 crates/visionclaw-tri-layout/src/drift.rs:133"]
    CLI["xr-client links the same crate<br/>xr-client/rust/Cargo.toml:24"]
    BPC["BinaryProtocolClient.agent_drift_offset, fed by applied 0x23 actions and memory flashes<br/>xr-client/rust/src/binary_protocol.rs:1486"]
    SLOT["_apply_drift_rest_slots moves the slot, _drift_rest_slot adds the ring spread<br/>xr-client/scripts/graph_scene.gd:2490 xr-client/scripts/graph_scene.gd:2503"]
    CH["AgentChoreography.set_rim_slot, still the only pose writer<br/>xr-client/scripts/agent_choreography.gd:101"]

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
2026-10-08 (`src/actors/gpu/force_compute_actor.rs:2040-2045`, ADR-2135), so the
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
        base RefCounted : memory_route.rs:1566
    }
    class MemoryQuery {
        base RefCounted : memory_query.rs:466
    }
    class CloudFrame {
        base RefCounted : cloud_frame.rs:276
    }
    class FrameBudget {
        base RefCounted : frame_budget.rs:252
        allocate : frame_budget.rs:148
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
`cfg(not(test))` (`xr-client/rust/src/lib.rs:42-43`).
