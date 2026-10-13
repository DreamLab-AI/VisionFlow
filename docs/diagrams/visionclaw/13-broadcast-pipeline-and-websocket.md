---
id: VC-13
title: Position broadcast pipeline and WebSocket
area: visionclaw
governing:
  - ../project/docs/PROTOCOL-registry.md
  - ../project/docs/BASELINE-architecture.md
adrs: [ADR-2003, ADR-2018, ADR-2009, ADR-2002]
sources:
  - ../project/src/gpu/broadcast_optimizer.rs
  - ../project/src/gpu/backpressure.rs
  - ../project/src/actors/gpu/force_compute_actor.rs
  - ../project/src/actors/graph_service_supervisor.rs
  - ../project/src/actors/physics_orchestrator_actor.rs
  - ../project/src/actors/client_coordinator_actor.rs
  - ../project/src/actors/graph_state_actor.rs
  - ../project/src/actors/messages/graph_messages.rs
  - ../project/src/handlers/socket_flow_handler/mod.rs
  - ../project/src/handlers/socket_flow_handler/http_handler.rs
  - ../project/src/handlers/socket_flow_handler/types.rs
  - ../project/src/handlers/socket_flow_handler/message_routing.rs
  - ../project/src/handlers/socket_flow_handler/position_updates.rs
  - ../project/src/handlers/socket_flow_handler/filter_auth.rs
  - ../project/src/handlers/socket_flow_handler/heartbeat.rs
  - ../project/src/handlers/socket_flow_handler/session_relay.rs
  - ../project/src/handlers/socket_flow_handler/binary_protocol.rs
  - ../project/src/utils/nip98.rs
  - ../project/src/utils/socket_flow_constants.rs
  - ../project/src/utils/binary_protocol.rs
  - ../project/src/gpu/mod.rs
  - ../project/client/src/store/websocket/binaryFrameDispatcher.ts
  - ../project/client/src/store/websocket/binaryProtocol.ts
  - ../project/client/src/features/graph/managers/graphDataManager.ts
  - ../project/client/src/features/graph/managers/dataManager/wsClient.ts
  - ../project/client/src/features/graph/managers/graphWorkerProxy.ts
  - ../project/client/src/features/graph/workers/graph.worker.ts
  - ../project/client/src/services/BinaryWebSocketProtocol.ts
  - ../project/src/handlers/socket_flow_handler/actor_messages.rs
  - ../project/src/utils/auth.rs
  - ../project/nginx.production.conf
  - ../project/docs/PROTOCOL-registry.md
  - ../project/crates/visionclaw-domain/src/config/system.rs
  - ../project/crates/visionclaw-domain/src/types/physics_config.rs
verified_commit: 06ce9fe75497ec73985b6d684ba6e94b9be5f4a5
---

## VC-13.3 GPU broadcast frame end to end
```mermaid
sequenceDiagram
    autonumber
    participant FC as ForceComputeActor<br/>src/actors/gpu/force_compute_actor.rs:76
    participant BO as BroadcastOptimizer<br/>src/gpu/broadcast_optimizer.rs:205
    participant BP as NetworkBackpressure<br/>src/gpu/backpressure.rs:213
    participant GSS as GraphServiceSupervisor<br/>src/actors/graph_service_supervisor.rs:421
    participant GSA as GraphStateActor<br/>src/actors/graph_state_actor.rs:66
    participant PO as PhysicsOrchestratorActor<br/>src/actors/physics_orchestrator_actor.rs:37
    participant CC as ClientCoordinatorActor<br/>src/actors/client_coordinator_actor.rs:562
    participant WS as SocketFlowServer<br/>src/handlers/socket_flow_handler/types.rs:25

    rect rgb(225,228,245)
    Note over FC: one broadcast decision per ComputeForces step (handler at<br/>src/actors/gpu/force_compute_actor.rs:1713), the orchestrator drives the cadence<br/>at 16ms in Continuous mode, src/actors/physics_orchestrator_actor.rs:282
    alt frame is bad (NaN/Inf, coordinate beyond MAX_COORD, speed beyond 1000, or KE beyond 1e9)<br/>src/actors/gpu/force_compute_actor.rs:2106
        FC->>BP: try_acquire() then re-broadcast last-known-good positions<br/>src/actors/gpu/force_compute_actor.rs:2124-2146
        Note right of FC: MAX_CONSECUTIVE_BAD_FRAMES=5 trips the circuit breaker<br/>src/actors/gpu/force_compute_actor.rs:54,2150
    else healthy frame
        Note over FC: snapshot last_good_positions, then project_display (ADR-2135) rewrites the<br/>broadcast buffer for DISPLAY ONLY and the physics buffer is restored afterwards<br/>src/actors/gpu/force_compute_actor.rs:2198,2335-2340
        Note over FC: branch order is force_full_broadcast, suppress_intermediate_broadcasts,<br/>else continuous, this diagram is the continuous branch, see VC-13.4 for the other two<br/>src/actors/gpu/force_compute_actor.rs:2218,2249,2254
        FC->>BO: process_frame(position_velocity_buffer, node_id_buffer)<br/>src/actors/gpu/force_compute_actor.rs:2258
        Note right of BO: BROADCAST-001: FULL SNAPSHOT ONLY, no delta filter exists<br/>visible_indices are visibility-culled only (culling off by default) and<br/>the caller ignores them and sends every finite node<br/>src/gpu/broadcast_optimizer.rs:7-11,235-259
        alt rate-limit slot is due (default 8 Hz, DEFAULT_BROADCAST_FPS)<br/>crates/visionclaw-domain/src/types/physics_config.rs:23
            Note right of BO: slots are phase-locked to the previous slot, not to the frame that took it<br/>so the long-run rate equals target_fps<br/>src/gpu/broadcast_optimizer.rs:56-72,46-52
            BO-->>FC: (should_broadcast=true, indices)
            FC->>BP: backpressure.try_acquire()<br/>src/actors/gpu/force_compute_actor.rs:2261
            alt token available (max_tokens=100, cost=1, refill 30 per second)<br/>src/gpu/backpressure.rs:67-70
                BP-->>FC: Some(sequence_id)<br/>src/gpu/backpressure.rs:262
                Note over FC: skip any node whose position is not finite<br/>src/actors/gpu/force_compute_actor.rs:2266
                FC->>GSS: UpdateNodePositions{positions, correlation_id}<br/>src/actors/messages/graph_messages.rs:58
                Note right of FC: last_full_broadcast_iteration = iteration_count<br/>src/actors/gpu/force_compute_actor.rs:2282-2287
                GSS->>GSA: do_send(UpdateNodePositions clone)<br/>src/actors/graph_service_supervisor.rs:2030
                GSA-->>GSA: mutate graph_data.nodes in-place<br/>src/actors/graph_state_actor.rs:865
                Note right of GSA: the one-shot snapshot path (subscribe_position_updates)<br/>returns this GPU-computed layout, see VC-13.6
                GSS->>PO: do_send(UpdateNodePositions)<br/>src/actors/graph_service_supervisor.rs:2041
                PO->>PO: throttle to 60fps (16ms gate), a no-op at 8 Hz input<br/>src/actors/physics_orchestrator_actor.rs:1122-1125
                opt user is dragging this node
                    PO->>PO: override with pinned (x,y,z), vel=0<br/>src/actors/physics_orchestrator_actor.rs:1133-1142
                end
                PO->>CC: do_send(BroadcastPositions{positions})<br/>src/actors/physics_orchestrator_actor.rs:1158
                CC->>CC: broadcast_sequence += 1 before the broadcast so it rides in the frame<br/>src/actors/client_coordinator_actor.rs:1570-1571
                CC->>CC: broadcast_with_filter(positions, node_type_arrays, seq, analytics)<br/>src/actors/client_coordinator_actor.rs:1585-1591
                CC->>CC: serialize_positions builds V5 frame (0x05 tag, u64 seq LE, V3 52B per node)<br/>src/actors/client_coordinator_actor.rs:492-498
                loop for each registered client, no subscribe gate<br/>src/actors/client_coordinator_actor.rs:431
                    alt client.filter.enabled == false (default per-client node filter)
                        CC->>WS: try_send(SendToClientBinary(unfiltered_binary clone))<br/>src/actors/client_coordinator_actor.rs:432-434,454-455
                    else client has an active node-id filter
                        CC->>CC: filter positions by filtered_node_ids, skip the client when none remain<br/>src/actors/client_coordinator_actor.rs:437-444
                        CC->>WS: try_send(SendToClientBinary(filtered_binary))<br/>src/actors/client_coordinator_actor.rs:445-455
                    end
                end
                Note over CC,WS: INVARIANT visibility filter default is fail-closed (ON)<br/>ADR-2003 — this per-client node-id filter is a<br/>SEPARATE mechanism from the pubkey visibility<br/>drop-set filter, which gates the one-shot snapshot (fetch_nodes, see VC-13.6)<br/>and initial state sync, not this ClientCoordinatorActor broadcast<br/>src/handlers/socket_flow_handler/position_updates.rs:15-24
                Note over CC,WS: a full mailbox (SendError Full) only SKIPS that one frame and the client stays<br/>registered, SendError Closed reports it for eviction. A client congested for its whole stall<br/>timeout (the heartbeat timeout) is evicted and its transport shut, other clients are never blocked<br/>src/actors/client_coordinator_actor.rs:231-243,315-356,630-648,src/handlers/socket_flow_handler/types.rs:498-499
                WS->>WS: ctx.binary(payload) over the WebSocket<br/>src/handlers/socket_flow_handler/actor_messages.rs:93-98
            else backpressure exhausted
                BP-->>FC: None
                FC->>BP: backpressure.record_skip()<br/>src/actors/gpu/force_compute_actor.rs:2289
                Note right of BP: congestion tracked, warn every<br/>log_interval_frames=60 skipped frames<br/>src/gpu/backpressure.rs:73,286-316
            end
        else rate-limited (slot not yet due)
            BO-->>FC: (should_broadcast=false, [])
            Note right of FC: see VC-13.4 for the 300-iteration periodic full broadcast
        end
    end
    end
    Note over WS,BP: tokens come back two ways. The bucket refills with time (enable_time_refill) and each<br/>client BroadcastAck is relayed ClientBroadcastAck, PositionBroadcastAck, backpressure.acknowledge()<br/>src/handlers/socket_flow_handler/binary_protocol.rs:50-70<br/>src/actors/client_coordinator_actor.rs:1830-1840<br/>src/actors/gpu/force_compute_actor.rs:4181-4192
    Note over CC: DRIFT the bytes_sent statistic assumes 48 bytes per node plus 9 header bytes while the<br/>frame it just built is 52 bytes per node, so the counter undercounts by about 8 percent<br/>src/actors/client_coordinator_actor.rs:1597-1598,src/utils/binary_protocol.rs:72-82
```

## VC-13.1 WS connect, upgrade and authentication
```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant HH as socket_flow_handler<br/>src/handlers/socket_flow_handler/http_handler.rs:53
    participant NS as NostrService get_session<br/>src/handlers/socket_flow_handler/http_handler.rs:226
    participant N98 as nip98::validate_nip98_token<br/>src/utils/nip98.rs:428
    participant WSS as SocketFlowServer actor<br/>src/handlers/socket_flow_handler/types.rs:489
    participant FA as filter_auth::handle_authenticate<br/>src/handlers/socket_flow_handler/filter_auth.rs:10

    rect rgb(225,228,245)
    Note over C,HH: process boundary — HTTP/WS upgrade at /wss
    C->>HH: GET /wss (Upgrade: websocket)
    HH->>HH: WEBSOCKET_RATE_LIMITER.is_allowed(client_ip)<br/>src/handlers/socket_flow_handler/http_handler.rs:60
    alt rate limited
        HH-->>C: 429 create_rate_limit_response<br/>src/handlers/socket_flow_handler/http_handler.rs:62
    else Origin header check
        HH->>HH: validate Origin against CORS_ALLOWED_ORIGINS or same-host<br/>src/handlers/socket_flow_handler/http_handler.rs:71-115
        alt Origin invalid and not same-host
            HH-->>C: 403 Forbidden<br/>src/handlers/socket_flow_handler/http_handler.rs:122
        else Origin missing (release build)
            HH-->>C: 400 Origin header required<br/>src/handlers/socket_flow_handler/http_handler.rs:142-144
        else Origin OK
            HH->>HH: read Sec-WebSocket-Protocol, nip98::websocket_auth_header takes the single nostr. subprotocol<br/>src/handlers/socket_flow_handler/http_handler.rs:150-158
            Note right of HH: browsers cannot set Authorization on a WebSocket upgrade, so the signed NIP-98 event rides as an<br/>unpadded URL-safe nostr.TOKEN subprotocol and only the neutral visionclaw protocol is echoed<br/>src/utils/nip98.rs:636-655,src/handlers/socket_flow_handler/http_handler.rs:10
            opt signed subprotocol present
                HH->>NS: verify_nip98_auth(header, signed_upgrade_url, GET) single-use<br/>src/handlers/socket_flow_handler/http_handler.rs:160-165
                alt NostrService absent, or invalid or replayed
                    HH-->>C: 401 Invalid or replayed WebSocket signature<br/>src/handlers/socket_flow_handler/http_handler.rs:162-170
                else verified
                    HH->>HH: signed_user = Some(user)<br/>src/handlers/socket_flow_handler/http_handler.rs:166
                end
            end
            Note right of HH: DRIFT the upgrade-time URL is rebuilt by signed_upgrade_url from connection_info<br/>alone while the post-connect authenticate frame uses nip98_request_url, whose docs claim to be the<br/>one reconstruction for every NIP-98 entry point, so the two agree only while connection_info<br/>and the X-Forwarded-Proto and X-Forwarded-Host headers agree<br/>src/handlers/socket_flow_handler/http_handler.rs:42-50,src/utils/auth.rs:159-189
            opt no signed user, so the Authorization Bearer token is the carrier
                HH->>HH: read Authorization Bearer token<br/>src/handlers/socket_flow_handler/http_handler.rs:182-188
                Note right of HH: RESOLVED ADR-2058 the ?token= query-string fallback is compiled<br/>in only for dev builds (http_handler.rs:191-206), release builds<br/>log and ignore it (http_handler.rs:208-218) and use the header alone
                alt token present
                    HH->>NS: nostr_service.get_session(token)<br/>src/handlers/socket_flow_handler/http_handler.rs:226
                    alt session valid
                        NS-->>HH: Some(session)
                        HH->>HH: signed_user = session, so the header-token client is attributed like the signed one<br/>src/handlers/socket_flow_handler/http_handler.rs:257-266
                        Note right of HH: RESOLVED 0d636fd0e (S-7f, verified in VC-03.2) the validated session used to be checked and then dropped<br/>so a header-session client reached the pubkey and is_power_user assignment below as anonymous.<br/>Authorization is therefore NOT the only carrier, the signed subprotocol is the browser path
                    else session invalid
                        NS-->>HH: None
                        alt release build (or dev without ALLOW_INSECURE_DEFAULTS)
                            HH-->>C: 401 Invalid or expired authentication token<br/>src/handlers/socket_flow_handler/http_handler.rs:254-255
                        else dev build with ALLOW_INSECURE_DEFAULTS=1
                            Note right of HH: dev-only bypass, compile-gated out of release, connection stays anonymous<br/>src/handlers/socket_flow_handler/http_handler.rs:228-235
                        end
                    end
                    opt NostrService absent
                        HH-->>C: 401 Authentication service unavailable (dev builds allow it only with ALLOW_INSECURE_DEFAULTS)<br/>src/handlers/socket_flow_handler/http_handler.rs:269-296
                    end
                else no token at all
                    alt release build
                        HH-->>C: 401 Authentication required<br/>src/handlers/socket_flow_handler/http_handler.rs:321-326
                    else dev build with ALLOW_INSECURE_DEFAULTS=1
                        Note right of HH: unauthenticated WS permitted, dev only<br/>src/handlers/socket_flow_handler/http_handler.rs:312-316
                    end
                end
            end
            HH->>WSS: ws_server = SocketFlowServer::new(...)<br/>src/handlers/socket_flow_handler/http_handler.rs:388
            HH->>WSS: per-connection heartbeat timings from system.websocket, and a closer for the raw TCP socket<br/>src/handlers/socket_flow_handler/http_handler.rs:394-413
            HH->>WSS: signed user, if any, sets pubkey and is_power_user (signed subprotocol and header session alike)<br/>src/handlers/socket_flow_handler/http_handler.rs:416-419
            HH->>WSS: connection_url = nip98_request_url(req) for the post-connect authenticate frame<br/>src/handlers/socket_flow_handler/http_handler.rs:431
            Note over HH: INVARIANT: X-Forwarded-Proto and X-Forwarded-Host win over connection_info in nip98_request_url,<br/>and nginx forwards X-Forwarded-Host $http_host for /wss so the dialled port survives<br/>src/utils/auth.rs:170-188,nginx.production.conf:248
            opt ?token= present a second time (dev builds only, src/handlers/socket_flow_handler/http_handler.rs:375-383)
                HH->>NS: nostr_service.get_session(token_from_qs)<br/>src/handlers/socket_flow_handler/http_handler.rs:436
                NS-->>HH: user with pubkey, is_power_user
                HH->>WSS: ws_server.pubkey = Some(user.pubkey)<br/>src/handlers/socket_flow_handler/http_handler.rs:437-438
                Note right of HH: DEBT redundant since 0d636fd0e, the first extraction already routes a dev query token<br/>through get_session into signed_user, see VC-03.2<br/>src/handlers/socket_flow_handler/http_handler.rs:191-206,266
            end
            HH->>C: 101 Switching Protocols (WsResponseBuilder, PUBLIC_WS_PROTOCOLS echoed)<br/>src/handlers/socket_flow_handler/http_handler.rs:451-454
            WSS->>WSS: started registers anonymous ClientRecipients with the coordinator<br/>src/handlers/socket_flow_handler/types.rs:502-516
            WSS->>WSS: SetClientId forwards an upgrade-time pubkey to the coordinator as AuthenticateClient<br/>src/handlers/socket_flow_handler/actor_messages.rs:17-33
            Note right of WSS: before this forward the coordinator held a session authenticated at the upgrade as anonymous<br/>so the same-user relay and per-user filters missed it until an authenticate frame arrived<br/>src/handlers/socket_flow_handler/actor_messages.rs:20-24
            WSS->>WSS: ping timer from the heartbeat settings, then send_full_state_sync and connection_established<br/>src/handlers/socket_flow_handler/types.rs:543-583
        end
    end
    end

    rect rgb(225,245,230)
    Note over C,WSS: post-connect NIP-98 "authenticate" message (kind 27235, ADR-2009 primary realm)
    C->>WSS: text message type=authenticate, event=base64 kind-27235 event<br/>src/handlers/socket_flow_handler/message_routing.rs:64-66
    WSS->>FA: handle_authenticate(act, msg, ctx)<br/>src/handlers/socket_flow_handler/filter_auth.rs:10
    alt VISIONCLAW_DEV_MODE full bypass (dev builds only)
        FA-->>C: authenticate_success (DEV_MODE_PUBKEY, power_user)<br/>src/handlers/socket_flow_handler/filter_auth.rs:26-59
    else event field present (NIP-98 path)
        FA->>N98: verify_nip98_auth auth_header=Nostr plus event_b64, ws_url, method=GET<br/>src/handlers/socket_flow_handler/filter_auth.rs:67-74
        N98->>N98: check kind==27235<br/>src/utils/nip98.rs:20,462
        N98->>N98: freshness ±TOKEN_MAX_AGE_SECONDS=60s<br/>src/utils/nip98.rs:169,467-484
        N98->>N98: url and method match, body binding, then Schnorr signature<br/>src/utils/nip98.rs:510,518,526,560-565
        N98->>N98: claim event id in replay cache (TTL=120s, cap 100000, 1000 per signer, fail-closed ReplayCacheFull)<br/>src/utils/nip98.rs:178,198,215,258-262,322-324,573 (ADR-2002)
        alt validation and single-use claim succeed
            N98-->>FA: Ok(user)
            FA-->>C: authenticate_success{pubkey, is_power_user}<br/>src/handlers/socket_flow_handler/filter_auth.rs:104-112
            FA->>WSS: AuthenticateClient to the coordinator, bump_sync_generation, then send_full_state_sync re-push filtered by new pubkey<br/>src/handlers/socket_flow_handler/filter_auth.rs:92-123
        else signature/URL/method/freshness/replay fails
            N98-->>FA: Err(e)
            FA-->>C: error NIP-98 WebSocket authentication failed<br/>src/handlers/socket_flow_handler/filter_auth.rs:124-132
        end
    else legacy token+pubkey fields (ADR-2009 secondary realm)
        Note right of FA: legacy { token, pubkey, ephemeral? } path, get_session must return the same pubkey<br/>src/handlers/socket_flow_handler/filter_auth.rs:137-145,213-216
    end
    end
```

## VC-13.7 Client decode: binary frame to SharedArrayBuffer
```mermaid
sequenceDiagram
    autonumber
    participant WS as WebSocket.onmessage<br/>client/src/store/websocket/binaryFrameDispatcher.ts:97
    participant BFD as BinaryFrameDispatcher<br/>client/src/store/websocket/binaryFrameDispatcher.ts:43
    participant BP as processBinaryData<br/>client/src/store/websocket/binaryProtocol.ts:462
    participant GDM as graphDataManager<br/>client/src/features/graph/managers/graphDataManager.ts:497
    participant WSC as handleBinaryFrame<br/>client/src/features/graph/managers/dataManager/wsClient.ts:19
    participant GWP as graphWorkerProxy<br/>client/src/features/graph/managers/graphWorkerProxy.ts:205
    participant WRK as graph.worker.ts<br/>client/src/features/graph/workers/graph.worker.ts:213

    rect rgb(222,236,250)
    WS->>WS: validateBinaryData — lead byte in {3,5,AGENT_ACTION}<br/>client/src/store/websocket/binaryProtocol.ts:187-208
    WS->>BFD: dispatcher.handle(buffer)<br/>client/src/store/websocket/binaryFrameDispatcher.ts:51
    alt frame already in flight
        BFD->>BFD: pendingLatest = buffer (newest-wins, drop older)<br/>client/src/store/websocket/binaryFrameDispatcher.ts:52-60
    else no frame in flight
        BFD->>BP: processBinaryData(buffer, get, set)<br/>client/src/store/websocket/binaryFrameDispatcher.ts:64
        BP->>BP: read lead byte, V3/V5 routes to the legacy path (V2 is no longer accepted, ADR-2057)<br/>client/src/store/websocket/binaryProtocol.ts:473-479
        BP->>BP: parseBinaryFrameData(data)<br/>client/src/store/websocket/binaryProtocol.ts:370
        opt full V3/V5 frame
            BP->>BP: nodeAnalyticsStore.ingest(parsedNodes)<br/>client/src/store/websocket/binaryProtocol.ts:381-385
        end
        BP->>GDM: graphDataManager.updateNodePositions(data)<br/>client/src/store/websocket/binaryProtocol.ts:407
        GDM->>WSC: handleBinaryFrame(positionData, lastUpdateTime, onUpdateTime)<br/>client/src/features/graph/managers/graphDataManager.ts:500-504
        alt within 16ms of last update (~60fps throttle)
            WSC-->>GDM: drop frame<br/>client/src/features/graph/managers/dataManager/wsClient.ts:27
        else
            WSC->>GWP: processBinaryFrame(frame: Uint8Array)<br/>client/src/features/graph/managers/dataManager/wsClient.ts:45-46
            alt frame already in flight (single-flight, D2)
                GWP->>GWP: pendingLatest = frame (newest-wins)<br/>client/src/features/graph/managers/graphWorkerProxy.ts:210-217
            else dispatch now
                GWP->>GWP: transfer(frame.buffer) — zero-copy neuter<br/>client/src/features/graph/managers/graphWorkerProxy.ts:248
                alt SAB mode (SharedArrayBuffer available, cross-origin isolated)<br/>client/src/features/graph/managers/graphWorkerProxy.ts:170-186
                    GWP->>WRK: workerApi.processBinaryFrame(transferable)<br/>client/src/features/graph/managers/graphWorkerProxy.ts:253
                    WRK->>WRK: parse the frame and write the node updates into targetPositions, not currentPositions<br/>client/src/features/graph/workers/graph.worker.ts:261-301
                    WRK->>WRK: with tweening on, ensureTweenLoop interpolates currentPositions toward the targets at 16ms<br/>client/src/features/graph/workers/graph.worker.ts:319-321,334-362
                    WRK->>WRK: syncToSharedBuffer on every tween tick writes positionView (the SharedArrayBuffer)<br/>client/src/features/graph/workers/graph.worker.ts:225-226,354
                    Note right of WRK: with tweening disabled the frame snaps currentPositions to the targets and syncs at once<br/>client/src/features/graph/workers/graph.worker.ts:314-318
                    Note right of WRK: SAB write complete — renderer reads<br/>positionView directly, no return value needed<br/>client/src/features/graph/workers/graph.worker.ts:204-217
                else Comlink transfer mode (SAB unavailable)
                    WRK-->>GWP: transferred ArrayBuffer of stride-3 positions<br/>client/src/features/graph/workers/graph.worker.ts:218-222
                    GWP->>GWP: lastTransferredView = new Float32Array(returned)<br/>client/src/features/graph/managers/graphWorkerProxy.ts:262-264
                end
            end
        end
    end
    end
    Note over WRK: render side beyond SAB see VC-31/VC-32
```

## VC-13.4 Full-snapshot escape hatches: ResetPositions, ForceFullBroadcast and the 300-iteration periodic send
```mermaid
sequenceDiagram
    autonumber
    participant PO as PhysicsOrchestratorActor<br/>src/actors/physics_orchestrator_actor.rs:37
    participant FC as ForceComputeActor<br/>src/actors/gpu/force_compute_actor.rs:76
    participant BO as BroadcastOptimizer<br/>src/gpu/broadcast_optimizer.rs:205
    participant BP as NetworkBackpressure<br/>src/gpu/backpressure.rs:213
    participant GSS as GraphServiceSupervisor<br/>src/actors/graph_service_supervisor.rs:421

    rect rgb(225,228,245)
    Note over PO,FC: two things can ask for a full snapshot outside the rate limiter
    PO->>FC: ForceFullBroadcast when FastSettle converges or hits its cap<br/>src/actors/physics_orchestrator_actor.rs:1888-1889,1924-1925
    FC->>BO: mark_broadcast() so the limiter waits a full interval after the snapshot<br/>src/actors/gpu/force_compute_actor.rs:2923
    FC->>FC: read GPU positions and velocities with NO physics step, project_display, skip non-finite rows<br/>src/actors/gpu/force_compute_actor.rs:2980-3030
    FC->>GSS: UpdateNodePositions ALL nodes (IMMEDIATE full broadcast, takes no backpressure token)<br/>src/actors/gpu/force_compute_actor.rs:3025-3028
    Note over FC: ResetPositions re-randomises the layout on a sphere, re-uploads it, reheats and sets<br/>force_full_broadcast=true for the next healthy frame<br/>src/actors/gpu/force_compute_actor.rs:3724,3798-3801
    alt actor.force_full_broadcast is true<br/>src/actors/gpu/force_compute_actor.rs:2218
        FC->>FC: force_full_broadcast=false, suppress_intermediate_broadcasts=false<br/>src/actors/gpu/force_compute_actor.rs:2220-2221
        FC->>BP: backpressure.try_acquire()<br/>src/actors/gpu/force_compute_actor.rs:2223
        FC->>BO: broadcast_optimizer.mark_broadcast()<br/>src/actors/gpu/force_compute_actor.rs:2224
        FC->>GSS: UpdateNodePositions ALL finite nodes (final converged positions)<br/>src/actors/gpu/force_compute_actor.rs:2225-2247
        Note right of FC: FINAL full broadcast logged every time<br/>src/actors/gpu/force_compute_actor.rs:2240-2243
    else actor.suppress_intermediate_broadcasts is true<br/>src/actors/gpu/force_compute_actor.rs:2249
        FC->>BO: process_frame — advances the rate-limit timer only, no send<br/>src/actors/gpu/force_compute_actor.rs:2252
        Note right of FC: DEBT this branch is unreachable, nothing ever sets the flag to true<br/>(declaration, initialiser and four resets only)<br/>src/actors/gpu/force_compute_actor.rs:111,273,2221,2869,2921,3800
    else continuous mode, normal rate-limited path
        FC->>BO: process_frame(position_velocity_buffer, node_id_buffer)<br/>src/actors/gpu/force_compute_actor.rs:2258
        alt should_broadcast true and backpressure token acquired
            FC->>GSS: UpdateNodePositions (rate-limited full snapshot)<br/>src/actors/gpu/force_compute_actor.rs:2261-2285
            FC->>FC: last_full_broadcast_iteration = iteration_count<br/>src/actors/gpu/force_compute_actor.rs:2287
        else should_broadcast false (rate-limited this tick)
            alt iteration_count minus last_full_broadcast_iteration is at least 300<br/>src/actors/gpu/force_compute_actor.rs:2291
                Note right of FC: periodic full broadcast for late-connecting<br/>clients — N=300 iterations, independent of the<br/>8 Hz rate limiter above<br/>src/actors/gpu/force_compute_actor.rs:2291-2325
                FC->>BP: backpressure.try_acquire()<br/>src/actors/gpu/force_compute_actor.rs:2293
                FC->>GSS: UpdateNodePositions ALL nodes (periodic full)<br/>src/actors/gpu/force_compute_actor.rs:2315-2318
                FC->>FC: last_full_broadcast_iteration = iteration_count<br/>src/actors/gpu/force_compute_actor.rs:2321
                FC->>BO: broadcast_optimizer.mark_broadcast()<br/>src/actors/gpu/force_compute_actor.rs:2323
            else below 300 iterations since last full broadcast
                Note right of FC: no broadcast this tick — waits for either<br/>the 8 Hz gate or the 300-iteration escape hatch
            end
        end
    end
    Note over FC,GSS: DIVERGENCE this mechanism sends ALL nodes on every<br/>emission (BROADCAST-001 full-snapshot design) — there is<br/>no delta/diff path to re-synchronise, unlike the historical<br/>delta-filter design implied by older docs<br/>src/gpu/broadcast_optimizer.rs:7-11
    Note over FC,BO: reset_broadcast_timer (next frame broadcasts at once) is for parameter changes and new clients,<br/>mark_broadcast is for a snapshot just sent, the two were conflated before<br/>src/gpu/broadcast_optimizer.rs:330-346
    end
```

## VC-13.2 Subscribe to position updates, and drag/pin control messages
```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant WSS as SocketFlowServer<br/>src/handlers/socket_flow_handler/message_routing.rs:16
    participant PU as position_updates handlers<br/>src/handlers/socket_flow_handler/position_updates.rs:596
    participant GSS as GraphServiceSupervisor<br/>src/actors/graph_service_supervisor.rs:421
    participant GPU as GpuComputeActor
    participant CC as ClientCoordinatorActor<br/>src/actors/client_coordinator_actor.rs:562

    rect rgb(222,236,250)
    C->>WSS: text subscribe_position_updates {binary, nodeTypes}, interval is ignored
    WSS->>PU: handle_subscribe_position_updates(act, msg, ctx)<br/>src/handlers/socket_flow_handler/message_routing.rs:50-52,src/handlers/socket_flow_handler/position_updates.rs:596
    Note over PU: ONE-SHOT, a subscribe returns one full snapshot now for a client that joins while physics is settled.<br/>The continuous stream is the ClientCoordinatorActor push at physics.broadcastFps (VC-13.3)<br/>so there is no per-socket polling loop and no subscribe gate on the push<br/>src/handlers/socket_flow_handler/position_updates.rs:601-607
    alt subscribed within 2s of the last accepted subscribe
        PU-->>C: ignored, no reply (per-session guard)<br/>src/handlers/socket_flow_handler/position_updates.rs:616-621
    else accepted
        PU->>PU: last_position_subscribe = now, position_sub_generation += 1<br/>src/handlers/socket_flow_handler/position_updates.rs:622,626-627
        PU->>PU: parse optional nodeTypes filter into subscribed_node_types<br/>src/handlers/socket_flow_handler/position_updates.rs:638-652
        PU-->>C: subscription_confirmed{mode push, binary}<br/>src/handlers/socket_flow_handler/position_updates.rs:657-666
        PU->>PU: fetch_nodes(app_state, settings_addr)<br/>src/handlers/socket_flow_handler/position_updates.rs:669
        alt act.position_sub_generation != my_generation
            Note right of PU: a newer subscribe superseded this fetch — drop the result<br/>src/handlers/socket_flow_handler/position_updates.rs:674-676
        else current generation
            opt subscribed_node_types non-empty
                PU->>PU: nodes.retain by node type flag bits<br/>src/handlers/socket_flow_handler/position_updates.rs:681-695
            end
            opt PUBKEY_VISIBILITY_FILTER on and visibility non-empty
                PU->>PU: compute_private_opaque_ids + apply_drop_set<br/>fail-closed for anon pubkey<br/>src/handlers/socket_flow_handler/position_updates.rs:704-715 (ADR-2003)
            end
            Note right of PU: DRIFT the comment above that branch says the filter is Default OFF, the truth table<br/>returns ON for an absent variable, so the comment is stale and the code is right<br/>src/handlers/socket_flow_handler/position_updates.rs:697-703,32-40
            PU->>PU: encode_node_data_extended_with_sssp, full-state frame, no delta<br/>src/handlers/socket_flow_handler/position_updates.rs:717-730
            PU->>C: ctx.binary(binary_data)<br/>src/handlers/socket_flow_handler/position_updates.rs:756
        end
    end
    end

    rect rgb(250,228,228)
    Note over C,GPU: process boundary — drag/pin handlers ALL require an authenticated pubkey (VULN-01)
    C->>WSS: text nodeDragStart {nodeId, x, y, z}
    WSS->>PU: handle_node_drag_start(act, msg, ctx)<br/>src/handlers/socket_flow_handler/position_updates.rs:873
    alt act.pubkey.is_none()
        PU-->>C: rejected — unauthenticated, dropped silently<br/>src/handlers/socket_flow_handler/position_updates.rs:879-882
    else authenticated
        alt dragged_nodes.len() at least MAX_DRAGGED_NODES_PER_CLIENT (5)<br/>src/handlers/socket_flow_handler/position_updates.rs:86,930-938
            PU-->>C: rejected — too many simultaneous drags, dropped silently
        else under the cap
            PU->>GSS: NodeInteractionMessage Dragged (resume physics if auto-paused)<br/>src/handlers/socket_flow_handler/position_updates.rs:954-961
            PU->>GSS: UpdateNodePositions pinned_data, velocity=0<br/>src/handlers/socket_flow_handler/position_updates.rs:964-977
            PU->>GPU: PinNodePositions{pins:[(node_id,[x,y,z])], reheat:true}<br/>src/handlers/socket_flow_handler/position_updates.rs:982-988
            PU-->>C: nodeDragStartAck<br/>src/handlers/socket_flow_handler/position_updates.rs:995-1002
            PU->>PU: start drag timeout checker<br/>src/handlers/socket_flow_handler/position_updates.rs:1007-1012
        end
    end
    C->>WSS: text nodeDragUpdate {nodeId, x, y, z}
    WSS->>PU: handle_node_drag_update(act, msg, ctx)<br/>src/handlers/socket_flow_handler/position_updates.rs:1024
    alt act.pubkey.is_none()
        PU-->>C: rejected — unauthenticated<br/>src/handlers/socket_flow_handler/position_updates.rs:1030-1033
    else within MIN_DRAG_INTERVAL_MS=16 rate cap<br/>src/handlers/socket_flow_handler/position_updates.rs:89,1049-1054
        PU-->>C: ignored (rate limited)
    else accepted update
        PU->>GSS: UpdateNodePositions moved node, velocity=0<br/>src/handlers/socket_flow_handler/position_updates.rs:1128-1131
        PU->>GPU: PinNodePositions updated pin, reheat:false<br/>src/handlers/socket_flow_handler/position_updates.rs:1138-1142
        PU->>GSS: up to 10 SimulationStep within DRAG_SETTLE_BUDGET_MS=50ms<br/>src/handlers/socket_flow_handler/position_updates.rs:83,1148-1165
        PU->>CC: BroadcastNodePositions, every graph node encoded with encode_node_data_with_live_analytics<br/>src/handlers/socket_flow_handler/position_updates.rs:1175-1211
        Note right of PU: TENSION the drag and drag-end broadcasts encode the whole graph with default class ids and<br/>reach every client through broadcast_to_all, so neither the pubkey drop-set of the snapshot path<br/>nor the pre-flagged type bits of fetch_nodes apply to these frames, ADR-2003 is not honoured here<br/>src/handlers/socket_flow_handler/position_updates.rs:1204-1211,src/utils/binary_protocol.rs:580-586,src/actors/client_coordinator_actor.rs:1475
    end
    C->>WSS: text nodeDragEnd {nodeId}
    WSS->>PU: handle_node_drag_end(act, msg, ctx)<br/>src/handlers/socket_flow_handler/position_updates.rs:1230
    PU->>GSS: NodeInteractionMessage Released, then one final settle and broadcast, the node STAYS pinned<br/>src/handlers/socket_flow_handler/position_updates.rs:1261-1284,1288-1308,1352-1354
    PU-->>C: nodeDragEndAck<br/>src/handlers/socket_flow_handler/position_updates.rs:1362-1369
    C->>WSS: text nodeUnpin {nodeId}
    WSS->>PU: handle_node_unpin(act, msg, ctx)<br/>src/handlers/socket_flow_handler/position_updates.rs:1381
    alt act.pubkey.is_none()
        PU-->>C: rejected — unauthenticated<br/>src/handlers/socket_flow_handler/position_updates.rs:1387-1390
    else authenticated
        PU->>GPU: PinNodePositions{unpin:[node_id], reheat:true}<br/>src/handlers/socket_flow_handler/position_updates.rs:1417-1424
        PU-->>C: nodeUnpinAck<br/>src/handlers/socket_flow_handler/position_updates.rs:1429-1436
    end
    Note over PU,GPU: a stalled drag releases its pin without a reheat after drag_timeout_ms, the failure path does not<br/>route through nodeDragEnd, which would leave a stale anchor<br/>src/handlers/socket_flow_handler/position_updates.rs:1459-1481
    end
```

## VC-13.9 Same-user session relay: beatClock and memoryRoute
```mermaid
sequenceDiagram
    autonumber
    participant C as Desktop client
    participant WSS as handle_text_message<br/>src/handlers/socket_flow_handler/message_routing.rs:16
    participant SR as session_relay handle_relay<br/>src/handlers/socket_flow_handler/session_relay.rs:331
    participant TH as RelayThrottle<br/>src/handlers/socket_flow_handler/session_relay.rs:263
    participant CC as ClientCoordinatorActor<br/>src/actors/client_coordinator_actor.rs:1915
    participant CM as ClientManager relay_text_to_pubkey<br/>src/actors/client_coordinator_actor.rs:521
    participant SIB as sibling session actor<br/>src/handlers/socket_flow_handler/actor_messages.rs:122

    rect rgb(222,236,250)
    C->>WSS: text beatClock or memoryRoute (ADR-2134)<br/>src/handlers/socket_flow_handler/message_routing.rs:94-98
    WSS->>SR: handle_relay(kind, text.len(), msg, ctx)<br/>src/handlers/socket_flow_handler/session_relay.rs:331-337
    alt act.pubkey is none
        SR-->>C: one error frame per session, every later frame dropped silently<br/>src/handlers/socket_flow_handler/session_relay.rs:338-347
    else authenticated
        SR->>SR: validate size at most 16 KiB, then per-kind fields (bpm 40 to 220, route limits, see VC-38)<br/>src/handlers/socket_flow_handler/session_relay.rs:233-245,32,42-43
        alt rejected
            SR-->>C: error frame naming the reject reason<br/>src/handlers/socket_flow_handler/session_relay.rs:352-358
        else accepted, re-serialised from validated fields only
            SR->>TH: offer(now, frame), at most one frame per kind per 250ms<br/>src/handlers/socket_flow_handler/session_relay.rs:361-365,270-287
            alt Send
                SR->>CC: RelayToUserSessions{pubkey, exclude_client_id, message}<br/>src/handlers/socket_flow_handler/session_relay.rs:311-326
            else Schedule
                SR->>SR: run_later(delay), the flush sends the newest pending frame so the last state lands<br/>src/handlers/socket_flow_handler/session_relay.rs:368-380,290-297
            else Coalesced
                Note right of TH: replaces the pending frame, a flush is already scheduled<br/>src/handlers/socket_flow_handler/session_relay.rs:280-282
            end
            CC->>CM: relay_text_to_pubkey(pubkey, exclude, message)<br/>src/actors/client_coordinator_actor.rs:1918-1922
            loop each client whose ClientState.pubkey equals the sender and id is not the sender<br/>src/actors/client_coordinator_actor.rs:531-534
                CM->>SIB: try_send(SendToClientText(message))<br/>src/actors/client_coordinator_actor.rs:535-542
                SIB->>SIB: ctx.text(msg.0)<br/>src/handlers/socket_flow_handler/actor_messages.rs:122-127
            end
        end
    end
    Note over CM: INVARIANT the match is on the coordinator's ClientState.pubkey, which only AuthenticateClient sets,<br/>so SetClientId forwarding an upgrade-time pubkey (VC-13.1) is what lets a header or subprotocol<br/>session receive a relayed frame without sending an authenticate message first<br/>src/actors/client_coordinator_actor.rs:1944-1947,src/handlers/socket_flow_handler/actor_messages.rs:25-33
    end
```

## VC-13.6 One-shot snapshot — GetGraphData to binary encode
```mermaid
sequenceDiagram
    autonumber
    participant PU as fetch_nodes<br/>src/handlers/socket_flow_handler/position_updates.rs:112
    participant GSS as GraphServiceSupervisor<br/>src/actors/graph_service_supervisor.rs:421
    participant GSA as GraphStateActor<br/>src/actors/graph_state_actor.rs:66
    participant BIN as binary_protocol::encode_node_data_extended_with_sssp<br/>src/utils/binary_protocol.rs:426

    rect rgb(225,228,245)
    Note over PU: invoked from handle_subscribe_position_updates, see VC-13.2 — this is the one-shot<br/>snapshot a subscribe returns, independent of the ClientCoordinatorActor push<br/>path in VC-13.3
    PU->>GSS: send(GetGraphData)<br/>src/handlers/socket_flow_handler/position_updates.rs:120
    GSS->>GSA: forward GetGraphData<br/>src/actors/graph_service_supervisor.rs:1483-1494
    GSA-->>GSS: shared GraphData snapshot, GPU-computed positions<br/>updated by VC-13.3's UpdateNodePositions forward<br/>src/actors/graph_state_actor.rs:942
    GSS-->>PU: shared GraphData snapshot
    alt graph_data.nodes is empty
        PU-->>PU: return None (transient) — the subscribe sends nothing this time<br/>src/handlers/socket_flow_handler/position_updates.rs:135-139,757-762
    else nodes present
        PU->>GSS: send(GetNodeTypeArrays)<br/>src/handlers/socket_flow_handler/position_updates.rs:144
        GSS-->>PU: NodeTypeArrays (agent/knowledge/ontology id sets)
        loop for each graph node<br/>src/handlers/socket_flow_handler/position_updates.rs:185
            PU->>PU: flag compact_id with agent/knowledge/ontology bit<br/>src/handlers/socket_flow_handler/position_updates.rs:189-201
            opt PUBKEY_VISIBILITY_FILTER on (default)<br/>src/handlers/socket_flow_handler/position_updates.rs:177
                PU->>PU: push NodeVisibility{wire_id, is_public, owner_pubkey}<br/>src/handlers/socket_flow_handler/position_updates.rs:202-204
            end
        end
        Note right of PU: INVARIANT visibility filter default is fail-closed (ON)<br/>ADR-2003 — absent env means filtered, anon caller<br/>sees public-only nodes<br/>src/handlers/socket_flow_handler/position_updates.rs:17-40
        PU-->>PU: Some((nodes, detailed_debug, visibility))<br/>src/handlers/socket_flow_handler/position_updates.rs:214
    end
    PU->>BIN: encode_node_data_extended_with_sssp(nodes, default class ids, analytics)<br/>src/handlers/socket_flow_handler/position_updates.rs:724-730
    BIN-->>PU: V3 binary frame (WIRE_V3_ITEM_SIZE=52 bytes per node)<br/>src/utils/binary_protocol.rs:72-82,90-93 (ADR-2018)
    Note right of BIN: DOC-DRIFT docs/PROTOCOL-registry.md now cites the redundant 52-byte<br/>unit-test assertions at binary_protocol.rs lines 1077 and 1174, HEAD has them at<br/>lines 1095 and 1191, the const lock it cites at line 93 is right<br/>docs/PROTOCOL-registry.md:88-92,src/utils/binary_protocol.rs:1095,1191
    end
```

## VC-13.8 Client session lifecycle
```mermaid
stateDiagram-v2
    [*] --> Connecting
    Connecting --> Rejected: rate limit, origin, signature or token check fails<br/>src/handlers/socket_flow_handler/http_handler.rs:60-63,117-126,142-144,168-170,254-256,325-326
    Connecting --> Connected: 101 Switching Protocols<br/>src/handlers/socket_flow_handler/http_handler.rs:451-454
    Rejected --> [*]

    Connected --> Authenticated: release builds always, the upgrade carried a signed subprotocol or a validated header session so pubkey is set before the actor starts<br/>src/handlers/socket_flow_handler/http_handler.rs:416-419
    Connected --> Anonymous: dev or dev-auth builds only, ALLOW_INSECURE_DEFAULTS with no valid credential, public-only view<br/>src/handlers/socket_flow_handler/http_handler.rs:228-235,312-316
    Anonymous --> Authenticated: authenticate message<br/>NIP-98 kind-27235 or legacy token/pubkey<br/>src/handlers/socket_flow_handler/filter_auth.rs:10 (ADR-2009)

    Authenticated --> Dragging: nodeDragStart requires pubkey<br/>src/handlers/socket_flow_handler/position_updates.rs:879-882
    Dragging --> Authenticated: nodeDragEnd leaves the node pinned, nodeUnpin or a drag timeout releases it<br/>src/handlers/socket_flow_handler/position_updates.rs:1278-1284,1417-1424,1471-1481

    Connected --> Streaming: started registers the client with the coordinator, every registered client receives the push, no subscribe needed<br/>src/handlers/socket_flow_handler/types.rs:502-516,src/actors/client_coordinator_actor.rs:431
    Streaming --> Congested: ClientCoordinatorActor try_send hits SendError Full<br/>on the client mailbox, only that frame is skipped<br/>src/actors/client_coordinator_actor.rs:237-239
    Congested --> Streaming: mailbox drains, the next broadcast_with_filter try_send succeeds<br/>src/actors/client_coordinator_actor.rs:315-321
    Congested --> Closed: congested for the whole stall timeout, evicted and the socket shut from outside<br/>src/actors/client_coordinator_actor.rs:329-351

    Streaming --> Closed: WS Close frame, or no inbound frame for heartbeatTimeout (server ping every heartbeatInterval)<br/>src/handlers/socket_flow_handler/mod.rs:35,52-56,src/handlers/socket_flow_handler/types.rs:543-568
    Streaming --> Closed: SendError Closed, client evicted<br/>src/actors/client_coordinator_actor.rs:240-242,641-647
    Closed --> [*]

    note right of Streaming
        Identity (Anonymous or Authenticated, with Dragging) and delivery (Streaming, Congested) are
        two parallel lanes of one connection, both entered at Connected
        HEARTBEAT defaults are a 10s ping and a 60s timeout, read per connection at upgrade
        and raised to at least 1s and two intervals. Only INBOUND frames refresh liveness
        crates/visionclaw-domain/src/config/system.rs:88-97
        src/handlers/socket_flow_handler/heartbeat.rs:35-39,79-90
        RESOLVED the old DIVERGENCE (CLIENT_TIMEOUT unused, last_activity never checked) is gone,
        the constants were deleted and the heartbeat module owns the timeout
        src/utils/socket_flow_constants.rs:7-9
        RESOLVED ADR-2054 PushDirective mechanism removed, zero senders
        tree-wide, send_pong and get_pending_directives were never called
        on the live ping/pong path
        src/handlers/socket_flow_handler/actor_messages.rs:262-265
    end note
```
