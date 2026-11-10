---
id: VC-13
title: Position broadcast pipeline and WebSocket
area: visionclaw
governing:
  - ../project/docs/PROTOCOL-registry.md
  - ../project/docs/BASELINE-architecture.md
adrs: [ADR-2003, ADR-2018, ADR-2009, ADR-2002, ADR-2139, ADR-2141, ADR-2183]
sources:
  - ../project/src/actors/gpu/force_compute_actor.rs
  - ../project/src/actors/position_stream.rs
  - ../project/src/actors/graph_service_supervisor.rs
  - ../project/src/actors/physics_orchestrator_actor.rs
  - ../project/src/actors/client_coordinator_actor.rs
  - ../project/src/actors/graph_state_actor.rs
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
  - ../project/client/src/store/websocket/binaryFrameDispatcher.ts
  - ../project/client/src/store/websocket/binaryProtocol.ts
  - ../project/client/src/features/graph/managers/graphDataManager.ts
  - ../project/client/src/features/graph/managers/dataManager/wsClient.ts
  - ../project/client/src/features/graph/managers/graphWorkerProxy.ts
  - ../project/client/src/features/graph/workers/graph.worker.ts
  - ../project/src/handlers/socket_flow_handler/actor_messages.rs
  - ../project/src/utils/auth.rs
  - ../project/nginx.production.conf
  - ../project/docs/PROTOCOL-registry.md
  - ../project/docs/BASELINE-architecture.md
  - ../project/crates/visionclaw-domain/src/config/system.rs
  - ../project/crates/visionclaw-domain/src/types/physics_config.rs
  - ../project/src/settings/api/settings_routes.rs
  - ../project/docs/adr/ADR-2141-positions-are-streamed-per-client-until-acknowledged.md
  - ../project/src/actors/client_filter.rs
  - ../project/crates/visionclaw-domain/src/utils/visibility_filter.rs
  - ../project/src/handlers/graph_access.rs
  - ../project/src/settings/models.rs
  - ../project/client/src/features/graph/explore/nodeFilter.ts
  - ../project/docs/adr/ADR-2183-interactive-filters-grey-out-through-a-highlight-mask.md
  - ../project/docs/KNOWN_ISSUES.md
  - ../project/client/src/features/graph/explore/__tests__/nodeFilter.test.ts
  - ../project/tests/node-filter-parity.json
  - ../project/xr-client/rust/src/settings_sync.rs
verified_commit: {visionclaw: 46edc196b58a46c64fffa5c10e3dd72c0a36238e}
---

## For developers

Positions are streamed per client and paced by acknowledgements (ADR-2141). ForceComputeActor publishes every step, and ClientCoordinatorActor sends each client the nodes it has not acknowledged, after the entitlement drop-set and the client's node filter. At 809a859aa the initial graph load and the `filter_update` reply both use the same visible_subgraph gate as the REST reads, the per-client node filter removes nodes only on an explicit server_remove opt-in, and every copy of the filter defaults is pinned to one fixture.

## For the business

Every viewer sees the graph move live, but only nodes it may see. A private node reaches only its owner, including when a viewer changes its node filter, which once re-sent the whole graph without that check.

## VC-13.3 Position stream end to end: GPU step to per-client chunk
```mermaid
sequenceDiagram
    autonumber
    participant PO as PhysicsOrchestratorActor<br/>src/actors/physics_orchestrator_actor.rs:36
    participant FC as ForceComputeActor<br/>src/actors/gpu/force_compute_actor.rs:78
    participant GSS as GraphServiceSupervisor<br/>src/actors/graph_service_supervisor.rs:421
    participant GSA as GraphStateActor<br/>src/actors/graph_state_actor.rs:67
    participant CC as ClientCoordinatorActor<br/>src/actors/client_coordinator_actor.rs:723
    participant WS as SocketFlowServer<br/>src/handlers/socket_flow_handler/types.rs:31

    rect rgb(225,228,245)
    PO->>FC: ComputeForces each physics step, 16ms target in Continuous mode, 0ms in FastSettle,<br/>0ms while any node is held (effective_interval, src/actors/physics_orchestrator_actor.rs:310-315)<br/>src/actors/physics_orchestrator_actor.rs:264-277, :586-589
    Note over FC: one GPU step per ComputeForces, nothing in this actor paces, throttles or caps a broadcast<br/>src/actors/gpu/force_compute_actor.rs:1779,2335-2337
    alt frame is bad (NaN/Inf, coordinate beyond MAX_COORD, speed beyond 1000, or KE beyond 1e9)<br/>src/actors/gpu/force_compute_actor.rs:2194-2224
        FC->>GSS: UpdateNodePositions with the last-known-good layout, display-projected the same way<br/>src/actors/gpu/force_compute_actor.rs:2239-2262
        Note right of FC: MAX_CONSECUTIVE_BAD_FRAMES=5 trips the circuit breaker and halts stepping<br/>src/actors/gpu/force_compute_actor.rs:56,2265-2274
    else healthy frame
        Note over FC: snapshot last_good_positions, then project_display (ADR-2135) rewrites the<br/>broadcast buffer for DISPLAY ONLY, pinned nodes are shown at their pin<br/>src/actors/gpu/force_compute_actor.rs:2297-2322
        FC->>FC: complete_snapshot, one entry per node in node order, a non-finite row is omitted and logged as a physics fault<br/>src/actors/gpu/force_compute_actor.rs:2338-2344, :4819-4844
        FC->>GSS: do_send UpdateNodePositions{positions, correlation_id}, every finite node, every step<br/>src/actors/gpu/force_compute_actor.rs:2345-2350
        Note right of FC: the physics buffer is restored from last_good_positions so the integrator never sees the projection<br/>src/actors/gpu/force_compute_actor.rs:2352-2364
        GSS->>CC: do_send PublishPositions{positions} built from the same set<br/>src/actors/graph_service_supervisor.rs:612-630
        GSS->>GSA: do_send UpdateNodePositions, the same message<br/>src/actors/graph_service_supervisor.rs:631-633
        GSA-->>GSA: mutate graph_data.nodes in place, so REST and the connect-time topology read GPU positions<br/>src/actors/graph_state_actor.rs:807-820
        CC->>CC: update_position_cache replaces the cached position of each node, then pump(None)<br/>src/actors/client_coordinator_actor.rs:1419-1426, :949-979
        Note over CC: nothing is sent until the visibility policy has loaded, the node classification arrives<br/>from GraphStateActor through push_node_type_arrays and UpdateNodeTypeArrays<br/>src/actors/client_coordinator_actor.rs:511-519, :1459-1466,src/actors/graph_service_supervisor.rs:1542-1555
        loop for each registered client, no subscribe gate<br/>src/actors/client_coordinator_actor.rs:520-524
            CC->>CC: pump_one: skip while a chunk is in flight and inside its effective timeout<br/>src/actors/client_coordinator_actor.rs:560
            CC->>CC: entitlement, compute_private_opaque_ids for this pubkey, once per distinct identity per pump<br/>src/actors/client_coordinator_actor.rs:566-577
            CC->>CC: ClientStream.plan picks the nodes this client has not acknowledged, see VC-13.4<br/>src/actors/client_coordinator_actor.rs:602-609
            alt nothing to send, the client is current
                Note right of CC: no frame at all, an idle graph costs a current client nothing<br/>src/actors/client_coordinator_actor.rs:607-609
            else a chunk is due
                CC->>CC: broadcast_sequence takes the next number, serialize_positions builds a V5 frame (0x05, u64 seq LE, V3 52B per node)<br/>src/actors/client_coordinator_actor.rs:614-615, :638-669
                CC->>WS: try_send SendToClientBinary(chunk), one frame per client, each built for that client<br/>src/actors/client_coordinator_actor.rs:617,288-305
                CC->>CC: record_sent only when the mailbox took the frame, otherwise the same nodes are planned again<br/>src/actors/client_coordinator_actor.rs:616-626
                WS->>WS: ctx.binary(payload) over the WebSocket<br/>src/handlers/socket_flow_handler/actor_messages.rs:37-43
            end
        end
        Note over CC,WS: INVARIANT the per-client entitlement drop-set is applied to every chunk, the private-node rule of ADR-2003<br/>is no longer a separate mechanism from the broadcast, and fails closed until the policy has loaded<br/>src/actors/client_coordinator_actor.rs:511-519, :566-578,src/handlers/socket_flow_handler/position_updates.rs:10-12
        Note over CC,WS: a full mailbox (SendError Full) only SKIPS that frame and the client stays registered, SendError Closed<br/>reports it for eviction. A client congested for its whole stall timeout (the heartbeat timeout) is evicted<br/>and its transport shut, other clients are never blocked<br/>src/actors/client_coordinator_actor.rs:295-305, :385-425, :778-797,src/handlers/socket_flow_handler/types.rs:329-331
        CC->>CC: broadcast_count and bytes_sent count the chunks and bytes actually handed to mailboxes<br/>src/actors/client_coordinator_actor.rs:977-979
    end
    end
    Note over PO: DRIFT the orchestrator comment still routes positions through BroadcastPositions, which ADR-2141 removed,<br/>the live path is UpdateNodePositions then PublishPositions<br/>src/actors/physics_orchestrator_actor.rs:577-581
    Note over PO,CC: DRIFT docs/BASELINE-architecture.md draws the position path as ForceComputeActor, PhysicsOrchestratorActor, ClientCoordinatorActor<br/>and says ForceComputeActor drives the periodic full-position broadcast, in the code the orchestrator only sends ComputeForces and takes no part in<br/>delivery, ForceComputeActor publishes every step through GraphServiceSupervisor and the coordinator paces each client<br/>docs/BASELINE-architecture.md:169-172, :245-246, :251-252, :436,src/actors/graph_service_supervisor.rs:612-633,src/actors/client_coordinator_actor.rs:1419-1426
    Note over WS,CC: acknowledgements close the loop, see VC-13.4 and VC-14.10
```

**Drift (KNOWN_ISSUES vs code):** BROADCAST-001 itself now describes the ADR-2141 per-client stream (`docs/KNOWN_ISSUES.md:77-81`), but the resolved-issues row for the delta filter still says Continuous mode "always sends full snapshots at 10fps" (`docs/KNOWN_ISSUES.md:117`). The coordinator streams each client only the nodes it has not acknowledged (`src/actors/client_coordinator_actor.rs:606-609`).


**Tension (parallel):** the disconnect path releases the GPU by holder, because its authors note that releasing by list misses a hold that arrives late (`src/handlers/socket_flow_handler/types.rs:432-434`). The orchestrator is still released by that list (`src/handlers/socket_flow_handler/types.rs:435`, `:442`). Whether a `Dragged` can reach it after its node left `dragged_nodes` is not shown: drag start inserts the node before it spawns the send (`src/handlers/socket_flow_handler/position_updates.rs:229`, `:232-233`). A stale entry would keep the 0 ms step and block the auto-pause (`src/actors/physics_orchestrator_actor.rs:310-311`, `:673`).

## VC-13.1 WS connect, upgrade and authentication
```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant HH as socket_flow_handler<br/>src/handlers/socket_flow_handler/http_handler.rs:49
    participant NS as NostrService get_session<br/>src/handlers/socket_flow_handler/http_handler.rs:222
    participant N98 as nip98::validate_nip98_token<br/>src/utils/nip98.rs:428
    participant WSS as SocketFlowServer actor<br/>src/handlers/socket_flow_handler/types.rs:320
    participant FA as filter_auth::handle_authenticate<br/>src/handlers/socket_flow_handler/filter_auth.rs:54

    rect rgb(225,228,245)
    Note over C,HH: process boundary — HTTP/WS upgrade at /wss
    C->>HH: GET /wss (Upgrade: websocket)
    HH->>HH: WEBSOCKET_RATE_LIMITER.is_allowed(client_ip)<br/>src/handlers/socket_flow_handler/http_handler.rs:56
    alt rate limited
        HH-->>C: 429 create_rate_limit_response<br/>src/handlers/socket_flow_handler/http_handler.rs:58
    else Origin header check
        HH->>HH: validate Origin against CORS_ALLOWED_ORIGINS or same-host<br/>src/handlers/socket_flow_handler/http_handler.rs:67-111
        alt Origin invalid and not same-host
            HH-->>C: 403 Forbidden<br/>src/handlers/socket_flow_handler/http_handler.rs:118-121
        else Origin missing (release build)
            HH-->>C: 400 Origin header required<br/>src/handlers/socket_flow_handler/http_handler.rs:138-140
        else Origin OK
            HH->>HH: read Sec-WebSocket-Protocol, nip98::websocket_auth_header takes the single nostr. subprotocol<br/>src/handlers/socket_flow_handler/http_handler.rs:146-154
            Note right of HH: browsers cannot set Authorization on a WebSocket upgrade, so the signed NIP-98 event rides as an<br/>unpadded URL-safe nostr.TOKEN subprotocol and only the neutral visionclaw protocol is echoed<br/>src/utils/nip98.rs:636-655,src/handlers/socket_flow_handler/http_handler.rs:10
            opt signed subprotocol present
                HH->>NS: verify_nip98_auth(header, signed_upgrade_url, GET) single-use<br/>src/handlers/socket_flow_handler/http_handler.rs:156-161
                alt NostrService absent, or invalid or replayed
                    HH-->>C: 401 Invalid or replayed WebSocket signature<br/>src/handlers/socket_flow_handler/http_handler.rs:158-167
                else verified
                    HH->>HH: signed_user = Some(user)<br/>src/handlers/socket_flow_handler/http_handler.rs:162
                end
            end
            Note right of HH: RESOLVED (estate register ER-25) the upgrade-time URL is now signed_upgrade_url, which delegates to nip98_request_url,<br/>the same single reconstruction the post-connect authenticate frame uses, a test pins them equal when a proxy sends both<br/>Forwarded and X-Forwarded-* headers<br/>src/handlers/socket_flow_handler/http_handler.rs:40-46, :503-519,src/utils/auth.rs:159-189
            opt no signed user, so the Authorization Bearer token is the carrier
                HH->>HH: read Authorization Bearer token<br/>src/handlers/socket_flow_handler/http_handler.rs:178-184
                Note right of HH: RESOLVED ADR-2058 the ?token= query-string fallback is compiled<br/>in only for dev builds (http_handler.rs:187-202), release builds<br/>log and ignore it (http_handler.rs:204-214) and use the header alone
                alt token present
                    HH->>NS: nostr_service.get_session(token)<br/>src/handlers/socket_flow_handler/http_handler.rs:222
                    alt session valid
                        NS-->>HH: Some(session)
                        HH->>HH: signed_user = session, so the header-token client is attributed like the signed one<br/>src/handlers/socket_flow_handler/http_handler.rs:258-262
                        Note right of HH: RESOLVED 0d636fd0e (S-7f, verified in VC-03.2) the validated session used to be checked and then dropped<br/>so a header-session client reached the pubkey and is_power_user assignment below as anonymous.<br/>Authorization is therefore NOT the only carrier, the signed subprotocol is the browser path
                    else session invalid
                        NS-->>HH: None
                        alt release build (or dev without ALLOW_INSECURE_DEFAULTS)
                            HH-->>C: 401 Invalid or expired authentication token<br/>src/handlers/socket_flow_handler/http_handler.rs:250-251
                        else dev build with ALLOW_INSECURE_DEFAULTS=1
                            Note right of HH: dev-only bypass, compile-gated out of release, connection stays anonymous<br/>src/handlers/socket_flow_handler/http_handler.rs:226-231
                        end
                    end
                    opt NostrService absent
                        HH-->>C: 401 Authentication service unavailable (dev builds allow it only with ALLOW_INSECURE_DEFAULTS)<br/>src/handlers/socket_flow_handler/http_handler.rs:265-292
                    end
                else no token at all
                    alt release build
                        HH-->>C: 401 Authentication required<br/>src/handlers/socket_flow_handler/http_handler.rs:321-322
                    else dev build with ALLOW_INSECURE_DEFAULTS=1
                        Note right of HH: unauthenticated WS permitted, dev only<br/>src/handlers/socket_flow_handler/http_handler.rs:308-312
                    end
                end
            end
            HH->>WSS: ws_server = SocketFlowServer::new(...)<br/>src/handlers/socket_flow_handler/http_handler.rs:384
            HH->>WSS: per-connection heartbeat timings from system.websocket, and a closer for the raw TCP socket<br/>src/handlers/socket_flow_handler/http_handler.rs:390-409
            HH->>WSS: signed user, if any, sets pubkey and is_power_user (signed subprotocol and header session alike)<br/>src/handlers/socket_flow_handler/http_handler.rs:412-415
            HH->>WSS: connection_url = nip98_request_url(req) for the post-connect authenticate frame<br/>src/handlers/socket_flow_handler/http_handler.rs:427
            Note over HH: INVARIANT: X-Forwarded-Proto and X-Forwarded-Host win over connection_info in nip98_request_url,<br/>and nginx forwards X-Forwarded-Host $http_host for /wss so the dialled port survives<br/>src/utils/auth.rs:170-188,nginx.production.conf:248
            opt ?token= present a second time (dev builds only, src/handlers/socket_flow_handler/http_handler.rs:371-379)
                HH->>NS: nostr_service.get_session(token_from_qs)<br/>src/handlers/socket_flow_handler/http_handler.rs:432
                NS-->>HH: user with pubkey, is_power_user
                HH->>WSS: ws_server.pubkey = Some(user.pubkey)<br/>src/handlers/socket_flow_handler/http_handler.rs:433-434
                Note right of HH: DEBT redundant since 0d636fd0e, the first extraction already routes a dev query token<br/>through get_session into signed_user, see VC-03.2<br/>src/handlers/socket_flow_handler/http_handler.rs:187-202, :262
            end
            Note over HH: DRIFT the registry's /wss row says NIP-98 Bearer or a ?token= query with the handler at http_handler.rs:139-150 and the Origin<br/>requirement at :130, the query token is dev-build only, the Bearer path is 178-184 and the Origin rejections are 118-121 and 138-140<br/>docs/PROTOCOL-registry.md:263,src/handlers/socket_flow_handler/http_handler.rs:118-121, :138-140, :178-214
            HH->>C: 101 Switching Protocols (WsResponseBuilder, PUBLIC_WS_PROTOCOLS echoed)<br/>src/handlers/socket_flow_handler/http_handler.rs:447-450
            WSS->>WSS: started registers anonymous ClientRecipients with the coordinator, which gives the session a fresh position stream, see VC-13.6<br/>src/handlers/socket_flow_handler/types.rs:333-347,src/actors/client_coordinator_actor.rs:1214-1216
            WSS->>WSS: SetClientId forwards an upgrade-time pubkey to the coordinator as AuthenticateClient<br/>src/handlers/socket_flow_handler/actor_messages.rs:14-30
            Note right of WSS: before this forward the coordinator held a session authenticated at the upgrade as anonymous<br/>so the same-user relay and per-user filters missed it until an authenticate frame arrived<br/>src/handlers/socket_flow_handler/actor_messages.rs:17-21
            WSS->>WSS: ping timer from the heartbeat settings, then send_full_state_sync (JSON topology only, no positions) and connection_established<br/>src/handlers/socket_flow_handler/types.rs:374-419
        end
    end
    end

    rect rgb(225,245,230)
    Note over C,WSS: post-connect NIP-98 "authenticate" message (kind 27235, ADR-2009 primary realm)
    C->>WSS: text message type=authenticate, event=base64 kind-27235 event<br/>src/handlers/socket_flow_handler/message_routing.rs:36-38
    WSS->>FA: handle_authenticate(act, msg, ctx)<br/>src/handlers/socket_flow_handler/filter_auth.rs:54
    alt VISIONCLAW_DEV_MODE full bypass (dev builds only)
        FA-->>C: authenticate_success (DEV_MODE_PUBKEY, power_user)<br/>src/handlers/socket_flow_handler/filter_auth.rs:70-103
    else event field present (NIP-98 path)
        FA->>N98: verify_nip98_auth auth_header=Nostr plus event_b64, ws_url, method=GET<br/>src/handlers/socket_flow_handler/filter_auth.rs:111-118
        N98->>N98: check kind==27235<br/>src/utils/nip98.rs:20,462
        N98->>N98: freshness ±TOKEN_MAX_AGE_SECONDS=60s<br/>src/utils/nip98.rs:169,467-484
        N98->>N98: url and method match, body binding, then Schnorr signature<br/>src/utils/nip98.rs:510,518,526,560-565
        N98->>N98: claim event id in replay cache (TTL=120s, cap 100000, 1000 per signer, fail-closed ReplayCacheFull)<br/>src/utils/nip98.rs:178,198,215,258-262,322-324,573 (ADR-2002)
        alt validation and single-use claim succeed
            N98-->>FA: Ok(user)
            FA-->>C: authenticate_success{pubkey, is_power_user}<br/>src/handlers/socket_flow_handler/filter_auth.rs:148-156
            FA->>WSS: AuthenticateClient to the coordinator, bump_sync_generation, then send_full_state_sync re-push of the topology filtered by the new pubkey, positions follow from the stream<br/>src/handlers/socket_flow_handler/filter_auth.rs:136-167
        else signature/URL/method/freshness/replay fails
            N98-->>FA: Err(e)
            FA-->>C: error NIP-98 WebSocket authentication failed<br/>src/handlers/socket_flow_handler/filter_auth.rs:168-176
        end
    else legacy token+pubkey fields (ADR-2009 secondary realm)
        Note right of FA: legacy { token, pubkey, ephemeral? } path, get_session must return the same pubkey<br/>src/handlers/socket_flow_handler/filter_auth.rs:181-189, :257-260
    end
    end
```

## VC-13.7 Client decode: ordered chunk to SharedArrayBuffer and acknowledgement
```mermaid
sequenceDiagram
    autonumber
    participant WS as WebSocket.onmessage<br/>client/src/store/websocket/binaryFrameDispatcher.ts:101
    participant BFD as BinaryFrameDispatcher<br/>client/src/store/websocket/binaryFrameDispatcher.ts:47
    participant BP as processBinaryData<br/>client/src/store/websocket/binaryProtocol.ts:456
    participant GDM as graphDataManager<br/>client/src/features/graph/managers/graphDataManager.ts:497
    participant WSC as handleBinaryFrame<br/>client/src/features/graph/managers/dataManager/wsClient.ts:19
    participant GWP as graphWorkerProxy<br/>client/src/features/graph/managers/graphWorkerProxy.ts:227
    participant WRK as graph.worker.ts<br/>client/src/features/graph/workers/graph.worker.ts:216

    rect rgb(222,236,250)
    WS->>WS: validateBinaryData, lead byte in {3,5,AGENT_ACTION}<br/>client/src/store/websocket/binaryProtocol.ts:193-200
    WS->>BFD: dispatcher.handle(buffer)<br/>client/src/store/websocket/binaryFrameDispatcher.ts:157
    Note over BFD: a chunk is NOT a snapshot a newer frame could replace, so frames queue in arrival<br/>order and none is coalesced,<br/>one is processed at a time across the Comlink await, a backstop of 256 queued frames<br/>drops the oldest, which the server<br/>rebuilds after its ack timeout<br/>client/src/store/websocket/binaryFrameDispatcher.ts:1-10, :27-28, :55-81
    BFD->>BP: processBinaryData(buffer, get, set)<br/>client/src/store/websocket/binaryFrameDispatcher.ts:60
    BP->>BP: read lead byte, V3 and V5 route to handleLegacyBinaryData (V2 is no longer accepted,<br/>ADR-2057)<br/>client/src/store/websocket/binaryProtocol.ts:466-473
    BP->>BP: parseBinaryFrameData(data), every frame carries absolute positions, there is no delta<br/>frame type (VC-14.6)<br/>client/src/store/websocket/binaryProtocol.ts:352-363
    BP->>BP: nodeAnalyticsStore.ingest(parsedNodes) for V3 and V5 frames<br/>client/src/store/websocket/binaryProtocol.ts:370-373
    BP->>GDM: await graphDataManager.updateNodePositions(data)<br/>client/src/store/websocket/binaryProtocol.ts:393-394
    GDM->>GDM: every 120th frame, sample for node ids the topology does not hold and refetch<br/>topology on a 30s cooldown<br/>client/src/features/graph/managers/graphDataManager.ts:515-540
    GDM->>WSC: handleBinaryFrame(positionData, lastUpdateTime, onUpdateTime)<br/>client/src/features/graph/managers/graphDataManager.ts:500-504
    Note over WSC: no throttle, every chunk goes to the worker, a worker error is rethrown so the chunk is not acknowledged<br/>client/src/features/graph/managers/dataManager/wsClient.ts:39-42, :78
    rect rgb(232,240,252)
        WSC->>GWP: processBinaryFrame(frame: Uint8Array)<br/>client/src/features/graph/managers/dataManager/wsClient.ts:41-42
        GWP->>GWP: queue the frame in arrival order, none replaced, a 256-frame backstop rejects the<br/>oldest<br/>client/src/features/graph/managers/graphWorkerProxy.ts:232-239
        GWP->>GWP: _drain dispatches one at a time and yields a microtask between frames<br/>client/src/features/graph/managers/graphWorkerProxy.ts:243-263
        GWP->>GWP: transfer(frame.buffer), zero-copy neuter<br/>client/src/features/graph/managers/graphWorkerProxy.ts:270
        alt SAB mode (SharedArrayBuffer available, cross-origin isolated)<br/>client/src/features/graph/managers/graphWorkerProxy.ts:191-207
            GWP->>WRK: workerApi.processBinaryFrame(transferable)<br/>client/src/features/graph/managers/graphWorkerProxy.ts:272-277
            WRK->>WRK: parse the chunk and write its node updates into targetPositions, not<br/>currentPositions, a frame before setGraphData is queued<br/>client/src/features/graph/workers/graph.worker.ts:273-311
            WRK->>WRK: with tweening on, ensureTweenLoop interpolates currentPositions toward the targets at<br/>16ms<br/>client/src/features/graph/workers/graph.worker.ts:329-331, :344-372
            WRK->>WRK: syncToSharedBuffer on every tween tick writes positionView (the SharedArrayBuffer), a short buffer is logged once<br/>client/src/features/graph/workers/graph.worker.ts:230-238, :364
            Note right of WRK: with tweening disabled the chunk snaps currentPositions to the targets and syncs at<br/>once<br/>client/src/features/graph/workers/graph.worker.ts:324-328
            Note right of WRK: SAB write complete, the renderer reads positionView directly, no return value needed<br/>client/src/features/graph/workers/graph.worker.ts:216-220
        else Comlink transfer mode (SAB unavailable)
            WRK-->>GWP: transferred ArrayBuffer of stride-3 positions<br/>client/src/features/graph/workers/graph.worker.ts:221-225
            GWP->>GWP: lastTransferredView = new Float32Array(returned)<br/>client/src/features/graph/managers/graphWorkerProxy.ts:281-286
        end
    end
    BP->>BP: updateNodePositions resolved, so acknowledge THIS chunk by the sequence it carried, a<br/>thrown error returns without an ack<br/>client/src/store/websocket/binaryProtocol.ts:393-409
    Note over BP,WSC: RESOLVED the 16ms gate that dropped a chunk and still acknowledged it is removed, ADR-2141 decision 5 now holds,<br/>an applied chunk is acknowledged and a failed one is not<br/>docs/adr/ADR-2141-positions-are-streamed-per-client-until-acknowledged.md:42,client/src/features/graph/managers/dataManager/wsClient.ts:39-42,client/src/store/websocket/binaryProtocol.ts:393-409
    end
    Note over WRK: render side beyond SAB see VC-31/VC-32
```

## VC-13.4 Per-client chunk lifecycle: plan, in flight, acknowledge, time out
```mermaid
sequenceDiagram
    autonumber
    participant TRG as pump triggers<br/>src/actors/client_coordinator_actor.rs:949
    participant CC as ClientManager pump_one<br/>src/actors/client_coordinator_actor.rs:548
    participant CS as ClientStream<br/>src/actors/position_stream.rs:79
    participant WS as SocketFlowServer
    participant CL as Client

    rect rgb(225,228,245)
    Note over TRG: five triggers call pump, new positions (PublishPositions), a client's acknowledgement (that client only),<br/>RegisterClient (the new client only), ConfigureStream, and a 50ms safety tick for a timed-out chunk or a visible set that just changed<br/>src/actors/client_coordinator_actor.rs:1419-1426, :1469-1500, :1214-1216, :1430-1446, :1110-1112
    TRG->>CC: pump_all or pump_client
    CC->>CS: is_waiting(now, ack_timeout), true while the newest chunk is inside its effective timeout<br/>src/actors/position_stream.rs:104-109
    alt a chunk is in flight and not timed out
        CS-->>CC: waiting, build nothing for this client
    else free to send
        CC->>CS: retain_nodes, forget what left the graph or the client's entitlement<br/>src/actors/position_stream.rs:244-246,src/actors/client_coordinator_actor.rs:600-602
        CC->>CS: plan(cache, visible, analytics_key, settings)<br/>src/actors/position_stream.rs:125-182
        Note right of CS: a node is a candidate when never acknowledged (infinite displacement), or its distance from the<br/>ACKNOWLEDGED position exceeds streamMoveThreshold, or only its analytics key differs (displacement 0)<br/>src/actors/position_stream.rs:137-159
        Note right of CS: more candidates than streamChunkNodes, a quarter of the chunk goes to the longest-waiting<br/>(FAIR_SHARE_DIVISOR=4), the rest to the largest movers, then sorted by displacement<br/>src/actors/position_stream.rs:66,162-181
        CS-->>CC: ids, empty when the client already holds the latest positions
        CC->>WS: V5 chunk with the next sequence, try_send<br/>src/actors/client_coordinator_actor.rs:614-617
        CC->>CS: record_sent(seq, now, positions): a still-unacknowledged earlier chunk counts one more consecutive timeout<br/>src/actors/position_stream.rs:185-216
        Note right of CS: the last IN_FLIGHT_HISTORY=8 chunks stay on the list so a slow client still makes progress<br/>src/actors/position_stream.rs:64,195-197
    end
    WS->>CL: binary frame
    CL->>CL: apply the chunk, then send [0x34][u64 seq][u32 nodes][u64 ms], see VC-14.10
    CL->>WS: acknowledgement frame
    WS->>CC: ClientBroadcastAck{sequence_id, client_id}<br/>src/handlers/socket_flow_handler/binary_protocol.rs:67-86
    CC->>CS: acknowledge(seq)<br/>src/actors/client_coordinator_actor.rs:629-631
    alt seq is on the in-flight list
        CS->>CS: commit that chunk's nodes as held, never move a node to older state, drop older in-flight chunks, reset the timeout backoff<br/>src/actors/position_stream.rs:225-240
        CC->>CC: pump(Some(client_id)), the next chunk is built at once, so delivery runs at the client's own round trip<br/>src/actors/client_coordinator_actor.rs:1488-1496
    else unknown or aged-out sequence
        CC->>CC: trace_stale_ack, changes nothing<br/>src/actors/client_coordinator_actor.rs:1497-1507
    end
    Note over CS: no acknowledgement within effective_timeout = ack_timeout times 2 to the consecutive timeouts, capped at 8x,<br/>so the safety tick rebuilds a fresh chunk from current state under a new sequence, a never-acknowledging client costs a bounded trickle<br/>src/actors/position_stream.rs:68,99-102,src/actors/client_coordinator_actor.rs:1110-1112
    Note over CC,CS: INVARIANT absolute positions, no deltas, a chunk is always rebuilt from current state and never resent as old bytes, ADR-2139 and ADR-2141<br/>docs/PROTOCOL-registry.md:326
    Note over CC: stored settings physics.streamChunkNodes, streamMoveThreshold and streamAckTimeoutMs default to 2500, 0.05 and 500 ms,<br/>loaded from the saved physics row at start and re-applied by the physics PUT<br/>crates/visionclaw-domain/src/types/physics_config.rs:25,38,47,src/actors/client_coordinator_actor.rs:1091-1106,src/settings/api/settings_routes.rs:477
    Note over CC,WS: TENSION ADR-2141 decision 6 says clients drop no chunk, the web dispatcher bounds its queue at 256 frames and discards the oldest,<br/>which is safe only because the server rebuilds an unacknowledged chunk after the timeout<br/>docs/adr/ADR-2141-positions-are-streamed-per-client-until-acknowledged.md:43,client/src/store/websocket/binaryFrameDispatcher.ts:27-28, :71-77
    end
```

## VC-13.2 Drag, hold and pin control messages: the GPU is the only physics
```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant WSS as SocketFlowServer<br/>src/handlers/socket_flow_handler/message_routing.rs:18
    participant PU as position_updates handlers<br/>src/handlers/socket_flow_handler/position_updates.rs:193
    participant GSS as GraphServiceSupervisor<br/>src/actors/graph_service_supervisor.rs:2067
    participant PO as PhysicsOrchestratorActor<br/>src/actors/physics_orchestrator_actor.rs:1195
    participant GPU as ForceComputeActor<br/>src/actors/gpu/force_compute_actor.rs:3027
    participant CC as ClientCoordinatorActor<br/>src/actors/client_coordinator_actor.rs:723

    Note over C,WSS: there is NO position request or subscribe message, every registered session is streamed by the coordinator<br/>and acknowledges chunks (ADR-2141), src/handlers/socket_flow_handler/message_routing.rs:15-16
    Note over PU: a drag is a HOLD, not a pin, the node rejoins the simulation when the hand lets go, a lasting anchor is the separate<br/>nodePin and nodeUnpin pair (ADR-2169), src/handlers/socket_flow_handler/position_updates.rs:81-84
    rect rgb(250,228,228)
    Note over C,GPU: process boundary, every drag and pin handler requires an authenticated pubkey (VULN-01)
    C->>WSS: text nodeDragStart {nodeId, position}
    WSS->>PU: handle_node_drag_start(act, msg, ctx)<br/>src/handlers/socket_flow_handler/message_routing.rs:51-53
    alt act.pubkey.is_none()
        PU-->>C: dropped with a warning, unauthenticated<br/>src/handlers/socket_flow_handler/position_updates.rs:199-202
    else authenticated
        PU->>PU: nodeId must fit u32, position must be finite and within 10000 on every axis (VULN-03, VULN-05)<br/>src/handlers/socket_flow_handler/position_updates.rs:92-100, :103-118, :21-34
        alt the connection already holds MAX_DRAGGED_NODES_PER_CLIENT (2, one per hand) other nodes<br/>src/handlers/socket_flow_handler/position_updates.rs:14, :87-89, :214-221
            PU-->>C: refused with a warning, no third hand
        else may_hold
            PU->>GSS: NodeInteractionMessage Dragged per newly held node (send_holds with wake), each carrying the holder<br/>src/handlers/socket_flow_handler/position_updates.rs:232, :131-142
            GSS->>PO: forwarded, track_hold_owner records node to holder, last grab wins (src/actors/physics_orchestrator_actor.rs:1183-1187), and only when<br/>auto_pause_config.resume_on_interaction is set a grab resumes physics, clears the FastSettle latch and re-kicks the pipeline (:1203-1226)
            PU->>GPU: HoldNodes{holder: hold_id, holds:[(node_id,[x,y,z])]}<br/>src/handlers/socket_flow_handler/position_updates.rs:143-145
            GPU->>GPU: holds.hold, mark the pinned mask dirty, clear reheat, impulse and the settle window<br/>src/actors/gpu/force_compute_actor.rs:3030-3043
            PU-->>C: nodeDragStartAck<br/>src/handlers/socket_flow_handler/position_updates.rs:235-242
            PU->>PU: start drag timeout checker<br/>src/handlers/socket_flow_handler/position_updates.rs:244-250
        end
    end
    C->>WSS: text nodeDragUpdate {nodeId, position}
    WSS->>PU: handle_node_drag_update(act, msg, ctx)<br/>src/handlers/socket_flow_handler/message_routing.rs:57-59
    alt act.pubkey.is_none()
        PU-->>C: rejected, unauthenticated<br/>src/handlers/socket_flow_handler/position_updates.rs:269-272
    else update for a node not held, a lost nodeDragStart
        PU->>PU: implicit start under the same two-hand cap, with its own timeout checker<br/>src/handlers/socket_flow_handler/position_updates.rs:284-305
    else within MIN_DRAG_INTERVAL_MS=16 of this node's last update<br/>src/handlers/socket_flow_handler/position_updates.rs:17, :307-320
        PU->>PU: COALESCED, not dropped - the newest position is kept in drag_pending and a trailing timer applies it<br/>src/handlers/socket_flow_handler/position_updates.rs:311-316, :340-354
    else accepted update
        PU->>GPU: HoldNodes with the moved position, the ONLY effect of a drag update<br/>src/handlers/socket_flow_handler/position_updates.rs:326-336
        Note right of GPU: the hold is a display-space position, the held node is shown exactly where the hand holds it<br/>and the neighbours spring around it on the next step, show_pinned_at_pin<br/>src/actors/gpu/force_compute_actor.rs:566, :2322
        GPU->>CC: the moved layout reaches every client through the ordinary entitled chunk stream, see VC-13.3<br/>src/actors/gpu/force_compute_actor.rs:2345-2350
        Note right of PU: RESOLVED the old drag path ran CPU SimulationStep calls and broadcast the whole graph as bare V3 to every session,<br/>bypassing the pubkey drop-set of ADR-2003. It is deleted, drags go through the GPU stream only<br/>src/handlers/socket_flow_handler/position_updates.rs:326-329
    end
    C->>WSS: text nodeDragEnd {nodeId}
    WSS->>PU: handle_node_drag_end(act, msg, ctx)<br/>src/handlers/socket_flow_handler/message_routing.rs:54-56
    PU->>GPU: the newest coalesced position first (send_holds), then release_holds, both to the same actor in order<br/>src/handlers/socket_flow_handler/position_updates.rs:390-403
    PU->>GSS: NodeInteractionMessage Released with the holder, the orchestrator drops the node only if that holder still owns it<br/>src/handlers/socket_flow_handler/position_updates.rs:163-172, src/actors/physics_orchestrator_actor.rs:1188-1190
    PU->>GPU: ReleaseNodes{holder, nodes}, the node REJOINS the simulation and the 2 s settle window opens<br/>src/handlers/socket_flow_handler/position_updates.rs:177, src/actors/gpu/force_compute_actor.rs:3048-3058
    PU-->>C: nodeDragEndAck<br/>src/handlers/socket_flow_handler/position_updates.rs:405-412
    C->>WSS: text nodePin {nodeId, position} or nodeUnpin {nodeId}
    WSS->>PU: handle_node_pin or handle_node_unpin<br/>src/handlers/socket_flow_handler/message_routing.rs:60-65
    alt act.pubkey.is_none()
        PU-->>C: rejected, unauthenticated<br/>src/handlers/socket_flow_handler/position_updates.rs:429-432, :481-484
    else authenticated
        PU->>GPU: PinNodePositions{pins, reheat:false} for nodePin, PinNodePositions{unpin, reheat:true} for nodeUnpin<br/>src/handlers/socket_flow_handler/position_updates.rs:449-453, :499-503
        GPU->>GPU: an explicit pin survives any drag, an unpin opens the settle window and a reheat queues the one-shot impulse<br/>src/actors/gpu/force_compute_actor.rs:2996-3015
        PU-->>C: nodePinAck or nodeUnpinAck<br/>src/handlers/socket_flow_handler/position_updates.rs:458-465, :508-515
    end
    Note over PU,GPU: a stalled drag (no update for drag_timeout_ms) is released and rejoins the simulation, an explicit pin is untouched<br/>src/handlers/socket_flow_handler/position_updates.rs:521-555
    Note over PU,GPU: on disconnect the connection releases BY HOLDER (ReleaseHolder), not by the nodes it believes it holds, because a late hold<br/>update can reach the GPU after the release meant to end it, src/handlers/socket_flow_handler/types.rs:429-442, src/handlers/socket_flow_handler/position_updates.rs:173-175
    Note over GPU: backstop, a hold nobody refreshed for HOLD_EXPIRY 5 s is released by the actor itself<br/>src/actors/gpu/force_compute_actor.rs:4540, :1920-1933
    end
```

## VC-13.9 Same-user session relay: beatClock and memoryRoute
```mermaid
sequenceDiagram
    autonumber
    participant C as Desktop client
    participant WSS as handle_text_message<br/>src/handlers/socket_flow_handler/message_routing.rs:18
    participant SR as session_relay handle_relay<br/>src/handlers/socket_flow_handler/session_relay.rs:331
    participant TH as RelayThrottle<br/>src/handlers/socket_flow_handler/session_relay.rs:263
    participant CC as ClientCoordinatorActor<br/>src/actors/client_coordinator_actor.rs:1554
    participant CM as ClientManager relay_text_to_pubkey<br/>src/actors/client_coordinator_actor.rs:690
    participant SIB as sibling session actor<br/>src/handlers/socket_flow_handler/actor_messages.rs:66

    rect rgb(222,236,250)
    C->>WSS: text beatClock or memoryRoute (ADR-2134)<br/>src/handlers/socket_flow_handler/message_routing.rs:66-70
    WSS->>SR: handle_relay(kind, text.len(), msg, ctx)<br/>src/handlers/socket_flow_handler/session_relay.rs:331-337
    alt act.pubkey is none
        SR-->>C: one error frame per session, every later frame dropped silently<br/>src/handlers/socket_flow_handler/session_relay.rs:338-347
    else authenticated
        SR->>SR: validate size at most 16 KiB, then per-kind fields (bpm 40 to 220, route limits, see VC-38)<br/>src/handlers/socket_flow_handler/session_relay.rs:233-245, :32, :42-43
        alt rejected
            SR-->>C: error frame naming the reject reason<br/>src/handlers/socket_flow_handler/session_relay.rs:352-358
        else accepted, re-serialised from validated fields only
            SR->>TH: offer(now, frame), at most one frame per kind per 250ms<br/>src/handlers/socket_flow_handler/session_relay.rs:361-365, :270-287
            alt Send
                SR->>CC: RelayToUserSessions{pubkey, exclude_client_id, message}<br/>src/handlers/socket_flow_handler/session_relay.rs:311-326
            else Schedule
                SR->>SR: run_later(delay), the flush sends the newest pending frame so the last state lands<br/>src/handlers/socket_flow_handler/session_relay.rs:368-380, :290-297
            else Coalesced
                Note right of TH: replaces the pending frame, a flush is already scheduled<br/>src/handlers/socket_flow_handler/session_relay.rs:280-282
            end
            CC->>CM: relay_text_to_pubkey(pubkey, exclude, message)<br/>src/actors/client_coordinator_actor.rs:1557-1561
            loop each client whose ClientState.pubkey equals the sender and id is not the sender<br/>src/actors/client_coordinator_actor.rs:700-703
                CM->>SIB: try_send(SendToClientText(message))<br/>src/actors/client_coordinator_actor.rs:704-711
                SIB->>SIB: ctx.text(msg.0)<br/>src/handlers/socket_flow_handler/actor_messages.rs:66-71
            end
        end
    end
    Note over CM: INVARIANT the match is on the coordinator's ClientState.pubkey, which only AuthenticateClient sets,<br/>so SetClientId forwarding an upgrade-time pubkey (VC-13.1) is what lets a header or subprotocol<br/>session receive a relayed frame without sending an authenticate message first<br/>src/actors/client_coordinator_actor.rs:1583-1586,src/handlers/socket_flow_handler/actor_messages.rs:22-30
    end
```

## VC-13.6 Connect-time sync: topology over JSON, positions from the stream
```mermaid
sequenceDiagram
    autonumber
    participant WSS as SocketFlowServer started<br/>src/handlers/socket_flow_handler/types.rs:323
    participant SYNC as send_full_state_sync<br/>src/handlers/socket_flow_handler/types.rs:177
    participant GSS as GraphServiceSupervisor<br/>src/actors/graph_service_supervisor.rs:421
    participant CC as ClientCoordinatorActor<br/>src/actors/client_coordinator_actor.rs:723

    rect rgb(225,228,245)
    WSS->>CC: RegisterClient{recipients} with the stall timeout taken from the heartbeat timeout<br/>src/handlers/socket_flow_handler/types.rs:329-347
    CC->>CC: register_client gives the session a fresh ClientStream, nothing acknowledged<br/>src/actors/client_coordinator_actor.rs:345-363, :1186
    CC->>CC: pump(Some(client_id)), every entitled node is a candidate, so the initial position sync is chunks through the ordinary path<br/>src/actors/client_coordinator_actor.rs:1214-1216
    Note right of CC: there is no connect-time position frame, the 3,000-node quality-top-N binary frame the old sync sent is deleted<br/>src/handlers/socket_flow_handler/types.rs:175-177
    WSS->>SYNC: self.send_full_state_sync(ctx), then connection_established{protocol supported 3 and 5}<br/>src/handlers/socket_flow_handler/types.rs:402-419
    SYNC->>SYNC: bump the sync generation, a newer sync supersedes an older one<br/>src/handlers/socket_flow_handler/types.rs:193-197
    SYNC->>GSS: send GetGraphData, then GetSettings<br/>src/handlers/socket_flow_handler/types.rs:199-208
    GSS-->>SYNC: shared GraphData snapshot, GPU-computed positions<br/>src/actors/graph_service_supervisor.rs:1490-1505
    SYNC->>WSS: GenGatedText state_sync{nodes_count, edges_count, settings version}, the counts are this session's view, visible_counts<br/>src/handlers/socket_flow_handler/types.rs:209-221, :215-248
    SYNC->>SYNC: initial_graph_load_projection, first visible_subgraph, the same call the REST graph reads use, drops private nodes not owned<br/>by this pubkey (anonymous sees public-only, fail-closed), drops their metadata rows and scrubs hidden page names from visible nodes<br/>src/handlers/socket_flow_handler/types.rs:271-276, :492-513,crates/visionclaw-domain/src/utils/visibility_filter.rs:171-183,src/handlers/graph_access.rs:23-42
    SYNC->>SYNC: then the top initial_node_limit nodes by quality score, default 3000, and only edges with both ends kept<br/>src/handlers/socket_flow_handler/types.rs:260-267, :520-537, :561-566
    SYNC->>WSS: GenGatedInitialGraphLoad{nodes, edges both inside the cap}, JSON, positions inside are advisory<br/>src/handlers/socket_flow_handler/types.rs:278-285
    WSS->>WSS: ctx.text(json), a superseded generation is dropped at the socket write<br/>src/handlers/socket_flow_handler/actor_messages.rs:121-130
    CC->>WSS: V5 chunks follow the acknowledgements, see VC-13.4<br/>src/actors/client_coordinator_actor.rs:1488-1496
    Note over WSS,CC: a client that asks again gets the JSON topology only, at most once per 30 seconds per connection<br/>src/handlers/socket_flow_handler/position_updates.rs:40-75,src/handlers/socket_flow_handler/message_routing.rs:33-35
    Note over SYNC,CC: after an authenticate frame the sync is re-pushed for the new identity, and AuthenticateClient sets the coordinator's pubkey,<br/>so the next pump recomputes the drop-set and sends the newly entitled nodes as never-acknowledged candidates<br/>src/handlers/socket_flow_handler/filter_auth.rs:136-167,src/actors/client_coordinator_actor.rs:1583-1586, :1110-1112
    Note over SYNC,CC: OPEN the stream carries every entitled node while the JSON initial load stops at initialNodeLimit, so a client sees position<br/>records for ids its topology does not hold, the web client counts them and refetches topology on a 30 second cooldown<br/>src/handlers/socket_flow_handler/types.rs:536-537,client/src/features/graph/managers/graphDataManager.ts:515-540
    end
```

## VC-13.10 Per-client node filter: population scope, opt-in removal and the gated filter reply
```mermaid
flowchart TB
    subgraph IN["filter_update text frame, any connected session"]
        FU["handle_filter_update builds UpdateClientFilter<br/>src/handlers/socket_flow_handler/filter_auth.rs:324<br/>the only guard is a registered client_id, not a pubkey, src/handlers/socket_flow_handler/filter_auth.rs:331"]
        FB["filter_update_from_wire: a missing key takes NodeFilterSettings default,<br/>quality 0.4, include_linked_pages false, server_remove false<br/>src/handlers/socket_flow_handler/filter_auth.rs:12-17, :45-49"]
        FU --> FB
    end
    subgraph CC["ClientCoordinatorActor"]
        UH["Handler UpdateClientFilter copies the fields, server_remove included<br/>src/actors/client_coordinator_actor.rs:1645, :1663-1670"]
        REP["filter_update_reply: visible_subgraph for the client's pubkey first,<br/>then recompute_filtered_nodes over that view, nodes and edges of the view only<br/>src/actors/client_coordinator_actor.rs:58-73, :1688-1704"]
        RC["recompute_filtered_nodes: linked_page stubs out unless include_linked_pages,<br/>predicates remove only when enabled AND server_remove<br/>src/actors/client_filter.rs:26-36"]
        PR["passes_predicates: only enabled predicates combine, an unscored node is neutral<br/>src/actors/client_filter.rs:74-93"]
        PUMP["pump_one: a node is sent when (filter off OR in filtered_node_ids) AND not in the entitlement drop-set<br/>src/actors/client_coordinator_actor.rs:578-582"]
        UH --> REP --> RC --> PR
        RC --> PUMP
    end
    subgraph DEF["four written copies of the shipped defaults, plus the derived fallbacks, all pinned"]
        D1["ClientFilter default: off, 0.4, stubs out, server_remove false<br/>src/actors/client_coordinator_actor.rs:135-151"]
        D2["NodeFilterSettings default, the same values plus min_connections and min_maturity<br/>src/settings/models.rs:170-189"]
        D3["NODE_FILTER_DEFAULTS, the same values plus minConnections and minMaturity<br/>client/src/features/graph/explore/nodeFilter.ts:57-68"]
        D5["XR NodeFilter default, the shared fields only, no server_remove<br/>xr-client/rust/src/settings_sync.rs:44-55"]
        D4["the filter_update fallbacks, taken from D2<br/>src/handlers/socket_flow_handler/filter_auth.rs:17"]
        FIX["tests/node-filter-parity.json default case :72<br/>pinned by src/actors/client_filter.rs:406, src/handlers/socket_flow_handler/filter_auth.rs:592<br/>xr-client/rust/src/settings_sync.rs:653-654<br/>and client/src/features/graph/explore/__tests__/nodeFilter.test.ts:14"]
        D1 --> FIX
        D2 --> FIX
        D3 --> FIX
        D4 --> FIX
        D5 --> FIX
    end
    FB --> UH
    REP --> I1["INVARIANT the filter reply passes the same ADR-2003 visibility gate as initialGraphLoad and REST<br/>src/actors/client_coordinator_actor.rs:54-57, :68-72"]
    PUMP --> T2["TENSION ADR-2183 says the stream always applies the population scope, but with enabled false<br/>filtered_node_ids is skipped, so linked_page stubs stream to a default client, and initialGraphLoad applies no stub scope either<br/>docs/adr/ADR-2183-interactive-filters-grey-out-through-a-highlight-mask.md:64, src/actors/client_filter.rs:3-5, src/handlers/socket_flow_handler/types.rs:492-537"]
```

**Invariant:** every copy of the node-filter defaults, the `filter_update` fallbacks included, is pinned to the `default` case of `tests/node-filter-parity.json` (`tests/node-filter-parity.json:72`), by `shipped_defaults_match_the_fixture_default_case` (`src/actors/client_filter.rs:406`), `an_empty_filter_update_takes_the_shipped_defaults` (`src/handlers/socket_flow_handler/filter_auth.rs:592`), the XR fixture test (`xr-client/rust/src/settings_sync.rs:653-654`) and the client suite (`client/src/features/graph/explore/__tests__/nodeFilter.test.ts:14`).

**Debt (parallel):** the defaults are still written four times and pinned by tests rather than shared (`src/actors/client_coordinator_actor.rs:135-151`, `src/settings/models.rs:170-189`, `client/src/features/graph/explore/nodeFilter.ts:57-68`, `xr-client/rust/src/settings_sync.rs:44-55`), and only two of them carry the degree and maturity floors (`src/settings/models.rs:185-186`, `client/src/features/graph/explore/nodeFilter.ts:66-67`); and `filtered_node_ids` is recomputed over the whole graph on register and authenticate (`src/actors/client_coordinator_actor.rs:1248-1251`, `:1621-1624`) but over the caller's view on `filter_update` (`src/actors/client_coordinator_actor.rs:73`), so the same client's set has two bases.

## VC-13.8 Client session lifecycle
```mermaid
stateDiagram-v2
    [*] --> Connecting
    Connecting --> Rejected: rate limit, origin, signature or token check fails<br/>src/handlers/socket_flow_handler/http_handler.rs:56-59, :113-122, :138-140, :163-167, :250-252, :321-322
    Connecting --> Connected: 101 Switching Protocols<br/>src/handlers/socket_flow_handler/http_handler.rs:447-450
    Rejected --> [*]

    Connected --> Authenticated: release builds always, the upgrade carried a signed subprotocol or a validated header session so pubkey is set before the actor starts<br/>src/handlers/socket_flow_handler/http_handler.rs:412-415
    Connected --> Anonymous: dev or dev-auth builds only, ALLOW_INSECURE_DEFAULTS with no valid credential, public-only view<br/>src/handlers/socket_flow_handler/http_handler.rs:226-231, :308-312
    Anonymous --> Authenticated: authenticate message<br/>NIP-98 kind-27235 or legacy token/pubkey<br/>src/handlers/socket_flow_handler/filter_auth.rs:54 (ADR-2009)

    Authenticated --> Dragging: nodeDragStart requires pubkey, at most two held nodes per connection<br/>src/handlers/socket_flow_handler/position_updates.rs:199-202, :214-221
    Dragging --> Authenticated: nodeDragEnd, a drag timeout or the disconnect releases the hold and the node rejoins the simulation<br/>src/handlers/socket_flow_handler/position_updates.rs:390-403, :537-545, src/handlers/socket_flow_handler/types.rs:442

    Connected --> Streaming: started registers the client with the coordinator, every registered client receives the push, no subscribe needed<br/>src/handlers/socket_flow_handler/types.rs:333-347,src/actors/client_coordinator_actor.rs:520-524
    Streaming --> Congested: ClientCoordinatorActor try_send hits SendError Full<br/>on the client mailbox, only that chunk is skipped<br/>src/actors/client_coordinator_actor.rs:295-302
    Congested --> Streaming: mailbox drains, the next pump_one try_send succeeds and the clock is cleared<br/>src/actors/client_coordinator_actor.rs:385-396
    Congested --> Closed: congested for the whole stall timeout, evicted and the socket shut from outside<br/>src/actors/client_coordinator_actor.rs:399-425

    Streaming --> Closed: WS Close frame, or no inbound frame for heartbeatTimeout (server ping every heartbeatInterval)<br/>src/handlers/socket_flow_handler/mod.rs:34,51-55,src/handlers/socket_flow_handler/types.rs:374-400
    Streaming --> Closed: SendError Closed, client evicted<br/>src/actors/client_coordinator_actor.rs:301-305, :788-796
    Closed --> [*]

    note right of Streaming
        Identity (Anonymous or Authenticated, with Dragging) and delivery (Streaming, Congested) are
        two parallel lanes of one connection, both entered at Connected
        HEARTBEAT defaults are a 10s ping and a 60s timeout, read per connection at upgrade
        and raised to at least 1s and two intervals. Only INBOUND frames refresh liveness
        crates/visionclaw-domain/src/config/system.rs:88-97
        src/handlers/socket_flow_handler/heartbeat.rs:35-39, :79-90
        RESOLVED the old DIVERGENCE (CLIENT_TIMEOUT unused, last_activity never checked) is gone,
        the constants were deleted and the heartbeat module owns the timeout
        src/utils/socket_flow_constants.rs:7-10
        RESOLVED ADR-2054 PushDirective mechanism removed, zero senders
        tree-wide, send_pong and get_pending_directives were never called
        on the live ping/pong path
        src/handlers/socket_flow_handler/actor_messages.rs:174-177
    end note
```
