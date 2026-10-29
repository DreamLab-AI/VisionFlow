---
id: VC-13
title: Position broadcast pipeline and WebSocket
area: visionclaw
governing:
  - ../project/docs/PROTOCOL-registry.md
  - ../project/docs/BASELINE-architecture.md
adrs: [ADR-2003, ADR-2018, ADR-2009, ADR-2002, ADR-2139, ADR-2141]
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
verified_commit: e74bcddc772d47a45b1971156a2bab540eaa027c
---

## VC-13.3 Position stream end to end: GPU step to per-client chunk
```mermaid
sequenceDiagram
    autonumber
    participant PO as PhysicsOrchestratorActor<br/>src/actors/physics_orchestrator_actor.rs:36
    participant FC as ForceComputeActor<br/>src/actors/gpu/force_compute_actor.rs:77
    participant GSS as GraphServiceSupervisor<br/>src/actors/graph_service_supervisor.rs:421
    participant GSA as GraphStateActor<br/>src/actors/graph_state_actor.rs:67
    participant CC as ClientCoordinatorActor<br/>src/actors/client_coordinator_actor.rs:617
    participant WS as SocketFlowServer<br/>src/handlers/socket_flow_handler/types.rs:22

    rect rgb(225,228,245)
    PO->>FC: ComputeForces each physics step, 16ms target in Continuous mode, 0ms in FastSettle<br/>src/actors/physics_orchestrator_actor.rs:259-272,571-574
    Note over FC: one GPU step per ComputeForces, nothing in this actor paces, throttles or caps a broadcast<br/>src/actors/gpu/force_compute_actor.rs:1746,2262-2264
    alt frame is bad (NaN/Inf, coordinate beyond MAX_COORD, speed beyond 1000, or KE beyond 1e9)<br/>src/actors/gpu/force_compute_actor.rs:2121-2151
        FC->>GSS: UpdateNodePositions with the last-known-good layout, display-projected the same way<br/>src/actors/gpu/force_compute_actor.rs:2166-2189
        Note right of FC: MAX_CONSECUTIVE_BAD_FRAMES=5 trips the circuit breaker and halts stepping<br/>src/actors/gpu/force_compute_actor.rs:55,2192-2201
    else healthy frame
        Note over FC: snapshot last_good_positions, then project_display (ADR-2135) rewrites the<br/>broadcast buffer for DISPLAY ONLY, pinned nodes are shown at their pin<br/>src/actors/gpu/force_compute_actor.rs:2224-2249
        FC->>FC: complete_snapshot, one entry per node in node order, a non-finite row is omitted and logged as a physics fault<br/>src/actors/gpu/force_compute_actor.rs:2265-2271,4091-4116
        FC->>GSS: do_send UpdateNodePositions{positions, correlation_id}, every finite node, every step<br/>src/actors/gpu/force_compute_actor.rs:2272-2277
        Note right of FC: the physics buffer is restored from last_good_positions so the integrator never sees the projection<br/>src/actors/gpu/force_compute_actor.rs:2279-2291
        GSS->>CC: do_send PublishPositions{positions} built from the same set<br/>src/actors/graph_service_supervisor.rs:607-625
        GSS->>GSA: do_send UpdateNodePositions, the same message<br/>src/actors/graph_service_supervisor.rs:626-628
        GSA-->>GSA: mutate graph_data.nodes in place, so REST and the connect-time topology read GPU positions<br/>src/actors/graph_state_actor.rs:905-918
        CC->>CC: update_position_cache replaces the cached position of each node, then pump(None)<br/>src/actors/client_coordinator_actor.rs:1315-1322,843-873
        Note over CC: nothing is sent until the visibility policy has loaded, the node classification arrives<br/>from GraphStateActor through push_node_type_arrays and UpdateNodeTypeArrays<br/>src/actors/client_coordinator_actor.rs:405-413,1355-1362,src/actors/graph_service_supervisor.rs:1537-1550
        loop for each registered client, no subscribe gate<br/>src/actors/client_coordinator_actor.rs:414-418
            CC->>CC: pump_one: skip while a chunk is in flight and inside its effective timeout<br/>src/actors/client_coordinator_actor.rs:454
            CC->>CC: entitlement, compute_private_opaque_ids for this pubkey, once per distinct identity per pump<br/>src/actors/client_coordinator_actor.rs:460-471
            CC->>CC: ClientStream.plan picks the nodes this client has not acknowledged, see VC-13.4<br/>src/actors/client_coordinator_actor.rs:496-503
            alt nothing to send, the client is current
                Note right of CC: no frame at all, an idle graph costs a current client nothing<br/>src/actors/client_coordinator_actor.rs:501-503
            else a chunk is due
                CC->>CC: broadcast_sequence takes the next number, serialize_positions builds a V5 frame (0x05, u64 seq LE, V3 52B per node)<br/>src/actors/client_coordinator_actor.rs:508-509,532-563
                CC->>WS: try_send SendToClientBinary(chunk), one frame per client, each built for that client<br/>src/actors/client_coordinator_actor.rs:511,225-243
                CC->>CC: record_sent only when the mailbox took the frame, otherwise the same nodes are planned again<br/>src/actors/client_coordinator_actor.rs:510-520
                WS->>WS: ctx.binary(payload) over the WebSocket<br/>src/handlers/socket_flow_handler/actor_messages.rs:37-43
            end
        end
        Note over CC,WS: INVARIANT the per-client entitlement drop-set is applied to every chunk, the private-node rule of ADR-2003<br/>is no longer a separate mechanism from the broadcast, and fails closed until the policy has loaded<br/>src/actors/client_coordinator_actor.rs:405-413,460-472,src/handlers/socket_flow_handler/position_updates.rs:13-15
        Note over CC,WS: a full mailbox (SendError Full) only SKIPS that frame and the client stays registered, SendError Closed<br/>reports it for eviction. A client congested for its whole stall timeout (the heartbeat timeout) is evicted<br/>and its transport shut, other clients are never blocked<br/>src/actors/client_coordinator_actor.rs:232-242,322-362,672-691,src/handlers/socket_flow_handler/types.rs:389-391
        CC->>CC: broadcast_count and bytes_sent count the chunks and bytes actually handed to mailboxes<br/>src/actors/client_coordinator_actor.rs:871-873
    end
    end
    Note over PO: DRIFT the orchestrator comment still routes positions through BroadcastPositions, which ADR-2141 removed,<br/>the live path is UpdateNodePositions then PublishPositions<br/>src/actors/physics_orchestrator_actor.rs:562-566
    Note over PO,CC: DRIFT docs/BASELINE-architecture.md draws the position path as ForceComputeActor, PhysicsOrchestratorActor, ClientCoordinatorActor<br/>and says ForceComputeActor drives the periodic full-position broadcast, in the code the orchestrator only sends ComputeForces and takes no part in<br/>delivery, ForceComputeActor publishes every step through GraphServiceSupervisor and the coordinator paces each client<br/>docs/BASELINE-architecture.md:150-153,226-227,232-233,325-326,src/actors/graph_service_supervisor.rs:607-628,src/actors/client_coordinator_actor.rs:1315-1322
    Note over WS,CC: acknowledgements close the loop, see VC-13.4 and VC-14.10
```

## VC-13.1 WS connect, upgrade and authentication
```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant HH as socket_flow_handler<br/>src/handlers/socket_flow_handler/http_handler.rs:49
    participant NS as NostrService get_session<br/>src/handlers/socket_flow_handler/http_handler.rs:222
    participant N98 as nip98::validate_nip98_token<br/>src/utils/nip98.rs:428
    participant WSS as SocketFlowServer actor<br/>src/handlers/socket_flow_handler/types.rs:380
    participant FA as filter_auth::handle_authenticate<br/>src/handlers/socket_flow_handler/filter_auth.rs:10

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
            Note right of HH: RESOLVED (estate register ER-25) the upgrade-time URL is now signed_upgrade_url, which delegates to nip98_request_url,<br/>the same single reconstruction the post-connect authenticate frame uses, a test pins them equal when a proxy sends both<br/>Forwarded and X-Forwarded-* headers<br/>src/handlers/socket_flow_handler/http_handler.rs:40-46,503-519,src/utils/auth.rs:159-189
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
                Note right of HH: DEBT redundant since 0d636fd0e, the first extraction already routes a dev query token<br/>through get_session into signed_user, see VC-03.2<br/>src/handlers/socket_flow_handler/http_handler.rs:187-202,262
            end
            Note over HH: DRIFT the registry's /wss row says NIP-98 Bearer or a ?token= query with the handler at http_handler.rs:139-150 and the Origin<br/>requirement at :130, the query token is dev-build only, the Bearer path is 178-184 and the Origin rejections are 118-121 and 138-140<br/>docs/PROTOCOL-registry.md:231,src/handlers/socket_flow_handler/http_handler.rs:118-121,138-140,178-214
            HH->>C: 101 Switching Protocols (WsResponseBuilder, PUBLIC_WS_PROTOCOLS echoed)<br/>src/handlers/socket_flow_handler/http_handler.rs:447-450
            WSS->>WSS: started registers anonymous ClientRecipients with the coordinator, which gives the session a fresh position stream, see VC-13.6<br/>src/handlers/socket_flow_handler/types.rs:393-407,src/actors/client_coordinator_actor.rs:1108-1110
            WSS->>WSS: SetClientId forwards an upgrade-time pubkey to the coordinator as AuthenticateClient<br/>src/handlers/socket_flow_handler/actor_messages.rs:14-30
            Note right of WSS: before this forward the coordinator held a session authenticated at the upgrade as anonymous<br/>so the same-user relay and per-user filters missed it until an authenticate frame arrived<br/>src/handlers/socket_flow_handler/actor_messages.rs:17-21
            WSS->>WSS: ping timer from the heartbeat settings, then send_full_state_sync (JSON topology only, no positions) and connection_established<br/>src/handlers/socket_flow_handler/types.rs:434-479
        end
    end
    end

    rect rgb(225,245,230)
    Note over C,WSS: post-connect NIP-98 "authenticate" message (kind 27235, ADR-2009 primary realm)
    C->>WSS: text message type=authenticate, event=base64 kind-27235 event<br/>src/handlers/socket_flow_handler/message_routing.rs:34-36
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
            FA->>WSS: AuthenticateClient to the coordinator, bump_sync_generation, then send_full_state_sync re-push of the topology filtered by the new pubkey, positions follow from the stream<br/>src/handlers/socket_flow_handler/filter_auth.rs:92-123
        else signature/URL/method/freshness/replay fails
            N98-->>FA: Err(e)
            FA-->>C: error NIP-98 WebSocket authentication failed<br/>src/handlers/socket_flow_handler/filter_auth.rs:124-132
        end
    else legacy token+pubkey fields (ADR-2009 secondary realm)
        Note right of FA: legacy { token, pubkey, ephemeral? } path, get_session must return the same pubkey<br/>src/handlers/socket_flow_handler/filter_auth.rs:137-145,213-216
    end
    end
```

## VC-13.7 Client decode: ordered chunk to SharedArrayBuffer and acknowledgement
```mermaid
sequenceDiagram
    autonumber
    participant WS as WebSocket.onmessage<br/>client/src/store/websocket/binaryFrameDispatcher.ts:101
    participant BFD as BinaryFrameDispatcher<br/>client/src/store/websocket/binaryFrameDispatcher.ts:47
    participant BP as processBinaryData<br/>client/src/store/websocket/binaryProtocol.ts:450
    participant GDM as graphDataManager<br/>client/src/features/graph/managers/graphDataManager.ts:497
    participant WSC as handleBinaryFrame<br/>client/src/features/graph/managers/dataManager/wsClient.ts:19
    participant GWP as graphWorkerProxy<br/>client/src/features/graph/managers/graphWorkerProxy.ts:212
    participant WRK as graph.worker.ts<br/>client/src/features/graph/workers/graph.worker.ts:214

    rect rgb(222,236,250)
    WS->>WS: validateBinaryData, lead byte in {3,5,AGENT_ACTION}<br/>client/src/store/websocket/binaryProtocol.ts:192-199
    WS->>BFD: dispatcher.handle(buffer)<br/>client/src/store/websocket/binaryFrameDispatcher.ts:157
    Note over BFD: a chunk is NOT a snapshot a newer frame could replace, so frames queue in arrival<br/>order and none is coalesced,<br/>one is processed at a time across the Comlink await, a backstop of 256 queued frames<br/>drops the oldest, which the server<br/>rebuilds after its ack timeout<br/>client/src/store/websocket/binaryFrameDispatcher.ts:1-10,27-28,55-81
    BFD->>BP: processBinaryData(buffer, get, set)<br/>client/src/store/websocket/binaryFrameDispatcher.ts:60
    BP->>BP: read lead byte, V3 and V5 route to handleLegacyBinaryData (V2 is no longer accepted,<br/>ADR-2057)<br/>client/src/store/websocket/binaryProtocol.ts:460-467
    BP->>BP: parseBinaryFrameData(data), every frame carries absolute positions, there is no delta<br/>frame type (VC-14.6)<br/>client/src/store/websocket/binaryProtocol.ts:351-362
    BP->>BP: nodeAnalyticsStore.ingest(parsedNodes) for V3 and V5 frames<br/>client/src/store/websocket/binaryProtocol.ts:369-372
    BP->>GDM: await graphDataManager.updateNodePositions(data)<br/>client/src/store/websocket/binaryProtocol.ts:392-393
    GDM->>GDM: every 120th frame, sample for node ids the topology does not hold and refetch<br/>topology on a 30s cooldown<br/>client/src/features/graph/managers/graphDataManager.ts:515-540
    GDM->>WSC: handleBinaryFrame(positionData, lastUpdateTime, onUpdateTime)<br/>client/src/features/graph/managers/graphDataManager.ts:500-504
    alt within 16ms of the last accepted frame
        WSC-->>GDM: returns normally, the frame is dropped and never reaches the worker<br/>client/src/features/graph/managers/dataManager/wsClient.ts:26-28
    else
        WSC->>GWP: processBinaryFrame(frame: Uint8Array)<br/>client/src/features/graph/managers/dataManager/wsClient.ts:42-43
        GWP->>GWP: queue the frame in arrival order, none replaced, a 256-frame backstop rejects the<br/>oldest<br/>client/src/features/graph/managers/graphWorkerProxy.ts:217-224
        GWP->>GWP: _drain dispatches one at a time and yields a microtask between frames<br/>client/src/features/graph/managers/graphWorkerProxy.ts:228-248
        GWP->>GWP: transfer(frame.buffer), zero-copy neuter<br/>client/src/features/graph/managers/graphWorkerProxy.ts:255
        alt SAB mode (SharedArrayBuffer available, cross-origin isolated)<br/>client/src/features/graph/managers/graphWorkerProxy.ts:176-192
            GWP->>WRK: workerApi.processBinaryFrame(transferable)<br/>client/src/features/graph/managers/graphWorkerProxy.ts:257-262
            WRK->>WRK: parse the chunk and write its node updates into targetPositions, not<br/>currentPositions, a frame before setGraphData is queued<br/>client/src/features/graph/workers/graph.worker.ts:263-301
            WRK->>WRK: with tweening on, ensureTweenLoop interpolates currentPositions toward the targets at<br/>16ms<br/>client/src/features/graph/workers/graph.worker.ts:319-321,334-362
            WRK->>WRK: syncToSharedBuffer on every tween tick writes positionView (the SharedArrayBuffer)<br/>client/src/features/graph/workers/graph.worker.ts:226-228,354
            Note right of WRK: with tweening disabled the chunk snaps currentPositions to the targets and syncs at<br/>once<br/>client/src/features/graph/workers/graph.worker.ts:314-318
            Note right of WRK: SAB write complete, the renderer reads positionView directly, no return value needed<br/>client/src/features/graph/workers/graph.worker.ts:214-218
        else Comlink transfer mode (SAB unavailable)
            WRK-->>GWP: transferred ArrayBuffer of stride-3 positions<br/>client/src/features/graph/workers/graph.worker.ts:219-223
            GWP->>GWP: lastTransferredView = new Float32Array(returned)<br/>client/src/features/graph/managers/graphWorkerProxy.ts:266-271
        end
    end
    BP->>BP: updateNodePositions resolved, so acknowledge THIS chunk by the sequence it carried, a<br/>thrown error returns without an ack<br/>client/src/store/websocket/binaryProtocol.ts:392-408
    Note over BP,WSC: TENSION ADR-2141 decision 5 acknowledges a chunk only after it is applied, but the<br/>16ms gate above returns normally,<br/>so a chunk that arrived inside the window is dropped, never reaches the worker, and<br/>is still acknowledged, the server then<br/>records those nodes as held by this client and does not resend them until they move<br/>past the threshold<br/>docs/adr/ADR-2141-positions-are-streamed-per-client-until-acknowledged.md:32,client/src/features/graph/managers/dataManager/wsClient.ts:26-28,client/src/store/websocket/binaryProtocol.ts:392-408
    end
    Note over WRK: render side beyond SAB see VC-31/VC-32
```

## VC-13.4 Per-client chunk lifecycle: plan, in flight, acknowledge, time out
```mermaid
sequenceDiagram
    autonumber
    participant TRG as pump triggers<br/>src/actors/client_coordinator_actor.rs:843
    participant CC as ClientManager pump_one<br/>src/actors/client_coordinator_actor.rs:442
    participant CS as ClientStream<br/>src/actors/position_stream.rs:79
    participant WS as SocketFlowServer
    participant CL as Client

    rect rgb(225,228,245)
    Note over TRG: five triggers call pump, new positions (PublishPositions), a client's acknowledgement (that client only),<br/>RegisterClient (the new client only), ConfigureStream, and a 50ms safety tick for a timed-out chunk or a visible set that just changed<br/>src/actors/client_coordinator_actor.rs:1315-1322,1365-1396,1108-1110,1326-1342,1004-1006
    TRG->>CC: pump_all or pump_client
    CC->>CS: is_waiting(now, ack_timeout), true while the newest chunk is inside its effective timeout<br/>src/actors/position_stream.rs:104-109
    alt a chunk is in flight and not timed out
        CS-->>CC: waiting, build nothing for this client
    else free to send
        CC->>CS: retain_nodes, forget what left the graph or the client's entitlement<br/>src/actors/position_stream.rs:244-246,src/actors/client_coordinator_actor.rs:494-496
        CC->>CS: plan(cache, visible, analytics_key, settings)<br/>src/actors/position_stream.rs:125-182
        Note right of CS: a node is a candidate when never acknowledged (infinite displacement), or its distance from the<br/>ACKNOWLEDGED position exceeds streamMoveThreshold, or only its analytics key differs (displacement 0)<br/>src/actors/position_stream.rs:137-159
        Note right of CS: more candidates than streamChunkNodes, a quarter of the chunk goes to the longest-waiting<br/>(FAIR_SHARE_DIVISOR=4), the rest to the largest movers, then sorted by displacement<br/>src/actors/position_stream.rs:66,162-181
        CS-->>CC: ids, empty when the client already holds the latest positions
        CC->>WS: V5 chunk with the next sequence, try_send<br/>src/actors/client_coordinator_actor.rs:508-511
        CC->>CS: record_sent(seq, now, positions): a still-unacknowledged earlier chunk counts one more consecutive timeout<br/>src/actors/position_stream.rs:185-216
        Note right of CS: the last IN_FLIGHT_HISTORY=8 chunks stay on the list so a slow client still makes progress<br/>src/actors/position_stream.rs:64,195-197
    end
    WS->>CL: binary frame
    CL->>CL: apply the chunk, then send [0x34][u64 seq][u32 nodes][u64 ms], see VC-14.10
    CL->>WS: acknowledgement frame
    WS->>CC: ClientBroadcastAck{sequence_id, client_id}<br/>src/handlers/socket_flow_handler/binary_protocol.rs:67-86
    CC->>CS: acknowledge(seq)<br/>src/actors/client_coordinator_actor.rs:523-525
    alt seq is on the in-flight list
        CS->>CS: commit that chunk's nodes as held, never move a node to older state, drop older in-flight chunks, reset the timeout backoff<br/>src/actors/position_stream.rs:225-240
        CC->>CC: pump(Some(client_id)), the next chunk is built at once, so delivery runs at the client's own round trip<br/>src/actors/client_coordinator_actor.rs:1384-1392
    else unknown or aged-out sequence
        CC->>CC: trace_stale_ack, changes nothing<br/>src/actors/client_coordinator_actor.rs:1393-1403
    end
    Note over CS: no acknowledgement within effective_timeout = ack_timeout times 2 to the consecutive timeouts, capped at 8x,<br/>so the safety tick rebuilds a fresh chunk from current state under a new sequence, a never-acknowledging client costs a bounded trickle<br/>src/actors/position_stream.rs:68,99-102,src/actors/client_coordinator_actor.rs:1004-1006
    Note over CC,CS: INVARIANT absolute positions, no deltas, a chunk is always rebuilt from current state and never resent as old bytes, ADR-2139 and ADR-2141<br/>docs/PROTOCOL-registry.md:284
    Note over CC: stored settings physics.streamChunkNodes, streamMoveThreshold and streamAckTimeoutMs default to 2500, 0.05 and 500 ms,<br/>loaded from the saved physics row at start and re-applied by the physics PUT<br/>crates/visionclaw-domain/src/types/physics_config.rs:24,37,46,src/actors/client_coordinator_actor.rs:985-1000,src/settings/api/settings_routes.rs:472
    Note over CC,WS: TENSION ADR-2141 decision 6 says clients drop no chunk, the web dispatcher bounds its queue at 256 frames and discards the oldest,<br/>which is safe only because the server rebuilds an unacknowledged chunk after the timeout<br/>docs/adr/ADR-2141-positions-are-streamed-per-client-until-acknowledged.md:33,client/src/store/websocket/binaryFrameDispatcher.ts:27-28,71-77
    end
```

## VC-13.2 Drag and pin control messages: the GPU is the only physics
```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant WSS as SocketFlowServer<br/>src/handlers/socket_flow_handler/message_routing.rs:16
    participant PU as position_updates handlers<br/>src/handlers/socket_flow_handler/position_updates.rs:188
    participant GSS as GraphServiceSupervisor<br/>src/actors/graph_service_supervisor.rs:421
    participant PO as PhysicsOrchestratorActor<br/>src/actors/physics_orchestrator_actor.rs:36
    participant GPU as ForceComputeActor<br/>src/actors/gpu/force_compute_actor.rs:3009
    participant CC as ClientCoordinatorActor<br/>src/actors/client_coordinator_actor.rs:617

    Note over C,WSS: there is NO position request or subscribe message, every registered session is streamed by the coordinator<br/>and acknowledges chunks, the subscribe_position_updates handler and its fetch_nodes snapshot were removed (ADR-2141)<br/>src/handlers/socket_flow_handler/message_routing.rs:13-14
    rect rgb(250,228,228)
    Note over C,GPU: process boundary, drag and pin handlers ALL require an authenticated pubkey (VULN-01)
    C->>WSS: text nodeDragStart {nodeId, position}
    WSS->>PU: handle_node_drag_start(act, msg, ctx)<br/>src/handlers/socket_flow_handler/message_routing.rs:52-54
    alt act.pubkey.is_none()
        PU-->>C: dropped silently, unauthenticated<br/>src/handlers/socket_flow_handler/position_updates.rs:194-197
    else authenticated
        PU->>PU: nodeId must fit u32, position must be finite and within 10000 on every axis (VULN-03, VULN-05)<br/>src/handlers/socket_flow_handler/position_updates.rs:26-37,233-243
        alt dragged_nodes.len() at least MAX_DRAGGED_NODES_PER_CLIENT (5)<br/>src/handlers/socket_flow_handler/position_updates.rs:19,245-252
            PU-->>C: rejected, too many simultaneous drags, dropped silently
        else under the cap
            PU->>GSS: NodeInteractionMessage Dragged, resumes physics if auto-paused<br/>src/handlers/socket_flow_handler/position_updates.rs:270-276
            GSS->>PO: forwarded, a grab clears the FastSettle latch and re-kicks the pipeline<br/>src/actors/physics_orchestrator_actor.rs:1161-1197
            PU->>GPU: PinNodePositions{pins:[(node_id,[x,y,z])], reheat:true}<br/>src/handlers/socket_flow_handler/position_updates.rs:281-288
            PU-->>C: nodeDragStartAck<br/>src/handlers/socket_flow_handler/position_updates.rs:294-301
            PU->>PU: start drag timeout checker<br/>src/handlers/socket_flow_handler/position_updates.rs:303-311
        end
    end
    C->>WSS: text nodeDragUpdate {nodeId, position}
    WSS->>PU: handle_node_drag_update(act, msg, ctx)<br/>src/handlers/socket_flow_handler/message_routing.rs:58-60
    alt act.pubkey.is_none()
        PU-->>C: rejected, unauthenticated<br/>src/handlers/socket_flow_handler/position_updates.rs:329-332
    else within MIN_DRAG_INTERVAL_MS=16 of this node's last update<br/>src/handlers/socket_flow_handler/position_updates.rs:22,349-353
        PU-->>C: ignored (rate limited)
    else accepted update
        PU->>GPU: PinNodePositions{updated pin, reheat:false}, the ONLY effect of a drag update<br/>src/handlers/socket_flow_handler/position_updates.rs:418-427
        Note right of GPU: the pin is a display-space position, the node is shown exactly where the hand holds it<br/>and the neighbours spring around it on the next step, show_pinned_at_pin<br/>src/actors/gpu/force_compute_actor.rs:553-566,2249
        GPU->>CC: the moved layout reaches every client through the ordinary entitled chunk stream, see VC-13.3<br/>src/actors/gpu/force_compute_actor.rs:2272-2277
        Note right of PU: RESOLVED the old drag path ran CPU SimulationStep calls on the graph-state copy and broadcast the whole graph to every<br/>session with default class ids, which bypassed the pubkey drop-set of ADR-2003. It is deleted, drags go through the GPU stream only<br/>src/handlers/socket_flow_handler/position_updates.rs:411-417
        Note right of PU: DRIFT the comment says the other users see the stream broadcast at the interaction rate, ADR-2141 removed<br/>that rate, a grab is simply a pin whose neighbours move past the threshold and are streamed as fast as each client acknowledges<br/>src/handlers/socket_flow_handler/position_updates.rs:413-414,docs/adr/ADR-2141-positions-are-streamed-per-client-until-acknowledged.md:36
    end
    C->>WSS: text nodeDragEnd {nodeId}
    WSS->>PU: handle_node_drag_end(act, msg, ctx)<br/>src/handlers/socket_flow_handler/message_routing.rs:55-57
    PU->>GSS: NodeInteractionMessage Released, no CPU settle and no side-channel broadcast, the node STAYS pinned on the GPU<br/>src/handlers/socket_flow_handler/position_updates.rs:479-492
    PU-->>C: nodeDragEndAck<br/>src/handlers/socket_flow_handler/position_updates.rs:496-504
    C->>WSS: text nodeUnpin {nodeId}
    WSS->>PU: handle_node_unpin(act, msg, ctx)<br/>src/handlers/socket_flow_handler/message_routing.rs:61-63
    alt act.pubkey.is_none()
        PU-->>C: rejected, unauthenticated<br/>src/handlers/socket_flow_handler/position_updates.rs:522-525
    else authenticated
        PU->>GPU: PinNodePositions{unpin:[node_id], reheat:true}<br/>src/handlers/socket_flow_handler/position_updates.rs:551-561
        PU-->>C: nodeUnpinAck<br/>src/handlers/socket_flow_handler/position_updates.rs:564-571
    end
    Note over PU,GPU: a stalled drag releases its pin without a reheat after drag_timeout_ms, the failure path does not<br/>route through nodeDragEnd, which would leave a stale anchor<br/>src/handlers/socket_flow_handler/position_updates.rs:594-619
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
    participant CC as ClientCoordinatorActor<br/>src/actors/client_coordinator_actor.rs:1450
    participant CM as ClientManager relay_text_to_pubkey<br/>src/actors/client_coordinator_actor.rs:584
    participant SIB as sibling session actor<br/>src/handlers/socket_flow_handler/actor_messages.rs:66

    rect rgb(222,236,250)
    C->>WSS: text beatClock or memoryRoute (ADR-2134)<br/>src/handlers/socket_flow_handler/message_routing.rs:64-68
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
            CC->>CM: relay_text_to_pubkey(pubkey, exclude, message)<br/>src/actors/client_coordinator_actor.rs:1453-1457
            loop each client whose ClientState.pubkey equals the sender and id is not the sender<br/>src/actors/client_coordinator_actor.rs:594-597
                CM->>SIB: try_send(SendToClientText(message))<br/>src/actors/client_coordinator_actor.rs:598-605
                SIB->>SIB: ctx.text(msg.0)<br/>src/handlers/socket_flow_handler/actor_messages.rs:66-71
            end
        end
    end
    Note over CM: INVARIANT the match is on the coordinator's ClientState.pubkey, which only AuthenticateClient sets,<br/>so SetClientId forwarding an upgrade-time pubkey (VC-13.1) is what lets a header or subprotocol<br/>session receive a relayed frame without sending an authenticate message first<br/>src/actors/client_coordinator_actor.rs:1479-1482,src/handlers/socket_flow_handler/actor_messages.rs:22-30
    end
```

## VC-13.6 Connect-time sync: topology over JSON, positions from the stream
```mermaid
sequenceDiagram
    autonumber
    participant WSS as SocketFlowServer started<br/>src/handlers/socket_flow_handler/types.rs:383
    participant SYNC as send_full_state_sync<br/>src/handlers/socket_flow_handler/types.rs:158
    participant GSS as GraphServiceSupervisor<br/>src/actors/graph_service_supervisor.rs:421
    participant CC as ClientCoordinatorActor<br/>src/actors/client_coordinator_actor.rs:617

    rect rgb(225,228,245)
    WSS->>CC: RegisterClient{recipients} with the stall timeout taken from the heartbeat timeout<br/>src/handlers/socket_flow_handler/types.rs:389-407
    CC->>CC: register_client gives the session a fresh ClientStream, nothing acknowledged<br/>src/actors/client_coordinator_actor.rs:282-300,1080
    CC->>CC: pump(Some(client_id)), every entitled node is a candidate, so the initial position sync is chunks through the ordinary path<br/>src/actors/client_coordinator_actor.rs:1108-1110
    Note right of CC: there is no connect-time position frame, the 3,000-node quality-top-N binary frame the old sync sent is deleted<br/>src/handlers/socket_flow_handler/types.rs:156-158
    WSS->>SYNC: self.send_full_state_sync(ctx), then connection_established{protocol supported 3 and 5}<br/>src/handlers/socket_flow_handler/types.rs:462-479
    SYNC->>SYNC: bump the sync generation, a newer sync supersedes an older one<br/>src/handlers/socket_flow_handler/types.rs:174-178
    SYNC->>GSS: send GetGraphData, then GetSettings<br/>src/handlers/socket_flow_handler/types.rs:180-189
    GSS-->>SYNC: shared GraphData snapshot, GPU-computed positions<br/>src/actors/graph_service_supervisor.rs:1485-1500
    SYNC->>WSS: GenGatedText state_sync{nodes_count, edges_count, settings version}<br/>src/handlers/socket_flow_handler/types.rs:190-223
    SYNC->>SYNC: initial_node_limit from settings, default 3000, top-N by quality score<br/>src/handlers/socket_flow_handler/types.rs:237-244,279-296
    SYNC->>SYNC: drop private nodes not owned by this pubkey, anonymous sees public-only, fail-closed<br/>src/handlers/socket_flow_handler/types.rs:257-273
    SYNC->>WSS: GenGatedInitialGraphLoad{nodes, edges both inside the cap}, JSON, positions inside are advisory<br/>src/handlers/socket_flow_handler/types.rs:338-345
    WSS->>WSS: ctx.text(json), a superseded generation is dropped at the socket write<br/>src/handlers/socket_flow_handler/actor_messages.rs:121-130
    CC->>WSS: V5 chunks follow the acknowledgements, see VC-13.4<br/>src/actors/client_coordinator_actor.rs:1384-1392
    Note over WSS,CC: a client that asks again gets the JSON topology only, at most once per 30 seconds per connection<br/>src/handlers/socket_flow_handler/position_updates.rs:45-80,src/handlers/socket_flow_handler/message_routing.rs:31-33
    Note over SYNC,CC: after an authenticate frame the sync is re-pushed for the new identity, and AuthenticateClient sets the coordinator's pubkey,<br/>so the next pump recomputes the drop-set and sends the newly entitled nodes as never-acknowledged candidates<br/>src/handlers/socket_flow_handler/filter_auth.rs:92-123,src/actors/client_coordinator_actor.rs:1479-1482,1004-1006
    Note over SYNC,CC: OPEN the stream carries every entitled node while the JSON initial load stops at initialNodeLimit, so a client sees position<br/>records for ids its topology does not hold, the web client counts them and refetches topology on a 30 second cooldown<br/>src/handlers/socket_flow_handler/types.rs:295-296,client/src/features/graph/managers/graphDataManager.ts:515-540
    end
```

## VC-13.8 Client session lifecycle
```mermaid
stateDiagram-v2
    [*] --> Connecting
    Connecting --> Rejected: rate limit, origin, signature or token check fails<br/>src/handlers/socket_flow_handler/http_handler.rs:56-59,113-122,138-140,163-167,250-252,321-322
    Connecting --> Connected: 101 Switching Protocols<br/>src/handlers/socket_flow_handler/http_handler.rs:447-450
    Rejected --> [*]

    Connected --> Authenticated: release builds always, the upgrade carried a signed subprotocol or a validated header session so pubkey is set before the actor starts<br/>src/handlers/socket_flow_handler/http_handler.rs:412-415
    Connected --> Anonymous: dev or dev-auth builds only, ALLOW_INSECURE_DEFAULTS with no valid credential, public-only view<br/>src/handlers/socket_flow_handler/http_handler.rs:226-231,308-312
    Anonymous --> Authenticated: authenticate message<br/>NIP-98 kind-27235 or legacy token/pubkey<br/>src/handlers/socket_flow_handler/filter_auth.rs:10 (ADR-2009)

    Authenticated --> Dragging: nodeDragStart requires pubkey<br/>src/handlers/socket_flow_handler/position_updates.rs:194-197
    Dragging --> Authenticated: nodeDragEnd leaves the node pinned, nodeUnpin or a drag timeout releases it<br/>src/handlers/socket_flow_handler/position_updates.rs:479-483,551-561,600-619

    Connected --> Streaming: started registers the client with the coordinator, every registered client receives the push, no subscribe needed<br/>src/handlers/socket_flow_handler/types.rs:393-407,src/actors/client_coordinator_actor.rs:414-418
    Streaming --> Congested: ClientCoordinatorActor try_send hits SendError Full<br/>on the client mailbox, only that chunk is skipped<br/>src/actors/client_coordinator_actor.rs:232-239
    Congested --> Streaming: mailbox drains, the next pump_one try_send succeeds and the clock is cleared<br/>src/actors/client_coordinator_actor.rs:322-333
    Congested --> Closed: congested for the whole stall timeout, evicted and the socket shut from outside<br/>src/actors/client_coordinator_actor.rs:336-362

    Streaming --> Closed: WS Close frame, or no inbound frame for heartbeatTimeout (server ping every heartbeatInterval)<br/>src/handlers/socket_flow_handler/mod.rs:34,51-55,src/handlers/socket_flow_handler/types.rs:434-460
    Streaming --> Closed: SendError Closed, client evicted<br/>src/actors/client_coordinator_actor.rs:238-242,682-690
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
        src/utils/socket_flow_constants.rs:7-10
        RESOLVED ADR-2054 PushDirective mechanism removed, zero senders
        tree-wide, send_pong and get_pending_directives were never called
        on the live ping/pong path
        src/handlers/socket_flow_handler/actor_messages.rs:174-177
    end note
```
