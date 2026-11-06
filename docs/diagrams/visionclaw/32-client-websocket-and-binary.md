---
id: VC-32
title: Client WebSocket transport and binary position protocol
area: visionclaw
governing:
  - ../project/docs/PROTOCOL-registry.md
  - ../project/docs/BASELINE-architecture.md
adrs: [ADR-2002, ADR-2019, ADR-2020, ADR-2047, ADR-2057, ADR-2078, ADR-2080, ADR-2139, ADR-2141, ADR-2142, ADR-2169, ADR-2183]
sources:
  - ../project/docs/BASELINE-architecture.md
  - ../project/docs/PROTOCOL-registry.md
  - ../project/client/src/store/websocket/connectionManager.ts
  - ../project/client/src/store/websocketStore.ts
  - ../project/client/src/store/websocket/index.ts
  - ../project/client/src/store/websocket/storeState.ts
  - ../project/client/src/store/websocket/types.ts
  - ../project/client/src/store/websocket/binaryFrameDispatcher.ts
  - ../project/client/src/store/websocket/binaryProtocol.ts
  - ../project/client/src/store/agentTargetStore.ts
  - ../project/client/src/store/websocket/filterSync.ts
  - ../project/client/src/store/websocket/textMessageHandler.ts
  - ../project/client/src/store/websocket/solidWebSocket.ts
  - ../project/client/src/services/solidPod/podNotifications.ts
  - ../project/client/src/services/BinaryWebSocketProtocol.ts
  - ../project/client/src/services/binaryProtocol/frameTypes.ts
  - ../project/client/src/services/binaryProtocol/agentMessages.ts
  - ../project/client/src/services/binaryProtocol/agentActionPayload.ts
  - ../project/client/src/services/binaryProtocol/backpressure.ts
  - ../project/client/src/services/binaryProtocol/ssspVoice.ts
  - ../project/client/src/services/nostrAuthService.ts
  - ../project/client/src/types/binaryProtocol.ts
  - ../project/client/src/features/graph/hooks/useGraphEventHandlers.ts
  - ../project/client/src/features/graph/hooks/dragHeartbeat.ts
  - ../project/client/src/app/AppInitializer.tsx
  - ../project/client/src/store/transientBeamStore.ts
  - ../project/xr-client/rust/src/binary_protocol.rs
  - ../project/src/utils/binary_protocol.rs
  - ../project/src/actors/position_stream.rs
  - ../project/src/handlers/socket_flow_handler/binary_protocol.rs
  - ../project/crates/visionclaw-domain/src/types/physics_config.rs
  - ../project/src/actors/client_coordinator_actor.rs
  - ../project/src/actors/client_filter.rs
  - ../project/src/handlers/socket_flow_handler/actor_messages.rs
  - ../project/docs/IDENTIFIER-taxonomy.md
  - ../project/client/src/utils/validation.ts
  - ../project/client/src/store/settings/settingsHelpers.ts
  - ../project/src/handlers/socket_flow_handler/http_handler.rs
  - ../project/src/settings/api/settings_routes.rs
  - ../project/client/src/services/signedWebSocketProtocols.ts
  - ../project/client/src/features/graph/managers/graphDataManager.ts
  - ../project/client/src/features/graph/managers/graphWorkerProxy.ts
  - ../project/src/utils/nip98.rs
  - ../project/src/handlers/socket_flow_handler/heartbeat.rs
  - ../project/src/handlers/socket_flow_handler/transport.rs
  - ../project/src/handlers/socket_flow_handler/mod.rs
  - ../project/src/handlers/socket_flow_handler/types.rs
  - ../project/src/handlers/socket_flow_handler/message_routing.rs
  - ../project/src/handlers/socket_flow_handler/position_updates.rs
verified_commit: {visionclaw: 809a859aae9e3d96d4d6b99690378018e48e3ad8}
---

## For developers

The browser keeps one `/wss` socket for everything that moves: JSON control frames (filter, drag holds, settings broadcasts, graph-change notices) and binary position frames. Since ADR-2141 the server streams position chunks per client and sends the next only after the client acknowledges the last (VC-32.12); a drag is a server-side hold, not a pin (ADR-2169, VC-32.10). At ec785afa9 the node filter gained a `serverRemove` opt-in (ADR-2183 D2): by default the quality and authority gates only grey nodes in the browser, so the client arms its shrink-guard acceptance window only when the filter can actually narrow the graph (`filterSync.ts:127-128`), matching the server's removal condition (`src/actors/client_filter.rs:28`). The `visual` settings bucket now joins the ADR-2047 broadcast (`src/settings/api/settings_routes.rs:1546`, VC-32.15).

**Debt (parallel):** the "server is removing nodes" condition `enabled && serverRemove` is restated in four places that must stay in step: the shrink-guard arm (`client/src/store/websocket/filterSync.ts:128`), the refetch re-assert (`client/src/store/websocket/textMessageHandler.ts:47`), the self-heal skip (`client/src/features/graph/managers/graphDataManager.ts:524`) and the server predicate (`src/actors/client_filter.rs:28`).

**Open (risk):** the `visual` bucket is one shared settings key (`src/settings/api/settings_routes.rs:1519`, `src/settings/api/settings_routes.rs:1528-1529`), so any signed-in user's visual write now reaches every open session live through the all-client broadcast (`src/settings/api/settings_routes.rs:454-464`); is that shared scope intended?

## For the business

Everything that changes on screen while you watch, nodes moving, someone dragging a node, a colleague changing a shared setting, arrives over one live connection. Turning on the quality filter now greys weaker pages out rather than deleting them from your view; pages are only removed when you ask for that explicitly. A change to the shared look of the graph made in one browser now shows up in every other open browser without a reload.

## VC-32.1 Connect: signed-subprotocol upgrade, dev-mode authenticate message
```mermaid
sequenceDiagram
    autonumber
    participant C as connect<br/>websocket/index.ts:154
    participant SP as signedWebSocketProtocols<br/>signedWebSocketProtocols.ts:7
    participant WS as WebSocket
    participant AH as sendAuthOnConnect<br/>connectionManager.ts:327
    participant S as socket_flow_handler<br/>http_handler.rs:49

    C->>C: signedUpgrade = !nostrAuth.isDevMode() index.ts:165
    alt signedUpgrade (every build without VITE_DEV_MODE_AUTH)
        C->>SP: signedWebSocketProtocols(state.url, nostrAuth) index.ts:166-168
        SP->>SP: sign a NIP-98 kind 27235 GET over the http(s) form of the ws URL signedWebSocketProtocols.ts:13-14
        SP-->>C: visionclaw plus nostr.EVENT, EVENT being unpadded base64url signedWebSocketProtocols.ts:16
    else dev mode
        Note over C: protocols stays empty, no credential rides the upgrade index.ts:166-168
    end
    C->>WS: new WebSocket(state.url, protocols) index.ts:169
    WS->>S: HTTP Upgrade /wss, Sec-WebSocket-Protocol visionclaw and nostr.EVENT
    S->>S: websocket_auth_header decodes the single nostr. token, two tokens or over 16384 chars is refused http_handler.rs:151-154, nip98.rs:639-655
    alt signed subprotocol present
        S->>S: verify_nip98_auth(header, signed_upgrade_url, GET) single-use http_handler.rs:156-169
        Note right of S: the signed url is rebuilt by crate::utils::auth::nip98_request_url, the same function REST NIP-98 uses, so a proxy sending both Forwarded and X-Forwarded-* cannot make the two paths disagree (ER-25) http_handler.rs:42-46
        alt invalid or replayed
            S-->>WS: 401 Invalid or replayed WebSocket signature http_handler.rs:163-166
        else verified
            Note over S: signed_user = Some(user) http_handler.rs:162
        end
    else no signed subprotocol, Authorization Bearer token
        S->>S: token = Authorization Bearer header http_handler.rs:178-184
        S->>S: get_session(token), a valid session becomes signed_user http_handler.rs:222, http_handler.rs:258-262
        alt token missing or session invalid in a release build
            S-->>WS: 401 http_handler.rs:250-251, http_handler.rs:317-322
        end
    end
    S->>S: ws_server.pubkey and is_power_user set from signed_user http_handler.rs:412-415
    S-->>WS: 101 Switching Protocols, only visionclaw and permessage-deflate are echoed http_handler.rs:10, http_handler.rs:447-450
    WS->>C: onopen index.ts:172
    C->>C: webSocketRegistry.register(graph,url,socket) index.ts:178
    alt signedUpgrade
        Note over AH: sendAuthOnConnect is skipped, identity was fixed at the upgrade index.ts:185
    else dev mode
        C->>AH: sendAuthOnConnect(socket,url) index.ts:185
        AH->>S: authenticate token=dev-session-token pubkey ephemeral connectionManager.ts:330-337
    end
    opt nodeFilter settings present
        C->>S: filter_update, forced once per connect and de-duplicated afterwards index.ts:189, filterSync.ts:110-118
    end
    C->>C: initializeBatchQueue(get) index.ts:191
    C->>C: setupFilterSubscription(get) index.ts:192
    C->>C: notifyConnectionStatusHandlers(true) index.ts:193
    C->>C: startHeartbeat(get) index.ts:194
    C->>C: processMessageQueue(get,set) index.ts:195
    Note over S: DRIFT: PROTOCOL-registry.md:344-346 says the Authorization header is now the only accepted carrier in a release build.<br/>http_handler.rs:151-169 also accepts the signed Sec-WebSocket-Protocol NIP-98 subprotocol, and that is the carrier this client<br/>uses in every non-dev build (index.ts:165-169). ADR-2058 removed only the ?token= query fallback, which is compiled out of release<br/>(http_handler.rs:186-214). BASELINE-architecture.md:404 still lists the query token as an open divergence, written before ADR-2058 landed.
    Note over S: RESOLVED 0d636fd0e (S-7f, accounted for in VC-03.2): a validated Authorization-header session is now carried into signed_user<br/>(http_handler.rs:258-262), so pubkey and is_power_user are set for a header-token client exactly as for the signed-subprotocol client<br/>(http_handler.rs:412-415). Before it the session was checked then dropped and that client stayed anonymous
    Note over AH: DEBT: the NIP-98 authenticate event branch of sendAuthOnConnect (connectionManager.ts:338-348) is unreachable from its only caller.<br/>index.ts:185 calls it only when signedUpgrade is false, which is exactly when isDevMode() is true, so the dev branch (connectionManager.ts:330)<br/>always wins. The server still handles authenticate (message_routing.rs:36-38), the client just no longer sends the event form
    Note over S: INVARIANT: the credential subprotocol is never echoed back, only PUBLIC_WS_PROTOCOLS (http_handler.rs:10, http_handler.rs:447-450)
```
## VC-32.2 Reconnect backoff and ping/pong heartbeat (binary silence is not a failure signal)
```mermaid
sequenceDiagram
    autonumber
    participant WS as WebSocket
    participant C as store<br/>websocket/index.ts:211
    participant HB as heartbeat<br/>connectionManager.ts:209
    participant SRV as SocketFlowServer<br/>types.rs:323

    WS-->>C: onclose(event) index.ts:211
    alt event.code in 1000,1001 and event.wasClean index.ts:230-233
        C->>C: updateConnectionState(disconnected) index.ts:237
    else abnormal closure
        C->>C: updateConnectionState(reconnecting,event.reason) index.ts:234
        C->>C: attemptReconnect(get,set,updateConnectionState) index.ts:235, connectionManager.ts:289
        loop while reconnectAttempts less than maxReconnectAttempts(10) connectionManager.ts:300
            Note over C: delay = min(1000 times 2 pow attempts + jitter(0-500ms), MAX_RECONNECT_DELAY=30000ms) connectionManager.ts:303-305
            C->>WS: setTimeout(get().connect, delay) connectionManager.ts:313
        end
        Note over C: reconnectAttempts exhausted -> updateConnectionState(failed) connectionManager.ts:320-321
    end
    Note over HB: HEARTBEAT_INTERVAL_MS=30000, HEARTBEAT_TIMEOUT_MS=45000 connectionManager.ts:25-26
    loop every 30000ms after startHeartbeat(get) connectionManager.ts:209-216
        HB->>SRV: socket.send(ping) as a plain text frame connectionManager.ts:250
        SRV-->>HB: text frame pong, message_routing.rs:20-23
        HB->>HB: pong -> handleHeartbeatResponse() binaryFrameDispatcher.ts:127-130, connectionManager.ts:266
        opt no pong within 45000ms
            HB->>WS: handleHeartbeatTimeout closes the socket with 4000 Heartbeat timeout connectionManager.ts:277-283
        end
    end
    Note over C: INVARIANT. liveness is ping/pong only. Binary silence is NOT a failure signal: with acknowledged streaming (ADR-2141) a client that holds the latest positions is sent nothing at all, so the old binary-silence watchdog (probe subscribe, request_full_snapshot, recycle after two unanswered probes) is removed connectionManager.ts:205-207
    Note over SRV: server side, a second watchdog the client does not see. Since 8d17f1800 the server pings every system.websocket.heartbeatInterval<br/>(types.rs:374-385) and closes with code 1001 once no inbound frame of any kind has arrived for heartbeatTimeout (heartbeat.rs:84-90, types.rs:386-396).<br/>Text and binary frames count as inbound (mod.rs:27-35), so the client text ping above keeps the socket alive. Defaults are 10 s and 60 s (heartbeat.rs:106-112),<br/>a timeout under two intervals is raised to two (heartbeat.rs:35-39)
    Note over SRV: a connected peer that stops reading is a third case. The server evicts it after the same heartbeat timeout of congestion and closes the socket<br/>from outside the session actor through a dup of the connection fd (types.rs:329-346, transport.rs:10-15), since a blocked write<br/>stops the actor, its heartbeat timer included. A merely full mailbox no longer evicts, it skips one frame (d90064950, VC-13.3), and the chunk it carried is rebuilt after the ack timeout (VC-32.8)
```
## VC-32.3 Binary frame demux: onmessage tag dispatch and the ordered frame queue
```mermaid
sequenceDiagram
    autonumber
    participant WS as WebSocket
    participant MH as createMessageHandler<br/>binaryFrameDispatcher.ts:101
    participant VB as validateBinaryData<br/>store/websocket/binaryProtocol.ts:183
    participant SF as BinaryFrameDispatcher<br/>binaryFrameDispatcher.ts:47
    participant PB as processBinaryData<br/>store/websocket/binaryProtocol.ts:456

    WS->>MH: onmessage(event) binaryFrameDispatcher.ts:124
    alt event.data equals pong
        MH->>MH: handleHeartbeatResponse() connectionManager.ts:266
    else event.data instanceof Blob
        MH->>VB: arrayBuffer().then(validateBinaryData) binaryFrameDispatcher.ts:139-144
    else event.data instanceof ArrayBuffer
        MH->>VB: validateBinaryData(event.data) binaryFrameDispatcher.ts:156
    else text frame
        MH->>MH: JSON.parse then handleTextMessage(message) binaryFrameDispatcher.ts:171-178
    end
    VB->>VB: read firstByte = view.getUint8(0) store/websocket/binaryProtocol.ts:193
    alt firstByte is 3(V3), 5(V5) or 0x23(AGENT_ACTION) store/websocket/binaryProtocol.ts:197
        VB-->>SF: dispatcher.handle(buffer) binaryFrameDispatcher.ts:141,157
    else unknown lead byte
        VB-->>MH: reject, log warn, drop frame store/websocket/binaryProtocol.ts:197, :198
    end
    SF->>SF: handle(buffer) pushes onto an ordered queue, MAX_QUEUED_FRAMES=256 binaryFrameDispatcher.ts:28,71-80
    opt queue already holds 256 frames
        SF->>SF: drop the OLDEST frame, overflowed++ (warn every 100th), the server rebuilds the unacknowledged chunk after its ack timeout binaryFrameDispatcher.ts:72-77
    end
    alt a frame is already in flight
        SF->>SF: wait, frames are applied in arrival order and none is coalesced binaryFrameDispatcher.ts:56
    else idle
        SF->>PB: processBinaryData(queue.shift(),get,set) binaryFrameDispatcher.ts:57-60
    end
    Note over SF,PB: the graph worker proxy applies the same rule, an ordered FIFO with a 256-frame backstop where each call resolves when THAT frame has been applied, replacing its newest-wins slot (ADR-2141)<br/>graphWorkerProxy.ts:44, graphWorkerProxy.ts:227, graphWorkerProxy.ts:243
    PB->>PB: firstByte switch: PROTOCOL_V3/V5 only -> handleLegacyBinaryData store/websocket/binaryProtocol.ts:469, :470
    Note over PB: RESOLVED ADR-2078: PROTOCOL_V2 removed from the accept list. The server<br/>rejects V2 outright (src/utils/binary_protocol.rs:610) so a 0x02 lead byte can<br/>only be a stale sender or a mis-routed payload - it is no longer routed here
    PB->>PB: firstByte MessageType.AGENT_ACTION(0x23) -> handleAgentActionTagged store/websocket/binaryProtocol.ts:476, :477
    PB->>PB: else parseHeader(data) then switch(header.type) store/websocket/binaryProtocol.ts:482, :488
    Note over VB,PB: DEBT: the parseHeader switch is unreachable from the socket. Its only caller is the dispatcher (binaryFrameDispatcher.ts:60), which is fed<br/>only frames validateBinaryData passed, and that accepts lead bytes 3, 5 and 0x23 alone (store/websocket/binaryProtocol.ts:197). The 0x23 and V3/V5 branches<br/>return first (:469-479), so GRAPH_UPDATE, VOICE_DATA, POSITION_UPDATE, AGENT_POSITIONS and the default arm (:488-504) never run. Delete them or admit their tags
    PB-->>SF: promise settles -> finally clears inFlight and re-pumps the queue via queueMicrotask binaryFrameDispatcher.ts:64-68
    Note over PB: ADR-2019 tag byte is registry-governed: byte 0 selects the codec, unknown tags rejected not reinterpreted (docs/PROTOCOL-registry.md tag registry, src/utils/binary_protocol.rs:610, :620)
```
## VC-32.4 Outbound framed-message header (client-originated only)
```mermaid
classDiagram
    class FramedMessageHeader {
      +uint8 type  offset0
      +uint8 version  offset1
      +uint32 payloadLength  offset2 littleEndian
      +uint8 graphTypeFlag  offset6 GRAPH_UPDATE only
    }
    class MessageType {
      GRAPH_UPDATE 0x01
      VOICE_DATA 0x02
      POSITION_UPDATE 0x10
      AGENT_POSITIONS 0x11
      AGENT_STATE_FULL 0x20
      AGENT_ACTION 0x23
      CONTROL_BITS 0x30
      SSSP_DATA 0x31
      BROADCAST_ACK 0x34
      VOICE_CHUNK 0x40
      VOICE_START 0x41
      VOICE_END 0x42
      SYNC_UPDATE 0x50
      ANNOTATION_UPDATE 0x51
      SELECTION_UPDATE 0x52
      USER_POSITION 0x53
      VR_PRESENCE 0x54
      ERROR 0xFF
    }
    class GraphTypeFlag {
      KNOWLEDGE_GRAPH 0x00
      ONTOLOGY 0x01
    }
    class ControlFlags {
      PAUSE_UPDATES bit0
      HIGH_FREQUENCY bit1
      LOW_BANDWIDTH bit2
      VOICE_ENABLED bit3
      DEBUG_MODE bit4
      FORCE_FULL_UPDATE bit5
      USER_INTERACTING bit6
      BACKGROUND_MODE bit7
    }
    FramedMessageHeader --> MessageType : selects codec
    FramedMessageHeader --> GraphTypeFlag : graphTypeFlag field
    note for FramedMessageHeader "createMessage/parseHeader BinaryWebSocketProtocol.ts:73-116. Header size<br/>MESSAGE_HEADER_SIZE=6, GRAPH_UPDATE_HEADER_SIZE=7 (frameTypes.ts:225-226). Used<br/>only by client-to-server encoders: encodePositionUpdates, encodeAgentState,<br/>encodeSSSPData, encodeControlBits, encodeVoiceChunk, flushPositionUpdates and<br/>encodeAgentAction (BinaryWebSocketProtocol.ts:129-224, :293-295). createBroadcastAck is NOT framed:<br/>it writes a bare [0x34] then the 20-byte payload (BinaryWebSocketProtocol.ts:245-255)."
    note for FramedMessageHeader "DEBT: no client code sends a framed message. The six encode* methods have no caller in client/src<br/>outside tests, and flushPositionUpdates drains pendingPositionUpdates, which only<br/>encodePositionUpdates fills (BinaryWebSocketProtocol.ts:136, :213-216), so the flush sent on<br/>interaction end is always null (graphDataManager.ts:611-614). The envelope is dead outbound as well."
    note for MessageType "RESOLVED ADR-2078. The framed header's version byte is written from<br/>PROTOCOL_VERSION (BinaryWebSocketProtocol.ts:86), which is PROTOCOL_V3=3 (frameTypes.ts:30), and the<br/>comment above it now says so (BinaryWebSocketProtocol.ts:81-84). The old header-local PROTOCOL_V4=4 is<br/>deleted from frameTypes.ts (frameTypes.ts:33-37) so PROTOCOL_V4 now names only the delta node<br/>encoding of VC-32.5, and SUPPORTED_HEADER_VERSIONS is V3 and V5 (frameTypes.ts:39). ADR-2078 also<br/>deleted five enum members with no reference anywhere in client/src: VELOCITY_UPDATE 0x12,<br/>AGENT_STATE_DELTA 0x21, AGENT_HEALTH 0x22, HANDSHAKE 0x32 and HEARTBEAT 0x33 (frameTypes.ts:59-63)."
    note for MessageType "The enum is frameTypes.ts:65-101. The server's own MessageType has only five tags. The code comment says this client decodes<br/>AGENT_ACTION and BROADCAST_ACK from the wire (frameTypes.ts:45-57). Only AGENT_ACTION is decoded:<br/>decodeBroadcastAck (backpressure.ts:47) has no caller but its own wrapper (BinaryWebSocketProtocol.ts:260),<br/>since 0x34 travels client to server. Every other member is client-outbound only. Live position frames are unframed and<br/>lead with a bare 0x03 or 0x05 byte, so they never carry a MessageType at all."
    note for FramedMessageHeader "DOC-DRIFT. The docs/PROTOCOL-registry.md frame tag registry lists 0x03, the retired 0x04, 0x05,<br/>0x34 (client to server, no generic header), 0x23, 0x43 and 0x44 (PROTOCOL-registry.md:64-73). This<br/>client-only framed envelope and the rest of its MessageType space (0x01, 0x02, 0x10-0x54) do not<br/>appear in that registry at all."
    note for MessageType "BROADCAST_ACK 0x34 payload (encodeBroadcastAckPayload/decodeBroadcastAck<br/>backpressure.ts:18-69): BROADCAST_ACK_PAYLOAD_SIZE=20 bytes - sequenceId split<br/>low/high uint32 offset0/4, nodesReceived uint32 offset8, timestamp split<br/>low/high uint32 offset12/16, all littleEndian to avoid BigInt64<br/>(backpressure.ts:26-29,36-39). Every applied V5 chunk is acknowledged, there is no batch<br/>cadence (VC-32.8)."
    note for MessageType "SSSP_DATA 0x31 and VOICE_CHUNK 0x40 payloads (ssspVoice.ts:8-91).<br/>SSSPDataRecord: uint32 nodeId, float32 distance, uint32 parentId, uint16 flags<br/>= 14 bytes (SSSP_DATA_SIZE_V2 frameTypes.ts:230, layout ssspVoice.ts:8-21). VoiceChunkHeader: uint16<br/>agentId, uint16 chunkId, uint8 format, uint16 dataLength = 7 bytes<br/>(VOICE_HEADER_SIZE frameTypes.ts:237) then raw audioData (ssspVoice.ts:49-65). Voice agentId is<br/>uint16 (max 65535) - a wire format separate from the uint32 flag-bit node/agent<br/>id of VC-32.6, no alignment issue (ssspVoice.ts:54-56,82-84)."
```
## VC-32.5 Server-to-client position wire records: V3, V5 envelope (V4 delta retired, V2 declined)
```mermaid
classDiagram
    class V3NodeRecord {
      +uint32 id  offset0  bits0to25 id, bits26to31 flags
      +float32x3 position  offset4
      +float32x3 velocity  offset16
      +float32 sssp_distance  offset28  default Infinity
      +int32 sssp_parent  offset32  default -1
      +uint32 cluster_id  offset36
      +float32 anomaly_score  offset40
      +uint32 community_id  offset44
      +float32 centrality  offset48
      52 bytes per record, leading tag byte 0x03
    }
    class V2NodeRecord {
      +uint32 id  offset0
      +float32x3 position  offset4
      +float32x3 velocity  offset16
      +float32 sssp_distance  offset28
      +int32 sssp_parent  offset32
      36 bytes per record, no analytics fields
    }
    class V5Envelope {
      +uint8 version  offset0  equals 5
      +uint64 broadcast_seq  offset1  8 bytes littleEndian
      +V3NodeRecord[] body  offset9 onward
    }
    class V4DeltaFrame_RETIRED {
      +uint8 version  offset0  equals 4
      +uint8 frame_number  offset1
      +uint16 num_changed  offset2
      +DeltaItem[] items  offset4 onward, 20 bytes each
    }
    class DeltaItem {
      +uint32 node_id  offset0
      +uint8 change_flags  offset4  bit0 pos, bit1 vel
      +int16 dx_dy_dz  offset8,10,12  divide by DELTA_SCALE_FACTOR 100
      +int16 dvx_dvy_dvz  offset14,16,18  divide by 100
    }
    V5Envelope --> V3NodeRecord : wraps
    V4DeltaFrame_RETIRED --> DeltaItem : contained
    note for V3NodeRecord "CROSS-CHECK OK. client/src/types/binaryProtocol.ts:84-94 offsets match<br/>xr-client/rust/src/binary_protocol.rs:8-11,947-969 exactly: id, pos, vel,<br/>sssp_distance, sssp_parent, cluster_id, anomaly, community_id, centrality. Godot<br/>NODE_RECORD_BYTES=52 (:32) equals client BINARY_NODE_SIZE_V3=52<br/>(types/binaryProtocol.ts:79). No DIVERGENCE."
    note for V2NodeRecord "Declined, not live. The server answers a 0x02 lead byte with Err (src/utils/binary_protocol.rs:610),<br/>its 36-byte stride constant is deleted, and the client declares no 36-byte stride at all (ADR-2057,<br/>types/binaryProtocol.ts:76-79). The shape is kept here only to show what the 52-byte V3 record replaced."
    note for V5Envelope "types/binaryProtocol.ts:328-349 parseV5Nodes reads seqLow at offset1 and seqHigh<br/>at offset5, reconstructs a V3 buffer at offset9. Matches Godot PROTOCOL_V5=0x05<br/>and V5_SEQ_BYTES=8, docs/PROTOCOL-registry.md 0x05 row<br/>(xr-client/rust/src/binary_protocol.rs:29-30)."
    note for V4DeltaFrame_RETIRED "RETIRED (ADR-2139, ADR-2141). No encoder exists on the server and every decoder refuses the tag: the web parser logs a<br/>decline and applies nothing (client/src/types/binaryProtocol.ts:190-198), the registry keeps 0x04 reserved, never to be reused<br/>(docs/PROTOCOL-registry.md:67). Every position frame is an absolute chunk, so there is nothing to add offsets to.<br/>PROTOCOL_V4 now means only the retired delta tag (client/src/services/binaryProtocol/frameTypes.ts:37)."
```
## VC-32.6 Wire node-id: type flag bits over the 26-bit id
```mermaid
classDiagram
    class WireNodeId {
      +uint32 raw  the full id as sent on the wire
      bit31 AGENT_NODE_FLAG 0x80000000
      bit30 KNOWLEDGE_NODE_FLAG 0x40000000
      bits26to28 ONTOLOGY_TYPE_MASK 0x1C000000
      bits0to25 NODE_ID_MASK 0x03FFFFFF  actual id, 0 to 67108863
    }
    class OntologySubtype {
      ONTOLOGY_CLASS_FLAG 0x04000000
      ONTOLOGY_INDIVIDUAL_FLAG 0x08000000
      ONTOLOGY_PROPERTY_FLAG 0x10000000
    }
    WireNodeId --> OntologySubtype : bits26to28 when ontology
    note for WireNodeId "INVARIANT, no DIVERGENCE. Client client/src/types/binaryProtocol.ts:96-107, Godot<br/>decoder xr-client/rust/src/binary_protocol.rs:41-47, and<br/>docs/IDENTIFIER-taxonomy.md:125-130 all agree exactly on mask and flag hex values.<br/>getActualNodeId(nodeId) is nodeId AND NODE_ID_MASK<br/>(types/binaryProtocol.ts:133-135)."
    note for WireNodeId "docs/IDENTIFIER-taxonomy.md:132-142: ids are sequential u32 from a NEXT_NODE_ID<br/>atomic counter. Every encoder branch, the untyped one included, routes through<br/>enforce_wire_id_bounds (src/utils/binary_protocol.rs:185-206): debug_assert fails fast in<br/>development (:186-193) but release does NOT truncate silently. remap_wire_id masks the id into<br/>26 bits and an error! log names the class and both ids (:194-204, ADR-2024). The masked id<br/>still aliases another node, so an over-range id is loud, not harmless."
    note for WireNodeId "INVARIANT (2026-09-08, 6c47b9833): an id that ALREADY carries a recognised<br/>class pattern is forwarded unchanged by the encoder's last branch. Running the 26-bit<br/>bound check on a stamped id panicked every broadcast in debug builds - each XR client<br/>saw a 2 s connection-reset loop - and would have stripped the class flag in release<br/>(src/utils/binary_protocol.rs:475-484). Only a high bit that is NOT a class<br/>pattern is still an overflow, handled by the untyped branch (src/utils/binary_protocol.rs:490)."
    note for WireNodeId "DRIFT: the comment at src/utils/binary_protocol.rs:476-480 still says position_updates.rs and<br/>actor_messages.rs stamp the flag themselves. Neither does at this revision: the stamping<br/>requestSwarmTelemetry handler was removed (ADR-2142, message_routing.rs:11-13), and the one live<br/>position encoder passes real class sets (client_coordinator_actor.rs:650-661). The branch is now defensive."
```
## VC-32.7 getNodeType(): flag precedence decision order
```mermaid
flowchart TD
    A["nodeId: uint32"] --> B{"nodeId AND AGENT_NODE_FLAG != 0?<br/>types/binaryProtocol.ts:119"}
    B -- yes --> AG["NodeType.Agent"]
    B -- no --> C{"nodeId AND KNOWLEDGE_NODE_FLAG != 0?<br/>types/binaryProtocol.ts:121"}
    C -- yes --> KN["NodeType.Knowledge"]
    C -- no --> D{"nodeId AND ONTOLOGY_TYPE_MASK == CLASS_FLAG?<br/>types/binaryProtocol.ts:123"}
    D -- yes --> OC["NodeType.OntologyClass"]
    D -- no --> E{"nodeId AND ONTOLOGY_TYPE_MASK == INDIVIDUAL_FLAG?<br/>types/binaryProtocol.ts:125"}
    E -- yes --> OI["NodeType.OntologyIndividual"]
    E -- no --> F{"nodeId AND ONTOLOGY_TYPE_MASK == PROPERTY_FLAG?<br/>types/binaryProtocol.ts:127"}
    F -- yes --> OP["NodeType.OntologyProperty"]
    F -- no --> UN["NodeType.Unknown  types/binaryProtocol.ts:130"]
    AG --> G["updateNodeTypeMapFromParsed caches actualId to type<br/>evicting oldest past MAX_NODE_TYPE_ENTRIES=65536<br/>websocket/binaryProtocol.ts:151, :168"]
    KN --> G
    OC --> G
    OI --> G
    OP --> G
```
## VC-32.8 Live position frame processing: parse, analytics ingest, apply, then acknowledge the chunk
```mermaid
sequenceDiagram
    autonumber
    participant PB as processBinaryData<br/>store/websocket/binaryProtocol.ts:456
    participant FR as parseBinaryFrameData<br/>types/binaryProtocol.ts:357
    participant AN as nodeAnalyticsStore<br/>features/analytics/store/nodeAnalyticsStore
    participant GDM as graphDataManager<br/>features/graph/managers/graphDataManager
    participant SRV as Server /wss

    PB->>PB: handleLegacyBinaryData(data,get,set) store/websocket/binaryProtocol.ts:352
    PB->>FR: parseBinaryFrameData(data) store/websocket/binaryProtocol.ts:359
    FR-->>PB: ParsedBinaryFrame nodes, broadcastSequence (V5 only) store/websocket/binaryProtocol.ts:365
    alt protoByte is PROTOCOL_V3 or PROTOCOL_V5 store/websocket/binaryProtocol.ts:370-371
        PB->>AN: nodeAnalyticsStore.ingest(parsedNodes) store/websocket/binaryProtocol.ts:372
    else frame without analytics, reachable only from the dead default arm (VC-32.3)
        Note over PB: V2 omits cluster_id/anomaly/community offsets 36/40/44 and must NOT overwrite live analytics store/websocket/binaryProtocol.ts:367-369
    end
    PB->>PB: updateNodeTypeMapFromParsed(parsedNodes,set) store/websocket/binaryProtocol.ts:375
    opt hasBotsData: any node isAgentNode() store/websocket/binaryProtocol.ts:377
        PB->>PB: emit(bots-position-update, data) store/websocket/binaryProtocol.ts:379, :380
    end
    PB->>GDM: graphDataManager.updateNodePositions(data) store/websocket/binaryProtocol.ts:394
    alt updateNodePositions rejected
        PB-->>PB: log the error and return WITHOUT acknowledging, the server rebuilds this chunk after its ack timeout store/websocket/binaryProtocol.ts:398-402
    else applied
        opt frame.broadcastSequence defined (V5) store/websocket/binaryProtocol.ts:407
            PB->>SRV: sendPositionAck(frame.broadcastSequence, parsedNodes.length) -> createBroadcastAck, [0x34][u64 seq][u32 nodes][u64 ts] with no generic header store/websocket/binaryProtocol.ts:408, :127-138, client/src/services/BinaryWebSocketProtocol.ts:245
        end
        Note over PB,SRV: a bare V3 frame carries no sequence and so nothing to name, it is not acknowledged store/websocket/binaryProtocol.ts:404-406
    end
    Note over PB: framed-header path handlePositionUpdate (MessageType.POSITION_UPDATE/AGENT_POSITIONS, store/websocket/binaryProtocol.ts:303, :497) updates the node-type cache and emits bots-position-update (:320-329)<br/>but ingests NO analytics and sends NO acknowledgement, it carries no stream sequence (:348-349). It is unreachable from the socket, see the VC-32.3 DEBT
```
## VC-32.9 Position stream start (no subscribe message), de-duplicated filter sync, REST-aware shrink-guard
```mermaid
sequenceDiagram
    autonumber
    participant AI as AppInitializer<br/>app/AppInitializer.tsx:260
    participant C as store<br/>websocket/index.ts
    participant FS as filterSync<br/>websocket/filterSync.ts:135
    participant TH as textMessageHandler<br/>websocket/textMessageHandler.ts:229
    participant GDM as graphDataManager<br/>graphDataManager.ts:258
    participant SRV as Server /wss

    C->>AI: onConnectionStatusChange(connected=true) AppInitializer.tsx:260,270
    alt websocketService.isReady() AppInitializer.tsx:272
        AI->>GDM: setBinaryUpdatesEnabled(true) AppInitializer.tsx:275
        Note over AI,SRV: nothing is subscribed to. The server streams positions to every registered session and sends the next chunk when this client acknowledges the last, see VC-32.12. The old subscribe_position_updates message and its idempotency guard are removed AppInitializer.tsx:277-278
    else not fully established yet
        AI->>GDM: enableBinaryUpdates() AppInitializer.tsx:284
    end
    C->>FS: syncFilterToServer(get, force) once per open index.ts:189
    FS->>FS: return if not connected or no nodeFilter snapshot filterSync.ts:111-114
    FS->>FS: unforced calls return when the snapshot equals lastFilterSnapshot filterSync.ts:115-116
    FS->>C: sendFilterUpdate(snapshot) filterSync.ts:117
    C->>FS: expectFilterResponseFor(filter) index.ts:370
    alt filter.enabled and filter.serverRemove filterSync.ts:127-128
        FS->>FS: expectFilterResponse() arms the 15000ms acceptance window filterSync.ts:129, filterSync.ts:29-34
    else disabled, or enabled without the serverRemove opt-in
        Note over FS: window not armed. The server answers with the full scoped graph and the predicates only grey on the client (ADR-2183 D2), so nothing smaller may be accepted filterSync.ts:120-126
    end
    C->>SRV: filter_update message, now carrying server_remove (default false) index.ts:372-381
    Note over SRV: predicate removal only when enabled and server_remove, scope removal always (src/actors/client_filter.rs:28, src/actors/client_filter.rs:34).<br/>A fresh ClientFilter is enabled false, server_remove false (src/actors/client_coordinator_actor.rs:140, src/actors/client_coordinator_actor.rs:149)
    Note over SRV: the reply is filter_update_reply, the caller's visible_subgraph first and then this client's filter, so private nodes never ride it (src/actors/client_coordinator_actor.rs:58-73)
    C->>FS: setupFilterSubscription(get) index.ts:192
    FS->>FS: ONE whole-store useSettingsStore.subscribe, replacing seven per-path subscriptions filterSync.ts:139-146
    loop on every settings-store change while isConnected
        FS->>FS: syncFilterToServer(get) unforced, only a real filter-field change reaches the wire filterSync.ts:144, filterSync.ts:91-102
    end
    SRV-->>TH: initialGraphLoad nodes,edges textMessageHandler.ts:117-119
    TH->>GDM: isRestLoadInFlight() textMessageHandler.ts:238, graphDataManager.ts:248-250
    alt a REST graph/data load is in flight
        TH->>TH: isFilterResponse = isFilterResponseExpected(), then clear the window textMessageHandler.ts:240-242
    else no REST load in flight
        alt nodes.length less than existingNodeCount textMessageHandler.ts:244-245
            alt isFilterResponseExpected() textMessageHandler.ts:252
                TH->>TH: accept the filtered load, clearFilterResponseExpectation() textMessageHandler.ts:253-258
            else smaller unsolicited payload
                TH->>TH: skip setGraphData, positions arrive via binary stream textMessageHandler.ts:259-269
            end
        end
    end
    TH->>GDM: offerServerGraphLoad(data, isFilterResponse) textMessageHandler.ts:319-321
    alt REST load in flight
        GDM-->>TH: deferred, held latest-wins graphDataManager.ts:262-268
        Note right of GDM: when REST settles the held load is DROPPED on success unless it answers our own filter_update,<br/>and applied on REST failure graphDataManager.ts:274-287
    else
        GDM->>GDM: setGraphData(data) graphDataManager.ts:270
        TH->>TH: emit graphDataUpdated source websocket_filter textMessageHandler.ts:323-329
    end
    Note over TH: the connect-time initialGraphLoad is capped at 3,000 nodes, which is why a smaller payload needs the guard textMessageHandler.ts:246-251.<br/>The cap is server-side: physics initialNodeLimit, 0 meaning INITIAL_NODE_LIMIT_DEFAULT 3000, clamped to 100000 (types.rs:255-260, types.rs:472)
    Note over FS: DRIFT: the filterSync header comment still calls it a 200-node connect-time payload (filterSync.ts:25)
```
## VC-32.10 Drag send path: a drag is a server-side hold over JSON control frames (ADR-2169)
```mermaid
sequenceDiagram
    autonumber
    participant U as PointerEvent
    participant EH as useGraphEventHandlers<br/>features/graph/hooks/useGraphEventHandlers.ts:87
    participant HB as startDragHeartbeat<br/>features/graph/hooks/dragHeartbeat.ts:11
    participant SRV as SocketFlowServer drag handlers<br/>socket_flow_handler/position_updates.rs:193

    U->>EH: handlePointerDown(event) useGraphEventHandlers.ts:87
    EH->>EH: dragDataRef.current = pointerDown,nodeId,startNodePos3D,currentNodePos3D useGraphEventHandlers.ts:104-113
    EH->>EH: startInteraction(node.id) useGraphEventHandlers.ts:129
    U->>EH: handlePointerMove(event) useGraphEventHandlers.ts:136
    alt distance greater than DRAG_THRESHOLD(5) useGraphEventHandlers.ts:16, useGraphEventHandlers.ts:145
        EH->>EH: drag.isDragging = true useGraphEventHandlers.ts:146
        alt graphDataManager.webSocketService.isReady() useGraphEventHandlers.ts:158
            EH->>SRV: sendMessage(nodeDragStart, nodeId, startPosition) useGraphEventHandlers.ts:159-166
            SRV->>SRV: authenticated, a free hand (two per connection), then HoldNodes and nodeDragStartAck position_updates.rs:199-202, position_updates.rs:215-221, position_updates.rs:228-242
            SRV->>SRV: arm check_drag_timeout after drag_timeout_ms (2000) plus 100 position_updates.rs:244-250, types.rs:23
        else not connected or not server-ready
            Note over EH: send skipped, caught by try/catch - drag state machine still proceeds useGraphEventHandlers.ts:157,168-170
        end
        EH->>HB: startDragHeartbeat re-sends nodeDragUpdate at currentNodePos3D every 500ms useGraphEventHandlers.ts:171-184, dragHeartbeat.ts:9, dragHeartbeat.ts:15
    end
    opt drag.isDragging useGraphEventHandlers.ts:193
        EH->>EH: updateNodePosition(nodeId,intersection) local optimistic move useGraphEventHandlers.ts:224, useGraphEventHandlers.ts:230-234
        EH->>SRV: throttledWebSocketUpdate(nodeId,position) throttle 100ms useGraphEventHandlers.ts:18,55-77,237-241
        alt shouldSendPositionUpdates() and webSocketService.isReady() useGraphEventHandlers.ts:58-60
            EH->>SRV: sendMessage(nodeDragUpdate, nodeId, position, timestamp) useGraphEventHandlers.ts:65-69
        end
        SRV->>SRV: an update for an unheld node is an implicit start, one inside MIN_DRAG_INTERVAL_MS is coalesced, not dropped position_updates.rs:283-285, position_updates.rs:307-319
    end
    U->>EH: handlePointerUp(event) useGraphEventHandlers.ts:246
    alt drag.isDragging useGraphEventHandlers.ts:264
        EH->>EH: flushPositionUpdates() useGraphEventHandlers.ts:274
        alt webSocketService.isReady() useGraphEventHandlers.ts:277
            EH->>SRV: sendMessage(nodeDragEnd, nodeId) useGraphEventHandlers.ts:278-280
            SRV->>SRV: land the last pending position, then release_holds, the node rejoins the simulation position_updates.rs:390-402
        end
    else click, not a drag
        EH->>EH: onNodeSelect(nodeId) useGraphEventHandlers.ts:288-291
    end
    EH->>HB: stopHeartbeat() useGraphEventHandlers.ts:296
    Note over SRV: INVARIANT (ADR-2169): a drag is a HOLD, not a pin. nodeDragEnd, a 2 s update gap or a disconnect releases the node back into<br/>the simulation (position_updates.rs:81-84, position_updates.rs:521-545). A lasting anchor is the separate nodePin and nodeUnpin pair<br/>(position_updates.rs:423, position_updates.rs:475, message_routing.rs:60-65), which this web client never sends.<br/>ADR-2169 is accepted with implementation partial and activation inactive, so the hold is source-true, not yet runtime-proven
    Note over EH: DRIFT: the client comments still describe the retired model, nodeDragStart pins the node server-side and nodeDragEnd unpins<br/>it (useGraphEventHandlers.ts:151-155, useGraphEventHandlers.ts:269-276). The server pins nothing on a drag (position_updates.rs:183-187)
    Note over EH: the old binary sendNodePositionUpdates second-frame path was removed 2026-06-03 useGraphEventHandlers.ts:61-64
```
## VC-32.11 Agent frame 0x23 (AGENT_ACTION): decode, state-frame split, beam and registry dispatch
```mermaid
sequenceDiagram
    autonumber
    participant PB as processBinaryData<br/>store/websocket/binaryProtocol.ts:456
    participant DEC as decodeAgentActions<br/>binaryProtocol/agentMessages.ts:183
    participant BEAM as pushTransientBeams<br/>store/transientBeamStore
    participant AT as useAgentTargetStore.pushActions<br/>client/src/store/agentTargetStore.ts:48

    PB->>PB: firstByte equals MessageType.AGENT_ACTION 0x23 store/websocket/binaryProtocol.ts:476
    PB->>PB: handleAgentActionTagged(data) store/websocket/binaryProtocol.ts:477 calling fn def :427
    Note over PB: shipping wire is a bare tag, not a 6-byte framed header - [0x23][u16 count]([u16 len][event])... store/websocket/binaryProtocol.ts:412-417
    alt data.byteLength greater-or-equal 18 store/websocket/binaryProtocol.ts:428
        PB->>DEC: decodeAgentActions(data.slice(1)) store/websocket/binaryProtocol.ts:429
    else too small
        PB->>PB: actions = empty array store/websocket/binaryProtocol.ts:430
    end
    DEC->>DEC: eventCount = view.getUint16(0) agentMessages.ts:192
    loop for each event, offset advances by 2 plus eventLen agentMessages.ts:195-215
        DEC->>DEC: eventLen = view.getUint16(offset) agentMessages.ts:201
        DEC->>DEC: decodeAgentAction(eventPayload) sourceAgentId,targetNodeId,actionType,timestamp,durationMs agentMessages.ts:153-181,210
    end
    DEC-->>PB: AgentActionEvent[] passed to dispatchAgentActions store/websocket/binaryProtocol.ts:431
    opt actions.length greater than 0 store/websocket/binaryProtocol.ts:435
        PB->>PB: work = the batch minus state frames (payload carries a lifecycle state) store/websocket/binaryProtocol.ts:438, agentActionPayload.ts:52-55
        opt work.length greater than 0 store/websocket/binaryProtocol.ts:439
            PB->>PB: emit(agent-action, work) live transcript and attention heat store/websocket/binaryProtocol.ts:441
            PB->>BEAM: pushTransientBeams(work) TransientBeamsLayer sink, which filters state frames again store/websocket/binaryProtocol.ts:444, transientBeamStore.ts:119, transientBeamStore.ts:70-72
        end
        PB->>AT: useAgentTargetStore.getState().pushActions(actions), the WHOLE batch state frames included<br/>store/websocket/binaryProtocol.ts:447
        Note over PB,AT: the agent registry (ADR-2142, ADR-2135): per source id the latest target node, presence time and lifecycle.<br/>A retired state frame removes the agent at once, the 30 s presence window is the backstop<br/>agentTargetStore.ts:1-13, agentTargetStore.ts:55-57, agentTargetStore.ts:62-65
        Note over PB,AT: a state frame (the 5 s keep-alive) reaches the registry only, never a beam, heat, drift credit or transcript line store/websocket/binaryProtocol.ts:436-437
    end
    Note over PB: AGENT_ACTION_HEADER_SIZE=15 bytes: sourceAgentId u32@0, targetNodeId u32@4, actionType u8@8, timestamp u32@9, durationMs u16@13, optional payload from @15 frameTypes.ts:215,agentMessages.ts:161-165
```
## VC-32.12 Acknowledged chunk stream (ADR-2141): server pump, client apply, ack
```mermaid
sequenceDiagram
    autonumber
    participant CC as ClientCoordinator.pump_one<br/>src/actors/client_coordinator_actor.rs:548
    participant CS as ClientStream<br/>src/actors/position_stream.rs:79
    participant CL as web client<br/>store/websocket/binaryProtocol.ts:352
    participant DEC as decode_broadcast_ack<br/>src/utils/binary_protocol.rs:1866

    CC->>CS: is_waiting(now, ack_timeout)? src/actors/position_stream.rs:104
    alt latest chunk in flight and not yet timed out
        CC-->>CC: send nothing to this client src/actors/client_coordinator_actor.rs:560
    else idle, or the in-flight chunk timed out
        CC->>CS: plan(cache, visible, analytics, settings) src/actors/position_stream.rs:125
        Note over CS: candidates = entitled nodes never acknowledged, or moved more than streamMoveThreshold from the ACKNOWLEDGED position, or with changed analytics.<br/>Never-sent first, a quarter of the chunk is reserved for the longest-waiting, the rest most-displaced first.<br/>FAIR_SHARE_DIVISOR=4 src/actors/position_stream.rs:66
        CS-->>CC: at most streamChunkNodes ids, empty when the client is current
        CC->>CL: V5 frame [0x05][u64 sequence][52-byte records], absolute positions, fresh sequence src/actors/client_coordinator_actor.rs:636
        CC->>CS: record_sent(seq, positions) -- a still-unacknowledged earlier chunk lengthens the backoff src/actors/position_stream.rs:185
    end
    CL->>CL: apply the chunk in arrival order (VC-32.3, VC-32.8) store/websocket/binaryProtocol.ts:393
    CL->>DEC: 0x34 acknowledgement of that sequence, only after the chunk was applied store/websocket/binaryProtocol.ts:407-408
    DEC->>CC: ClientBroadcastAck { sequence } src/handlers/socket_flow_handler/binary_protocol.rs:67-81
    CC->>CS: acknowledge(seq) -- the chunk's nodes become the client's held state, never moving it backwards src/actors/position_stream.rs:225
    Note over CS: LATE ACKS: the last 8 sent chunks stay on the in-flight list so a slow client's acknowledgement still commits (IN_FLIGHT_HISTORY=8 src/actors/position_stream.rs:64).<br/>BACKOFF: each consecutive timeout doubles the wait up to 8x, any acknowledgement resets it (MAX_BACKOFF_DOUBLINGS=3, effective_timeout src/actors/position_stream.rs:68, :99)
    Note over CC,CL: settings physics.streamChunkNodes 2500, streamMoveThreshold 0.05, streamAckTimeoutMs 500 are the stored defaults crates/visionclaw-domain/src/types/physics_config.rs:25, :38, :47
    Note over CC,CL: INVARIANT. there is no connect-time position frame and no deltas. A new or reconnecting client has acknowledged nothing, so its first sync is this same mechanism. Delivery stops entirely once every client is current. ADR-2141 is accepted and complete with activation_status staged
    Note over CL: DEBT. the web queue is bounded at 256 frames and drops the OLDEST on overflow (binaryFrameDispatcher.ts:28,72-76), so "clients drop none" in ADR-2141 decision 6 is not literally true. The dropped chunk is unacknowledged and the server rebuilds it after the timeout
```

## VC-32.15 JSON control-frame catalogue
```mermaid
classDiagram
    class ClientToServer {
      authenticate  token/pubkey/ephemeral in dev mode only, the event form is unreachable from the client since the signed subprotocol (VC-32.1)  connectionManager.ts:332-343
      filter_update  enabled,quality_threshold,authority_threshold,filter_by_quality,filter_by_authority,filter_mode,include_linked_pages  index.ts:360-383
      BroadcastAck (binary 0x34, not JSON)  [0x34][u64 sequence][u32 nodes][u64 timestamp_ms], sent after a chunk is applied  BinaryWebSocketProtocol.ts:245, store/websocket/binaryProtocol.ts:408
      REMOVED ADR-2141  subscribe_position_updates and request_full_snapshot have no sender or handler, message_routing.rs:15-16 states there is no position request message
      nodeDragStart  nodeId,position  useGraphEventHandlers.ts:159-166
      nodeDragUpdate  nodeId,position,timestamp  useGraphEventHandlers.ts:65-69
      nodeDragEnd  nodeId  useGraphEventHandlers.ts:278-280
      ping  raw string, not JSON, answered by a raw pong  connectionManager.ts:250, message_routing.rs:20-23
    }
    class ServerToClient {
      connection_established  sets isServerReady  textMessageHandler.ts:91-95
      error  category:validation|server|protocol|auth|rate_limit, code, retryable, retryAfter  types.ts:14-23,store/websocket/binaryProtocol.ts:207, dispatched textMessageHandler.ts:98-105
      filter_update_success  data.visible_nodes,data.total_nodes  textMessageHandler.ts:107-115
      initialGraphLoad  nodes,edges  textMessageHandler.ts:117-119,229-331
      memory_flash  data forwarded to memoryFlash event  textMessageHandler.ts:122-124
      graphUpdated  revision,reason, debounced 750ms refetch  textMessageHandler.ts:27-60,129-133
      pong  raw string, not JSON  binaryFrameDispatcher.ts:127-130
      settingsUpdated  category,updatedBy,timestamp,settings? textMessageHandler.ts:139-141,155-192
    }
    ClientToServer --> ServerToClient : request/response over one socket
    note for ClientToServer "sent via store websocket/index.ts sendMessage(type,data) JSON.stringify(type,data)<br/>index.ts:301-330. Queued to messageQueue (MAX_QUEUE_SIZE=100) when not connected,<br/>connectionManager.ts:27,149-168."
    note for ServerToClient "dispatched by handleTextMessage after JSON.parse and validateMessage(type is<br/>nonempty string, length less-or-equal 100)<br/>websocket/binaryFrameDispatcher.ts:86-94,textMessageHandler.ts:81-144."
    note for ServerToClient "RESOLVED ADR-2080 settingsUpdated now HAS a consumer. The server emitted<br/>it from src/settings/api/settings_routes.rs:455,1148 while the only client knowledge was<br/>validation.ts:274 case settings_update (snake_case, different shape) - every broadcast fell<br/>through to the default arm and did nothing, so two viewers saw different physics until one<br/>reloaded. nodeFilter applies the supplied settings directly. other categories re-read via<br/>getSectionPaths plus getSettingsByPaths. own-write echo and stale timestamps are dropped.<br/>Contract fixed by vc-core ADR-2047. settings_update is dead on the wire and left in place.<br/>The visual bucket now broadcasts too (settings_routes.rs:1546), and the client re-reads<br/>every visualisation.* path outside rendering and physics plus<br/>VISUAL_ONLY_SECTION_PATHS (settingsHelpers.ts:254-262), cache invalidated first (VC-32.17)"
    note for ServerToClient "GRAPH_UPDATED shrink-guard: a smaller initialGraphLoad is discarded unless<br/>isFilterResponseExpected() is true within the 15000ms window, armed only for an<br/>enabled filter that opted in to serverRemove, and a REST load in flight defers<br/>the whole payload - see VC-32.9. textMessageHandler.ts:235-270,filterSync.ts:29-42,127-131"
```
## VC-32.16 solidWebSocket: JSS resource-notification boundary
```mermaid
sequenceDiagram
    autonumber
    participant C as store<br/>websocket/index.ts:448
    participant SW as solidWebSocket store adapter<br/>websocket/solidWebSocket.ts:71
    participant PN as podNotificationManager (singleton)<br/>services/solidPod/podNotifications.ts:300
    participant JSS as JSS WebSocket<br/>VITE_JSS_WS_URL

    C->>SW: connectSolid() index.ts:448-450
    SW->>SW: bindLifecycle(set) once, never torn down solidWebSocket.ts:30-34
    alt podNotificationManager.isConnected solidWebSocket.ts:74
        SW->>SW: re-sync mirror from the socket the other consumer opened solidWebSocket.ts:77
    else
        SW->>PN: connect() solidWebSocket.ts:81
        alt VITE_JSS_WS_URL unset or not ws:/wss: podNotifications.ts:70,82
            PN->>PN: logger.warn/error, return podNotifications.ts:71-72,83-84
        else
            PN->>JSS: new WebSocket(validatedUrl.href) podNotifications.ts:87
            JSS-->>PN: onopen -> webSocketRegistry.register("solid-pod") podNotifications.ts:89-95
            PN-->>SW: onLifecycle{type:"open"} -> isSolidConnected=true, emit(solid-connected) solidWebSocket.ts:38-41
            JSS-->>PN: onmessage(msg) -> handleMessage podNotifications.ts:97-101,225
            alt msg starts with "protocol " podNotifications.ts:226
                PN->>JSS: resubscribe send("sub url") for every tracked URL podNotifications.ts:229-231
                PN-->>SW: onLifecycle{protocol} -> emit(solid-protocol) solidWebSocket.ts:52-54
            else msg starts with "ack " podNotifications.ts:233
                PN->>PN: notifySubscribers(url,{type:"ack"}) podNotifications.ts:236
            else msg starts with "pub " podNotifications.ts:237
                PN->>PN: notifySubscribers(url,{type:"pub"}) + container URL podNotifications.ts:240,251-259
                PN-->>SW: onLifecycle{pub} -> emit(solid-resource-changed) solidWebSocket.ts:55-57
            else msg starts with "error " podNotifications.ts:242
                PN-->>SW: onLifecycle{server-error} -> emit(solid-error) solidWebSocket.ts:49-51
            end
            JSS-->>PN: onclose -> unregister, emit connection:close podNotifications.ts:109-117
            PN-->>SW: onLifecycle{close} -> isSolidConnected=false, solidSocket=null solidWebSocket.ts:42-45
            PN->>PN: handleReconnect - SOLID_RECONNECT_DELAY_MS * 2^attempt podNotifications.ts:275-289
        end
    end
    Note over SW,PN: INVARIANT: ADR-2100 - ONE socket for VITE_JSS_WS_URL. solidWebSocket.ts no longer<br/>constructs a WebSocket, keeps no subscription wire and owns no backoff ladder - it is a thin<br/>store adapter over podNotificationManager, registered once as "solid-pod". The retired<br/>"solid-store" registry entry and its 10-attempt ladder are gone - see VC-30.5, VC-26.9
    Note over SW: subscribeSolidResource delegates to podNotificationManager.subscribe (sub on the first<br/>callback for a URL) and mirrors into state.solidSubscriptions for getSolidSubscriptions()<br/>only, so a callback fires exactly once solidWebSocket.ts:107-140, podNotifications.ts:130-149
    Note over JSS: see VC-33 for the JSS/Solid Pod ownership boundary this socket crosses into
```
## VC-32.17 Settings broadcast: the shared emitter, the nodeFilter payload, and the peer re-read
```mermaid
sequenceDiagram
    autonumber
    participant W as writer session
    participant RT as settings_routes<br/>src/settings/api/settings_routes.rs:453
    participant CC as ClientCoordinator<br/>BroadcastMessage
    participant TH as handleSettingsUpdated<br/>textMessageHandler.ts:139
    participant API as settingsApi
    W->>RT: PUT rendering, row persisted before the actor update src/settings/api/settings_routes.rs:1005-1017
    RT->>CC: broadcast_settings_change(rendering) src/settings/api/settings_routes.rs:1030
    W->>RT: PUT visual, deep_merge_json into the stored document src/settings/api/settings_routes.rs:1519-1525
    RT->>CC: broadcast_settings_change(visual) src/settings/api/settings_routes.rs:1546
    W->>RT: PUT node-filter
    RT->>CC: its own payload with settings, now minConnections and minMaturity src/settings/api/settings_routes.rs:1147-1165
    CC-->>TH: settingsUpdated category,updatedBy,timestamp
    TH->>TH: drop own echo and stale timestamps textMessageHandler.ts:163-178
    alt category nodeFilter
        TH->>TH: set nodeFilter to the payload whole textMessageHandler.ts:181-184
    else any other category
        TH->>TH: getSectionPaths(category) textMessageHandler.ts:196
        Note over TH: visual adds VISUAL_ONLY_SECTION_PATHS, embeddingCloud, the agent node colours and agentColors (settingsHelpers.ts:161-165, settingsHelpers.ts:254-262)
        TH->>API: invalidateSettingsCache() then getSettingsByPaths(paths) textMessageHandler.ts:203-204
        API-->>TH: fresh values, set with skipSave textMessageHandler.ts:205-212
    end
    Note over RT: DEBT: the doc comment calls broadcast_settings_change the single emitter (src/settings/api/settings_routes.rs:450),<br/>yet nodeFilter still builds and sends its own payload (src/settings/api/settings_routes.rs:1147-1171)
```
**Debt (loose end):** `PUT /api/settings/visual` only deep-merges (`src/settings/api/settings_routes.rs:1426-1437`, called at `src/settings/api/settings_routes.rs:1525`), so a key can be added or overwritten but never removed; retired visual keys stay in the stored document for good.

**Open:** `filter_update` carries no `minConnections` or `minMaturity` (`client/src/store/websocket/index.ts:372-381`) and `ClientFilter` has no such fields (`src/actors/client_coordinator_actor.rs:111-125`), so the degree and maturity floors only grey on the client even when `serverRemove` is on; is that intended?
