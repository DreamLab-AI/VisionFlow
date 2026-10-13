---
id: VC-32
title: Client WebSocket transport and binary position protocol
area: visionclaw
governing:
  - ../project/docs/PROTOCOL-registry.md
  - ../project/docs/BASELINE-architecture.md
adrs: [ADR-2002, ADR-2019, ADR-2020, ADR-2047, ADR-2057, ADR-2078, ADR-2080]
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
  - ../project/client/src/services/binaryProtocol/backpressure.ts
  - ../project/client/src/services/binaryProtocol/ssspVoice.ts
  - ../project/client/src/services/nostrAuthService.ts
  - ../project/client/src/types/binaryProtocol.ts
  - ../project/client/src/features/graph/hooks/useGraphEventHandlers.ts
  - ../project/client/src/app/AppInitializer.tsx
  - ../project/client/src/store/transientBeamStore.ts
  - ../project/xr-client/rust/src/binary_protocol.rs
  - ../project/src/utils/binary_protocol.rs
  - ../project/docs/IDENTIFIER-taxonomy.md
  - ../project/client/src/utils/validation.ts
  - ../project/src/handlers/socket_flow_handler/http_handler.rs
  - ../project/src/settings/api/settings_routes.rs
  - ../project/client/src/services/signedWebSocketProtocols.ts
  - ../project/client/src/features/graph/managers/graphDataManager.ts
  - ../project/src/utils/nip98.rs
  - ../project/src/handlers/socket_flow_handler/heartbeat.rs
  - ../project/src/handlers/socket_flow_handler/transport.rs
  - ../project/src/handlers/socket_flow_handler/mod.rs
  - ../project/src/handlers/socket_flow_handler/types.rs
  - ../project/src/handlers/socket_flow_handler/message_routing.rs
  - ../project/src/handlers/socket_flow_handler/position_updates.rs
verified_commit: {visionclaw: 06ce9fe75497ec73985b6d684ba6e94b9be5f4a5}
---
## VC-32.1 Connect: signed-subprotocol upgrade, dev-mode authenticate message
```mermaid
sequenceDiagram
    autonumber
    participant C as connect<br/>websocket/index.ts:154
    participant SP as signedWebSocketProtocols<br/>signedWebSocketProtocols.ts:7
    participant WS as WebSocket
    participant AH as sendAuthOnConnect<br/>connectionManager.ts:372
    participant S as socket_flow_handler<br/>http_handler.rs:53

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
    S->>S: websocket_auth_header decodes the single nostr. token, two tokens or over 16384 chars is refused http_handler.rs:155-158, nip98.rs:639-655
    alt signed subprotocol present
        S->>S: verify_nip98_auth(header, signed_upgrade_url, GET) single-use http_handler.rs:160-173
        Note right of S: the signed url is connection_info scheme, host and uri http_handler.rs:42-50
        alt invalid or replayed
            S-->>WS: 401 Invalid or replayed WebSocket signature http_handler.rs:167-170
        else verified
            Note over S: signed_user = Some(user) http_handler.rs:166
        end
    else no signed subprotocol, Authorization Bearer token
        S->>S: token = Authorization Bearer header http_handler.rs:182-188
        S->>S: get_session(token), a valid session becomes signed_user http_handler.rs:226, http_handler.rs:262-266
        alt token missing or session invalid in a release build
            S-->>WS: 401 http_handler.rs:254-255, http_handler.rs:321-326
        end
    end
    S->>S: ws_server.pubkey and is_power_user set from signed_user http_handler.rs:416-419
    S-->>WS: 101 Switching Protocols, only visionclaw and permessage-deflate are echoed http_handler.rs:8, http_handler.rs:451-454
    WS->>C: onopen index.ts:172
    C->>C: webSocketRegistry.register(graph,url,socket) index.ts:178
    alt signedUpgrade
        Note over AH: sendAuthOnConnect is skipped, identity was fixed at the upgrade index.ts:185
    else dev mode
        C->>AH: sendAuthOnConnect(socket,url) index.ts:185
        AH->>S: authenticate token=dev-session-token pubkey ephemeral connectionManager.ts:375-382
    end
    opt nodeFilter settings present
        C->>S: filter_update, forced once per connect and de-duplicated afterwards index.ts:189, filterSync.ts:108-116
    end
    C->>C: initializeBatchQueue(get) index.ts:191
    C->>C: setupFilterSubscription(get) index.ts:192
    C->>C: notifyConnectionStatusHandlers(true) index.ts:193
    C->>C: startHeartbeat(get) index.ts:194
    C->>C: processMessageQueue(get,set) index.ts:195
    Note over S: DRIFT: PROTOCOL-registry.md:260-263 says the Authorization header is now the only accepted carrier in a release build.<br/>http_handler.rs:155-173 also accepts the signed Sec-WebSocket-Protocol NIP-98 subprotocol, and that is the carrier this client<br/>uses in every non-dev build (index.ts:165-169). ADR-2058 removed only the ?token= query fallback, which is compiled out of release<br/>(http_handler.rs:190-218). BASELINE-architecture.md:248 still lists the query token as an open divergence, written before ADR-2058 landed.
    Note over S: RESOLVED 0d636fd0e (S-7f, accounted for in VC-03.2): a validated Authorization-header session is now carried into signed_user<br/>(http_handler.rs:262-266), so pubkey and is_power_user are set for a header-token client exactly as for the signed-subprotocol client<br/>(http_handler.rs:416-419). Before it the session was checked then dropped and that client stayed anonymous
    Note over AH: DEBT: the NIP-98 authenticate event branch of sendAuthOnConnect (connectionManager.ts:383-393) is unreachable from its only caller.<br/>index.ts:185 calls it only when signedUpgrade is false, which is exactly when isDevMode() is true, so the dev branch (connectionManager.ts:375)<br/>always wins. The server still handles authenticate (message_routing.rs:64-66), the client just no longer sends the event form
    Note over S: INVARIANT: the credential subprotocol is never echoed back, only PUBLIC_WS_PROTOCOLS (http_handler.rs:8, http_handler.rs:451-454)
```
## VC-32.2 Reconnect backoff and heartbeat / binary-silence watchdog
```mermaid
sequenceDiagram
    autonumber
    participant WS as WebSocket
    participant C as store<br/>websocket/index.ts:211
    participant HB as heartbeat<br/>connectionManager.ts:226
    participant SRV as SocketFlowServer<br/>types.rs:492

    WS-->>C: onclose(event) index.ts:211
    alt event.code in 1000,1001 and event.wasClean index.ts:230-233
        C->>C: updateConnectionState(disconnected) index.ts:237
    else abnormal closure
        C->>C: updateConnectionState(reconnecting,event.reason) index.ts:234
        C->>C: attemptReconnect(get,set,updateConnectionState) index.ts:235, connectionManager.ts:334
        loop while reconnectAttempts less than maxReconnectAttempts(10) connectionManager.ts:345
            Note over C: delay = min(1000 times 2 pow attempts + jitter(0-500ms), MAX_RECONNECT_DELAY=30000ms) connectionManager.ts:348-350
            C->>WS: setTimeout(get().connect, delay) connectionManager.ts:358
        end
        Note over C: reconnectAttempts exhausted -> updateConnectionState(failed) connectionManager.ts:365-366
    end
    Note over HB: HEARTBEAT_INTERVAL_MS=30000, HEARTBEAT_TIMEOUT_MS=45000, BINARY_SILENCE_PROBE_MS=60000 connectionManager.ts:25-26,216
    loop every 30000ms after startHeartbeat(get) connectionManager.ts:226-236
        HB->>SRV: socket.send(ping) as a plain text frame connectionManager.ts:271
        SRV-->>HB: text frame pong, message_routing.rs:18-21
        HB->>HB: pong -> handleHeartbeatResponse() binaryFrameDispatcher.ts:123-126, connectionManager.ts:311
        opt no pong within 45000ms
            HB->>WS: socket.close(4000, Heartbeat timeout) connectionManager.ts:328
        end
        opt Date.now() - lastInboundBinaryAt greater than 60000ms connectionManager.ts:279
            alt unansweredProbes greater or equal 2 connectionManager.ts:280
                HB->>WS: recycle connection via handleHeartbeatTimeout connectionManager.ts:283
            else
                HB->>SRV: send subscribe_position_updates interval=200 binary=true connectionManager.ts:288-291
                HB->>SRV: send request_full_snapshot connectionManager.ts:299
            end
        end
    end
    Note over C: noteInboundBinary() resets lastInboundBinaryAt and unansweredProbes on every inbound binary frame connectionManager.ts:221-224
    Note over C: INVARIANT. binary silence alone never forces reconnect - only unanswered probes do (connectionManager.ts:208-215)
    Note over SRV: server side, a second watchdog the client does not see. Since 8d17f1800 the server pings every system.websocket.heartbeatInterval<br/>(types.rs:543-554) and closes with code 1001 once no inbound frame of any kind has arrived for heartbeatTimeout (heartbeat.rs:84-90, types.rs:555-565).<br/>Text and binary frames count as inbound (mod.rs:28-36), so the client text ping above keeps the socket alive. Defaults are 10 s and 60 s (heartbeat.rs:106-112),<br/>a timeout under two intervals is raised to two (heartbeat.rs:35-39)
    Note over SRV: the probe's interval=200 is dead weight. handle_subscribe_position_updates treats a subscribe as ONE snapshot and ignores any interval<br/>(position_updates.rs:601-606), and a second subscribe inside 2 s is dropped (position_updates.rs:616-621), so the 30 s probe cadence clears that guard.<br/>The request_full_snapshot that follows is the frame that actually answers the probe (message_routing.rs:35-37)
    Note over SRV: a connected peer that stops reading is a third case. The server evicts it after the same heartbeat timeout of congestion and closes the socket<br/>from outside the session actor through a dup of the connection fd (types.rs:498-515, transport.rs:10-15), since a blocked write<br/>stops the actor, its heartbeat timer included. A merely full mailbox no longer evicts, it skips one frame (d90064950, VC-13.3), which the client<br/>sees as binary silence and answers with the probes above
```
## VC-32.3 Binary frame demux: onmessage tag dispatch and single-flight discipline
```mermaid
sequenceDiagram
    autonumber
    participant WS as WebSocket
    participant MH as createMessageHandler<br/>binaryFrameDispatcher.ts:97
    participant VB as validateBinaryData<br/>store/websocket/binaryProtocol.ts:187
    participant SF as BinaryFrameDispatcher<br/>binaryFrameDispatcher.ts:43
    participant PB as processBinaryData<br/>store/websocket/binaryProtocol.ts:462

    WS->>MH: onmessage(event) binaryFrameDispatcher.ts:120
    alt event.data equals pong
        MH->>MH: handleHeartbeatResponse() connectionManager.ts:311
    else event.data instanceof Blob
        MH->>MH: noteInboundBinary() binaryFrameDispatcher.ts:133
        MH->>VB: arrayBuffer().then(validateBinaryData) binaryFrameDispatcher.ts:136-141
    else event.data instanceof ArrayBuffer
        MH->>MH: noteInboundBinary() binaryFrameDispatcher.ts:152
        MH->>VB: validateBinaryData(event.data) binaryFrameDispatcher.ts:154
    else text frame
        MH->>MH: JSON.parse then handleTextMessage(message) binaryFrameDispatcher.ts:169-176
    end
    VB->>VB: read firstByte = view.getUint8(0) store/websocket/binaryProtocol.ts:197
    alt firstByte is 3(V3), 5(V5) or 0x23(AGENT_ACTION) store/websocket/binaryProtocol.ts:201
        VB-->>SF: dispatcher.handle(buffer) binaryFrameDispatcher.ts:138,155
    else unknown lead byte
        VB-->>MH: reject, log warn, drop frame store/websocket/binaryProtocol.ts:201, :202
    end
    alt inFlight already processing binaryFrameDispatcher.ts:52
        SF->>SF: pendingLatest = buffer, dropCount++ (newest-wins) binaryFrameDispatcher.ts:53-60
    else idle
        SF->>PB: processBinaryData(buffer,get,set) binaryFrameDispatcher.ts:64
    end
    PB->>PB: firstByte switch: PROTOCOL_V3/V5 only -> handleLegacyBinaryData store/websocket/binaryProtocol.ts:475, :476
    Note over PB: RESOLVED ADR-2078: PROTOCOL_V2 removed from the accept list. The server<br/>rejects V2 outright (src/utils/binary_protocol.rs:610) so a 0x02 lead byte can<br/>only be a stale sender or a mis-routed payload - it is no longer routed here
    PB->>PB: firstByte MessageType.AGENT_ACTION(0x23) -> handleAgentActionTagged store/websocket/binaryProtocol.ts:482, :483
    PB->>PB: else parseHeader(data) then switch(header.type) store/websocket/binaryProtocol.ts:488, :494
    PB-->>SF: promise settles -> finally drains pendingLatest via queueMicrotask binaryFrameDispatcher.ts:68-76
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
    note for FramedMessageHeader "createMessage/parseHeader BinaryWebSocketProtocol.ts:73-116. Header size<br/>MESSAGE_HEADER_SIZE=6, GRAPH_UPDATE_HEADER_SIZE=7 (frameTypes.ts:224-225). Used<br/>only for client-to-server encode calls: encodePositionUpdates, encodeAgentState,<br/>encodeSSSPData, encodeControlBits, encodeVoiceChunk, createBroadcastAck,<br/>encodeAgentAction (BinaryWebSocketProtocol.ts:129-290)."
    note for MessageType "RESOLVED ADR-2078. The framed header's version byte is written from<br/>PROTOCOL_VERSION (BinaryWebSocketProtocol.ts:86), which is PROTOCOL_V3=3 (frameTypes.ts:30), and the<br/>comment above it now says so (BinaryWebSocketProtocol.ts:81-84). The old header-local PROTOCOL_V4=4 is<br/>deleted from frameTypes.ts (frameTypes.ts:33-37) so PROTOCOL_V4 now names only the delta node<br/>encoding of VC-32.5, and SUPPORTED_HEADER_VERSIONS is V3 and V5 (frameTypes.ts:38). ADR-2078 also<br/>deleted five enum members with no reference anywhere in client/src: VELOCITY_UPDATE 0x12,<br/>AGENT_STATE_DELTA 0x21, AGENT_HEALTH 0x22, HANDSHAKE 0x32 and HEARTBEAT 0x33 (frameTypes.ts:58-62)."
    note for MessageType "The enum is frameTypes.ts:64-100. The server's own MessageType has only five tags and this client decodes<br/>just AGENT_ACTION and BROADCAST_ACK from the wire, every other member is client-outbound<br/>only and no evidence the server emits it (frameTypes.ts:44-56). Live position frames are unframed and<br/>lead with a bare 0x03 or 0x05 byte, so they never carry a MessageType at all."
    note for FramedMessageHeader "DOC-DRIFT. docs/PROTOCOL-registry.md frame tag registry lists only<br/>0x03,0x05,0x23,0x43,0x44 as live server-to-client tags; this client-only outbound<br/>envelope and its MessageType space (0x01,0x02,0x10-0x54) do not appear in that<br/>registry at all."
    note for MessageType "BROADCAST_ACK 0x34 payload (encodeBroadcastAckPayload/decodeBroadcastAck<br/>backpressure.ts:18-69): BROADCAST_ACK_PAYLOAD_SIZE=20 bytes - sequenceId split<br/>low/high uint32 offset0/4, nodesReceived uint32 offset8, timestamp split<br/>low/high uint32 offset12/16, all littleEndian to avoid BigInt64<br/>(backpressure.ts:26-29,36-39). Ack cadence (ACK_BATCH_SIZE=10) is in VC-32.8."
    note for MessageType "SSSP_DATA 0x31 and VOICE_CHUNK 0x40 payloads (ssspVoice.ts:8-91).<br/>SSSPDataRecord: uint32 nodeId, float32 distance, uint32 parentId, uint16 flags<br/>= 14 bytes (SSSP_DATA_SIZE_V2 frameTypes.ts:229, layout ssspVoice.ts:8-21). VoiceChunkHeader: uint16<br/>agentId, uint16 chunkId, uint8 format, uint16 dataLength = 7 bytes<br/>(VOICE_HEADER_SIZE frameTypes.ts:236) then raw audioData (ssspVoice.ts:49-65). Voice agentId is<br/>uint16 (max 65535) - a wire format separate from the uint32 flag-bit node/agent<br/>id of VC-32.6, no alignment issue (ssspVoice.ts:54-56,82-84)."
```
## VC-32.5 Server-to-client position wire records: V3, V4 delta, V5 envelope (V2 declined)
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
    class V4DeltaFrame {
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
    V4DeltaFrame --> DeltaItem : contains
    note for V3NodeRecord "CROSS-CHECK OK. client/src/types/binaryProtocol.ts:94-104 offsets match<br/>xr-client/rust/src/binary_protocol.rs:8-11,750-771 exactly: id, pos, vel,<br/>sssp_distance, sssp_parent, cluster_id, anomaly, community_id, centrality. Godot<br/>NODE_RECORD_BYTES=52 (:32) equals client BINARY_NODE_SIZE_V3=52<br/>(types/binaryProtocol.ts:89). No DIVERGENCE."
    note for V2NodeRecord "Declined, not live. The server answers a 0x02 lead byte with Err (src/utils/binary_protocol.rs:610),<br/>its 36-byte stride constant is deleted, and the client declares no 36-byte stride at all (ADR-2057,<br/>types/binaryProtocol.ts:86-89). The shape is kept here only to show what the 52-byte V3 record replaced."
    note for V5Envelope "types/binaryProtocol.ts:410-431 parseV5Nodes reads seqLow at offset1 and seqHigh<br/>at offset5, reconstructs a V3 buffer at offset9. Matches Godot PROTOCOL_V5=0x05<br/>and V5_SEQ_BYTES=8, docs/PROTOCOL-registry.md 0x05 row<br/>(xr-client/rust/src/binary_protocol.rs:29-30)."
    note for V4DeltaFrame "DOC-DRIFT. docs/PROTOCOL-registry.md frame tag registry (0x03,0x05,0x23,0x43,0x44)<br/>has no 0x04 row; this delta codec exists only in client decode<br/>(types/binaryProtocol.ts:56-65,327-397) with no confirmed live sender in the<br/>registry. PROTOCOL_V4 here (delta node encoding) is a DIFFERENT meaning from the<br/>same-named PROTOCOL_V4 in services/binaryProtocol/frameTypes.ts:14 (framed-header<br/>version) - see VC-32.4."
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
    note for WireNodeId "INVARIANT, no DIVERGENCE. Client client/src/types/binaryProtocol.ts:106-117, Godot<br/>decoder xr-client/rust/src/binary_protocol.rs:41-47, and<br/>docs/IDENTIFIER-taxonomy.md:124-129 all agree exactly on mask and flag hex values.<br/>getActualNodeId(nodeId) is nodeId AND NODE_ID_MASK<br/>(types/binaryProtocol.ts:143-145)."
    note for WireNodeId "docs/IDENTIFIER-taxonomy.md:131-141: ids are sequential u32 from a NEXT_NODE_ID<br/>atomic counter. Every encoder branch, the untyped one included, routes through<br/>enforce_wire_id_bounds (src/utils/binary_protocol.rs:185-206): debug_assert fails fast in<br/>development (:186-193) but release does NOT truncate silently. remap_wire_id masks the id into<br/>26 bits and an error! log names the class and both ids (:194-204, ADR-2024). The masked id<br/>still aliases another node, so an over-range id is loud, not harmless."
    note for WireNodeId "INVARIANT (2026-09-08, 6c47b9833): an id that ALREADY carries a recognised<br/>class pattern is forwarded unchanged by the encoder's last branch, because<br/>position_updates.rs and actor_messages.rs stamp the flag themselves and pass<br/>empty class sets. Running the 26-bit bound check on a stamped id panicked<br/>every broadcast in debug builds - each XR client saw a 2 s connection-reset<br/>loop - and would have stripped the class flag in release<br/>(src/utils/binary_protocol.rs:475-484). Only a high bit that is NOT a class<br/>pattern is still an overflow, handled by the untyped branch (src/utils/binary_protocol.rs:490)."
```
## VC-32.7 getNodeType(): flag precedence decision order
```mermaid
flowchart TD
    A["nodeId: uint32"] --> B{"nodeId AND AGENT_NODE_FLAG != 0?<br/>types/binaryProtocol.ts:129"}
    B -- yes --> AG["NodeType.Agent"]
    B -- no --> C{"nodeId AND KNOWLEDGE_NODE_FLAG != 0?<br/>types/binaryProtocol.ts:131"}
    C -- yes --> KN["NodeType.Knowledge"]
    C -- no --> D{"nodeId AND ONTOLOGY_TYPE_MASK == CLASS_FLAG?<br/>types/binaryProtocol.ts:133"}
    D -- yes --> OC["NodeType.OntologyClass"]
    D -- no --> E{"nodeId AND ONTOLOGY_TYPE_MASK == INDIVIDUAL_FLAG?<br/>types/binaryProtocol.ts:135"}
    E -- yes --> OI["NodeType.OntologyIndividual"]
    E -- no --> F{"nodeId AND ONTOLOGY_TYPE_MASK == PROPERTY_FLAG?<br/>types/binaryProtocol.ts:137"}
    F -- yes --> OP["NodeType.OntologyProperty"]
    F -- no --> UN["NodeType.Unknown  types/binaryProtocol.ts:140"]
    AG --> G["updateNodeTypeMapFromParsed caches actualId to type<br/>evicting oldest past MAX_NODE_TYPE_ENTRIES=65536<br/>websocket/binaryProtocol.ts:155, :172"]
    KN --> G
    OC --> G
    OI --> G
    OP --> G
```
## VC-32.8 Live position frame processing: parse, analytics ingest, ack
```mermaid
sequenceDiagram
    autonumber
    participant PB as processBinaryData<br/>store/websocket/binaryProtocol.ts:462
    participant FR as parseBinaryFrameData<br/>types/binaryProtocol.ts:440
    participant AN as nodeAnalyticsStore<br/>features/analytics/store/nodeAnalyticsStore
    participant GDM as graphDataManager<br/>features/graph/managers/graphDataManager
    participant SRV as Server /wss

    PB->>PB: handleLegacyBinaryData(data,get,set) store/websocket/binaryProtocol.ts:361
    PB->>FR: parseBinaryFrameData(data) store/websocket/binaryProtocol.ts:370
    FR-->>PB: ParsedBinaryFrame type=full|delta, nodes, broadcastSequence
    opt frame.type equals full store/websocket/binaryProtocol.ts:381
        alt protoByte is PROTOCOL_V3 or PROTOCOL_V5 store/websocket/binaryProtocol.ts:382
            PB->>AN: nodeAnalyticsStore.ingest(parsedNodes) store/websocket/binaryProtocol.ts:384
        else V4 delta frame
            Note over PB: V4 delta omits cluster_id/anomaly/community offsets 36/40/44 - must NOT overwrite live analytics store/websocket/binaryProtocol.ts:378, :380
        end
    end
    PB->>PB: updateNodeTypeMapFromParsed(parsedNodes,set) store/websocket/binaryProtocol.ts:388
    opt hasBotsData: any node isAgentNode() store/websocket/binaryProtocol.ts:390
        PB->>PB: emit(bots-position-update, data) store/websocket/binaryProtocol.ts:392, :393
    end
    PB->>GDM: graphDataManager.updateNodePositions(data) store/websocket/binaryProtocol.ts:407
    PB->>PB: positionUpdateSequence++ store/websocket/binaryProtocol.ts:415
    opt positionUpdateSequence minus lastAckSentSequence greater-or-equal ACK_BATCH_SIZE(10) store/websocket/binaryProtocol.ts:38, :417
        PB->>SRV: sendPositionAck -> BroadcastAck message (createBroadcastAck), carrying the V5 frame.broadcastSequence when present else the local counter store/websocket/binaryProtocol.ts:416, :131, :142, :418
    end
    Note over PB: framed-header path handlePositionUpdate (MessageType.POSITION_UPDATE/AGENT_POSITIONS, store/websocket/binaryProtocol.ts:307, :354, :503) mirrors this same analytics-cache/ack flow but is reached only via the 6-byte outbound envelope of VC-32.4, not the bare V3/V5 tag the live server actually sends.
```
## VC-32.9 Position subscription: idempotent one-shot subscribe, de-duplicated filter sync, REST-aware shrink-guard
```mermaid
sequenceDiagram
    autonumber
    participant AI as AppInitializer<br/>app/AppInitializer.tsx:269
    participant C as store<br/>websocket/index.ts
    participant FS as filterSync<br/>websocket/filterSync.ts:132
    participant TH as textMessageHandler<br/>websocket/textMessageHandler.ts:226
    participant GDM as graphDataManager<br/>graphDataManager.ts:258
    participant SRV as Server /wss

    C->>AI: onConnectionStatusChange(connected=true) AppInitializer.tsx:269,282
    alt websocketService.isReady() AppInitializer.tsx:284
        alt not hasSubscribedToPositions AppInitializer.tsx:290
            AI->>SRV: subscribe_position_updates binary=true, no interval field AppInitializer.tsx:293-295
            Note right of SRV: a subscribe yields ONE full snapshot and any interval is ignored, the stream itself is the<br/>coordinator push at physics.broadcastFps (position_updates.rs:601-606).<br/>A second subscribe inside 2 s is dropped (position_updates.rs:616-621)
        else already subscribed on this connection
            Note over AI: idempotency guard skips duplicate subscribe AppInitializer.tsx:65,290
        end
    else not fully established yet
        AI->>C: onMessage waiting for connection_established AppInitializer.tsx:309
        C-->>AI: message.type equals connection_established AppInitializer.tsx:310
        alt not hasSubscribedToPositions AppInitializer.tsx:315
            AI->>SRV: subscribe_position_updates binary=true AppInitializer.tsx:319
        end
    end
    Note over C: !connected clears hasSubscribedToPositions so the next connection resubscribes exactly once AppInitializer.tsx:277-278
    C->>FS: syncFilterToServer(get, force) once per open index.ts:189
    FS->>FS: return if not connected or no nodeFilter snapshot filterSync.ts:109-112
    FS->>FS: unforced calls return when the snapshot equals lastFilterSnapshot filterSync.ts:113-114
    FS->>C: sendFilterUpdate(snapshot) filterSync.ts:115
    C->>FS: expectFilterResponseFor(filter) index.ts:370
    alt filter.enabled
        FS->>FS: expectFilterResponse() arms the 15000ms acceptance window filterSync.ts:124-128, filterSync.ts:29-34
    else filter disabled
        Note over FS: window not armed, the server answers a disabled filter with the full graph so nothing smaller may be accepted filterSync.ts:118-123
    end
    C->>SRV: filter_update message index.ts:372-380
    C->>FS: setupFilterSubscription(get) index.ts:192
    FS->>FS: ONE whole-store useSettingsStore.subscribe, replacing seven per-path subscriptions filterSync.ts:136-143
    loop on every settings-store change while isConnected
        FS->>FS: syncFilterToServer(get) unforced, only a real filter-field change reaches the wire filterSync.ts:141, filterSync.ts:90-101
    end
    SRV-->>TH: initialGraphLoad nodes,edges textMessageHandler.ts:116-118
    TH->>GDM: isRestLoadInFlight() textMessageHandler.ts:235, graphDataManager.ts:248-250
    alt a REST graph/data load is in flight
        TH->>TH: isFilterResponse = isFilterResponseExpected(), then clear the window textMessageHandler.ts:237-239
    else no REST load in flight
        alt nodes.length less than existingNodeCount textMessageHandler.ts:241-242
            alt isFilterResponseExpected() textMessageHandler.ts:249
                TH->>TH: accept the filtered load, clearFilterResponseExpectation() textMessageHandler.ts:250-255
            else smaller unsolicited payload
                TH->>TH: skip setGraphData, positions arrive via binary stream textMessageHandler.ts:256-266
            end
        end
    end
    TH->>GDM: offerServerGraphLoad(data, isFilterResponse) textMessageHandler.ts:316-318
    alt REST load in flight
        GDM-->>TH: deferred, held latest-wins graphDataManager.ts:262-268
        Note right of GDM: when REST settles the held load is DROPPED on success unless it answers our own filter_update,<br/>and applied on REST failure graphDataManager.ts:274-287
    else
        GDM->>GDM: setGraphData(data) graphDataManager.ts:270
        TH->>TH: emit graphDataUpdated source websocket_filter textMessageHandler.ts:320-326
    end
    Note over TH: the connect-time initialGraphLoad is capped at 3,000 nodes, which is why a smaller payload needs the guard textMessageHandler.ts:243-248
```
## VC-32.10 Drag / pin send path: server-authoritative pin over JSON control frames
```mermaid
sequenceDiagram
    autonumber
    participant U as PointerEvent
    participant EH as useGraphEventHandlers<br/>features/graph/hooks/useGraphEventHandlers.ts:78
    participant WSS as webSocketService<br/>graphDataManager.webSocketService
    participant SRV as Server /wss

    U->>EH: handlePointerDown(event) useGraphEventHandlers.ts:78
    EH->>EH: dragDataRef.current = pointerDown,nodeId,startNodePos3D useGraphEventHandlers.ts:95-104
    EH->>EH: startInteraction(node.id) useGraphEventHandlers.ts:120
    U->>EH: handlePointerMove(event) useGraphEventHandlers.ts:127
    alt distance greater than DRAG_THRESHOLD(5) useGraphEventHandlers.ts:136
        EH->>EH: drag.isDragging = true useGraphEventHandlers.ts:137
        alt graphDataManager.webSocketService.isReady() useGraphEventHandlers.ts:149
            EH->>SRV: sendMessage(nodeDragStart, nodeId, startPosition) useGraphEventHandlers.ts:150-157
        else not connected or not server-ready
            Note over EH: send skipped, caught by try/catch - drag state machine still proceeds useGraphEventHandlers.ts:148,159-161
        end
    end
    opt drag.isDragging useGraphEventHandlers.ts:170
        EH->>EH: updateNodePosition(nodeId,intersection) local optimistic move useGraphEventHandlers.ts:207-211
        EH->>WSS: throttledWebSocketUpdate(nodeId,position) throttle 100ms useGraphEventHandlers.ts:17,54-76,214-218
        alt shouldSendPositionUpdates() and webSocketService.isReady() useGraphEventHandlers.ts:57-59
            WSS->>SRV: sendMessage(nodeDragUpdate, nodeId, position, timestamp) useGraphEventHandlers.ts:64-68
        end
    end
    U->>EH: handlePointerUp(event) useGraphEventHandlers.ts:223
    alt drag.isDragging useGraphEventHandlers.ts:241
        EH->>EH: flushPositionUpdates() useGraphEventHandlers.ts:251
        alt webSocketService.isReady() useGraphEventHandlers.ts:254
            EH->>SRV: sendMessage(nodeDragEnd, nodeId) useGraphEventHandlers.ts:255-257
        end
    else click, not a drag
        EH->>EH: onNodeSelect(nodeId) useGraphEventHandlers.ts:265-268
    end
    Note over SRV: ADR-03 D7 (legacy). pin state is server-authoritative - nodeDragStart pins, nodeDragEnd unpins and runs final settle. the old binary sendNodePositionUpdates second-frame path was removed 2026-06-03 useGraphEventHandlers.ts:60-63,142-144,246-247
```
## VC-32.11 Agent frame 0x23 (AGENT_ACTION): decode and beam dispatch
```mermaid
sequenceDiagram
    autonumber
    participant PB as processBinaryData<br/>store/websocket/binaryProtocol.ts:462
    participant DEC as decodeAgentActions<br/>binaryProtocol/agentMessages.ts:183
    participant BEAM as pushTransientBeams<br/>store/transientBeamStore
    participant AT as useAgentTargetStore.pushActions<br/>client/src/store/agentTargetStore.ts:25

    PB->>PB: firstByte equals MessageType.AGENT_ACTION 0x23 store/websocket/binaryProtocol.ts:482
    PB->>PB: handleAgentActionTagged(data) store/websocket/binaryProtocol.ts:483 calling fn def :438
    Note over PB: shipping wire is a bare tag, not a 6-byte framed header - [0x23][u16 count]([u16 len][event])... store/websocket/binaryProtocol.ts:423-428
    alt data.byteLength greater-or-equal 18 store/websocket/binaryProtocol.ts:439
        PB->>DEC: decodeAgentActions(data.slice(1)) store/websocket/binaryProtocol.ts:440
    else too small
        PB->>PB: actions = empty array store/websocket/binaryProtocol.ts:441
    end
    DEC->>DEC: eventCount = view.getUint16(0) agentMessages.ts:192
    loop for each event, offset advances by 2 plus eventLen agentMessages.ts:195-215
        DEC->>DEC: eventLen = view.getUint16(offset) agentMessages.ts:201
        DEC->>DEC: decodeAgentAction(eventPayload) sourceAgentId,targetNodeId,actionType,timestamp,durationMs agentMessages.ts:153-181,210
    end
    DEC-->>PB: AgentActionEvent[] passed to dispatchAgentActions store/websocket/binaryProtocol.ts:442
    opt actions.length greater than 0 store/websocket/binaryProtocol.ts:446
        PB->>PB: emit(agent-action, actions) live transcript and attention heat store/websocket/binaryProtocol.ts:448
        PB->>BEAM: pushTransientBeams(actions) TransientBeamsLayer sink store/websocket/binaryProtocol.ts:451
        PB->>AT: useAgentTargetStore.getState().pushActions(actions)<br/>store/websocket/binaryProtocol.ts:453
        Note over PB,AT: 2026-09-11 (dcf5a1069): a THIRD sink on the same decoded batch,<br/>feeding the desktop momentum nudge in BotsVisualization. Each sink is<br/>pushed in parallel so none can starve the others.<br/>client/src/store/agentTargetStore.ts:16, :22
    end
    Note over PB: AGENT_ACTION_HEADER_SIZE=15 bytes: sourceAgentId u32@0, targetNodeId u32@4, actionType u8@8, timestamp u32@9, durationMs u16@13, optional payload from @15 frameTypes.ts:214,agentMessages.ts:161-165
```
## VC-32.15 JSON control-frame catalogue
```mermaid
classDiagram
    class ClientToServer {
      authenticate  token/pubkey/ephemeral in dev mode only, the event form is unreachable from the client since the signed subprotocol (VC-32.1)  connectionManager.ts:377-388
      filter_update  enabled,quality_threshold,authority_threshold,filter_by_quality,filter_by_authority,filter_mode,include_linked_pages  index.ts:360-383
      subscribe_position_updates  binary only from AppInitializer.tsx:295,319, binary and interval=200 from the watchdog probe connectionManager.ts:288-291, the server ignores interval position_updates.rs:601-606
      request_full_snapshot  no payload  connectionManager.ts:299, handled message_routing.rs:35-37
      nodeDragStart  nodeId,position  useGraphEventHandlers.ts:150-157
      nodeDragUpdate  nodeId,position,timestamp  useGraphEventHandlers.ts:64-68
      nodeDragEnd  nodeId  useGraphEventHandlers.ts:255-257
      ping  raw string, not JSON, answered by a raw pong  connectionManager.ts:271, message_routing.rs:18-21
    }
    class ServerToClient {
      connection_established  sets isServerReady  textMessageHandler.ts:90-95
      error  category:validation|server|protocol|auth|rate_limit, code, retryable, retryAfter  types.ts:14-23,store/websocket/binaryProtocol.ts:211, dispatched textMessageHandler.ts:97-104
      filter_update_success  data.visible_nodes,data.total_nodes  textMessageHandler.ts:106-114
      initialGraphLoad  nodes,edges  textMessageHandler.ts:116-118,226-330
      memory_flash  data forwarded to memoryFlash event  textMessageHandler.ts:121-123
      graphUpdated  revision,reason, debounced 750ms refetch  textMessageHandler.ts:27-60,128-132
      pong  raw string, not JSON  binaryFrameDispatcher.ts:123-126
      settingsUpdated  category,updatedBy,timestamp,settings? textMessageHandler.ts:138-140,154-191
    }
    ClientToServer --> ServerToClient : request/response over one socket
    note for ClientToServer "sent via store websocket/index.ts sendMessage(type,data) JSON.stringify(type,data)<br/>index.ts:301-330. Queued to messageQueue (MAX_QUEUE_SIZE=100) when not connected,<br/>connectionManager.ts:27,149-168."
    note for ServerToClient "dispatched by handleTextMessage after JSON.parse and validateMessage(type is<br/>nonempty string, length less-or-equal 100)<br/>websocket/binaryFrameDispatcher.ts:82-90,textMessageHandler.ts:80-143."
    note for ServerToClient "RESOLVED ADR-2080 settingsUpdated now HAS a consumer. The server emitted<br/>it from src/settings/api/settings_routes.rs:454,1012 while the only client knowledge was<br/>validation.ts:274 case settings_update (snake_case, different shape) - every broadcast fell<br/>through to the default arm and did nothing, so two viewers saw different physics until one<br/>reloaded. nodeFilter applies the supplied settings directly. other categories re-read via<br/>getSectionPaths plus getSettingsByPaths. own-write echo and stale timestamps are dropped.<br/>Contract fixed by vc-core ADR-2047. settings_update is dead on the wire and left in place"
    note for ServerToClient "GRAPH_UPDATED shrink-guard: a smaller initialGraphLoad is discarded unless<br/>isFilterResponseExpected() is true within the 15000ms window, armed only for an<br/>enabled filter, and a REST load in flight defers the whole payload - see VC-32.9.<br/>textMessageHandler.ts:232-267,filterSync.ts:29-42,124-128"
```
## VC-32.16 solidWebSocket: JSS resource-notification boundary
```mermaid
sequenceDiagram
    autonumber
    participant C as store<br/>websocket/index.ts:447
    participant SW as solidWebSocket store adapter<br/>websocket/solidWebSocket.ts:71
    participant PN as podNotificationManager (singleton)<br/>services/solidPod/podNotifications.ts:300
    participant JSS as JSS WebSocket<br/>VITE_JSS_WS_URL

    C->>SW: connectSolid() index.ts:447-449
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
