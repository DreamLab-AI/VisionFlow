---
id: VC-14
title: Wire frames and tag registry
area: visionclaw
governing:
  - ../project/docs/PROTOCOL-registry.md
  - ../project/docs/GPU-wire-abi.md
adrs: [ADR-2018, ADR-2019, ADR-2020, ADR-2024, ADR-2057, ADR-2060, ADR-2139, ADR-2141]
sources:
  - ../project/src/utils/binary_protocol.rs
  - ../project/src/protocols/binary_settings_protocol.rs
  - ../project/crates/visionclaw-protocol/src/lib.rs
  - ../project/crates/visionclaw-protocol/src/wire_fixtures.rs
  - ../project/crates/visionclaw-wire/src/lib.rs
  - ../project/docs/PROTOCOL-registry.md
  - ../project/docs/adr/ADR-2141-positions-are-streamed-per-client-until-acknowledged.md
  - ../project/crates/visionclaw-xr-presence/src/wire.rs
  - ../project/crates/visionclaw-xr-presence/src/agent_presence.rs
  - ../project/xr-client/rust/src/binary_protocol.rs
  - ../project/client/src/services/binaryProtocol/frameTypes.ts
  - ../project/client/src/services/BinaryWebSocketProtocol.ts
  - ../project/src/actors/agent_beam_actor.rs
  - ../project/client/src/types/__tests__/wireFixtures.test.ts
  - ../project/client/src/types/binaryProtocol.ts
  - ../project/src/actors/presence_actor.rs
  - ../project/client/src/store/websocket/binaryProtocol.ts
  - ../project/src/actors/client_coordinator_actor.rs
  - ../project/src/handlers/socket_flow_handler/position_updates.rs
  - ../project/src/handlers/socket_flow_handler/binary_protocol.rs
  - ../project/client/src/features/graph/managers/dataManager/wsClient.ts
verified_commit: e4ba5eb35ee6715cfd7f2daf1cd9af05e8a18cce
---

## VC-14.1 V3 position record — 52 bytes, little-endian

```mermaid
classDiagram
    class WireNodeDataItemV3 {
        +u32 id__at0_4B
        +Vec3Data position__at4_12B
        +Vec3Data velocity__at16_12B
        +f32 sssp_distance__at28_4B
        +i32 sssp_parent__at32_4B
        +u32 cluster_id__at36_4B
        +f32 anomaly_score__at40_4B
        +u32 community_id__at44_4B
        +f32 centrality__at48_4B
        +TOTAL_52_BYTES()
    }
    class WireSizeConstants {
        +WIRE_V2_ID_SIZE_4 L67
        +WIRE_VEC3_SIZE_12 L68
        +WIRE_F32_SIZE_4 L69
        +WIRE_I32_SIZE_4 L70
        +WIRE_U32_SIZE_4 L71
        +WIRE_V3_ITEM_SIZE_52 L72_L80
    }
    WireNodeDataItemV3 --> WireSizeConstants : size checked by

    class SizeLockEvidence {
        +struct_at_binary_protocol_rs L43
        +v5_seq_const_assert L99
        +const_assert_52 L90
        +redundant_test_asserts L1095_L1191
        +frame_is_tag_byte_then_N_records()
        +little_endian_throughout()
    }
    WireSizeConstants --> SizeLockEvidence

    note for SizeLockEvidence "INVARIANT ADR-2018 WIRE_V3_ITEM_SIZE == 52"
    note for SizeLockEvidence "RESOLVED ADR-2060: registry citations corrected"
    note for SizeLockEvidence "RESOLVED ADR-2057: now a compile-time const assert"
    note for WireSizeConstants "RESOLVED the WIRE_V2_ITEM_SIZE and WIRE_ITEM_SIZE aliases are deleted, the two redundant tests now compare WIRE_V3_ITEM_SIZE and WIRE_V2_ID_SIZE directly - utils/binary_protocol.rs:1092-1100"
```

## VC-14.2 Wire node id — 26-bit id plus type flag bits

```mermaid
classDiagram
    class WireNodeId {
        +bits_0_to_25_node_id
        +bit_26_ONTOLOGY_CLASS_0x04000000 L22
        +bit_27_ONTOLOGY_INDIVIDUAL_0x08000000 L23
        +bit_28_ONTOLOGY_PROPERTY_0x10000000 L24
        +bit_30_KNOWLEDGE_0x40000000 L18
        +bit_31_AGENT_0x80000000 L17
    }
    class Masks {
        +NODE_ID_MASK_0x03FFFFFF L28
        +ONTOLOGY_TYPE_MASK_0x1C000000 L21
        +MAX_ID_67108863()
    }
    class RemapFunctions {
        +remap_wire_id() L217
        +to_wire_id() L319
        +clear_all_flags_and_get_actual_node_id() L233_L247
    }
    WireNodeId --> Masks
    Masks --> RemapFunctions

    note for Masks "INVARIANT ADR-2024 NODE_ID_MASK = 0x03FF_FFFF - flag bits 26-31 are stripped before analytics and SSSP map lookups"
    note for RemapFunctions "Ids above the mask are remapped and logged NODE ID OVERFLOW at utils/binary_protocol.rs:197 inside enforce_wire_id_bounds (utils/binary_protocol.rs:185) - remap_wire_id (utils/binary_protocol.rs:217) returns (masked_id, overflowed)"
    note for RemapFunctions "Encoder strips flag bits before SSSP and analytics lookups - the maps are keyed by compact id, skipping this is a real bug class"
    note for WireNodeId "Ontology sub-type bits are only meaningful when the node belongs to GraphType::Ontology (comment utils/binary_protocol.rs:20)"
```

## VC-14.3 Tag byte 0 registry — two disjoint socket spaces

```mermaid
flowchart TD
    subgraph GRAPH["/wss graph socket - tag space A"]
        T03["0x03 Graph position frame V3<br/>52 B per node<br/>utils/binary_protocol.rs:11 PROTOCOL_V3, dispatch :608"]
        T05["0x05 V5 broadcast envelope wrapping V3<br/>PROTOCOL_V5 branch utils/binary_protocol.rs:612"]
        T23["0x23 AGENT_ACTION beam event<br/>MessageType::AgentAction :1792"]
        T34["0x34 BroadcastAck client to server, 21 bytes with no generic header, one per applied position chunk (ADR-2141)<br/>MessageType::BroadcastAck :1788, decode_message arm :1853"]
        T02["0x02 VoiceData :1782"]
        T03c["0x03 ControlFrame in MessageType :1784"]
        T04["0x04 RETIRED V4 delta frame, reserved and never reused<br/>no encoder exists, every decoder refuses it (ADR-2139)<br/>docs/PROTOCOL-registry.md:57"]
    end
    subgraph PRESENCE["/ws/presence socket - tag space A sibling"]
        T43["0x43 OPCODE_AVATAR_POSE<br/>crates/visionclaw-xr-presence/src/wire.rs:9"]
        T44["0x44 OPCODE_AGENT_PRESENCE<br/>crates/visionclaw-xr-presence/src/agent_presence.rs:40"]
    end
    subgraph SETTINGS["settings binary socket - tag space B, DISJOINT"]
        S05["0x05 settings batch or value-change message<br/>src/protocols/binary_settings_protocol.rs"]
        S03["0x03 BatchGet in the same space"]
    end

    N1["INVARIANT ADR-2019 tag allocation is per socket - the numeric overlap between 0x05 graph V5 and 0x05 settings is safe only because the sockets are demultiplexed independently"]
    N2["RESOLVED ADR-2060: BASELINE cited the SETTINGS 0x05 as the graph V5 envelope. Corrected, and the registry now warns that a 0x05 citation must always name its socket"]
    N3["DIVERGENCE MessageType enum (utils/binary_protocol.rs:1776) reuses 0x03 for ControlFrame while PROTOCOL_V3 also uses 0x03 as the position-frame lead byte - disambiguated only by direction and call site"]
    N4["RESOLVED ADR-2057: the V5 envelope now has an owning ADR fixing its layout [0x05] u64 seq then V3 body, and the broadcast_seq contract"]
    N5["OPEN: the settings binary protocol is still not fully enumerated in the registry - unchanged by the 2026-09-05 remediation"]
    N6["RESOLVED ADR-2060: the legacy 48B/28B ADR figures are marked retired in the governing docs - the wire is 52B, now compile-time locked"]
    N7["RESOLVED ADR-2139: tag 0x04 (V4 delta) is now an explicit RETIRED row in the registry, so nothing can be allocated over it"]
    N8["INVARIANT ADR-2143: Codescape is recorded as a second, independent producer of the frozen 0x03, 0x05, 0x23 and 0x34 contract (its encoders exist, serving on port 4100 is phased) and allocates no tag byte and no flag bit, so this registry stays the only allocator - docs/PROTOCOL-registry.md:190-197"]
    T05 -.- N2
    T03 -.- N3
    T05 -.- N4
    SETTINGS -.- N5
    T03 -.- N6
    T04 -.- N7
    T05 -.- N8
    SETTINGS -.- N1
```

## VC-14.4 V5 envelope and tag dispatch with unknown-tag rejection

```mermaid
sequenceDiagram
    autonumber
    participant SRV as Server encoder<br/>src/utils/binary_protocol.rs
    participant SOCK as WebSocket /wss
    participant DEC as decode dispatch<br/>utils/binary_protocol.rs:608 match protocol_version
    participant V3 as decode_node_data_v3<br/>utils/binary_protocol.rs:630

    SRV->>SOCK: frame bytes
    SOCK->>DEC: data[0] read as protocol_version :605, payload = data[1..] :606
    alt version == 1
        Note over DEC: Err "Protocol V1 is no longer supported. Please upgrade client." :609
    else version == 2
        Note over DEC: Err "V2 protocol no longer supported. Please upgrade client to V3+." :610
    else version == PROTOCOL_V3 (0x03)
        DEC->>V3: decode_node_data_v3(payload) :611
    else version == PROTOCOL_V5 (0x05) :612
        Note over DEC: V5 layout is [0x05] then u64 broadcast_seq LE then the V3 body
        alt payload.len() < WIRE_V5_SEQ_SIZE
            Note over DEC: Err "V5 frame too small for broadcast sequence" :615
        else
            DEC->>V3: decode_node_data_v3(&payload[WIRE_V5_SEQ_SIZE..]) :618
        end
    else any other byte
        Note over DEC: Err "Unknown protocol version" :620 - INVARIANT unknown tags are REJECTED, never reinterpreted
    end
    V3->>V3: reject when data.len() is not a multiple of WIRE_V3_ITEM_SIZE :631
    V3->>V3: expected_nodes = data.len() / WIRE_V3_ITEM_SIZE :639 then as_chunks of 52 bytes :657
    Note over DEC,V3: The V5 body carries NO inner 0x03 byte - the receiver distinguishes 0x03 from 0x05 by the lead byte alone
    Note over DEC: broadcast_seq gives clients a monotonic ordering and drop-detection handle
    Note over DEC,V3: RESOLVED ADR-2060 - the V5 decode citation is corrected, and ADR-2057 replaced the bare literal 5 with a named PROTOCOL_V5 constant and WIRE_V5_SEQ_SIZE
```

## VC-14.5 0x23 AGENT_ACTION frame

```mermaid
classDiagram
    class AgentActionEvent {
        +u32 source_agent_id__at0_4B L1562
        +u32 target_node_id__at4_4B L1563
        +u8 action_type__at8_1B L1564
        +u32 timestamp_ms__at9_4B L1565
        +u16 duration_ms__at13_2B L1566
        +Vec~u8~ payload__at15_variable L1567
    }
    class AgentActionType {
        <<enum>>
        +Query_0 L1535
        +Update_1 L1536
        +Create_2 L1537
        +Delete_3 L1538
        +Link_4 L1539
        +Transform_5 L1540
    }
    class FrameLayout {
        +AGENT_ACTION_HEADER_SIZE_15 L1571
        +lead_byte_MessageType_AgentAction_0x23 L1792
        +capacity_1_plus_15_plus_payload L1596
    }
    AgentActionEvent --> AgentActionType : action_type
    AgentActionEvent --> FrameLayout : encoded by

    note for FrameLayout "Frame = [0x23][u16 count LE] then per event [u16 event_len LE][15-byte header plus optional payload] - multiple actions coalesce into ONE frame, encode_agent_actions utils/binary_protocol.rs:1652 and decode_agent_actions :1675"
    note for AgentActionEvent "ADR-2020 identity-blind by design - carries agent-id-space numeric ids only, never a pubkey or DID"
    note for AgentActionEvent "RESOLVED ADR-2060: the ~+340-line 0x23 citation drift is corrected"
    note for AgentActionType "Coalescing and fan-out live in src/actors/agent_beam_actor.rs - see VC-16.4, ingest side is ES-02"
```

## VC-14.6 Encode to decode of one V5 frame, server to clients

```mermaid
sequenceDiagram
    autonumber
    participant FCA as ClientCoordinatorActor serialize_positions<br/>src/actors/client_coordinator_actor.rs:532
    participant ENC as encode_node_data_extended_with_sssp<br/>src/utils/binary_protocol.rs:426
    participant WS as WebSocket /wss
    participant RS as XR Rust decoder<br/>xr-client/rust/src/binary_protocol.rs:458
    participant TS as Web TS decoder<br/>client/src/types/binaryProtocol.ts:328

    FCA->>ENC: the coordinator pump encodes each client's chunk, node tuples plus the NodeClassIds<br/>sets, src/actors/client_coordinator_actor.rs:540-554
    ENC->>ENC: per node stamp type flag bits (utils/binary_protocol.rs:465-474) then<br/>enforce_wire_id_bounds (utils/binary_protocol.rs:490), to_wire_id_v2<br/>(utils/binary_protocol.rs:508)
    ENC->>ENC: PRE-STAMPED ids forwarded unchanged (utils/binary_protocol.rs:475-484) when get_node_type is not Unknown
    Note over ENC: INVARIANT (2026-09-08, 6c47b9833) — a caller that stamps the class flag itself and<br/>passes EMPTY class sets, for example the swarm telemetry encode in<br/>position_updates.rs:118,152, reaches the<br/>untyped fall-through with bits 26-31 already set. The chunk path passes real class<br/>sets instead. A compact raw id (ADR-2024) never carries<br/>those bits, so a recognised class pattern means stamped and the encoder forwards it.<br/>Running the 26-bit bound check on a stamped id panicked every broadcast in the<br/>dev-runtime<br/>build — each /wss client saw Connection reset without closing handshake on a 2 s loop<br/>—<br/>and a release build would have silently stripped the class flag instead.<br/>Regression reads the id back off the wire, utils/binary_protocol.rs:1241<br/>DRIFT the encoder comment still names actor_messages.rs as a pre-stamping caller, that file no longer calls<br/>any encoder, its position broadcast handler was removed by ADR-2141, utils/binary_protocol.rs:476-478
    ENC->>ENC: strip flag bits before SSSP and analytics map lookups
    ENC->>ENC: write 52-byte record - id@0 pos@4 vel@16 sssp_dist@28 sssp_parent@32 cluster@36<br/>anomaly@40 community@44 centrality@48
    Note over ENC: sssp_distance defaults to f32::INFINITY and sssp_parent to -1 when absent
    ENC->>WS: [tag][records...]
    par XR Rust client
        WS->>RS: bytes
        RS->>RS: decode_position_frame_with_sequence :458
        alt lead byte == PROTOCOL_V5 (0x05) :473
            RS->>RS: need = HEADER_BYTES + V5_SEQ_BYTES :474 then read u64 seq :481-483
        end
        RS->>RS: mask each id with NODE_ID_MASK :759
        Note over RS: constants PROTOCOL_V5 :29, V5_SEQ_BYTES 8 :30, NODE_RECORD_BYTES 52 :32, NODE_ID_MASK :41
    and Web TS client
        WS->>TS: bytes
        TS->>TS: parseBinaryNodeData dispatch on lead byte :174-215
        alt lead byte == PROTOCOL_V5 (0x05) :199-202
            TS->>TS: parseV5Nodes :328 - reject if byteLength < 9 :330, read u64 seq LE :335-337, decode body from offset 9 :340
            TS->>TS: surface broadcastSequence :368 - handleLegacyBinaryData acknowledges that sequence<br/>after applying the chunk store/websocket/binaryProtocol.ts:406-408
        else lead byte == 0x02 (V2)
            TS->>TS: DECLINED with a diagnostic - the server rejects V2 at utils/binary_protocol.rs:610
        else lead byte == 0x04 (V4 delta)
            TS->>TS: DECLINED with a diagnostic, position frames are never deltas (ADR-2139),<br/>client/src/types/binaryProtocol.ts:190-198
        else unrecognised version
            TS->>TS: DECLINED - no size autodetection
        end
        Note over TS: CORRECTED ADR-2078. ADR-2057 Finding 1 was WRONG: the live TS path always<br/>had V5, including the short-payload guard that mirrors the server at<br/>utils/binary_protocol.rs:615.<br/>client/src/services/binaryProtocol/ is NOT the live position path - it has no<br/>52-byte decoding at all, so a V5 branch there would have been a SECOND decoder.<br/>The real defect was Finding 2 and it was worse than reported - see the next note
        Note over TS: RESOLVED ADR-2078. V2 was DECODED, not merely advertised (36-byte records, the<br/>V2 case at client/src/types/binaryProtocol.ts:175-185, routed in at<br/>client/src/store/websocket/binaryProtocol.ts:463-467),<br/>and the default arm re-read any unknown frame from offset 0 as 36-byte records<br/>whenever its length divided by 36 - fabricating nodes from arbitrary payloads.<br/>Both now decline. BINARY_NODE_SIZE_V2 and the size-swap heuristic are deleted
        Note over TS: RESOLVED ADR-2078. The TS decoder now has the fixture cross-check the two Rust<br/>decoders always had - client/src/types/__tests__/wireFixtures.test.ts, 12 tests<br/>pinning the same constants as wire_fixtures.rs plus a synthetic V5 round-trip
    end
    Note over ENC,TS: Shared fixtures (crates/visionclaw-protocol/src/wire_fixtures.rs:1-6 now only<br/>re-exports visionclaw_wire, whose node-record and ack constants are at<br/>crates/visionclaw-wire/src/lib.rs:83,92-95) pin the format for both decoders -<br/>utils/binary_protocol.rs:960-965 asserts fx::NODE_RECORD_BYTES == WIRE_V3_ITEM_SIZE<br/>and fx::NODE_ID_MASK == NODE_ID_MASK, and :968 guards the shared 0x23 fixture against<br/>encoder drift
```

## VC-14.7 The coexisting binary codecs

```mermaid
classDiagram
    class ServerEncoder {
        +path_src_utils_binary_protocol_rs()
        +WIRE_V3_ITEM_SIZE_52 L72
        +encode_node_data_with_types() L380
        +encode_node_data_extended_with_sssp() L426
        +encode_node_data_with_flags() L556
        +encode_node_data() L571
        +decode_dispatch() L588
        +decode_broadcast_ack() L1866
        +CANONICAL_source_of_truth()
    }
    class XrRustDecoder {
        +path_xr_client_rust_src_binary_protocol_rs()
        +PROTOCOL_V5_0x05 L29
        +V5_SEQ_BYTES_8 L30
        +NODE_RECORD_BYTES_52 L32
        +NODE_ID_MASK L41
        +decode_position_frame() L446
        +decode_position_frame_with_sequence() L458
        +encode_broadcast_ack() L515
        +admit_and_ack() L701
    }
    class WebTsDecoder {
        +path_client_src_services_binaryProtocol()
        +PROTOCOL_V2_removed_comment L5_L8
        +PROTOCOL_V3_3 L9
        +PROTOCOL_V4_removed_ADR_2078 L10_L16
        +SUPPORTED_HEADER_VERSIONS L39
        +no_node_record_decoding()
    }
    class SharedFixtures {
        +path_crates_visionclaw_protocol_src_wire_fixtures_rs()
        +NODE_RECORD_BYTES()
        +NODE_ID_MASK()
        +agent_action_0x23_fixture()
        +broadcast_ack_frame_0x34()
        +reexport_of_visionclaw_wire L6
    }
    ServerEncoder --> SharedFixtures : asserted against L960_L965
    XrRustDecoder --> SharedFixtures : same source file included
    WebTsDecoder --> ServerEncoder : must track manually

    note for ServerEncoder "visionclaw-protocol crate's own binary_protocol module was REMOVED as a stale 48-byte copy - crates/visionclaw-protocol/src/lib.rs:9-21 now points here as the single encoder"
    note for WebTsDecoder "DOC-DRIFT — this module now EXPORTS PROTOCOL_V5 and includes it in SUPPORTED_HEADER_VERSIONS (frameTypes.ts:23,39); PROTOCOL_V4 was removed from here under ADR-2078 and survives only as the RETIRED delta tag in types/binaryProtocol.ts, which declines a V4 frame (ADR-2139, types/binaryProtocol.ts:190-198). It still has NO 52-byte node-record decode logic at all (grep confirmed) — the live V5 position decoder is client/src/types/binaryProtocol.ts, see VC-14.6"
    note for WebTsDecoder "ROOT CAUSE (ADR-2057): the TS decoder is the only one with no fixture cross-check"
    note for SharedFixtures "ADR-2018 consumer freshness and ADR-2019 per-opcode frame policy - the fixtures now live in the dependency-free visionclaw-wire crate (crates/visionclaw-wire/src/lib.rs:83,92-95) and visionclaw-protocol re-exports them (wire_fixtures.rs:6), so the isolated xr-client workspace and the Codescape build share one source"
```

## VC-14.8 V4 framed-message header — client-emitted only

```mermaid
classDiagram
    class V4FramedHeader {
        +u8 type__at0_1B
        +u8 version__at1_1B
        +u32 payloadLength_LE__at2_4B
        +TOTAL_6_BYTES()
    }
    class TagsDeclaredClientSide {
        +GRAPH_UPDATE_0x01 L67
        +VOICE_DATA_0x02 L70
        +POSITION_UPDATE_0x10 L73
        +AGENT_POSITIONS_0x11 L74
        +AGENT_STATE_FULL_0x20 L77
        +AGENT_ACTION_0x23 L78
        +CONTROL_BITS_0x30 L81
        +SSSP_DATA_0x31 L82
        +BROADCAST_ACK_0x34 L90
        +VOICE_CHUNK_0x40 L85
        +VOICE_START_0x41 L86
        +VOICE_END_0x42 L87
        +SYNC_UPDATE_0x50 L93
        +ANNOTATION_UPDATE_0x51 L94
        +SELECTION_UPDATE_0x52 L95
        +USER_POSITION_0x53 L96
        +VR_PRESENCE_0x54 L97
        +ERROR_0xFF L100
    }
    V4FramedHeader --> TagsDeclaredClientSide : type byte drawn from

    class ServerMessageTypeEnum {
        +BinaryPositions_0 L1780
        +VoiceData_0x02 L1782
        +ControlFrame_0x03 L1784
        +AgentAction_0x23 L1792
        +BroadcastAck_0x34 L1788
        +enum_declared_at L1776_L1793
    }
    TagsDeclaredClientSide --> ServerMessageTypeEnum : only 3 of 18 share a value with a server tag

    note for V4FramedHeader "Shape at BinaryWebSocketProtocol.ts:81 - 6-byte header"
    note for V4FramedHeader "By design: V4 is not the wire format - live streams are UNFRAMED"
    note for V4FramedHeader "RESOLVED ADR-2141: the BROADCAST_ACK is no longer sent through this header, createBroadcastAck builds [0x34][20-byte payload] by hand because the server's decode_message reads no generic header - client/src/services/BinaryWebSocketProtocol.ts:245-255 and utils/binary_protocol.rs:1853"
    note for TagsDeclaredClientSide "OPEN: the client declares 18 tags, the server enum defines 5, and the file's own comment says the rest are client-outbound only - routed to vc-clients, client/src/services/binaryProtocol/frameTypes.ts:43-57"
    note for TagsDeclaredClientSide "RESOLVED ADR-2078: VELOCITY_UPDATE, AGENT_STATE_DELTA, AGENT_HEALTH, HANDSHAKE and HEARTBEAT were deleted from the enum, the earlier 20-tag count is now 18 - client/src/services/binaryProtocol/frameTypes.ts:59-63"
    note for TagsDeclaredClientSide "DOC-DRIFT the SERVER TAG comments in this enum cite server lines 1711, 1717 and 1721 and the header comment cites lines 1705 to 1722, the enum now sits at src/utils/binary_protocol.rs:1776-1793 with VoiceData :1782, BroadcastAck :1788 and AgentAction :1792 - client/src/services/binaryProtocol/frameTypes.ts:45,70,78,90"
    note for ServerMessageTypeEnum "RESOLVED ADR-2057: the 48-byte comment is corrected to 52 - utils/binary_protocol.rs:1777-1779"
```

## VC-14.9 Presence sibling opcodes 0x43 and 0x44

```mermaid
classDiagram
    class AvatarPose_0x43 {
        +u8 opcode_0x43 wire_rs_L9
        +u16 frame_len_LE
        +u8_16 room_id_hash
        +u8 avatar_id_len
        +avatar_id_utf8
        +u64 timestamp_us_LE
        +u8 transform_mask
        +transforms_28B_each_head_left_right
    }
    class TransformMask {
        +head_0b001 wire_rs_L18
        +left_0b010 wire_rs_L19
        +right_0b100 wire_rs_L20
    }
    class AgentPresence_0x44 {
        +u8 opcode_0x44 agent_presence_rs_L40
        +u16 body_len_LE
        +u64 seq_LE
        +u16 agent_count_LE
        +per_agent_u32_local_id
        +per_agent_u8_field_mask
    }
    class FieldMask {
        +bit0_state_u8_idle_working_awaiting_speaking
        +bit1_gaze_i16x3_quantised_scale_32767
        +bit2_attention_u8_tag_plus_u32_node_id_when_tag_2
    }
    AvatarPose_0x43 --> TransformMask
    AgentPresence_0x44 --> FieldMask

    note for TransformMask "Presence BITMASK not a count - asymmetric hand presence round-trips correctly"
    note for AgentPresence_0x44 "ADR-2020 additive sibling of 0x43 carrying social state, not skeleton - see VC-17 for the presence pipeline"
    note for AgentPresence_0x44 "Two logical channels share one codec - reliable for state and attention, high-rate 10-20 Hz for gaze-only"
    note for AvatarPose_0x43 "Server fan-out envelope [0x43][u64 broadcast_seq LE][u32 room_id LE][u16 user_count LE] built at src/actors/presence_actor.rs:370-380"
```

## VC-14.10 Position acknowledgement 0x34 — one 21-byte frame per applied chunk

```mermaid
sequenceDiagram
    autonumber
    participant SRV as ClientCoordinatorActor<br/>src/actors/client_coordinator_actor.rs:617
    participant WS as WebSocket /wss
    participant WEB as Web client<br/>client/src/store/websocket/binaryProtocol.ts:351
    participant XR as XR Rust client<br/>xr-client/rust/src/binary_protocol.rs:701
    participant HDL as SocketFlowServer binary handler<br/>src/handlers/socket_flow_handler/binary_protocol.rs:30

    SRV->>WS: V5 chunk [0x05][u64 seq][52 B per node], seq is the chunk's own number<br/>src/actors/client_coordinator_actor.rs:532-563
    par web client
        WS->>WEB: chunk bytes
        WEB->>WEB: apply through graphDataManager.updateNodePositions, an exception returns WITHOUT an ack<br/>client/src/store/websocket/binaryProtocol.ts:392-401
        WEB->>WEB: sendPositionAck(seq, node count) after the apply, a V3 frame has no sequence and is never acknowledged<br/>client/src/store/websocket/binaryProtocol.ts:406-408
        WEB->>WS: createBroadcastAck builds [0x34][u64 seq][u32 nodes][u64 timestamp_ms], 21 bytes, no 6-byte header<br/>client/src/services/BinaryWebSocketProtocol.ts:245-255
    and XR client
        WS->>XR: chunk bytes
        XR->>XR: freshness gate admits the frame, only Accept and AcceptResync name a sequence<br/>xr-client/rust/src/binary_protocol.rs:528-533
        XR->>WS: encode_broadcast_ack after the records are applied, a stale or duplicate frame is not acknowledged<br/>xr-client/rust/src/binary_protocol.rs:515-522,701-712
    end
    WS->>HDL: binary frame
    HDL->>HDL: is_stream_ack, the first byte 0x34, is exempt from the per-IP message rate limiter<br/>src/handlers/socket_flow_handler/binary_protocol.rs:10-12,22-24,35
    HDL->>HDL: BinaryProtocol::decode_message, the 0x34 arm reads the 20-byte payload with no generic header<br/>src/utils/binary_protocol.rs:1853,1866-1892
    HDL->>SRV: ClientBroadcastAck{sequence_id, nodes_received, timestamp, client_id}<br/>src/handlers/socket_flow_handler/binary_protocol.rs:67-86
    Note over SRV: the ack commits that chunk's nodes as held by this client and the next chunk is built at once, see VC-13.4<br/>src/actors/client_coordinator_actor.rs:1365-1396
    Note over WS,SRV: INVARIANT the ack bytes are asserted against one golden frame by the server, the web client and the XR client<br/>crates/visionclaw-wire/src/lib.rs:92-95,334 and src/utils/binary_protocol.rs:2159-2178
    Note over WEB: TENSION ADR-2141 decision 5 says a chunk that was not applied is not acknowledged, but handleBinaryFrame returns normally when a frame<br/>arrives within 16 ms of the last accepted one, so the ack is still sent for a chunk the worker never saw, VC-13.7<br/>docs/adr/ADR-2141-positions-are-streamed-per-client-until-acknowledged.md:32 and client/src/features/graph/managers/dataManager/wsClient.ts:26-28 and client/src/store/websocket/binaryProtocol.ts:392-408
```
