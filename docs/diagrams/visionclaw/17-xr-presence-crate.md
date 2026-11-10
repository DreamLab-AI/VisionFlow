---
id: VC-17
title: XR presence crate and co-presence
area: visionclaw
governing:
  - ../project/docs/PROTOCOL-registry.md
  - ../project/docs/XR-client.md
adrs: [ADR-2019, ADR-2020]
sources:
  - ../project/docs/XR-client.md
  - ../project/crates/visionclaw-xr-presence/src/lib.rs
  - ../project/crates/visionclaw-xr-presence/src/wire.rs
  - ../project/crates/visionclaw-xr-presence/src/agent_presence.rs
  - ../project/crates/visionclaw-xr-presence/src/validate.rs
  - ../project/src/handlers/presence_handler.rs
  - ../project/src/actors/presence_actor.rs
  - ../project/src/main.rs
verified_commit: {visionclaw: 010e1109bd78f253f0d82382bb94908b5023b1e9}
---

## For developers

The `visionclaw-xr-presence` crate holds the presence wire types and validation. The server's `presence_handler` and `PresenceActor` use it to authenticate a headset session, relay its pose to peers on the 0x43 frame and carry agent co-presence on the sibling 0x44 channel. Re-stamped at ec785afa9: no presence source changed, and the only drift is in the governing XR-client and PROTOCOL-registry docs and in `main.rs` line positions.

## For the business

People wearing headsets in the same space see each other's head and hands move in real time, and AI agents appear beside them as present participants. A session must prove who it is before its movements are shared.

## VC-17.1 Presence session handshake and authentication

```mermaid
sequenceDiagram
    autonumber
    participant C as XR client<br/>xr-client/rust
    participant PS as PresenceSession<br/>src/handlers/presence_handler.rs:433
    participant PA as PresenceActor<br/>src/actors/presence_actor.rs:655

    C->>PS: GET /ws/presence upgrade (src/main.rs:1153, ws_presence presence_handler.rs:501)
    PS->>PS: phase = SessionPhase::Challenged{nonce, ts_us} :156
    PS->>C: started :436 then send_challenge :163 - ServerHandshake::Challenge{nonce hex, ts}
    C->>PS: ws::Message::Text routed to handle_auth :461
    alt phase is not Challenged
        Note over PS: warn "auth attempted in wrong phase" then close_with CLOSE_CODE_VALIDATION 4400 :192-195
    else malformed JSON
        Note over PS: close_with_code ws::CloseCode::Unsupported "malformed json" :204
    else client ts more than MAX_HANDSHAKE_SKEW_US 30s from server clock
        Note over PS: close CLOSE_CODE_AUTH_FAIL "stale handshake timestamp" :218-228
    else signature invalid or verified DID differs from claimed DID
        Note over PS: identity_verifier.verify_signed_challenge :237-248, mismatch :249-252, close CLOSE_CODE_AUTH_FAIL 4401 (constant :37)
    else nonce already in the shared SeenNonces LRU
        Note over PS: close CLOSE_CODE_AUTH_FAIL "replayed challenge" :259-271, LRU capacity SEEN_NONCE_CAPACITY 4096 :57
    else verified and fresh
        PS->>PS: room actor from the registry, a disconnected Addr is replaced by a fresh PresenceActor :281-294
        PS->>PA: JoinRoom (presence_handler.rs:305)
        PA->>PA: handle_join :664, local_id_for(avatar_id) :695 assigns the compact u32 wire id (fn presence_actor.rs:360)
        PA-->>PS: JoinAck{avatar_id, members}
        PS->>C: ServerHandshake::Joined (presence_handler.rs:327-334)
        PS->>PS: phase = SessionPhase::Joined :335
    end
    loop every HEARTBEAT_INTERVAL 15s presence_handler.rs:35, run_interval presence_handler.rs:438
        PS->>C: heartbeat :403 pings, and stops the session after two silent intervals :404-407
        PS->>PS: enforce_handshake_deadline :395 closes 4401 while phase is still Challenged :396 past HANDSHAKE_TIMEOUT 10s (presence_handler.rs:34)
    end
    Note over PS: TENSION the 10s HANDSHAKE_TIMEOUT is only tested on the 15s heartbeat tick (presence_handler.rs:438-441), so an unauthenticated socket lives about 15s, not 10s
    Note over PS: INVARIANT authentication is a DID signature over a server-issued nonce, and a nonce is accepted at most once across all sessions (SeenNonces presence_handler.rs:51)
    Note over PS: StreamHandler dispatch presence_handler.rs:461-462 routes Text to handle_auth, Binary to handle_pose_frame<br/>handle_pose_frame rejects any binary frame received before phase is Joined with CLOSE_CODE_VALIDATION "binary before auth" :361-364
    Note over PS,PA: On stopping() the session sends LeaveRoom to the room actor only if phase was Joined :444-453
```

## VC-17.2 Headset pose to peers — 0x43 path

```mermaid
sequenceDiagram
    autonumber
    participant HMD as Headset / Godot client
    participant PS as PresenceSession<br/>presence_handler.rs:355 handle_pose_frame
    participant PA as PresenceActor<br/>presence_actor.rs:751 handle_ingest
    participant W as wire::decode<br/>crates/visionclaw-xr-presence/src/wire.rs
    participant V as validators<br/>crates/visionclaw-xr-presence/src/validate.rs
    participant PEERS as Other sessions in room

    HMD->>PS: ws::Message::Binary routed to handle_pose_frame :462
    alt phase is not Joined
        Note over PS: close CLOSE_CODE_VALIDATION "binary before auth" :361-364
    else check_rate_limit presence_handler.rs:175 fails
        Note over PS: 120 frames per RATE_LIMIT_WINDOW 1s :32-33, close with CLOSE_CODE_RATE_LIMIT 4429 :38 (call :366-369)
    else within budget
        PS->>PA: IngestPose{avatar_id, frame_bytes} :373 (handler presence_actor.rs:742)
        PA->>W: wire::decode(frame_bytes) presence_actor.rs:757
        Note over W: [0x43][u16 frame_len LE][u8_16 room_id_hash][u8 avatar_id_len][avatar_id utf8][u64 timestamp_us LE][u8 transform_mask] then 28-byte transforms (wire.rs:24-34)
        Note over W: transform_mask bits head 0b001 (wire.rs:18), left 0b010 (wire.rs:19), right 0b100 (wire.rs:20) - a PRESENCE bitmask not a count, so asymmetric hand presence round-trips
        PA->>PA: reject decoded avatar_id differing from the session's (spoofing) :770-778, room_hash mismatch :780-783
        PA->>V: run_validators (presence_actor.rs:315)
        V->>V: world_bounds on head and each present hand (validate.rs:77, presence_actor.rs:321-332)
        V->>V: hand_reach against hand_reach_m from configured_hand_reach_m, default 1.5m (validate.rs:147, presence_actor.rs:40, :45)
        opt member has a previous frame
            V->>V: monotonic_timestamp (validate.rs:96) then velocity_gate against max_velocity_mps (validate.rs:10, presence_actor.rs:341-342)
        end
        V->>V: joint_anatomy on the two wrist-only HandPoses (validate.rs:118, presence_actor.rs:348)
        alt decode, spoof or validator error
            PA->>PA: record_violation(avatar_id) :299
            Note over PA: the frame is dropped and the session warned, a member reaching VIOLATION_KICK_THRESHOLD 10 within VIOLATION_WINDOW 1s is removed and kicked (presence_actor.rs:33-34, :785-793)
        else all pass
            PA->>PA: pending_poses.insert then dispatch_broadcast :807-809
            PA->>PA: build_broadcast_frame :370
            Note over PA: fan-out envelope [0x43][u64 broadcast_seq LE][u32 room_id LE][u16 user_count LE] - PREAMBLE_OPCODE :54, layout comment :53, push at :379
            PA->>PEERS: BroadcastFrame to every subscriber except the sender, oldest dropped past BACKPRESSURE_LIMIT 3 (presence_actor.rs:35, :562-583)
        end
    end
    Note over PA,PEERS: Production rooms use the defaults set in PresenceActor::new, bounds Aabb::symmetric 50 and 20 m/s (presence_actor.rs:284-285). with_bounds :293 is called only from tests
```

## VC-17.3 Agent co-presence — 0x44 sibling channel

```mermaid
sequenceDiagram
    autonumber
    participant AG as Caller (tests only at this pin)
    participant PA as PresenceActor<br/>presence_actor.rs:814 IngestAgentPresence
    participant AP as agent_presence codec<br/>crates/visionclaw-xr-presence/src/agent_presence.rs
    participant PEERS as Room members

    AG->>PA: IngestAgentPresence :814
    PA->>PA: handle_agent_presence :424
    alt avatar_id is not a joined subscriber
        Note over PA: PermissionDenied :432-438
    else check_node_correlation :415 fails
        Note over PA: attention GraphNode id above WIRE_NODE_ID_MASK 26-bit (presence_actor.rs:67, :417) returns InvalidNodeCorrelation :440-446
    else accepted
        PA->>PA: local_id_for :448, first publication is a full delta, later ones AgentPresenceDelta::between :454-457
        alt delta empty at wire resolution
            Note over PA: Unchanged, nothing broadcast :467-469
        else changed
            PA->>AP: encode_agent_presence(seq, deltas) (presence_actor.rs:492)
            Note over AP: [0x44][u16 body_len LE][u64 seq LE][u16 agent_count LE] then per agent [u32 local_id LE][u8 field_mask] (agent_presence.rs:21-31)
            Note over AP: field_mask bit0 state u8 - 0 idle, 1 working, 2 awaiting_approval, 3 speaking
            Note over AP: field_mask bit1 gaze i16x3 quantised unit direction via quantise_dir agent_presence.rs:398 and dequantise_dir agent_presence.rs:409
            Note over AP: field_mask bit2 attention u8 tag - 0 none, 1 user, 2 node - plus u32 node_id when tag == 2
            PA->>PEERS: broadcast_agent_presence :482 to every subscriber except the publisher :508-516
        end
    end
    AG->>PA: SweepStaleAgentPresence :822, ttl default AGENT_PRESENCE_TTL 10s (presence_actor.rs:63)
    Note over PA: A retired agent is announced on the JSON event channel rather than as a 0x44 delta :230-235, sweep presence_actor.rs:526-549
    Note over PA: The 0x44 broadcast sequence is SEPARATE from the 0x43 pose sequence :264
    Note over PA,PEERS: Queries GetAgentPresence :831 and GetAttentionNode :839 read this state without broadcasting, attention_node presence_actor.rs:403
    Note over AG,PA: DEBT no production code sends IngestAgentPresence, SweepStaleAgentPresence, GetAgentPresence or GetAttentionNode - every sender is in the test module (presence_actor.rs:1177), and started() schedules no sweep (presence_actor.rs:634-640)
```

## VC-17.6 PresenceActor message surface

```mermaid
classDiagram
    class PresenceActor {
        +JoinRoom() L655
        +LeaveRoom() L733
        +IngestPose() L742
        +IngestAgentPresence() L814
        +SweepStaleAgentPresence() L822
        +GetAgentPresence() L831
        +GetAttentionNode() L839
        +ListMembers() L847
        +RoomStats() L855
    }
    class PresenceActorInherent {
        +new() L271
        +with_bounds() L293
        +record_violation() L299
        +run_validators() L315
        +local_id_for() L360
        +build_broadcast_frame() L370
        +attention_node() L403
        +check_node_correlation() L415
        +handle_agent_presence() L424
        +broadcast_agent_presence() L482
        +sweep_stale_agent_presence() L526
        +dispatch_broadcast() L551
        +handle_join() L664
        +handle_ingest() L751
    }
    class PresenceSession {
        +send_challenge() L163
        +check_rate_limit() L175
        +handle_auth() L191
        +handle_pose_frame() L355
        +enforce_handshake_deadline() L395
        +heartbeat() L403
        +close_with() L413
        +close_with_code() L421
    }
    PresenceSession --> PresenceActor : JoinRoom / IngestPose / LeaveRoom
    PresenceActor --> PresenceActorInherent : impl blocks of the same type

    note for PresenceSession "Actix ws actor at presence_handler.rs:433, StreamHandler at :458"
    note for PresenceActorInherent "Not a separate type: the inherent impl PresenceActor blocks at presence_actor.rs:270, :663 and :750, while agent_presence_sequence (presence_actor.rs:264) is a field, configured_hand_reach_m (presence_actor.rs:45) a free fn"
    note for PresenceActor "Test-only CollectActor handles BroadcastFrame L882 and RoomEventEnvelope L889"
    note for PresenceSession "Crate is transport-agnostic (visionclaw-xr-presence/src/lib.rs:9-10): no transport assumed, each consumer injects the ports::Broadcaster and ports::IdentityVerifier traits"
    note for PresenceActor "PresenceActor implements ports::Broadcaster as a no-op (presence_actor.rs:647-652); the live fan-out is dispatch_broadcast"
    note for PresenceActor "Same crate is the single source of truth for the 0x43 wire format and is consumed by both this server (lib.rs:5-6) and the Godot client at xr-client/rust/ (lib.rs:6-7)"
```

**Debt:** the 0x44 agent co-presence channel (ADR-2020) is implemented inside `PresenceActor` but has no production ingress at the pin. `IngestAgentPresence`, `SweepStaleAgentPresence`, `GetAgentPresence` and `GetAttentionNode` are sent only from the test module (`src/actors/presence_actor.rs:1177`), and `PresenceActor::started` schedules only subscriber eviction (`src/actors/presence_actor.rs:634-640`), so a published agent state would never be swept. Closing it needs a server path that publishes agent state into a room and a periodic sweep.

**Tension:** `HANDSHAKE_TIMEOUT` is 10 s (`src/handlers/presence_handler.rs:34`) but is enforced only on the 15 s heartbeat interval (`src/handlers/presence_handler.rs:438-441`), so an unauthenticated socket survives roughly 15 s.

**Drift:** before this restamp the topic credited room bounds to `with_bounds`; production rooms are created with `PresenceActor::new` (`src/handlers/presence_handler.rs:285`) and keep the defaults at `src/actors/presence_actor.rs:284-285`.

**Open:** the XR client gained a Codescape backend mode (ADR-2143 A6) in which the same presence socket targets a Codescape server that "serves the graph stream, presence, voice" itself (`../project/docs/XR-client.md:1319`), so the `PresenceSession` and `PresenceActor` above are the VisionClaw backend only. See [VC-43](43-codescape-presence-lifecycle.md) for the separate Codescape lifecycle audit.

The [1 November admission review](../../estate-review/2026-11-01-room-admission-and-voice.md) traces the current join handler and proposed membership controls in [VC-49](49-room-admission.md). The historical nonce-authentication diagrams above do not imply membership admission or a room cap.
