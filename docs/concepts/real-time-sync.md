---
title: Real-Time Synchronisation
description: Understanding VisionFlow's WebSocket binary protocol for high-performance real-time graph streaming
category: explanation
tags:
  - websocket
  - binary-protocol
  - real-time
  - performance
related-docs:
  - concepts/actor-model.md
  - reference/protocols/binary-websocket.md
  - reference/websocket-protocol.md
updated-date: 2025-12-18
difficulty-level: intermediate
---

# Real-Time Synchronisation

VisionFlow achieves 60 FPS real-time graph updates via a custom binary WebSocket protocol that reduces bandwidth by 80% compared to JSON.

---

## Core Concept

Real-time graph visualisation requires streaming position updates for potentially 100,000+ nodes at 60 FPS. JSON serialisation would consume:

```
100K nodes x 150 bytes/node x 60 FPS = 900 MB/s
```

The binary protocol achieves:

```
100K nodes x 36 bytes/node x 60 FPS = 216 MB/s
```

Combined with adaptive frame rates and delta encoding, actual bandwidth drops to 5-50 MB/s.

---

## Protocol Architecture

### Hybrid Protocol

VisionFlow uses hybrid JSON + binary messaging:

| Type | Format | Usage |
|------|--------|-------|
| Control Messages | JSON | Authentication, filters, state sync |
| Initial Graph | JSON | Nodes + edges + metadata on connect |
| Position Updates | Binary | Real-time streaming (V2/V3) |
| Position Snapshots | Binary | Full state sync on request |

### Protocol Versions

| Version | Bytes/Node | Status | Use Case |
|---------|------------|--------|----------|
| V1 | 34 | DEPRECATED | Legacy (node ID truncation bug) |
| V2 | 36 | **CURRENT** | Production, full u32 node IDs |
| V3 | 48 | STABLE | Analytics extension |
| V4 | 16 | EXPERIMENTAL | Delta encoding (60-80% reduction) |

---

## Binary Protocol V2 (Current)

### Wire Format: 36 Bytes Per Node

```
┌─────────────────────────────────────────────────────────────┐
│              Binary Node Data (36 bytes)                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  [0-3]   Node ID (u32 with type flags in high bits)         │
│  [4-7]   Position X (f32, IEEE 754)                         │
│  [8-11]  Position Y (f32)                                   │
│  [12-15] Position Z (f32)                                   │
│  [16-19] Velocity X (f32)                                   │
│  [20-23] Velocity Y (f32)                                   │
│  [24-27] Velocity Z (f32)                                   │
│  [28-31] SSSP Distance (f32)                                │
│  [32-35] SSSP Parent (i32)                                  │
│                                                              │
│  Total message: 1 byte (version) + N x 36 bytes             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Node Type Flags

Node types are encoded in the high bits of the u32 ID:

```rust
const AGENT_NODE_FLAG: u32 = 0x80000000;      // Bit 31
const KNOWLEDGE_NODE_FLAG: u32 = 0x40000000;  // Bit 30
const NODE_ID_MASK: u32 = 0x3FFFFFFF;         // Bits 0-29 (1B max)
```

**Client decoding**:

```typescript
const nodeIdRaw = view.getUint32(offset, true);
const actualId = nodeIdRaw & 0x3FFFFFFF;
const isAgent = (nodeIdRaw & 0x80000000) !== 0;
const isKnowledge = (nodeIdRaw & 0x40000000) !== 0;
```

---

## Dual-Graph Broadcasting

VisionFlow unifies knowledge graph and agent graph into a single WebSocket stream.

### Graph Types

| Graph | Typical Nodes | Flag | Content |
|-------|--------------|------|---------|
| Knowledge Graph | ~185 | Bit 30 | Scientific concepts, papers |
| Agent Graph | ~3-10 | Bit 31 | AI agents (Claude via MCP) |

### Unified Broadcast Flow

```
┌─────────────────────────────────────────────────────────────┐
│                Unified Broadcast Pipeline                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Physics Loop (60 FPS)                                      │
│       ↓                                                      │
│  Graph Actor                                                │
│       ↓                                                      │
│  Collect knowledge nodes (flag: 0x40000000)                 │
│  Collect agent nodes (flag: 0x80000000)                     │
│       ↓                                                      │
│  Binary Encoder                                             │
│  encode_node_data_with_types()                              │
│       ↓                                                      │
│  Single broadcast (188 nodes: 185 knowledge + 3 agents)     │
│       ↓                                                      │
│  All WebSocket clients                                      │
│       ↓                                                      │
│  Client-side separation by flag bits                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Benefits

1. **No broadcast conflicts**: Single source of truth
2. **Synchronised updates**: Both graphs update together
3. **Efficient bandwidth**: One message, not two streams
4. **Type safety**: Flags ensure proper separation

---

## Connection Lifecycle

```
┌─────────────────────────────────────────────────────────────┐
│               WebSocket Connection Lifecycle                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Connecting                                                  │
│       ↓                                                      │
│  Connected (handshake success)                              │
│       ↓                                                      │
│  Authenticating (JWT token)                                 │
│       ↓                                                      │
│  Authenticated                                               │
│       ↓                                                      │
│  Active                                                      │
│  ├── Receive: Binary position updates (throttled)          │
│  ├── Send: Filter updates (JSON)                           │
│  └── Heartbeat: Ping/pong every 30s                        │
│       ↓                                                      │
│  Reconnecting (on connection loss)                          │
│       ↓                                                      │
│  Closed (graceful shutdown)                                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Adaptive Broadcast Rate

### Frame Rate Modes

| Mode | Interval | Trigger |
|------|----------|---------|
| Active | 16.7 ms (60 FPS) | Movement detected |
| Settling | 50 ms (20 FPS) | Low velocity |
| Settled | 200 ms (5 FPS) | Near-zero kinetic energy |

### Energy-Based Detection

```rust
fn determine_broadcast_mode(&self) -> BroadcastMode {
    let kinetic_energy = self.calculate_kinetic_energy();
    let max_velocity = self.get_max_velocity();

    if max_velocity > 1.0 {
        BroadcastMode::Active
    } else if kinetic_energy > 0.1 {
        BroadcastMode::Settling
    } else {
        BroadcastMode::Settled
    }
}
```

---

## Bandwidth Reduction Techniques

### 1. Binary Protocol (76% reduction)

| Format | Per Node | 50 Nodes |
|--------|----------|----------|
| JSON | ~150 bytes | 7.5 KB |
| Binary V2 | 36 bytes | 1.8 KB |

### 2. Adaptive Frame Rate (60-90% reduction)

| State | Bandwidth |
|-------|-----------|
| Active (60 FPS) | 100% |
| Settling (20 FPS) | 33% |
| Settled (5 FPS) | 8% |

### 3. Delta Encoding (V4, experimental)

Only send changed positions:

| Scenario | Bandwidth |
|----------|-----------|
| 100% nodes moving | 100% |
| 10% nodes moving | 10% |
| Static graph | ~0% (periodic sync) |

### Combined Impact

For a 10K node graph:

| Technique | Bandwidth |
|-----------|-----------|
| JSON @ 60 FPS | 900 MB/s |
| Binary V2 @ 60 FPS | 216 MB/s |
| Binary V2 @ adaptive | 50-100 MB/s |
| Binary V4 delta | 5-20 MB/s |

---

## Client Implementation

### TypeScript Decoder

```typescript
class BinaryProtocolClient {
    private parseNodeUpdate(buffer: ArrayBuffer): NodeData[] {
        const view = new DataView(buffer);
        const version = view.getUint8(0);

        if (version !== 2) throw new Error(`Unsupported version: ${version}`);

        const nodes: NodeData[] = [];
        const BINARY_NODE_SIZE = 36;
        let offset = 1; // Skip version byte

        while (offset < buffer.byteLength) {
            // Node ID with type flags
            const rawId = view.getUint32(offset, true);
            const actualId = rawId & 0x3FFFFFFF;
            const isAgent = (rawId & 0x80000000) !== 0;
            const isKnowledge = (rawId & 0x40000000) !== 0;
            offset += 4;

            // Position
            const x = view.getFloat32(offset, true); offset += 4;
            const y = view.getFloat32(offset, true); offset += 4;
            const z = view.getFloat32(offset, true); offset += 4;

            // Velocity
            const vx = view.getFloat32(offset, true); offset += 4;
            const vy = view.getFloat32(offset, true); offset += 4;
            const vz = view.getFloat32(offset, true); offset += 4;

            // SSSP
            const ssspDistance = view.getFloat32(offset, true); offset += 4;
            const ssspParent = view.getInt32(offset, true); offset += 4;

            nodes.push({
                id: actualId,
                nodeType: isAgent ? 'agent' : isKnowledge ? 'knowledge' : 'normal',
                position: { x, y, z },
                velocity: { vx, vy, vz },
                ssspDistance,
                ssspParent
            });
        }

        return nodes;
    }
}
```

---

## Server Implementation

### Rust Encoder

```rust
pub fn encode_node_data_with_types(
    nodes: &[(u32, BinaryNodeData)],
    agent_ids: &[u32],
    knowledge_ids: &[u32]
) -> Vec<u8> {
    let mut buffer = Vec::with_capacity(1 + nodes.len() * 36);

    // Version byte
    buffer.push(2u8);

    for (node_id, data) in nodes {
        // Apply type flags
        let wire_id = if agent_ids.contains(node_id) {
            node_id | 0x80000000
        } else if knowledge_ids.contains(node_id) {
            node_id | 0x40000000
        } else {
            *node_id
        };

        // Encode node data (36 bytes)
        buffer.extend_from_slice(&wire_id.to_le_bytes());
        buffer.extend_from_slice(&data.x.to_le_bytes());
        buffer.extend_from_slice(&data.y.to_le_bytes());
        buffer.extend_from_slice(&data.z.to_le_bytes());
        buffer.extend_from_slice(&data.vx.to_le_bytes());
        buffer.extend_from_slice(&data.vy.to_le_bytes());
        buffer.extend_from_slice(&data.vz.to_le_bytes());
        buffer.extend_from_slice(&data.sssp_distance.to_le_bytes());
        buffer.extend_from_slice(&data.sssp_parent.to_le_bytes());
    }

    buffer
}
```

---

## Heartbeat and Reconnection

### Heartbeat Protocol

- **Ping interval**: 30 seconds
- **Pong timeout**: 45 seconds
- **On timeout**: Mark connection dead, attempt reconnect

### Reconnection Strategy

```rust
fn calculate_backoff(attempt: u32) -> Duration {
    let base_delay = 1000;  // 1 second
    let exponential = base_delay * 2u64.pow(attempt - 1);
    let max_delay = 30000;  // 30 seconds

    Duration::from_millis(exponential.min(max_delay))
}
```

Backoff sequence: 1s, 2s, 4s, 8s, 16s, 30s, 30s, ...

---

## Performance Characteristics

### Latency

| Operation | P50 | P95 | P99 |
|-----------|-----|-----|-----|
| Serialisation | 0.2 ms | 0.5 ms | 1.0 ms |
| Deserialisation | 0.3 ms | 0.7 ms | 1.2 ms |
| Network RTT | 8 ms | 18 ms | 30 ms |
| End-to-end | 10 ms | 20 ms | 35 ms |

### Throughput

- Messages/second: 300 sustained, 600 burst
- Bytes/second: 500 KB sustained, 1 MB burst
- Nodes/update: Up to 100K

---

## Error Handling

### Error Codes

| Code | Name | Recovery |
|------|------|----------|
| 0x0001 | ParseError | Log and discard |
| 0x0002 | AuthFailure | Reconnect with auth |
| 0x0003 | RateLimitExceeded | Backoff and retry |
| 0x0004 | InvalidState | Reset connection |
| 0x0005 | InternalError | Retry with backoff |

### Client-Side Validation

```typescript
if (buffer.byteLength < 1) {
    throw new Error('Empty binary message');
}

const version = view.getUint8(0);
const expectedSize = version === 2 ? 36 : 48;
const payloadSize = buffer.byteLength - 1;

if (payloadSize % expectedSize !== 0) {
    console.error(`Invalid message: ${payloadSize} bytes`);
    return;
}
```

---

## Related Concepts

- **[Actor Model](actor-model.md)**: ClientCoordinatorActor WebSocket handling
- **[GPU Acceleration](gpu-acceleration.md)**: Physics updates driving broadcasts
- **[Multi-Agent System](multi-agent-system.md)**: Agent graph integration
