---
title: WebSocket Binary Protocol Specification
description: Complete binary WebSocket protocol specification for VisionFlow real-time graph updates, including V1-V4 versions, XR collaboration, and performance characteristics.
category: reference
tags:
  - api
  - websocket
  - protocol
  - binary
  - real-time
  - visionflow
updated-date: 2025-01-29
difficulty-level: advanced
---

# WebSocket Binary Protocol Specification

**Protocol Version**: 2.0 (Current Standard)
**Last Updated**: January 29, 2025
**Status**: Production

---

## Overview

VisionFlow uses a **hybrid JSON + Binary protocol** for WebSocket communication:
- **JSON messages**: Control flow, authentication, initial loads, metadata
- **Binary messages**: High-frequency position streaming (V1/V2/V3/V4)

The binary protocol achieves **~80% bandwidth reduction** compared to pure JSON for large graphs while maintaining sub-10ms latency.

---

## Protocol Versions

| Version | Bytes/Node | Status | Use Case |
|---------|------------|--------|----------|
| **V1** | 34 | DEPRECATED | Legacy clients, node IDs <= 16383 |
| **V2** | 36 | **CURRENT** | Production, full u32 node IDs |
| **V3** | 48 | STABLE | Analytics extension (clustering, anomaly) |
| **V4** | 16 | EXPERIMENTAL | Delta encoding (60-80% bandwidth reduction) |

### Version Detection

Server sends protocol version as first byte of binary messages:

```
[0] = Protocol Version (u8)
[1..N] = Payload (version-specific)
```

---

## Protocol V2 (Current Standard)

### Wire Format: 36 Bytes Per Node

**Total Message Size**: 1 + (36 x node_count) bytes

```
Byte Layout (Little-Endian):
+----------+--------------------------------------------+
| Offset   | Field (Type, Bytes)                        |
+----------+--------------------------------------------+
| [0]      | Protocol Version (u8) = 2                  |
+----------+--------------------------------------------+
| [1-4]    | Node ID (u32) with type flags              |
| [5-8]    | Position X (f32)                           |
| [9-12]   | Position Y (f32)                           |
| [13-16]  | Position Z (f32)                           |
| [17-20]  | Velocity X (f32)                           |
| [21-24]  | Velocity Y (f32)                           |
| [25-28]  | Velocity Z (f32)                           |
| [29-32]  | SSSP Distance (f32)                        |
| [33-36]  | SSSP Parent (i32)                          |
+----------+--------------------------------------------+
```

### Field Specifications

| Field | Type | Bytes | Endianness | Description |
|-------|------|-------|------------|-------------|
| **Node ID** | u32 | 4 | Little | Bits 0-29: ID (0 to 1,073,741,823), Bits 30-31: Type flags |
| **Position X/Y/Z** | f32 | 12 | Little | 3D world coordinates (arbitrary units) |
| **Velocity X/Y/Z** | f32 | 12 | Little | Physics velocity (units/sec) |
| **SSSP Distance** | f32 | 4 | Little | Single-source shortest path distance (default: `f32::INFINITY`) |
| **SSSP Parent** | i32 | 4 | Little | Parent node in SSSP tree (default: `-1`) |

### Node Type Flags (u32 ID Field)

Server encodes node types in the high bits of the node ID:

```rust
// Flag constants (bits 30-31)
const AGENT_NODE_FLAG: u32 = 0x80000000;      // Bit 31
const KNOWLEDGE_NODE_FLAG: u32 = 0x40000000;  // Bit 30
const NODE_ID_MASK: u32 = 0x3FFFFFFF;         // Bits 0-29

// Ontology type flags (bits 26-28, only for GraphType::Ontology)
const ONTOLOGY_CLASS_FLAG: u32 = 0x04000000;      // Bit 26
const ONTOLOGY_INDIVIDUAL_FLAG: u32 = 0x08000000; // Bit 27
const ONTOLOGY_PROPERTY_FLAG: u32 = 0x10000000;   // Bit 28
```

**Client decoding example (TypeScript)**:
```typescript
const nodeIdRaw = view.getUint32(offset, true); // Little-endian
const actualId = nodeIdRaw & 0x3FFFFFFF;
const isAgent = (nodeIdRaw & 0x80000000) !== 0;
const isKnowledge = (nodeIdRaw & 0x40000000) !== 0;
```

---

## Protocol V1 (Legacy, Deprecated)

### Wire Format: 34 Bytes Per Node

**BUG**: Only supports node IDs 0-16383 (14 bits). IDs > 16383 get truncated!

- Protocol Version (u8) = 1 at offset [0]
- Node ID (u16) with type flags at [1-2]
- Position X/Y/Z (3xf32) at [3-14]
- Velocity X/Y/Z (3xf32) at [15-26]
- SSSP Distance (f32) at [27-30]
- SSSP Parent (i32) at [31-34]

**Migration Note**: V1 is automatically used only when all node IDs <= 16383. Otherwise, server upgrades to V2.

---

## Protocol V3 (Analytics Extension)

### Wire Format: 48 Bytes Per Node

Extends V2 with machine learning analytics fields:

- Protocol Version (u8) = 3 at offset [0]
- V2 Fields (Node ID, Pos, Vel, SSSP) at [1-36]
- Cluster ID (u32, K-means) at [37-40]
- Anomaly Score (f32, LOF 0.0-1.0) at [41-44]
- Community ID (u32, Louvain) at [45-48]

**Additional Fields**:

| Field | Type | Description |
|-------|------|-------------|
| **Cluster ID** | u32 | K-means cluster assignment (0 = unassigned) |
| **Anomaly Score** | f32 | LOF (Local Outlier Factor) score: 0.0 = normal, 1.0 = anomaly |
| **Community ID** | u32 | Louvain community detection (0 = unassigned) |

---

## Protocol V4 (Delta Encoding - Experimental)

### Motivation

Full state updates send redundant data for static nodes. Delta encoding achieves **60-80% bandwidth reduction** by only sending changes.

### Wire Format: 16 Bytes Per Changed Node

```
+----------+--------------------------------------------+
| Offset   | Field (Type, Bytes)                        |
+----------+--------------------------------------------+
| [0]      | Protocol Version (u8) = 4                  | Header
+----------+--------------------------------------------+
| [1-4]    | Node ID (u32) with type flags             | Per Change
| [5]      | Change Flags (u8, bit field)              | (16 bytes)
| [6-8]    | Padding (3 bytes, reserved)               |
| [9-10]   | Delta Position X (i16, scaled)            |
| [11-12]  | Delta Position Y (i16, scaled)            |
| [13-14]  | Delta Position Z (i16, scaled)            |
| [15-16]  | Delta Velocity X (i16, scaled)            |
| [17-18]  | Delta Velocity Y (i16, scaled)            |
| [19-20]  | Delta Velocity Z (i16, scaled)            |
+----------+--------------------------------------------+
```

### Delta Encoding Details

**Change Flags (bit field)**:
```rust
const DELTA_POSITION_CHANGED: u8 = 0x01;
const DELTA_VELOCITY_CHANGED: u8 = 0x02;
```

**Scale Factor**: 100.0 (converts f32 to i16 with 0.01 precision)

**Resync Interval**: Frame 0 and every 60 frames send full V2 state

---

## XR Collaboration Protocol

VisionFlow uses a custom binary WebSocket protocol optimized for real-time XR collaboration, semantic graph synchronization, and low-latency node updates. The protocol achieves 36 bytes per node update at 90 Hz, enabling smooth multi-user immersive experiences.

### Connection Management

#### Handshake

**Client to Server**:
```
MESSAGE-TYPE: 0x00 (HELLO)
PROTOCOL-VERSION: u32 (current: 1)
CLIENT-ID: UUID (128 bits)
CAPABILITIES: u32 (bitmask)
  bit 0: hand-tracking
  bit 1: eye-tracking
  bit 2: voice-enabled
  bit 3: ar-supported
  bit 4: vr-supported
PLATFORM: u8
  0: WebXR
  1: Meta Quest
  2: Apple Vision Pro
  3: SteamVR
  4: Desktop/Fallback

Total: 26 bytes
```

**Server to Client**:
```
MESSAGE-TYPE: 0x01 (WELCOME)
SESSION-ID: UUID (128 bits)
WORLD-ID: UUID (128 bits)
PROTOCOL-VERSION: u32
CAPABILITY-FLAGS: u32 (server capabilities)
TIMESTAMP: u64 (server time in milliseconds)
STATE-SNAPSHOT-SIZE: u32
[STATE-SNAPSHOT] (variable, gzip compressed)
```

#### Connection Keepalive

**Heartbeat (bidirectional, 30-second interval)**:
```
MESSAGE-TYPE: 0x02 (PING)
TIMESTAMP: u64
SEQUENCE: u32

Response:
MESSAGE-TYPE: 0x03 (PONG)
TIMESTAMP: u64
SEQUENCE: u32

Total: 13 bytes
```

### Message Frame Structure

#### Header (fixed 8 bytes)

```
+------------------+------------------+------------+-------------+
| Message Type     | User ID          | Timestamp  | Data Length |
| (1 byte)         | (4 bytes)        | (4 bytes)  | (2 bytes)   |
+------------------+------------------+------------+-------------+
| u8               | u32 (hash)       | u32 (delta)| u16         |
+------------------+------------------+------------+-------------+
| Payload (variable, up to 512 bytes)                            |
+----------------------------------------------------------------+
```

### Message Types

#### 0x01-0x0F: Control Messages

| Type | Name | Purpose | Response |
|------|------|---------|----------|
| 0x01 | WELCOME | Server greeting + snapshot | None |
| 0x02 | PING | Connection check | PONG (0x03) |
| 0x03 | PONG | Ping response | None |
| 0x04 | SYNC-REQUEST | Request full sync | SYNC-RESPONSE |
| 0x05 | SYNC-RESPONSE | Full world state | None |

#### 0x10-0x1F: Presence & Avatar

| Type | Name | Purpose | Frequency |
|------|------|---------|-----------|
| 0x10 | POSE-UPDATE | User head/hand transforms | 90 Hz |
| 0x11 | AVATAR-STATE | Avatar appearance/status | On change |
| 0x12 | USER-JOIN | New user entered space | On event |
| 0x13 | USER-LEAVE | User left space | On event |
| 0x14 | VOICE-DATA | Audio stream | ~50 Hz (16kHz mono) |

#### 0x20-0x2F: Interaction

| Type | Name | Purpose | Frequency |
|------|------|---------|-----------|
| 0x20 | GESTURE-EVENT | Hand gesture recognized | On gesture |
| 0x21 | VOICE-COMMAND | Voice command | On speech |
| 0x22 | OBJECT-SELECT | Object interaction | On action |
| 0x23 | OBJECT-GRAB | Object grabbed | On action |
| 0x24 | OBJECT-RELEASE | Object released | On action |

#### 0x30-0x3F: Graph Updates

| Type | Name | Purpose | Frequency |
|------|------|---------|-----------|
| 0x30 | NODE-CREATE | New ontology node | On creation |
| 0x31 | NODE-UPDATE | Update node properties | On change |
| 0x32 | NODE-DELETE | Remove node | On deletion |
| 0x33 | EDGE-CREATE | New relationship | On creation |
| 0x34 | EDGE-DELETE | Remove relationship | On deletion |
| 0x35 | CONSTRAINT-APPLY | Physics constraint | On change |

#### 0x40-0x4F: Agent Actions

| Type | Name | Purpose | Frequency |
|------|------|---------|-----------|
| 0x40 | AGENT-ACTION | Agent-initiated action | On action |
| 0x41 | AGENT-RESPONSE | Agent response data | On response |
| 0x42 | AGENT-STATUS | Agent status update | 1 Hz |

#### 0x50-0x5F: Errors & Acknowledgments

| Type | Name | Purpose | Frequency |
|------|------|---------|-----------|
| 0x50 | ERROR | Error notification | On error |
| 0x51 | ACK | Message acknowledgment | On receipt |
| 0x52 | NACK | Negative acknowledgment | On reject |

### Payload Specifications

#### POSE-UPDATE (0x10) - 36 bytes

Optimized transform update for user pose (head + hands):

```
+---------------+---------------+---------------+---------------+
| Position X    | Position Y    | Position Z    | Rotation X    |
| float16 (2)   | float16 (2)   | float16 (2)   | float16 (2)   |
+---------------+---------------+---------------+---------------+
| Rotation Y    | Rotation Z    | Rotation W    | Hand State    |
| float16 (2)   | float16 (2)   | float16 (2)   | u16 (2)       |
+---------------+---------------+---------------+---------------+
| Velocity (velocity estimation for smooth interpolation)       |
| float16 x3 (6 bytes) = [vx, vy, vz]                          |
+---------------------------------------------------------------+
| Hand State: 16-bit packed                                     |
|  Left Hand: 4 bits (open, pinch, point, fist)                |
|  Right Hand: 4 bits (open, pinch, point, fist)               |
|  Head Rotation Confidence: 4 bits (0-15)                     |
|  Tracking State: 4 bits (calibrated, tracking, lost, etc)    |
+---------------------------------------------------------------+

Total: 8 + 8 + 12 + 2 + 6 = 36 bytes (efficient!)
```

#### VOICE-DATA (0x14) - 160 bytes per frame

Opus-encoded audio at 16kHz mono:

```
+---------------+---------------+------------------------------+
| Sequence      | Frame Type    | Opus Payload                 |
| u16           | u8            | (variable, ~160 bytes)       |
+---------------+---------------+------------------------------+
| Frame Types:  |               |                              |
| 0: speech     | 1: noise      | 2: silence                   |
| 3: end-frame  |               |                              |
+---------------+---------------+------------------------------+

Total: ~160 bytes at 50 fps (20 ms frames) = 8 KB/s per user
```

---

## Client Implementation

### TypeScript Example

```typescript
class BinaryProtocolParser {
  private view: DataView;
  private offset: number = 1; // Skip version byte

  constructor(buffer: ArrayBuffer) {
    this.view = new DataView(buffer);
  }

  parseNodeUpdates(): NodeUpdate[] {
    const version = this.view.getUint8(0);
    const bytesPerNode = version === 3 ? 48 : 36;
    const nodeCount = Math.floor((this.view.byteLength - 1) / bytesPerNode);
    const updates: NodeUpdate[] = [];

    for (let i = 0; i < nodeCount; i++) {
      const offset = 1 + (i * bytesPerNode);
      const nodeIdRaw = this.view.getUint32(offset, true);

      updates.push({
        id: nodeIdRaw & 0x3FFFFFFF,
        isAgent: (nodeIdRaw & 0x80000000) !== 0,
        isKnowledge: (nodeIdRaw & 0x40000000) !== 0,
        x: this.view.getFloat32(offset + 4, true),
        y: this.view.getFloat32(offset + 8, true),
        z: this.view.getFloat32(offset + 12, true),
        vx: this.view.getFloat32(offset + 16, true),
        vy: this.view.getFloat32(offset + 20, true),
        vz: this.view.getFloat32(offset + 24, true),
        ssspDistance: this.view.getFloat32(offset + 28, true),
        ssspParent: this.view.getInt32(offset + 32, true)
      });
    }

    return updates;
  }
}

// Usage
ws.binaryType = 'arraybuffer';
ws.onmessage = (event) => {
  if (event.data instanceof ArrayBuffer) {
    const parser = new BinaryProtocolParser(event.data);
    const updates = parser.parseNodeUpdates();
    updates.forEach(node => {
      updateNodePosition(node.id, [node.x, node.y, node.z]);
    });
  }
};
```

### Rust Server Implementation

```rust
use crate::utils::binary_protocol;

// Encode nodes with type flags
let nodes: Vec<(u32, BinaryNodeData)> = vec![
    (1, BinaryNodeData { node_id: 1, x: 0.0, y: 0.0, z: 0.0, vx: 0.0, vy: 0.0, vz: 0.0 }),
    (2, BinaryNodeData { node_id: 2, x: 1.0, y: 1.0, z: 1.0, vx: 0.0, vy: 0.0, vz: 0.0 }),
];
let agent_ids = vec![2]; // Node 2 is an agent
let knowledge_ids = vec![1]; // Node 1 is knowledge

let binary_data = binary_protocol::encode_node_data_with_types(
    &nodes,
    &agent_ids,
    &knowledge_ids
);

// Send via WebSocket
ctx.binary(binary_data);
```

---

## Performance Characteristics

### Bandwidth Comparison (100K nodes)

| Protocol | Message Size | vs JSON | Latency (1Gbps) |
|----------|--------------|---------|-----------------|
| JSON | 18 MB | - | 144 ms |
| Binary V1 | 3.4 MB | 81% smaller | 27 ms |
| Binary V2 | 3.6 MB | 80% smaller | 29 ms |
| Binary V3 | 4.8 MB | 73% smaller | 38 ms |
| Binary V4 (delta) | 0.7-1.4 MB | 92-96% smaller | 5-11 ms |

### CPU Overhead (per frame, 100K nodes)

| Operation | V2 Time | V4 Delta Time |
|-----------|---------|---------------|
| Server Encode | 1.2 ms | 3.5 ms (first frame), 0.4 ms (delta) |
| Client Decode | 0.8 ms | 0.2 ms (delta) |

### Per-User Bandwidth (XR Collaboration)

| Content | Message Type | Frequency | Bandwidth |
|---------|--------------|-----------|-----------|
| **Pose** | POSE-UPDATE | 90 Hz | 36 bytes x 90 = 3.24 KB/s |
| **Voice** | VOICE-DATA | 50 Hz (20ms frames) | ~160 bytes x 50 = 8 KB/s |
| **Gestures** | GESTURE-EVENT | ~5-10 per sec | ~50 bytes x 10 = 500 B/s |
| **Graph** | NODE-UPDATE | Variable | ~1-10 KB/s |
| **Overhead** | Headers + keepalive | Constant | ~1 KB/s |
| **TOTAL** | - | - | **~13-15 KB/s per user** |

### Scaling Example

- **10 concurrent users**: 130-150 KB/s (1 Mb/s bandwidth)
- **100 concurrent users**: 1.3-1.5 MB/s (10 Mb/s bandwidth)
- **1000 concurrent users**: 13-15 MB/s (100 Mb/s bandwidth)

---

## Compression & Delta Encoding

### Transform Delta Encoding

```typescript
// Only send changed fields
class DeltaPose {
  flags: u8;  // Bitmask of which fields changed
  // bit 0: position changed
  // bit 1: rotation changed
  // bit 2: velocity changed
  // bits 3-7: reserved

  payload: Buffer;  // Only includes changed fields

  // If position changed: 6 bytes (3x float16)
  // If rotation changed: 8 bytes (4x float16)
  // If velocity changed: 6 bytes (3x float16)

  // Example: position + rotation = 1 + 6 + 8 = 15 bytes
  // vs full update = 36 bytes (58% reduction!)
}
```

### Graph Delta Compression

Uses gzip for graph updates:

```typescript
// On server
const graphDelta = computeChanges(previousState, currentState);
const compressed = gzip(serialize(graphDelta));

// Threshold: send full state if delta > 80% of full size
if (compressed.length > fullState.length * 0.8) {
  sendFullState();
} else {
  sendDeltaUpdate(compressed);
}
```

---

## Error Handling

### Message Validation

```typescript
if (buffer.byteLength < 1) {
    throw new Error('Empty binary message');
}

const version = view.getUint8(0);
const expectedSize = version === 2 ? 36 : version === 3 ? 48 : 34;
const payloadSize = buffer.byteLength - 1;

if (payloadSize % expectedSize !== 0) {
    console.error(`Invalid V${version} message: ${payloadSize} bytes`);
    return;
}
```

### Malformed Data

```typescript
if (!isFinite(position.x) || !isFinite(position.y) || !isFinite(position.z)) {
    console.warn(`Node ${actualId} has invalid position: NaN or Infinity`);
    return; // Skip this node
}
```

---

## Conflict Resolution

### Last-Write-Wins (LWW)

For concurrent edits:

```typescript
User1: NODE-UPDATE { id: 'node-1', value: 100, timestamp: 1000 }
User2: NODE-UPDATE { id: 'node-1', value: 200, timestamp: 1001 }

// Server resolves with later timestamp
Result: value = 200 (User2 wins)

// User1 receives NACK + corrected value
Server: NACK { reason: "concurrent-edit", correctValue: 200 }
```

---

## Security Considerations

### Message Validation

- All messages validated against schema
- Payload sizes capped at 512 bytes
- User IDs verified against authentication context

### Data Encryption

- All traffic uses WSS (WebSocket Secure = TLS)
- Sensitive data (voice, positioning) encrypted end-to-end
- Eye gaze data encrypted per-frame

### Rate Limiting

```typescript
const LIMITS = {
  POSE_UPDATE: 100,    // max per second
  NODE_UPDATE: 10,
  GESTURE_EVENT: 20,
  VOICE_DATA: 60
};
```

---

## Version Negotiation

Server automatically selects protocol version based on graph characteristics:

```rust
pub fn needs_v2_protocol(nodes: &[(u32, BinaryNodeData)]) -> bool {
    nodes.iter().any(|(node_id, _)| {
        let actual_id = get_actual_node_id(*node_id);
        actual_id > 0x3FFF // 16383
    })
}
```

**Decision logic**:
1. If any node ID > 16383 -> **V2 required**
2. If analytics requested -> **V3**
3. If delta encoding enabled -> **V4**
4. Otherwise -> **V2 (default)**

---

## Known Limitations

### Protocol V1
- Node IDs > 16383 get truncated (critical bug)
- Only supports 16,384 unique nodes per graph type
- **Recommendation**: Migrate to V2 immediately

### Protocol V2
- Supports 1,073,741,823 unique node IDs
- Production-stable since Nov 2025
- No compression (use `permessage-deflate` for 2-3x savings)

### Protocol V3
- Analytics fields populated by backend ML pipeline
- Requires clustering/anomaly detection modules enabled
- 33% larger than V2 (trade-off for richer data)

### Protocol V4
- Experimental (not production-ready)
- Requires client state tracking (complex)
- Resync every 60 frames adds latency spikes

---

## Performance Tuning

### Recommended Settings

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Max Connections** | 1000 | Per server instance |
| **Pose Update Rate** | 90 Hz | Match HMD refresh rate |
| **Heartbeat Interval** | 30 sec | Keep-alive |
| **Max Message Size** | 512 B | Prevents flooding |
| **Compression** | gzip | For graph updates |
| **Voice Codec** | Opus 16kHz | High quality, low latency |
| **Buffer Size** | 64 KB | Per-connection |

---

## Related Documentation

- [WebSocket Endpoints](../api/websocket-endpoints.md)
- [REST API Reference](../api/rest-api.md)
- [Database Schema Reference](../database/schema-catalog.md)
- [Error Codes Reference](../error-codes.md)

---

## References

- **Server Code**: `src/utils/binary_protocol.rs`
- **Client Code**: `client/src/services/BinaryWebSocketProtocol.ts`
- **WebSocket Handler**: `src/handlers/socket_flow_handler.rs`
- **Analytics Pipeline**: `src/services/analytics_service.rs` (V3 fields)

---

**Specification Version**: 3.0
**Last Verified**: January 29, 2025
**Implementation**: VisionFlow Server v0.1.0+
**Maintainer**: VisionFlow Core Team
