---
title: WebSocket API Reference
description: Real-time WebSocket API for VisionFlow including binary protocol specification
category: reference
difficulty-level: intermediate
tags:
  - api
  - websocket
  - real-time
updated-date: 2025-01-29
---

# WebSocket API Reference

Real-time communication protocol for VisionFlow graph updates.

---

## Connection

### Establishing Connection

```typescript
// Binary protocol (recommended)
const ws = new WebSocket('ws://localhost:9090/ws?token=YOUR-JWT-TOKEN');
ws.binaryType = 'arraybuffer';

// Legacy JSON protocol (deprecated)
const wsJson = new WebSocket('ws://localhost:9090/ws?token=TOKEN&protocol=json');
```

### Connection Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `token` | string | Yes | JWT authentication token |
| `protocol` | string | No | Protocol version: `binary-v2` (default), `json` |

---

## Message Types

### Client to Server (JSON)

#### Subscribe to Position Updates

```json
{
  "type": "subscribe_position_updates",
  "data": {
    "rate": 60,
    "nodeFilter": "all",
    "protocol": "binary-v2"
  }
}
```

#### Filter Update

```json
{
  "type": "filter_update",
  "data": {
    "quality": 0.7,
    "maxNodes": 10000,
    "types": ["concept", "entity", "class"]
  }
}
```

#### Heartbeat (Keep-alive)

```json
{
  "type": "heartbeat",
  "timestamp": 1702915200000
}
```

### Server to Client (JSON)

#### Subscription Confirmed

```json
{
  "type": "subscription_confirmed",
  "data": {
    "rate": 60,
    "protocol": "binary-v2",
    "nodeCount": 50000
  }
}
```

#### State Sync

```json
{
  "type": "state_sync",
  "data": {
    "nodes": 50000,
    "edges": 120000,
    "graphType": "knowledge-graph",
    "timestamp": 1702915200000
  }
}
```

---

## Binary Protocol

### Protocol Versions

| Version | Status | Bytes/Node | Use Case |
|---------|--------|------------|----------|
| **V2** | **Current** | 36 | Production standard |
| V3 | Stable | 48 | Analytics extension |
| V4 | Experimental | 16 | Delta encoding |
| V1 | Deprecated | 34 | Legacy (ID limit: 16383) |

### Protocol V2 Wire Format

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

| Field | Type | Bytes | Range | Description |
|-------|------|-------|-------|-------------|
| **Version** | u8 | 1 | 2 | Protocol version |
| **Node ID** | u32 | 4 | 0-1,073,741,823 | Bits 0-29: ID, Bits 30-31: Type flags |
| **Position X/Y/Z** | f32 | 12 | -inf to +inf | 3D world coordinates |
| **Velocity X/Y/Z** | f32 | 12 | -inf to +inf | Physics velocity |
| **SSSP Distance** | f32 | 4 | 0.0 to +inf | Shortest path distance |
| **SSSP Parent** | i32 | 4 | -1 to max | Parent in path tree |

### Node Type Flags

Encoded in high bits of Node ID field:

```typescript
const AGENT_NODE_FLAG = 0x80000000;      // Bit 31
const KNOWLEDGE_NODE_FLAG = 0x40000000;  // Bit 30
const NODE_ID_MASK = 0x3FFFFFFF;         // Bits 0-29

// Decoding example
const nodeIdRaw = view.getUint32(offset, true); // Little-endian
const actualId = nodeIdRaw & NODE_ID_MASK;
const isAgent = (nodeIdRaw & AGENT_NODE_FLAG) !== 0;
const isKnowledge = (nodeIdRaw & KNOWLEDGE_NODE_FLAG) !== 0;
```

---

## Protocol V3 (Analytics Extension)

Extends V2 with 12 additional bytes for machine learning analytics.

**Additional Fields**:

| Field | Type | Offset | Description |
|-------|------|--------|-------------|
| **Cluster ID** | u32 | 37-40 | K-means cluster (0 = unassigned) |
| **Anomaly Score** | f32 | 41-44 | LOF score: 0.0-1.0 |
| **Community ID** | u32 | 45-48 | Louvain community |

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
```

---

## Performance Characteristics

### Bandwidth (100K nodes @ 60 FPS)

| Protocol | Message Size | Parse Time | Latency | vs JSON |
|----------|--------------|------------|---------|---------|
| **Binary V2** | 3.6 MB | 0.8 ms | 10 ms | **80% smaller** |
| Binary V3 | 4.8 MB | 1.1 ms | 13 ms | 73% smaller |
| Binary V4 (delta) | 0.7-1.4 MB | 0.2 ms | 5 ms | 92-96% smaller |
| JSON (deprecated) | 18 MB | 12 ms | 69 ms | Baseline |

### CPU Usage (100K nodes @ 60 FPS)

| Operation | Binary V2 | JSON V1 |
|-----------|-----------|---------|
| Server Encode | 1.2 ms | 15 ms |
| Client Decode | 0.8 ms | 12 ms |
| Server CPU | 5% | 28% |
| Client CPU | 3% | 18% |

---

## Client-Side WebSocket Management (February 2026)

### WebSocketEventBus

The client routes all WebSocket events through a typed event bus (`client/src/services/WebSocketEventBus.ts`). This replaces direct coupling between WebSocket handlers and consuming services.

**Event Categories:**
- `connection:open/close/error` -- Connection lifecycle
- `message:graph/voice/bots/pod` -- Typed message routing by service domain
- `registry:registered/unregistered/closedAll` -- Connection tracking

### WebSocketRegistry

All WebSocket connections register with a central registry (`client/src/services/WebSocketRegistry.ts`). This provides:
- Single point for connection health monitoring
- Coordinated shutdown via `closeAll()`
- Named connection lookup via `getConnection(name)`

**Registered connections:** Voice, Bots, SolidPod, Graph

### Settings Pipeline

Backend physics and quality-gate PUT handlers now accept partial JSON patches. The `useSelectiveSettingsStore` hook has been simplified from 548 to 152 lines, using native Zustand selectors instead of manual caching/TTL/debouncing. The quality-gate `maxNodeCount` default has been raised from 10,000 to 500,000.

---

## Related Documentation

- [Binary Protocol Specification](../protocols/binary-websocket.md)
- [REST API Reference](./rest-api.md)
- [Protocol Reference](../protocols/README.md)
