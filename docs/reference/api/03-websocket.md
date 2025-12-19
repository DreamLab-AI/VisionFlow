---
title: WebSocket Protocol
description: **Version**: 2.0 (Binary Protocol) **Last Updated**: November 3, 2025 **Status**: Production
category: reference
tags:
  - api
  - api
  - api
  - backend
  - frontend
updated-date: 2025-12-18
difficulty-level: advanced
---


# WebSocket Protocol

**Version**: 2.0 (Binary Protocol)
**Last Updated**: November 3, 2025
**Status**: Production

---

## Table of Contents

1. [Overview](#overview)
2. [Binary Protocol V2 (Current)](#binary-protocol-v2-current)
3. 
4. [Connection](#connection)
5. [Client Implementation](#client-implementation)
6. [Performance](#performance)

---

## Overview

VisionFlow uses a **36-byte binary WebSocket protocol** for real-time graph updates, achieving:

- **80% bandwidth reduction** vs. JSON
- **Sub-10ms latency** for node updates
- **60 FPS** streaming at 100k+ nodes
- **Zero parsing overhead** (direct TypedArray access)

### Protocol Versions

| Version | Status | Use Case | Bandwidth (100k nodes) |
|---------|--------|----------|------------------------|
| **Binary V2** | ✅ Current | Real-time physics updates | 3.6 MB/frame |
| JSON V1 | ⚠️ Deprecated | Legacy compatibility | 18 MB/frame |

---

## Binary Protocol V2 (Current)

### Message Format

Each node update is **exactly 36 bytes**:

```
Byte Layout (Little-Endian):
┌──────────┬───────────────────────────────────────────┐
│ Offset   │ Field                                     │
├──────────┼───────────────────────────────────────────┤
│ [0-3]    │ Node ID (u32)                             │
│ [4-7]    │ X position (f32)                          │
│ [8-11]   │ Y position (f32)                          │
│ [12-15]  │ Z position (f32)                          │
│ [16-19]  │ VX velocity (f32)                         │
│ [20-23]  │ VY velocity (f32)                         │
│ [24-27]  │ VZ velocity (f32)                         │
│ [28-31]  │ Mass (f32)                                │
│ [32-35]  │ Charge (f32)                              │
└──────────┴───────────────────────────────────────────┘

Total: 36 bytes per node
```

### Byte Layout Diagram

```
 0               4               8              12              16
 ├───────────────┼───────────────┼───────────────┼───────────────┤
 │   Node ID     │   Position X  │   Position Y  │   Position Z  │
 │    (u32)      │    (f32)      │    (f32)      │    (f32)      │
 ├───────────────┼───────────────┼───────────────┼───────────────┤
16              20              24              28              32
 ├───────────────┼───────────────┼───────────────┼───────────────┤
 │  Velocity X   │  Velocity Y   │  Velocity Z   │     Mass      │
 │    (f32)      │    (f32)      │    (f32)      │    (f32)      │
 ├───────────────┼───────────────┼───────────────┼───────────────┤
32              36
 ├───────────────┤
 │    Charge     │
 │    (f32)      │
 └───────────────┘
```

### Field Descriptions

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| **Node ID** | u32 | 0 - 4,294,967,295 | Unique node identifier (matches database `id`) |
| **Position X** | f32 | -∞ to +∞ | 3D position (world coordinates) |
| **Position Y** | f32 | -∞ to +∞ | 3D position (world coordinates) |
| **Position Z** | f32 | -∞ to +∞ | 3D position (world coordinates) |
| **Velocity X** | f32 | -∞ to +∞ | Physics velocity (m/s) |
| **Velocity Y** | f32 | -∞ to +∞ | Physics velocity (m/s) |
| **Velocity Z** | f32 | -∞ to +∞ | Physics velocity (m/s) |
| **Mass** | f32 | 0.0 - +∞ | Node mass (affects physics) |
| **Charge** | f32 | -∞ to +∞ | Semantic charge (ontology-driven) |

### Example: Parsing in TypeScript

```typescript
interface NodeUpdate {
    id: number;
    position: [number, number, number];
    velocity: [number, number, number];
    mass: number;
    charge: number;
}

class BinaryProtocolParser {
    private view: DataView;

    constructor(buffer: ArrayBuffer) {
        this.view = new DataView(buffer);
    }

    parseNodeUpdates(): NodeUpdate[] {
        const nodeCount = this.view.byteLength / 36;
        const updates: NodeUpdate[] = [];

        for (let i = 0; i < nodeCount; i++) {
            const offset = i * 36;

            updates.push({
                id: this.view.getUint32(offset + 0, true),  // Little-endian
                position: [
                    this.view.getFloat32(offset + 4, true),
                    this.view.getFloat32(offset + 8, true),
                    this.view.getFloat32(offset + 12, true),
                ],
                velocity: [
                    this.view.getFloat32(offset + 16, true),
                    this.view.getFloat32(offset + 20, true),
                    this.view.getFloat32(offset + 24, true),
                ],
                mass: this.view.getFloat32(offset + 28, true),
                charge: this.view.getFloat32(offset + 32, true),
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
            updateNodePosition(node.id, node.position);
            updateNodeVelocity(node.id, node.velocity);
        });
    }
};
```

### Example: Parsing in Rust (Server-Side)

```rust
use byteorder::{LittleEndian, ReadBytesExt};
use std::io::Cursor;

#[repr(C, packed)]
pub struct NodeUpdateBinary {
    pub id: u32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub vx: f32,
    pub vy: f32,
    pub vz: f32,
    pub mass: f32,
    pub charge: f32,
}

impl NodeUpdateBinary {
    pub fn serialize(nodes: &[Node]) -> Vec<u8> {
        let mut buffer = Vec::with-capacity(nodes.len() * 36);

        for node in nodes {
            buffer.extend-from-slice(&node.id.to-le-bytes());
            buffer.extend-from-slice(&node.x.to-le-bytes());
            buffer.extend-from-slice(&node.y.to-le-bytes());
            buffer.extend-from-slice(&node.z.to-le-bytes());
            buffer.extend-from-slice(&node.vx.to-le-bytes());
            buffer.extend-from-slice(&node.vy.to-le-bytes());
            buffer.extend-from-slice(&node.vz.to-le-bytes());
            buffer.extend-from-slice(&node.mass.to-le-bytes());
            buffer.extend-from-slice(&node.charge.to-le-bytes());
        }

        buffer
    }

    pub fn deserialize(data: &[u8]) -> Result<Vec<NodeUpdateBinary>, std::io::Error> {
        let node-count = data.len() / 36;
        let mut nodes = Vec::with-capacity(node-count);
        let mut cursor = Cursor::new(data);

        for - in 0..node-count {
            nodes.push(NodeUpdateBinary {
                id: cursor.read-u32::<LittleEndian>()?,
                x: cursor.read-f32::<LittleEndian>()?,
                y: cursor.read-f32::<LittleEndian>()?,
                z: cursor.read-f32::<LittleEndian>()?,
                vx: cursor.read-f32::<LittleEndian>()?,
                vy: cursor.read-f32::<LittleEndian>()?,
                vz: cursor.read-f32::<LittleEndian>()?,
                mass: cursor.read-f32::<LittleEndian>()?,
                charge: cursor.read-f32::<LittleEndian>()?,
            });
        }

        Ok(nodes)
    }
}
```

---

## Legacy JSON Protocol (DEPRECATED - Historical Reference Only)

> **🚨 DEPRECATION NOTICE**: The JSON WebSocket protocol is **DEPRECATED** and maintained for historical reference only.
> **All new implementations MUST use the Binary V2 protocol (36-byte format).**
> **Legacy JSON support may be removed in future versions.**

### Connection (DEPRECATED)

**⚠️ DO NOT USE - For historical reference only**

```javascript
// DEPRECATED: This protocol is obsolete - use Binary V2 instead
const ws = new WebSocket('ws://localhost:9090/ws?token=YOUR-JWT-TOKEN&protocol=json');
```

### Message Format

All messages use JSON:

```json
{
  "type": "message-type",
  "data": {}
}
```

### Event Types

#### Processing Status

```json
{
  "type": "processing.status",
  "data": {
    "jobId": "uuid",
    "status": "processing",
    "progress": 45
  }
}
```

#### Notifications

```json
{
  "type": "notification",
  "data": {
    "id": "uuid",
    "title": "Processing Complete",
    "message": "Your job finished successfully"
  }
}
```

#### Subscribe to Events

```json
{
  "type": "subscribe",
  "data": {
    "channels": ["projects.uuid", "notifications"]
  }
}
```

### Example (DEPRECATED)

**⚠️ DO NOT USE - For historical reference only**

```javascript
// DEPRECATED: This is the old JSON protocol - DO NOT USE
const ws = new WebSocket('ws://localhost:9090/ws?token=YOUR-TOKEN&protocol=json');

ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'subscribe',
    data: { channels: ['projects.123'] }
  }));
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  console.log('Received:', message);
};

// MIGRATE TO BINARY V2 - See migration guide below
```

---

## Connection

### Establishing Connection

```typescript
class VisionFlowWebSocket {
    private ws: WebSocket;
    private protocol: 'binary' | 'json';

    constructor(url: string, token: string, protocol: 'binary' | 'json' = 'binary') {
        this.protocol = protocol;
        const protocolParam = protocol === 'json' ? '&protocol=json' : '';  // DEPRECATED: JSON protocol
        this.ws = new WebSocket(`${url}?token=${token}${protocolParam}`);

        if (protocol === 'binary') {
            this.ws.binaryType = 'arraybuffer';
        }

        this.setupHandlers();
    }

    private setupHandlers() {
        this.ws.onopen = () => console.log('Connected (protocol:', this.protocol + ')');
        this.ws.onerror = (err) => console.error('WebSocket error:', err);
        this.ws.onclose = (event) => {
            console.log('Disconnected:', event.code, event.reason);
            this.reconnect();
        };

        this.ws.onmessage = (event) => {
            if (this.protocol === 'binary') {
                this.handleBinaryMessage(event.data as ArrayBuffer);
            } else {
                this.handleJsonMessage(JSON.parse(event.data));
            }
        };
    }

    private handleBinaryMessage(buffer: ArrayBuffer) {
        const parser = new BinaryProtocolParser(buffer);
        const updates = parser.parseNodeUpdates();
        this.onNodeUpdates(updates);
    }

    private handleJsonMessage(message: any) {
        // Legacy JSON handling
    }

    private reconnect() {
        setTimeout(() => {
            console.log('Reconnecting...');
            this.ws = new WebSocket(this.ws.url);
            this.setupHandlers();
        }, 1000);
    }
}
```

### Authentication

```bash
# Obtain JWT token
curl -X POST http://localhost:9090/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "password": "pass"}'

# Response: {"token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."}

# Connect with token
const ws = new WebSocket('ws://localhost:9090/ws?token=eyJhbGci...');
```

---

## Client Implementation

### React Three Fiber Example

```typescript
import { useEffect, useRef } from 'react';
import { useThree } from '@react-three/fiber';

export function usePhysicsStream() {
    const wsRef = useRef<VisionFlowWebSocket | null>(null);
    const nodesRef = useRef<Map<number, THREE.Mesh>>(new Map());

    useEffect(() => {
        wsRef.current = new VisionFlowWebSocket(
            'ws://localhost:9090/ws',
            localStorage.getItem('jwt-token')!,
            'binary'
        );

        wsRef.current.onNodeUpdates = (updates) => {
            updates.forEach(update => {
                const mesh = nodesRef.current.get(update.id);
                if (mesh) {
                    mesh.position.set(...update.position);
                    // Optional: Use velocity for motion blur
                    mesh.userData.velocity = update.velocity;
                }
            });
        };

        return () => wsRef.current?.close();
    }, []);

    return { ws: wsRef.current, nodes: nodesRef.current };
}
```

---

## Performance

### Benchmarks (100k nodes @ 60 FPS)

| Metric | Binary V2 | JSON V1 | Improvement |
|--------|-----------|---------|-------------|
| **Message Size** | 3.6 MB | 18 MB | 80% smaller |
| **Parse Time** | 0.8 ms | 12 ms | 15x faster |
| **Network Latency** | <10 ms | 45 ms | 4.5x faster |
| **CPU Usage** | 5% | 28% | 5.6x lower |
| **Memory Allocation** | 3.6 MB | 22 MB | 84% less |

**Hardware**: Client @ Chrome 120, RTX 4080, 1Gbps LAN

### Optimization Tips

1. **Use Binary Protocol**: Always prefer binary for real-time updates
2. **Batch Updates**: Server sends 16ms batches (60 FPS)
3. **Typed Arrays**: Zero-copy parsing with `DataView`
4. **WebSocket Compression**: Enable `permessage-deflate` for 2x compression
5. **Connection Pooling**: Reuse WebSocket connections

---

## Migration Guide

### Upgrading from JSON to Binary (REQUIRED)

**All clients using the deprecated JSON protocol MUST migrate to Binary V2.**

```typescript
// ❌ BEFORE (DEPRECATED JSON Protocol)
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);      // 18 MB/frame
    updateNodes(data.nodes);
};

// ✅ AFTER (Binary V2 Protocol - REQUIRED)
ws.binaryType = 'arraybuffer';
ws.onmessage = (event) => {
    const parser = new BinaryProtocolParser(event.data);
    const updates = parser.parseNodeUpdates();  // 3.6 MB/frame (80% reduction!)
    updateNodes(updates);
};
```

### Migration Steps

1. **Set Binary Type**: Configure WebSocket to receive `ArrayBuffer`
   ```typescript
   ws.binaryType = 'arraybuffer';
   ```

2. **Update Message Handler**: Replace JSON.parse with binary parser
   ```typescript
   const parser = new BinaryProtocolParser(event.data);
   const updates = parser.parseNodeUpdates();
   ```

3. **Update Data Structures**: Use typed arrays for node data
   ```typescript
   interface NodeUpdate {
       id: number;
       position: [number, number, number];
       velocity: [number, number, number];
       mass: number;
       charge: number;
   }
   ```

4. **Remove JSON Serialization**: No more `JSON.parse()` or `JSON.stringify()`

5. **Test Performance**: Verify 80% bandwidth reduction and <10ms latency

**For detailed migration instructions, see [Binary Protocol Migration Guide](../../guides/migration/json-to-binary-protocol.md)**

---

## References

- **[Binary Protocol Specification](../binary-websocket.md)** - Complete technical specification of the 36-byte binary format
- **[REST API Documentation](./rest-api-complete.md)**
- **[Performance Benchmarks](../performance-benchmarks.md)** - Comprehensive performance testing results
- ****

---

**Last Updated**: November 3, 2025
**Maintainer**: VisionFlow Documentation Team
**Protocol Version**: Binary V2 (Current), JSON V1 (Deprecated)
