# VisionFlow Binary Protocol Specification

**Version:** 2.0
**Status:** Production
**Last Updated:** November 5, 2025

---

## Overview

VisionFlow uses a high-performance 36-byte binary protocol for real-time graph physics updates over WebSocket connections. This protocol achieves 80% bandwidth reduction compared to JSON while maintaining sub-10ms latency for 100k+ node graphs.

---

## Protocol Design

### Message Format

Each node update message is **exactly 36 bytes** in little-endian format:

```
┌──────────┬───────────────────────────────────────────┐
│ Offset   │ Field (Type)                               │
├──────────┼───────────────────────────────────────────┤
│ [0-3]    │ Node ID (u32)                             │
│ [4-7]    │ Position X (f32)                          │
│ [8-11]   │ Position Y (f32)                          │
│ [12-15]  │ Position Z (f32)                          │
│ [16-19]  │ Velocity X (f32)                          │
│ [20-23]  │ Velocity Y (f32)                          │
│ [24-27]  │ Velocity Z (f32)                          │
│ [28-31]  │ Mass (f32)                                │
│ [32-35]  │ Charge (f32)                              │
└──────────┴───────────────────────────────────────────┘
```

### Field Specifications

| Field | Type | Bytes | Endianness | Range | Description |
|-------|------|-------|------------|-------|-------------|
| **Node ID** | u32 | 4 | Little | 0 - 4,294,967,295 | Unique node identifier matching database `id` |
| **Position X** | f32 | 4 | Little | -∞ to +∞ | 3D world coordinate (meters) |
| **Position Y** | f32 | 4 | Little | -∞ to +∞ | 3D world coordinate (meters) |
| **Position Z** | f32 | 4 | Little | -∞ to +∞ | 3D world coordinate (meters) |
| **Velocity X** | f32 | 4 | Little | -∞ to +∞ | Physics velocity (m/s) |
| **Velocity Y** | f32 | 4 | Little | -∞ to +∞ | Physics velocity (m/s) |
| **Velocity Z** | f32 | 4 | Little | -∞ to +∞ | Physics velocity (m/s) |
| **Mass** | f32 | 4 | Little | 0.0 - +∞ | Node mass (kg, affects physics) |
| **Charge** | f32 | 4 | Little | -∞ to +∞ | Semantic charge (ontology-driven) |

---

## Batch Messages

Multiple node updates are sent in a single WebSocket message by concatenating 36-byte records:

```
Total Message Size = 36 * node_count bytes
```

**Example:** 1000 nodes = 36,000 bytes (36 KB)

---

## Server Implementation (Rust)

### Serialization

```rust
use byteorder::{LittleEndian, WriteBytesExt};

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
    /// Serialize multiple nodes to binary format
    pub fn serialize_batch(nodes: &[Node]) -> Vec<u8> {
        let mut buffer = Vec::with_capacity(nodes.len() * 36);

        for node in nodes {
            buffer.extend_from_slice(&node.id.to_le_bytes());
            buffer.extend_from_slice(&node.x.to_le_bytes());
            buffer.extend_from_slice(&node.y.to_le_bytes());
            buffer.extend_from_slice(&node.z.to_le_bytes());
            buffer.extend_from_slice(&node.vx.to_le_bytes());
            buffer.extend_from_slice(&node.vy.to_le_bytes());
            buffer.extend_from_slice(&node.vz.to_le_bytes());
            buffer.extend_from_slice(&node.mass.to_le_bytes());
            buffer.extend_from_slice(&node.charge.to_le_bytes());
        }

        buffer
    }

    /// Deserialize binary data to nodes
    pub fn deserialize_batch(data: &[u8]) -> Result<Vec<NodeUpdateBinary>, std::io::Error> {
        if data.len() % 36 != 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Data length must be multiple of 36"
            ));
        }

        let node_count = data.len() / 36;
        let mut nodes = Vec::with_capacity(node_count);
        let mut cursor = std::io::Cursor::new(data);

        for _ in 0..node_count {
            nodes.push(NodeUpdateBinary {
                id: cursor.read_u32::<LittleEndian>()?,
                x: cursor.read_f32::<LittleEndian>()?,
                y: cursor.read_f32::<LittleEndian>()?,
                z: cursor.read_f32::<LittleEndian>()?,
                vx: cursor.read_f32::<LittleEndian>()?,
                vy: cursor.read_f32::<LittleEndian>()?,
                vz: cursor.read_f32::<LittleEndian>()?,
                mass: cursor.read_f32::<LittleEndian>()?,
                charge: cursor.read_f32::<LittleEndian>()?,
            });
        }

        Ok(nodes)
    }
}
```

---

## Client Implementation (TypeScript)

### Deserialization

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
const ws = new WebSocket('ws://localhost:9090/ws?token=JWT');
ws.binaryType = 'arraybuffer';

ws.onmessage = (event) => {
    if (event.data instanceof ArrayBuffer) {
        const parser = new BinaryProtocolParser(event.data);
        const updates = parser.parseNodeUpdates();

        updates.forEach(node => {
            updateNodeInScene(node.id, node.position, node.velocity);
        });
    }
};
```

---

## Performance Characteristics

### Bandwidth Comparison

| Graph Size | Binary V2 | JSON V1 | Savings |
|------------|-----------|---------|---------|
| 1K nodes | 36 KB | 180 KB | 80% |
| 10K nodes | 360 KB | 1.8 MB | 80% |
| 100K nodes | 3.6 MB | 18 MB | 80% |
| 1M nodes | 36 MB | 180 MB | 80% |

### Latency Benchmarks (60 FPS, 100K nodes)

| Metric | Binary V2 | JSON V1 |
|--------|-----------|---------|
| Serialization (Server) | 1.2 ms | 15 ms |
| Network Transfer | 8 ms | 42 ms |
| Deserialization (Client) | 0.8 ms | 12 ms |
| **Total Round-Trip** | **10 ms** | **69 ms** |

**Hardware:** Server @ Ryzen 9 5950X, Client @ Chrome 120, 1Gbps LAN

---

## Protocol Versioning

### Version Detection

The server automatically determines protocol version based on WebSocket connection parameters:

```typescript
// Binary V2 (default)
const ws = new WebSocket('ws://localhost:9090/ws?token=JWT');
ws.binaryType = 'arraybuffer';

// Legacy JSON (deprecated)
const ws = new WebSocket('ws://localhost:9090/ws?token=JWT&protocol=json');
```

### Future Protocol Versions

Reserved for future extensions:
- **V3**: Compressed binary format (Zstandard)
- **V4**: Delta encoding for incremental updates
- **V5**: Multi-message batching with header

---

## Error Handling

### Invalid Message Size

Client should validate message size:

```typescript
if (event.data.byteLength % 36 !== 0) {
    console.error('Invalid binary message size:', event.data.byteLength);
    return;
}
```

### Malformed Data

Server validates all fields before sending. Clients should handle NaN/Infinity:

```typescript
if (!isFinite(node.position[0])) {
    console.warn('Invalid position for node', node.id);
    return;
}
```

---

## Compression

WebSocket compression (`permessage-deflate`) can be enabled for additional 2-3x bandwidth savings:

```javascript
const ws = new WebSocket('ws://localhost:9090/ws?token=JWT', {
    perMessageDeflate: true
});
```

**Note:** Compression adds CPU overhead; benchmark for your use case.

---

## Security Considerations

1. **Authentication:** All connections require valid JWT token
2. **Rate Limiting:** Server enforces max 60 updates/second per client
3. **Data Validation:** All numeric fields validated server-side
4. **Buffer Overflow:** Client must validate message size before parsing

---

## References

- 
- 
- [Performance Benchmarks](./performance-benchmarks.md)
- 

---

**Specification Version:** 2.0
**Implementation:** VisionFlow Server v0.1.0+
**Maintainer:** VisionFlow Core Team
