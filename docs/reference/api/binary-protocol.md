# Binary Protocol Specification

## Overview

VisionFlow uses a highly optimized binary protocol for real-time position updates, providing 95% bandwidth reduction compared to JSON. This protocol transmits node positions, velocities, and graph traversal data at 60 FPS with minimal latency.

## Protocol Version

**Current Version**: `1.0`  
**Wire Format**: 34 bytes per node  
**Update Rate**: 60 FPS (16.67ms intervals)  
**Encoding**: Little-endian  

## Binary Frame Format

### Complete 34-Byte Structure

```
┌─────────────┬──────────────┬───────────────────────────────┐
│ Offset      │ Field        │ Description                   │
├─────────────┼──────────────┼───────────────────────────────┤
│ 0-1         │ node_id      │ u16 with control bits         │
│ 2-5         │ position.x   │ f32 - X coordinate            │
│ 6-9         │ position.y   │ f32 - Y coordinate            │
│ 10-13       │ position.z   │ f32 - Z coordinate            │
│ 14-17       │ velocity.x   │ f32 - X velocity              │
│ 18-21       │ velocity.y   │ f32 - Y velocity              │
│ 22-25       │ velocity.z   │ f32 - Z velocity              │
│ 26-29       │ sssp_dist    │ f32 - SSSP distance           │
│ 30-33       │ sssp_parent  │ i32 - SSSP parent node        │
└─────────────┴──────────────┴───────────────────────────────┘
```

### Control Bits (Node ID)

The 16-bit node ID includes control flags:

```
Bit 15: Agent node flag (0x8000)
Bit 14: Knowledge node flag (0x4000)  
Bits 0-13: Actual node ID (max 16,383)
```

Example:
```
0x8005 = Agent node with ID 5
0x4005 = Knowledge node with ID 5
0x0005 = Regular node with ID 5
```

## Data Types

| Type | Size | Range | Description |
|------|------|-------|-------------|
| `u16` | 2 bytes | 0 to 65,535 | Unsigned 16-bit integer |
| `f32` | 4 bytes | ±3.4×10³⁸ | IEEE 754 single precision |
| `i32` | 4 bytes | ±2,147,483,647 | Signed 32-bit integer |

## Implementation

### Server-Side (Rust)

```rust
// Binary protocol constants
const WIRE_ID_SIZE: usize = 2;
const WIRE_VEC3_SIZE: usize = 12;
const WIRE_F32_SIZE: usize = 4;
const WIRE_I32_SIZE: usize = 4;
const WIRE_ITEM_SIZE: usize = 34;

// Control flags
const WIRE_AGENT_FLAG: u16 = 0x8000;
const WIRE_KNOWLEDGE_FLAG: u16 = 0x4000;
const WIRE_NODE_ID_MASK: u16 = 0x3FFF;

// Encoding function
pub fn encode_node_data(nodes: &[NodePosition]) -> Vec<u8> {
    let mut buffer = Vec::with_capacity(nodes.len() * WIRE_ITEM_SIZE);
    
    for node in nodes {
        // Encode node ID with control bits
        let wire_id = to_wire_id(node.id);
        buffer.extend_from_slice(&wire_id.to_le_bytes());
        
        // Encode position (12 bytes)
        buffer.extend_from_slice(&node.position.x.to_le_bytes());
        buffer.extend_from_slice(&node.position.y.to_le_bytes());
        buffer.extend_from_slice(&node.position.z.to_le_bytes());
        
        // Encode velocity (12 bytes)
        buffer.extend_from_slice(&node.velocity.x.to_le_bytes());
        buffer.extend_from_slice(&node.velocity.y.to_le_bytes());
        buffer.extend_from_slice(&node.velocity.z.to_le_bytes());
        
        // Encode SSSP data (8 bytes)
        buffer.extend_from_slice(&node.sssp_distance.to_le_bytes());
        buffer.extend_from_slice(&node.sssp_parent.to_le_bytes());
    }
    
    buffer
}
```

### Client-Side (TypeScript)

```typescript
// Binary protocol constants
export const BINARY_NODE_SIZE = 34;
export const AGENT_NODE_FLAG = 0x8000;
export const KNOWLEDGE_NODE_FLAG = 0x4000;
export const NODE_ID_MASK = 0x3FFF;

// Parsing function
export function parseBinaryNodeData(buffer: ArrayBuffer): BinaryNodeData[] {
    const view = new DataView(buffer);
    const nodes: BinaryNodeData[] = [];
    const nodeCount = Math.floor(buffer.byteLength / BINARY_NODE_SIZE);
    
    for (let i = 0; i < nodeCount; i++) {
        const offset = i * BINARY_NODE_SIZE;
        
        // Parse node ID and control bits
        const nodeId = view.getUint16(offset, true);
        const isAgent = (nodeId & AGENT_NODE_FLAG) !== 0;
        const isKnowledge = (nodeId & KNOWLEDGE_NODE_FLAG) !== 0;
        const actualId = nodeId & NODE_ID_MASK;
        
        // Parse position (offset 2-13)
        const position = {
            x: view.getFloat32(offset + 2, true),
            y: view.getFloat32(offset + 6, true),
            z: view.getFloat32(offset + 10, true)
        };
        
        // Parse velocity (offset 14-25)
        const velocity = {
            x: view.getFloat32(offset + 14, true),
            y: view.getFloat32(offset + 18, true),
            z: view.getFloat32(offset + 22, true)
        };
        
        // Parse SSSP data (offset 26-33)
        const ssspDistance = view.getFloat32(offset + 26, true);
        const ssspParent = view.getInt32(offset + 30, true);
        
        nodes.push({
            nodeId: actualId,
            nodeType: isAgent ? 'agent' : isKnowledge ? 'knowledge' : 'normal',
            position,
            velocity,
            ssspDistance,
            ssspParent
        });
    }
    
    return nodes;
}
```

## WebSocket Integration

### Message Flow

```
1. Server calculates physics (60 FPS)
2. Server encodes positions to binary
3. Server sends WebSocket binary frame
4. Client receives ArrayBuffer
5. Client parses binary data
6. Client updates Three.js scene
```

### WebSocket Handler

```javascript
ws.onmessage = (event) => {
    if (event.data instanceof ArrayBuffer) {
        // Binary position update
        const updates = parseBinaryNodeData(event.data);
        
        // Update node positions
        updates.forEach(update => {
            const node = scene.getNodeById(update.nodeId);
            if (node) {
                node.position.copy(update.position);
                node.velocity = update.velocity;
                node.ssspDistance = update.ssspDistance;
            }
        });
    }
};
```

## Performance Characteristics

### Bandwidth Usage

| Nodes | JSON Size | Binary Size | Reduction |
|-------|-----------|-------------|-----------|
| 1 | ~180 bytes | 34 bytes | 81% |
| 100 | ~18 KB | 3.4 KB | 81% |
| 1000 | ~180 KB | 34 KB | 81% |
| 10000 | ~1.8 MB | 340 KB | 81% |

### Latency

- **Encoding**: <0.1ms for 1000 nodes
- **Network**: Varies by connection
- **Decoding**: <0.05ms for 1000 nodes
- **Total overhead**: <0.2ms typical

### Update Rate

At 60 FPS with 1000 nodes:
- **Data rate**: 2.04 MB/s
- **Messages/sec**: 60
- **Bytes/message**: 34,000

## SSSP (Single Source Shortest Path) Data

The protocol includes SSSP data for graph traversal visualisation:

- **sssp_distance**: Distance from source node (f32)
  - `Infinity` = No path exists
  - `0.0` = Source node
  - `> 0` = Path distance

- **sssp_parent**: Parent node in shortest path (i32)
  - `-1` = No parent (source or unreachable)
  - `>= 0` = Parent node ID

## Compression

For large graphs, consider enabling WebSocket compression:

```javascript
// Server-side compression detection
if (binaryData.length > 1024) {
    const compressed = zlib.deflateSync(binaryData);
    if (compressed.length < binaryData.length * 0.9) {
        ws.send(compressed);
    }
}
```

## Validation and Error Handling

### Client-Side Validation

```javascript
function validateBinaryData(buffer: ArrayBuffer): boolean {
    // Check size is multiple of 34
    if (buffer.byteLength % BINARY_NODE_SIZE !== 0) {
        console.warn('Invalid binary data size');
        return false;
    }
    
    // Validate reasonable bounds
    const view = new DataView(buffer);
    for (let i = 0; i < buffer.byteLength; i += BINARY_NODE_SIZE) {
        const x = view.getFloat32(i + 2, true);
        if (Math.abs(x) > 10000) {
            console.warn('Position out of bounds');
            return false;
        }
    }
    
    return true;
}
```

### Server-Side Validation

```rust
fn validate_node_data(node: &NodePosition) -> bool {
    // Check position bounds
    if node.position.x.abs() > 10000.0 ||
       node.position.y.abs() > 10000.0 ||
       node.position.z.abs() > 10000.0 {
        return false;
    }
    
    // Check velocity limits
    if node.velocity.magnitude() > 100.0 {
        return false;
    }
    
    true
}
```

## Debugging Tools

### Binary Inspector

```javascript
function inspectBinaryFrame(buffer: ArrayBuffer): void {
    const view = new DataView(buffer);
    console.log('Binary Frame Inspection:');
    console.log(`Size: ${buffer.byteLength} bytes`);
    console.log(`Nodes: ${buffer.byteLength / 34}`);
    
    // First node details
    if (buffer.byteLength >= 34) {
        const nodeId = view.getUint16(0, true);
        console.log(`First Node ID: ${nodeId & 0x3FFF}`);
        console.log(`Is Agent: ${(nodeId & 0x8000) !== 0}`);
        console.log(`Position: (${view.getFloat32(2, true).toFixed(2)}, ${view.getFloat32(6, true).toFixed(2)}, ${view.getFloat32(10, true).toFixed(2)})`);
    }
}
```

### Hex Dump

```bash
# Capture and analyze binary frame
websocat ws://localhost:3001/ws | head -c 34 | xxd

# Expected output for one node:
00000000: 0500 0000 0040 0000 8040 0000 c040 0000  .....@...@...@..
00000010: 0000 0000 0000 0000 0000 ff7f ffff ffff  ................
00000020: ffff                                     ..
```

## Migration Guide

### From JSON to Binary

Before (JSON):
```json
{
    "type": "position_update",
    "nodes": [{
        "id": 5,
        "position": {"x": 2.0, "y": 4.0, "z": 6.0},
        "velocity": {"x": 0.0, "y": 0.0, "z": 0.0}
    }]
}
```

After (Binary):
```
[05 00] [00 00 00 40] [00 00 80 40] [00 00 c0 40] [00 00 00 00] [00 00 00 00] [00 00 00 00] [00 00 80 7f] [ff ff ff ff]
```

## Best Practices

1. **Always validate buffer size** before parsing
2. **Use typed arrays** for efficient memory access
3. **Batch updates** to reduce draw calls
4. **Cache DataView** objects when possible
5. **Handle endianness** explicitly (always little-endian)
6. **Monitor performance** with built-in metrics

## Future Extensions

### Version 2.0 Proposal

```
[0-1]   Protocol version (u16)
[2-3]   Message type (u16)
[4-5]   Node count (u16)
[6-39]  First node data (34 bytes)
[40-73] Second node data (34 bytes)
...
```

This would support:
- Protocol versioning
- Multiple message types
- Variable node counts
- Backward compatibility

## Related Documentation

- [WebSocket API](websocket-api.md)
- [MCP Protocol](mcp-protocol.md)
- [Performance Optimization](../../guides/performance.md)
- [Legacy Binary Protocol](../binary-protocol.md)

---

**[← WebSocket API](websocket-api.md)** | **[MCP Protocol →](mcp-protocol.md)**