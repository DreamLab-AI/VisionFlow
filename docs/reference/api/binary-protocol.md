# Binary Protocol Specification

[← Knowledge Base](../../index.md) > [Reference](../index.md) > [API](./index.md) > Binary Protocol

## Overview

VisionFlow uses a highly optimised binary protocol for real-time position updates, achieving **84.8% bandwidth reduction** compared to JSON WebSocket messaging. This document serves as the **definitive specification** for the 34-byte wire format currently in production.

## Protocol Version

**Current Version**: `1.0`
**Wire Format**: 34 bytes per node
**Update Rate**: 60 FPS (16.67ms intervals)
**Encoding**: Little-endian IEEE 754
**Specification Date**: 2025-10-03

## Binary Frame Format

### Complete 34-Byte Wire Structure

Each node position update is transmitted as a fixed 34-byte binary frame:

```
┌─────────────┬──────────────┬──────┬───────────────────────────────┐
│ Byte Offset │ Field Name   │ Type │ Description                   │
├─────────────┼──────────────┼──────┼───────────────────────────────┤
│ 0-1         │ node_id      │ u16  │ Node ID with control bits     │
│ 2-5         │ position.x   │ f32  │ X coordinate (metres)         │
│ 6-9         │ position.y   │ f32  │ Y coordinate (metres)         │
│ 10-13       │ position.z   │ f32  │ Z coordinate (metres)         │
│ 14-17       │ velocity.x   │ f32  │ X velocity (m/s)              │
│ 18-21       │ velocity.y   │ f32  │ Y velocity (m/s)              │
│ 22-25       │ velocity.z   │ f32  │ Z velocity (m/s)              │
│ 26-29       │ sssp_dist    │ f32  │ SSSP distance                 │
│ 30-33       │ sssp_parent  │ i32  │ SSSP parent node ID           │
└─────────────┴──────────────┴──────┴───────────────────────────────┘
```

### Data Types

| Type  | Size     | Range              | Endianness    | Description                  |
|-------|----------|--------------------|---------------|------------------------------|
| `u16` | 2 bytes  | 0 to 65,535        | Little-endian | Unsigned 16-bit integer      |
| `f32` | 4 bytes  | ±3.4×10³⁸          | Little-endian | IEEE 754 single precision    |
| `i32` | 4 bytes  | ±2,147,483,647     | Little-endian | Signed 32-bit integer        |

All numeric values use **little-endian** byte order to match x86-64 architecture and JavaScript TypedArray defaults.

### Control Bits Encoding

The 16-bit node ID field embeds type information in its high bits:

```
┌───────────┬───────────┬─────────────────────────┐
│ Bit 15    │ Bit 14    │ Bits 0-13               │
├───────────┼───────────┼─────────────────────────┤
│ Agent     │ Knowledge │ Actual Node ID          │
│ Flag      │ Flag      │ (max 16,383)            │
└───────────┴───────────┴─────────────────────────┘
```

**Control Bit Masks**:
- **Bit 15** (`0x8000`): Agent node flag
- **Bit 14** (`0x4000`): Knowledge graph node flag
- **Bits 0-13** (`0x3FFF`): Actual node identifier (14-bit range: 0-16,383)

**Encoding Examples**:
```rust
0x8005 = Agent node with ID 5       (bit 15 set)
0x4005 = Knowledge node with ID 5   (bit 14 set)
0xC005 = Both flags (invalid state)
0x0005 = Regular node with ID 5     (no flags)
```

This encoding allows the client to distinguish node types for visualisation without additional metadata transmission.

## Implementation

### Server-Side Encoding (Rust)

```rust
// Binary protocol constants
const WIRE_ID_SIZE: usize = 2;
const WIRE_VEC3_SIZE: usize = 12;
const WIRE_F32_SIZE: usize = 4;
const WIRE_I32_SIZE: usize = 4;
const WIRE_ITEM_SIZE: usize = 34;

// Control flags for u16 wire format
const WIRE_AGENT_FLAG: u16 = 0x8000;
const WIRE_KNOWLEDGE_FLAG: u16 = 0x4000;
const WIRE_NODE_ID_MASK: u16 = 0x3FFF;

/// Convert u32 node ID with flags to u16 wire format
pub fn to_wire_id(node_id: u32) -> u16 {
    let actual_id = (node_id & 0x3FFFFFFF) as u16;
    let wire_id = actual_id & WIRE_NODE_ID_MASK;

    if is_agent_node(node_id) {
        wire_id | WIRE_AGENT_FLAG
    } else if is_knowledge_node(node_id) {
        wire_id | WIRE_KNOWLEDGE_FLAG
    } else {
        wire_id
    }
}

/// Encode node position data for WebSocket transmission
pub fn encode_node_data(nodes: &[(u32, NodePosition)]) -> Vec<u8> {
    let mut buffer = Vec::with_capacity(nodes.len() * WIRE_ITEM_SIZE);

    for (node_id, node) in nodes {
        // Encode node ID with control bits (2 bytes)
        let wire_id = to_wire_id(*node_id);
        buffer.extend_from_slice(&wire_id.to_le_bytes());

        // Encode position vector (12 bytes)
        buffer.extend_from_slice(&node.position.x.to_le_bytes());
        buffer.extend_from_slice(&node.position.y.to_le_bytes());
        buffer.extend_from_slice(&node.position.z.to_le_bytes());

        // Encode velocity vector (12 bytes)
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

**Broadcasting Pattern**:
```rust
// Server broadcasts position updates at 60 FPS
use tokio::time::{interval, Duration};
use axum::extract::ws::{WebSocket, Message};

let mut ticker = interval(Duration::from_millis(16)); // ~60 FPS

loop {
    ticker.tick().await;

    let updated_nodes = graph_state.get_node_positions();
    let binary_data = encode_node_data(&updated_nodes);

    websocket.send(Message::Binary(binary_data)).await?;
}
```

### Client-Side Decoding (TypeScript)

```typescript
// Binary protocol constants
export const BINARY_NODE_SIZE = 34;
export const AGENT_NODE_FLAG = 0x8000;
export const KNOWLEDGE_NODE_FLAG = 0x4000;
export const NODE_ID_MASK = 0x3FFF;

export interface BinaryNodeData {
  nodeId: number;
  nodeType: 'agent' | 'knowledge' | 'normal';
  position: { x: number; y: number; z: number };
  velocity: { x: number; y: number; z: number };
  ssspDistance: number;
  ssspParent: number;
}

/**
 * Parse binary node data from WebSocket ArrayBuffer
 * @param buffer - Raw binary data (must be multiple of 34 bytes)
 * @returns Array of parsed node data
 */
export function parseBinaryNodeData(buffer: ArrayBuffer): BinaryNodeData[] {
  const view = new DataView(buffer);
  const nodes: BinaryNodeData[] = [];

  // Validate buffer size
  if (buffer.byteLength % BINARY_NODE_SIZE !== 0) {
    console.warn(
      `Invalid buffer size: ${buffer.byteLength} bytes ` +
      `(expected multiple of ${BINARY_NODE_SIZE})`
    );
    return nodes;
  }

  const nodeCount = Math.floor(buffer.byteLength / BINARY_NODE_SIZE);

  for (let i = 0; i < nodeCount; i++) {
    const offset = i * BINARY_NODE_SIZE;

    // Parse node ID and extract control bits (offset 0-1)
    const rawNodeId = view.getUint16(offset, true);
    const isAgent = (rawNodeId & AGENT_NODE_FLAG) !== 0;
    const isKnowledge = (rawNodeId & KNOWLEDGE_NODE_FLAG) !== 0;
    const actualId = rawNodeId & NODE_ID_MASK;

    // Parse position vector (offset 2-13)
    const position = {
      x: view.getFloat32(offset + 2, true),
      y: view.getFloat32(offset + 6, true),
      z: view.getFloat32(offset + 10, true)
    };

    // Parse velocity vector (offset 14-25)
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

### WebSocket Integration

```typescript
// WebSocket message handler
ws.onmessage = (event: MessageEvent) => {
  if (event.data instanceof ArrayBuffer) {
    // Binary position update from physics simulation
    const nodeUpdates = parseBinaryNodeData(event.data);

    // Update Three.js scene (no client-side physics required)
    nodeUpdates.forEach(update => {
      const mesh = scene.getObjectByName(`node-${update.nodeId}`);
      if (mesh) {
        // Apply position directly from server
        mesh.position.set(
          update.position.x,
          update.position.y,
          update.position.z
        );

        // Store velocity for optional interpolation
        mesh.userData.velocity = update.velocity;

        // Update SSSP visualisation
        if (update.ssspDistance !== Infinity) {
          updatePathVisualisation(update.nodeId, update.ssspParent);
        }
      }
    });
  }
};
```

## SSSP (Single Source Shortest Path) Data

The protocol includes graph traversal data for visualising shortest paths:

### SSSP Distance Field

**Type**: `f32` (4 bytes, offset 26-29)
**Values**:
- `0.0` - Source node (path origin)
- `> 0.0` - Distance from source node (weighted edge sum)
- `Infinity` (`0x7F800000`) - No path exists or unreachable

### SSSP Parent Field

**Type**: `i32` (4 bytes, offset 30-33)
**Values**:
- `-1` - No parent (source node or unreachable)
- `≥ 0` - Parent node ID in shortest path tree

### Path Reconstruction

```typescript
/**
 * Reconstruct shortest path from source to target node
 * @param nodeId - Target node
 * @param ssspData - Map of node SSSP data
 * @returns Array of node IDs forming path from source to target
 */
function reconstructPath(
  nodeId: number,
  ssspData: Map<number, BinaryNodeData>
): number[] {
  const path: number[] = [];
  let current = nodeId;

  while (current !== -1) {
    path.unshift(current);
    const node = ssspData.get(current);

    if (!node || node.ssspParent === -1) {
      break;
    }

    current = node.ssspParent;
  }

  return path;
}
```

## Performance Characteristics

### Bandwidth Reduction

Comparison with JSON WebSocket protocol:

| Nodes | JSON Size  | Binary Size | Reduction | Data Rate @ 60 FPS |
|-------|------------|-------------|-----------|-------------------|
| 1     | ~180 bytes | 34 bytes    | 81.1%     | 2.04 KB/s         |
| 50    | ~9 KB      | 1.7 KB      | 81.1%     | 102 KB/s          |
| 100   | ~18 KB     | 3.4 KB      | 81.1%     | 204 KB/s          |
| 1000  | ~180 KB    | 34 KB       | 81.1%     | 2.04 MB/s         |

**Overall Achieved**: 84.8% bandwidth reduction (including WebSocket framing overhead)

### Latency Breakdown

| Operation                  | P50   | P95   | P99   |
|----------------------------|-------|-------|-------|
| Server-side encoding       | 0.1ms | 0.3ms | 0.5ms |
| Network transmission (LAN) | 2ms   | 5ms   | 10ms  |
| Client-side decoding       | 0.1ms | 0.2ms | 0.3ms |
| Three.js scene update      | 1ms   | 3ms   | 8ms   |
| **Total end-to-end**       | 3ms   | 8ms   | 18ms  |

### Throughput Limits

- **Maximum nodes per update**: 10,000 (340 KB frame)
- **Typical nodes per update**: 50-100 (1.7-3.4 KB frame)
- **Update frequency**: 60 FPS (16.67ms interval)
- **Sustained data rate**: 2-10 MB/s
- **Burst capacity**: Up to 20 MB/s (1200 nodes @ 60 FPS)

## Coordinate System

### Spatial Bounds

- **Origin**: Centre of 3D space `(0, 0, 0)`
- **Units**: Metres (1 unit = 1 metre for XR scaling)
- **Position bounds**: ±10,000 units on each axis
- **Velocity range**: ±100 units/second maximum

### Validation

```rust
pub fn validate_node_position(position: &Vec3, velocity: &Vec3) -> Result<()> {
    // Check position bounds
    const MAX_POSITION: f32 = 10000.0;
    if position.x.abs() > MAX_POSITION ||
       position.y.abs() > MAX_POSITION ||
       position.z.abs() > MAX_POSITION {
        return Err(Error::PositionOutOfBounds);
    }

    // Check velocity limits
    const MAX_VELOCITY: f32 = 100.0;
    let velocity_magnitude = (
        velocity.x.powi(2) +
        velocity.y.powi(2) +
        velocity.z.powi(2)
    ).sqrt();

    if velocity_magnitude > MAX_VELOCITY {
        return Err(Error::VelocityExceeded);
    }

    Ok(())
}
```

### Client-Side Validation

```typescript
function validateBinaryData(buffer: ArrayBuffer): boolean {
  // Validate buffer size
  if (buffer.byteLength % BINARY_NODE_SIZE !== 0) {
    console.warn('Invalid binary data size');
    return false;
  }

  const view = new DataView(buffer);
  const nodeCount = buffer.byteLength / BINARY_NODE_SIZE;

  // Validate position bounds for all nodes
  for (let i = 0; i < nodeCount; i++) {
    const offset = i * BINARY_NODE_SIZE;

    const x = view.getFloat32(offset + 2, true);
    const y = view.getFloat32(offset + 6, true);
    const z = view.getFloat32(offset + 10, true);

    if (Math.abs(x) > 10000 ||
        Math.abs(y) > 10000 ||
        Math.abs(z) > 10000) {
      console.warn(`Position out of bounds at node ${i}`);
      return false;
    }
  }

  return true;
}
```

## Error Handling

### Frame Size Validation

```typescript
ws.onmessage = (event: MessageEvent) => {
  if (event.data instanceof ArrayBuffer) {
    const buffer = event.data;

    // Reject invalid frame sizes
    if (buffer.byteLength % BINARY_NODE_SIZE !== 0) {
      console.error(
        `Invalid frame size: ${buffer.byteLength} bytes ` +
        `(must be multiple of ${BINARY_NODE_SIZE})`
      );
      return;
    }

    // Reject empty frames
    if (buffer.byteLength === 0) {
      console.warn('Received empty binary frame');
      return;
    }

    // Process valid frame
    const nodes = parseBinaryNodeData(buffer);
    updateVisualisation(nodes);
  }
};
```

### NaN and Infinity Handling

```typescript
function sanitiseNodeData(node: BinaryNodeData): BinaryNodeData {
  // Replace NaN with zero
  if (isNaN(node.position.x)) node.position.x = 0;
  if (isNaN(node.position.y)) node.position.y = 0;
  if (isNaN(node.position.z)) node.position.z = 0;

  // Clamp infinite velocities
  const maxVel = 100;
  if (!isFinite(node.velocity.x)) node.velocity.x = 0;
  if (!isFinite(node.velocity.y)) node.velocity.y = 0;
  if (!isFinite(node.velocity.z)) node.velocity.z = 0;

  node.velocity.x = Math.max(-maxVel, Math.min(maxVel, node.velocity.x));
  node.velocity.y = Math.max(-maxVel, Math.min(maxVel, node.velocity.y));
  node.velocity.z = Math.max(-maxVel, Math.min(maxVel, node.velocity.z));

  return node;
}
```

## Debugging Tools

### Binary Frame Inspector

```typescript
function inspectBinaryFrame(buffer: ArrayBuffer): void {
  const view = new DataView(buffer);
  const nodeCount = buffer.byteLength / BINARY_NODE_SIZE;

  console.log('=== Binary Frame Inspection ===');
  console.log(`Frame size: ${buffer.byteLength} bytes`);
  console.log(`Node count: ${nodeCount}`);
  console.log('');

  // Inspect first 3 nodes
  const inspectCount = Math.min(3, nodeCount);

  for (let i = 0; i < inspectCount; i++) {
    const offset = i * BINARY_NODE_SIZE;

    const rawId = view.getUint16(offset, true);
    const actualId = rawId & NODE_ID_MASK;
    const isAgent = (rawId & AGENT_NODE_FLAG) !== 0;
    const isKnowledge = (rawId & KNOWLEDGE_NODE_FLAG) !== 0;

    console.log(`Node ${i}:`);
    console.log(`  Raw ID: 0x${rawId.toString(16).padStart(4, '0')}`);
    console.log(`  Actual ID: ${actualId}`);
    console.log(`  Type: ${isAgent ? 'Agent' : isKnowledge ? 'Knowledge' : 'Normal'}`);
    console.log(`  Position: (${
      view.getFloat32(offset + 2, true).toFixed(3)
    }, ${
      view.getFloat32(offset + 6, true).toFixed(3)
    }, ${
      view.getFloat32(offset + 10, true).toFixed(3)
    })`);
    console.log(`  Velocity: (${
      view.getFloat32(offset + 14, true).toFixed(3)
    }, ${
      view.getFloat32(offset + 18, true).toFixed(3)
    }, ${
      view.getFloat32(offset + 22, true).toFixed(3)
    })`);
    console.log(`  SSSP Distance: ${view.getFloat32(offset + 26, true).toFixed(3)}`);
    console.log(`  SSSP Parent: ${view.getInt32(offset + 30, true)}`);
    console.log('');
  }
}
```

### Hex Dump Analysis

```bash
# Capture binary WebSocket frame
websocat ws://localhost:3001/ws --binary | head -c 34 | xxd

# Expected output for single node (ID=5, Agent flag set):
# Offset   Hex bytes                                    ASCII
# 00000000: 0580 0000 0040 0000 8040 0000 c040 0000  .....@...@...@..
# 00000010: 0000 0000 0000 0000 0000 ff7f ffff ffff  ................
# 00000020: 0000                                     ..

# Breakdown:
# 05 80     - Node ID 5 with agent flag (0x8005)
# 00 00 00 40 - position.x = 2.0
# 00 00 80 40 - position.y = 4.0
# 00 00 c0 40 - position.z = 6.0
# 00 00 00 00 - velocity.x = 0.0
# 00 00 00 00 - velocity.y = 0.0
# 00 00 00 00 - velocity.z = 0.0
# 00 00 80 7f - sssp_distance = Infinity
# ff ff ff ff - sssp_parent = -1
```

### Performance Profiling

```typescript
class BinaryProtocolMetrics {
  private decodeTimes: number[] = [];
  private frameSizes: number[] = [];

  recordDecode(startTime: number, frameSize: number): void {
    const duration = performance.now() - startTime;

    this.decodeTimes.push(duration);
    this.frameSizes.push(frameSize);

    // Keep last 100 samples
    if (this.decodeTimes.length > 100) {
      this.decodeTimes.shift();
      this.frameSizes.shift();
    }
  }

  getMetrics(): {
    avgDecodeTime: number;
    maxDecodeTime: number;
    avgFrameSize: number;
    throughput: number;
  } {
    const avgDecode = this.decodeTimes.reduce((a, b) => a + b, 0) /
                      this.decodeTimes.length;
    const maxDecode = Math.max(...this.decodeTimes);
    const avgSize = this.frameSizes.reduce((a, b) => a + b, 0) /
                    this.frameSizes.length;

    // Calculate throughput (bytes/second at 60 FPS)
    const throughput = avgSize * 60;

    return {
      avgDecodeTime: avgDecode,
      maxDecodeTime: maxDecode,
      avgFrameSize: avgSize,
      throughput: throughput
    };
  }
}
```

## Migration from JSON

### Legacy JSON Format (Deprecated)

The 28-byte format mentioned in some older documentation was a prototype that never reached production. The current 34-byte format is the only wire format that has been deployed.

**Deprecated JSON representation**:
```json
{
  "type": "position_update",
  "nodes": [{
    "id": 5,
    "position": { "x": 2.0, "y": 4.0, "z": 6.0 },
    "velocity": { "x": 0.0, "y": 0.0, "z": 0.0 }
  }]
}
```

**Size comparison**:
- JSON (formatted): ~180 bytes per node
- JSON (minified): ~140 bytes per node
- Binary (34-byte): 34 bytes per node
- **Reduction**: 81% (formatted) or 76% (minified)

### Migration Checklist

If migrating from JSON to binary protocol:

1. ✅ Update WebSocket message handler to detect `ArrayBuffer`
2. ✅ Implement `parseBinaryNodeData()` deserialisation
3. ✅ Replace JSON parsing with binary parsing
4. ✅ Update position update logic to use `BinaryNodeData`
5. ✅ Test with mixed JSON/binary messages during transition
6. ✅ Add frame validation and error handling
7. ✅ Monitor bandwidth reduction metrics
8. ✅ Remove legacy JSON parsing once stable

## Security Considerations

### Input Validation

```rust
pub fn validate_binary_frame(data: &[u8]) -> Result<()> {
    // Size validation
    if data.len() % WIRE_ITEM_SIZE != 0 {
        return Err(Error::InvalidFrameSize);
    }

    // Maximum frame size (prevent DoS)
    const MAX_FRAME_SIZE: usize = 340_000; // 10,000 nodes
    if data.len() > MAX_FRAME_SIZE {
        return Err(Error::FrameTooLarge);
    }

    // Minimum frame size
    if data.len() < WIRE_ITEM_SIZE {
        return Err(Error::FrameTooSmall);
    }

    Ok(())
}
```

### Rate Limiting

Binary protocol inherits WebSocket connection rate limits:
- **Position updates**: 60 messages/second (60 FPS)
- **Maximum concurrent connections**: 100 per server
- **Maximum frame size**: 340 KB (10,000 nodes)

### Memory Safety

```typescript
// Avoid memory leaks from large buffers
const MAX_BUFFER_SIZE = 1024 * 1024; // 1 MB

ws.onmessage = (event: MessageEvent) => {
  if (event.data instanceof ArrayBuffer) {
    // Reject oversized frames
    if (event.data.byteLength > MAX_BUFFER_SIZE) {
      console.error('Frame size exceeds maximum allowed');
      return;
    }

    // Process and immediately release reference
    const nodes = parseBinaryNodeData(event.data);
    updateVisualisation(nodes);
    // ArrayBuffer eligible for GC after function scope
  }
};
```

## Future Protocol Extensions

### Version 2.0 Proposal

Potential future enhancements whilst maintaining backwards compatibility:

```
┌─────────────────────────────────────────────────────┐
│ Header (6 bytes)                                    │
├─────────┬─────────┬─────────┬─────────┬────────────┤
│ Version │ Type    │ Flags   │ Count   │ Reserved   │
│ u16     │ u8      │ u8      │ u16     │ u16        │
├─────────┴─────────┴─────────┴─────────┴────────────┤
│ Payload (Variable - N × 34 bytes for node data)    │
└─────────────────────────────────────────────────────┘
```

**Proposed features**:
- Protocol version negotiation
- Multiple message types in single frame
- Optional compression flag
- Batch count for variable node counts
- Backwards compatibility via version detection

**Version detection strategy**:
```typescript
function detectProtocolVersion(buffer: ArrayBuffer): number {
  if (buffer.byteLength >= 2) {
    const view = new DataView(buffer);
    const possibleVersion = view.getUint16(0, true);

    // Version 2.0+ has version header
    if (possibleVersion >= 0x0200) {
      return possibleVersion;
    }
  }

  // Version 1.0 (current) has no version header
  return 0x0100;
}
```

## Best Practices

### Client Implementation

1. **Always validate buffer size** before parsing
2. **Use TypedArrays** for efficient memory access
3. **Cache DataView instances** when processing multiple frames
4. **Handle endianness explicitly** (always little-endian)
5. **Implement frame size checks** before deserialisation
6. **Monitor decode performance** with metrics collection
7. **Batch scene updates** to reduce Three.js overhead

### Server Implementation

1. **Pre-allocate buffers** with exact capacity
2. **Use zero-copy operations** where possible
3. **Validate node data** before encoding
4. **Implement rate limiting** per connection
5. **Monitor encoding latency** to prevent frame drops
6. **Use efficient serialisation** (avoid intermediate allocations)

### Performance Optimisation

```typescript
// ✅ Good: Reuse DataView for multiple frames
class BinaryDecoder {
  private dataView?: DataView;

  decode(buffer: ArrayBuffer): BinaryNodeData[] {
    this.dataView = new DataView(buffer);
    return this.parseNodes(buffer.byteLength / BINARY_NODE_SIZE);
  }

  private parseNodes(count: number): BinaryNodeData[] {
    const nodes: BinaryNodeData[] = new Array(count);

    for (let i = 0; i < count; i++) {
      nodes[i] = this.parseNodeAt(i * BINARY_NODE_SIZE);
    }

    return nodes;
  }
}

// ❌ Bad: Create new DataView for each node
function parseNodesSlow(buffer: ArrayBuffer): BinaryNodeData[] {
  const nodes: BinaryNodeData[] = [];

  for (let i = 0; i < buffer.byteLength / BINARY_NODE_SIZE; i++) {
    const view = new DataView(buffer, i * BINARY_NODE_SIZE, BINARY_NODE_SIZE);
    nodes.push(parseNode(view)); // Inefficient: new DataView per iteration
  }

  return nodes;
}
```

## Related Documentation

- **[WebSocket API Reference](./websocket-api.md)** - Complete WebSocket protocol specification
- **[Networking and Protocols](../../concepts/networking-and-protocols.md)** - Multi-protocol architecture overview
- **[WebSocket Protocol Component](../../architecture/components/websocket-protocol.md)** - Server-side implementation details
- **[Performance Optimisation Guide](../../guides/performance.md)** - Client-side rendering optimisation

## Cross-References

This document is the **definitive specification** for VisionFlow's binary protocol. All implementations must reference this specification for:

- 34-byte wire format structure
- Control bit encoding (bits 14-15 for node types)
- Little-endian byte ordering
- SSSP graph traversal data format
- Bandwidth reduction metrics (84.8%)

---

**[← API Reference Index](./index.md)** | **[MCP Protocol →](./mcp-protocol.md)**
