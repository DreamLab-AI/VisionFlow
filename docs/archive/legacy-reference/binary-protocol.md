# Binary Protocol Specification (Legacy)

> **⚠️ DEPRECATED**: This document describes an early prototype (28-byte format) that was never deployed to production.
>
> **See**: [API Reference: Binary Protocol](./api/binary-protocol.md) for the **definitive 34-byte specification** currently in use.

## Overview

This document describes the 28-byte prototype format explored during early development. VisionFlow's actual production system uses a 34-byte wire format with SSSP data fields.

## Protocol Version (Prototype)

**Prototype Version**: `1.0`
**Specification Date**: 2024-12-19
**Format**: 28-byte fixed-length binary packets (never deployed)  

## Binary Frame Format

### Frame Structure (28 bytes total)

```
Byte Offset | Field Name    | Data Type | Size (bytes) | Description
------------|---------------|-----------|--------------|------------------
0-3         | node_id       | u32       | 4            | Unique node identifier
4-7         | x_position    | f32       | 4            | X coordinate (IEEE 754)
8-11        | y_position    | f32       | 4            | Y coordinate (IEEE 754)
12-15       | z_position    | f32       | 4            | Z coordinate (IEEE 754)
16-19       | x_velocity    | f32       | 4            | X velocity component
20-23       | y_velocity    | f32       | 4            | Y velocity component
24-27       | z_velocity    | f32       | 4            | Z velocity component
```

### Data Types

- **u32**: 32-bit unsigned integer (little-endian)
- **f32**: 32-bit IEEE 754 floating-point (little-endian)

### Coordinate System

- **Origin**: Centre of 3D space (0, 0, 0)
- **Units**: Arbitrary units (typically 1 unit = 1 metre for XR scaling)
- **Bounds**: ±1000 units on each axis
- **Velocity Range**: ±100 units/second maximum

## WebSocket Protocol

### Connection Endpoint

```
ws://localhost:8080/ws/positions
```

### Message Flow

1. **Client Connection**: WebSocket handshake
2. **Server Stream**: Continuous binary frames at 60fps
3. **Frame Processing**: Client deserialises 28-byte packets
4. **Position Updates**: Real-time node position synchronisation

### Error Handling

- **Invalid Frame Size**: Frames not exactly 28 bytes are discarded
- **Out-of-Bounds Values**: Positions/velocities exceeding limits are clamped
- **Unknown Node IDs**: New nodes are dynamically added to graph

## Client Implementation

### TypeScript Interface

```typescript
interface PositionUpdate {
  nodeId: number;
  position: {
    x: number;
    y: number;
    z: number;
  };
  velocity: {
    x: number;
    y: number;
    z: number;
  };
}
```

### Binary Deserialisation

```typescript
function deserialisePositionUpdate(buffer: ArrayBuffer): PositionUpdate {
  const view = new DataView(buffer);
  
  return {
    nodeId: view.getUint32(0, true),  // little-endian
    position: {
      x: view.getFloat32(4, true),
      y: view.getFloat32(8, true),
      z: view.getFloat32(12, true),
    },
    velocity: {
      x: view.getFloat32(16, true),
      y: view.getFloat32(20, true),
      z: view.getFloat32(24, true),
    }
  };
}
```

### WebSocket Integration

```typescript
// client/src/services/WebSocketService.ts
export class WebSocketService {
  private handleBinaryMessage(event: MessageEvent) {
    if (event.data instanceof ArrayBuffer && event.data.byteLength === 28) {
      const update = deserialisePositionUpdate(event.data);
      this.graphDataManager.updateNodePosition(update);
    }
  }
}
```

## Server Implementation

### Rust Serialisation

```rust
// server/src/graph/binary_protocol.rs
#[repr(C, packed)]
pub struct PositionFrame {
    pub node_id: u32,
    pub x_position: f32,
    pub y_position: f32,
    pub z_position: f32,
    pub x_velocity: f32,
    pub y_velocity: f32,
    pub z_velocity: f32,
}

impl PositionFrame {
    pub fn to_bytes(&self) -> [u8; 28] {
        unsafe { std::mem::transmute(*self) }
    }
}
```

### Broadcasting

```rust
// Server broadcasts position updates at 60fps
tokio::time::interval(Duration::from_millis(16)); // ~60fps

for node in &updated_nodes {
    let frame = PositionFrame {
        node_id: node.id,
        x_position: node.position.x,
        y_position: node.position.y,
        z_position: node.position.z,
        x_velocity: node.velocity.x,
        y_velocity: node.velocity.y,
        z_velocity: node.velocity.z,
    };
    
    websocket.send(Message::Binary(frame.to_bytes().to_vec())).await?;
}
```

## Performance Characteristics

### Bandwidth Usage

- **Binary Format**: 28 bytes per node update
- **JSON Equivalent**: ~180 bytes per node update
- **Compression Ratio**: 85.4% reduction
- **Network Load**: ~1.7KB/s for 1000 nodes at 60fps

### Latency

- **Serialisation**: <0.1ms (server-side)
- **Deserialisation**: <0.05ms (client-side)
- **Total Overhead**: <0.2ms per frame

### Throughput

- **Maximum Nodes**: 10,000+ simultaneous updates
- **Update Rate**: 60fps stable
- **Memory Usage**: 280KB buffer for 10,000 nodes

## Versioning and Compatibility

### Version Detection

Future protocol versions will include a version header:

```
Version 2.0+ Format:
Byte 0-1: Protocol Version (u16)
Byte 2-29: Position data (28 bytes)
```

### Backward Compatibility

- Version 1.0 clients will continue to work with fixed 28-byte frames
- Server can detect version based on frame size
- Graceful degradation for older clients

## Security Considerations

### Data Validation

- All numeric values are validated on the server side
- Position bounds are enforced to prevent coordinate overflow
- Velocity limits prevent acceleration exploits

### Authentication

- Binary protocol inherits WebSocket connection authentication
- No additional security headers required in binary frames
- Frame integrity verified by fixed-size validation

## Testing and Debugging

### Frame Inspector

```typescript
function inspectFrame(buffer: ArrayBuffer): string {
  const view = new DataView(buffer);
  return `
    Node ID: ${view.getUint32(0, true)}
    Position: (${view.getFloat32(4, true).toFixed(3)}, 
               ${view.getFloat32(8, true).toFixed(3)}, 
               ${view.getFloat32(12, true).toFixed(3)})
    Velocity: (${view.getFloat32(16, true).toFixed(3)}, 
               ${view.getFloat32(20, true).toFixed(3)}, 
               ${view.getFloat32(24, true).toFixed(3)})
  `;
}
```

### Validation Tools

```bash
# Hex dump of binary frame
xxd -l 28 position_frame.bin

# Expected output:
# 00000000: 0100 0000 0000 8040 0000 0041 0000 8041  .......@...A...A
# 00000010: 0000 0000 0000 0000 0000 0000            ............
```

## Migration from JSON

### JSON Format (Deprecated)

```json
{
  "type": "position_update",
  "nodeId": 1,
  "position": { "x": 4.0, "y": 8.0, "z": 16.0 },
  "velocity": { "x": 0.0, "y": 0.0, "z": 0.0 }
}
```

### Migration Steps

1. Update client WebSocket handlers to process binary messages
2. Replace JSON parsing with binary deserialisation
3. Update position update interfaces to match binary format
4. Test with both formats during transition period

## Migration to Production Format

The 28-byte format was superseded by the 34-byte format which adds:
- SSSP distance field (4 bytes, f32)
- SSSP parent field (4 bytes, i32)

These fields enable real-time shortest path visualisation without additional WebSocket messages.

## Related Documentation

**Current Production Documentation**:
- **[Binary Protocol (Definitive)](./api/binary-protocol.md)** - 34-byte production specification
- [WebSocket API Reference](./api/websocket-api.md) - Complete WebSocket API documentation
- [Networking and Protocols](../concepts/networking-and-protocols.md) - Multi-protocol architecture overview

**Legacy References**:
- This document (28-byte prototype - deprecated)

## Status

**ARCHIVED**: This specification describes a prototype format that was never deployed to production. All implementations use the 34-byte format documented in [api/binary-protocol.md](./api/binary-protocol.md).

---

**[← Reference Index](./index.md)** | **[View Production Spec →](./api/binary-protocol.md)**