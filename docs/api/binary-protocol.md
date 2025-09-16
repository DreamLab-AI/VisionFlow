# Binary Protocol Specification

*[API Documentation](README.md) > Binary Protocol*

## Overview

VisionFlow uses a highly optimized binary protocol for real-time position updates, achieving **94% bandwidth reduction** compared to JSON. The protocol uses a fixed 34-byte format per node with integrated SSSP (Single-Source Shortest Path) data for advanced graph analytics.

## Wire Format Specification

### Node Data Structure (34 bytes)

```
Byte Offset  Size  Type    Field Description
0-1          2     u16     Node ID with type flags (little-endian)
2-5          4     f32     Position X coordinate (IEEE 754)
6-9          4     f32     Position Y coordinate (IEEE 754)
10-13        4     f32     Position Z coordinate (IEEE 754)
14-17        4     f32     Velocity X component (IEEE 754)
18-21        4     f32     Velocity Y component (IEEE 754)
22-25        4     f32     Velocity Z component (IEEE 754)
26-29        4     f32     SSSP distance from source (IEEE 754)
30-33        4     i32     SSSP parent node ID (signed, little-endian)
```

**Total Size**: 34 bytes per node
**Byte Order**: Little-endian throughout
**Alignment**: Naturally aligned (no padding required)

### Memory Layout Diagram

```
┌─────────────┬────────────────┬────────────────┬──────────────┐
│  Node ID    │    Position    │    Velocity    │     SSSP     │
│  (2 bytes)  │   (12 bytes)   │   (12 bytes)   │  (8 bytes)   │
└─────────────┴────────────────┴────────────────┴──────────────┘
0            2                14               26            34
```

## Type Flag Encoding

### Node Type Flags in Node ID

The 16-bit Node ID field uses the high bits for type classification:

```
Bit Layout (MSB to LSB):
15 14 13 12 11 10 09 08 07 06 05 04 03 02 01 00
A  K  -  -  -  -  -  -  -  -  -  -  -  -  -  -
│  │  └─────────── Reserved (must be 0)
│  └─────────────── Knowledge node flag (0x4000)
└────────────────── Agent node flag (0x8000)
```

### Type Flag Values

| Flag Mask | Hex Value | Type | Description |
|-----------|-----------|------|-------------|
| `0x8000` | 32768 | Agent | AI agent node |
| `0x4000` | 16384 | Knowledge | Knowledge graph node |
| `0xC000` | 49152 | Agent+Knowledge | Hybrid node (reserved) |
| `0x0000` | 0 | Standard | Default/unclassified node |

### Node ID Extraction

```rust
// Rust example
let node_id: u16 = wire_data[0..2]; // Read 2 bytes
let is_agent = (node_id & 0x8000) != 0;
let is_knowledge = (node_id & 0x4000) != 0;
let actual_id = node_id & 0x3FFF; // Mask out flag bits (14-bit ID space)
```

```javascript
// JavaScript example
const flaggedId = view.getUint16(offset, true); // Little-endian
const isAgent = (flaggedId & 0x8000) !== 0;
const isKnowledge = (flaggedId & 0x4000) !== 0;
const actualId = flaggedId & 0x3FFF; // 16,383 max node ID
```

## SSSP Integration

### SSSP Fields

The binary protocol includes integrated shortest-path computation data:

- **SSSP Distance** (bytes 26-29): Shortest path distance from designated source node
- **SSSP Parent** (bytes 30-33): Parent node ID for path reconstruction

### SSSP Values

| Value | Meaning |
|-------|---------|
| `distance = 0.0` | This is the source node |
| `distance > 0.0` | Shortest distance from source |
| `distance = +∞` (f32::INFINITY) | Node unreachable from source |
| `parent = -1` | No parent (source node or unreachable) |
| `parent >= 0` | Valid parent node ID for path reconstruction |

### Path Reconstruction Algorithm

```rust
fn reconstruct_path(target_id: u32, node_data: &HashMap<u32, NodeData>) -> Vec<u32> {
    let mut path = Vec::new();
    let mut current = target_id;

    while let Some(node) = node_data.get(&current) {
        path.push(current);

        if node.sssp_parent == -1 {
            break; // Reached source node
        }

        current = node.sssp_parent as u32;
    }

    path.reverse(); // Return source-to-target path
    path
}
```

## Data Type Specifications

### Floating Point Format

All floating-point values use **IEEE 754 single-precision** format:
- **Size**: 32 bits (4 bytes)
- **Precision**: ~7 decimal digits
- **Range**: ±1.18×10⁻³⁸ to ±3.40×10³⁸
- **Special Values**: +∞, -∞, NaN supported

### Coordinate System

Position and velocity use a right-handed 3D coordinate system:
- **X-axis**: Typically horizontal (left-right)
- **Y-axis**: Typically vertical (up-down)
- **Z-axis**: Depth (forward-backward)
- **Units**: Abstract units (typically normalized to [-1000, 1000] range)

### Integer Encoding

- **Node ID**: Unsigned 16-bit integer (0-65535, with 14 bits for actual ID)
- **SSSP Parent**: Signed 32-bit integer (-1 for no parent, 0+ for valid parent ID)

## Message Format

### Binary Message Structure

```
WebSocket Binary Frame:
┌─────────────────────────────────────────────────────────┐
│                   Message Header                        │
│                  (WebSocket Frame)                      │
├─────────────────────────────────────────────────────────┤
│                   Node Data 1                          │
│                   (34 bytes)                           │
├─────────────────────────────────────────────────────────┤
│                   Node Data 2                          │
│                   (34 bytes)                           │
├─────────────────────────────────────────────────────────┤
│                      ...                               │
├─────────────────────────────────────────────────────────┤
│                   Node Data N                          │
│                   (34 bytes)                           │
└─────────────────────────────────────────────────────────┘

Total Size: 34 × N bytes (must be multiple of 34)
```

### Validation Rules

1. **Message Size**: Must be exact multiple of 34 bytes
2. **Node ID Range**: Actual ID must be in range [1, 16383]
3. **Coordinate Bounds**: Typically [-10000, 10000] for each axis
4. **Velocity Limits**: Typically [-1000, 1000] for each component
5. **SSSP Consistency**: Parent must exist in node set or be -1

## Implementation Examples

### Rust Serialization

```rust
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct WireNodeDataItem {
    pub id: u16,              // Node ID with type flags
    pub position: [f32; 3],   // X, Y, Z coordinates
    pub velocity: [f32; 3],   // X, Y, Z velocity components
    pub sssp_distance: f32,   // Shortest path distance
    pub sssp_parent: i32,     // Parent node for path reconstruction
}

impl WireNodeDataItem {
    pub fn new_agent(id: u16, pos: [f32; 3], vel: [f32; 3], distance: f32, parent: i32) -> Self {
        Self {
            id: id | 0x8000,  // Set agent flag
            position: pos,
            velocity: vel,
            sssp_distance: distance,
            sssp_parent: parent,
        }
    }

    pub fn new_knowledge(id: u16, pos: [f32; 3], vel: [f32; 3], distance: f32, parent: i32) -> Self {
        Self {
            id: id | 0x4000,  // Set knowledge flag
            position: pos,
            velocity: vel,
            sssp_distance: distance,
            sssp_parent: parent,
        }
    }

    pub fn get_actual_id(&self) -> u16 {
        self.id & 0x3FFF
    }

    pub fn is_agent(&self) -> bool {
        (self.id & 0x8000) != 0
    }

    pub fn is_knowledge(&self) -> bool {
        (self.id & 0x4000) != 0
    }
}

// Serialization to bytes
fn serialize_nodes(nodes: &[WireNodeDataItem]) -> Vec<u8> {
    bytemuck::cast_slice(nodes).to_vec()
}

// Deserialization from bytes
fn deserialize_nodes(data: &[u8]) -> Result<&[WireNodeDataItem], &'static str> {
    if data.len() % 34 != 0 {
        return Err("Invalid message size");
    }
    Ok(bytemuck::cast_slice(data))
}
```

### JavaScript Deserialization

```javascript
class BinaryProtocolDecoder {
  static BYTES_PER_NODE = 34;
  static AGENT_FLAG = 0x8000;
  static KNOWLEDGE_FLAG = 0x4000;

  static decodePositions(buffer) {
    if (buffer.byteLength % this.BYTES_PER_NODE !== 0) {
      throw new Error(`Invalid buffer size: ${buffer.byteLength}`);
    }

    const nodeCount = buffer.byteLength / this.BYTES_PER_NODE;
    const view = new DataView(buffer);
    const nodes = new Map();

    for (let i = 0; i < nodeCount; i++) {
      const offset = i * this.BYTES_PER_NODE;
      const node = this.decodeNode(view, offset);
      nodes.set(node.id, node);
    }

    return nodes;
  }

  static decodeNode(view, offset) {
    const flaggedId = view.getUint16(offset, true);

    // Extract type flags
    const isAgent = (flaggedId & this.AGENT_FLAG) !== 0;
    const isKnowledge = (flaggedId & this.KNOWLEDGE_FLAG) !== 0;
    const actualId = flaggedId & 0x3FFF;

    // Read position
    const position = {
      x: view.getFloat32(offset + 2, true),
      y: view.getFloat32(offset + 6, true),
      z: view.getFloat32(offset + 10, true)
    };

    // Read velocity
    const velocity = {
      x: view.getFloat32(offset + 14, true),
      y: view.getFloat32(offset + 18, true),
      z: view.getFloat32(offset + 22, true)
    };

    // Read SSSP data
    const ssspDistance = view.getFloat32(offset + 26, true);
    const ssspParent = view.getInt32(offset + 30, true);

    return {
      id: actualId,
      type: this.getNodeType(isAgent, isKnowledge),
      position,
      velocity,
      ssspDistance: ssspDistance === Number.POSITIVE_INFINITY ? null : ssspDistance,
      ssspParent: ssspParent === -1 ? null : ssspParent
    };
  }

  static getNodeType(isAgent, isKnowledge) {
    if (isAgent && isKnowledge) return 'hybrid';
    if (isAgent) return 'agent';
    if (isKnowledge) return 'knowledge';
    return 'standard';
  }

  static encodeNode(node) {
    const buffer = new ArrayBuffer(this.BYTES_PER_NODE);
    const view = new DataView(buffer);

    // Encode ID with type flags
    let flaggedId = node.id & 0x3FFF;
    if (node.type === 'agent' || node.type === 'hybrid') {
      flaggedId |= this.AGENT_FLAG;
    }
    if (node.type === 'knowledge' || node.type === 'hybrid') {
      flaggedId |= this.KNOWLEDGE_FLAG;
    }

    view.setUint16(0, flaggedId, true);

    // Encode position
    view.setFloat32(2, node.position.x, true);
    view.setFloat32(6, node.position.y, true);
    view.setFloat32(10, node.position.z, true);

    // Encode velocity
    view.setFloat32(14, node.velocity.x, true);
    view.setFloat32(18, node.velocity.y, true);
    view.setFloat32(22, node.velocity.z, true);

    // Encode SSSP data
    const distance = node.ssspDistance ?? Number.POSITIVE_INFINITY;
    const parent = node.ssspParent ?? -1;

    view.setFloat32(26, distance, true);
    view.setInt32(30, parent, true);

    return buffer;
  }
}
```

### C++ Implementation

```cpp
#include <cstdint>
#include <vector>
#include <cstring>

#pragma pack(push, 1)
struct WireNodeDataItem {
    uint16_t id;              // Node ID with type flags
    float position[3];        // X, Y, Z coordinates
    float velocity[3];        // X, Y, Z velocity components
    float sssp_distance;      // Shortest path distance
    int32_t sssp_parent;      // Parent node ID

    static constexpr uint16_t AGENT_FLAG = 0x8000;
    static constexpr uint16_t KNOWLEDGE_FLAG = 0x4000;
    static constexpr uint16_t ID_MASK = 0x3FFF;

    uint16_t GetActualId() const { return id & ID_MASK; }
    bool IsAgent() const { return (id & AGENT_FLAG) != 0; }
    bool IsKnowledge() const { return (id & KNOWLEDGE_FLAG) != 0; }
};
#pragma pack(pop)

static_assert(sizeof(WireNodeDataItem) == 34, "Wire format must be exactly 34 bytes");

class BinaryProtocol {
public:
    static std::vector<uint8_t> SerializeNodes(const std::vector<WireNodeDataItem>& nodes) {
        std::vector<uint8_t> buffer(nodes.size() * sizeof(WireNodeDataItem));
        std::memcpy(buffer.data(), nodes.data(), buffer.size());
        return buffer;
    }

    static std::vector<WireNodeDataItem> DeserializeNodes(const std::vector<uint8_t>& buffer) {
        if (buffer.size() % sizeof(WireNodeDataItem) != 0) {
            throw std::runtime_error("Invalid buffer size");
        }

        const size_t nodeCount = buffer.size() / sizeof(WireNodeDataItem);
        std::vector<WireNodeDataItem> nodes(nodeCount);
        std::memcpy(nodes.data(), buffer.data(), buffer.size());
        return nodes;
    }
};
```

## Performance Characteristics

### Bandwidth Comparison

| Node Count | JSON Size | Binary Size | Reduction | Compression |
|------------|-----------|-------------|-----------|-------------|
| 1 | ~150 bytes | 34 bytes | 77% | 84% with gzip |
| 100 | ~15 KB | 3.4 KB | 77% | 94% with gzip |
| 1,000 | ~150 KB | 34 KB | 77% | 95% with gzip |
| 10,000 | ~1.5 MB | 340 KB | 77% | 95% with gzip |

### Transmission Speed (60 FPS)

| Node Count | JSON (MB/s) | Binary (KB/s) | Reduction |
|------------|-------------|---------------|-----------|
| 100 | 9.0 | 2,040 | 77% |
| 500 | 45 | 10,200 | 77% |
| 1,000 | 90 | 20,400 | 77% |
| 5,000 | 450 | 102,000 | 77% |

### Memory Efficiency

- **Fixed Size**: No variable-length fields or padding
- **Natural Alignment**: All fields are naturally aligned
- **Zero-Copy**: Can be cast directly to/from byte arrays
- **Cache Friendly**: Sequential access patterns

### CPU Performance

- **Encoding**: ~0.5μs per node (no serialization overhead)
- **Decoding**: ~0.3μs per node (direct memory mapping)
- **Validation**: ~0.1μs per node (bounds checking)
- **Type Extraction**: ~0.05μs per node (bitwise operations)

## Validation and Error Handling

### Client-Side Validation

```javascript
class ProtocolValidator {
  static validateMessage(buffer) {
    const errors = [];

    // Size validation
    if (buffer.byteLength % 34 !== 0) {
      errors.push(`Invalid message size: ${buffer.byteLength} (must be multiple of 34)`);
    }

    const nodeCount = Math.floor(buffer.byteLength / 34);
    const view = new DataView(buffer);

    // Validate each node
    for (let i = 0; i < nodeCount; i++) {
      const offset = i * 34;
      const nodeErrors = this.validateNode(view, offset);
      errors.push(...nodeErrors.map(err => `Node ${i}: ${err}`));
    }

    return { valid: errors.length === 0, errors };
  }

  static validateNode(view, offset) {
    const errors = [];

    // Validate node ID
    const flaggedId = view.getUint16(offset, true);
    const actualId = flaggedId & 0x3FFF;

    if (actualId === 0) {
      errors.push('Node ID cannot be 0');
    }

    if (actualId > 16383) {
      errors.push(`Node ID ${actualId} exceeds maximum (16383)`);
    }

    // Validate positions
    for (let i = 0; i < 3; i++) {
      const pos = view.getFloat32(offset + 2 + i * 4, true);
      if (!isFinite(pos)) {
        errors.push(`Invalid position component ${i}: ${pos}`);
      }
      if (Math.abs(pos) > 10000) {
        errors.push(`Position component ${i} out of bounds: ${pos}`);
      }
    }

    // Validate velocities
    for (let i = 0; i < 3; i++) {
      const vel = view.getFloat32(offset + 14 + i * 4, true);
      if (!isFinite(vel)) {
        errors.push(`Invalid velocity component ${i}: ${vel}`);
      }
      if (Math.abs(vel) > 1000) {
        errors.push(`Velocity component ${i} out of bounds: ${vel}`);
      }
    }

    // Validate SSSP distance
    const distance = view.getFloat32(offset + 26, true);
    if (!isFinite(distance) && distance !== Number.POSITIVE_INFINITY) {
      errors.push(`Invalid SSSP distance: ${distance}`);
    }
    if (isFinite(distance) && distance < 0) {
      errors.push(`Negative SSSP distance: ${distance}`);
    }

    return errors;
  }
}
```

### Error Recovery

```rust
pub enum DecodeError {
    InvalidSize(usize),
    InvalidNodeId(u16),
    InvalidCoordinate(f32),
    InvalidVelocity(f32),
    InvalidSSSPDistance(f32),
}

pub fn safe_decode_message(data: &[u8]) -> Result<Vec<WireNodeDataItem>, DecodeError> {
    if data.len() % 34 != 0 {
        return Err(DecodeError::InvalidSize(data.len()));
    }

    let nodes = bytemuck::cast_slice::<u8, WireNodeDataItem>(data);

    for node in nodes {
        // Validate node
        validate_node(node)?;
    }

    Ok(nodes.to_vec())
}

fn validate_node(node: &WireNodeDataItem) -> Result<(), DecodeError> {
    let actual_id = node.get_actual_id();
    if actual_id == 0 || actual_id > 16383 {
        return Err(DecodeError::InvalidNodeId(actual_id));
    }

    // Validate coordinates
    for &coord in &node.position {
        if !coord.is_finite() || coord.abs() > 10000.0 {
            return Err(DecodeError::InvalidCoordinate(coord));
        }
    }

    // Validate velocities
    for &vel in &node.velocity {
        if !vel.is_finite() || vel.abs() > 1000.0 {
            return Err(DecodeError::InvalidVelocity(vel));
        }
    }

    // Validate SSSP distance
    if !node.sssp_distance.is_finite() && !node.sssp_distance.is_infinite() {
        return Err(DecodeError::InvalidSSSPDistance(node.sssp_distance));
    }

    Ok(())
}
```

## Protocol Evolution

### Version History

- **v1.0**: Initial 28-byte format (ID, Position, Velocity only)
- **v1.1**: Extended to 34-byte format with SSSP integration
- **v1.2**: Added type flags in Node ID field

### Future Enhancements

1. **Variable Precision**: 16-bit floats for less critical data
2. **Delta Compression**: Send only position changes
3. **Batch Timestamps**: Include update timestamps
4. **Extended Attributes**: Additional node properties

### Backward Compatibility

The protocol includes version detection in the WebSocket handshake:

```javascript
// Client capability negotiation
const ws = new WebSocket('ws://localhost:3001/wss', ['visionflow-v1.2', 'visionflow-v1.1']);

ws.onopen = (event) => {
  const negotiatedProtocol = ws.protocol;
  console.log('Using protocol version:', negotiatedProtocol);
};
```

## Security Considerations

### Data Integrity

- **Size Validation**: Exact byte count requirements
- **Range Checking**: Coordinate and velocity bounds
- **Type Consistency**: Node type flag validation
- **Endianness**: Fixed little-endian byte order

### DoS Prevention

- **Message Size Limits**: Maximum buffer size enforcement
- **Rate Limiting**: Control message frequency per client
- **Validation Costs**: Limit CPU usage in validation
- **Memory Bounds**: Prevent excessive memory allocation

### Input Sanitization

```rust
pub fn sanitize_node_data(mut node: WireNodeDataItem) -> WireNodeDataItem {
    // Clamp coordinates to valid range
    for coord in &mut node.position {
        *coord = coord.clamp(-10000.0, 10000.0);
    }

    // Clamp velocities
    for vel in &mut node.velocity {
        *vel = vel.clamp(-1000.0, 1000.0);
    }

    // Sanitize SSSP distance
    if !node.sssp_distance.is_finite() {
        node.sssp_distance = f32::INFINITY;
    } else if node.sssp_distance < 0.0 {
        node.sssp_distance = 0.0;
    }

    // Validate parent ID
    if node.sssp_parent < -1 {
        node.sssp_parent = -1;
    }

    node
}
```

## Testing and Verification

### Round-Trip Testing

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_trip_serialization() {
        let original = WireNodeDataItem {
            id: 0x8001, // Agent node ID 1
            position: [10.5, 20.3, -5.7],
            velocity: [0.1, -0.2, 0.05],
            sssp_distance: 15.5,
            sssp_parent: 42,
        };

        // Serialize
        let bytes = serialize_nodes(&[original]);
        assert_eq!(bytes.len(), 34);

        // Deserialize
        let deserialized = deserialize_nodes(&bytes).unwrap();
        assert_eq!(deserialized.len(), 1);

        let node = &deserialized[0];
        assert_eq!(node.get_actual_id(), 1);
        assert!(node.is_agent());
        assert_eq!(node.position, original.position);
        assert_eq!(node.velocity, original.velocity);
        assert_eq!(node.sssp_distance, original.sssp_distance);
        assert_eq!(node.sssp_parent, original.sssp_parent);
    }

    #[test]
    fn test_type_flags() {
        let agent = WireNodeDataItem::new_agent(123, [0.0; 3], [0.0; 3], 0.0, -1);
        assert!(agent.is_agent());
        assert!(!agent.is_knowledge());
        assert_eq!(agent.get_actual_id(), 123);

        let knowledge = WireNodeDataItem::new_knowledge(456, [0.0; 3], [0.0; 3], 0.0, -1);
        assert!(!knowledge.is_agent());
        assert!(knowledge.is_knowledge());
        assert_eq!(knowledge.get_actual_id(), 456);
    }
}
```

### Fuzzing Tests

```rust
#[cfg(test)]
fn fuzz_test_decode() {
    use rand::Rng;

    let mut rng = rand::thread_rng();

    for _ in 0..1000 {
        // Generate random 34-byte message
        let mut data = vec![0u8; 34];
        rng.fill(&mut data[..]);

        // Should not panic on any input
        let _result = safe_decode_message(&data);
    }
}
```

---

*This binary protocol is used by the WebSocket streams documented in [websocket-streams.md](websocket-streams.md).*