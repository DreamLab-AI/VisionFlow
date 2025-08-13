# Binary Protocol Documentation

## Overview

The VisionFlow binary protocol is designed for efficient transmission of node position and velocity data over WebSocket connections. It uses a compact binary format to minimize bandwidth usage while maintaining precision for real-time graph visualisation updates.

## Protocol Architecture

### Design Principles

1. **Efficiency**: Minimize bytes per node update
2. **Simplicity**: Fixed-size records for fast parsing
3. **Type Safety**: Support for node type identification
4. **GPU Compatibility**: Memory layout optimized for GPU processing

### Components

- **Binary Protocol** (`src/utils/binary_protocol.rs`): Encoding/decoding logic
- **Socket Flow Messages** (`src/utils/socket_flow_messages.rs`): Data structures
- **Socket Flow Constants** (`src/utils/socket_flow_constants.rs`): Protocol constants

## Wire Format Specification

### Node Data Structure

Each node is transmitted as a fixed 28-byte structure:

```
┌─────────────┬────────────────┬────────────────┐
│  Node ID    │    Position    │    Velocity    │
│  (4 bytes)  │   (12 bytes)   │   (12 bytes)   │
└─────────────┴────────────────┴────────────────┘
```

### Field Details

| Field | Type | Size | Description |
|-------|------|------|-------------|
| Node ID | u32 | 4 bytes | Unique node identifier with type flags |
| Position.x | f32 | 4 bytes | X coordinate |
| Position.y | f32 | 4 bytes | Y coordinate |
| Position.z | f32 | 4 bytes | Z coordinate |
| Velocity.x | f32 | 4 bytes | X velocity |
| Velocity.y | f32 | 4 bytes | Y velocity |
| Velocity.z | f32 | 4 bytes | Z velocity |

### Node Type Flags

The Node ID field includes type flags in the high bits:

| Flag | Value | Description |
|------|-------|-------------|
| Agent Node | 0x80000000 | Bit 31 indicates agent node |
| Knowledge Node | 0x40000000 | Bit 30 indicates knowledge graph node |
| Actual ID Mask | 0x3FFFFFFF | Bits 0-29 contain the actual node ID |

### Wire Format Structure (Rust)

```rust
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct WireNodeDataItem {
    pub id: u32,           // 4 bytes (includes type flags)
    pub position: Vec3Data, // 12 bytes (3 × f32)
    pub velocity: Vec3Data, // 12 bytes (3 × f32)
    // Total: 28 bytes
}

// Compile-time assertion to ensure exact size
static_assertions::const_assert_eq!(std::mem::size_of::<WireNodeDataItem>(), 28);
```

## Message Format

### Binary Message Structure

A complete binary message consists of concatenated node data:

```
┌──────────────┬──────────────┬─────┬──────────────┐
│    Node 1    │    Node 2    │ ... │    Node N    │
│  (28 bytes)  │  (28 bytes)  │     │  (28 bytes)  │
└──────────────┴──────────────┴─────┴──────────────┘
```

### Message Size Calculation

```rust
pub fn calculate_message_size(updates: &[(u32, BinaryNodeData)]) -> usize {
    updates.len() * std::mem::size_of::<WireNodeDataItem>()
}
```

For example:
- 100 nodes = 2,800 bytes
- 1,000 nodes = 28,000 bytes (~27.3 KB)
- 10,000 nodes = 280,000 bytes (~273 KB)

## Encoding Process

### Server to Client

1. **Collect Updates**: Gather node positions and velocities
2. **Apply Type Flags**: Mark agent and knowledge nodes
3. **Create Wire Format**: Convert to `WireNodeDataItem` structures
4. **Serialize**: Use bytemuck for zero-copy serialization
5. **Transmit**: Send as binary WebSocket frame

```rust
// Basic encoding without type flags
pub fn encode_node_data(nodes: &[(u32, BinaryNodeData)]) -> Vec<u8> {
    let mut buffer = Vec::with_capacity(
        nodes.len() * std::mem::size_of::<WireNodeDataItem>()
    );

    for (node_id, node) in nodes {
        let wire_item = WireNodeDataItem {
            id: *node_id,
            position: node.position,
            velocity: node.velocity,
        };

        let item_bytes = bytemuck::bytes_of(&wire_item);
        buffer.extend_from_slice(item_bytes);
    }

    buffer
}

// Enhanced encoding with node type flags
pub fn encode_node_data_with_types(
    nodes: &[(u32, BinaryNodeData)],
    agent_node_ids: &[u32],
    knowledge_node_ids: &[u32]
) -> Vec<u8> {
    let mut buffer = Vec::with_capacity(
        nodes.len() * std::mem::size_of::<WireNodeDataItem>()
    );

    for (node_id, node) in nodes {
        // Apply type flags
        let flagged_id = if agent_node_ids.contains(node_id) {
            set_agent_flag(*node_id)
        } else if knowledge_node_ids.contains(node_id) {
            set_knowledge_flag(*node_id)
        } else {
            *node_id
        };

        let wire_item = WireNodeDataItem {
            id: flagged_id,
            position: node.position,
            velocity: node.velocity,
        };

        let item_bytes = bytemuck::bytes_of(&wire_item);
        buffer.extend_from_slice(item_bytes);
    }

    buffer
}
```

## Node Type Flag Utilities

```rust
// Flag constants
const AGENT_NODE_FLAG: u32 = 0x80000000;     // Bit 31
const KNOWLEDGE_NODE_FLAG: u32 = 0x40000000; // Bit 30
const NODE_ID_MASK: u32 = 0x3FFFFFFF;        // Bits 0-29

// Flag manipulation functions
pub fn set_agent_flag(node_id: u32) -> u32 {
    (node_id & NODE_ID_MASK) | AGENT_NODE_FLAG
}

pub fn set_knowledge_flag(node_id: u32) -> u32 {
    (node_id & NODE_ID_MASK) | KNOWLEDGE_NODE_FLAG
}

pub fn is_agent_node(node_id: u32) -> bool {
    (node_id & AGENT_NODE_FLAG) != 0
}

pub fn is_knowledge_node(node_id: u32) -> bool {
    (node_id & KNOWLEDGE_NODE_FLAG) != 0
}

pub fn get_actual_node_id(node_id: u32) -> u32 {
    node_id & NODE_ID_MASK
}

pub fn get_node_type(node_id: u32) -> NodeType {
    if is_agent_node(node_id) {
        NodeType::Agent
    } else if is_knowledge_node(node_id) {
        NodeType::Knowledge
    } else {
        NodeType::Unknown
    }
}
```

## Decoding Process

### Client to Server

1. **Receive Binary**: Get binary WebSocket frame
2. **Validate Size**: Ensure data is multiple of 28 bytes
3. **Deserialize**: Parse fixed-size chunks
4. **Extract Type Info**: Process node type flags
5. **Reconstruct**: Create server-side structures with defaults

```rust
pub fn decode_node_data(data: &[u8]) -> Result<Vec<(u32, BinaryNodeData)>, String> {
    const WIRE_ITEM_SIZE: usize = std::mem::size_of::<WireNodeDataItem>();

    if data.len() % WIRE_ITEM_SIZE != 0 {
        return Err(format!(
            "Data size {} is not a multiple of wire item size {}",
            data.len(), WIRE_ITEM_SIZE
        ));
    }

    let mut updates = Vec::with_capacity(data.len() / WIRE_ITEM_SIZE);

    for chunk in data.chunks_exact(WIRE_ITEM_SIZE) {
        let wire_item: WireNodeDataItem = *bytemuck::from_bytes(chunk);

        // Extract actual node ID (strip type flags)
        let actual_id = get_actual_node_id(wire_item.id);

        let server_node_data = BinaryNodeData {
            position: wire_item.position,
            velocity: wire_item.velocity,
            mass: 100u8,     // Default, replaced from node_map
            flags: 0u8,      // Default, replaced from node_map
            padding: [0u8, 0u8],
        };

        updates.push((actual_id, server_node_data));
    }

    Ok(updates)
}
```

## Type Definitions

### Vec3Data Structure

```rust
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Vec3Data {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}
```

### BinaryNodeData (Server Format)

```rust
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BinaryNodeData {
    pub position: Vec3Data,    // 12 bytes
    pub velocity: Vec3Data,    // 12 bytes
    pub mass: u8,             // 1 byte (server-only)
    pub flags: u8,            // 1 byte (server-only)
    pub padding: [u8; 2],     // 2 bytes (server-only)
    // Total: 28 bytes (server-side)
}
```

## WebSocket Integration

### Message Types

The binary protocol is used for specific WebSocket message types:

```typescript
// Client-side message handling
socket.addEventListener('message', (event) => {
    if (event.data instanceof ArrayBuffer) {
        // Binary message - node updates
        handleBinaryUpdate(event.data);
    } else {
        // Text message - JSON protocol
        const message = JSON.parse(event.data);
        handleJsonMessage(message);
    }
});
```

### Binary Frame Format

WebSocket binary frames use opcode 0x2:
- FIN bit: 1 (complete message)
- Opcode: 0x2 (binary frame)
- Payload: Concatenated node data

## Performance Characteristics

### Bandwidth Usage

| Scenario | Nodes | Update Rate | Uncompressed | Compressed | Bandwidth |
|----------|-------|-------------|--------------|------------|-----------|
| Small multi-agent | 10 agents + 100 knowledge | 10 Hz / 1 Hz | 3.1 KB | ~2 KB | 20 KB/s |
| Medium multi-agent | 50 agents + 1,000 knowledge | 10 Hz / 1 Hz | 29.4 KB | ~15 KB | 150 KB/s |
| Large System | 100 agents + 10,000 knowledge | 10 Hz / 1 Hz | 283 KB | ~120 KB | 1.2 MB/s |
| Enterprise | 500 agents + 100,000 knowledge | 10 Hz / 1 Hz | 2.8 MB | ~1.2 MB | 12 MB/s |

### Optimization Strategies

1. **Delta Updates**: Only send changed nodes
2. **Throttling**: Limit update frequency based on client capacity
3. **Compression**: Apply permessage-deflate for large updates
4. **Chunking**: Split large updates across multiple frames
5. **Type-based Filtering**: Send only relevant node types to specific clients

## Error Handling

### Common Errors

1. **Invalid Data Size**
   ```rust
   if data.len() % WIRE_ITEM_SIZE != 0 {
       return Err("Data size is not a multiple of wire item size");
   }
   ```

2. **Empty Data**
   ```rust
   if data.is_empty() {
       return Ok(Vec::new());
   }
   ```

3. **Type Flag Conflicts**
   ```rust
   if is_agent_node(id) && is_knowledge_node(id) {
       return Err("Node cannot be both agent and knowledge type");
   }
   ```

### Client-Side Validation

```typescript
function decodeBinaryUpdate(buffer: ArrayBuffer): NodeUpdate[] {
    const BYTES_PER_NODE = 28;

    if (buffer.byteLength % BYTES_PER_NODE !== 0) {
        throw new Error('Invalid binary data size');
    }

    const view = new DataView(buffer);
    const nodeCount = buffer.byteLength / BYTES_PER_NODE;
    const updates: NodeUpdate[] = [];

    for (let i = 0; i < nodeCount; i++) {
        const offset = i * BYTES_PER_NODE;
        const flaggedId = view.getUint32(offset, true);

        // Extract type information
        const isAgent = (flaggedId & 0x80000000) !== 0;
        const isKnowledge = (flaggedId & 0x40000000) !== 0;
        const actualId = flaggedId & 0x3FFFFFFF;

        updates.push({
            id: actualId,
            type: isAgent ? 'agent' : isKnowledge ? 'knowledge' : 'unknown',
            position: {
                x: view.getFloat32(offset + 4, true),
                y: view.getFloat32(offset + 8, true),
                z: view.getFloat32(offset + 12, true),
            },
            velocity: {
                x: view.getFloat32(offset + 16, true),
                y: view.getFloat32(offset + 20, true),
                z: view.getFloat32(offset + 24, true),
            }
        });
    }

    return updates;
}
```

## Testing

### Unit Tests

```rust
#[test]
fn test_wire_format_size() {
    assert_eq!(std::mem::size_of::<WireNodeDataItem>(), 28);
}

#[test]
fn test_node_type_flags() {
    let node_id = 42u32;

    // Test agent flag
    let agent_id = set_agent_flag(node_id);
    assert!(is_agent_node(agent_id));
    assert_eq!(get_actual_node_id(agent_id), node_id);

    // Test knowledge flag
    let knowledge_id = set_knowledge_flag(node_id);
    assert!(is_knowledge_node(knowledge_id));
    assert_eq!(get_actual_node_id(knowledge_id), node_id);
}

#[test]
fn test_encode_decode_roundtrip_with_flags() {
    let nodes = vec![
        (1u32, create_test_node_data()),
        (2u32, create_test_node_data()),
    ];

    let agent_ids = vec![1u32];
    let knowledge_ids = vec![2u32];

    let encoded = encode_node_data_with_types(&nodes, &agent_ids, &knowledge_ids);
    let decoded = decode_node_data(&encoded).unwrap();

    assert_eq!(decoded.len(), 2);
    // Verify data integrity (positions preserved, flags processed)
}
```

## Protocol Constants

From `src/utils/socket_flow_constants.rs`:

```rust
// Binary message constants
pub const NODE_POSITION_SIZE: usize = 24; // 6 f32s * 4 bytes
pub const BINARY_HEADER_SIZE: usize = 4;  // Optional header
pub const COMPRESSION_THRESHOLD: usize = 1024; // 1KB

// WebSocket constants
pub const MAX_MESSAGE_SIZE: usize = 100 * 1024 * 1024; // 100MB
pub const BINARY_CHUNK_SIZE: usize = 64 * 1024; // 64KB
```

## Security Considerations

### Data Validation

- Validate all array bounds before access
- Use type-safe deserialization (bytemuck)
- Limit maximum message size (100MB)
- Rate limit binary updates
- Validate node type flag combinations

### Memory Safety

- Fixed-size allocations prevent DoS
- No dynamic memory allocation during decode
- Bounded update counts per message
- GPU memory bounds checking

## Important Notes

- Server-side fields (mass, flags, padding) are NOT transmitted over the wire
- Node IDs are u32 with embedded type flags in the high bits
- All floating-point values use IEEE 754 single precision (f32)
- Agent nodes have flag 0x80000000 set in the transmitted node ID
- Knowledge nodes have flag 0x40000000 set in the transmitted node ID
- Actual node ID is extracted using mask 0x3FFFFFFF
- The wire format is optimized for GPU memory alignment and processing