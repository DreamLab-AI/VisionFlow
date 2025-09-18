# VisionFlow WebSocket Binary Protocol v2.0

*Last Updated: 2025-09-18*

## Overview

The VisionFlow WebSocket protocol provides real-time, bidirectional streaming of comprehensive graph data including positions, velocities, voice, control bits, and shortest path calculations for all nodes across both Logseq knowledge graphs and AI agent swarms.

## Protocol Philosophy

**WebSocket handles ALL real-time data streams:**
- Position and velocity updates for every node in every graph
- Voice data streaming (bidirectional)
- Control bits and system flags
- SSSP (Shortest Single-Source Path) real-time calculations
- Complete graph state synchronisation

**REST API handles complex operations:**
- Swarm creation and management
- Task orchestration and submission
- Configuration changes
- Authentication and authorisation
- Metadata queries

## Binary Protocol Specification

### Frame Structure

All WebSocket messages use a binary frame format optimised for real-time performance:

```
Frame Header (8 bytes):
[0-1]   : Message Type (u16)
[2-3]   : Frame Length (u16) - excluding header
[4-7]   : Timestamp (u32) - milliseconds since epoch

Payload:
[8+]    : Message-specific binary data
```

### Message Types

| Type | Value | Description |
|------|-------|-------------|
| NODE_UPDATE | 0x0001 | Position/velocity updates for nodes |
| VOICE_FRAME | 0x0002 | Opus-encoded audio data |
| CONTROL_MSG | 0x0003 | System control and flags |
| SSSP_UPDATE | 0x0004 | Shortest path calculations |
| SYNC_STATE | 0x0005 | Full graph state synchronisation |
| HEARTBEAT | 0x0006 | Connection health monitoring |

## Node Update Protocol (Type 0x0001)

### Single Node Update (40 bytes)

```
Bytes 0-3:   Node ID (u32, little-endian)
Bytes 4-7:   Graph Type (u32)
              0 = Logseq Knowledge Node
              1 = AI Agent Node  
              2 = Hybrid/Linked Node
              3 = System/Control Node
Bytes 8-11:  X Position (f32, IEEE 754)
Bytes 12-15: Y Position (f32, IEEE 754)
Bytes 16-19: Z Position (f32, IEEE 754)
Bytes 20-23: X Velocity (f32, IEEE 754)
Bytes 24-27: Y Velocity (f32, IEEE 754)
Bytes 28-31: Z Velocity (f32, IEEE 754)
Bytes 32-35: Control Bits/State Flags (u32)
Bytes 36-39: SSSP Distance/Weight (f32, IEEE 754)
```

### Batch Node Updates

For efficiency, multiple node updates can be sent in a single frame:

```
Frame Header (8 bytes)
Node Count (u16, 2 bytes)
Reserved (u16, 2 bytes) - for future use
Node Data (40 bytes × node count)
```

### Control Bits Specification (Bytes 32-35)

```
Bit 0-7:   Node State
           0x00 = Idle
           0x01 = Active
           0x02 = Spawning
           0x03 = Terminating
           0x04 = Error
           0x05 = Paused
           0x06 = Selected
           0x07 = Highlighted

Bit 8-15:  Agent-Specific Flags (for Agent nodes)
           0x00 = Ready
           0x01 = Processing Task
           0x02 = Waiting for Input
           0x03 = Communicating
           0x04 = Learning
           0x05 = Coordinating
           0x06 = Spawning Child
           0x07 = Reporting

Bit 16-23: Knowledge Graph Flags (for Logseq nodes)
           0x00 = Static
           0x01 = Recently Modified
           0x02 = Has Backlinks
           0x03 = Tagged
           0x04 = Search Result
           0x05 = Favourited
           0x06 = Archive
           0x07 = External Link

Bit 24-31: System Flags
           0x00 = Normal
           0x01 = Debug Mode
           0x02 = Performance Monitor
           0x03 = Network Issue
           0x04 = High Load
           0x05 = Synchronising
           0x06 = Backup in Progress
           0x07 = Emergency Stop
```

## Voice Protocol (Type 0x0002)

### Voice Frame Structure

```
Bytes 0-3:   Session ID (u32)
Bytes 4-7:   Sequence Number (u32)
Bytes 8-11:  Audio Length (u32)
Bytes 12-15: Encoding Info (u32)
             Bit 0-7:   Codec (0x01 = Opus, 0x02 = PCM)
             Bit 8-15:  Sample Rate (0x01 = 16kHz, 0x02 = 48kHz)
             Bit 16-23: Channels (0x01 = Mono, 0x02 = Stereo)
             Bit 24-31: Quality (0x01 = Low, 0x02 = High)
Bytes 16+:   Audio Data (Opus-encoded frames)
```

### Voice Features

- **Bidirectional Streaming**: Both client-to-server and server-to-client
- **Opus Encoding**: High-quality, low-latency audio compression
- **Session Management**: Multiple concurrent voice sessions
- **Echo Cancellation**: Built-in acoustic echo cancellation
- **Noise Suppression**: Real-time noise reduction

## Control Message Protocol (Type 0x0003)

### Control Frame Structure

```
Bytes 0-3:   Command Type (u32)
Bytes 4-7:   Target ID (u32) - Node, Swarm, or System ID
Bytes 8-11:  Parameter Count (u32)
Bytes 12+:   Parameters (variable length)
```

### Command Types

| Command | Value | Description |
|---------|-------|-------------|
| SET_STATE | 0x0001 | Change node state |
| HIGHLIGHT | 0x0002 | Highlight specific nodes |
| FOCUS_VIEW | 0x0003 | Centre camera on node |
| TOGGLE_PHYSICS | 0x0004 | Enable/disable physics |
| ADJUST_FORCES | 0x0005 | Modify physics parameters |
| START_RECORDING | 0x0006 | Begin voice recording |
| STOP_RECORDING | 0x0007 | End voice recording |
| EMERGENCY_STOP | 0x0008 | Halt all operations |

## SSSP Update Protocol (Type 0x0004)

### Shortest Path Frame Structure

```
Bytes 0-3:   Source Node ID (u32)
Bytes 4-7:   Algorithm Type (u32)
             0x01 = Dijkstra
             0x02 = A*
             0x03 = Floyd-Warshall
             0x04 = Bellman-Ford
Bytes 8-11:  Path Count (u32)
Bytes 12+:   Path Data (variable length)
```

### Path Data Format

For each path:
```
Bytes 0-3:   Destination Node ID (u32)
Bytes 4-7:   Total Distance (f32)
Bytes 8-11:  Hop Count (u32)
Bytes 12+:   Node IDs in path (u32 × hop count)
```

## State Synchronisation Protocol (Type 0x0005)

### Full Sync Frame Structure

```
Bytes 0-3:   Sync Version (u32)
Bytes 4-7:   Graph Count (u32)
Bytes 8+:    Graph Data (variable length)
```

### Graph Data Format

For each graph:
```
Bytes 0-3:   Graph ID (u32)
Bytes 4-7:   Graph Type (u32)
Bytes 8-11:  Node Count (u32)
Bytes 12-15: Edge Count (u32)
Bytes 16+:   Node Data (40 bytes × node count)
Bytes X+:    Edge Data (16 bytes × edge count)
```

### Edge Data Format

```
Bytes 0-3:   Source Node ID (u32)
Bytes 4-7:   Target Node ID (u32)
Bytes 8-11:  Edge Weight (f32)
Bytes 12-15: Edge Type/Flags (u32)
```

## Flow Control and Backpressure

### Client Buffering Strategy

```typescript
interface BufferManager {
    maxBufferSize: number;      // Maximum buffer size (default: 1MB)
    dropPolicy: 'oldest' | 'newest' | 'priority';
    compressionEnabled: boolean;
    batchSize: number;          // Nodes per batch (default: 50)
}
```

### Server Rate Limiting

- **Position Updates**: 60 FPS maximum (16.67ms intervals)
- **Voice Frames**: Real-time (no artificial limiting)
- **Control Messages**: 100 messages/second maximum
- **SSSP Updates**: 10 calculations/second maximum
- **Sync Messages**: 1 full sync/minute maximum

## Error Handling

### Error Frame Structure (Type 0x0007)

```
Bytes 0-3:   Error Code (u32)
Bytes 4-7:   Error Category (u32)
Bytes 8-11:  Context Length (u32)
Bytes 12+:   Error Context (UTF-8 string)
```

### Error Codes

| Code | Category | Description |
|------|----------|-------------|
| 0x1001 | PROTOCOL | Invalid message type |
| 0x1002 | PROTOCOL | Malformed frame |
| 0x1003 | PROTOCOL | Version mismatch |
| 0x2001 | DATA | Invalid node ID |
| 0x2002 | DATA | Graph not found |
| 0x2003 | DATA | Insufficient permissions |
| 0x3001 | SYSTEM | Server overloaded |
| 0x3002 | SYSTEM | GPU compute error |
| 0x3003 | SYSTEM | Memory exhausted |

## Performance Characteristics

### Bandwidth Utilisation

- **100 Active Nodes**: ~240 KB/s (40 bytes × 100 × 60 FPS)
- **500 Active Nodes**: ~1.2 MB/s
- **1000 Active Nodes**: ~2.4 MB/s
- **Voice Stream**: ~32 KB/s per session (Opus @ 32kbps)

### Latency Targets

- **Node Updates**: < 5ms end-to-end
- **Voice Transmission**: < 50ms end-to-end
- **Control Commands**: < 2ms response time
- **SSSP Calculations**: < 100ms for 1000+ node graphs

### Compression Benefits

| Data Type | Uncompressed (JSON) | Binary Protocol | Reduction |
|-----------|-------------------|-----------------|-----------|
| Node Update | 220 bytes | 40 bytes | 82% |
| Voice Frame | 1024 bytes | 512 bytes | 50% |
| Control Command | 150 bytes | 16 bytes | 89% |
| SSSP Path | 300 bytes | 64 bytes | 79% |

## Implementation Example

### Client-Side TypeScript

```typescript
class VisionFlowWebSocket {
    private ws: WebSocket;
    private nodeBuffer: Map<number, NodeData> = new Map();
    
    private handleBinaryMessage(data: ArrayBuffer) {
        const view = new DataView(data);
        const messageType = view.getUint16(0, true);
        const frameLength = view.getUint16(2, true);
        const timestamp = view.getUint32(4, true);
        
        switch (messageType) {
            case 0x0001: // NODE_UPDATE
                this.handleNodeUpdate(data.slice(8));
                break;
            case 0x0002: // VOICE_FRAME
                this.handleVoiceFrame(data.slice(8));
                break;
            case 0x0003: // CONTROL_MSG
                this.handleControlMessage(data.slice(8));
                break;
        }
    }
    
    private handleNodeUpdate(payload: ArrayBuffer) {
        const view = new DataView(payload);
        
        const nodeId = view.getUint32(0, true);
        const graphType = view.getUint32(4, true);
        const position = {
            x: view.getFloat32(8, true),
            y: view.getFloat32(12, true),
            z: view.getFloat32(16, true)
        };
        const velocity = {
            x: view.getFloat32(20, true),
            y: view.getFloat32(24, true),
            z: view.getFloat32(28, true)
        };
        const controlBits = view.getUint32(32, true);
        const sspDistance = view.getFloat32(36, true);
        
        this.nodeBuffer.set(nodeId, {
            nodeId, graphType, position, velocity,
            controlBits, sspDistance
        });
        
        this.updateVisualization(nodeId);
    }
}
```

### Server-Side Rust

```rust
impl GraphActor {
    async fn broadcast_node_updates(&self, nodes: &[NodeData]) {
        let mut buffer = Vec::with_capacity(8 + 4 + nodes.len() * 40);
        
        // Frame header
        buffer.extend_from_slice(&(0x0001u16).to_le_bytes()); // NODE_UPDATE
        buffer.extend_from_slice(&((4 + nodes.len() * 40) as u16).to_le_bytes());
        buffer.extend_from_slice(&(SystemTime::now().duration_since(UNIX_EPOCH)
            .unwrap().as_millis() as u32).to_le_bytes());
        
        // Node count
        buffer.extend_from_slice(&(nodes.len() as u16).to_le_bytes());
        buffer.extend_from_slice(&0u16.to_le_bytes()); // Reserved
        
        // Node data
        for node in nodes {
            buffer.extend_from_slice(&node.id.to_le_bytes());
            buffer.extend_from_slice(&node.graph_type.to_le_bytes());
            buffer.extend_from_slice(&node.position.x.to_le_bytes());
            buffer.extend_from_slice(&node.position.y.to_le_bytes());
            buffer.extend_from_slice(&node.position.z.to_le_bytes());
            buffer.extend_from_slice(&node.velocity.x.to_le_bytes());
            buffer.extend_from_slice(&node.velocity.y.to_le_bytes());
            buffer.extend_from_slice(&node.velocity.z.to_le_bytes());
            buffer.extend_from_slice(&node.control_bits.to_le_bytes());
            buffer.extend_from_slice(&node.sssp_distance.to_le_bytes());
        }
        
        self.broadcast_binary(buffer).await;
    }
}
```

## Security Considerations

### Authentication

- WebSocket connection requires valid JWT token
- Token validation on every frame (cached for performance)
- Automatic disconnection on token expiry

### Rate Limiting

- Per-client message rate limiting
- Bandwidth throttling for excessive clients
- Priority queuing for control messages

### Data Validation

- All binary data validated before processing
- Range checking for positions and velocities
- Malformed frame detection and rejection

---

*This binary protocol enables VisionFlow to stream comprehensive real-time graph data whilst maintaining low latency and high throughput for immersive visualisation experiences.*