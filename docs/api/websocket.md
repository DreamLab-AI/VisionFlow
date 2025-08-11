# WebSocket API Reference

## Overview
The VisionFlow WebSocket implementation provides real-time graph updates using an optimized binary protocol alongside JSON control messages.

## Connection

Connect to: `wss://your-domain/wss` (Primary graph streaming endpoint)

### Additional WebSocket Endpoints
- `/wss` - Binary graph position updates
- `/ws/speech` - Voice interaction streaming
- `/ws/mcp-relay` - MCP protocol relay  
- `/ws/bots_visualization` - Agent swarm visualization

### Connection Flow
1. Client connects to WebSocket endpoint
2. Server sends: `{"type": "connection_established", "timestamp": <timestamp>}`
3. Client sends authentication (handled via HTTP session before WebSocket upgrade)
4. Client sends: `{"type": "requestInitialData"}`
5. Server begins binary updates (rate controlled by server settings)
6. Server sends: `{"type": "updatesStarted", "timestamp": <timestamp>}`
7. Server may send: `{"type": "loading", "message": "Calculating initial layout..."}` during processing

### Authentication

WebSocket connections in VisionFlow inherit authentication from the HTTP session established before the WebSocket upgrade. This means user authentication (e.g., via Nostr) should occur through standard HTTP endpoints before establishing the WebSocket connection.

**Session-based Authentication:**
- Session cookie sent during WebSocket handshake
- No additional authentication messages required over WebSocket
- Session validates user permissions and feature access

**Token-based Alternative:**
```javascript
const ws = new WebSocket('wss://your-domain/wss?token=<session-token>');
```

## Message Types

### Control Messages (JSON)

All JSON control messages follow this structure:
```typescript
interface WebSocketMessage {
  type: string;
  payload?: any;
  timestamp?: number;
  id?: string;
}
```

#### Server → Client Messages

**Connection Established**
```json
{
  "type": "connection_established",
  "timestamp": 1679417762000
}
```

**Updates Started**
```json
{
  "type": "updatesStarted", 
  "timestamp": 1679417763000
}
```

**Loading State**
```json
{
  "type": "loading",
  "message": "Calculating initial layout..."
}
```

**Heartbeat Response**
```json
{
  "type": "pong",
  "timestamp": 1679417764000
}
```

#### Client → Server Messages

**Request Initial Data**
```json
{
  "type": "requestInitialData"
}
```

**Heartbeat**
```json
{
  "type": "ping",
  "timestamp": 1679417764000
}
```

### Binary Messages - Position Updates

Position updates are transmitted as binary messages using a highly optimized 28-byte format per node.

#### Wire Format (28 bytes per node)

```
┌─────────────┬────────────────┬────────────────┐
│  Node ID    │    Position    │    Velocity    │
│  (4 bytes)  │   (12 bytes)   │   (12 bytes)   │
└─────────────┴────────────────┴────────────────┘
```

**Field Breakdown:**
- **Node ID**: u32 (4 bytes) - Includes type flags in high bits
- **Position**: Vec3 (12 bytes) - X, Y, Z coordinates as f32 values  
- **Velocity**: Vec3 (12 bytes) - X, Y, Z velocity components as f32 values

#### Node Type Flags

The Node ID field embeds type information in the high bits:

| Flag | Value | Type | Description |
|------|-------|------|-------------|
| 0x80000000 | Bit 31 | Agent | AI agent node |
| 0x40000000 | Bit 30 | Knowledge | Knowledge graph node |
| 0x00000000 | Default | Unknown | Standard node |

#### Binary Message Processing

**Server → Client Updates:**
1. Server continuously computes physics simulation
2. Changed node positions are batched
3. Type flags applied based on node classification
4. Data encoded as 28-byte records
5. Binary WebSocket frame sent to clients
6. Update frequency varies: 5-60 Hz based on activity

**Client → Server Updates:**
1. Client can send position updates during interaction
2. Same 28-byte binary format used
3. Server processes updates through physics system
4. Changes validated and broadcast to other clients
5. Modifications may be adjusted by physics constraints

#### Example Binary Data

For a single agent node with ID=1, position=(10.0, 20.0, 30.0), velocity=(0.1, 0.2, 0.3):

```
01 00 00 80  // Node ID: 1 with agent flag (0x80000001)
00 00 20 41  // X position: 10.0 (little-endian f32)
00 00 A0 41  // Y position: 20.0 (little-endian f32)
00 00 F0 41  // Z position: 30.0 (little-endian f32)
CD CC CC 3D  // X velocity: 0.1 (little-endian f32)
CD CC 4C 3E  // Y velocity: 0.2 (little-endian f32) 
9A 99 99 3E  // Z velocity: 0.3 (little-endian f32)
```

### Position Synchronization Protocol

VisionFlow implements bidirectional position synchronization:

1. **Server Authority**: Server maintains authoritative graph state
2. **Client Updates**: Clients can send position updates during user interactions
3. **Physics Integration**: Server applies physics constraints to all updates
4. **Broadcast**: Validated changes broadcast to all connected clients
5. **Late Join**: New clients receive complete current state upon connection

## Implementation Details

### Server-Side Configuration

The server uses dynamic update rate control based on graph activity:

```rust
// From socket_flow_constants.rs
pub const POSITION_UPDATE_RATE: u32 = 5; // Hz minimum
// Maximum rate determined by motion_threshold and activity
```

**Settings (from settings.yaml):**
- `min_update_rate`: Minimum updates per second when graph is stable
- `max_update_rate`: Maximum updates per second during high activity  
- `motion_threshold`: Sensitivity threshold for detecting node movement
- `heartbeat_interval_ms`: Client heartbeat interval

### Client-Side Handling

```typescript
// Binary message decoding
function decodeBinaryUpdate(buffer: ArrayBuffer): NodeUpdate[] {
  const BYTES_PER_NODE = 28;
  const view = new DataView(buffer);
  const updates: NodeUpdate[] = [];
  
  for (let i = 0; i < buffer.byteLength; i += BYTES_PER_NODE) {
    const flaggedId = view.getUint32(i, true);
    
    // Extract type flags
    const isAgent = (flaggedId & 0x80000000) !== 0;
    const isKnowledge = (flaggedId & 0x40000000) !== 0;
    const actualId = flaggedId & 0x3FFFFFFF;
    
    updates.push({
      id: actualId,
      type: isAgent ? 'agent' : isKnowledge ? 'knowledge' : 'unknown',
      position: {
        x: view.getFloat32(i + 4, true),
        y: view.getFloat32(i + 8, true),
        z: view.getFloat32(i + 12, true),
      },
      velocity: {
        x: view.getFloat32(i + 16, true),
        y: view.getFloat32(i + 20, true),
        z: view.getFloat32(i + 24, true),
      }
    });
  }
  
  return updates;
}

// WebSocket message handling
socket.addEventListener('message', (event) => {
  if (event.data instanceof ArrayBuffer) {
    // Binary position update
    const updates = decodeBinaryUpdate(event.data);
    handlePositionUpdates(updates);
  } else {
    // JSON control message
    const message = JSON.parse(event.data);
    handleControlMessage(message);
  }
});
```

## Optimization Features

### Binary Protocol Optimizations

1. **Fixed-Size Records**: 28 bytes per node enables fast parsing without delimiters
2. **Zero-Copy Serialization**: Uses Rust's `bytemuck` for direct memory mapping
3. **Batch Updates**: Multiple nodes in single WebSocket frame
4. **Compression**: permessage-deflate for messages >1KB
5. **Type Flags**: Node classification without additional lookups

### Performance Characteristics

| Node Count | Uncompressed Size | Compressed Size | Bandwidth (5Hz) |
|------------|-------------------|-----------------|-----------------|
| 100        | 2.8 KB           | ~2 KB           | 10 KB/s         |
| 1,000      | 28 KB            | ~15 KB          | 75 KB/s         |
| 10,000     | 280 KB           | ~120 KB         | 600 KB/s       |

### Client-Side Optimizations

- **Web Worker Processing**: Binary decoding runs off main thread
- **TypedArray Views**: Efficient binary data access
- **Object Pooling**: Reuse Vector3 instances
- **Throttled Updates**: Frame rate-based update application
- **Differential Updates**: Only changed nodes processed

## Rate Limiting & Throttling

### Server-Side Rate Control

The server implements intelligent rate limiting:

1. **Dynamic Frequency**: Update rate varies with graph activity
2. **Motion Detection**: Uses `motion_threshold` to detect significant changes
3. **Stability Mode**: Reduced updates when graph reaches equilibrium
4. **Client Capacity**: Adapts to individual client processing capabilities

### Rate Limiting Configuration

```rust
// From socket_flow_constants.rs
pub const HEARTBEAT_INTERVAL: u64 = 30; // seconds
pub const CLIENT_TIMEOUT: u64 = 60; // seconds  
pub const MAX_MESSAGE_SIZE: usize = 100 * 1024 * 1024; // 100MB
```

**Per-Client Limits:**
- Binary updates: Server-controlled throttling
- JSON messages: 1000 messages/minute
- Heartbeat timeout: 60 seconds
- Maximum message size: 100MB

## Error Handling

### Connection Issues

The `socket_flow_handler.rs` handles errors through connection management rather than structured error messages:

1. **Invalid Messages**: Connection closed for malformed data
2. **Timeout**: Client connections dropped after heartbeat timeout
3. **Protocol Violations**: Automatic disconnection for protocol breaches
4. **Resource Limits**: Connection throttling when server capacity exceeded

### Client-Side Error Detection

```typescript
// Connection monitoring
class ConnectionMonitor {
  private lastMessage = Date.now();
  private reconnectAttempts = 0;
  
  onMessage(event: MessageEvent) {
    this.lastMessage = Date.now();
    this.reconnectAttempts = 0;
  }
  
  checkConnection() {
    if (Date.now() - this.lastMessage > 60000) {
      this.reconnect();
    }
  }
  
  private async reconnect() {
    if (this.reconnectAttempts >= 5) return;
    
    const delay = Math.pow(2, this.reconnectAttempts) * 1000;
    await sleep(delay);
    
    this.reconnectAttempts++;
    this.connect();
  }
}
```

## Diagnostics

### Common Issues

1. **No Position Updates**
   - Verify WebSocket connection state
   - Check that `requestInitialData` message was sent
   - Confirm agent/knowledge node flags if expecting specific types

2. **High Latency**
   - Monitor network bandwidth utilization
   - Check if compression is enabled for large updates
   - Verify client processing keeps up with server updates

3. **Connection Drops**
   - Implement proper heartbeat/ping-pong handling  
   - Check proxy/firewall WebSocket support
   - Monitor client memory usage for leaks

4. **Binary Data Corruption**
   - Validate message size is multiple of 28 bytes
   - Verify little-endian byte order assumption
   - Check for proper ArrayBuffer handling

### Debug Logging

**Server-side:**
```bash
RUST_LOG=webxr::handlers::socket_flow_handler=debug,webxr::utils::binary_protocol=trace cargo run
```

**Client-side:**
```typescript
// Enable comprehensive logging
localStorage.setItem('debug', 'websocket:*,binary:*');

// Message size monitoring
ws.addEventListener('message', (event) => {
  if (event.data instanceof ArrayBuffer) {
    console.log(`[WS Binary] ${event.data.byteLength} bytes, ${event.data.byteLength / 28} nodes`);
  }
});
```

### Binary Message Inspection

For debugging binary protocol issues:

```bash
# Capture binary messages with hex dump
xxd -g 1 -l 84 binary_message.bin  # First 3 nodes (84 bytes)
```

Example output:
```
00000000: 01 00 00 80 00 00 20 41 00 00 a0 41 00 00 f0 41  ......A...A...A
00000010: cd cc cc 3d cd cc 4c 3e 9a 99 99 3e 02 00 00 40  ...=..L>...>...@
00000020: 00 00 80 41 00 00 c0 41 00 00 00 42 33 33 33 3e  ...A...A...B33>
```

## Integration Examples

### React Hook for Real-time Positions

```typescript
function useWebSocketPositions() {
  const [positions, setPositions] = useState(new Map());
  const [connectionState, setConnectionState] = useState('disconnected');
  
  useEffect(() => {
    const ws = new WebSocket('wss://localhost:3001/wss');
    
    ws.onopen = () => {
      setConnectionState('connected');
      ws.send(JSON.stringify({ type: 'requestInitialData' }));
    };
    
    ws.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) {
        const updates = decodeBinaryUpdate(event.data);
        setPositions(prev => {
          const newPositions = new Map(prev);
          updates.forEach(update => {
            newPositions.set(update.id, update);
          });
          return newPositions;
        });
      } else {
        const message = JSON.parse(event.data);
        if (message.type === 'connection_established') {
          console.log('WebSocket connection established');
        }
      }
    };
    
    ws.onclose = () => setConnectionState('disconnected');
    
    // Heartbeat
    const heartbeat = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'ping', timestamp: Date.now() }));
      }
    }, 30000);
    
    return () => {
      clearInterval(heartbeat);
      ws.close();
    };
  }, []);
  
  return { positions, connectionState };
}
```

### Three.js Scene Integration

```typescript
// Update 3D scene from WebSocket positions
function updateSceneFromWebSocket(scene: THREE.Scene, positions: Map<number, NodeUpdate>) {
  positions.forEach((data, id) => {
    const mesh = scene.getObjectByName(`node-${id}`);
    if (mesh) {
      // Update position
      mesh.position.set(data.position.x, data.position.y, data.position.z);
      
      // Store velocity for interpolation
      mesh.userData.velocity = data.velocity;
      
      // Apply type-specific styling
      if (data.type === 'agent') {
        (mesh.material as THREE.MeshBasicMaterial).color.setHex(0xff6b6b);
      } else if (data.type === 'knowledge') {
        (mesh.material as THREE.MeshBasicMaterial).color.setHex(0x4ecdc4);
      }
    }
  });
}
```

## Security Considerations

### Binary Protocol Security

1. **Input Validation**
   - All node IDs validated against known nodes
   - Position/velocity values bounds-checked
   - Message size limits prevent memory exhaustion

2. **Authentication**
   - Binary updates require valid session
   - Node modifications access-controlled  
   - Type flag validation prevents spoofing

3. **Rate Limiting**
   - Client update frequency throttled server-side
   - Maximum nodes per update enforced
   - Connection count limits prevent DoS

### Memory Safety

- Fixed-size allocations prevent DoS attacks
- No dynamic memory allocation during decode
- Bounded update counts per message
- GPU memory bounds checking when applicable

## Protocol Evolution

### Current Version (v1.0)
- 28-byte binary format
- u32 node IDs with type flags
- permessage-deflate compression
- Dynamic update rate control

### Future Enhancements (v2.0)
- Variable-length encoding for node IDs  
- Differential updates (position deltas only)
- Custom float precision (16-bit where appropriate)
- Protocol versioning negotiation

### Backward Compatibility
- Version detection during handshake
- Fallback to JSON for unsupported clients
- Graceful degradation of features

For detailed information on the binary protocol implementation, see [Binary Protocol Documentation](./binary-protocol.md).