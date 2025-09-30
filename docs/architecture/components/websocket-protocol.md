# WebSocket Binary Protocol

**Version**: 1.2.0
**Status**: Production Ready
**Performance**: 85% bandwidth reduction achieved

## Overview

The WebSocket binary protocol provides high-performance real-time communication between the Rust backend and TypeScript clients. The protocol achieves significant bandwidth reduction through custom binary serialisation and selective compression.

## Protocol Architecture

### Connection Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Connecting
    Connecting --> Connected: Handshake Success
    Connecting --> Failed: Handshake Failure
    Connected --> Authenticating: Send Auth
    Authenticating --> Authenticated: Auth Success
    Authenticating --> Failed: Auth Failure
    Authenticated --> Active: Ready
    Active --> Active: Message Exchange
    Active --> Reconnecting: Connection Lost
    Reconnecting --> Connecting: Retry
    Active --> Closed: Graceful Shutdown
    Failed --> [*]
    Closed --> [*]
```

### Message Frame Structure

All WebSocket messages follow this binary frame structure:

```
┌─────────────────────────────────────────┐
│          Frame Header (8 bytes)         │
├─────────┬────────┬─────────┬────────────┤
│ Version │  Type  │ Flags   │   Length   │
│ 1 byte  │ 1 byte │ 2 bytes │  4 bytes   │
├─────────┴────────┴─────────┴────────────┤
│          Payload (Variable)             │
│                                         │
│         (Binary Data)                   │
└─────────────────────────────────────────┘
```

**Header Fields**:
- **Version** (1 byte): Protocol version (current: 1)
- **Type** (1 byte): Message type identifier
- **Flags** (2 bytes): Optional flags for compression, priority, etc.
- **Length** (4 bytes): Payload length in bytes

### Message Types

| Type ID | Name | Description | Direction |
|---------|------|-------------|-----------|
| 0x01 | NodeUpdate | Node position/velocity updates | Server → Client |
| 0x02 | EdgeUpdate | Edge state changes | Server → Client |
| 0x03 | SettingsSync | Settings synchronisation | Bidirectional |
| 0x04 | AgentState | Agent status updates | Server → Client |
| 0x05 | Command | Client commands | Client → Server |
| 0x06 | Heartbeat | Connection keep-alive | Bidirectional |
| 0x07 | Error | Error notifications | Server → Client |
| 0x08 | BatchUpdate | Multiple updates batched | Server → Client |

## Binary Data Formats

### Node Update Format

Each node update uses exactly **34 bytes**:

```rust
struct WireNodeDataItem {
    id: u16,                // 2 bytes - With type flags in high bits
    position: Vec3Data,     // 12 bytes (3 × f32)
    velocity: Vec3Data,     // 12 bytes (3 × f32)
    sssp_distance: f32,     // 4 bytes - SSSP distance
    sssp_parent: i32,       // 4 bytes - Parent for path reconstruction
}
```

**Total**: 34 bytes per node (vs ~150 bytes JSON)

### Vec3Data Format

3D vectors are serialised as three consecutive 32-bit floats:

```
┌────────┬────────┬────────┐
│   X    │   Y    │   Z    │
│ 4 bytes│ 4 bytes│ 4 bytes│
└────────┴────────┴────────┘
```

**Byte order**: Little-endian
**Precision**: IEEE 754 single precision

### Settings Update Format

Settings use a variable-length format with path compression:

```
┌──────────┬────────────┬─────────────┬──────────┐
│ Path ID  │ Value Type │ Value Length│  Value   │
│ 2 bytes  │  1 byte    │  2 bytes    │ Variable │
└──────────┴────────────┴─────────────┴──────────┘
```

**Path ID**: Pre-registered common paths (e.g., physics parameters)
**Value Types**:
- `0x01`: i32 (4 bytes)
- `0x02`: i64 (8 bytes)
- `0x03`: f32 (4 bytes)
- `0x04`: f64 (8 bytes)
- `0x05`: String (length-prefixed)
- `0x06`: Bytes (length-prefixed)
- `0x07`: Array (length-prefixed, recursive)
- `0x08`: Object (key-value pairs, recursive)

### Batch Update Format

Batch updates combine multiple node updates:

```
┌────────────┬──────────┬─────────────────┐
│ Node Count │ Flags    │  Node Updates   │
│  2 bytes   │ 2 bytes  │  N × 34 bytes   │
└────────────┴──────────┴─────────────────┘
```

**Batch size**: Up to 50 nodes per batch (default)
**Frequency**: 5Hz (200ms intervals)

## Compression Strategy

### Selective Compression

Compression is applied selectively based on message size and type:

```rust
fn should_compress(payload: &[u8], msg_type: u8) -> bool {
    // Settings messages over 256 bytes
    if msg_type == 0x03 && payload.len() > 256 {
        return true;
    }

    // Batch updates over 1KB
    if msg_type == 0x08 && payload.len() > 1024 {
        return true;
    }

    false
}
```

**Compression Algorithm**: GZIP (level 6)
**Typical Ratios**:
- Settings messages: 70-85% reduction
- Large batch updates: 40-60% reduction
- Small updates: No compression (overhead exceeds benefit)

### Delta Encoding

Settings use delta encoding to transmit only changed values:

```rust
struct DeltaUpdate {
    changed_count: u16,        // Number of changed settings
    changes: Vec<SettingChange>
}

struct SettingChange {
    path_id: u16,
    value_type: u8,
    value: BinaryValue
}
```

**Change Detection**: Blake3 hashing for efficient delta identification
**Full Sync**: Every 10th update or on reconnection

## Rate Limiting

### Client-Side Batching

```typescript
const batchConfig: BatchQueueConfig = {
    batchSize: 50,           // Max items per batch
    flushIntervalMs: 200,    // 5Hz update rate
    maxQueueSize: 1000,      // Memory bounds
    priorityField: 'nodeId'  // Priority processing
};
```

### Server-Side Rate Limits

```rust
pub fn socket_flow_updates() -> RateLimitConfig {
    RateLimitConfig {
        requests_per_minute: 300,  // 5Hz × 60s
        burst_size: 50,
        cleanup_interval: Duration::from_secs(600),
        ban_duration: Duration::from_secs(600),
        max_violations: 10,
    }
}
```

## Connection Management

### Heartbeat Protocol

```mermaid
sequenceDiagram
    participant Client
    participant Server

    loop Every 30 seconds
        Client->>Server: Ping (0x06)
        Server-->>Client: Pong (0x06)
    end

    Note over Client,Server: If no Pong within 45s
    Client->>Client: Mark Connection Dead
    Client->>Client: Attempt Reconnection
```

**Ping Interval**: 30 seconds
**Pong Timeout**: 45 seconds
**Reconnection Strategy**: Exponential backoff (1s to 30s max)

### Reconnection Flow

```rust
fn calculate_backoff(attempt: u32) -> Duration {
    let base_delay = 1000; // 1 second
    let exponential_delay = base_delay * 2u64.pow(attempt - 1);
    let max_delay = 30000; // 30 seconds

    Duration::from_millis(exponential_delay.min(max_delay))
}
```

### Connection Resilience

```typescript
class WebSocketService {
    private reconnect(): void {
        const delay = this.calculateBackoff();

        setTimeout(() => {
            this.connect();
        }, delay);

        this.reconnectAttempts++;
    }

    private queueMessage(message: BinaryMessage): void {
        // Queue messages during disconnection
        if (this.messageQueue.length < MAX_QUEUE_SIZE) {
            this.messageQueue.push(message);
        }
    }
}
```

## Performance Characteristics

### Bandwidth Reduction

| Message Type | JSON Size | Binary Size | Reduction |
|--------------|-----------|-------------|-----------|
| Single Node | ~150 bytes | 34 bytes | 77% |
| 50 Nodes | ~7.5 KB | 1.7 KB | 77% |
| Settings Update | ~800 bytes | ~120 bytes | 85% |
| Batch (compressed) | ~10 KB | ~2 KB | 80% |

**Overall Achieved**: 84.8% bandwidth reduction

### Latency Metrics

| Operation | P50 | P95 | P99 |
|-----------|-----|-----|-----|
| Serialisation | 0.2ms | 0.5ms | 1.0ms |
| Deserialisation | 0.3ms | 0.7ms | 1.2ms |
| Network RTT | 8ms | 18ms | 30ms |
| End-to-End | 10ms | 20ms | 35ms |

### Throughput

- **Messages/second**: 300 sustained, 600 burst
- **Bytes/second**: 500 KB sustained, 1 MB burst
- **Nodes/update**: Up to 100 (typical: 20-50)

## Error Handling

### Error Frame Format

```
┌──────────┬──────────┬───────────────┬────────────┐
│Error Code│ Severity │ Message Length│  Message   │
│ 2 bytes  │  1 byte  │   2 bytes     │  Variable  │
└──────────┴──────────┴───────────────┴────────────┘
```

### Error Codes

| Code | Name | Description | Recovery |
|------|------|-------------|----------|
| 0x0001 | ParseError | Failed to parse message | Log and discard |
| 0x0002 | AuthFailure | Authentication failed | Reconnect with auth |
| 0x0003 | RateLimitExceeded | Too many requests | Backoff and retry |
| 0x0004 | InvalidState | Invalid connection state | Reset connection |
| 0x0005 | InternalError | Server internal error | Retry with backoff |

### Error Severity Levels

- **0x01 - Info**: Informational, no action required
- **0x02 - Warning**: Warning condition, may require attention
- **0x03 - Error**: Error condition, requires retry
- **0x04 - Critical**: Critical failure, connection termination

## Implementation Examples

### Rust Server Implementation

```rust
use actix_web_actors::ws;

impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for WebSocketActor {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        match msg {
            Ok(ws::Message::Binary(bin)) => {
                // Parse binary frame
                if let Ok(frame) = BinaryFrame::parse(&bin) {
                    self.handle_binary_frame(frame, ctx);
                }
            }
            Ok(ws::Message::Ping(msg)) => {
                ctx.pong(&msg);
            }
            _ => {}
        }
    }
}

fn serialize_node_update(nodes: &[NodeData]) -> Vec<u8> {
    let mut buffer = Vec::with_capacity(8 + nodes.len() * 34);

    // Write header
    buffer.push(1); // Version
    buffer.push(0x01); // NodeUpdate type
    buffer.extend_from_slice(&0u16.to_le_bytes()); // Flags
    buffer.extend_from_slice(&((nodes.len() * 34) as u32).to_le_bytes());

    // Write node data
    for node in nodes {
        buffer.extend_from_slice(&node.id.to_le_bytes());
        buffer.extend_from_slice(&node.position.x.to_le_bytes());
        buffer.extend_from_slice(&node.position.y.to_le_bytes());
        buffer.extend_from_slice(&node.position.z.to_le_bytes());
        // ... velocity and SSSP data
    }

    buffer
}
```

### TypeScript Client Implementation

```typescript
class BinaryProtocolClient {
    private parseFrame(data: ArrayBuffer): BinaryFrame {
        const view = new DataView(data);
        let offset = 0;

        const version = view.getUint8(offset++);
        const type = view.getUint8(offset++);
        const flags = view.getUint16(offset, true); offset += 2;
        const length = view.getUint32(offset, true); offset += 4;

        const payload = new Uint8Array(data, offset, length);

        return { version, type, flags, length, payload };
    }

    private parseNodeUpdate(payload: Uint8Array): NodeData[] {
        const view = new DataView(payload.buffer, payload.byteOffset);
        const nodes: NodeData[] = [];
        let offset = 0;

        while (offset < payload.length) {
            const id = view.getUint16(offset, true); offset += 2;
            const x = view.getFloat32(offset, true); offset += 4;
            const y = view.getFloat32(offset, true); offset += 4;
            const z = view.getFloat32(offset, true); offset += 4;
            const vx = view.getFloat32(offset, true); offset += 4;
            const vy = view.getFloat32(offset, true); offset += 4;
            const vz = view.getFloat32(offset, true); offset += 4;
            const sssp_distance = view.getFloat32(offset, true); offset += 4;
            const sssp_parent = view.getInt32(offset, true); offset += 4;

            nodes.push({ id, position: {x,y,z}, velocity: {vx,vy,vz}, sssp_distance, sssp_parent });
        }

        return nodes;
    }
}
```

## Testing and Validation

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_serialization_roundtrip() {
        let original = NodeData {
            id: 42,
            position: Vec3::new(1.0, 2.0, 3.0),
            velocity: Vec3::new(0.1, 0.2, 0.3),
            sssp_distance: 5.5,
            sssp_parent: 10,
        };

        let serialized = serialize_node(&original);
        let deserialized = deserialize_node(&serialized).unwrap();

        assert_eq!(original, deserialized);
        assert_eq!(serialized.len(), 34);
    }
}
```

### Integration Tests

```typescript
describe('BinaryProtocol', () => {
    it('should handle node updates correctly', async () => {
        const client = new BinaryProtocolClient();
        await client.connect();

        const updates = await client.waitForNodeUpdate();
        expect(updates).toHaveLength(50);
        expect(updates[0]).toHaveProperty('position');
    });
});
```

## Performance Optimisation Tips

### Server-Side Optimisations

1. **Pre-allocate Buffers**: Reuse buffers for serialisation
2. **Batch Updates**: Combine multiple small updates
3. **Zero-Copy**: Use `bytes::Bytes` for zero-copy operations
4. **Compression Threshold**: Only compress large messages
5. **Connection Pooling**: Reuse WebSocket connections

### Client-Side Optimisations

1. **ArrayBuffer Pooling**: Reuse ArrayBuffers for parsing
2. **Worker Threads**: Parse in Web Workers for large updates
3. **Incremental Updates**: Apply updates incrementally to UI
4. **Throttling**: Limit UI updates to 60 FPS
5. **Priority Queues**: Process critical updates first

## Monitoring and Debugging

### Performance Metrics

```rust
struct WebSocketMetrics {
    messages_sent: AtomicU64,
    messages_received: AtomicU64,
    bytes_sent: AtomicU64,
    bytes_received: AtomicU64,
    compression_ratio: AtomicF64,
    delta_messages: AtomicU64,
    full_sync_messages: AtomicU64,
}
```

### Debug Logging

Enable debug logging for protocol inspection:

```rust
RUST_LOG=visionflow::websocket=debug cargo run
```

### Browser DevTools

Monitor WebSocket traffic in Chrome DevTools:
1. Network tab → WS filter
2. View frame data with binary inspector
3. Monitor bandwidth usage in real-time

## Security Considerations

### Authentication

```rust
impl WebSocketActor {
    fn authenticate(&self, token: &str) -> Result<UserId, AuthError> {
        // Verify JWT token
        // Check token expiration
        // Validate user permissions
    }
}
```

### Input Validation

All binary input is validated before processing:

```rust
fn validate_frame(frame: &BinaryFrame) -> Result<(), ValidationError> {
    // Check version compatibility
    // Validate message type
    // Verify length bounds
    // Check flag combinations
}
```

### Rate Limiting

Per-connection and per-IP rate limits prevent abuse:

```rust
pub struct RateLimiter {
    requests_per_minute: u32,
    burst_size: u32,
    violations: HashMap<IpAddr, u32>,
}
```

## Migration Guide

### From JSON to Binary Protocol

1. **Implement Binary Serialisation**: Add binary encode/decode methods
2. **Update Client**: Integrate `BinaryProtocolClient`
3. **Test Thoroughly**: Validate roundtrip serialisation
4. **Deploy Gradually**: Use feature flags for rollout
5. **Monitor Performance**: Track bandwidth and latency improvements

## References

- [Binary Protocol Specification](../../reference/binary-protocol-spec.md)
- [WebSocket API Reference](../../reference/websocket-api.md)
- [Performance Benchmarks](../../reports/performance-benchmarks.md)
- [Security Best Practices](../security.md)
