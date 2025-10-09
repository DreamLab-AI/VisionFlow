# WebSocket Protocol Specification

[← Knowledge Base](../index.md) > [Reference](./index.md) > [API](./index.md) > WebSocket Protocol

**Version**: 2.0 (Binary Protocol V2)
**Status**: Production Ready
**Last Updated**: October 2025
**Performance**: ~80% bandwidth reduction vs JSON

---

## Overview

The WebSocket protocol provides high-performance real-time communication between the Rust backend and TypeScript clients. Binary Protocol V2 uses 36-byte node frames with u32 IDs and dual-graph type flags, fixing the V1 node ID truncation bug while supporting unified knowledge + agent graph broadcasting.

**WebSocket Endpoints**:
- General: `ws://localhost:3001/ws` - Binary position updates + JSON messages
- Voice Commands: `ws://localhost:3001/ws/voice` - Voice command streaming
- Agent Visualisation: `ws://localhost:3001/ws/agents` - Agent status updates
- Real-time Analytics: `ws://localhost:3001/ws/analytics` - GPU computation results

---

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

#### Establishing Connection

```javascript
// Basic connection
const ws = new WebSocket('ws://localhost:3001/ws');

// Authenticated connection
const ws = new WebSocket(`ws://localhost:3001/ws?token=${jwtToken}`);

// Connection event handlers
ws.onopen = (event) => {
  console.log('WebSocket connected');
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = (event) => {
  console.log('WebSocket closed:', event.code, event.reason);
};
```

#### Authentication

WebSocket connections support two authentication methods:

**1. Query Parameter Authentication**:
```javascript
const ws = new WebSocket(`ws://localhost:3001/ws?token=${jwtToken}`);
```

**2. First Message Authentication**:
```javascript
ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'auth',
    token: jwtToken,
    pubkey: nostrPublicKey
  }));
};
```

**Rust Server Authentication Handler**:
```rust
impl WebSocketActor {
    fn authenticate(&self, token: &str) -> Result<UserId, AuthError> {
        // Verify JWT token
        // Check token expiration
        // Validate user permissions
    }
}
```

---

## Message Frame Structure

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
- **Version** (1 byte): Protocol version (current: 2)
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

---

## Binary Data Formats

### Node Update Format (Binary Protocol V2)

Each node update uses exactly **36 bytes**:

```rust
struct WireNodeDataItemV2 {
    id: u32,                // 4 bytes - With type flags in high bits (31/30)
    position: Vec3Data,     // 12 bytes (3 × f32)
    velocity: Vec3Data,     // 12 bytes (3 × f32)
    sssp_distance: f32,     // 4 bytes - SSSP distance
    sssp_parent: i32,       // 4 bytes - Parent for path reconstruction
}
```

**Total**: 36 bytes per node (vs ~150 bytes JSON)

**Type Flags (Dual-Graph Support)**:
- **Bit 31** (`0x80000000`): Agent node flag
- **Bit 30** (`0x40000000`): Knowledge graph node flag
- **Bits 0-29** (`0x3FFFFFFF`): Actual node ID (30-bit, max 1 billion nodes)

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
│  2 bytes   │ 2 bytes  │  N × 36 bytes   │
└────────────┴──────────┴─────────────────┘
```

**Batch size**: Up to 50 nodes per batch (default)
**Frequency**: Adaptive (60 FPS active, 5 Hz settled)

---

## Dual-Graph Broadcasting Architecture

### Overview

VisionFlow supports **unified dual-graph broadcasting** where both knowledge graphs and agent graphs are transmitted in a single WebSocket stream with type identification flags.

### Graph Structure

**Knowledge Graph:**
- Scientific concepts, papers, entities
- Physics-simulated positions
- ~185 nodes typical
- Flag: `0x40000000` (bit 30)

**Agent Graph:**
- LLM agents (Claude via MCP)
- Physics-simulated positions
- ~3-10 nodes typical
- Flag: `0x80000000` (bit 31)

### Unified Broadcast Flow

```mermaid
sequenceDiagram
    participant PhysicsLoop as Physics Loop (60 FPS)
    participant GraphActor as Graph Actor
    participant Encoder as Binary Encoder
    participant Clients as WebSocket Clients

    PhysicsLoop->>GraphActor: Update node positions
    GraphActor->>GraphActor: Collect knowledge nodes
    GraphActor->>GraphActor: Collect agent nodes
    GraphActor->>Encoder: encode_node_data_with_types()
    Note over Encoder: Sets KNOWLEDGE_NODE_FLAG on knowledge IDs<br/>Sets AGENT_NODE_FLAG on agent IDs
    Encoder->>GraphActor: Binary buffer (N × 36 bytes)
    GraphActor->>Clients: Single broadcast (188 nodes)
    Clients->>Clients: Decode and separate by flags
```

### Implementation (`src/actors/graph_actor.rs`)

**Unified Physics Broadcast** (lines 2087-2150):

```rust
// Collect BOTH graphs in single broadcast
let mut position_data: Vec<(u32, BinaryNodeData)> = Vec::new();
let mut knowledge_ids: Vec<u32> = Vec::new();
let mut agent_ids: Vec<u32> = Vec::new();

// Knowledge graph nodes
for (node_id, node) in self.node_map.iter() {
    position_data.push((*node_id, BinaryNodeDataClient::new(...)));
    knowledge_ids.push(*node_id);
}

// Agent graph nodes
for node in &self.bots_graph_data.nodes {
    position_data.push((node.id, BinaryNodeDataClient::new(...)));
    agent_ids.push(node.id);
}

// Single unified broadcast with type flags
let binary_data = encode_node_data_with_types(
    &position_data,
    &agent_ids,
    &knowledge_ids
);
```

**Result:** Single WebSocket message containing all 188 nodes with proper type separation.

### Client-Side Separation

Clients decode the unified stream and separate nodes by type:

```typescript
const nodeId = view.getUint32(offset, true);
const isAgent = (nodeId & 0x80000000) !== 0;
const isKnowledge = (nodeId & 0x40000000) !== 0;
const actualId = nodeId & 0x3FFFFFFF;

if (isAgent) {
    agentGraph.updateNode(actualId, position, velocity);
} else if (isKnowledge) {
    knowledgeGraph.updateNode(actualId, position, velocity);
}
```

### Benefits

1. **No Broadcast Conflicts**: Single source of truth eliminates race conditions
2. **Synchronized Updates**: Both graphs update simultaneously
3. **Efficient Bandwidth**: One message instead of two separate streams
4. **Type Safety**: Flags ensure proper graph separation on client

---

## Client → Server Messages

### Settings Update
Updates user settings in real-time.

```json
{
  "type": "settings",
  "path": "visualisation.rendering.enable_shadows",
  "value": true
}
```

### Graph Interaction
Notifies server of user interactions with the graph.

```json
{
  "type": "graph_interaction",
  "action": "node_selected",
  "nodeId": "node_123",
  "position": { "x": 100, "y": 200, "z": 0 }
}
```

### Voice Commands
Streams voice commands through WebSocket for real-time agent coordination.

```json
{
  "type": "voice_command",
  "command": "spawn a researcher agent",
  "sessionId": "session_123",
  "userId": "user_456",
  "audioData": "base64_encoded_audio",
  "format": "wav",
  "timestamp": 1706006400000
}
```

### Agent Control
Controls agent behaviour and coordination.

```json
{
  "type": "agent_control",
  "action": "pause",
  "agentId": "agent_123"
}
```

### Subscribe/Unsubscribe
Manages event subscriptions.

```json
{
  "type": "subscribe",
  "events": ["agent_status", "task_progress"]
}
```

```json
{
  "type": "unsubscribe",
  "events": ["task_progress"]
}
```

---

## Server → Client Messages

### Binary Position Updates

Real-time physics simulation results streamed at 60 FPS from the backend.

```javascript
ws.onmessage = (event) => {
  if (event.data instanceof ArrayBuffer) {
    // Real-time binary position updates from backend physics
    const nodeUpdates = parseBinaryNodeData(event.data);
    // Apply updates directly to visualisation (no client-side physics)
    updateVisualization(nodeUpdates);
  }
};
```

The backend sends continuous streams of binary frames at 60 FPS. Client-side physics simulation is not required - only position smoothing for display.

### Settings Confirmation
Confirms settings updates.

```json
{
  "type": "settings_updated",
  "path": "visualisation.rendering.enable_shadows",
  "value": true,
  "timestamp": 1706006400000
}
```

### Agent Status Updates
Real-time agent status changes from live MCP swarms.

```json
{
  "type": "agent_status",
  "agentId": "agent_1757967065850_dv2zg7",
  "status": "active",
  "swarmId": "swarm_1757880683494_yl81sece5",
  "health": {
    "cpu": 45.2,
    "memory": 1024,
    "taskQueue": 3,
    "mcpConnected": true,
    "lastPing": "2025-01-22T10:15:30Z"
  },
  "capabilities": ["code", "review", "rust", "python"],
  "currentTask": {
    "taskId": "task_1757967065850_abc123",
    "description": "Analysing authentication module",
    "progress": 65,
    "estimatedCompletion": "2025-01-22T10:20:00Z"
  },
  "mcpMetrics": {
    "messagesProcessed": 847,
    "responsesGenerated": 234,
    "errorRate": 0.02
  },
  "timestamp": 1706006400000
}
```

### Voice Response
Real-time voice command execution results.

```json
{
  "type": "voice_response",
  "sessionId": "session_123",
  "response": {
    "intent": "SpawnAgent",
    "success": true,
    "message": "Successfully spawned researcher agent in swarm swarm_1757880683494_yl81sece5",
    "data": {
      "agentId": "agent_1757967065850_dv2zg7",
      "swarmId": "swarm_1757880683494_yl81sece5",
      "mcpTaskId": "mcp_task_1757967065850_xyz789"
    }
  },
  "audioResponse": {
    "available": true,
    "url": "/api/voice/tts/audio_1757967065850.wav"
  }
}
```

### Task Progress
Live task execution updates.

```json
{
  "type": "task_progress",
  "taskId": "task_456",
  "progress": 75,
  "status": "processing",
  "message": "Analysing module dependencies",
  "agentId": "agent_123"
}
```

### GPU Analytics Streaming
Real-time GPU computation results and progress updates.

```json
{
  "type": "gpu_analytics",
  "operation": "clustering",
  "algorithm": "louvain",
  "status": "in_progress",
  "progress": 0.65,
  "data": {
    "clustersFound": 5,
    "currentIteration": 12,
    "maxIterations": 100,
    "modularity": 0.743,
    "gpuUtilization": 89,
    "memoryUsage": "2.4 GB",
    "estimatedCompletion": "2025-01-22T10:16:30Z"
  },
  "performance": {
    "kernelExecutions": 89,
    "avgKernelTime": 3.2,
    "throughput": "1.2M nodes/sec"
  }
}
```

### System Events
Important system-wide events from actual swarm operations.

```json
{
  "type": "system_event",
  "event": "swarm_initialized",
  "data": {
    "swarmId": "swarm_1757880683494_yl81sece5",
    "agentCount": 5,
    "topology": "mesh",
    "mcpConnected": true,
    "agents": [
      {
        "id": "agent_1757967065850_dv2zg7",
        "type": "coordinator",
        "status": "active"
      }
    ],
    "consensusThreshold": 0.7,
    "initialisationTime": 2847
  }
}
```

### Error Messages
Error notifications and alerts.

```json
{
  "type": "error",
  "code": "AGENT_FAILURE",
  "message": "Agent agent_123 has crashed",
  "details": {
    "agentId": "agent_123",
    "reason": "out_of_memory"
  }
}
```

---

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

---

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

**WebSocket Rate Limits**:
- **Messages per minute**: 1000
- **Binary updates**: Unlimited (server-controlled at 60 FPS)
- **Maximum message size**: 10MB
- **Connection limit per IP**: 10

---

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

#### Heartbeat Implementation

```javascript
// Client-side heartbeat
const heartbeatInterval = setInterval(() => {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({
      type: 'ping',
      timestamp: Date.now()
    }));
  }
}, 30000); // Every 30 seconds

// Clean up on close
ws.onclose = () => {
  clearInterval(heartbeatInterval);
};
```

**Server Response**:
```json
{
  "type": "pong",
  "timestamp": 1706006400000,
  "serverTime": 1706006400100
}
```

### Reconnection Strategy

Implement exponential backoff for reconnections:

```javascript
class WebSocketReconnect {
  constructor(url) {
    this.url = url;
    this.reconnectDelay = 1000;
    this.maxDelay = 30000;
    this.attempts = 0;
  }

  connect() {
    this.ws = new WebSocket(this.url);

    this.ws.onclose = () => {
      this.scheduleReconnect();
    };

    this.ws.onopen = () => {
      this.reconnectDelay = 1000;
      this.attempts = 0;
    };
  }

  scheduleReconnect() {
    setTimeout(() => {
      this.attempts++;
      this.reconnectDelay = Math.min(
        this.reconnectDelay * 2,
        this.maxDelay
      );
      this.connect();
    }, this.reconnectDelay);
  }
}
```

#### Rust Server Backoff Calculation

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

---

## Performance Characteristics

### Bandwidth Reduction

| Message Type | JSON Size | Binary V2 Size | Reduction |
|--------------|-----------|----------------|-----------|
| Single Node | ~150 bytes | 36 bytes | 76% |
| 50 Nodes | ~7.5 KB | 1.8 KB | 76% |
| 188 Nodes (dual-graph) | ~28 KB | 6.8 KB | 76% |
| Settings Update | ~800 bytes | ~120 bytes | 85% |
| Batch (compressed) | ~10 KB | ~2 KB | 80% |

**Overall Achieved**: ~80% bandwidth reduction vs JSON

**V2 Improvements over V1:**
- +2 bytes per node (36 vs 34)
- Supports 1 billion nodes vs 16K
- Fixes node ID truncation bug
- Enables dual-graph type separation

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

---

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

### Connection Error Codes

| Code | Reason | Description |
|------|--------|-------------|
| 1000 | Normal Closure | Connection closed normally |
| 1001 | Going Away | Server going down |
| 1002 | Protocol Error | Protocol error detected |
| 1003 | Unsupported Data | Received unsupported data type |
| 1006 | Abnormal Closure | Connection lost |
| 1008 | Policy Violation | Message violated policy |
| 1009 | Message Too Big | Message exceeded size limit |
| 1011 | Internal Error | Server internal error |
| 4000 | Authentication Failed | Invalid or expired token |
| 4001 | Rate Limited | Too many messages |
| 4002 | Invalid Message | Malformed message format |

### Error Recovery

```javascript
ws.onerror = (error) => {
  console.error('WebSocket error:', error);

  // Attempt to reconnect for recoverable errors
  if (error.code >= 1006 && error.code < 4000) {
    reconnect();
  }
};
```

---

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

fn serialize_node_update(
    nodes: &[(u32, BinaryNodeData)],
    agent_ids: &[u32],
    knowledge_ids: &[u32]
) -> Vec<u8> {
    let mut buffer = Vec::with_capacity(nodes.len() * 36);

    // Encode nodes with dual-graph type flags
    for (node_id, data) in nodes {
        // Determine type flag
        let wire_id = if agent_ids.contains(node_id) {
            node_id | 0x80000000  // AGENT_NODE_FLAG
        } else if knowledge_ids.contains(node_id) {
            node_id | 0x40000000  // KNOWLEDGE_NODE_FLAG
        } else {
            *node_id
        };

        // Write node data (36 bytes)
        buffer.extend_from_slice(&wire_id.to_le_bytes());         // 4 bytes
        buffer.extend_from_slice(&data.position.x.to_le_bytes()); // 4 bytes
        buffer.extend_from_slice(&data.position.y.to_le_bytes()); // 4 bytes
        buffer.extend_from_slice(&data.position.z.to_le_bytes()); // 4 bytes
        buffer.extend_from_slice(&data.velocity.x.to_le_bytes()); // 4 bytes
        buffer.extend_from_slice(&data.velocity.y.to_le_bytes()); // 4 bytes
        buffer.extend_from_slice(&data.velocity.z.to_le_bytes()); // 4 bytes
        buffer.extend_from_slice(&data.sssp_distance.to_le_bytes()); // 4 bytes
        buffer.extend_from_slice(&data.sssp_parent.to_le_bytes());   // 4 bytes
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
        const BINARY_NODE_SIZE = 36;
        let offset = 0;

        while (offset < payload.length) {
            // Parse node ID with type flags (V2: u32, 4 bytes)
            const rawId = view.getUint32(offset, true); offset += 4;
            const isAgent = (rawId & 0x80000000) !== 0;
            const isKnowledge = (rawId & 0x40000000) !== 0;
            const id = rawId & 0x3FFFFFFF;

            // Parse position (12 bytes)
            const x = view.getFloat32(offset, true); offset += 4;
            const y = view.getFloat32(offset, true); offset += 4;
            const z = view.getFloat32(offset, true); offset += 4;

            // Parse velocity (12 bytes)
            const vx = view.getFloat32(offset, true); offset += 4;
            const vy = view.getFloat32(offset, true); offset += 4;
            const vz = view.getFloat32(offset, true); offset += 4;

            // Parse SSSP data (8 bytes)
            const sssp_distance = view.getFloat32(offset, true); offset += 4;
            const sssp_parent = view.getInt32(offset, true); offset += 4;

            nodes.push({
                id,
                nodeType: isAgent ? 'agent' : isKnowledge ? 'knowledge' : 'normal',
                position: {x, y, z},
                velocity: {vx, vy, vz},
                sssp_distance,
                sssp_parent
            });
        }

        return nodes;
    }
}
```

### Binary Protocol Parser

Parser for the 36-byte binary protocol used for real-time position updates:

```javascript
function parseBinaryNodeData(buffer) {
  const view = new DataView(buffer);
  const nodes = [];
  const BINARY_NODE_SIZE = 36; // V2: 36 bytes
  const nodeCount = buffer.byteLength / BINARY_NODE_SIZE;

  for (let i = 0; i < nodeCount; i++) {
    const offset = i * BINARY_NODE_SIZE;

    // Parse node ID with type flags (V2: u32, 4 bytes)
    const nodeId = view.getUint32(offset, true);
    const isAgent = (nodeId & 0x80000000) !== 0;     // Bit 31: agent flag
    const isKnowledge = (nodeId & 0x40000000) !== 0; // Bit 30: knowledge flag
    const actualId = nodeId & 0x3FFFFFFF;            // Bits 0-29: node ID

    nodes.push({
      id: actualId,
      isAgent,
      isKnowledge,
      position: {
        x: view.getFloat32(offset + 4, true),   // Position X
        y: view.getFloat32(offset + 8, true),   // Position Y
        z: view.getFloat32(offset + 12, true)   // Position Z
      },
      velocity: {
        x: view.getFloat32(offset + 16, true),  // Velocity X
        y: view.getFloat32(offset + 20, true),  // Velocity Y
        z: view.getFloat32(offset + 24, true)   // Velocity Z
      },
      ssspDistance: view.getFloat32(offset + 28, true),  // SSSP distance
      ssspParent: view.getInt32(offset + 32, true)       // SSSP parent (-1 = root)
    });
  }

  return nodes;
}

// Usage: Apply updates directly to visualisation
ws.onmessage = (event) => {
  if (event.data instanceof ArrayBuffer) {
    const nodeUpdates = parseBinaryNodeData(event.data);
    // No client-side physics - just apply positions
    nodeUpdates.forEach(node => {
      updateNodePosition(node.id, node.position);
      if (node.isAgent) {
        updateAgentVisualization(node.id, node);
      }
    });
  }
};
```

### React Hook Integration

```javascript
import { useEffect, useState, useRef } from 'react';

function useWebSocket(url) {
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState(null);
  const ws = useRef(null);

  useEffect(() => {
    ws.current = new WebSocket(url);

    ws.current.onopen = () => setIsConnected(true);
    ws.current.onclose = () => setIsConnected(false);
    ws.current.onmessage = (event) => {
      setLastMessage(event.data);
    };

    return () => {
      ws.current.close();
    };
  }, [url]);

  const sendMessage = (message) => {
    if (ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(message));
    }
  };

  return { isConnected, lastMessage, sendMessage };
}
```

---

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

### Message Batching

For optimal performance, batch multiple updates:

```javascript
// Instead of sending multiple messages
settings.forEach(setting => {
  ws.send(JSON.stringify({
    type: 'settings',
    path: setting.path,
    value: setting.value
  }));
});

// Send a single batched message
ws.send(JSON.stringify({
  type: 'settings_batch',
  updates: settings
}));
```

### Binary vs JSON Protocol Selection

**Binary Protocol (36-byte format)** for:
- Real-time position updates at 60 FPS
- Physics simulation results from backend
- High-frequency numeric data (>10 Hz)
- Bandwidth-critical applications (95% size reduction vs JSON)

**JSON Messages** for:
- Voice command streaming
- Agent status updates
- Control messages and configuration
- Event notifications and errors
- Human-readable data and debugging

---

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

---

## Security Considerations

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

---

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
        assert_eq!(serialized.len(), 36); // V2: 36 bytes
    }

    #[test]
    fn test_dual_graph_type_flags() {
        let nodes = vec![
            (10, create_test_node()),  // Regular
            (20, create_test_node()),  // Knowledge
            (30, create_test_node()),  // Agent
        ];
        let agent_ids = vec![30];
        let knowledge_ids = vec![20];

        let binary = encode_node_data_with_types(&nodes, &agent_ids, &knowledge_ids);

        // Verify flags in binary data
        let id1 = u32::from_le_bytes([binary[0], binary[1], binary[2], binary[3]]);
        let id2 = u32::from_le_bytes([binary[36], binary[37], binary[38], binary[39]]);
        let id3 = u32::from_le_bytes([binary[72], binary[73], binary[74], binary[75]]);

        assert_eq!(id1, 10); // No flags
        assert_eq!(id2 & 0x40000000, 0x40000000); // KNOWLEDGE_NODE_FLAG
        assert_eq!(id3 & 0x80000000, 0x80000000); // AGENT_NODE_FLAG
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

### WebSocket Test Client

```javascript
// Simple test client
const testWebSocket = async () => {
  const ws = new WebSocket('ws://localhost:3001/ws');

  ws.onopen = () => {
    console.log('Connected');

    // Test ping
    ws.send(JSON.stringify({
      type: 'ping',
      timestamp: Date.now()
    }));

    // Test settings update
    ws.send(JSON.stringify({
      type: 'settings',
      path: 'test.value',
      value: 123
    }));
  };

  ws.onmessage = (event) => {
    console.log('Received:', event.data);
  };
};
```

---

## Compression

Messages larger than 1KB are automatically compressed using zlib:

```javascript
// Server automatically compresses large messages
// Client needs to handle decompression if needed
ws.onmessage = (event) => {
  if (event.data instanceof Blob) {
    // Handle compressed data
    event.data.arrayBuffer().then(buffer => {
      const decompressed = pako.inflate(buffer);
      // Process decompressed data
    });
  }
};
```

---

## Migration Guide

### From JSON to Binary Protocol

1. **Implement Binary Serialisation**: Add binary encode/decode methods
2. **Update Client**: Integrate `BinaryProtocolClient`
3. **Test Thoroughly**: Validate roundtrip serialisation
4. **Deploy Gradually**: Use feature flags for rollout
5. **Monitor Performance**: Track bandwidth and latency improvements

### From V1 to V2

1. **Update Node ID Parsing**: Change from `getUint16()` to `getUint32()`
2. **Update Type Flag Checks**: Change from bits 15/14 to bits 31/30
3. **Update Buffer Size**: Change `BINARY_NODE_SIZE` from 34 to 36
4. **Update Offsets**: Adjust all field offsets by +2 bytes
5. **Test Dual-Graph**: Verify knowledge/agent node separation

---

## Version History

### Version 2.0 (October 2025) - Current

**Changes:**
- Upgraded to Binary Protocol V2 (36 bytes per node)
- Changed node ID from u16 to u32 (fixes truncation bug)
- Moved type flags from bits 15/14 to bits 31/30
- Implemented unified dual-graph broadcasting
- Added adaptive broadcast rate (60 FPS active, 5 Hz settled)

**Bug Fixes:**
- Fixed node ID truncation for IDs > 16,383
- Fixed dual-graph broadcast conflicts (knowledge + agent graphs)
- Eliminated race conditions from separate broadcast paths
- Fixed 2-byte misalignment causing "corrupted node data" errors

### Version 1.2.0 (Legacy) - DEPRECATED

**Format:**
- Binary Protocol V1 (34 bytes per node)
- u16 node IDs (bits 15/14 for type flags, bits 0-13 for ID)
- Separate broadcasts for knowledge and agent graphs
- Fixed 60 FPS broadcast rate

**Known Issues:**
- Node ID truncation bug (IDs > 16,383 collision)
- Dual-graph broadcast conflicts
- Limited to 16,383 nodes maximum

---

## Troubleshooting

### Common Issues

**Issue: "Corrupted node data" errors on client**
- **Cause**: Client using V1 parser (34 bytes) with V2 server (36 bytes)
- **Fix**: Update client `BINARY_NODE_SIZE` constant from 34 to 36
- **Fix**: Update parse offsets (position at offset 4, not 2)

**Issue: Nodes with wrong type (agent vs knowledge)**
- **Cause**: Client checking wrong flag bits (15/14 instead of 31/30)
- **Fix**: Update flag constants to `0x80000000` and `0x40000000`
- **Fix**: Use `getUint32()` instead of `getUint16()` for node ID

**Issue: Agent nodes not moving**
- **Cause**: Agent graph not included in unified broadcast
- **Fix**: Verify server logs show "188 nodes: 185 knowledge + 3 agents"
- **Fix**: Check `encode_node_data_with_types()` receives both ID lists

---

## Related Documentation

- **[Binary Protocol V2 Specification](binary-protocol.md)** - Complete wire format details
- **[Dual-Graph Architecture](../../architecture/core/visualization.md)** - Knowledge + agent graph visualization
- **[REST API](rest-api.md)** - HTTP endpoint documentation
- **[MCP Protocol](mcp-protocol.md)** - Agent communication protocol
- **[Performance Benchmarks](../../reports/performance-benchmarks.md)** - Bandwidth and latency metrics

---

**[← REST API](rest-api.md)** | **[Binary Protocol →](binary-protocol.md)**
