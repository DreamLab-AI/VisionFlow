# WebSocket Protocols

*[Api](../index.md)*

## Overview

VisionFlow uses WebSocket connections for real-time bidirectional communication between clients and the backend. The system implements multiple specialised WebSocket endpoints, each optimised for specific data types and update patterns.

## WebSocket Architecture

```mermaid
graph TB
    subgraph "Client Layer"
        Browser[Browser Client]
        XR[XR Client]
        API[API Client]
    end

    subgraph "WebSocket Endpoints"
        Flow[/wss<br/>Binary Graph Updates]
        Speech[/ws/speech<br/>Voice Streaming]
        MCP[/ws/mcp-relay<br/>MCP Protocol]
        Bots[/api/visualisation/agents/ws<br/>Agent Updates]
    end

    subgraph "Backend Handlers"
        FlowHandler[Socket Flow Handler]
        SpeechHandler[Speech Handler]
        MCPHandler[MCP Relay Handler]
        BotsHandler[Bots Viz Handler]
    end

    Browser --> Flow
    Browser --> Speech
    Browser --> MCP
    Browser --> Bots
    XR --> Flow
    API --> MCP

    Flow --> FlowHandler
    Speech --> SpeechHandler
    MCP --> MCPHandler
    Bots --> BotsHandler
```

## WebSocket Endpoints

### 1. Socket Flow Stream
- **Endpoint**: `/wss` (Production: `wss://your-domain.com/wss`)
- **Purpose**: Real-time graph position updates
- **Protocol**: Binary (28-byte format) + JSON control
- **Update Rate**: 5-60 FPS (dynamic)
- **Binary Format**: Defined in [Binary Protocol](../binary-protocol.md)

### 2. Speech Stream
- **Endpoint**: `/ws/speech`
- **Purpose**: Voice interaction and audio streaming
- **Protocol**: JSON + Binary audio data
- **Features**: Real-time transcription, TTS, voice commands

### 3. MCP Relay
- **Endpoint**: `/ws/mcp-relay`
- **Purpose**: Claude Flow MCP protocol relay
- **Protocol**: JSON-RPC 2.0
- **Features**: Tool invocation, agent orchestration

### 4. Agent Visualisation
- **Endpoint**: `/api/visualisation/agents/ws`
- **Purpose**: Multi-Agent system visualisation and monitoring
- **Protocol**: JSON with agent states, metrics, and position updates
- **Update Rate**: 16 ms intervals (~60 FPS)
- **Features**: Agent status tracking, performance metrics, swarm coordination

## Binary Protocol Specification

### Position Update Format (28 bytes)

**Reference:** This format is defined authoritatively in [Binary Protocol Specification](../binary-protocol.md).

```
Offset  Size  Type      Description
0       4     u32       Node/Agent ID (with type flags)
4       4     f32       Position X (IEEE 754)
8       4     f32       Position Y (IEEE 754)
12      4     f32       Position Z (IEEE 754)
16      4     f32       Velocity X
20      4     f32       Velocity Y
24      4     f32       Velocity Z

All values are little-endian format.
```

### Node Type Flags

| Flag Value | Type | Description |
|-----------|------|-------------|
| 0x80000000 | Agent | AI agent node |
| 0x40000000 | Knowledge | Knowledge graph node |
| 0x00000000 | Unknown | Default/other node type |

**Note:** Node ID uses lower 30 bits (0x3FFFFFFF) for actual ID.

### Binary Message Structure

```typescript
// Client request for positions
interface PositionRequest {
  type: 'requestInitialData' | 'requestPositions';
  clientId?: string;
}

// Server binary response
// Raw binary data: Array of 28-byte position records
```

### Implementation Example

```typescript
// Decoding binary positions with type flags
function decodeBinaryPositions(data: ArrayBuffer): Map<number, PositionUpdate> {
  const view = new DataView(data);
  const positions = new Map();

  for (let i = 0; i < data.byteLength; i += 28) {
    const flaggedId = view.getUint32(i, true);

    // Extract type flags
    const isAgent = (flaggedId & 0x80000000) !== 0;
    const isKnowledge = (flaggedId & 0x40000000) !== 0;
    const actualId = flaggedId & 0x3FFFFFFF;

    const position = {
      x: view.getFloat32(i + 4, true),
      y: view.getFloat32(i + 8, true),
      z: view.getFloat32(i + 12, true)
    };
    const velocity = {
      x: view.getFloat32(i + 16, true),
      y: view.getFloat32(i + 20, true),
      z: view.getFloat32(i + 24, true)
    };

    positions.set(actualId, {
      position,
      velocity,
      type: isAgent ? 'agent' : isKnowledge ? 'knowledge' : 'unknown'
    });
  }

  return positions;
}
```

## JSON Message Protocols

### Base Message Format

```typescript
interface WebSocketMessage<T = any> {
  type: string;          // Message type identifier
  payload?: T;           // Type-specific payload
  timestamp?: string;    // ISO 8601 timestamp
  id?: string;          // Message ID for tracking
}
```

### Socket Flow Messages (/wss)

#### Connection Established
```json
{
  "type": "connection_established",
  "timestamp": 1679417762000
}
```

#### Request Initial Data
```json
{
  "type": "requestInitialData"
}
```

#### Request Full Snapshot
```json
{
  "type": "request_full_snapshot",
  "graphs": ["knowledge", "agent"]
}
```

#### Updates Started
```json
{
  "type": "updatesStarted",
  "timestamp": 1679417763000
}
```

#### Enable Randomization (Legacy)
```json
{
  "type": "enableRandomization",
  "enabled": true
}
```
**Note**: Server-side randomization is deprecated. Client-side randomization is now preferred.

#### Request Bots Positions
```json
{
  "type": "requestBotsPositions"
}
```

#### Bots Updates Started
```json
{
  "type": "botsUpdatesStarted",
  "timestamp": 1679417763000
}
```

#### Heartbeat
```json
{
  "type": "ping",
  "timestamp": 1679417764000
}

{
  "type": "pong",
  "timestamp": 1679417764000
}
```

### Speech Socket Messages (/ws/speech)

#### Text-to-Speech Request
```json
{
  "type": "textToSpeech",
  "payload": {
    "text": "Hello world",
    "voice": "neural",
    "speed": 1.0,
    "stream": true
  }
}
```

#### Speech-to-Text Control
```json
{
  "type": "sttAction",
  "payload": {
    "action": "start",
    "language": "en-US",
    "model": "whisper"
  }
}
```

#### Provider Configuration
```json
{
  "type": "setProvider",
  "payload": {
    "provider": "openai"
  }
}
```

#### Audio Data (Binary)
- Binary WebSocket frames containing raw audio data
- Used for streaming audio input/output
- Format depends on configured audio codec

### Agent Visualisation Messages (/ws/bots_visualization)

#### Initialization Message
```json
{
  "type": "multi-agent-init",
  "payload": {
    "multi-agentId": "multi-agent-001",
    "topology": "hierarchical",
    "agents": []
  }
}
```

#### Agent State Update
```json
{
  "type": "agent-state-update",
  "payload": {
    "agentId": "agent-001",
    "status": "executing",
    "currentTask": "Analyzing research papers",
    "progress": 0.65,
    "metrics": {
      "tokensUsed": 1523,
      "tasksCompleted": 42,
      "successRate": 0.95
    }
  }
}
```

#### Position Update (JSON alternative to binary)
```json
{
  "type": "position-update",
  "payload": {
    "positions": [{
      "id": 1,
      "type": "agent",
      "position": [100.0, 200.0, 50.0],
      "velocity": [0.1, 0.2, 0.3]
    }]
  }
}
```

#### Full Agent Update
```json
{
  "type": "bots-full-update",
  "payload": {
    "agents": [{
      "id": "agent-001",
      "name": "Research Agent Alpha",
      "type": "researcher",
      "status": "active",
      "health": 0.95,
      "workload": 0.7,
      "capabilities": ["web_search", "document_analysis"],
      "metrics": {
        "tasksCompleted": 42,
        "successRate": 0.95,
        "tokenRate": 1523.4,
        "averageResponseTime": 250
      }
    }],
    "connections": [{
      "from": "agent-001",
      "to": "agent-002",
      "strength": 0.8,
      "messageRate": 12.5,
      "bandwidth": 1024
    }]
  }
}
```

### MCP Relay Messages (/ws/mcp-relay)

#### Tool Invocation
```json
{
  "jsonrpc": "2.0",
  "id": "call-123",
  "method": "tools/call",
  "params": {
    "name": "multi-agent_init",
    "arguments": {
      "topology": "hierarchical",
      "maxAgents": 10
    }
  }
}
```

#### Tool Response
```json
{
  "jsonrpc": "2.0",
  "id": "call-123",
  "result": {
    "content": [{
      "type": "text",
      "text": "multi-agent initialized with 10 agents in hierarchical topology"
    }]
  }
}
```

#### Agent Command
```json
{
  "type": "execute-command",
  "payload": {
    "agentId": "agent-001",
    "command": "analyze_document",
    "parameters": {
      "documentUrl": "https://example.com/doc.pdf",
      "extractInsights": true
    }
  }
}
```

### System Messages (All Endpoints)

#### Error Message
```json
{
  "type": "error",
  "payload": {
    "code": "AGENT_NOT_FOUND",
    "message": "Agent with ID 'agent-999' not found",
    "details": {
      "requestId": "req-123",
      "recoverable": true,
      "endpoint": "/ws/bots_visualization"
    }
  }
}
```

#### Connection Status
```json
{
  "type": "connection-status",
  "payload": {
    "status": "connected",
    "endpoint": "/wss",
    "clientCount": 5,
    "serverTime": "2024-01-10T12:00:00.123Z"
  }
}
```

## Compression and Optimisation

### Compression Strategy
- Messages > 1KB automatically use permessage-deflate
- Binary position data typically compresses 40-60%
- JSON messages compress 70-85%

### Batching Strategy
```typescript
// Position updates batching
const batchSize = 100; // positions per message
const updateRate = 60; // Hz for agents, 1 Hz for knowledge nodes

// Agent visualisation batching
const agentBatchSize = 50; // agents per update
const updateInterval = 16; // ms (~60fps)
```

## Connection Management

### Reconnection Strategy
```typescript
class WebSocketManager {
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private baseDelay = 1000; // ms

  async reconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      throw new Error('Max reconnection attempts reached');
    }

    // Exponential backoff with jitter
    const delay = this.baseDelay * Math.pow(2, this.reconnectAttempts) +
                  Math.random() * 1000;
    await sleep(delay);

    this.reconnectAttempts++;
    await this.connect();
  }
}
```

### Connection States
```typescript
enum ConnectionState {
  CONNECTING = 'connecting',
  CONNECTED = 'connected',
  RECONNECTING = 'reconnecting',
  DISCONNECTED = 'disconnected',
  ERROR = 'error'
}
```

### Heartbeat Configuration
```typescript
// Per-endpoint heartbeat intervals
const HEARTBEAT_CONFIG = {
  '/wss': { interval: 30000, timeout: 60000 },
  '/ws/speech': { interval: 5000, timeout: 10000 },
  '/ws/mcp-relay': { interval: 30000, timeout: 60000 },
  '/ws/bots_visualization': { interval: 30000, timeout: 60000 }
};
```

## Performance Metrics

### Bandwidth Usage

| Scenario | JSON | Binary | Reduction |
|----------|------|--------|----------|
| 100 agents @ 60fps | 3 MB/s | 168 KB/s | 94% |
| 500 agents @ 60fps | 15 MB/s | 840 KB/s | 94% |
| 1000 agents @ 60fps | 30 MB/s | 1.68 MB/s | 94% |
| Mixed (agents + knowledge) | 25 MB/s | 1.2 MB/s | 95% |

### Latency Targets
- Position updates: < 16.67ms (60 FPS)
- Agent state changes: < 100ms
- Speech processing: < 200ms
- MCP command execution: < 500ms

## Security Considerations

### Authentication
```typescript
// WebSocket with session-based auth
const ws = new WebSocket('wss://api.example.com/wss', {
  headers: {
    'Cookie': `session=${sessionToken}`,
    'X-Client-Version': '1.0.0'
  }
});

// Token-based auth alternative
const wsWithToken = new WebSocket(
  `wss://api.example.com/wss?token=${authToken}`
);
```

### Message Validation
- All JSON messages validated against schemas
- Binary messages checked for correct size alignment
- Rate limiting per client connection (see limits below)
- Maximum message size: 100MB (configurable)

### Rate Limiting

| Endpoint | Message Rate | Binary Rate | Burst Allowance |
|----------|-------------|-------------|-----------------|
| /wss | 1000 msg/min | Unlimited* | 100 messages |
| /ws/speech | 500 msg/min | 10 MB/min | 50 messages |
| /ws/mcp-relay | 100 msg/min | N/A | 20 messages |
| /ws/bots_visualization | 1000 msg/min | 1 MB/min | 100 messages |

*Binary updates are rate-controlled by server-side throttling, not client limits

## Integration Examples

### React Hook for Position Updates
```typescript
function useWebSocketPositions() {
  const [positions, setPositions] = useState(new Map());

  useEffect(() => {
    const ws = new WebSocket('wss://localhost:3001/wss');

    ws.onopen = () => {
      // Request initial data
      ws.send(JSON.stringify({ type: 'requestInitialData' }));
    };

    ws.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) {
        const newPositions = decodeBinaryPositions(event.data);
        setPositions(prev => new Map([...prev, ...newPositions]));
      } else {
        const message = JSON.parse(event.data);
        handleControlMessage(message);
      }
    };

    return () => ws.close();
  }, []);

  return positions;
}
```

### Three.js Integration
```typescript
// Update 3D positions from WebSocket
function updateSceneFromWebSocket(
  scene: THREE.Scene,
  positions: Map<number, PositionUpdate>
) {
  positions.forEach((data, id) => {
    const mesh = scene.getObjectByName(`node-${id}`);
    if (mesh) {
      // Update position
      mesh.position.set(data.position.x, data.position.y, data.position.z);

      // Store velocity for interpolation
      mesh.userData.velocity = data.velocity;
      mesh.userData.type = data.type;

      // Apply type-specific styling
      if (data.type === 'agent') {
        mesh.material.colour.setHex(0xff6b6b);
      } else if (data.type === 'knowledge') {
        mesh.material.colour.setHex(0x4ecdc4);
      }
    }
  });
}
```

### Agent Visualisation Integration
```typescript
function useAgentVisualization() {
  const [agents, setAgents] = useState([]);

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:3001/ws/bots_visualization');

    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);

      switch (message.type) {
        case 'bots-full-update':
          setAgents(message.payload.agents);
          break;

        case 'agent-state-update':
          setAgents(prev => prev.map(agent =>
            agent.id === message.payload.agentId
              ? { ...agent, ...message.payload }
              : agent
          ));
          break;
      }
    };

    return () => ws.close();
  }, []);

  return agents;
}
```

## Troubleshooting

### Common Issues

1. **Binary Data Corruption**
   - Check byte alignment (must be multiple of 28)
   - Verify little-endian byte order
   - Validate node type flags

2. **High Memory Usage**
   - Implement position update throttling
   - Use object pooling for frequent updates
   - Monitor WebSocket message queue size

3. **Connection Instability**
   - Implement proper heartbeat handling
   - Use exponential backoff for reconnections
   - Monitor network conditions

### Debug Configuration

```typescript
// Enable comprehensive WebSocket debugging
const DEBUG_CONFIG = {
  logBinaryMessages: process.env.NODE_ENV === 'development',
  logJSONMessages: true,
  logHeartbeats: false,
  logPerformanceMetrics: true
};

if (DEBUG_CONFIG.logBinaryMessages) {
  ws.addEventListener('message', (event) => {
    if (event.data instanceof ArrayBuffer) {
      console.log(`[WS Binary] ${event.data.byteLength} bytes:`,
        Array.from(new Uint8Array(event.data, 0, 32)) // First 32 bytes
      );
    }
  });
}
```

### Performance Monitoring

```typescript
class WebSocketMetrics {
  private messageCount = 0;
  private bytesReceived = 0;
  private lastUpdate = Date.now();

  onMessage(event: MessageEvent) {
    this.messageCount++;
    this.bytesReceived += event.data instanceof ArrayBuffer
      ? event.data.byteLength
      : new Blob([event.data]).size;

    // Log metrics every 10 seconds
    if (Date.now() - this.lastUpdate > 10000) {
      console.log(`WebSocket Metrics: ${this.messageCount} msgs, ${this.bytesReceived} bytes`);
      this.reset();
    }
  }

  private reset() {
    this.messageCount = 0;
    this.bytesReceived = 0;
    this.lastUpdate = Date.now();
  }
}
```

## References

- [Binary Protocol Specification](../binary-protocol.md)
- [REST API Documentation](rest/index.md)
- [WebSocket API Reference](websocket/index.md)
- [VisionFlow Architecture](../architecture/system-overview.md)



## See Also

- [Request Handlers Architecture](../server/handlers.md) - Server implementation

## Related Topics

- [AI Services Documentation](../server/ai-services.md) - Implementation
- [Actor System](../server/actors.md) - Implementation
- [Analytics API Endpoints](../api/analytics-endpoints.md)
- [Binary Protocol Specification](../binary-protocol.md)
- [Graph API Reference](../api/rest/graph.md)
- [MCP WebSocket Relay Architecture](../architecture/mcp-websocket-relay.md)
- [Modern Settings API - Path-Based Architecture](../MODERN_SETTINGS_API.md)
- [Multi-MCP Agent Visualisation API Reference](../api/multi-mcp-visualization-api.md)
- [REST API Bloom/Glow Field Validation Fix](../REST_API_BLOOM_GLOW_VALIDATION_FIX.md)
- [REST API Reference](../api/rest/index.md)
- [Request Handlers Architecture](../server/handlers.md) - Implementation
- [Services Architecture](../server/services.md) - Implementation
- [Settings API Reference](../api/rest/settings.md)
- [Single-Source Shortest Path (SSSP) API](../api/shortest-path-api.md)
- [VisionFlow API Documentation](../api/index.md)
- [VisionFlow MCP Integration Documentation](../api/mcp/index.md)
- [VisionFlow WebSocket API Documentation](../api/websocket/index.md)
- [WebSocket API Reference](../api/websocket.md)
- [WebSocket Communication](../client/websocket.md)
- [dev-backend-api](../reference/agents/development/backend/dev-backend-api.md)
- [docs-api-openapi](../reference/agents/documentation/api-docs/docs-api-openapi.md)
