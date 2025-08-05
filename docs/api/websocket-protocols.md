# WebSocket Protocols and Communication

## Overview

This document consolidates all WebSocket protocols used in the system, including binary position updates, JSON message formats, and real-time agent communication.

## WebSocket Endpoints

### Main Graph Data Stream
- **Endpoint**: `/ws` or `/wss`
- **Purpose**: Real-time graph position updates
- **Protocol**: Binary (28-byte format)
- **Update Rate**: 60 FPS

### Bots/Agent Data Stream
- **Endpoint**: Shares main WebSocket, uses node ID flags
- **Purpose**: AI agent position and state updates
- **Protocol**: Binary + JSON state messages
- **Bot Identification**: Node ID & 0x80 flag

## Binary Protocol Specification

### Position Update Format (28 bytes)

```
Offset  Size  Type      Description
0       4     uint32    Node/Agent ID (0x80 flag for bots)
4       4     float32   Position X
8       4     float32   Position Y
12      4     float32   Position Z
16      4     float32   Velocity X
20      4     float32   Velocity Y
24      4     float32   Velocity Z
```

### Binary Message Structure

```typescript
// Client request for positions
interface PositionRequest {
  type: 'requestPositions' | 'requestBotsPositions';
  clientId?: string;
}

// Server binary response
// Header: [messageType(1), nodeCount(4)]
// Body: Array of 28-byte position records
```

### Implementation Example

```typescript
// Decoding binary positions
function decodeBinaryPositions(data: ArrayBuffer): Map<number, Position> {
  const view = new DataView(data);
  const positions = new Map();
  
  for (let i = 0; i < data.byteLength; i += 28) {
    const id = view.getUint32(i, true);
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
    
    positions.set(id, { position, velocity });
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

### Agent State Messages

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
        "tokenRate": 1523.4
      }
    }],
    "connections": [{
      "from": "agent-001",
      "to": "agent-002",
      "strength": 0.8,
      "messageRate": 12.5
    }]
  }
}
```

#### Agent Status Update
```json
{
  "type": "agent-status-update",
  "payload": {
    "agentId": "agent-001",
    "status": "executing",
    "currentTask": "Analyzing research papers",
    "progress": 0.65
  }
}
```

### Control Messages

#### Initialize Swarm Request
```json
{
  "type": "initialize-swarm",
  "payload": {
    "topology": "hierarchical",
    "maxAgents": 10,
    "agentTypes": ["coordinator", "researcher", "coder"],
    "enableNeural": true,
    "customPrompt": "Build a REST API"
  }
}
```

#### Command Execution
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

### System Messages

#### Heartbeat
```json
{
  "type": "ping",
  "timestamp": "2024-01-10T12:00:00Z"
}

{
  "type": "pong",
  "timestamp": "2024-01-10T12:00:00Z",
  "serverTime": "2024-01-10T12:00:00.123Z"
}
```

#### Error Message
```json
{
  "type": "error",
  "payload": {
    "code": "AGENT_NOT_FOUND",
    "message": "Agent with ID 'agent-999' not found",
    "details": {
      "requestId": "req-123",
      "recoverable": true
    }
  }
}
```

## Compression and Optimization

### Compression Threshold
- Messages > 1KB are automatically compressed
- Uses zlib compression
- Compression flag set in binary header

### Batching Strategy
```typescript
// Batch multiple position updates
const batchSize = 100; // positions per message
const updateRate = 60; // Hz

// Results in:
// - 100 agents: 1 message/frame
// - 500 agents: 5 messages/frame
// - 1000 agents: 10 messages/frame
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
    
    // Exponential backoff
    const delay = this.baseDelay * Math.pow(2, this.reconnectAttempts);
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

## Performance Metrics

### Bandwidth Usage
| Scenario | JSON | Binary | Reduction |
|----------|------|--------|----------|
| 100 agents @ 60fps | 3 MB/s | 168 KB/s | 94% |
| 500 agents @ 60fps | 15 MB/s | 840 KB/s | 94% |
| 1000 agents @ 60fps | 30 MB/s | 1.68 MB/s | 94% |

### Latency Targets
- Position updates: < 16.67ms (60 FPS)
- State changes: < 100ms
- Command execution: < 500ms

## Security Considerations

### Authentication
```typescript
// WebSocket with auth token
const ws = new WebSocket('wss://api.example.com/ws', {
  headers: {
    'Authorization': `Bearer ${authToken}`,
    'X-Client-Version': '1.0.0'
  }
});
```

### Message Validation
- All JSON messages validated against schemas
- Binary messages checked for correct size
- Rate limiting per client connection
- Maximum message size: 1MB (configurable)

## Integration Examples

### React Hook Usage
```typescript
function useWebSocketPositions() {
  const [positions, setPositions] = useState(new Map());
  
  useEffect(() => {
    const ws = new WebSocket(WS_URL);
    
    ws.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) {
        const newPositions = decodeBinaryPositions(event.data);
        setPositions(newPositions);
      }
    };
    
    // Request positions at 60 FPS
    const interval = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'requestPositions' }));
      }
    }, 16.67);
    
    return () => {
      clearInterval(interval);
      ws.close();
    };
  }, []);
  
  return positions;
}
```

### Three.js Integration
```typescript
// Update 3D positions from WebSocket
function updateAgentPositions(scene: THREE.Scene, positions: Map<number, Position>) {
  positions.forEach((data, id) => {
    const mesh = scene.getObjectByName(`agent-${id}`);
    if (mesh) {
      mesh.position.set(data.position.x, data.position.y, data.position.z);
      // Store velocity for interpolation
      mesh.userData.velocity = data.velocity;
    }
  });
}
```

## Troubleshooting

### Common Issues

1. **No position updates**
   - Check WebSocket connection state
   - Verify requestPositions messages are sent
   - Check for bot flag (0x80) if expecting agent data

2. **High latency**
   - Monitor network bandwidth
   - Check compression is enabled
   - Reduce update frequency if needed

3. **Connection drops**
   - Implement heartbeat/ping-pong
   - Check proxy/firewall settings
   - Monitor for memory leaks

### Debug Mode
```typescript
// Enable WebSocket debug logging
if (process.env.NODE_ENV === 'development') {
  ws.addEventListener('message', (event) => {
    console.log('[WS]', event.data instanceof ArrayBuffer 
      ? `Binary: ${event.data.byteLength} bytes` 
      : `JSON: ${event.data}`);
  });
}
```

## References

- [Binary Protocol Specification](./binary-protocol.md)
- [REST API Documentation](./rest.md)
- [Agent Control System](../server/features/claude-flow-mcp-integration.md)
- [GPU Migration Architecture](../architecture/visionflow-gpu-migration.md)