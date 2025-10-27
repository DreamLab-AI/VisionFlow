# WebSocket API Reference

## Overview

VisionFlow uses WebSocket connections for real-time bidirectional communication between clients and the server. The WebSocket API handles high-frequency position updates, real-time graph changes, and live system events.

**WebSocket Endpoints**:
- General: `ws://localhost:3030/ws` - Binary position updates + JSON messages
- Voice Commands: `ws://localhost:3030/ws/voice` - Voice command streaming
- Agent Visualization: `ws://localhost:3030/ws/agents` - Agent status updates
- Real-time Analytics: `ws://localhost:3030/ws/analytics` - GPU computation results

## Connection

### Establishing Connection

```javascript
// Basic connection
const ws = new WebSocket('ws://localhost:3030/ws');

// Authenticated connection
const ws = new WebSocket(`ws://localhost:3030/ws?token=${jwtToken}`);

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

### Authentication

WebSocket connections can be authenticated using:
1. Query parameter: `?token=<jwt_token>`
2. First message authentication:

```javascript
ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'auth',
    token: jwtToken,
    pubkey: nostrPublicKey
  }));
};
```

## Message Types

### Binary Messages

Real-time node position updates are streamed at 60 FPS using a compact binary protocol. Each node is encoded as a 34-byte frame:

**Binary Frame Format (34 bytes per node)**:
- `node_id`: u16 (2 bytes) - Node identifier with control bits
- `position`: Vec3 (12 bytes) - X, Y, Z coordinates as f32
- `velocity`: Vec3 (12 bytes) - Velocity vector as f32
- `sssp_distance`: f32 (4 bytes) - Shortest path distance
- `sssp_parent`: i32 (4 bytes) - Parent node in shortest path tree

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

The backend sends continuous streams of these binary frames at 60 FPS. Client-side physics simulation is not required - only position smoothing for display.

### JSON Messages

All other messages use JSON format:

```javascript
ws.onmessage = (event) => {
  if (typeof event.data === 'string') {
    const message = JSON.parse(event.data);
    handleMessage(message);
  }
};
```

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

**Real-time Voice Response:**
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

### Heartbeat
Keeps connection alive and measures latency.

```json
{
  "type": "ping",
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

## Server → Client Messages

### Binary Position Updates
Real-time physics simulation results streamed at 60 FPS from the backend.

**Binary Protocol (34 bytes per node)**:
- `node_id`: u16 (2 bytes) - Node identifier (bit 15: agent flag, bits 14-0: node ID)
- `position`: Vec3 (12 bytes) - Current world position (x, y, z as f32)
- `velocity`: Vec3 (12 bytes) - Current velocity vector (x, y, z as f32)
- `sssp_distance`: f32 (4 bytes) - Distance in shortest path computation
- `sssp_parent`: i32 (4 bytes) - Parent node ID in shortest path tree (-1 if root)

The server computes physics on the backend and streams position updates continuously. Clients receive authoritative position data without needing local physics simulation.

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

### Agent Status Updates (Real MCP Data)
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
    "description": "Analyzing authentication module",
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

### Task Progress
Live task execution updates.

```json
{
  "type": "task_progress",
  "taskId": "task_456",
  "progress": 75,
  "status": "processing",
  "message": "Analyzing module dependencies",
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

### Real-time Anomaly Detection
Live anomaly detection results as they're computed.

```json
{
  "type": "anomaly_detected",
  "method": "isolation_forest",
  "anomaly": {
    "nodeId": "node_1247",
    "score": 0.78,
    "confidence": 0.94,
    "features": {
      "x": 156.3,
      "y": 287.1,
      "connectivity": 12
    },
    "severity": "high"
  },
  "context": {
    "totalNodes": 1500,
    "anomaliesFound": 8,
    "detectionTime": 167
  }
}
```

### System Events (Real MCP Integration)
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

### Heartbeat Response
Server response to client ping.

```json
{
  "type": "pong",
  "timestamp": 1706006400000,
  "serverTime": 1706006400100
}
```

## Connection Management

### Heartbeat Protocol

The WebSocket connection uses a heartbeat mechanism to detect disconnections:

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

## Rate Limiting

WebSocket connections have the following rate limits:
- **Messages per minute**: 1000
- **Binary updates**: Unlimited (server-controlled at 60 FPS)
- **Maximum message size**: 10MB
- **Connection limit per IP**: 10

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

## Error Handling
### Timeouts

While WebSockets are persistent connections, the server-side operations they trigger are still subject to timeouts to ensure system stability.

-   **Actor-Level Timeout:** When a WebSocket message triggers a long-running operation via an actor (e.g., a complex graph query), that operation is subject to a **5-second timeout**. If the actor does not respond within this time, the operation will fail, and an error message will be sent back to the client over the WebSocket.

-   **Connection Heartbeat:** The WebSocket connection relies on a heartbeat mechanism (ping/pong messages) to detect disconnections. If the server does not receive a `ping` from the client within a certain interval (typically 30 seconds), it may close the connection.

### Connection Errors

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

## Performance Considerations

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

### Binary vs JSON

**Binary Protocol (34-byte format)** for:
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

## Integration Examples

### React Hook

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

### Binary Protocol Parser

Parser for the 34-byte binary protocol used for real-time position updates:

```javascript
function parseBinaryNodeData(buffer) {
  const view = new DataView(buffer);
  const nodes = [];
  const nodeCount = buffer.byteLength / 34;

  for (let i = 0; i < nodeCount; i++) {
    const offset = i * 34;

    // Parse node ID with control bits
    const nodeId = view.getUint16(offset, true);
    const isAgent = (nodeId & 0x8000) !== 0;  // Bit 15: agent flag
    const actualId = nodeId & 0x7FFF;         // Bits 14-0: node ID

    nodes.push({
      id: actualId,
      isAgent,
      position: {
        x: view.getFloat32(offset + 2, true),   // Position X
        y: view.getFloat32(offset + 6, true),   // Position Y
        z: view.getFloat32(offset + 10, true)   // Position Z
      },
      velocity: {
        x: view.getFloat32(offset + 14, true),  // Velocity X
        y: view.getFloat32(offset + 18, true),  // Velocity Y
        z: view.getFloat32(offset + 22, true)   // Velocity Z
      },
      ssspDistance: view.getFloat32(offset + 26, true),  // SSSP distance
      ssspParent: view.getInt32(offset + 30, true)       // SSSP parent (-1 = root)
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

## Testing

### WebSocket Test Client

```javascript
// Simple test client
const testWebSocket = async () => {
  const ws = new WebSocket('ws://localhost:3030/ws');
  
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

## Related Documentation

- [Binary Protocol Specification](binary-protocol.md)
- [REST API](rest-api.md)
- [MCP Protocol](mcp-protocol.md)
- [Client WebSocket Integration](../../guides/websocket-integration.md)

---

**[← REST API](rest-api.md)** | **[Binary Protocol →](binary-protocol.md)**