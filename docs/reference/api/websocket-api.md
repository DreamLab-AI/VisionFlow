# WebSocket API Reference

## Overview

VisionFlow uses WebSocket connections for real-time bidirectional communication between clients and the server. The WebSocket API handles high-frequency position updates, real-time graph changes, and live system events.

**WebSocket Endpoint**: `ws://localhost:3001/ws`

## Connection

### Establishing Connection

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

High-frequency position updates use the binary protocol (34 bytes per node):

```javascript
ws.onmessage = (event) => {
  if (event.data instanceof ArrayBuffer) {
    // Binary position update
    const updates = parseBinaryNodeData(event.data);
    // Process position updates...
  }
};
```

See [Binary Protocol Specification](binary-protocol.md) for detailed format.

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

### Voice Command
Sends voice commands for processing.

```json
{
  "type": "voice_command",
  "command": "show me all agents",
  "confidence": 0.95
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
Controls agent behavior and coordination.

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
Continuous stream of node positions at 60 FPS.

**Format**: 34-byte binary frames containing:
- Node ID with control bits (2 bytes)
- Position XYZ (12 bytes)
- Velocity XYZ (12 bytes)
- SSSP distance (4 bytes)
- SSSP parent (4 bytes)

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
Real-time agent status changes.

```json
{
  "type": "agent_status",
  "agentId": "agent_123",
  "status": "active",
  "health": {
    "cpu": 45.2,
    "memory": 1024,
    "taskQueue": 3
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

### System Events
Important system-wide events.

```json
{
  "type": "system_event",
  "event": "swarm_initialized",
  "data": {
    "swarmId": "swarm_789",
    "agentCount": 5,
    "topology": "mesh"
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

Use binary protocol for:
- Position updates (95% bandwidth reduction)
- High-frequency data (>10 Hz)
- Large numeric arrays

Use JSON for:
- Control messages
- Configuration updates
- Event notifications
- Human-readable data

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

### Binary Data Parser

```javascript
function parseBinaryNodeData(buffer) {
  const view = new DataView(buffer);
  const nodes = [];
  const nodeCount = buffer.byteLength / 34;

  for (let i = 0; i < nodeCount; i++) {
    const offset = i * 34;
    
    const nodeId = view.getUint16(offset, true);
    const isAgent = (nodeId & 0x8000) !== 0;
    const actualId = nodeId & 0x3FFF;
    
    nodes.push({
      id: actualId,
      isAgent,
      position: {
        x: view.getFloat32(offset + 2, true),
        y: view.getFloat32(offset + 6, true),
        z: view.getFloat32(offset + 10, true)
      },
      velocity: {
        x: view.getFloat32(offset + 14, true),
        y: view.getFloat32(offset + 18, true),
        z: view.getFloat32(offset + 22, true)
      },
      ssspDistance: view.getFloat32(offset + 26, true),
      ssspParent: view.getInt32(offset + 30, true)
    });
  }

  return nodes;
}
```

## Testing

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

## Related Documentation

- [Binary Protocol Specification](binary-protocol.md)
- [REST API](rest-api.md)
- [MCP Protocol](mcp-protocol.md)
- [Client WebSocket Integration](../../guides/websocket-integration.md)

---

**[← REST API](rest-api.md)** | **[Binary Protocol →](binary-protocol.md)**