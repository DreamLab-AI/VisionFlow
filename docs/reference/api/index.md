# API Reference

## Overview

VisionFlow provides multiple API interfaces for different use cases:

- **[REST API](rest-api.md)** - Traditional HTTP endpoints for data management and control
- **[WebSocket API](websocket-api.md)** - Real-time bidirectional communication
- **[Binary Protocol](binary-protocol.md)** - High-performance position updates
- **[MCP Protocol](mcp-protocol.md)** - Multi-agent system integration

## Quick Start

### REST API Example
```bash
# Get agent data
curl http://localhost:3001/api/bots/data

# Submit a task
curl -X POST http://localhost:3001/api/bots/submit-task \
  -H "Content-Type: application/json" \
  -d '{"task": "Analyze codebase", "priority": "high"}'
```

### WebSocket Connection
```javascript
const ws = new WebSocket('ws://localhost:3001/ws');
ws.onmessage = (event) => {
  if (event.data instanceof ArrayBuffer) {
    // Binary protocol message
    const update = parseBinaryNodeData(event.data);
  } else {
    // JSON message
    const message = JSON.parse(event.data);
  }
};
```

## API Categories

### 1. Agent Management APIs
- Agent spawning and lifecycle
- Task submission and tracking
- Swarm coordination
- Status monitoring

### 2. Graph Data APIs
- Node and edge management
- Physics simulation parameters
- Clustering and constraints
- Visualization settings

### 3. System APIs
- Authentication and sessions
- Settings management
- Health monitoring
- Telemetry and metrics

### 4. Integration APIs
- GitHub integration
- RAGFlow knowledge base
- Voice services (TTS/STT)
- External AI services

## Protocol Overview

### Communication Layers

```
┌─────────────────────┐
│   Client (React)    │
├─────────────────────┤
│  REST │ WS │ Binary │  ← API Layers
├─────────────────────┤
│   Rust Backend      │
├─────────────────────┤
│ MCP │ Redis │ PG    │  ← Integration Layer
└─────────────────────┘
```

### Data Flow Architecture

1. **High-Frequency Updates** (60 FPS)
   - Binary WebSocket protocol
   - Position and velocity data
   - Minimal latency (<2ms)

2. **Metadata and Control** (On-demand)
   - REST API endpoints
   - Configuration changes
   - Task management

3. **Agent Communication** (Event-driven)
   - MCP TCP protocol
   - Inter-agent messaging
   - Swarm coordination

## Authentication

All APIs support Nostr-based authentication:

```javascript
// REST API Authentication
const headers = {
  'Authorization': `Bearer ${jwtToken}`,
  'X-Nostr-Pubkey': publicKey
};

// WebSocket Authentication
const ws = new WebSocket(`ws://localhost:3001/ws?token=${jwtToken}`);
```

## Rate Limiting

Default rate limits:
- REST API: 1000 requests per 15 minutes
- WebSocket: 1000 messages per minute
- Binary updates: Unlimited (server-controlled)

## Error Handling

All APIs follow consistent error response format:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": {
      "field": "task",
      "reason": "Task description is required"
    }
  }
}
```

## API Versioning

Current API version: `v1`

Future versions will be available at:
- REST: `/api/v2/...`
- WebSocket: `/ws/v2`

## Related Documentation

- [Configuration Reference](../configuration.md)
- [Authentication Guide](../../guides/authentication.md)
- [Integration Guide](../../guides/integration.md)

---

**[← Back to Reference](../index.md)** | **[REST API →](rest-api.md)**