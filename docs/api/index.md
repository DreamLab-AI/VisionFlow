# VisionFlow API Documentation

## Overview

VisionFlow provides a comprehensive API surface consisting of REST endpoints for data operations, WebSocket connections for real-time updates, and an efficient binary protocol for high-performance streaming.

## API Architecture

```mermaid
graph LR
    subgraph "Client Applications"
        Web[Web Client]
        XR[XR Client]
        API[API Client]
        Mobile[Mobile Client]
    end

    subgraph "API Layer"
        REST[REST API<br/>:3001/api]
        WS[WebSocket<br/>:3001/ws/*]
        Binary[Binary Protocol]
    end

    subgraph "Core Endpoints"
        Graph[/api/graph/*]
        Settings[/api/settings/*]
        Bots[/api/bots/*]
        Files[/api/files/*]
        Analytics[/api/analytics/*]
    end

    subgraph "WebSocket Streams"
        Flow[/wss<br/>Graph Updates]
        Speech[/ws/speech<br/>Voice I/O]
        MCPRelay[/ws/mcp-relay<br/>Agent Control]
        BotsViz[/ws/bots_visualization<br/>multi-agent State]
    end

    Web --> REST
    Web --> WS
    XR --> WS
    API --> REST
    Mobile --> REST

    REST --> Graph
    REST --> Settings
    REST --> Bots
    REST --> Files
    REST --> Analytics

    WS --> Flow
    WS --> Speech
    WS --> MCPRelay
    WS --> BotsViz

    Flow --> Binary
    BotsViz --> Binary
```

## API Components

### [REST API](rest.md)
HTTP-based API for CRUD operations, configuration management, and service integration.

**Key Features:**
- Graph data management and analytics
- AI agent orchestration and monitoring
- Settings and configuration control
- File processing and content management
- Quest 3 / XR integration
- External service integrations (RAGFlow, Perplexity, GitHub)

**Base URL:** `http://localhost:3001/api`

### [WebSocket API](websocket.md)
Real-time bidirectional communication for streaming updates and interactive features.

**Primary Endpoints:**
- `/wss` - Binary graph position updates (28-byte protocol)
- `/ws/speech` - Voice interaction streaming
- `/ws/mcp-relay` - MCP protocol relay for Claude Flow
- `/ws/bots_visualization` - Multi Agent Visualisation

### [Binary Protocol](binary-protocol.md)
Highly optimized binary format for efficient data transmission.

**Protocol Features:**
- 28-byte fixed-size node format
- Node type flags for agent/knowledge classification
- GPU memory alignment optimization
- Differential updates and compression support

### [WebSocket Protocols](websocket-protocols.md)
Comprehensive documentation of WebSocket message formats and protocols.

**Message Types:**
- JSON control messages
- Binary position updates
- Agent state updates
- Speech streaming data
- MCP tool invocations

## API Overview

### REST Endpoints Summary

| Category | Endpoints | Purpose |
|----------|-----------|---------|
| **Graph** | `/api/graph/*` | Node/edge data, layouts, analytics |
| **Agents** | `/api/bots/*` | multi-agent management, metrics, orchestration |
| **Settings** | `/api/settings/*` | Configuration, physics, visualization |
| **Files** | `/api/files/*` | Content processing, metadata management |
| **Quest 3** | `/api/quest3/*` | XR session management, device status |
| **Visualization** | `/api/visualisation/*` | 3D rendering configuration |
| **Analytics** | `/api/analytics/*` | Performance metrics, system health |
| **Health** | `/api/health`, `/api/mcp/health` | Service status monitoring |
| **External** | `/api/ragflow/*`, `/api/perplexity/*` | AI service integration |

### WebSocket Endpoints Summary

| Endpoint | Protocol | Update Rate | Purpose |
|----------|----------|-------------|---------|
| `/wss` | Binary + JSON | 5-60 Hz | Graph position streaming |
| `/ws/speech` | JSON + Binary Audio | Real-time | Voice interaction |
| `/ws/mcp-relay` | JSON-RPC 2.0 | On-demand | Agent control via MCP |
| `/ws/bots_visualization` | JSON | ~60 FPS | multi-agent state visualization |

## Authentication

### Nostr-Based Authentication
VisionFlow uses decentralized authentication via the Nostr protocol (NIP-07).

**Primary Method:**
- Browser extension integration (Alby, nos2x, etc.)
- Event signing for identity verification
- Session-based authentication after initial verification
- Feature-based access control

**Supported Features:**
- Public key identity (`npub` encoding)
- Event signing and verification
- Relay-based identity resolution
- Power user detection and permissions

### Session Management
- HTTP sessions for REST API access
- WebSocket authentication inheritance from HTTP session
- Token-based alternative for programmatic access
- Automatic session refresh and timeout handling

## Rate Limiting

| API Type | Standard Limit | Burst Limit | Window |
|----------|----------------|-------------|--------|
| REST API | 100 req/min | 20 requests | 1 minute |
| WebSocket Control | 1000 msg/min | 100 messages | 1 minute |
| Binary Stream | Server-controlled | Unlimited* | N/A |
| File Operations | 10 req/min | 5 requests | 1 minute |

*Binary updates use intelligent server-side throttling based on graph activity

## Error Handling

### Standard Error Response
```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Human readable message",
    "details": {
      "field": "specific_field",
      "reason": "validation_error"
    },
    "requestId": "req-123"
  },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### Common Error Codes
- `UNAUTHORIZED` (401) - Authentication required
- `FORBIDDEN` (403) - Insufficient permissions
- `NOT_FOUND` (404) - Resource not found
- `INVALID_REQUEST` (400) - Invalid parameters
- `RATE_LIMITED` (429) - Too many requests
- `INTERNAL_ERROR` (500) - Server error
- `AGENT_NOT_FOUND` (404) - Agent does not exist
- `multi-agent_ERROR` (500) - multi-agent operation failed
- `GPU_ERROR` (500) - GPU computation error

## Performance Characteristics

### Binary Protocol Efficiency

| Scenario | JSON (MB/s) | Binary (KB/s) | Reduction |
|----------|-------------|---------------|-----------|
| 100 agents @ 60fps | 3.0 | 168 | 94.4% |
| 500 agents @ 60fps | 15.0 | 840 | 94.4% |
| 1000 agents @ 60fps | 30.0 | 1680 | 94.4% |
| Mixed graph (10k nodes) | 25.0 | 1200 | 95.2% |

### Optimization Features
1. **Compression**: permessage-deflate for WebSocket messages >1KB
2. **Batching**: Multiple updates in single frames
3. **Delta Updates**: Only changed nodes transmitted
4. **Type Flags**: Efficient node classification
5. **GPU Alignment**: Memory layout optimized for GPU processing

## Integration Examples

### JavaScript/TypeScript Client
```typescript
import { VisionFlowClient } from '@visionflow/client';

const client = new VisionFlowClient({
  baseUrl: 'http://localhost:3001',
  wsUrl: 'ws://localhost:3001',
  authentication: 'nostr'
});

// REST API usage
const graphData = await client.graph.getData();
const agents = await client.bots.listAgents();

// WebSocket streaming
client.websocket.onPositionUpdates(updates => {
  updateVisualization(updates);
});

client.websocket.onAgentStateChanged(agent => {
  updateAgentDisplay(agent);
});
```

### Python Client Example
```python
from visionflow import Client

client = Client(
    base_url="http://localhost:3001",
    ws_url="ws://localhost:3001",
    auth_method="session"
)

# Graph operations
graph_data = client.graph.get_data()
client.graph.update()

# Agent management
agents = client.bots.list_agents()
client.bots.initialize_multi-agent(topology="hierarchical", max_agents=10)
```

### cURL Examples
```bash
# Health check
curl http://localhost:3001/api/health

# Get graph data
curl http://localhost:3001/api/graph/data

# Initialize Multi Agent
curl -X POST http://localhost:3001/api/bots/initialize-multi-agent \
  -H "Content-Type: application/json" \
  -d '{"topology":"hierarchical","maxAgents":5}'

# Update settings
curl -X POST http://localhost:3001/api/settings \
  -H "Content-Type: application/json" \
  -d '{"path":"physics.gravity","value":-12.0}'
```

## Development Tools

### API Documentation
- **Interactive Docs**: Available at `/docs/swagger` (when enabled)
- **OpenAPI Spec**: Downloadable at `/docs/openapi.json`
- **Postman Collection**: Available in `/docs/postman/`

### WebSocket Testing
```bash
# Test WebSocket connection
wscat -c ws://localhost:3001/wss

# Send control message
{"type":"requestInitialData"}

# Monitor binary updates
# (binary data will appear as raw bytes)
```

### Debug Configuration
```bash
# Server-side logging
RUST_LOG=webxr::handlers=debug,webxr::utils::binary_protocol=trace cargo run

# Client-side debugging
localStorage.setItem('debug', 'visionflow:*,websocket:*,binary:*');
```

## Security Considerations

### Data Protection
- All inputs validated against schemas
- SQL injection prevention measures
- XSS prevention for user content
- File upload restrictions and scanning

### Network Security
- CORS properly configured for cross-origin requests
- WebSocket connection limits prevent DoS attacks
- Rate limiting per IP and authenticated user
- Binary message size limits (100MB max)

### Authentication Security
- Nostr cryptographic identity verification
- Session token rotation
- Feature-based permission enforcement
- Audit logging for sensitive operations

## CORS Configuration

```http
Access-Control-Allow-Origin: http://localhost:3001
Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS
Access-Control-Allow-Headers: Content-Type, Authorization, X-Request-ID
Access-Control-Allow-Credentials: true
Access-Control-Max-Age: 86400
```

## API Versioning

### Current Version (v1.0)
- Base URL: `http://localhost:3001/api`
- WebSocket: `ws://localhost:3001/ws/*`
- Binary protocol: 28-byte format with type flags
- Nostr-based authentication

### Version Compatibility
- **Backward compatibility** maintained within major versions
- **Deprecation notices** provided 6 months before removal
- **Migration guides** available for major version changes
- **Feature detection** available via capabilities endpoint

### Future Roadmap (v2.0)
- Enhanced binary protocol with variable-length encoding
- WebSocket protocol versioning and negotiation
- Extended agent capabilities and orchestration
- Advanced analytics and monitoring features

## Monitoring & Observability

### Health Endpoints
- `/api/health` - Overall system health
- `/api/mcp/health` - MCP connection status
- `/api/analytics/system` - Performance metrics

### Metrics Available
- **Performance**: FPS, frame time, memory usage
- **Network**: WebSocket client count, bandwidth, latency
- **Agents**: Task completion rates, token usage, success rates
- **Graph**: Node/edge counts, topology metrics, update frequencies

### Alerting Integration
- Prometheus metrics exposure (when enabled)
- Custom webhook notifications for critical errors
- Performance threshold monitoring
- Automatic failover detection

## Getting Started

### Quick Setup
1. **Start Server**: `cargo run` or use Docker
2. **Test Health**: `curl http://localhost:3001/api/health`
3. **Open WebSocket**: Connect to `ws://localhost:3001/wss`
4. **Initialize Graph**: `POST /api/graph/update`
5. **Start Multi Agent**: `POST /api/bots/initialize-multi-agent`

### Common Workflows
1. **Graph Visualization**: REST → WebSocket → Binary updates
2. **Agent Management**: Initialize multi-agent → Monitor via WebSocket → Control via MCP
3. **Voice Interaction**: Connect to `/ws/speech` → Stream audio → Receive transcriptions
4. **XR Integration**: Check Quest 3 status → Initialize XR session → Stream positions

### Best Practices
1. Use WebSocket connections for real-time features
2. Implement proper reconnection logic with exponential backoff
3. Handle binary message validation and error recovery
4. Cache REST API responses where appropriate
5. Monitor rate limits and implement client-side throttling
6. Use heartbeat/ping-pong for connection health monitoring

## Support & Resources

- **GitHub Repository**: [VisionFlow](https://github.com/your-org/visionflow)
- **Issues & Bug Reports**: [GitHub Issues](https://github.com/your-org/visionflow/issues)
- **API Documentation**: This documentation set
- **Community**: [Discord/Slack/Forum links]

For detailed technical specifications, see the individual documentation files:
- [REST API Reference](rest.md)
- [WebSocket API Reference](websocket.md)
- [Binary Protocol Specification](binary-protocol.md)
- [WebSocket Protocols Guide](websocket-protocols.md)