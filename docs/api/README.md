# VisionFlow API Documentation

*[Documentation Home](../index.md)*

## Overview

VisionFlow provides a comprehensive API ecosystem for real-time graph visualization, AI agent orchestration, and analytics. The API is built on a production-ready architecture with advanced performance optimizations, comprehensive validation, and enterprise-grade security features.

## Quick Start

### Base URLs
```
Development: http://localhost:3001/api
Production:  https://api.visionflow.dev/api
WebSocket:   ws://localhost:3001/wss (or wss://api.visionflow.dev/wss)
```

### Authentication
VisionFlow uses Nostr-based authentication for secure access:

```bash
curl -X POST http://localhost:3001/api/auth/nostr \
  -H "Content-Type: application/json" \
  -d '{
    "pubkey": "user_public_key_hex",
    "signature": "signature_hex",
    "challenge": "server_challenge"
  }'
```

## API Components

### 🌐 REST Endpoints
**File:** [rest-endpoints.md](rest-endpoints.md)

Complete REST API reference with:
- Graph management and data operations
- AI agent orchestration and swarm control
- System configuration and settings
- Analytics and performance monitoring
- Health checks and service status

**Key Features:**
- Automatic camelCase ↔ snake_case conversion
- Comprehensive input validation
- Standardized error responses
- Rate limiting and security controls

### ⚡ WebSocket Streams
**File:** [websocket-streams.md](websocket-streams.md)

Real-time communication protocols:
- Binary position updates (34-byte optimized format)
- Agent visualization streaming
- MCP protocol relay (JSON-RPC 2.0)
- Speech/voice interaction streams

**Performance:**
- 84.8% bandwidth reduction through binary protocol
- 5Hz real-time updates with burst handling
- Priority queuing for agent nodes

### 📡 Binary Protocol
**File:** [binary-protocol.md](binary-protocol.md)

Ultra-optimized wire format specification:
- **34 bytes per node** (ID: 2 bytes, Position: 12 bytes, Velocity: 12 bytes, SSSP: 8 bytes)
- Node type flags for agent/knowledge classification
- Little-endian byte order for cross-platform compatibility
- Zero-copy serialization optimizations

## Core API Categories

### Graph Management
- **GET /api/graph/data** - Retrieve graph structure and node data
- **POST /api/graph/update** - Update graph from file sources
- **POST /api/graph/refresh** - Refresh graph topology

### Agent Orchestration
- **GET /api/bots/data** - List active AI agents
- **POST /api/bots/initialize-swarm** - Create multi-agent swarms
- **WebSocket /ws/bots_visualization** - Real-time agent monitoring

### Analytics & Insights
- **POST /api/analytics/shortest-path** - GPU-accelerated SSSP computation
- **GET /api/analytics/clustering** - Semantic clustering analysis
- **GET /api/analytics/system** - System performance metrics

### Configuration
- **GET /api/settings** - Retrieve system configuration
- **POST /api/settings** - Update settings with validation
- **WebSocket binary settings** - Real-time settings synchronization

### MCP Integration
- **TCP Port 9500** - MCP server with JSON-RPC 2.0
- **WebSocket /ws/mcp-relay** - MCP protocol relay for web clients
- Multi-swarm addressing and lifecycle management

## Performance Characteristics

### Network Efficiency
| Protocol | 100 Nodes @ 60fps | 1000 Nodes @ 60fps | Bandwidth Reduction |
|----------|-------------------|---------------------|-------------------|
| JSON | 3 MB/s | 30 MB/s | - |
| Binary | 168 KB/s | 1.68 MB/s | **94%** |
| Compressed Binary | 100 KB/s | 1 MB/s | **97%** |

### Real-Time Updates
- **Update Rate**: 5Hz minimum, up to 60Hz for high activity
- **Latency**: <10ms processing overhead
- **Throughput**: 300 requests/minute with 50-message burst tolerance
- **Priority Processing**: Agent nodes receive preferential treatment

## Architecture Features

### Production-Ready Infrastructure
- **Comprehensive Validation**: Multi-tier input validation with detailed error feedback
- **Advanced Security**: Input sanitization, rate limiting, malicious content detection
- **High Performance**: Binary protocols and efficient data serialization
- **Monitoring**: Real-time metrics, health checks, and observability
- **Developer Experience**: Intuitive schemas and helpful error messages

### Scalability & Reliability
- **Connection Management**: Exponential backoff reconnection with heartbeat monitoring
- **Message Queuing**: Persistent queues with retry logic during disconnections
- **Rate Limiting**: Adaptive limits per endpoint with burst allowances
- **Error Recovery**: Graceful degradation and automatic failover

## Integration Examples

### TypeScript/JavaScript
```typescript
import { VisionFlowClient } from '@visionflow/sdk';

const client = new VisionFlowClient({
  baseUrl: 'http://localhost:3001/api',
  token: 'your_auth_token'
});

// Get graph data
const graph = await client.graph.getData();

// Connect to real-time updates
const ws = client.websocket.connect('/wss');
ws.onBinaryMessage = (data) => {
  const positions = decodeBinaryPositions(data);
  updateVisualization(positions);
};
```

### React Integration
```typescript
function useVisionFlowAPI() {
  const [graph, setGraph] = useState(null);
  const [positions, setPositions] = useState(new Map());

  useEffect(() => {
    // REST API for initial data
    client.graph.getData().then(setGraph);

    // WebSocket for real-time updates
    const ws = client.websocket.connect();
    ws.onBinaryMessage = (data) => {
      const updates = decodeBinaryPositions(data);
      setPositions(prev => new Map([...prev, ...updates]));
    };

    return () => ws.close();
  }, []);

  return { graph, positions };
}
```

## Error Handling

### Standard Response Format
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_FAILED",
    "message": "Request validation failed",
    "details": {
      "field": "graph.nodes[2].color",
      "provided": "#zzzzzz",
      "expected": "hex color format (#RRGGBB or #RGB)"
    },
    "suggestions": ["Use hex colors like '#ff0000' or '#f00'"]
  }
}
```

### Common Error Codes
| Code | Description | HTTP Status |
|------|-------------|-------------|
| `UNAUTHORIZED` | Authentication required | 401 |
| `VALIDATION_ERROR` | Input validation failed | 422 |
| `RATE_LIMITED` | Too many requests | 429 |
| `RESOURCE_LIMIT_EXCEEDED` | Request exceeds limits | 413 |

## Rate Limits

| Category | Requests/Min | Burst | Notes |
|----------|--------------|-------|-------|
| Graph Operations | 100 | 20 | CRUD operations |
| Agent Operations | 50 | 10 | Swarm management |
| Analytics | 200 | 50 | Queries and computations |
| WebSocket Messages | 1000 | 100 | Real-time updates |
| Binary Updates | Server-controlled | - | 5Hz default rate |

## Security Features

### Input Validation
- Multi-layer validation (syntax, semantics, security, business logic)
- Malicious content detection and blocking
- Bounds checking and resource limits
- Type safety enforcement

### Authentication & Authorization
- Nostr-based authentication (NIP-07 protocol)
- Session-based WebSocket authentication
- Permission-based feature access control
- Security headers and CORS configuration

## API Evolution

### Current Version (v1.0)
- 34-byte binary protocol format
- u32 node IDs with type flags
- Automatic case conversion (camelCase ↔ snake_case)
- Comprehensive validation system

### Backward Compatibility
- Version detection during handshake
- Graceful degradation for unsupported features
- Migration guides for breaking changes

## Documentation Structure

```
api/
├── README.md               # This overview
├── rest-endpoints.md       # Complete REST API reference
├── websocket-streams.md    # Real-time communication protocols
├── binary-protocol.md      # Wire format specification
├── index.md               # Legacy main API doc (comprehensive)
├── websocket.md           # Legacy WebSocket reference
├── websocket-protocols.md  # Legacy protocols overview
├── rest/
│   ├── index.md           # REST API summary
│   ├── graph.md           # Graph management endpoints
│   └── settings.md        # Settings API details
└── mcp/
    └── index.md           # MCP integration guide
```

## Support Resources

- **API Status**: [status.visionflow.dev](https://status.visionflow.dev)
- **OpenAPI Spec**: [/api/openapi.json](http://localhost:3001/api/openapi.json)
- **Health Check**: [/api/health](http://localhost:3001/api/health)

## Next Steps

1. Review [rest-endpoints.md](rest-endpoints.md) for complete REST API documentation
2. See [websocket-streams.md](websocket-streams.md) for real-time communication protocols
3. Check [binary-protocol.md](binary-protocol.md) for wire format specifications
4. Explore integration examples in respective endpoint documentation

---

*Last updated: 2025-09-16*