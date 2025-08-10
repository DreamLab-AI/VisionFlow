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
    end
    
    subgraph "API Layer"
        REST[REST API<br/>:3001/api]
        WS[WebSocket<br/>:3001/ws]
        Binary[Binary Protocol]
    end
    
    subgraph "Endpoints"
        Graph[/api/graph]
        Settings[/api/settings]
        Bots[/api/bots]
        Files[/api/files]
        Voice[/ws/speech]
        Flow[/ws/socket_flow]
        MCP[/ws/mcp_relay]
        Viz[/ws/bots_visualization]
    end
    
    Web --> REST
    Web --> WS
    XR --> WS
    API --> REST
    
    REST --> Graph
    REST --> Settings
    REST --> Bots
    REST --> Files
    
    WS --> Voice
    WS --> Flow
    WS --> MCP
    WS --> Viz
    
    Flow --> Binary
    Viz --> Binary
```

## API Types

### [REST API](rest.md)
HTTP-based API for CRUD operations, configuration management, and service integration.

**Key Endpoints:**
- `/api/graph` - Graph data management
- `/api/settings` - Configuration and settings
- `/api/bots` - AI agent management
- `/api/files` - File operations
- `/api/quest3` - Quest 3 specific features
- `/api/visualisation` - Visualization controls

### [WebSocket API](websocket.md)
Real-time bidirectional communication for streaming updates and interactive features.

**WebSocket Endpoints:**
- `/ws/socket_flow` - Binary graph updates
- `/ws/speech` - Voice interaction streaming
- `/ws/mcp_relay` - MCP protocol relay
- `/ws/bots_visualization` - Agent swarm visualization

### [Binary Protocol](binary-protocol.md)
Highly optimized binary format for efficient data transmission.

**Protocol Features:**
- 28-byte node format
- Differential updates
- GPU memory alignment
- Type flags for dual-graph support

## Authentication

### Nostr Authentication
- NIP-07 browser extension support
- Event signing and verification
- Decentralized identity management

### API Key Authentication
- Bearer token in Authorization header
- Session-based authentication
- Rate limiting per key

## Rate Limiting

| Endpoint Type | Rate Limit | Window |
|--------------|------------|--------|
| REST API | 100 req/min | 1 minute |
| WebSocket | 1000 msg/min | 1 minute |
| Binary Stream | Unlimited | N/A |
| File Upload | 10 req/min | 1 minute |

## Error Handling

### HTTP Status Codes
- `200` - Success
- `201` - Created
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `429` - Rate Limited
- `500` - Internal Server Error

### WebSocket Close Codes
- `1000` - Normal closure
- `1001` - Going away
- `1002` - Protocol error
- `1003` - Unsupported data
- `1008` - Policy violation
- `1011` - Server error

## Response Formats

### Success Response
```json
{
  "success": true,
  "data": { ... },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### Error Response
```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Human readable message",
    "details": { ... }
  },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## CORS Configuration

```javascript
// Allowed origins
Access-Control-Allow-Origin: http://localhost:3001
Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS
Access-Control-Allow-Headers: Content-Type, Authorization
Access-Control-Allow-Credentials: true
```

## API Versioning

The API uses URL-based versioning:
- Current version: `v1`
- Base URL: `http://localhost:3001/api/v1`
- WebSocket: `ws://localhost:3001/ws`

## Performance Considerations

### Pagination
All list endpoints support pagination:
- `?page=1&limit=100` - Page number and items per page
- `?cursor=abc123` - Cursor-based pagination for large datasets

### Compression
- HTTP responses use gzip compression
- WebSocket messages use MessagePack for JSON
- Binary protocol uses raw bytes (no compression needed)

### Caching
- ETags for resource versioning
- Cache-Control headers for static resources
- WebSocket uses differential updates

## SDK Support

### JavaScript/TypeScript
```typescript
import { VisionFlowClient } from '@visionflow/client';

const client = new VisionFlowClient({
  baseUrl: 'http://localhost:3001',
  wsUrl: 'ws://localhost:3001',
  apiKey: 'your-api-key'
});
```

### Python
```python
from visionflow import Client

client = Client(
    base_url="http://localhost:3001",
    api_key="your-api-key"
)
```

### Rust
```rust
use visionflow_client::Client;

let client = Client::new(
    "http://localhost:3001",
    "your-api-key"
);
```

## Testing

### API Testing Tools
- Postman collection available
- OpenAPI/Swagger specification
- WebSocket testing with wscat
- Binary protocol test client

### Example cURL Commands
```bash
# GET graph data
curl http://localhost:3001/api/graph

# POST new settings
curl -X POST http://localhost:3001/api/settings \
  -H "Content-Type: application/json" \
  -d '{"theme": "dark"}'

# WebSocket connection
wscat -c ws://localhost:3001/ws/socket_flow
```