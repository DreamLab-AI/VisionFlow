# VisionFlow API Quick Reference

## WebSocket Endpoints

```
ws://your-domain/wss                  # Main graph WebSocket (binary protocol)
ws://your-domain/ws/speech            # Speech recognition WebSocket
ws://your-domain/ws/mcp-relay         # MCP relay WebSocket (legacy)
ws://your-domain/ws/client-messages   # Agent → User messages WebSocket
ws://your-domain/api/mcp/ws           # MCP visualization WebSocket
ws://your-domain/api/agents/ws        # Agent visualization WebSocket
```

## REST API Endpoints

### Settings API
```
GET    /api/settings                      # Load all settings
POST   /api/settings                      # Update all settings
GET    /api/settings/health               # Health check
POST   /api/settings/reset                # Reset to defaults
GET    /api/settings/export               # Export as JSON
POST   /api/settings/import               # Import from JSON
POST   /api/settings/cache/clear          # Clear cache
GET    /api/settings/path/{path}          # Get single setting
PUT    /api/settings/path/{path}          # Update single setting
POST   /api/settings/batch                # Batch get settings
GET    /api/settings/physics/{graph}      # Get physics settings (logseq/visionflow)
PUT    /api/settings/physics/{graph}      # Update physics settings
```

### Graph API
```
GET    /api/graph/data                        # Get full graph with positions
GET    /api/graph/data/paginated              # Paginated graph data
POST   /api/graph/update                      # Update graph
POST   /api/graph/refresh                     # Refresh graph state
GET    /api/graph/auto-balance-notifications  # Get auto-balance events

# Graph State (CQRS)
GET    /api/graph/state                   # Current graph state
GET    /api/graph/statistics              # Graph statistics
POST   /api/graph/nodes                   # Add node
GET    /api/graph/nodes/{id}              # Get node
PUT    /api/graph/nodes/{id}              # Update node
DELETE /api/graph/nodes/{id}              # Remove node
POST   /api/graph/edges                   # Add edge
PUT    /api/graph/edges/{id}              # Update edge
POST   /api/graph/positions/batch         # Batch position update
```

### Files API
```
POST   /api/files/process                 # Process GitHub markdown
GET    /api/files/get_content/{filename}  # Get file content
POST   /api/files/refresh_graph           # Refresh from files
POST   /api/files/update_graph            # Update with new files
```

### Analytics API
```
GET    /api/analytics/params              # Get analytics params
POST   /api/analytics/params              # Update analytics params
POST   /api/analytics/anomaly/detect      # Run anomaly detection (GPU)
GET    /api/analytics/anomaly/status      # Anomaly status
GET    /api/analytics/anomaly/results     # Anomaly results
POST   /api/analytics/community/detect    # Community detection (GPU)
GET    /api/analytics/community/results   # Community results
```

### Clustering API
```
POST   /api/clustering/configure          # Configure clustering
POST   /api/clustering/start              # Start clustering
GET    /api/clustering/status             # Clustering status
GET    /api/clustering/results            # Clustering results
POST   /api/clustering/export             # Export assignments
```

### Constraints API
```
POST   /api/constraints/configure         # Configure constraints
GET    /api/constraints/status            # Constraint status
```

### Workspace API
```
GET    /api/workspace/list                # List workspaces
POST   /api/workspace/create              # Create workspace
GET    /api/workspace/count               # Workspace count
GET    /api/workspace/{id}                # Get workspace
PUT    /api/workspace/{id}                # Update workspace
DELETE /api/workspace/{id}                # Delete workspace
POST   /api/workspace/{id}/favorite       # Toggle favorite
POST   /api/workspace/{id}/archive        # Archive workspace
```

### Bots/Agents API
```
GET    /api/bots/data                     # Get bots data
POST   /api/bots/data                     # Update bots data
POST   /api/bots/update                   # Update bots (alias)
GET    /api/agents/visualization/snapshot # Agent snapshot
POST   /api/agents/visualization/init     # Init swarm visualization
```

### Ontology API
```
POST   /api/ontology/load                 # Load ontology
POST   /api/ontology/load-axioms          # Load axioms
GET    /api/ontology/axioms               # Get axioms
POST   /api/ontology/validate             # Validate ontology
POST   /api/ontology/query                # Query ontology
POST   /api/ontology/classify             # Classify ontology
```

### Quest 3 API
```
GET    /api/quest3/defaults               # Quest 3 optimized settings
POST   /api/quest3/calibrate              # Calibrate for Quest 3
```

### Sessions API
```
GET    /api/sessions/list                 # List sessions
GET    /api/sessions/{uuid}/status        # Session status
GET    /api/sessions/{uuid}/telemetry     # Session telemetry
```

### Client Logs API
```
POST   /api/client-logs                   # Receive browser logs
```

### Health/Monitoring API
```
GET    /api/health                        # Unified health check
GET    /api/health/physics                # Physics health
POST   /api/health/mcp/start              # Start MCP relay
GET    /api/health/mcp/logs               # MCP logs
```

### RAGFlow Chat API
```
POST   /api/ragflow/chat                  # Chat with RAGFlow
GET    /api/ragflow/sessions              # List chat sessions
```

### Nostr API
```
POST   /api/nostr/publish                 # Publish to Nostr
GET    /api/nostr/events                  # Get Nostr events
```

### Graph Export/Sharing
```
POST   /api/graph/export                  # Export graph
GET    /api/graph/share/{id}              # Get shared graph
```

### Validation API
```
POST   /api/validation/test/{type}        # Test validation
GET    /api/validation/stats              # Validation stats
```

### MCP API
```
GET    /api/mcp/status                    # MCP server status
POST   /api/mcp/refresh                   # Refresh MCP discovery
```

## WebSocket Message Formats

### Main Graph WebSocket (/wss)

**Client → Server:**
```json
{
  "type": "register",
  "clientId": "unique-client-id"
}

{
  "type": "ping",
  "timestamp": 1234567890
}
```

**Server → Client:**
```
Binary: BinaryNodeData (protocol buffer)
  - Node ID (u32)
  - Position (x, y, z) (f32)
  - Velocity (x, y, z) (f32)
  - Additional metadata

JSON:
{
  "type": "pong",
  "timestamp": 1234567890
}

{
  "type": "registered",
  "clientId": 123
}
```

### Client Logs Format (POST /api/client-logs)

```json
{
  "logs": [
    {
      "level": "error",
      "namespace": "GraphRenderer",
      "message": "Failed to render frame",
      "timestamp": "2025-10-23T12:00:00Z",
      "data": { "nodeCount": 1000 },
      "userAgent": "Quest3/...",
      "url": "https://visionflow.info",
      "stack": "Error: ...\n  at ..."
    }
  ],
  "sessionId": "uuid-v4",
  "timestamp": "2025-10-23T12:00:00Z"
}
```

## Response Formats

### Success Response
```json
{
  "success": true,
  "data": { ... },
  "message": "Operation completed"
}
```

### Error Response
```json
{
  "error": "error_code",
  "message": "Human-readable error message",
  "details": { ... }
}
```

### Graph Data Response
```json
{
  "nodes": [
    {
      "id": 1,
      "metadataId": "file1.md",
      "label": "Node Label",
      "position": { "x": 1.0, "y": 2.0, "z": 3.0 },
      "velocity": { "x": 0.1, "y": 0.2, "z": 0.3 },
      "metadata": { "key": "value" },
      "type": "document",
      "size": 1.0,
      "color": "#FF0000",
      "weight": 1.0,
      "group": "group1"
    }
  ],
  "edges": [
    {
      "source": 1,
      "target": 2,
      "weight": 1.0
    }
  ],
  "metadata": {
    "file1.md": {
      "title": "Document Title",
      "content": "...",
      "tags": ["tag1", "tag2"]
    }
  },
  "settlementState": {
    "isSettled": true,
    "stableFrameCount": 100,
    "kineticEnergy": 0.01
  }
}
```

## Common Headers

### Request Headers
```
Content-Type: application/json
X-Session-ID: uuid-v4                    # For session correlation
X-Client-ID: unique-client-id            # For client identification
```

### Response Headers
```
Content-Type: application/json
Cache-Control: no-store                  # For API endpoints
X-Correlation-ID: uuid-v4                # For request tracing
```

## Rate Limits

- **WebSocket Position Updates**: Dynamic (1-30 Hz based on motion)
- **REST API**: No hard limits (nginx connection limits apply)
- **Client Logs**: No hard limits (be reasonable)

## Error Codes

- `400` - Bad Request (invalid input)
- `404` - Not Found (resource doesn't exist)
- `500` - Internal Server Error (backend error)
- `503` - Service Unavailable (actor system down)

## Configuration

**Backend Server**: `http://localhost:4000` (internal)
**nginx Reverse Proxy**: Port 4000 (external)
**WebSocket Timeout**: 10 hours (600 minutes)
**API Timeout**: 120 seconds

## Authentication

⚠️ **Currently**: No authentication required
⚠️ **Production**: Implement JWT or API key authentication

## Notes

- All timestamps are Unix epoch (seconds or milliseconds)
- All coordinates use right-handed coordinate system
- Graph IDs are `u32` (0 to 4,294,967,295)
- WebSocket uses binary protocol for efficiency
- Settings API uses CQRS pattern (separate read/write paths)
- GPU acceleration is optional (falls back to CPU)

---

**Last Updated**: October 23, 2025
**Backend Version**: 0.1.0
**API Version**: 1.0
