# VisionFlow API Documentation

This directory contains comprehensive API reference documentation for all VisionFlow services.

## Available API Documentation

### Ontology API

**File**: [ontology-endpoints.md](./ontology-endpoints.md)

Complete REST and WebSocket API reference for the ontology validation system.

**Topics Covered**:
- 11 REST endpoints with curl examples
- WebSocket protocol and message types
- Request/response schemas
- Authentication and rate limiting
- Error codes and handling
- Best practices and example workflows

**Quick Links**:
- [Load Ontology](./ontology-endpoints.md#load-ontology)
- [Validate Graph](./ontology-endpoints.md#validate-graph-against-ontology)
- [Get Report](./ontology-endpoints.md#get-validation-report)
- [WebSocket API](./ontology-endpoints.md#websocket-api)
- [Error Codes](./ontology-endpoints.md#error-codes-reference)

## API Overview

### REST APIs

| API | Base Path | Endpoints | Status |
|-----|-----------|-----------|--------|
| Ontology | `/api/ontology` | 11 | Production |
| Graph | `/api/graph` | 19 | Production |
| Agents | `/api/agents` | 8 | Production |
| Analytics | `/api/analytics` | 5 | Production |

### WebSocket APIs

| Protocol | Path | Purpose | Status |
|----------|------|---------|--------|
| Binary | `/ws` | Real-time graph updates | Production |
| JSON | `/api/ontology/ws` | Validation progress | Production |

## Authentication

All API endpoints require authentication using JWT tokens:

```http
Authorization: Bearer <jwt_token>
```

Obtain tokens via the `/api/auth/login` endpoint.

## Rate Limiting

| Limit Type | Threshold | Window |
|------------|-----------|--------|
| Per-IP | 100 requests | 1 minute |
| Per-User | 1000 requests | 1 hour |
| WebSocket | 10 connections | Per client |

## Base URLs

| Environment | Base URL | WebSocket URL |
|-------------|----------|---------------|
| Development | `http://localhost:8080` | `ws://localhost:8080` |
| Production | `https://api.visionflow.io` | `wss://api.visionflow.io` |

## API Versioning

Current API version: **v1**

Future versions will maintain backward compatibility:
- `/api/v1/*` - Current stable API
- `/api/v2/*` - Future enhancements

## Error Handling

All APIs return standardized error responses:

```json
{
  "error": "Description of error",
  "code": "ERROR_CODE",
  "details": { },
  "timestamp": "2025-10-17T12:34:56Z",
  "trace_id": "uuid"
}
```

See individual API documentation for specific error codes.

## Client Libraries

### TypeScript

```bash
npm install @visionflow/client
```

```typescript
import { VisionFlowClient } from '@visionflow/client';

const client = new VisionFlowClient('http://localhost:8080');
await client.authenticate(token);

// Use ontology API
const { ontology_id } = await client.ontology.load(ontologyContent);
const report = await client.ontology.validate(ontology_id, 'Full');
```

### Python (Planned)

```bash
pip install visionflow-client
```

### Rust (Planned)

```toml
[dependencies]
visionflow-client = "1.0"
```

## Example Workflows

### Complete Validation Workflow

```bash
#!/bin/bash

# 1. Load ontology
ONTOLOGY_ID=$(curl -X POST http://localhost:8080/api/ontology/load \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d @ontology.json | jq -r '.ontology_id')

# 2. Validate graph
JOB_ID=$(curl -X POST http://localhost:8080/api/ontology/validate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d "{\"ontology_id\":\"$ONTOLOGY_ID\",\"mode\":\"Full\"}" \
  | jq -r '.job_id')

# 3. Get report
curl http://localhost:8080/api/ontology/reports/$JOB_ID \
  -H "Authorization: Bearer $TOKEN" | jq
```

### Real-Time Updates with WebSocket

```javascript
const ws = new WebSocket('ws://localhost:8080/api/ontology/ws?client_id=my-client');

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);

  switch (message.type) {
    case 'validation_progress':
      console.log(`Progress: ${message.progress * 100}%`);
      break;
    case 'validation_complete':
      console.log('Validation complete!');
      fetchReport(message.report_id);
      break;
  }
};
```

## Related Documentation

- [Ontology Feature Documentation](../features/ontology-system.md)
- [System Architecture](../architecture/system-overview.md)
- [Data Storage](../architecture/data-storage.md)
- [Getting Started Guide](../getting-started/01-installation.md)

## Support

For API issues or questions:

- [GitHub Issues](https://github.com/yourusername/VisionFlow/issues)
- [API Status Page](https://status.visionflow.io) (planned)
- [Developer Forum](https://forum.visionflow.io) (planned)

## Contributing

To add new API documentation:

1. Create a new markdown file for your API
2. Follow the ontology-endpoints.md template structure
3. Include working curl examples
4. Document all error codes
5. Add client library examples
6. Update this README

## Documentation Standards

All API documentation should include:

- Complete endpoint list with HTTP methods
- Request/response schemas
- Authentication requirements
- Rate limiting information
- curl command examples
- Error code reference
- Best practices
- Example workflows
- Client library examples
