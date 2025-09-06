# Modern Settings API - Path-Based Architecture

This document describes the modern, performant path-based settings API that replaced the legacy bulk settings endpoints.

## Overview

The legacy bulk settings endpoints (`GET /api/settings` and `POST /api/settings`) have been removed and replaced with granular, path-based endpoints for better performance and efficiency.

## ✅ Modern Endpoints (Active)

### 1. Granular Path Access

#### Get Single Setting
```
GET /api/settings/path?path=visualisation.physics.damping
```
**Response:**
```json
{
  "value": 0.95,
  "path": "visualisation.physics.damping",
  "success": true
}
```

#### Update Single Setting
```
PUT /api/settings/path
Content-Type: application/json

{
  "path": "visualisation.physics.damping",
  "value": 0.98
}
```
**Response:**
```json
{
  "success": true,
  "path": "visualisation.physics.damping",
  "value": 0.98,
  "previousValue": 0.95,
  "message": "Settings updated successfully"
}
```

### 2. Batch Operations

#### Batch Read Multiple Settings
```
POST /api/settings/batch
Content-Type: application/json

{
  "paths": [
    "visualisation.physics.damping",
    "visualisation.physics.gravity"
  ]
}
```
**Response:**
```json
{
  "success": true,
  "message": "Successfully read 2 settings",
  "values": [
    {
      "path": "visualisation.physics.damping",
      "value": 0.95,
      "success": true
    },
    {
      "path": "visualisation.physics.gravity", 
      "value": 0.001,
      "success": true
    }
  ]
}
```

#### Batch Update Multiple Settings
```
PUT /api/settings/batch
Content-Type: application/json

{
  "updates": [
    {
      "path": "visualisation.physics.damping",
      "value": 0.98
    },
    {
      "path": "visualisation.physics.gravity",
      "value": 0.002
    }
  ]
}
```
**Response:**
```json
{
  "success": true,
  "message": "Successfully updated 2 settings",
  "results": [
    {
      "path": "visualisation.physics.damping",
      "success": true,
      "value": 0.98,
      "previousValue": 0.95
    },
    {
      "path": "visualisation.physics.gravity",
      "success": true,
      "value": 0.002,
      "previousValue": 0.001
    }
  ]
}
```

### 3. Schema Introspection

#### Get Settings Schema
```
GET /api/settings/schema?path=visualisation.physics
```
**Response:**
```json
{
  "path": "visualisation.physics",
  "schema": {
    "type": "object",
    "properties": {
      "damping": {
        "type": "number",
        "format": "float",
        "path": "visualisation.physics.damping"
      },
      "gravity": {
        "type": "number", 
        "format": "float",
        "path": "visualisation.physics.gravity"
      }
    },
    "path": "visualisation.physics"
  },
  "success": true
}
```

## ❌ Deprecated Endpoints (Removed)

The following bulk endpoints have been **removed** for performance reasons:

- ❌ `GET /api/settings` - Returned entire settings object (performance issue)
- ❌ `POST /api/settings` - Updated entire settings object (performance issue)

## Benefits of Path-Based Architecture

1. **Performance**: Only transfers requested data instead of entire settings object
2. **Granular Control**: Update individual settings without affecting others
3. **Batch Efficiency**: Update multiple specific settings in one request
4. **Schema Discovery**: Introspect settings structure programmatically
5. **Validation**: Path-specific validation and error handling
6. **Rate Limiting**: Per-endpoint rate limiting for better resource management

## Path Format

Settings paths use dot notation to navigate nested objects:
- `visualisation.physics.damping`
- `system.performance.maxThreads`
- `auth.oauth.clientId`
- `graphs.logseq.physics.springStrength`

## Error Handling

All endpoints return standardized error responses:
```json
{
  "error": "Path not found: Invalid path 'invalid.path'",
  "path": "invalid.path", 
  "success": false
}
```

## Migration Guide

### Before (Legacy)
```javascript
// Get all settings
const settings = await fetch('/api/settings').then(r => r.json());
const damping = settings.visualisation.physics.damping;

// Update settings
await fetch('/api/settings', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    visualisation: {
      physics: {
        damping: 0.98
      }
    }
  })
});
```

### After (Modern)
```javascript
// Get specific setting
const response = await fetch('/api/settings/path?path=visualisation.physics.damping');
const { value: damping } = await response.json();

// Update specific setting
await fetch('/api/settings/path', {
  method: 'PUT',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    path: 'visualisation.physics.damping',
    value: 0.98
  })
});
```

## Implementation Files

- `/src/handlers/settings_paths.rs` - Modern path-based endpoints
- `/src/handlers/settings_handler.rs` - Legacy endpoints (commented out)
- `/src/handlers/api_handler/mod.rs` - Route configuration

## Rate Limiting

All endpoints support rate limiting (when enabled):
- `settings_get_path` - Read operations
- `settings_update_path` - Single updates  
- `settings_batch_update` - Batch updates
- `settings_schema` - Schema requests

Batch operations are limited to 50 items maximum to prevent abuse.



## See Also

- [Configuration Guide](getting-started/configuration.md)
- [Getting Started with VisionFlow](getting-started/index.md)
- [Guides](guides/README.md)
- [Installation Guide](getting-started/installation.md)
- [Quick Start Guide](getting-started/quickstart.md)
- [VisionFlow Quick Start Guide](guides/quick-start.md)
- [VisionFlow Settings System Guide](guides/settings-guide.md)

## Related Topics

- [Analytics API Endpoints](api/analytics-endpoints.md)
- [Graph API Reference](api/rest/graph.md)
- [Multi-MCP Agent Visualisation API Reference](api/multi-mcp-visualization-api.md)
- [REST API Bloom/Glow Field Validation Fix](REST_API_BLOOM_GLOW_VALIDATION_FIX.md)
- [REST API Reference](api/rest/index.md)
- [Settings API Reference](api/rest/settings.md)
- [Single-Source Shortest Path (SSSP) API](api/shortest-path-api.md)
- [VisionFlow API Documentation](api/index.md)
- [VisionFlow MCP Integration Documentation](api/mcp/index.md)
- [VisionFlow WebSocket API Documentation](api/websocket/index.md)
- [WebSocket API Reference](api/websocket.md)
- [WebSocket Protocols](api/websocket-protocols.md)
- [dev-backend-api](reference/agents/development/backend/dev-backend-api.md)
- [docs-api-openapi](reference/agents/documentation/api-docs/docs-api-openapi.md)
