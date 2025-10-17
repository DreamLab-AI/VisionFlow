# Settings API Reference

## Overview

The Settings API provides RESTful endpoints for managing global and user-specific configuration. All endpoints support JSON payloads with `camelCase` field names for frontend compatibility, automatically converted to `snake_case` for backend processing.

## Base URL

```
http://localhost:4000/api
```

## Authentication

All settings modification endpoints require authentication via Nostr public key. Include the public key in request headers:

```
Authorization: Nostr <pubkey>
```

Power user permissions are required for most write operations.

## Endpoints

### 1. Get Global Settings

Retrieve the current global default settings.

**Endpoint:** `GET /api/settings`

**Authentication:** Not required for read

**Response:** `200 OK`

```json
{
  "visualisation": {
    "rendering": {
      "ambientLightIntensity": 0.5,
      "backgroundColor": "#000000",
      "directionalLightIntensity": 0.4,
      "enableAmbientOcclusion": false,
      "enableAntialiasing": true,
      "enableShadows": false,
      "environmentIntensity": 0.3,
      "shadowMapSize": "2048",
      "shadowBias": 0.0001,
      "context": "desktop"
    },
    "animations": {
      "enableMotionBlur": true,
      "enableNodeAnimations": true,
      "motionBlurStrength": 0.1,
      "selectionWaveEnabled": false,
      "pulseEnabled": true,
      "pulseSpeed": 1.2,
      "pulseStrength": 0.8,
      "waveSpeed": 0.5
    },
    "glow": {
      "enabled": false,
      "intensity": 1.2,
      "radius": 1.2,
      "threshold": 0.39,
      "diffuseStrength": 1.5,
      "atmosphericDensity": 0.8,
      "volumetricIntensity": 1.2,
      "baseColor": "#00ffff",
      "emissionColor": "#00e5ff",
      "opacity": 0.6,
      "pulseSpeed": 1.2,
      "flowSpeed": 0.5,
      "nodeGlowStrength": 0.7,
      "edgeGlowStrength": 0.5,
      "environmentGlowStrength": 0.96
    },
    "graphs": {
      "logseq": {
        "nodes": { /* ... */ },
        "edges": { /* ... */ },
        "labels": { /* ... */ },
        "physics": { /* ... */ }
      },
      "visionflow": { /* ... */ }
    }
  },
  "system": {
    "network": {
      "bindAddress": "0.0.0.0",
      "port": 4000,
      "domain": "visionflow.info",
      "enableTls": false,
      "enableHttp2": false
    },
    "websocket": {
      "updateRate": 60,
      "maxConnections": 100,
      "heartbeatInterval": 10000
    }
  },
  "xr": { /* ... */ },
  "auth": { /* ... */ }
}
```

**Error Responses:**

- `500 Internal Server Error` - Database error

---

### 2. Update Global Settings

Update global default settings (requires power user permissions).

**Endpoint:** `PUT /api/settings`

**Authentication:** Required (Power User)

**Headers:**
```
Authorization: Nostr <pubkey>
Content-Type: application/json
```

**Request Body:** (Partial updates supported)

```json
{
  "visualisation": {
    "rendering": {
      "ambientLightIntensity": 0.7
    }
  }
}
```

**Response:** `200 OK`

```json
{
  "message": "Settings updated successfully",
  "updatedFields": [
    "visualisation.rendering.ambientLightIntensity"
  ]
}
```

**Error Responses:**

- `400 Bad Request` - Validation error
```json
{
  "error": "ValidationError",
  "field": "visualisation.rendering.ambientLightIntensity",
  "message": "Value must be between 0.0 and 10.0",
  "receivedValue": 15.0
}
```

- `401 Unauthorized` - Missing or invalid authentication
- `403 Forbidden` - User lacks power user permissions
```json
{
  "error": "PermissionDenied",
  "message": "User does not have power user permissions"
}
```

- `500 Internal Server Error` - Database error

---

### 3. Get User Settings

Retrieve settings for a specific user (merged with global defaults).

**Endpoint:** `GET /api/settings/user/:pubkey`

**Authentication:** Required (self or power user)

**Path Parameters:**
- `pubkey` - User's Nostr public key (hex format)

**Response:** `200 OK`

Returns the effective settings for the user, with user-specific overrides merged over global defaults.

```json
{
  "visualisation": {
    "rendering": {
      "ambientLightIntensity": 0.8  // User override
    }
  },
  // ... rest of settings from global defaults
}
```

**Error Responses:**

- `401 Unauthorized` - Missing or invalid authentication
- `403 Forbidden` - Cannot access other user's settings without power user permission
- `404 Not Found` - User not found
- `500 Internal Server Error` - Database error

---

### 4. Update User Settings

Update settings for a specific user (creates overrides).

**Endpoint:** `PUT /api/settings/user/:pubkey`

**Authentication:** Required (self or power user)

**Headers:**
```
Authorization: Nostr <pubkey>
Content-Type: application/json
```

**Path Parameters:**
- `pubkey` - User's Nostr public key (hex format)

**Request Body:**

```json
{
  "visualisation": {
    "rendering": {
      "ambientLightIntensity": 0.8
    },
    "graphs": {
      "logseq": {
        "physics": {
          "springK": 5.0,
          "repelK": 15.0
        }
      }
    }
  }
}
```

**Response:** `200 OK`

```json
{
  "message": "User settings updated successfully",
  "userId": "abc123...",
  "updatedFields": [
    "visualisation.rendering.ambientLightIntensity",
    "visualisation.graphs.logseq.physics.springK",
    "visualisation.graphs.logseq.physics.repelK"
  ]
}
```

**Error Responses:**

- `400 Bad Request` - Validation error
- `401 Unauthorized` - Missing or invalid authentication
- `403 Forbidden` - Cannot modify other user's settings
- `404 Not Found` - User not found
- `500 Internal Server Error` - Database error

---

### 5. Delete User Setting Override

Remove a user-specific override, reverting to global default.

**Endpoint:** `DELETE /api/settings/user/:pubkey/path/:path`

**Authentication:** Required (self or power user)

**Path Parameters:**
- `pubkey` - User's Nostr public key (hex format)
- `path` - Dot-separated setting path (e.g., `visualisation.rendering.ambientLightIntensity`)

**Response:** `200 OK`

```json
{
  "message": "User setting override removed",
  "path": "visualisation.rendering.ambientLightIntensity"
}
```

**Error Responses:**

- `401 Unauthorized` - Missing or invalid authentication
- `403 Forbidden` - Cannot modify other user's settings
- `404 Not Found` - Setting override not found
- `500 Internal Server Error` - Database error

---

### 6. Validate Settings

Validate settings without persisting (dry-run).

**Endpoint:** `POST /api/settings/validate`

**Authentication:** Not required

**Headers:**
```
Content-Type: application/json
```

**Request Body:**

```json
{
  "visualisation": {
    "rendering": {
      "ambientLightIntensity": 15.0  // Invalid value
    }
  }
}
```

**Response:** `200 OK` (validation passed)

```json
{
  "valid": true,
  "message": "Settings validation passed"
}
```

**Response:** `400 Bad Request` (validation failed)

```json
{
  "valid": false,
  "errors": [
    {
      "field": "visualisation.rendering.ambientLightIntensity",
      "message": "Value must be between 0.0 and 10.0",
      "receivedValue": 15.0,
      "constraint": "range(0.0, 10.0)"
    }
  ]
}
```

---

### 7. Get Setting by Path

Retrieve a specific setting value by path.

**Endpoint:** `GET /api/settings/path/:path`

**Authentication:** Not required

**Path Parameters:**
- `path` - Dot-separated setting path (e.g., `visualisation.rendering.ambientLightIntensity`)

**Query Parameters:**
- `userId` (optional) - User pubkey for user-specific value

**Response:** `200 OK`

```json
{
  "path": "visualisation.rendering.ambientLightIntensity",
  "value": 0.5,
  "source": "global"  // or "user" if override exists
}
```

**Error Responses:**

- `404 Not Found` - Setting path not found
- `500 Internal Server Error` - Database error

---

### 8. Update Setting by Path

Update a specific setting value by path.

**Endpoint:** `PUT /api/settings/path/:path`

**Authentication:** Required (Power User)

**Path Parameters:**
- `path` - Dot-separated setting path

**Query Parameters:**
- `userId` (optional) - User pubkey for user-specific update

**Request Body:**

```json
{
  "value": 0.7
}
```

**Response:** `200 OK`

```json
{
  "message": "Setting updated successfully",
  "path": "visualisation.rendering.ambientLightIntensity",
  "oldValue": 0.5,
  "newValue": 0.7
}
```

**Error Responses:**

- `400 Bad Request` - Validation error
- `401 Unauthorized` - Missing or invalid authentication
- `403 Forbidden` - Insufficient permissions
- `404 Not Found` - Setting path not found
- `500 Internal Server Error` - Database error

---

### 9. Get Audit Log

Retrieve settings change history.

**Endpoint:** `GET /api/settings/audit`

**Authentication:** Required (Power User)

**Query Parameters:**
- `userId` (optional) - Filter by user
- `startTime` (optional) - ISO 8601 timestamp
- `endTime` (optional) - ISO 8601 timestamp
- `limit` (optional, default: 100) - Max results
- `offset` (optional, default: 0) - Pagination offset

**Response:** `200 OK`

```json
{
  "total": 523,
  "limit": 100,
  "offset": 0,
  "entries": [
    {
      "id": 1,
      "userId": "abc123...",
      "timestamp": "2025-10-17T10:30:00Z",
      "settingKey": "visualisation.rendering.ambientLightIntensity",
      "oldValue": "0.5",
      "newValue": "0.7",
      "reason": "User preference update"
    },
    // ... more entries
  ]
}
```

**Error Responses:**

- `401 Unauthorized` - Missing or invalid authentication
- `403 Forbidden` - Insufficient permissions
- `500 Internal Server Error` - Database error

---

## WebSocket Settings Subscription

Subscribe to real-time settings updates via WebSocket.

**Endpoint:** `ws://localhost:4000/ws`

**Protocol:** JSON messages

### Subscribe to Settings Changes

**Client → Server:**

```json
{
  "type": "settings:subscribe",
  "userId": "abc123..."  // Optional, for user-specific updates
}
```

**Server → Client (on update):**

```json
{
  "type": "settings:update",
  "timestamp": "2025-10-17T10:30:00Z",
  "path": "visualisation.rendering.ambientLightIntensity",
  "value": 0.7,
  "userId": null  // null for global, pubkey for user-specific
}
```

### Unsubscribe

**Client → Server:**

```json
{
  "type": "settings:unsubscribe"
}
```

---

## Field Name Conventions

### API (Frontend)
Use `camelCase` for all field names:

```json
{
  "ambientLightIntensity": 0.5,
  "enableShadows": true,
  "springK": 5.0
}
```

### Database (Backend)
Stored as `snake_case`:

```sql
SELECT ambient_light_intensity, enable_shadows, spring_k FROM settings;
```

The API layer automatically converts between formats. Clients should always use `camelCase`.

---

## Rate Limiting

All endpoints are rate-limited to prevent abuse:

- **Read operations:** 1000 requests/10 minutes per IP
- **Write operations:** 100 requests/10 minutes per user
- **WebSocket connections:** 10 connections per IP

**Rate Limit Headers:**

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 987
X-RateLimit-Reset: 1634567890
```

**Rate Limit Exceeded:** `429 Too Many Requests`

```json
{
  "error": "RateLimitExceeded",
  "message": "Rate limit exceeded. Try again in 123 seconds.",
  "retryAfter": 123
}
```

---

## Examples

### Example 1: Update Physics Parameters

```bash
curl -X PUT http://localhost:4000/api/settings \
  -H "Authorization: Nostr abc123..." \
  -H "Content-Type: application/json" \
  -d '{
    "visualisation": {
      "graphs": {
        "logseq": {
          "physics": {
            "springK": 5.0,
            "repelK": 15.0,
            "damping": 0.7
          }
        }
      }
    }
  }'
```

### Example 2: Get User's Effective Settings

```bash
curl -X GET http://localhost:4000/api/settings/user/abc123... \
  -H "Authorization: Nostr abc123..."
```

### Example 3: Validate Settings Before Saving

```bash
curl -X POST http://localhost:4000/api/settings/validate \
  -H "Content-Type: application/json" \
  -d '{
    "visualisation": {
      "rendering": {
        "ambientLightIntensity": 0.7
      }
    }
  }'
```

### Example 4: WebSocket Subscription (JavaScript)

```javascript
const ws = new WebSocket('ws://localhost:4000/ws');

ws.onopen = () => {
  // Subscribe to settings updates
  ws.send(JSON.stringify({
    type: 'settings:subscribe',
    userId: 'abc123...'  // Optional
  }));
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);

  if (message.type === 'settings:update') {
    console.log('Setting updated:', message.path, '=', message.value);
    // Update UI accordingly
  }
};
```

---

## Error Codes Reference

| Code | Description |
|------|-------------|
| 200 | OK - Request succeeded |
| 400 | Bad Request - Validation error |
| 401 | Unauthorized - Missing/invalid authentication |
| 403 | Forbidden - Insufficient permissions |
| 404 | Not Found - Resource not found |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error - Database/server error |

---

## Related Documentation

- [Settings System Architecture](./settings-system.md)
- [Validation Rules](./settings-validation.md)
- [Database Schema](./settings-schema.md)
- [User Permissions](./user-permissions.md)
- [Migration Guide](./settings-migration-guide.md)
