# VisionFlow API Endpoint Mapping

**Complete list of all API endpoints the frontend expects**

---

## Legend

- ✅ **Likely Implemented** - Standard endpoints
- ⚠️ **Verify Implementation** - May need review
- ❌ **Missing** - Causes 404 errors

---

## Settings API (`/api/settings/*`)

| Status | Method | Endpoint | Frontend File | Purpose |
|--------|--------|----------|---------------|---------|
| ⚠️ | `GET` | `/api/settings/path/{encodedPath}` | `settingsApi.ts:240` | Get single setting by path |
| ⚠️ | `PUT` | `/api/settings/path/{encodedPath}` | `settingsApi.ts:152,260` | Update single setting |
| ⚠️ | `POST` | `/api/settings/batch` | `settingsApi.ts:248` | Batch fetch settings |
| ⚠️ | `POST` | `/api/settings/reset` | `settingsApi.ts:287` | Reset to defaults |
| ⚠️ | `GET` | `/api/settings/physics/{graphName}` | `settingsApi.ts:301` | Get graph physics |
| ⚠️ | `PUT` | `/api/settings/physics/{graphName}` | `settingsApi.ts:316` | Update graph physics |
| ✅ | `GET` | `/api/settings/health` | `settingsApi.ts:340` | Health check |
| ⚠️ | `POST` | `/api/settings/cache/clear` | `settingsApi.ts:354` | Clear cache |

**Notes:**
- Path parameter must be URL-encoded (e.g., `system.debug.enabled` → `system%2Edebug%2Eenabled`)
- Graph names: `logseq`, `visionflow`, `default`
- Batch endpoint expects: `{ paths: string[] }`

---

## Analytics API (`/analytics/*`)

| Status | Method | Endpoint | Frontend File | Purpose |
|--------|--------|----------|---------------|---------|
| ✅ | `GET` | `/analytics/params` | `analyticsApi.ts:136` | Get analytics parameters |
| ✅ | `POST` | `/analytics/params` | `analyticsApi.ts:159` | Update analytics params |
| ✅ | `GET` | `/analytics/constraints` | `analyticsApi.ts:180` | Get constraints |
| ✅ | `POST` | `/analytics/constraints` | `analyticsApi.ts:200` | Update constraints |
| ✅ | `POST` | `/analytics/clustering/run` | `analyticsApi.ts:228,388` | Start clustering task |
| ✅ | `GET` | `/analytics/clustering/status?task_id=X` | `analyticsApi.ts:278` | Get task status |
| ✅ | `POST` | `/analytics/clustering/cancel?task_id=X` | `analyticsApi.ts:322` | Cancel task |
| ❌ | `POST` | `/analytics/clustering/focus` | `SemanticClusteringControls.tsx:179` | Focus on cluster |
| ✅ | `GET` | `/analytics/stats` | `analyticsApi.ts:340` | GPU performance stats |
| ✅ | `GET` | `/analytics/gpu-status` | `analyticsApi.ts:366` | GPU status |
| ✅ | `POST` | `/analytics/insights` | `analyticsApi.ts:253` | Semantic analysis |
| ✅ | `POST` | `/analytics/anomaly/toggle` | `analyticsApi.ts:410` | Configure anomaly detection |
| ✅ | `GET` | `/analytics/anomaly/current` | `analyticsApi.ts:427` | Get current anomalies |
| ❌ | `POST` | `/analytics/shortest-path` | `analyticsStore.ts:264` | Calculate shortest path |

---

## Graph API (`/graph/*`, `/api/graph/*`)

| Status | Method | Endpoint | Frontend File | Purpose |
|--------|--------|----------|---------------|---------|
| ⚠️ | `POST` | `/graph/export` | `exportApi.ts:76` | Export graph |
| ⚠️ | `POST` | `/graph/share` | `exportApi.ts:109` | Create shareable link |
| ⚠️ | `GET` | `/graph/shared/{shareId}` | `exportApi.ts:129` | Get shared graph |
| ⚠️ | `DELETE` | `/graph/shared/{shareId}` | `exportApi.ts:140` | Delete shared graph |
| ⚠️ | `POST` | `/graph/publish` | `exportApi.ts:157` | Publish graph |
| ⚠️ | `GET` | `/graph/shared` | `exportApi.ts:178` | List user shares |
| ⚠️ | `PUT` | `/graph/shared/{shareId}` | `exportApi.ts:193` | Update share settings |
| ✅ | `PUT` | `/graph/nodes/batch-update` | `batchUpdateApi.ts:29` | Batch node updates |
| ✅ | `POST` | `/graph/nodes/batch-create` | `batchUpdateApi.ts:58` | Batch node creation |
| ✅ | `DELETE` | `/graph/nodes/batch-delete` | `batchUpdateApi.ts:86` | Batch node deletion |
| ✅ | `PUT` | `/graph/edges/batch-update` | `batchUpdateApi.ts:117` | Batch edge updates |
| ❌ | `GET` | `/api/graph/auto-balance-notifications` | `AutoBalanceIndicator.tsx:25` | Auto-balance status |
| ⚠️ | `GET` | `/api/graph` | `OntologyModeToggle.tsx:60` | Get knowledge graph |

---

## Workspace API (`/workspace/*`)

| Status | Method | Endpoint | Frontend File | Purpose |
|--------|--------|----------|---------------|---------|
| ⚠️ | `GET` | `/workspace/list?page=&limit=&status=&type=` | `workspaceApi.ts:153` | List workspaces |
| ⚠️ | `POST` | `/workspace/create` | `workspaceApi.ts:181` | Create workspace |
| ⚠️ | `GET` | `/workspace/{id}` | `workspaceApi.ts:289` | Get workspace |
| ⚠️ | `PUT` | `/workspace/{id}` | `workspaceApi.ts:210` | Update workspace |
| ⚠️ | `DELETE` | `/workspace/{id}` | `workspaceApi.ts:230` | Delete workspace |
| ⚠️ | `POST` | `/workspace/{id}/favorite` | `workspaceApi.ts:249` | Toggle favorite |
| ⚠️ | `POST` | `/workspace/{id}/archive` | `workspaceApi.ts:269` | Archive workspace |
| ⚠️ | `PUT` | `/workspace/{id}/settings` | `workspaceApi.ts:308` | Update settings |
| ⚠️ | `GET` | `/workspace/{id}/members` | `workspaceApi.ts:328` | Get members |

---

## Ontology API (`/api/ontology/*`)

| Status | Method | Endpoint | Frontend File | Purpose |
|--------|--------|----------|---------------|---------|
| ❌ | `GET` | `/api/ontology/graph` | `OntologyModeToggle.tsx:60` | Get ontology graph |
| ❌ | `POST` | `/api/ontology/load` | `useOntologyStore.ts:131` | Load ontology |
| ❌ | `POST` | `/api/ontology/validate` | `useOntologyStore.ts:158` | Validate ontology |

**Impact:** Ontology mode toggle completely broken

---

## Bots/Agents API (`/api/bots/*`)

| Status | Method | Endpoint | Frontend File | Purpose |
|--------|--------|----------|---------------|---------|
| ❌ | `GET` | `/api/bots/status` | `AgentTelemetry.ts:191` | Get agent status |
| ❌ | `GET` | `/api/bots/data` | `AgentTelemetry.ts:192` | Get agent data |
| ❌ | `GET` | `/api/bots/agents` | `AgentNodesLayer.tsx:336` | Get agent list |

**Impact:** Agent visualization and telemetry broken

---

## Telemetry/Logging API (`/api/*`)

| Status | Method | Endpoint | Frontend File | Purpose |
|--------|--------|----------|---------------|---------|
| ❌ | `POST` | `/api/client-logs` | `remoteLogger.ts:30` | Remote client logs (Quest 3) |
| ❌ | `POST` | `/api/telemetry/errors` | `useErrorHandler.tsx:298` | Error telemetry |
| ❌ | `POST` | `/api/telemetry/upload` | `AgentTelemetry.ts:142` | Upload telemetry |
| ❌ | `POST` | `/api/errors/log` | `ErrorBoundary.tsx:93` | Log errors |

**Impact:** Remote debugging for Quest 3 completely broken

---

## Health/System API

| Status | Method | Endpoint | Frontend File | Purpose |
|--------|--------|----------|---------------|---------|
| ✅ | `GET` | `/health` | `UnifiedApiClient.ts:485` | Global health check |
| ⚠️ | `GET` | `/api/settings/health` | `settingsApi.ts:340` | Settings health check |

---

## WebSocket Endpoints

| Status | Protocol | Endpoint | Frontend File | Purpose |
|--------|----------|----------|---------------|---------|
| ✅ | `WS` | `/wss` | `WebSocketService.ts:92` | Main WebSocket |
| ⚠️ | `WS` | `/api/settings/ws` | `useSettingsWebSocket.ts:92` | Settings updates |
| ⚠️ | `WS` | `/ws/analytics` | `analyticsApi.ts:468` | Analytics task updates |
| ⚠️ | `WS` | `/speech` or similar | `VoiceWebSocketService.ts` | Voice/speech WebSocket |

---

## API Client Configuration

### UnifiedApiClient Defaults

```typescript
// From: services/api/UnifiedApiClient.ts
baseURL: '/api'                    // All paths prefixed with /api
timeout: 30000                     // 30 second timeout
retryAttempts: 3                   // Max retry count
retryDelay: 1000                   // Initial retry delay (exponential backoff)
```

### Vite Proxy (Development)

```typescript
// From: vite.config.ts
proxy: {
  '/api': {
    target: 'http://visionflow_container:4000',
    changeOrigin: true,
    secure: false
  },
  '/ws': {
    target: 'ws://visionflow_container:4000',
    ws: true
  },
  '/wss': {
    target: 'ws://visionflow_container:4000',
    ws: true
  }
}
```

---

## Request/Response Formats

### Standard API Response

```typescript
{
  success: boolean,
  data?: T,
  error?: string,
  message?: string
}
```

### Settings API Response (Path-based)

```typescript
// GET /api/settings/path/{path}
{
  value: any
}

// POST /api/settings/batch
{
  [path: string]: any
}
```

### Analytics Task Response

```typescript
{
  success: true,
  task_id: string,
  task?: {
    task_id: string,
    status: 'pending' | 'running' | 'completed' | 'failed',
    progress: number,
    result?: any,
    error?: string
  }
}
```

### Workspace Response

```typescript
{
  success: true,
  data: {
    workspaces: Workspace[],
    total: number,
    page: number,
    limit: number,
    hasMore: boolean
  }
}
```

---

## Query Parameters

### Workspace List

```
/workspace/list?page=1&limit=20&status=active&type=personal&favorite=true
```

### Analytics Task Status

```
/analytics/clustering/status?task_id={taskId}
```

### Analytics Cancel Task

```
/analytics/clustering/cancel?task_id={taskId}
```

---

## Request Bodies

### Settings Batch Request

```json
{
  "paths": [
    "system.debug.enabled",
    "visualisation.nodes.baseColor",
    "visualisation.graphs.logseq.physics.springK"
  ]
}
```

### Physics Update Request

```json
{
  "springK": 0.5,
  "repelK": 10.0,
  "attractionK": 0.01,
  "gravity": 0.1,
  "damping": 0.95,
  "temperature": 1.0,
  "warmupIterations": 100,
  "coolingRate": 0.99
}
```

### Clustering Request

```json
{
  "algorithm": "louvain",
  "resolution": 1.0,
  "iterations": 100,
  "gpu_accelerated": true,
  "real_time_updates": false
}
```

### Workspace Create Request

```json
{
  "name": "My Workspace",
  "description": "Workspace description",
  "type": "personal",
  "settings": {
    "autoSave": true,
    "syncEnabled": true,
    "collaborationEnabled": false
  }
}
```

---

## WebSocket Message Formats

### Text Messages (JSON)

```json
{
  "type": "setting_updated",
  "data": {
    "path": "system.debug.enabled",
    "value": true
  },
  "timestamp": 1234567890
}
```

### Binary Messages (Position Updates)

**Header (6 bytes):**
```
[type: u8][graphTypeFlag: u8][payloadSize: u32]
```

**Position Data (28 bytes per node):**
```
[nodeId: u32][x: f32][y: f32][z: f32][vx: f32][vy: f32][vz: f32]
```

**Graph Type Flags:**
- `0x01` = knowledge_graph
- `0x02` = ontology

**Agent Node Flag:**
- `nodeId | 0x80000000` = Agent/bot node

---

## Error Response Format

### Standard Error

```json
{
  "success": false,
  "error": "Error message",
  "statusCode": 400,
  "data": {
    "field": "validation details"
  }
}
```

### WebSocket Error Frame

```json
{
  "type": "error",
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid setting path",
    "category": "validation",
    "retryable": false,
    "affectedPaths": ["system.invalid.path"],
    "timestamp": 1234567890
  }
}
```

---

## Authentication

### Header Format

```
Authorization: Bearer {jwt_token}
```

### Public Endpoints

To bypass authentication:

```typescript
unifiedApiClient.get('/public-endpoint', { skipAuth: true });
```

### Auth Interceptor

- Automatically adds token from `nostrAuth.getToken()`
- Handles 401 responses by clearing auth state
- Redirects to login on authentication failure

---

## Rate Limiting

**Expected Headers:**
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1234567890
```

**429 Response:**
```json
{
  "success": false,
  "error": "Rate limit exceeded",
  "retryAfter": 60
}
```

---

## CORS Requirements

**Required Headers:**
```
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS
Access-Control-Allow-Headers: Content-Type, Authorization
Access-Control-Max-Age: 86400
```

**Development:**
- Vite proxy handles CORS automatically

**Production:**
- Nginx must set CORS headers

---

## File Uploads

**Not currently implemented in frontend**

Potential future endpoints:
```
POST /api/upload/graph
POST /api/upload/ontology
POST /api/upload/settings
```

---

## Pagination

**Query Parameters:**
```
?page=1&limit=20
```

**Response:**
```json
{
  "success": true,
  "data": {
    "items": [...],
    "total": 100,
    "page": 1,
    "limit": 20,
    "hasMore": true
  }
}
```

---

## Search/Filter

**Query Parameters:**
```
?search=term&status=active&type=personal&favorite=true
```

**Multiple Values:**
```
?types=personal&types=team&types=public
```

---

## Sorting

**Query Parameters:**
```
?sortBy=createdAt&sortOrder=desc
```

**Valid Sort Fields:**
- `createdAt`
- `updatedAt`
- `lastAccessed`
- `name`
- `memberCount`

---

## Cache Control

**Headers:**
```
Cache-Control: no-cache, no-store, must-revalidate
Pragma: no-cache
Expires: 0
```

**Settings Cache:**
```
POST /api/settings/cache/clear
```

---

## Content Types

**Request:**
```
Content-Type: application/json
```

**Response:**
```
Content-Type: application/json
```

**WebSocket:**
```
- Text: JSON string
- Binary: ArrayBuffer
```

---

## HTTP Status Codes

| Code | Meaning | Frontend Handling |
|------|---------|-------------------|
| 200 | OK | Success |
| 201 | Created | Success (resource created) |
| 204 | No Content | Success (no body) |
| 400 | Bad Request | Show error, no retry |
| 401 | Unauthorized | Clear auth, redirect to login |
| 403 | Forbidden | Show error, no retry |
| 404 | Not Found | Show error, no retry |
| 409 | Conflict | Show error, allow retry |
| 429 | Rate Limited | Retry after delay |
| 500 | Server Error | Retry with exponential backoff |
| 502 | Bad Gateway | Retry with exponential backoff |
| 503 | Service Unavailable | Retry with exponential backoff |
| 504 | Gateway Timeout | Retry with exponential backoff |

---

## Retry Logic

**Conditions for Retry:**
- Network errors (status 0)
- 5xx errors (500-599)
- NOT for: 401, 403, 404, 400

**Retry Delays:**
```
Attempt 1: 1000ms
Attempt 2: 2000ms
Attempt 3: 4000ms
Max Attempts: 3
```

---

## Testing Endpoints

### Health Check

```bash
curl http://visionflow_container:4000/health
```

### Settings API

```bash
# Get setting
curl http://visionflow_container:4000/api/settings/path/system.debug.enabled

# Update setting
curl -X PUT http://visionflow_container:4000/api/settings/path/system.debug.enabled \
  -H "Content-Type: application/json" \
  -d '{"value": true}'

# Batch fetch
curl -X POST http://visionflow_container:4000/api/settings/batch \
  -H "Content-Type: application/json" \
  -d '{"paths": ["system.debug.enabled", "visualisation.nodes.baseColor"]}'
```

### WebSocket Test

```javascript
const ws = new WebSocket('ws://localhost:4000/wss');
ws.onopen = () => console.log('Connected');
ws.onmessage = (e) => console.log('Message:', e.data);
```

---

## Priority Order for Implementation

### Critical (Blocking Features)

1. `POST /api/client-logs` - Remote logging
2. `GET /api/bots/status` - Agent status
3. `GET /api/bots/data` - Agent data
4. `GET /api/settings/path/{path}` - Path-based settings
5. `PUT /api/settings/path/{path}` - Path-based updates

### High (Core Features)

6. `POST /api/settings/batch` - Batch settings
7. `GET /api/settings/physics/{graphName}` - Physics config
8. `GET /api/ontology/graph` - Ontology mode
9. `WS /wss` - Main WebSocket
10. `GET /analytics/stats` - Analytics stats

### Medium (Enhanced Features)

11. `/workspace/*` endpoints - Workspace management
12. `/graph/export` - Export functionality
13. `/graph/share` - Sharing
14. `POST /analytics/clustering/run` - Clustering

### Low (Nice to Have)

15. `/analytics/shortest-path` - Path analysis
16. `/analytics/clustering/focus` - Cluster focus
17. `/graph/auto-balance-notifications` - Auto-balance

---

## End of Mapping

For detailed architecture, see: `VISIONFLOW_FRONTEND_API_ANALYSIS.md`
For 404 errors, see: `FRONTEND_404_ERRORS_SUMMARY.md`
