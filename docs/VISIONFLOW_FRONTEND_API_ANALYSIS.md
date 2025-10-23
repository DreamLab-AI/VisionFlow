# VisionFlow Frontend API Analysis Report

**Generated:** 2025-10-23
**Purpose:** Complete mapping of frontend API architecture, endpoints, and error patterns

---

## Executive Summary

The VisionFlow React/TypeScript frontend uses a **unified API client architecture** with centralized HTTP handling, retry logic, authentication interceptors, and comprehensive error management. All API calls route through `/api/*` endpoints proxied to the backend at `http://visionflow_container:4000`.

### Key Findings

✅ **Well-Architected**: Clean separation of concerns with `UnifiedApiClient`, domain-specific API modules, and Zustand state management
⚠️ **404 Issues**: Several hardcoded endpoints not matching backend routes
🔧 **WebSocket Integration**: Dual WebSocket system (main WSS + voice WSS) with binary protocol support
📊 **Settings System**: Path-based REST API with debouncing, batching, and graph-specific physics separation

---

## 1. API Client Architecture

### 1.1 UnifiedApiClient (Core HTTP Layer)

**Location:** `/client/src/services/api/UnifiedApiClient.ts`

**Configuration:**
- **Base URL:** `/api` (default, configurable)
- **Default Timeout:** 30 seconds
- **Retry Logic:** Exponential backoff, max 3 attempts
- **Retry Conditions:** Network errors (status 0) and 5xx errors only (excludes 401/403)

**Features:**
```typescript
class UnifiedApiClient {
  // Request methods
  get<T>(url, config?)
  post<T>(url, data?, config?)
  put<T>(url, data?, config?)
  patch<T>(url, data?, config?)
  delete<T>(url, config?)

  // Convenience methods (auto-extract data)
  getData<T>(url, config?)
  postData<T>(url, data?, config?)
  putData<T>(url, data?, config?)

  // Auth management
  setAuthToken(token: string)
  removeAuthToken()

  // Interceptors
  setInterceptors(config: InterceptorConfig)

  // Health check
  healthCheck(): Promise<boolean>  // GET /health
}
```

**Singleton Instance:**
```typescript
// Exported from services/api/index.ts
import { unifiedApiClient } from '@/services/api/UnifiedApiClient';
```

---

## 2. Domain-Specific API Modules

### 2.1 Settings API (`/api/settings/*`)

**Location:** `/client/src/api/settingsApi.ts`

**Key Endpoints:**

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/api/settings/path/{encodedPath}` | Get single setting by dot-notation path |
| `PUT` | `/api/settings/path/{encodedPath}` | Update single setting (with debouncing) |
| `POST` | `/api/settings/batch` | Batch fetch multiple paths |
| `POST` | `/api/settings/reset` | Reset to defaults |
| `GET` | `/api/settings/physics/{graphName}` | Get physics for logseq/visionflow/default |
| `PUT` | `/api/settings/physics/{graphName}` | Update physics (with validation) |
| `GET` | `/api/settings/health` | Health check + cache status |
| `POST` | `/api/settings/cache/clear` | Clear backend cache |

**Features:**
- **Debouncing:** 50ms delay for UI responsiveness
- **Priority Queue:** Critical (physics) → High (visual) → Normal → Low
- **Batch Operations:** Up to 25 settings per batch
- **Graph Separation:** Validates `logseq` vs `visionflow` physics independence

**Path Format Examples:**
```typescript
'system.debug.enabled'
'visualisation.nodes.baseColor'
'visualisation.graphs.logseq.physics.springK'
```

---

### 2.2 WebSocket Service (`/wss`)

**Location:** `/client/src/services/WebSocketService.ts`

**Connection URL:**
```typescript
// Development: ws://localhost:3001/wss
// Production: wss://{window.location.host}/wss
```

**Message Types:**

1. **Text Messages (JSON)**
   ```typescript
   interface WebSocketMessage {
     type: string;
     data?: any;
     error?: WebSocketErrorFrame;
   }
   ```

2. **Binary Messages (Position Updates)**
   - **Header:** Message type, graph type flag, payload size
   - **Types:** `GRAPH_UPDATE`, `VOICE_DATA`, `POSITION_UPDATE`, `AGENT_POSITIONS`
   - **Routing:** Filters by current mode (knowledge_graph vs ontology)

**Features:**
- Automatic reconnection (exponential backoff, max 30s delay)
- Heartbeat (30s interval, 10s timeout)
- Message queue for offline periods (max 100 messages)
- Binary protocol with position batching
- Custom backend URL support via settings

**Key Methods:**
```typescript
webSocketService.sendMessage(type, data)
webSocketService.sendNodePositionUpdates(updates)
webSocketService.flushPositionUpdates()
webSocketService.on(eventName, handler)
```

---

### 2.3 Analytics API (`/analytics/*`)

**Location:** `/client/src/api/analyticsApi.ts`

**Endpoints:**

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/analytics/params` | Get visual analytics parameters |
| `POST` | `/analytics/params` | Update analytics config |
| `GET` | `/analytics/constraints` | Get constraint sets |
| `POST` | `/analytics/constraints` | Update constraints |
| `POST` | `/analytics/clustering/run` | Start clustering task |
| `GET` | `/analytics/clustering/status?task_id=X` | Poll task status |
| `POST` | `/analytics/clustering/cancel?task_id=X` | Cancel task |
| `GET` | `/analytics/stats` | GPU performance stats |
| `GET` | `/analytics/gpu-status` | Comprehensive GPU status |
| `POST` | `/analytics/insights` | Semantic analysis |
| `POST` | `/analytics/anomaly/toggle` | Configure anomaly detection |
| `GET` | `/analytics/anomaly/current` | Get current anomalies |

**WebSocket Support:**
- Connects to `ws://{host}/ws/analytics`
- Real-time task progress updates
- Auto-reconnection with exponential backoff

**Usage Pattern:**
```typescript
const analyticsAPI = new AnalyticsAPI();
const taskId = await analyticsAPI.runClustering({ algorithm: 'louvain' });
const unsubscribe = analyticsAPI.subscribeToTask(taskId, (task) => {
  console.log(task.progress);
});
```

---

### 2.4 Workspace API (`/workspace/*`)

**Location:** `/client/src/api/workspaceApi.ts`

**Endpoints:**

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/workspace/list?page=&limit=&status=&type=` | List workspaces (paginated) |
| `POST` | `/workspace/create` | Create new workspace |
| `GET` | `/workspace/{id}` | Get workspace details |
| `PUT` | `/workspace/{id}` | Update workspace |
| `DELETE` | `/workspace/{id}` | Delete workspace |
| `POST` | `/workspace/{id}/favorite` | Toggle favorite |
| `POST` | `/workspace/{id}/archive` | Archive/unarchive |
| `PUT` | `/workspace/{id}/settings` | Update workspace settings |
| `GET` | `/workspace/{id}/members` | Get team members |

**Features:**
- Client-side validation (name length < 100 chars)
- Custom error class with status codes
- Automatic date transformation
- Optimistic updates support

---

### 2.5 Export API (`/graph/*`)

**Location:** `/client/src/api/exportApi.ts`

**Endpoints:**

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/graph/export` | Export graph in format |
| `POST` | `/graph/share` | Create shareable link |
| `GET` | `/graph/shared/{shareId}` | Get shared graph |
| `DELETE` | `/graph/shared/{shareId}` | Delete share |
| `POST` | `/graph/publish` | Publish to public repo |
| `GET` | `/graph/shared` | List user's shares |
| `PUT` | `/graph/shared/{shareId}` | Update share settings |

**Supported Formats:**
`json`, `csv`, `graphml`, `gexf`, `svg`, `png`, `pdf`, `xlsx`, `dot`, `adjlist`

---

### 2.6 Batch Update API

**Location:** `/client/src/api/batchUpdateApi.ts`

**Endpoints:**

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `PUT` | `/graph/nodes/batch-update` | Bulk node updates |
| `POST` | `/graph/nodes/batch-create` | Bulk node creation |
| `DELETE` | `/graph/nodes/batch-delete` | Bulk node deletion |
| `PUT` | `/graph/edges/batch-update` | Bulk edge updates |

---

### 2.7 Optimization API

**Location:** `/client/src/api/optimizationApi.ts`

**Note:** Base URL is empty (`API_BASE = ''`), meaning endpoints are relative to `/api/`

---

### 2.8 Remote Logger

**Location:** `/client/src/services/remoteLogger.ts`

**Endpoint:**
```typescript
// POST /api/client-logs
// Default: http://visionflow_container:4000/api/client-logs
// Configurable via VITE_API_URL
```

**Features:**
- Intercepts all `console.*` methods
- Buffers logs (max 50, flush every 1s)
- Sends XR environment info
- Uses `sendBeacon` for page unload

---

## 3. State Management (Zustand)

### 3.1 Settings Store

**Location:** `/client/src/store/settingsStore.ts`

**Architecture:**
- **Lazy Loading:** Only loads essential paths on init
- **Partial State:** `DeepPartial<Settings>` - only holds loaded paths
- **Essential Paths:** Debug, WebSocket, auth, XR, physics for both graphs

**Key Methods:**
```typescript
useSettingsStore.getState().get<T>(path)            // Sync getter
useSettingsStore.getState().set<T>(path, value)    // Sync setter + REST
useSettingsStore.getState().ensureLoaded(paths)    // Async load
useSettingsStore.getState().loadSection(section)   // Load group
useSettingsStore.getState().updatePhysics(graph, params)
useSettingsStore.getState().flushPendingUpdates()
```

**Sections:**
`physics`, `rendering`, `xr`, `glow`, `hologram`, `nodes`, `edges`, `labels`

---

## 4. Missing/404 Endpoints Analysis

### 4.1 Likely 404 Errors

Based on frontend code, these endpoints may be missing from backend:

1. **Ontology API:**
   ```
   GET  /api/ontology/graph
   POST /api/ontology/load
   POST /api/ontology/validate
   ```

2. **Bots/Agents API:**
   ```
   GET /api/bots/status
   GET /api/bots/data
   GET /api/bots/agents
   ```

3. **Telemetry:**
   ```
   POST /api/telemetry/errors
   POST /api/telemetry/upload
   ```

4. **Error Logging:**
   ```
   POST /api/errors/log
   ```

5. **Auto-Balance:**
   ```
   GET /api/graph/auto-balance-notifications
   ```

6. **Shortest Path:**
   ```
   POST /api/analytics/shortest-path
   ```

7. **Clustering Focus:**
   ```
   POST /api/analytics/clustering/focus
   ```

---

## 5. GraphDataManager

**Location:** `/client/src/features/graph/managers/graphDataManager.ts`

**Purpose:** Manages graph data, handles WebSocket binary updates, coordinates with Three.js rendering

**Key Responsibilities:**
- Parse binary node position updates
- Update Three.js object positions
- Handle graph type switching (logseq vs visionflow vs ontology)
- Batch position updates for performance
- Detect and route agent/bot nodes

**Integration:**
```typescript
import { graphDataManager } from '@/features/graph/managers/graphDataManager';

// Used by WebSocketService for position updates
await graphDataManager.updateNodePositions(binaryData);
```

---

## 6. Authentication Flow

**Location:** `/client/src/services/api/authInterceptor.ts`

**Flow:**
1. **Initialization** (`main.tsx`):
   ```typescript
   initializeAuthInterceptor(unifiedApiClient);
   setupAuthStateListener();
   ```

2. **Interceptor Adds Token:**
   - Checks `nostrAuth.isAuthenticated()`
   - Adds `Authorization: Bearer {token}` header
   - Skips for public endpoints (`skipAuth: true`)

3. **401 Handling:**
   - Redirects to login
   - Clears auth state
   - Reloads settings

---

## 7. Environment Configuration

### 7.1 Development (Vite Dev Server)

**Port:** 5173 (proxied through Nginx on 3001)

**Proxy Rules (vite.config.ts):**
```typescript
proxy: {
  '/api': {
    target: process.env.VITE_API_URL || 'http://visionflow_container:4000',
    changeOrigin: true,
    secure: false
  },
  '/ws': {
    target: process.env.VITE_WS_URL || 'ws://visionflow_container:4000',
    ws: true
  },
  '/wss': {
    target: process.env.VITE_WS_URL || 'ws://visionflow_container:4000',
    ws: true
  }
}
```

### 7.2 Production Build

- **Output:** `/client/dist`
- **Routing:** Nginx serves static files, proxies `/api/*` to backend
- **WebSocket:** Nginx upgrades connection for `/wss`

---

## 8. Error Handling

### 8.1 ApiError Interface

```typescript
interface ApiError extends Error {
  status?: number;
  statusText?: string;
  data?: any;
  isApiError: true;
}
```

### 8.2 WebSocket Error Frames

```typescript
interface WebSocketErrorFrame {
  code: string;
  message: string;
  category: 'validation' | 'server' | 'protocol' | 'auth' | 'rate_limit';
  retryable: boolean;
  retryAfter?: number;
  affectedPaths?: string[];
  timestamp: number;
}
```

### 8.3 Error Boundaries

- `ErrorBoundary.tsx` - Catches React errors, reports to `/api/errors/log`
- `useErrorHandler.tsx` - Hook for async error handling

---

## 9. Critical Frontend-Backend Contract Points

### 9.1 Settings API

**Frontend Expects:**
- Path-based REST API (not bulk GET/PUT)
- Response format: `{ value: any }` for GET
- Graph-specific physics endpoints
- Batch endpoint with `{ paths: string[] }` payload

### 9.2 WebSocket Binary Protocol

**Frontend Expects:**
- Header format: `[type: u8][graphTypeFlag: u8][payloadSize: u32]`
- Graph type filtering (0x01 = knowledge_graph, 0x02 = ontology)
- Position data: `[nodeId: u32][x: f32][y: f32][z: f32][vx: f32][vy: f32][vz: f32]`
- Agent nodes: nodeId with high bit set (0x80000000)

### 9.3 Analytics Tasks

**Frontend Expects:**
- Task-based async API with polling
- Task structure:
  ```typescript
  {
    task_id: string,
    status: 'pending' | 'running' | 'completed' | 'failed',
    progress: number,
    result?: any,
    error?: string
  }
  ```

---

## 10. Performance Optimizations

1. **Settings Debouncing:** 50ms delay, priority queue for physics
2. **WebSocket Batching:** Position updates batched before sending
3. **Lazy Loading:** Settings loaded on-demand by section
4. **Request Retry:** Exponential backoff for transient failures
5. **Memory Optimization:** Partial settings store (only loaded paths)

---

## 11. Security Considerations

1. **CORS:** Enabled in dev, controlled by Nginx in prod
2. **COOP/COEP:** Headers set for SharedArrayBuffer support
3. **Auth Tokens:** Stored in localStorage, auto-attached to requests
4. **Public Endpoints:** Can bypass auth with `skipAuth: true`
5. **Input Validation:** Client-side checks before API calls

---

## 12. Recommendations for Backend Team

### High Priority

1. **Implement Missing Endpoints:**
   - `/api/bots/status`, `/api/bots/data`, `/api/bots/agents`
   - `/api/ontology/graph`, `/api/ontology/load`, `/api/ontology/validate`
   - `/api/telemetry/errors`, `/api/telemetry/upload`

2. **Settings API:**
   - Ensure path-based routes (`/api/settings/path/{encodedPath}`)
   - Implement batch endpoint (`POST /api/settings/batch`)
   - Support graph-specific physics (`/api/settings/physics/{graphName}`)

3. **WebSocket:**
   - Maintain binary protocol header format
   - Support graph type filtering (knowledge_graph vs ontology)
   - Handle heartbeat (ping/pong)

### Medium Priority

4. **Analytics:**
   - Implement task-based API with polling
   - Support WebSocket subscriptions at `/ws/analytics`

5. **Error Responses:**
   - Use consistent format: `{ success: boolean, data?: any, error?: string }`
   - Include proper HTTP status codes

6. **Health Checks:**
   - `/health` endpoint for load balancers
   - `/api/settings/health` for cache status

---

## 13. Files Reference

**Core API:**
- `/client/src/services/api/UnifiedApiClient.ts` - HTTP client
- `/client/src/services/api/authInterceptor.ts` - Auth handling
- `/client/src/services/WebSocketService.ts` - WebSocket client
- `/client/src/services/BinaryWebSocketProtocol.ts` - Binary message parsing

**Domain APIs:**
- `/client/src/api/settingsApi.ts` - Settings CRUD
- `/client/src/api/analyticsApi.ts` - GPU analytics
- `/client/src/api/workspaceApi.ts` - Workspace management
- `/client/src/api/exportApi.ts` - Graph export/share
- `/client/src/api/batchUpdateApi.ts` - Bulk operations
- `/client/src/api/optimizationApi.ts` - Graph optimization

**State Management:**
- `/client/src/store/settingsStore.ts` - Settings Zustand store
- `/client/src/store/autoSaveManager.ts` - Auto-save coordination
- `/client/src/store/settingsRetryManager.ts` - Retry logic

**Utilities:**
- `/client/src/services/remoteLogger.ts` - Remote logging
- `/client/src/utils/loggerConfig.ts` - Logger setup
- `/client/src/utils/debugConfig.ts` - Debug system

**Configuration:**
- `/client/vite.config.ts` - Dev proxy + build config
- `/client/.env.example` - Environment variables

---

## 14. Quick Reference: All API Endpoints

### Settings
```
GET    /api/settings/path/{encodedPath}
PUT    /api/settings/path/{encodedPath}
POST   /api/settings/batch
POST   /api/settings/reset
GET    /api/settings/physics/{graphName}
PUT    /api/settings/physics/{graphName}
GET    /api/settings/health
POST   /api/settings/cache/clear
```

### Analytics
```
GET    /analytics/params
POST   /analytics/params
GET    /analytics/constraints
POST   /analytics/constraints
POST   /analytics/clustering/run
GET    /analytics/clustering/status?task_id=X
POST   /analytics/clustering/cancel?task_id=X
POST   /analytics/clustering/focus
GET    /analytics/stats
GET    /analytics/gpu-status
POST   /analytics/insights
POST   /analytics/anomaly/toggle
GET    /analytics/anomaly/current
POST   /analytics/shortest-path
```

### Workspace
```
GET    /workspace/list
POST   /workspace/create
GET    /workspace/{id}
PUT    /workspace/{id}
DELETE /workspace/{id}
POST   /workspace/{id}/favorite
POST   /workspace/{id}/archive
PUT    /workspace/{id}/settings
GET    /workspace/{id}/members
```

### Graph
```
POST   /graph/export
POST   /graph/share
GET    /graph/shared/{shareId}
DELETE /graph/shared/{shareId}
POST   /graph/publish
GET    /graph/shared
PUT    /graph/shared/{shareId}
POST   /graph/nodes/batch-update
POST   /graph/nodes/batch-create
DELETE /graph/nodes/batch-delete
PUT    /graph/edges/batch-update
GET    /graph/auto-balance-notifications
```

### Ontology (May be missing)
```
GET    /api/ontology/graph
POST   /api/ontology/load
POST   /api/ontology/validate
```

### Bots/Agents (May be missing)
```
GET    /api/bots/status
GET    /api/bots/data
GET    /api/bots/agents
```

### Telemetry (May be missing)
```
POST   /api/telemetry/errors
POST   /api/telemetry/upload
POST   /api/client-logs
POST   /api/errors/log
```

### Health
```
GET    /health
GET    /api/settings/health
```

### WebSocket
```
WS     /wss                    (Main WebSocket)
WS     /ws/analytics           (Analytics updates)
WS     /speech (or similar)    (Voice WebSocket)
```

---

## End of Report

**Next Steps:**
1. Backend team: Review missing endpoints section
2. Backend team: Validate settings API path-based routing
3. Backend team: Confirm WebSocket binary protocol format
4. DevOps: Ensure Nginx proxy routes match frontend expectations
5. QA: Test 404 errors with browser DevTools Network tab

**Contact:** For questions about this analysis, check the frontend codebase or review the Vite proxy configuration.
