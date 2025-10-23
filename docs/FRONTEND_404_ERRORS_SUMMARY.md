# VisionFlow Frontend 404 Errors - Quick Summary

**Critical Issues Causing 404 Errors**

---

## 1. Missing Backend Endpoints

### High Priority (Causing Active Errors)

#### Bots/Agents API
```
❌ GET  /api/bots/status
❌ GET  /api/bots/data
❌ GET  /api/bots/agents
```

**Used by:**
- `AgentNodesLayer.tsx` (line 336, 348)
- `AgentTelemetry.ts` (line 191-192)
- `BotsWebSocketIntegration.ts`

**Impact:** Agent visualization and telemetry broken

---

#### Ontology API
```
❌ GET  /api/ontology/graph
❌ POST /api/ontology/load
❌ POST /api/ontology/validate
```

**Used by:**
- `OntologyModeToggle.tsx` (line 60)
- `useOntologyStore.ts` (line 131, 158)

**Impact:** Ontology mode switching broken

---

#### Telemetry/Logging
```
❌ POST /api/telemetry/errors
❌ POST /api/telemetry/upload
❌ POST /api/client-logs
❌ POST /api/errors/log
```

**Used by:**
- `remoteLogger.ts` (line 30) - Quest 3 debugging
- `ErrorBoundary.tsx` (line 93)
- `useErrorHandler.tsx` (line 298)
- `AgentTelemetry.ts` (line 142)

**Impact:** Remote logging broken (critical for Quest 3 debugging)

---

### Medium Priority (May Cause Errors)

#### Analytics Extensions
```
⚠️ POST /api/analytics/shortest-path
⚠️ POST /api/analytics/clustering/focus
```

**Used by:**
- `analyticsStore.ts` (line 264)
- `SemanticClusteringControls.tsx` (line 179)

**Impact:** Advanced analytics features broken

---

#### Auto-Balance
```
⚠️ GET  /api/graph/auto-balance-notifications
```

**Used by:**
- `AutoBalanceIndicator.tsx` (line 25)

**Impact:** Auto-balance UI indicator broken

---

## 2. Settings API Issues

### Expected Backend Routes

The frontend uses **path-based REST API** with URL-encoded dot-notation paths:

```typescript
// Frontend expects:
GET  /api/settings/path/system.debug.enabled
PUT  /api/settings/path/system.debug.enabled
POST /api/settings/batch                      // Body: { paths: string[] }

// NOT bulk operations like:
GET  /api/settings (get all)
PUT  /api/settings (update all)
```

### Graph-Specific Physics

```typescript
GET  /api/settings/physics/logseq      // Logseq graph physics
PUT  /api/settings/physics/logseq
GET  /api/settings/physics/visionflow  // VisionFlow graph physics
PUT  /api/settings/physics/visionflow
GET  /api/settings/physics/default     // Default/fallback physics
PUT  /api/settings/physics/default
```

**Critical:** Frontend validates that `logseq` and `visionflow` physics are independent!

---

## 3. WebSocket Protocol Requirements

### Main WebSocket (`/wss`)

**Binary Message Header:**
```
[type: u8][graphTypeFlag: u8][payloadSize: u32]
```

**Graph Type Flags:**
- `0x01` = knowledge_graph mode
- `0x02` = ontology mode

**Position Update Format:**
```
[nodeId: u32][x: f32][y: f32][z: f32][vx: f32][vy: f32][vz: f32]
```

**Agent Nodes:**
- Agent/bot nodes have high bit set: `nodeId | 0x80000000`

### Settings WebSocket

```
WS /api/settings/ws
```

**Expected Messages:**
```json
{
  "type": "setting_updated",
  "path": "system.debug.enabled",
  "value": true,
  "timestamp": 1234567890
}
```

---

## 4. API Response Format

### Standard Response

```typescript
{
  success: boolean,
  data?: any,
  error?: string,
  message?: string
}
```

### Analytics Task Response

```typescript
{
  success: true,
  task_id: "uuid-string",
  task?: {
    task_id: string,
    status: 'pending' | 'running' | 'completed' | 'failed',
    progress: number,
    result?: any,
    error?: string
  }
}
```

---

## 5. Immediate Action Items

### Backend Team

1. **Implement Missing Endpoints (Priority Order):**
   ```
   1. POST /api/client-logs          (Quest 3 debugging critical!)
   2. GET  /api/bots/status           (Agent visualization)
   3. GET  /api/bots/data             (Agent data)
   4. GET  /api/bots/agents           (Agent list)
   5. GET  /api/ontology/graph        (Ontology mode)
   6. POST /api/ontology/load         (Ontology loading)
   7. POST /api/ontology/validate     (Ontology validation)
   8. POST /api/telemetry/errors      (Error reporting)
   9. POST /api/errors/log            (Error logging)
   ```

2. **Verify Settings API Routes:**
   - Path-based GET/PUT: `/api/settings/path/{encodedPath}`
   - Batch endpoint: `POST /api/settings/batch`
   - Graph physics: `/api/settings/physics/{graphName}`
   - Health check: `GET /api/settings/health`

3. **Verify WebSocket Protocol:**
   - Binary header format matches frontend parser
   - Graph type filtering works
   - Agent node high-bit flag is set
   - Settings WebSocket at `/api/settings/ws`

### DevOps Team

4. **Nginx Configuration:**
   - Verify `/api/*` proxy to backend:4000
   - Verify `/wss` WebSocket upgrade
   - Verify `/ws/*` WebSocket upgrade
   - Check CORS headers for development

5. **Environment Variables:**
   - `VITE_API_URL` - Backend URL (default: http://visionflow_container:4000)
   - `VITE_WS_URL` - WebSocket URL (default: ws://visionflow_container:4000)

### QA Team

6. **Testing:**
   - Open browser DevTools → Network tab
   - Look for 404 errors to `/api/*` endpoints
   - Check WebSocket connection at `/wss`
   - Test ontology mode toggle
   - Test agent visualization
   - Verify settings load properly

---

## 6. How to Debug

### Browser DevTools

1. **Network Tab:**
   ```
   Filter: /api/
   Look for: Status 404
   ```

2. **Console Tab:**
   ```
   Filter: [RemoteLogger]
   Filter: [SettingsStore]
   Filter: [WebSocketService]
   ```

3. **Application Tab:**
   ```
   Check: localStorage → nostr_session_token
   Check: localStorage → graph-viz-settings-v2
   ```

### Backend Logs

```bash
# Check for 404 errors in backend logs
grep "404" /var/log/visionflow/backend.log

# Check API routes
grep "Registered route" /var/log/visionflow/backend.log
```

---

## 7. Frontend API Client Usage

### Making API Calls

```typescript
import { unifiedApiClient } from '@/services/api/UnifiedApiClient';

// GET request
const data = await unifiedApiClient.getData<MyType>('/endpoint');

// POST request
const result = await unifiedApiClient.postData<Response>('/endpoint', payload);

// With auth bypass
const publicData = await unifiedApiClient.get('/public', { skipAuth: true });
```

### Settings Access

```typescript
import { useSettingsStore } from '@/store/settingsStore';

// Get setting
const value = useSettingsStore.getState().get<boolean>('system.debug.enabled');

// Set setting (auto-syncs to backend)
useSettingsStore.getState().set('system.debug.enabled', true);

// Batch update
useSettingsStore.getState().batchUpdate([
  { path: 'key1', value: 'value1' },
  { path: 'key2', value: 'value2' }
]);
```

### WebSocket Usage

```typescript
import { webSocketService } from '@/services/WebSocketService';

// Send message
webSocketService.sendMessage('custom_event', { data: 'value' });

// Listen for messages
webSocketService.on('setting_updated', (data) => {
  console.log('Setting changed:', data);
});

// Update node positions
webSocketService.sendNodePositionUpdates([
  { nodeId: 1, x: 0, y: 0, z: 0 }
]);
```

---

## 8. Common Errors and Solutions

### Error: "Failed to fetch"

**Cause:** Backend not running or CORS issue

**Solution:**
1. Check backend is running: `curl http://visionflow_container:4000/health`
2. Check Vite proxy in `vite.config.ts`
3. Check CORS headers in backend

---

### Error: "404 Not Found: /api/..."

**Cause:** Endpoint not implemented in backend

**Solution:**
1. Check backend route registration
2. Implement missing endpoint
3. Verify route path matches exactly

---

### Error: "WebSocket connection failed"

**Cause:** WebSocket endpoint not configured

**Solution:**
1. Check `/wss` is proxied correctly
2. Verify WebSocket upgrade headers
3. Check firewall/security groups

---

### Error: "Setting not found"

**Cause:** Path-based routing not working

**Solution:**
1. Verify backend supports `/api/settings/path/{encodedPath}`
2. Check URL encoding of dot-notation paths
3. Verify settings cache is populated

---

## 9. Testing Checklist

### Backend Endpoints

- [ ] `POST /api/client-logs` returns 200 OK
- [ ] `GET /api/bots/status` returns agent data
- [ ] `GET /api/bots/data` returns bot data
- [ ] `GET /api/ontology/graph` returns graph
- [ ] `GET /api/settings/path/system.debug.enabled` returns value
- [ ] `POST /api/settings/batch` accepts array of paths
- [ ] `GET /api/settings/physics/logseq` returns physics config

### WebSocket

- [ ] Connection to `/wss` succeeds
- [ ] Binary messages are received
- [ ] Graph type filtering works
- [ ] Position updates render correctly
- [ ] Settings WebSocket at `/api/settings/ws` connects

### Integration

- [ ] Frontend loads without 404 errors
- [ ] Settings load successfully
- [ ] Graph renders with nodes/edges
- [ ] Ontology mode toggle works
- [ ] Agent nodes appear in visualization
- [ ] Remote logging sends to backend

---

## End of Summary

For detailed API documentation, see: `VISIONFLOW_FRONTEND_API_ANALYSIS.md`

**Questions?** Check browser DevTools Network tab for exact failing requests.
