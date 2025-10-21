# Frontend Settings Implementation Audit

## Executive Summary

**Critical Finding**: The TypeScript client sends and expects **camelCase** format exclusively, with **NO case conversion** anywhere in the client codebase.

## Client API Contract

### Request Format (What Client Sends)

**File**: `/client/src/api/settingsApi.ts`

#### Single Path Update (Line 93)
```typescript
await unifiedApiClient.putData(`${API_BASE}/path`, { path: update.path, value: update.value });
```
**Sends**: `{ "path": "visualisation.nodes.baseColor", "value": "#ff0000" }`

#### Batch Update (Line 166)
```typescript
await unifiedApiClient.putData(`${API_BASE}/batch`, { updates: chunk });
```
**Sends**:
```json
{
  "updates": [
    { "path": "visualisation.nodes.baseColor", "value": "#ff0000" },
    { "path": "visualisation.physics.springK", "value": 0.5 }
  ]
}
```

#### Get Single Path (SettingsCacheClient.ts Line 174)
```typescript
const response = await fetch(`/api/settings/path?path=${encodeURIComponent(path)}`)
```
**Sends**: `GET /api/settings/path?path=visualisation.nodes.baseColor`

#### Get Batch Paths (SettingsCacheClient.ts Line 229)
```typescript
body: JSON.stringify({ paths: uncachedPaths })
```
**Sends**:
```json
{
  "paths": [
    "visualisation.nodes.baseColor",
    "visualisation.physics.springK"
  ]
}
```

### Response Format (What Client Expects)

**File**: `/client/src/api/settingsApi.ts`

#### Single Path Response (Expected)
```json
{
  "success": true,
  "value": "#ff0000"
}
```

#### Batch Response (Expected - Line 171)
```json
{
  "results": [
    {
      "path": "visualisation.nodes.baseColor",
      "success": true,
      "value": "#ff0000"
    }
  ]
}
```

#### Get Paths Response (SettingsCacheClient.ts Line 244)
```typescript
Object.entries(batchResult).forEach(([path, value]) => {
  results[path] = value;
});
```
**Expected**:
```json
{
  "visualisation.nodes.baseColor": "#ff0000",
  "visualisation.physics.springK": 0.5
}
```

## Case Conversion Analysis

### NO Case Conversion Found

**Searched for**: `snake_case`, `camelCase`, `to_snake`, `toCamel`, `convertCase`, `snakeCase`, `kebabCase`, `convertKey`, `transformKey`

**Result**:
- ✅ **No case conversion utilities exist in client code**
- ✅ **No transformation of keys in UnifiedApiClient**
- ✅ **Raw JSON sent/received without modification**

### Comments Mentioning Case Handling

**Only mentions are documentation, NOT conversion code**:

1. `/client/src/features/settings/config/settings.ts:37`
   ```typescript
   // Physics settings - using camelCase for client
   ```

2. `/client/src/features/bots/components/AgentTelemetryStream.tsx:122`
   ```typescript
   // Handle both camelCase and snake_case
   ```
   *Note: This is for BOTS API, not settings*

3. `/client/src/features/graph/components/FlowingEdges.tsx:79`
   ```typescript
   // Handle both snake_case and camelCase field names
   ```
   *Note: This is for GRAPH API, not settings*

## Essential Paths Loaded on Startup

**File**: `/client/src/store/settingsStore.ts` (Lines 63-75)

```typescript
const ESSENTIAL_PATHS = [
  'system.debug.enabled',
  'system.websocket.updateRate',
  'system.websocket.reconnectAttempts',
  'auth.enabled',
  'auth.required',
  'visualisation.rendering.context',
  'xr.enabled',
  'xr.mode',
  // Add physics settings so control center has them on startup
  'visualisation.graphs.logseq.physics',
  'visualisation.graphs.visionflow.physics'
];
```

**Format**: All **camelCase** dot-notation paths

## Client-Side Key Naming Standards

### TypeScript Interface (settings.ts)

**All properties use camelCase**:

```typescript
export interface NodeSettings {
  baseColor: string;          // NOT base_color
  metalness: number;          // NOT metalness (no snake)
  nodeSize: number;           // NOT node_size
  enableInstancing: boolean;  // NOT enable_instancing
}

export interface PhysicsSettings {
  springK: number;            // NOT spring_k
  repelK: number;             // NOT repel_k
  attractionK: number;        // NOT attraction_k
  maxVelocity: number;        // NOT max_velocity
  enableBounds: boolean;      // NOT enable_bounds
  boundsSize: number;         // NOT bounds_size
  restLength: number;         // NOT rest_length
  repulsionCutoff: number;    // NOT repulsion_cutoff
}
```

## Cache Client Behavior

**File**: `/client/src/services/SettingsCacheClient.ts`

### Key Storage Format
```typescript
private cache = new Map<string, CachedSetting>();
```

**Keys stored as-is** (no transformation):
- Cache key: `"visualisation.nodes.baseColor"`
- Stored value: `{ value: "#ff0000", path: "visualisation.nodes.baseColor", ... }`

### WebSocket Message Handling (Lines 102-115)

```typescript
switch (message.type) {
  case 'settings_changed':
    this.handleSettingChanged(message.data);
    break;
  case 'settings_batch_changed':
    this.handleBatchSettingsChanged(message.data);
    break;
}
```

**Expected WebSocket message format**:
```json
{
  "type": "settings_changed",
  "data": {
    "path": "visualisation.nodes.baseColor",
    "value": "#ff0000",
    "timestamp": 1234567890
  }
}
```

## Settings Store Path Access

**File**: `/client/src/store/settingsStore.ts`

### Path-Based Getter (Lines 336-370)
```typescript
get: <T>(path: SettingsPath): T | undefined => {
  // Navigate using dot notation
  const pathParts = path.split('.');
  let current: any = partialSettings;

  for (const part of pathParts) {
    if (current?.[part] === undefined) {
      return undefined;
    }
    current = current[part];
  }
  return current as T;
}
```

**Usage Example**:
```typescript
const color = settingsStore.get('visualisation.nodes.baseColor');
// Accesses: partialSettings.visualisation.nodes.baseColor
```

## Auto-Save Manager

**File**: `/client/src/store/autoSaveManager.ts`

### Batch Update Construction (Lines 99-105)
```typescript
const updates: BatchOperation[] = Array.from(this.pendingChanges.entries())
  .map(([path, value]) => ({ path, value }));

await settingsApi.updateSettingsByPaths(updates);
```

**No transformation** - paths sent exactly as queued.

### Client-Only Paths (Lines 26-36)
```typescript
private readonly CLIENT_ONLY_PATHS = [
  'auth.nostr.connected',
  'auth.nostr.publicKey',
];
```

These paths are **NEVER sent to server** (filtered out before sync).

## Component Usage Examples

### Physics Controls (useGraphSettings.ts)
```typescript
const fullPath = `visualisation.graphs.${graphName}.${path}`;
// Example: "visualisation.graphs.logseq.physics.springK"

const value = useSelectiveSetting<T>(fullPath, {
  enableCache: true,
  fallbackToStore: true
});
```

### Update Pattern (settingsStore.ts Line 392)
```typescript
autoSaveManager.queueChange(path, value);
// Path example: "visualisation.physics.springK"
// Value example: 0.5
```

## API Endpoints Called by Client

### Settings API Endpoints

1. **GET** `/api/settings/path?path={path}`
   - Request: `path` query param (camelCase)
   - Response: `{ success: true, value: <any> }`

2. **PUT** `/api/settings/path`
   - Body: `{ "path": "...", "value": ... }`
   - Response: `{ success: true }`

3. **PUT** `/api/settings/batch`
   - Body: `{ "updates": [{ "path": "...", "value": ... }, ...] }`
   - Response: `{ results: [{ path, success, value }, ...] }`

4. **POST** `/api/settings/batch`
   - Body: `{ "paths": ["path1", "path2", ...] }`
   - Response: `{ "path1": value1, "path2": value2, ... }`

5. **POST** `/api/settings/reset`
   - Body: (empty)
   - Response: Full settings object (camelCase)

## Critical Integration Points

### 1. Physics Parameters (Most Common Updates)

**Client sends** (settingsStore.ts Line 623):
```typescript
updatePhysics: (graphName: string, params: Partial<GPUPhysicsParams>) => {
  // params = { springK: 0.5, repelK: 100, ... }
}
```

**Path format**: `visualisation.graphs.{graphName}.physics.{paramName}`

**Example paths**:
- `visualisation.graphs.logseq.physics.springK`
- `visualisation.graphs.logseq.physics.repelK`
- `visualisation.graphs.visionflow.physics.maxVelocity`

### 2. Rendering Settings

**Example paths**:
- `visualisation.rendering.ambientLightIntensity`
- `visualisation.rendering.backgroundColor`
- `visualisation.rendering.enableShadows`

### 3. XR Settings

**Example paths**:
- `xr.enabled`
- `xr.mode`
- `xr.enableHandTracking`
- `xr.quality`

## Validation Against Backend

### Backend Must Accept (camelCase)

**JSON payloads**:
```json
{
  "path": "visualisation.nodes.baseColor",
  "value": "#ff0000"
}
```

**NOT**:
```json
{
  "path": "visualisation.nodes.base_color",
  "value": "#ff0000"
}
```

### Backend Must Return (camelCase)

**Response format**:
```json
{
  "visualisation.nodes.baseColor": "#ff0000",
  "visualisation.nodes.metalness": 0.5
}
```

**NOT**:
```json
{
  "visualisation.nodes.base_color": "#ff0000",
  "visualisation.nodes.metalness": 0.5
}
```

## Performance Characteristics

### Debouncing & Batching (settingsApi.ts)

- **Debounce delay**: 50ms (Line 11)
- **Batch size limit**: 25 updates per chunk (Line 12)
- **Critical updates**: Immediate (no debounce for physics)
- **Priority levels**: Critical, High, Normal, Low (Lines 22-27)

### Caching (SettingsCacheClient.ts)

- **Default TTL**: 5 minutes (Line 47)
- **Max cache size**: 1000 entries (Line 48)
- **Storage quota**: 5MB (Line 49)
- **Cache maintenance**: Every 5 minutes (Line 482)

## Error Handling

### Retry Logic (autoSaveManager.ts)

- **Max retries**: 3 (Line 22)
- **Retry delay**: 1s with exponential backoff (Line 24)
- **Retry on**: Network errors, 5xx errors
- **No retry on**: 401, 403 (auth errors)

### Fallback Behavior (settingsApi.ts Line 194)

If batch fails → Falls back to individual path updates

## Conclusion

### ✅ Client API Contract (DEFINITIVE)

1. **Request format**: Pure JSON with camelCase keys
2. **Path format**: Dot-notation with camelCase segments
3. **Response format**: JSON object mapping paths to values
4. **NO case conversion**: Keys sent/received as-is
5. **Essential paths**: 10 camelCase paths loaded on startup

### 🔴 Server Responsibility

The backend MUST:
1. Accept camelCase keys in request bodies
2. Use camelCase keys in dot-notation paths
3. Return camelCase keys in response objects
4. Store settings with camelCase keys internally OR convert at API boundary

### 📊 Most Common Settings Updated

Based on code analysis:
1. **Physics parameters**: `springK`, `repelK`, `attractionK`, `gravity`, etc.
2. **Rendering settings**: `ambientLightIntensity`, `backgroundColor`
3. **Node settings**: `baseColor`, `metalness`, `opacity`, `nodeSize`
4. **XR settings**: `enabled`, `mode`, `enableHandTracking`

### 🎯 Testing Recommendations

Test these exact paths with backend:
```json
[
  "system.debug.enabled",
  "visualisation.rendering.context",
  "visualisation.graphs.logseq.physics.springK",
  "visualisation.graphs.logseq.nodes.baseColor",
  "xr.enabled"
]
```

**Expected behavior**: Backend should accept and return these paths unchanged.
