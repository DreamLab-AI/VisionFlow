---
title: Complete State Management Architecture
description: 1.  [Architecture Overview](#architecture-overview) 2.
category: explanation
tags:
  - architecture
  - patterns
  - structure
  - api
  - rest
related-docs:
  - diagrams/README.md
updated-date: 2025-12-18
difficulty-level: advanced
---

# Complete State Management Architecture

**Status**: Production Implementation
**Last Updated**: 2025-12-05
**Architecture**: Zustand + Immer + LocalStorage Persistence

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Store Catalog](#store-catalog)
3. [State Flow Patterns](#state-flow-patterns)
4. [Subscription System](#subscription-system)
5. [Persistence & Hydration](#persistence--hydration)
6. [Performance Optimizations](#performance-optimizations)
7. [Integration Patterns](#integration-patterns)
8. [Best Practices](#best-practices)

---

## Architecture Overview

### Technology Stack

```
┌─────────────────────────────────────────────────────────────┐
│                     STATE MANAGEMENT STACK                   │
├─────────────────────────────────────────────────────────────┤
│  Zustand 4.x          │ Core state management library       │
│  Immer 10.x           │ Immutable state updates            │
│  LocalStorage         │ Client-side persistence            │
│  React 18.x           │ Component integration              │
│  Custom Hooks         │ Selective loading & caching        │
└─────────────────────────────────────────────────────────────┘
```

### Core Principles

1. **Partial State Loading**: Load only required settings paths
2. **Lazy Loading**: Load sections on-demand to minimize startup time
3. **Batch Operations**: Group updates to reduce re-renders
4. **Selective Subscriptions**: Subscribe to specific paths, not entire store
5. **Immutable Updates**: Use Immer for safe state mutations
6. **Auto-persistence**: Automatic background saves with retry logic

---

## Store Catalog

### 1. SettingsStore (`settingsStore.ts`)

**Purpose**: Central settings management with partial loading and lazy initialization

**Location**: `/client/src/store/settingsStore.ts`

#### State Structure

```typescript
interface SettingsState {
  // Partial state (lazily loaded)
  partialSettings: "DeepPartial<Settings>"
  loadedPaths: "Set<string>"          // Tracks which paths are loaded
  loadingSections: "Set<string>"      // Tracks sections currently loading

  // Full settings mirror (deprecated, for backward compat)
  settings: "DeepPartial<Settings>"

  // Authentication state
  initialized: boolean
  authenticated: boolean
  user: "{ isPowerUser: boolean; pubkey: string } | null"
  isPowerUser: boolean

  // Subscription management
  subscribers: "Map<string, Set<() => void>>"
}
```

#### Essential Paths (Loaded at Startup)

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
  'visualisation.graphs.logseq.physics',
  'visualisation.graphs.visionflow.physics',
  'nodeFilter.enabled',
  'nodeFilter.qualityThreshold',
  'nodeFilter.authorityThreshold',
  'nodeFilter.filterByQuality',
  'nodeFilter.filterByAuthority',
  'nodeFilter.filterMode'
];
```

#### Key Actions

```typescript
// Initialization
initialize: () => Promise<void>
  - Waits for authentication (max 3s)
  - Loads essential paths only
  - Marks store as initialized
  - Triggers AutoSaveManager initialization

// Path-based access
"get: <T>(path: SettingsPath) => T"
  - Returns undefined if path not loaded
  - Validates path exists in loadedPaths
  - Traverses nested object by dot notation

"set: <T>(path: SettingsPath, value: T) => void"
  - Updates local state immediately
  - Triggers backend API update (async)
  - Marks path as loaded

// Lazy loading
"ensureLoaded: (paths: string[]) => Promise<void>"
  - Checks loadedPaths
  - Fetches unloaded paths via API
  - Updates state and loadedPaths

"loadSection: (section: string) => Promise<void>"
  - Maps section to paths
  - Calls ensureLoaded
  - Prevents duplicate loads

// Batch operations
"updateSettings: (updater: (draft: Settings) => void) => void"
  - Uses Immer produce()
  - Detects changed paths automatically
  - Sends batch update to backend
  - Notifies all subscribers

"batchUpdate: (updates: Array<{path, value}>) => void"
  - Updates multiple paths atomically
  - Single API call for all updates

// Special handlers
"updatePhysics: (graphName: string, params: Partial<GPUPhysicsParams>) => void"
  - Validates physics parameters
  - Clamps values to safe ranges
  - Sends WebSocket notification
  - Dispatches custom DOM event

"notifyPhysicsUpdate: (graphName: string, params: Partial<GPUPhysicsParams>) => void"
  - Sends WebSocket message if connected
  - Dispatches 'physicsParametersUpdated' event
  - Used by GPU physics engine

// Subscription system
"subscribe: (path: SettingsPath, callback: () => void, immediate?: boolean) => () => void"
  - Adds callback to subscribers map
  - Optionally calls immediately
  - Returns unsubscribe function

"unsubscribe: (path: SettingsPath, callback: () => void) => void"
  - Removes callback from subscribers
  - Cleans up empty subscription sets

"notifyViewportUpdate: (path: SettingsPath) => void"
  - Special handler for viewport settings
  - Triggers 'viewport.update' subscribers
  - Used for camera/view changes

// Import/Export
"exportSettings: () => Promise<string>"
  - Fetches all settings if only essentials loaded
  - Returns JSON string

"importSettings: (jsonString: string) => Promise<void>"
  - Parses JSON
  - Extracts all paths
  - Calls batchUpdate

"resetSettings: () => Promise<void>"
  - Calls backend reset API
  - Clears local state
  - Re-initializes with essentials
```

#### Persistence Strategy

```typescript
persist(
  storeCreator,
  {
    name: 'graph-viz-settings-v2',
    storage: createJSONStorage(() => localStorage),

    // Only persist essentials and auth
    partialize: (state) => ({
      authenticated: state.authenticated,
      user: state.user,
      isPowerUser: state.isPowerUser,
      essentialPaths: ESSENTIAL_PATHS.reduce((acc, path) => {
        const value = state.partialSettings[path];
        if (value !== undefined) {
          acc[path] = value;
        }
        return acc;
      }, {})
    }),

    // Restore auth state only
    merge: (persistedState, currentState) => ({
      ...currentState,
      authenticated: persistedState.authenticated || false,
      user: persistedState.user || null,
      isPowerUser: persistedState.isPowerUser || false
    })
  }
)
```

#### Section Path Mapping

```typescript
"function getSectionPaths(section: string): string[]" {
  const sectionPathMap = {
    'physics': [
      'visualisation.graphs.logseq.physics',
      'visualisation.graphs.visionflow.physics'
    ],
    'rendering': [
      'visualisation.rendering.ambientLightIntensity',
      'visualisation.rendering.backgroundColor',
      // ... 10+ rendering paths
    ],
    'xr': [
      'xr.enabled',
      'xr.mode',
      'xr.enableHandTracking',
      'xr.enableHaptics',
      'xr.quality'
    ],
    'glow': [ /* glow effect paths */ ],
    'hologram': [ /* hologram effect paths */ ],
    'nodes': [ /* node visualization paths */ ],
    'edges': [ /* edge visualization paths */ ],
    'labels': [ /* label settings paths */ ]
  };
  return sectionPathMap[section] || [];
}
```

---

### 2. AutoSaveManager (`autoSaveManager.ts`)

**Purpose**: Batch and persist settings changes with retry logic

**Location**: `/client/src/store/autoSaveManager.ts`

#### State Structure

```typescript
class AutoSaveManager {
  private pendingChanges: "Map<string, any>"
  private saveDebounceTimer: "NodeJS.Timeout | null"
  private isInitialized: boolean
  private retryCount: "Map<string, number>"

  private readonly MAX_RETRIES = 3
  private readonly DEBOUNCE_DELAY = 500 // ms
  private readonly RETRY_DELAY = 1000   // ms

  private readonly CLIENT_ONLY_PATHS = [
    'auth.nostr.connected',
    'auth.nostr.publicKey'
  ]
}
```

#### Core Methods

```typescript
// Queue management
"queueChange(path: string, value: any): void"
  - Skip if not initialized
  - Skip client-only paths
  - Add to pendingChanges map
  - Reset retry count
  - Schedule flush (debounced)

"queueChanges(changes: Map<string, any>): void"
  - Batch version of queueChange
  - Filters client-only paths
  - Single debounced flush

// Persistence
"scheduleFlush(): void"
  - Clear existing timer
  - Set 500ms debounce timer
  - Calls flushPendingChanges

"forceFlush(): Promise<void>"
  - Cancel debounce timer
  - Immediate flush
  - Used on unmount/critical saves

"flushPendingChanges(): Promise<void>"
  - Convert map to BatchOperation[]
  - Call settingsApi.updateSettingsByPaths
  - Clear successful updates
  - Retry failed updates

// Retry logic
"retryFailedChanges(failedUpdates: BatchOperation[], error: any): Promise<void>"
  - Track retry count per path
  - Exponential backoff: RETRY_DELAY * 2^retryCount
  - Max 3 retries per path
  - Show toast on max retries
  - Schedule individual retries

// Status
"hasPendingChanges(): boolean"
"getPendingCount(): number"
```

#### Flow Diagram

```
┌────────────────────────────────────────────────────────────┐
│                    AutoSaveManager Flow                     │
└────────────────────────────────────────────────────────────┘

User updates setting
       ↓
settingsStore.set(path, value)
       ↓
autoSaveManager.queueChange(path, value)
       ↓
[Wait 500ms - debounce]
       ↓
flushPendingChanges()
       ↓
settingsApi.updateSettingsByPaths([{path, value}, ...])
       ↓
   ┌───────────┴───────────┐
   │                       │
SUCCESS                FAILURE
   │                       │
Clear map           Increment retry
   │                Track in retryCount
   └───────────┬───────────┘
               │
         [Wait 1s, 2s, 4s - exponential]
               │
         Retry up to 3 times
               │
          ┌────┴────┐
          │         │
      SUCCESS   MAX RETRIES
          │         │
     Clear map  Show toast
```

---

### 3. AnalyticsStore (`analyticsStore.ts`)

**Purpose**: Shortest path computations with caching and metrics

**Location**: `/client/src/features/analytics/store/analyticsStore.ts`

#### State Structure

```typescript
interface AnalyticsState {
  // Current computation result
  currentResult: SSSPResult | null

  // Cache indexed by sourceNodeId
  cache: SSSPCache

  // Loading state
  loading: boolean
  error: string | null

  // Performance metrics
  metrics: AnalyticsMetrics

  // Graph change detection
  lastGraphHash: string | null
}

interface SSSPResult {
  sourceNodeId: string
  distances: "Record<string, number>"
  predecessors: "Record<string, string | null>"
  unreachableCount: number
  computationTime: number
  timestamp: number
  algorithm: "'dijkstra' | 'bellman-ford' | 'floyd-warshall'"
}

interface SSSPCache {
  "[sourceNodeId: string]": {
    result: SSSPResult
    lastAccessed: number
    graphHash: string
  }
}

interface AnalyticsMetrics {
  totalComputations: number
  cacheHits: number
  cacheMisses: number
  averageComputationTime: number
  lastComputationTime: number
}
```

#### Key Actions

```typescript
// Computation
"computeSSSP: (nodes: GraphNode[], edges: GraphEdge[], sourceNodeId: string, algorithm?: 'dijkstra' | 'bellman-ford') => Promise<SSSPResult>"
  - Validates inputs
  - Computes graph hash
  - Checks cache (by sourceNodeId + graphHash)
  - If cache hit: return cached result
  - If cache miss:
    1. Try server API: POST /api/analytics/shortest-path
    2. If server fails: fallback to local algorithm
  - Store result in cache
  - Update metrics
  - LRU eviction (max 50 entries)

// Algorithms
"dijkstra(nodes, edges, sourceNodeId): SSSPResult"
  - Classic Dijkstra's algorithm
  - Handles disconnected graphs
  - Returns distances + predecessors

"bellmanFord(nodes, edges, sourceNodeId): SSSPResult"
  - Bellman-Ford algorithm
  - Handles negative weights
  - Detects negative cycles

// Cache management
"getCachedResult(sourceNodeId, graphHash): SSSPResult | null"
  - Validates graph hash matches
  - Updates lastAccessed timestamp
  - Returns null on miss

"clearCache(): void"
  - Clears entire cache
  - Resets lastGraphHash

"invalidateCache(): void"
  - Clears cache + lastGraphHash
  - Used when graph structure changes

"cleanExpiredCache(maxAge = 24h): void"
  - Removes entries older than maxAge
  - Based on lastAccessed timestamp

// Utilities
"normalizeDistances(result): Record<string, number>"
  - Min-max normalization to [0, 1]
  - Handles Infinity values
  - Used for visualization

"getUnreachableNodes(result): string[]"
  - Returns node IDs with distance = Infinity

// Metrics
"updateMetrics(computationTime, fromCache): void"
  - Increments totalComputations
  - Tracks cache hits/misses
  - Updates averageComputationTime
  - Only averages non-cached computations

"resetMetrics(): void"
```

#### Persistence

```typescript
persist(
  storeCreator,
  {
    name: 'analytics-store',
    storage: createJSONStorage(() => localStorage),

    // Persist cache and metrics
    partialize: (state) => ({
      cache: state.cache,
      metrics: state.metrics
    })
  }
)
```

#### Cache Strategy

```
┌────────────────────────────────────────────────────────────┐
│                      SSSP Cache Strategy                    │
└────────────────────────────────────────────────────────────┘

Cache Key: sourceNodeId (e.g., "node_123")
Cache Value: {
  result: SSSPResult,
  lastAccessed: timestamp,
  graphHash: "btoa(nodeIds|edgeIds)"
}

Validation:
  - Check sourceNodeId exists in cache
  - Verify graphHash matches current graph
  - If mismatch: invalidate entry, recompute

Eviction Policy:
  - LRU (Least Recently Used)
  - Max 50 entries
  - Sort by lastAccessed on insert
  - Keep top 50 entries

Graph Hash:
  hashGraph(nodes, edges) => string
  - Sort node IDs alphabetically
  - Sort edge tuples (source-target-weight)
  - Concatenate with '|' separator
  - Base64 encode
  - Example: "bm9kZTEsbm9kZTJ8ZWRnZTEtZWRnZTI="

Invalidation Triggers:
  - Node added/removed
  - Edge added/removed
  - Edge weight changed
  - Manual invalidateCache() call
```

---

### 4. MultiUserStore (`multiUserStore.ts`)

**Purpose**: Multi-user collaboration state with WebSocket sync

**Location**: `/client/src/store/multiUserStore.ts`

#### State Structure

```typescript
interface MultiUserState {
  localUserId: string
  users: "Record<string, UserData>"
  connectionStatus: "'disconnected' | 'connecting' | 'connected'"
}

interface UserData {
  id: string
  name?: string
  position: "[number, number, number]"
  rotation: "[number, number, number]"
  isSelecting: boolean
  selectedNodeId?: string
  lastUpdate: number
  color?: string
}
```

#### Key Actions

```typescript
// User management
"setLocalUserId(userId: string): void"
"updateUser(userId: string, data: Partial<UserData>): void"
"removeUser(userId: string): void"

// Local user actions
"updateLocalPosition(position: [x, y, z], rotation: [x, y, z]): void"
  - Updates localUserId's position/rotation
  - Triggers WebSocket sync

"updateLocalSelection(isSelecting: boolean, selectedNodeId?: string): void"
  - Updates localUserId's selection state
  - Broadcasts to other users

// Connection
"setConnectionStatus(status): void"

// Maintenance
"clearStaleUsers(staleThreshold = 30s): void"
  - Removes users with lastUpdate > threshold
  - Keeps localUserId regardless
```

#### WebSocket Integration

```typescript
class MultiUserConnection {
  private ws: "WebSocket | null"
  private reconnectInterval: "NodeJS.Timeout | null"
  private heartbeatInterval: "NodeJS.Timeout | null"

  "connect(): void"
    - Opens WebSocket connection
    - Sends 'join' message
    - Starts heartbeat (5s interval)
    - Schedules reconnect on close

  "handleMessage(message): void"
    - 'userUpdate': Update remote user
    - 'userLeft': Remove user
    - 'sync': Batch user state sync
    - 'pong': Heartbeat response

  "send(data): void"
    - Sends JSON message if connected

  "disconnect(): void"
    - Stops heartbeat
    - Closes WebSocket
    - Clears intervals
}
```

#### Subscription Pattern

```typescript
// Hook for XR user tracking
export const useXRUserTracking = () => {
  const updateLocalPosition = useMultiUserStore(state => state.updateLocalPosition)
  const updateLocalSelection = useMultiUserStore(state => state.updateLocalSelection)
  const connection = useRef<MultiUserConnection | null>(null)

  useEffect(() => {
    // Subscribe to local user changes
    const unsubscribe = useMultiUserStore.subscribe(
      state => state.users[state.localUserId],
      (userData) => {
        if (userData && connection.current) {
          // Broadcast local changes via WebSocket
          connection.current.send({
            type: 'userUpdate',
            userId: userData.id,
            data: {
              position: userData.position,
              rotation: userData.rotation,
              isSelecting: userData.isSelecting,
              selectedNodeId: userData.selectedNodeId
            }
          })
        }
      }
    )
    return unsubscribe
  }, [])

  return { updatePosition: updateLocalPosition, updateSelection: updateLocalSelection }
}
```

---

### 5. OntologyStore (`useOntologyStore.ts`)

**Purpose**: Ontology constraint validation and metrics

**Location**: `/client/src/features/ontology/store/useOntologyStore.ts`

#### State Structure

```typescript
interface OntologyState {
  loaded: boolean
  validating: boolean
  violations: Violation[]
  constraintGroups: ConstraintGroup[]
  metrics: OntologyMetrics
}

interface Violation {
  axiomType: string
  description: string
  severity: "'error' | 'warning'"
  affectedEntities: "string[]"
}

interface ConstraintGroup {
  id: string
  name: string
  enabled: boolean
  strength: number              // 0.0 - 1.0
  description: string
  constraintCount: number
  icon?: string
}

interface OntologyMetrics {
  axiomCount: number
  classCount: number
  propertyCount: number
  individualCount: number
  constraintsByType: "Record<string, number>"
  cacheHitRate: number
  validationTimeMs: number
  lastValidated?: number
}
```

#### Key Actions

```typescript
// Ontology operations
"loadOntology(fileUrl: string): Promise<void>"
  - POST /api/ontology/load
  - Updates metrics and constraintGroups
  - Sets loaded = true

"validateOntology(): Promise<void>"
  - POST /api/ontology/validate
  - Sends enabled constraint groups
  - Updates violations and metrics
  - Records validation timestamp

// Constraint management
"toggleConstraintGroup(id: string): void"
  - Flips enabled state
  - Requires re-validation

"updateStrength(id: string, strength: number): void"
  - Updates constraint strength (0-1)
  - Used for soft constraints

// State setters
"setLoaded(loaded: boolean): void"
"setValidating(validating: boolean): void"
"setViolations(violations: Violation[]): void"
"setMetrics(metrics: OntologyMetrics): void"
```

#### Default Constraint Groups

```typescript
constraintGroups: [
  {
    id: 'subsumption',
    name: 'Subsumption',
    enabled: true,
    strength: 0.8,
    description: 'Class hierarchy constraints'
  },
  {
    id: 'disjointness',
    name: 'Disjointness',
    enabled: true,
    strength: 1.0,
    description: 'Disjoint class constraints'
  },
  {
    id: 'property_domain',
    name: 'Property Domain',
    enabled: true,
    strength: 0.9,
    description: 'Property domain restrictions'
  },
  {
    id: 'property_range',
    name: 'Property Range',
    enabled: true,
    strength: 0.9,
    description: 'Property range restrictions'
  },
  {
    id: 'cardinality',
    name: 'Cardinality',
    enabled: false,
    strength: 0.7,
    description: 'Property cardinality constraints'
  }
]
```

---

## State Flow Patterns

### Pattern 1: UI Event → Store → Backend

```
┌──────────────────────────────────────────────────────────────┐
│                   Standard Update Flow                        │
└──────────────────────────────────────────────────────────────┘

1. User interaction (slider, toggle, input)
       ↓
2. Event handler calls store action
   Example: settingsStore.set('physics.damping', 0.9)
       ↓
3. Store updates local state immediately (optimistic update)
   set: <T>(path, value) => {
     set(state => updateNestedValue(state, path, value))
       ↓
4. Store queues backend update
   autoSaveManager.queueChange(path, value)
       ↓
5. AutoSaveManager debounces (500ms)
       ↓
6. Batch update to backend
   settingsApi.updateSettingsByPaths([...])
       ↓
7. On success: clear queue
   On failure: retry with exponential backoff
       ↓
8. UI re-renders via store subscription
```

### Pattern 2: Backend → Store → UI

```
┌──────────────────────────────────────────────────────────────┐
│                   WebSocket Update Flow                       │
└──────────────────────────────────────────────────────────────┘

1. Server sends WebSocket message
   Example: { type: 'graph_data', data: {...} }
       ↓
2. WebSocketService processes message
   handleWebSocketMessage(message)
       ↓
3. Service updates relevant store(s)
   graphDataManager.handleGraphData(data)
   multiUserStore.updateUser(userId, data)
       ↓
4. Store triggers re-render
   Via Zustand subscribeWithSelector
       ↓
5. Components re-render with new data
   Using selective subscriptions
```

### Pattern 3: Lazy Loading

```
┌──────────────────────────────────────────────────────────────┐
│                   Lazy Section Loading                        │
└──────────────────────────────────────────────────────────────┘

1. User opens settings section (e.g., "Physics")
       ↓
2. Component checks if section loaded
   const isLoaded = settingsStore.isLoaded('visualisation.graphs.logseq.physics')
       ↓
3. If not loaded, trigger load
   await settingsStore.loadSection('physics')
       ↓
4. Store checks loadingSections to prevent duplicates
       ↓
5. Map section to paths
   getSectionPaths('physics') => [
     'visualisation.graphs.logseq.physics',
     'visualisation.graphs.visionflow.physics'
   ]
       ↓
6. Call ensureLoaded(paths)
   Filter out already-loaded paths
   Fetch unloaded paths from API
       ↓
7. Update state with new data
   set({ partialSettings, loadedPaths })
       ↓
8. Mark section as loaded
   loadingSections.delete('physics')
       ↓
9. Component re-renders with loaded settings
```

### Pattern 4: Batch Operations

```
┌──────────────────────────────────────────────────────────────┐
│                   Batch Update Pattern                        │
└──────────────────────────────────────────────────────────────┘

1. Multiple settings changed (e.g., preset applied)
       ↓
2. Use updateSettings with Immer
   settingsStore.updateSettings((draft) => {
     draft.physics.damping = 0.9
     draft.physics.repelK = 100
     draft.rendering.backgroundColor = '#000'
   })
       ↓
3. Immer produces new immutable state
       ↓
4. Store detects changed paths automatically
   findChangedPaths(oldSettings, newSettings)
   => ['physics.damping', 'physics.repelK', 'rendering.backgroundColor']
       ↓
5. Update local state atomically
   set({ partialSettings: newSettings, loadedPaths: [...] })
       ↓
6. Send single batched API call
   settingsApi.updateSettingsByPaths([
     { path: 'physics.damping', value: 0.9 },
     { path: 'physics.repelK', value: 100 },
     { path: 'rendering.backgroundColor', value: '#000' }
   ])
       ↓
7. Notify all subscribers
   subscribers.forEach(callbacks => callbacks.forEach(cb => cb()))
```

---

## Subscription System

### Zustand Subscriptions

#### Basic Subscription

```typescript
// Subscribe to entire store (inefficient)
useSettingsStore.subscribe(
  (state) => console.log('Store changed:', state)
)

// Subscribe with selector (efficient)
useSettingsStore.subscribe(
  (state) => state.get('physics.damping'),
  (newValue, oldValue) => {
    console.log('Damping changed:', oldValue, '->', newValue)
  }
)
```

#### React Hook Integration

```typescript
// useSettingsStore with selector
const damping = useSettingsStore(state => state.get('physics.damping'))

// With custom equality function
const physics = useSettingsStore(
  state => state.get('visualisation.graphs.logseq.physics'),
  (prev, next) => shallowEqual(prev, next)
)
```

### Custom Subscription API

```typescript
// settingsStore.subscribe (custom implementation)
const unsubscribe = settingsStore.subscribe(
  'physics.damping',
  () => {
    // Callback fired when physics.damping changes
    const newValue = settingsStore.get('physics.damping')
    console.log('Damping updated:', newValue)
  },
  true // immediate: call callback on subscribe
)

// Cleanup
unsubscribe()
```

#### Implementation Details

```typescript
interface SettingsState {
  subscribers: Map<string, Set<() => void>>

  subscribe: (path: SettingsPath, callback: () => void, immediate?: boolean) => () => void
  unsubscribe: (path: SettingsPath, callback: () => void) => void
}

subscribe: (path, callback, immediate = true) => {
  // Add callback to subscribers map
  set(state => {
    const subscribers = new Map(state.subscribers)
    if (!subscribers.has(path)) {
      subscribers.set(path, new Set())
    }
    subscribers.get(path)!.add(callback)
    return { subscribers }
  })

  // Optionally call immediately
  if (immediate && get().initialized) {
    callback()
  }

  // Return unsubscribe function
  return () => get().unsubscribe(path, callback)
}

unsubscribe: (path, callback) => {
  set(state => {
    const subscribers = new Map(state.subscribers)
    if (subscribers.has(path)) {
      const callbacks = subscribers.get(path)!
      callbacks.delete(callback)
      if (callbacks.size === 0) {
        subscribers.delete(path)
      }
    }
    return { subscribers }
  })
}
```

### Selective Subscription Hooks

#### useSelectiveSetting

```typescript
// Hook with caching and deduplication
const damping = useSelectiveSetting<number>(
  'physics.damping',
  {
    enableCache: true,          // Use response cache
    enableDeduplication: true,  // Deduplicate concurrent requests
    fallbackToStore: true       // Fallback to store if API fails
  }
)

// Implementation flow:
1. Subscribe to store path
2. Fetch from API (with cache/deduplication)
3. Return API value if available, else store value
4. Re-subscribe on path change
```

#### useSelectiveSettings (Batch)

```typescript
// Load multiple settings with batch API call
const { damping, repelK, maxVelocity } = useSelectiveSettings({
  damping: 'physics.damping',
  repelK: 'physics.repelK',
  maxVelocity: 'physics.maxVelocity'
}, {
  enableBatchLoading: true,   // Single API call for all paths
  enableCache: true,          // Cache responses
  fallbackToStore: true       // Fallback on API failure
})

// Implementation:
1. Debounce multiple path requests (50ms)
2. Send single batch API call
3. Cache individual responses
4. Subscribe to all paths in store
5. Return merged API + store values
```

#### useSettingSetter

```typescript
const { set, batchedSet, immediateSet, updateSettings } = useSettingSetter()

// Debounced single update (50ms)
set('physics.damping', 0.9)

// Debounced batch update (50ms)
batchedSet({
  'physics.damping': 0.9,
  'physics.repelK': 100
})

// Immediate update with Immer
immediateSet((draft) => {
  draft.physics.damping = 0.9
})

// Direct access to updateSettings
updateSettings((draft) => {
  draft.physics.damping = 0.9
})
```

#### useSettingsSubscription

```typescript
// Subscribe to path changes with custom callback
useSettingsSubscription(
  'physics.damping',
  (newValue) => {
    console.log('Damping changed:', newValue)
  },
  {
    immediate: true,           // Call on mount
    enableCache: false,        // Use store, not API
    dependencies: []           // React dependencies
  }
)

// Use case: side effects, logging, external sync
```

---

## Persistence & Hydration

### LocalStorage Persistence

#### SettingsStore Persistence

```typescript
persist(
  storeCreator,
  {
    name: 'graph-viz-settings-v2',
    storage: createJSONStorage(() => localStorage),

    // Serialize: Only persist auth state + essentials
    partialize: (state) => ({
      authenticated: state.authenticated,
      user: state.user,
      isPowerUser: state.isPowerUser,
      essentialPaths: ESSENTIAL_PATHS.reduce((acc, path) => {
        const value = state.partialSettings[path];
        if (value !== undefined) {
          acc[path] = value;
        }
        return acc;
      }, {})
    }),

    // Deserialize: Restore auth state only
    merge: (persistedState, currentState) => {
      if (!persistedState) return currentState;
      return {
        ...currentState,
        authenticated: persistedState.authenticated || false,
        user: persistedState.user || null,
        isPowerUser: persistedState.isPowerUser || false
        // Don't restore settings - refetch from API
      };
    },

    // Lifecycle
    onRehydrateStorage: () => (state) => {
      if (state && debugState.isEnabled()) {
        logger.info('Settings store rehydrated from storage');
      }
    }
  }
)
```

**Why only auth state?**
- Settings may change on server
- Avoid stale data issues
- Essential paths refetched on startup
- User-specific settings managed server-side

#### AnalyticsStore Persistence

```typescript
persist(
  storeCreator,
  {
    name: 'analytics-store',
    storage: createJSONStorage(() => localStorage),

    // Persist cache and metrics
    partialize: (state) => ({
      cache: state.cache,
      metrics: state.metrics
    }),

    onRehydrateStorage: () => (state) => {
      if (state && debugState.isEnabled()) {
        logger.info('Analytics store rehydrated', {
          cacheEntries: Object.keys(state.cache || {}).length,
          metrics: state.metrics
        });
      }
    }
  }
)
```

**Why persist cache?**
- SSSP computations are expensive
- Graph structure rarely changes
- Cache validated via graphHash
- Metrics track cache effectiveness

### Hydration Flow

```
┌──────────────────────────────────────────────────────────────┐
│                   Store Hydration Flow                        │
└──────────────────────────────────────────────────────────────┘

1. App initializes
       ↓
2. Zustand reads from localStorage
   const persisted = localStorage.getItem('graph-viz-settings-v2')
       ↓
3. Parse JSON
   const state = JSON.parse(persisted)
       ↓
4. Merge with default state
   merge(persistedState, currentState)
       ↓
5. Store initialized with merged state
   useSettingsStore.getState() => { authenticated: true, user: {...}, ... }
       ↓
6. Call store.initialize()
   - Wait for auth ready (max 3s)
   - Fetch essential paths from API
   - Update state with fresh data
   - Mark initialized = true
       ↓
7. Trigger onRehydrateStorage callback
   Log rehydration success
       ↓
8. Components mount and subscribe
   Use hydrated + initialized state
```

### State Migration

```typescript
// Version 1 → Version 2 migration example
const migrateSettingsV1ToV2 = (v1State: any): SettingsState => {
  return {
    ...v1State,
    // Add new fields
    loadedPaths: new Set(v1State.loadedPaths || []),
    loadingSections: new Set(),
    // Rename fields
    partialSettings: v1State.settings || {},
    // Remove deprecated fields
    // ...
  };
};

// Apply migration on hydrate
merge: (persistedState, currentState) => {
  if (!persistedState) return currentState;

  const version = persistedState._version || 1;
  let migratedState = persistedState;

  if (version === 1) {
    migratedState = migrateSettingsV1ToV2(persistedState);
  }

  return {
    ...currentState,
    ...migratedState,
    _version: 2
  };
}
```

---

## Performance Optimizations

### 1. Memoization & Selectors

#### Shallow Equality Checks

```typescript
"function shallowEqual<T>(a: T, b: T): boolean" {
  if (a === b) return true;
  if (!a || !b) return false;
  if (typeof a !== 'object' || typeof b !== 'object') return a === b;

  const keysA = Object.keys(a as any);
  const keysB = Object.keys(b as any);

  if (keysA.length !== keysB.length) return false;

  for (const key of keysA) {
    if (!(key in (b as any)) || (a as any)[key] !== (b as any)[key]) {
      return false;
    }
  }

  return true;
}

// Usage
const physics = useSettingsStore(
  state => state.get('visualisation.graphs.logseq.physics'),
  shallowEqual // Prevent re-renders if object contents unchanged
);
```

#### Selector Memoization

```typescript
"const useSettingsSelector = <T>(selector: (settings: any) => T, options: {...} = {}): T => {" {
  const {
    equalityFn = shallowEqual,
    enableCache = false,
    cacheTTL = 5000
  } = options;

  // Memoize selector by toString() (function source)
  const memoizedSelector = useCallback(selector, [selector.toString()]);

  // Use memoized selector with custom equality
  const value = useSettingsStore(
    state => memoizedSelector(state.settings),
    equalityFn
  );

  // Optional caching
  const cacheKey = useMemo(() => {
    if (!enableCache) return null;
    return `selector_${memoizedSelector.toString()}_${JSON.stringify(value)}`;
  }, [enableCache, memoizedSelector, value]);

  useEffect(() => {
    if (cacheKey && enableCache) {
      setCachedResponse(cacheKey, value, cacheTTL);
    }
  }, [cacheKey, enableCache, value, cacheTTL]);

  return value;
};
```

### 2. Request Deduplication

```typescript
// Global request map
const requestMap = new Map<string, Promise<any>>();

async function getDedicatedSetting<T>(path: SettingsPath): Promise<T> {
  // Check cache first
  const cached = getCachedResponse<T>(path);
  if (cached !== undefined) {
    return cached;
  }

  // Deduplicate concurrent requests
  if (requestMap.has(path)) {
    return requestMap.get(path) as Promise<T>;
  }

  // Make request
  const request = settingsApi.getSettingByPath(path)
    .then(value => {
      setCachedResponse(path, value);
      return value as T;
    })
    .finally(() => {
      requestMap.delete(path);
    });

  requestMap.set(path, request);
  return request;
}

// Multiple components requesting same path simultaneously
// → Only 1 API call made
// → All callers receive same promise
```

### 3. Response Caching

```typescript
"interface CacheEntry<T>" {
  value: T;
  timestamp: number;
  ttl: number;
}

"const responseCache = new Map<string, CacheEntry<any>>();"
const CACHE_TTL = 5000; // 5 seconds

"function getCachedResponse<T>(key: string): T | undefined" {
  const entry = responseCache.get(key);
  if (!entry) return undefined;

  const now = Date.now();
  if (now - entry.timestamp > entry.ttl) {
    responseCache.delete(key);
    return undefined;
  }

  return entry.value;
}

"function setCachedResponse<T>(key: string, value: T, ttl: number = CACHE_TTL): void" {
  responseCache.set(key, {
    value,
    timestamp: Date.now(),
    ttl
  });
}

// Cache benefits:
// - Reduce API calls for frequently accessed settings
// - Improve perceived performance
// - Stale data acceptable for short TTL (5s)
```

### 4. Debouncing

```typescript
const debounceMap = new Map<string, ReturnType<typeof setTimeout>>();
const DEBOUNCE_DELAY = 50; // ms

// Debounced setting update
const debouncedSet = useCallback((path: SettingsPath, value: any) => {
  const key = `single_${path}`;

  // Clear existing timer
  if (debounceMap.has(key)) {
    clearTimeout(debounceMap.get(key)!);
  }

  // Set new timer
  const timeout = setTimeout(() => {
    setByPath(path, value);
    debounceMap.delete(key);
  }, DEBOUNCE_DELAY);

  debounceMap.set(key, timeout);
}, [setByPath]);

// Benefits:
// - Coalesce rapid slider movements into single update
// - Reduce API calls
// - Reduce re-renders
// - User gets immediate visual feedback (optimistic update)
```

#### AutoSaveManager Debouncing

```typescript
// 500ms debounce for batching updates
private scheduleFlush() {
  if (this.saveDebounceTimer) {
    clearTimeout(this.saveDebounceTimer);
  }

  this.saveDebounceTimer = setTimeout(() => {
    this.flushPendingChanges();
  }, this.DEBOUNCE_DELAY); // 500ms
}

// User changes 10 settings rapidly:
// → 10 queueChange() calls
// → Timer resets 10 times
// → After 500ms idle: 1 batch API call with all 10 updates
```

### 5. Batch Loading

```typescript
"function debouncedBatchLoad(paths: string[], callback: (results: Record<string, any>) => void): void" {
  const key = paths.sort().join('|');

  if (debounceMap.has(key)) {
    clearTimeout(debounceMap.get(key)!);
  }

  const timeout = setTimeout(async () => {
    try {
      // Check cache first
      const cachedResults: Record<string, any> = {};
      const uncachedPaths: string[] = [];

      for (const path of paths) {
        const cached = getCachedResponse(path);
        if (cached !== undefined) {
          cachedResults[path] = cached;
        } else {
          uncachedPaths.push(path);
        }
      }

      // Fetch uncached paths in batch
      let apiResults: Record<string, any> = {};
      if (uncachedPaths.length > 0) {
        apiResults = await settingsApi.getSettingsByPaths(uncachedPaths);

        // Cache results
        for (const [path, value] of Object.entries(apiResults)) {
          setCachedResponse(path, value);
        }
      }

      // Merge cached + API results
      const allResults = { ...cachedResults, ...apiResults };
      callback(allResults);
    } catch (error) {
      logger.error('Batch load failed:', error);
      callback({});
    } finally {
      debounceMap.delete(key);
    }
  }, DEBOUNCE_DELAY);

  debounceMap.set(key, timeout);
}

// Example:
// Component A requests: ['physics.damping', 'physics.repelK']
// Component B requests: ['physics.maxVelocity']
// → Both debounced to 50ms
// → Single API call: getSettingsByPaths(['physics.damping', 'physics.repelK', 'physics.maxVelocity'])
```

### 6. LRU Cache Eviction

```typescript
// In analyticsStore.computeSSSP()
set(state => produce(state, draft => {
  // ... compute and cache result ...

  draft.cache[sourceNodeId] = {
    result,
    lastAccessed: Date.now(),
    graphHash
  };

  // LRU eviction: keep max 50 entries
  const cacheEntries = Object.entries(draft.cache);
  if (cacheEntries.length > 50) {
    // Sort by lastAccessed (most recent first)
    const sortedEntries = cacheEntries.sort(
      ([,a], [,b]) => b.lastAccessed - a.lastAccessed
    );
    // Keep top 50
    draft.cache = Object.fromEntries(sortedEntries.slice(0, 50));
  }
}));

// Benefits:
// - Prevent unbounded cache growth
// - Keep most-used entries
// - Automatic memory management
```

---

## Integration Patterns

### Pattern 1: Component → Store

```typescript
// Component: SettingsPanel.tsx
import { useSettingsStore } from '@/store/settingsStore';

const SettingsPanel: React.FC = () => {
  // Subscribe to specific setting
  const damping = useSettingsStore(state => state.get('physics.damping'));
  const set = useSettingsStore(state => state.set);

  return (
    <Slider
      value={damping}
      onChange={(value) => set('physics.damping', value)}
      min={0}
      max={1}
      step={0.01}
    />
  );
};

// Flow:
// 1. User drags slider
// 2. onChange fires
// 3. set('physics.damping', 0.85) called
// 4. Store updates immediately (optimistic)
// 5. AutoSaveManager queues update
// 6. Component re-renders with new value
// 7. After 500ms: API call sent
```

### Pattern 2: WebSocket → Store → Component

```typescript
// WebSocketService.ts
class WebSocketService {
  private handleBinaryMessage(data: ArrayBuffer) {
    const message = binaryProtocol.decode(data);

    switch (message.type) {
      case MessageType.GRAPH_DATA:
        graphDataManager.handleGraphData(message.data);
        break;

      case MessageType.MULTI_USER_UPDATE:
        multiUserStore.updateUser(message.userId, message.data);
        break;
    }
  }
}

// Component: UserAvatar.tsx
const UserAvatar: React.FC<{ userId: string }> = ({ userId }) => {
  // Subscribe to specific user
  const user = useMultiUserStore(
    state => state.users[userId],
    (prev, next) => shallowEqual(prev, next)
  );

  if (!user) return null;

  return (
    <Avatar
      position={user.position}
      rotation={user.rotation}
      color={user.color}
      isSelecting={user.isSelecting}
    />
  );
};

// Flow:
// 1. Server sends binary WebSocket message
// 2. WebSocketService decodes message
// 3. multiUserStore.updateUser() called
// 4. UserAvatar component subscribed to specific userId
// 5. Component re-renders with new position
```

### Pattern 3: Lazy Loading in Component

```typescript
// Component: PhysicsSettingsPanel.tsx
const PhysicsSettingsPanel: React.FC = () => {
  const loadSection = useSettingsStore(state => state.loadSection);
  const isLoaded = useSettingsStore(state =>
    state.isLoaded('visualisation.graphs.logseq.physics')
  );
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!isLoaded) {
      setLoading(true);
      loadSection('physics')
        .then(() => setLoading(false))
        .catch(err => {
          console.error('Failed to load physics settings:', err);
          setLoading(false);
        });
    }
  }, [isLoaded, loadSection]);

  if (loading) return <LoadingSpinner />;

  return <PhysicsControls />;
};

// Flow:
// 1. Component mounts
// 2. Check if 'physics' section loaded
// 3. If not: call loadSection('physics')
// 4. Store fetches physics paths from API
// 5. Store updates partialSettings + loadedPaths
// 6. isLoaded becomes true
// 7. Component re-renders with loaded data
```

### Pattern 4: Batch Operations

```typescript
// Component: PresetSelector.tsx
const PresetSelector: React.FC = () => {
  const updateSettings = useSettingsStore(state => state.updateSettings);

  const applyPreset = (preset: Preset) => {
    updateSettings((draft) => {
      // Batch update multiple settings atomically
      draft.physics.damping = preset.physics.damping;
      draft.physics.repelK = preset.physics.repelK;
      draft.physics.springK = preset.physics.springK;
      draft.rendering.backgroundColor = preset.rendering.backgroundColor;
      draft.rendering.ambientLightIntensity = preset.rendering.ambientLightIntensity;
    });
  };

  return (
    <Select onChange={(value) => applyPreset(PRESETS[value])}>
      <Option value="default">Default</Option>
      <Option value="high_performance">High Performance</Option>
      <Option value="high_quality">High Quality</Option>
    </Select>
  );
};

// Flow:
// 1. User selects preset
// 2. applyPreset() called with preset object
// 3. updateSettings with Immer draft
// 4. Immer produces new immutable state
// 5. Store detects 5 changed paths
// 6. Store updates local state atomically
// 7. Single batched API call with 5 updates
// 8. All subscribed components re-render once
```

### Pattern 5: Analytics Integration

```typescript
// Component: ShortestPathVisualizer.tsx
const ShortestPathVisualizer: React.FC<{ sourceNodeId: string }> = ({ sourceNodeId }) => {
  const nodes = useGraphStore(state => state.nodes);
  const edges = useGraphStore(state => state.edges);
  const computeSSSP = useAnalyticsStore(state => state.computeSSSP);
  const currentResult = useAnalyticsStore(state => state.currentResult);
  const loading = useAnalyticsStore(state => state.loading);

  useEffect(() => {
    if (nodes.length > 0 && sourceNodeId) {
      computeSSSP(nodes, edges, sourceNodeId, 'dijkstra');
    }
  }, [nodes, edges, sourceNodeId, computeSSSP]);

  if (loading) return <LoadingSpinner />;
  if (!currentResult) return null;

  return (
    <DistanceHeatmap
      nodes={nodes}
      distances={currentResult.distances}
      normalizedDistances={useAnalyticsStore(state =>
        state.normalizeDistances(currentResult)
      )}
    />
  );
};

// Flow:
// 1. Component mounts with sourceNodeId
// 2. useEffect triggers computeSSSP
// 3. Store checks cache (sourceNodeId + graphHash)
// 4. If cache miss: compute locally or fetch from server
// 5. Store updates currentResult
// 6. Component re-renders with SSSP result
// 7. DistanceHeatmap visualizes shortest paths
```

---

## Best Practices

### 1. Store Organization

**DO:**
- Use domain-specific stores (settings, analytics, multi-user, ontology)
- Keep stores focused and cohesive
- Use TypeScript for type safety
- Document complex state transformations

**DON'T:**
- Create a single monolithic store
- Mix unrelated state (e.g., UI state + domain data)
- Use nested stores (flat structure is simpler)

### 2. Subscription Patterns

**DO:**
- Subscribe to specific paths with selectors
- Use shallow equality for object comparisons
- Unsubscribe in cleanup (useEffect return)
- Use subscribeWithSelector middleware for fine-grained control

**DON'T:**
- Subscribe to entire store in components
- Use deep equality checks (expensive)
- Forget to unsubscribe (memory leaks)
- Create subscriptions in render functions

### 3. State Updates

**DO:**
- Use Immer for complex nested updates
- Batch related updates with updateSettings
- Validate and clamp values before setting
- Provide optimistic updates for better UX

**DON'T:**
- Mutate state directly (use Immer or spread syntax)
- Make multiple individual API calls for related updates
- Block UI on state updates (async where possible)
- Update state in render functions

### 4. Performance

**DO:**
- Use memoization (useMemo, useCallback) for expensive computations
- Implement request deduplication for concurrent requests
- Cache responses with appropriate TTL
- Debounce rapid updates (sliders, inputs)
- Use LRU eviction for bounded caches

**DON'T:**
- Compute derived state in render (use selectors)
- Make redundant API calls (cache and deduplicate)
- Subscribe to unused state (over-subscription)
- Store large data in state (use refs or external storage)

### 5. Error Handling

**DO:**
- Catch and log all store errors
- Provide fallbacks for failed API calls
- Retry failed operations with exponential backoff
- Show user-friendly error messages (toast)
- Track error metrics

**DON'T:**
- Swallow errors silently
- Retry indefinitely without limits
- Expose internal errors to users
- Ignore network failures (handle gracefully)

### 6. Testing

**DO:**
- Test store actions independently
- Mock API calls in tests
- Test subscription callbacks
- Test persistence and hydration
- Test error scenarios and retries

**DON'T:**
- Test implementation details (e.g., internal state structure)
- Rely on external services in unit tests
- Skip edge cases (null, undefined, empty arrays)
- Test React hooks without proper utilities (@testing-library/react-hooks)

### 7. Documentation

**DO:**
- Document store purpose and responsibilities
- Explain complex state transformations
- Provide usage examples for custom hooks
- Document performance characteristics
- Keep documentation updated with code changes

**DON'T:**
- Assume store behavior is self-explanatory
- Leave outdated comments in code
- Over-document trivial operations
- Skip documenting integration patterns

---

## Complete Data Flow Example

### Scenario: User Updates Physics Settings

```
┌─────────────────────────────────────────────────────────────────┐
│  Complete Flow: User Updates Damping Slider                     │
└─────────────────────────────────────────────────────────────────┘

1. USER ACTION
   User drags "Damping" slider to 0.85

2. EVENT HANDLER
   <Slider onChange={(value) => set('physics.damping', value)} />

3. STORE UPDATE (Optimistic)
   settingsStore.set('physics.damping', 0.85)

   set: <T>(path, value) => {
     // Update local state immediately
     set(state => {
       const newPartialSettings = { ...state.partialSettings };
       setNestedValue(newPartialSettings, path, value);
       return {
         partialSettings: newPartialSettings,
         settings: newPartialSettings,
         loadedPaths: new Set([...state.loadedPaths, path])
       };
     });

     // Queue backend update (async)
     settingsApi.updateSettingByPath(path, value).catch(error => {
       logger.error(`Failed to update setting ${path}:`, error);
     });
   }

4. COMPONENT RE-RENDER
   const damping = useSettingsStore(state => state.get('physics.damping'));
   // damping = 0.85 (immediate visual feedback)

5. AUTOSAVE MANAGER
   autoSaveManager.queueChange('physics.damping', 0.85)

   queueChange(path, value) {
     this.pendingChanges.set(path, value);
     this.resetRetryCount(path);
     this.scheduleFlush(); // 500ms debounce
   }

6. DEBOUNCE PERIOD
   [Wait 500ms]
   (If user continues dragging, timer resets)

7. FLUSH TO BACKEND
   flushPendingChanges() {
     const updates = Array.from(this.pendingChanges.entries())
       .map(([path, value]) => ({ path, value }));

     settingsApi.updateSettingsByPaths(updates)
       .then(() => {
         // Clear successful updates
         updates.forEach(({ path }) => {
           this.pendingChanges.delete(path);
           this.resetRetryCount(path);
         });
       })
       .catch(error => {
         // Retry with exponential backoff
         this.retryFailedChanges(updates, error);
       });
   }

8. BACKEND API CALL
   PUT /api/settings/physics
   Body: { damping: 0.85 }

9. SERVER PROCESSES UPDATE
   - Validates damping value (0.0 - 1.0)
   - Persists to database
   - Returns 200 OK

10. SUCCESS CLEANUP
    autoSaveManager clears pendingChanges
    No further action needed

11. WEBSOCKET NOTIFICATION (Optional)
    If physics settings trigger GPU update:

    notifyPhysicsUpdate('logseq', { damping: 0.85 })

    const wsService = window.webSocketService;
    wsService.send({
      type: 'physics_parameter_update',
      timestamp: Date.now(),
      graph: 'logseq',
      parameters: { damping: 0.85 }
    });

12. GPU PHYSICS ENGINE UPDATE
    Server receives WebSocket message
    Updates GPU compute buffers
    Next simulation frame uses new damping value

13. CUSTOM DOM EVENT
    window.dispatchEvent(new CustomEvent('physicsParametersUpdated', {
      detail: { graphName: 'logseq', params: { damping: 0.85 } }
    }));

14. EVENT LISTENERS
    Other components listening for physics updates react
    (e.g., update UI indicators, log metrics)

TOTAL LATENCY:
- Optimistic update: ~1ms (immediate)
- Backend persistence: ~500ms (debounced) + network latency
- WebSocket propagation: ~50ms (if applicable)
- GPU update: Next frame (~16ms @ 60fps)

USER EXPERIENCE:
- Slider moves immediately (optimistic update)
- No UI blocking or lag
- Settings persisted in background
- Physics engine updated in real-time
- No flash or jarring transitions
```

---

## Appendix: Store API Reference

### SettingsStore

```typescript
interface SettingsState {
  // State
  partialSettings: "DeepPartial<Settings>"
  settings: "DeepPartial<Settings>"
  loadedPaths: "Set<string>"
  loadingSections: "Set<string>"
  initialized: boolean
  authenticated: boolean
  user: "{ isPowerUser: boolean; pubkey: string } | null"
  isPowerUser: boolean
  subscribers: "Map<string, Set<() => void>>"

  // Initialization
  "initialize: () => Promise<void>"
  "setAuthenticated: (authenticated: boolean) => void"
  "setUser: (user: { isPowerUser: boolean; pubkey: string } | null) => void"

  // Access
  "get: <T>(path: SettingsPath) => T"
  "set: <T>(path: SettingsPath, value: T) => void"

  // Subscriptions
  "subscribe: (path: SettingsPath, callback: () => void, immediate?: boolean) => () => void"
  "unsubscribe: (path: SettingsPath, callback: () => void) => void"
  "notifyViewportUpdate: (path: SettingsPath) => void"

  // Lazy loading
  "ensureLoaded: (paths: string[]) => Promise<void>"
  "loadSection: (section: string) => Promise<void>"
  "isLoaded: (path: SettingsPath) => boolean"

  // Batch operations
  "updateSettings: (updater: (draft: Settings) => void) => void"
  "getByPath: <T>(path: SettingsPath) => Promise<T>"
  "setByPath: <T>(path: SettingsPath, value: T) => void"
  "batchUpdate: (updates: Array<{path: SettingsPath, value: any}>) => void"
  "flushPendingUpdates: () => Promise<void>"

  // Import/Export
  "resetSettings: () => Promise<void>"
  "exportSettings: () => Promise<string>"
  "importSettings: (jsonString: string) => Promise<void>"

  // Specialized updates
  "updateComputeMode: (mode: string) => void"
  "updateClustering: (config: ClusteringConfig) => void"
  "updateConstraints: (constraints: ConstraintConfig[]) => void"
  "updatePhysics: (graphName: string, params: Partial<GPUPhysicsParams>) => void"
  "updateWarmupSettings: (settings: WarmupSettings) => void"
  "notifyPhysicsUpdate: (graphName: string, params: Partial<GPUPhysicsParams>) => void"
}
```

### AnalyticsStore

```typescript
interface AnalyticsState {
  // State
  currentResult: "SSSPResult | null"
  cache: SSSPCache
  loading: boolean
  error: "string | null"
  metrics: AnalyticsMetrics
  lastGraphHash: "string | null"

  // Computation
  "computeSSSP: (nodes: GraphNode[], edges: GraphEdge[], sourceNodeId: string, algorithm?: 'dijkstra' | 'bellman-ford' | 'floyd-warshall') => Promise<SSSPResult>"

  // Cache management
  "clearResults: () => void"
  "clearCache: () => void"
  "getCachedResult: (sourceNodeId: string, graphHash: string) => SSSPResult | null"
  "invalidateCache: () => void"
  "cleanExpiredCache: (maxAge?: number) => void"

  // Utilities
  "normalizeDistances: (result: SSSPResult) => Record<string, number>"
  "getUnreachableNodes: (result: SSSPResult) => string[]"

  // Metrics
  "updateMetrics: (computationTime: number, fromCache: boolean) => void"
  "resetMetrics: () => void"
  "setError: (error: string | null) => void"
}
```

### MultiUserStore

```typescript
interface MultiUserState {
  // State
  localUserId: string
  users: "Record<string, UserData>"
  connectionStatus: "'disconnected' | 'connecting' | 'connected'"

  // User management
  "setLocalUserId: (userId: string) => void"
  "updateUser: (userId: string, data: Partial<UserData>) => void"
  "removeUser: (userId: string) => void"

  // Local user actions
  "updateLocalPosition: (position: [number, number, number], rotation: [number, number, number]) => void"
  "updateLocalSelection: (isSelecting: boolean, selectedNodeId?: string) => void"

  // Connection
  "setConnectionStatus: (status: 'disconnected' | 'connecting' | 'connected') => void"

  // Maintenance
  "clearStaleUsers: (staleThreshold?: number) => void"
}
```

### OntologyStore

```typescript
interface OntologyState {
  // State
  loaded: boolean
  validating: boolean
  violations: "Violation[]"
  constraintGroups: "ConstraintGroup[]"
  metrics: OntologyMetrics

  // State setters
  "setLoaded: (loaded: boolean) => void"
  "setValidating: (validating: boolean) => void"
  "setViolations: (violations: Violation[]) => void"
  "setMetrics: (metrics: OntologyMetrics) => void"

  // Constraint management
  "toggleConstraintGroup: (id: string) => void"
  "updateStrength: (id: string, strength: number) => void"

  // Ontology operations
  "loadOntology: (fileUrl: string) => Promise<void>"
  "validateOntology: () => Promise<void>"
}
```

---

---

---

## Related Documentation

- [Server-Side Actor System - Complete Architecture Documentation](../../server/actors/actor-system-complete.md)
- [X-FluxAgent Integration Plan for ComfyUI MCP Skill](../../../multi-agent-docker/x-fluxagent-adaptation-plan.md)
- [VisionFlow GPU CUDA Architecture - Complete Technical Documentation](../../infrastructure/gpu/cuda-architecture-complete.md)
- [Server Architecture](../../../concepts/architecture/core/server.md)
- [VisionFlow Documentation Modernization - Final Report](../../../DOCUMENTATION_MODERNIZATION_COMPLETE.md)

## Summary

This document provides a complete map of the client-side state management architecture:

1. **4 main stores**: Settings, Analytics, MultiUser, Ontology
2. **Partial loading**: Only essential paths loaded at startup, lazy load on demand
3. **Batch operations**: Group updates to reduce API calls and re-renders
4. **Auto-persistence**: AutoSaveManager handles background saves with retry logic
5. **Subscription system**: Fine-grained subscriptions with selective hooks
6. **Performance optimizations**: Caching, deduplication, debouncing, memoization, LRU eviction
7. **LocalStorage persistence**: Selective persistence of auth state and cache
8. **Integration patterns**: Clear data flow from UI → Store → Backend → WebSocket → GPU

All state transformations follow predictable patterns with comprehensive error handling and retry logic. The architecture supports real-time updates, multi-user collaboration, and high-performance visualization with minimal latency.
