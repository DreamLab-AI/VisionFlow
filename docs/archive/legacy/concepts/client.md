---
title: VisionFlow Client Architecture
description: The VisionFlow client is a high-performance 3D visualization platform built on React Three Fiber, implementing instanced rendering for 10,000+ node graphs with real-time WebSocket synchronization. ...
category: reference
tags:
  - architecture
  - design
  - api
  - api
  - api
related-docs:
  - guides/client/three-js-rendering.md
  - guides/client/state-management.md
  - guides/client/xr-integration.md
  - working/link-validation-report.md
  - concepts/architecture/core/server.md
updated-date: 2025-12-18
difficulty-level: advanced
---

# VisionFlow Client Architecture

**Status**: Production-Ready
**Technology Stack**: React 18 + TypeScript + Three.js + Zustand
**Performance**: 60 FPS @ 10,000 nodes | 90% bandwidth reduction with binary protocol
**Last Updated**: 2025-12-02

---

## Executive Summary

The VisionFlow client is a high-performance 3D visualization platform built on React Three Fiber, implementing instanced rendering for 10,000+ node graphs with real-time WebSocket synchronization. The architecture achieves 60 FPS rendering through aggressive optimization techniques including GPU instancing, binary WebSocket protocols, and lazy-loaded state management.

**Key Achievements**:
- **Instanced Rendering**: Single draw call for 10,000 nodes
- **Binary Protocol**: 34-byte node format (90% bandwidth reduction vs JSON)
- **Lazy Loading**: 87% faster initial load (203ms vs 1,650ms)
- **Auto-Save**: 93% reduction in API calls through debouncing
- **XR Support**: Quest 3 AR/VR with Babylon.js fallback

---

## Architecture Overview

```mermaid
graph TB
    subgraph "Application Layer"
        App[App.tsx<br/>Auth + Device Detection]
        MainLayout[MainLayout.tsx<br/>Desktop Mode]
        ImmersiveApp[ImmersiveApp.tsx<br/>VR/AR Mode]
    end

    subgraph "Rendering Layer"
        GraphManager[GraphManager.tsx<br/>Three.js Scene]
        InstancedMesh[InstancedMesh<br/>10k+ Nodes]
        HologramMaterial[HologramNodeMaterial<br/>Custom Shaders]
        SelectiveBloom[SelectiveBloom<br/>Post-Processing]
        FlowingEdges[FlowingEdges<br/>Edge Rendering]
    end

    subgraph "State Management"
        SettingsStore[Zustand Store<br/>Lazy Loading]
        GraphDataManager[graphDataManager<br/>Graph State]
        AnalyticsStore[analyticsStore<br/>SSSP/Clustering]
    end

    subgraph "Communication Layer"
        WebSocketService[WebSocketService<br/>Binary Protocol]
        BinaryProtocol[34-Byte Node Format]
        RESTAPIs[REST API Clients<br/>Settings/Analytics]
    end

    subgraph "Physics Engine"
        GraphWorker[Web Worker<br/>Force-Directed Layout]
        Float32Array[Float32Array<br/>Position Data]
    end

    App --> MainLayout
    App --> ImmersiveApp
    MainLayout --> GraphManager
    GraphManager --> InstancedMesh
    GraphManager --> FlowingEdges
    GraphManager --> SelectiveBloom
    InstancedMesh --> HologramMaterial

    GraphManager --> SettingsStore
    GraphManager --> GraphDataManager
    GraphManager --> AnalyticsStore

    GraphManager --> GraphWorker
    GraphWorker --> Float32Array
    Float32Array --> InstancedMesh

    GraphDataManager --> WebSocketService
    WebSocketService --> BinaryProtocol
    SettingsStore --> RESTAPIs
```

---

## Core Components

### 1. React 18 + Three.js Rendering

**File**: `client/src/features/graph/components/GraphManager.tsx` (1,047 lines)

#### Instanced Rendering Architecture

```typescript
// Single draw call for all nodes
<instancedMesh
  ref={meshRef}
  args={[undefined, undefined, visibleNodes.length]}
  frustumCulled={false}
>
  <sphereGeometry args={[0.5, 32, 32]} />
  <primitive object={materialRef.current} attach="material" />
</instancedMesh>
```

**Key Implementation**:
- **GPU Instancing**: All nodes share single geometry + material
- **Per-Instance Data**: Position, color, scale via `setMatrixAt()`
- **Custom Geometries**: Node type determines shape (sphere, box, octahedron, etc.)
- **Performance**: 10,000 nodes @ 60 FPS with single draw call

#### Node Type Geometry Mapping

```typescript
const getGeometryForNodeType = (type?: string): THREE.BufferGeometry => {
  switch (type?.toLowerCase()) {
    case 'folder': return new THREE.OctahedronGeometry(0.6, 0);
    case 'file': return new THREE.BoxGeometry(0.8, 0.8, 0.8);
    case 'concept': return new THREE.IcosahedronGeometry(0.5, 0);
    case 'todo': return new THREE.ConeGeometry(0.5, 1, 4);
    case 'reference': return new THREE.TorusGeometry(0.5, 0.2, 8, 16);
    default: return new THREE.SphereGeometry(0.5, 32, 32);
  }
};
```

**⚠️ Known Issue**: Instanced rendering currently uses sphere geometry for all nodes. Custom geometries require rebuilding instancedMesh, which causes frame drops with 10k+ nodes. **Solution**: Implement separate instanced meshes per geometry type.

#### Custom Shader Materials

**File**: `client/src/rendering/materials/HologramNodeMaterial.ts` (313 lines)

```glsl
// Fragment shader with holographic effects
uniform float time;
uniform vec3 baseColor;
uniform vec3 emissiveColor;
uniform float opacity;
uniform float glowStrength;
uniform float rimPower;

varying vec3 vNormal;
varying vec3 vWorldPosition;

void main() {
  vec3 viewDirection = normalize(cameraPosition - vWorldPosition);

  // Rim lighting for depth perception
  float rim = 1.0 - max(dot(viewDirection, vNormal), 0.0);
  rim = pow(rim, rimPower);

  // Animated scanlines
  float scanline = sin(vWorldPosition.y * 30.0 + time * 2.0);

  // Combine effects
  vec3 emission = emissiveColor * (rim + scanline) * glowStrength;
  gl_FragColor = vec4(baseColor + emission, opacity);
}
```

**Features**:
- **Rim Lighting**: Edge glow based on view angle
- **Animated Scanlines**: Vertical scanning effect
- **Instance Colors**: Per-node color overrides via `instanceColor`
- **Distance Fading**: Opacity decreases with camera distance
- **No Tone Mapping**: `toneMapped: false` for bloom compatibility

**Performance Benchmark**:
- Standard shaders: 10,000 nodes @ 60 FPS
- Hologram shaders: 10,000 nodes @ 58 FPS (3% overhead)
- **Browser Compatibility**: Chrome/Edge (WebGL 2), Safari (degraded, WebGL 1)

---

### 2. State Management with Zustand

**File**: `client/src/store/settingsStore.ts` (1,070 lines)

#### Lazy Loading Strategy

```mermaid
sequenceDiagram
    participant App
    participant Store
    participant Auth
    participant API

    App->>Store: initialize()
    Store->>Auth: waitForAuthReady()
    Auth-->>Store: Ready
    Store->>API: getSettingsByPaths(ESSENTIAL_PATHS)
    API-->>Store: Essential settings (12 paths)
    Store-->>App: initialized=true (203ms)

    Note over App,Store: User opens advanced panel
    App->>Store: ensureLoaded(['physics.*'])
    Store->>API: getSettingsByPaths(unloaded)
    API-->>Store: Physics settings (30 paths)
    Store-->>App: Settings merged (89ms)
```

**Essential Paths** (loaded at startup):
```typescript
const ESSENTIAL_PATHS = [
  'system.debug.enabled',
  'system.websocket.updateRate',
  'visualisation.rendering.context',
  'xr.enabled',
  'nodeFilter.enabled',          // Client-side filtering
  'nodeFilter.qualityThreshold',
  'nodeFilter.authorityThreshold'
];
```

**Performance Impact**:
- **Before**: Load all 150+ settings → 1,650ms
- **After**: Load 12 essential paths → 203ms
- **Result**: **87% reduction in initial load time**

#### Auto-Save System

**File**: `client/src/store/autoSaveManager.ts`

```typescript
// Debounced batch updates
const autoSaveManager = {
  queue: new Map<string, any>(),
  timeout: null,

  enqueue: (path: string, value: any) => {
    queue.set(path, value);
    clearTimeout(timeout);
    timeout = setTimeout(() => flushQueue(), 500); // 500ms debounce
  },

  flushQueue: async () => {
    const updates = Array.from(queue.entries());
    queue.clear();
    await settingsApi.updateSettingsByPaths(updates); // Single API call
  }
};
```

**Optimization Results**:
- **Before**: Slider change → immediate API call (10 calls/second while dragging)
- **After**: Batch all changes within 500ms → 1 API call
- **Result**: **93% reduction in API traffic** during rapid setting changes

**⚠️ Edge Case**: Settings changed within 500ms of tab close may be lost. **Mitigation**: `beforeunload` event triggers immediate flush.

#### Path-Based Subscriptions

```typescript
// Subscribe to specific setting paths
useSettingsStore.getState().subscribe(
  'visualisation.bloom.intensity',
  () => {
    // React to bloom changes
    updateBloomEffect();
  }
);
```

**Implementation**: Uses Zustand's `subscribeWithSelector` middleware for fine-grained reactivity without full store re-renders.

---

### 3. WebSocket Binary Protocol

**Files**:
- `client/src/services/WebSocketService.ts` (1,407 lines)
- `client/src/services/BinaryWebSocketProtocol.ts`
- `client/src/types/binaryProtocol.ts`

#### 34-Byte Node Format

```
┌─────────────────────────────────────────────────────────┐
│  Node ID (u16)  │  Position XYZ (f32×3)  │  Velocity XYZ (f32×3)  │
│     2 bytes     │      12 bytes          │       12 bytes         │
├─────────────────────────────────────────────────────────┤
│  SSSP Distance (f32)  │  SSSP Parent (i32)  │
│      4 bytes          │      4 bytes         │
└─────────────────────────────────────────────────────────┘
Total: 34 bytes per node
```

**Type Flags** (embedded in Node ID):
```typescript
const KNOWLEDGE_NODE_FLAG = 0x4000; // Bit 14
const AGENT_NODE_FLAG = 0x8000;     // Bit 15

// Dual graph support without additional bytes
export function isAgentNode(nodeId: number): boolean {
  return (nodeId & AGENT_NODE_FLAG) !== 0;
}

export function isKnowledgeNode(nodeId: number): boolean {
  return (nodeId & KNOWLEDGE_NODE_FLAG) !== 0;
}
```

#### Message Processing Pipeline

```mermaid
graph LR
    WS[WebSocket] --> Binary{Binary Data?}
    Binary -->|Yes| Parse[parseBinaryNodeData]
    Binary -->|No| JSON[JSON.parse]

    Parse --> Validate[validateNodePositions]
    Validate --> BatchQueue[NodePositionBatchQueue]
    BatchQueue --> GraphManager[Update InstancedMesh]

    JSON --> Filter{filter_confirmed?}
    JSON --> InitialLoad{initialGraphLoad?}
    Filter --> UpdateUI[Update Filter UI]
    InitialLoad --> SetGraph[graphDataManager.setGraphData]
```

**Binary vs JSON Comparison**:
```typescript
// JSON: 340 bytes/node
{
  "id": "node_12345",
  "position": {"x": 12.5, "y": -3.2, "z": 8.7},
  "velocity": {"x": 0.05, "y": -0.02, "z": 0.03},
  "sssp_distance": 5.2,
  "sssp_parent": 98
}

// Binary: 34 bytes/node
[u16 id][f32 x][f32 y][f32 z][f32 vx][f32 vy][f32 vz][f32 dist][i32 parent]
```

**Bandwidth Savings**: **90% reduction** (340 → 34 bytes)

#### Server-Side Filtering Synchronization

```typescript
// Client sends filter update to server
webSocketService.sendFilterUpdate({
  enabled: true,
  qualityThreshold: 0.7,
  authorityThreshold: 0.5,
  filterByQuality: true,
  filterByAuthority: true,
  filterMode: 'and'
});

// Server responds with filtered graph
{
  "type": "initialGraphLoad",
  "nodes": [...],  // Only nodes matching filter
  "edges": [...]
}

{
  "type": "filter_confirmed",
  "data": {
    "visible_nodes": 1247,
    "total_nodes": 10000
  }
}
```

**Two-Tier Filtering Architecture**:
1. **Server-Side**: Pre-filters nodes before transmission (reduces network traffic)
2. **Client-Side**: Additional hierarchical/quality filtering for instant UI updates

**⚠️ Performance Note**: With 10,000+ nodes, client-side filtering can take 15-30ms per frame. Consider using Web Worker for filtering logic.

---

### 4. Physics Simulation

**File**: `client/src/features/graph/workers/graphWorker.ts`

#### Web Worker Architecture

```mermaid
graph TB
    MainThread[Main Thread<br/>GraphManager] -->|postMessage| Worker[Web Worker<br/>graphWorker.ts]
    Worker -->|Float32Array| Proxy[graphWorkerProxy]
    Proxy -->|positions| MainThread

    Worker --> ForceDirected[Force-Directed Layout<br/>D3-Force Algorithm]
    ForceDirected --> Simulation[Position Updates<br/>60 times/second]
    Simulation --> Float32Array[Float32Array<br/>Zero-Copy Transfer]
```

**Implementation**:
```typescript
// GraphManager.tsx
useFrame(async (state, delta) => {
  // Get updated positions from Web Worker
  const positions = await graphWorkerProxy.tick(delta);
  nodePositionsRef.current = positions;

  if (positions) {
    // Update instanced mesh matrices
    for (let i = 0; i < visibleNodes.length; i++) {
      const i3 = i * 3;
      tempMatrix.setPosition(
        positions[i3],
        positions[i3 + 1],
        positions[i3 + 2]
      );
      meshRef.current.setMatrixAt(i, tempMatrix);
    }
    meshRef.current.instanceMatrix.needsUpdate = true;
  }
});
```

**Performance**:
- **Main Thread**: 60 FPS rendering loop unblocked
- **Worker Thread**: Physics simulation @ 60 ticks/second
- **Data Transfer**: `Float32Array` (zero-copy via Transferable Objects)
- **Overhead**: <2ms per frame for 10,000 nodes

**⚠️ Limitation**: Physics runs on ALL `graphData.nodes`, not just `visibleNodes`. This prevents layout collapse when toggling filters, but wastes CPU on hidden nodes.

---

### 5. Post-Processing Effects

**File**: `client/src/rendering/SelectiveBloom.tsx` (187 lines)

```typescript
<EffectComposer multisampling={0} renderPriority={1}>
  <Bloom
    intensity={bloomParams.intensity}           // 0.0-3.0 (default: 1.5)
    luminanceThreshold={bloomParams.threshold}  // 0.0-1.0 (default: 0.1)
    luminanceSmoothing={0.025}
    kernelSize={KernelSize.LARGE}               // Blur radius
    mipmapBlur={true}                           // Multi-resolution blur
    blendFunction={BlendFunction.ADD}
  />
</EffectComposer>
```

**Layer System**:
```typescript
const LAYERS = {
  BASE: 0,            // Default layer (no bloom)
  GRAPH_BLOOM: 1,     // Graph nodes with bloom effect
  ENVIRONMENT_GLOW: 2 // Environment objects with glow
};
```

**Performance Impact**:
- **Bloom Disabled**: 10,000 nodes @ 60 FPS
- **Bloom Enabled**: 10,000 nodes @ 57 FPS (5% overhead)
- **High-Resolution (1440p+)**: Drops to 45 FPS with `KernelSize.HUGE`

**⚠️ Browser Compatibility**:
- **Chrome/Edge**: Full support (WebGL 2 + EXT_color_buffer_float)
- **Safari**: Degraded (no floating-point textures, visible banding)
- **Mobile**: Disabled on iOS/Android (performance)

---

## Client-Side Hierarchical LOD

**Implementation**: `client/src/features/graph/hooks/useExpansionState.ts`

```typescript
// Hierarchy detection
const hierarchyMap = useMemo(() => {
  const hierarchy = detectHierarchy(graphData.nodes);
  logger.info(`Detected hierarchy: ${hierarchy.size} nodes, max depth: ${
    Math.max(...Array.from(hierarchy.values()).map(n => n.depth))
  }`);
  return hierarchy;
}, [graphData.nodes]);

// Visibility filtering
const visibleNodes = useMemo(() => {
  return graphData.nodes.filter(node => {
    // 1. Hierarchy filtering: hide children if parent collapsed
    const hierarchyNode = hierarchyMap.get(node.id);
    if (hierarchyNode && !hierarchyNode.isRoot) {
      if (!expansionState.isVisible(node.id, hierarchyNode.parentId)) {
        return false;
      }
    }

    // 2. Quality/Authority filtering (if enabled)
    if (filterEnabled) {
      let quality = node.metadata?.quality ?? computeFromConnections(node);
      let authority = node.metadata?.authority ?? computeFromHierarchy(node);

      const passesQuality = !filterByQuality || quality >= qualityThreshold;
      const passesAuthority = !filterByAuthority || authority >= authorityThreshold;

      return filterMode === 'and'
        ? passesQuality && passesAuthority
        : passesQuality || passesAuthority;
    }
    return true;
  });
}, [graphData.nodes, hierarchyMap, expansionState, filterEnabled]);
```

**Key Insight**: Physics still uses **ALL** `graphData.nodes`, but rendering only displays `visibleNodes`. This allows client-side filtering without affecting the force-directed layout.

**Performance**:
- **Hierarchy Detection**: 45ms for 10,000 nodes (runs once on data load)
- **Visibility Filtering**: 12-18ms per frame (runs every render)
- **Optimization**: Memoized with `useMemo`, only recomputes when dependencies change

---

## XR/AR Integration

### Device Detection

**File**: `client/src/app/App.tsx`

```typescript
const shouldUseImmersiveClient = () => {
  const userAgent = navigator.userAgent;

  const isQuest3Browser = userAgent.includes('Quest 3') ||
                          userAgent.includes('Quest3') ||
                          userAgent.includes('OculusBrowser') ||
                          (userAgent.includes('VR') && userAgent.includes('Quest'));

  const forceQuest3 = window.location.search.includes('force=quest3') ||
                      window.location.search.includes('directar=true') ||
                      window.location.search.includes('immersive=true');

  return (isQuest3Browser || forceQuest3) && initialized;
};
```

**⚠️ Known Issue**: User-agent sniffing is fragile. Quest 3 browser reports different strings depending on firmware version:
- v55-v57: `Quest 3` or `Quest3`
- v58+: `OculusBrowser` + `VR`
- **Failure Case**: Some Quest 2 browsers also report `OculusBrowser`, causing incorrect detection

**Mitigation**: URL parameter `?force=quest3` allows manual override.

### Babylon.js XR Scene

**File**: `client/src/immersive/components/ImmersiveApp.tsx` (212 lines)

```typescript
export const ImmersiveApp: React.FC<ImmersiveAppProps> = ({ onExit }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [babylonScene, setBabylonScene] = useState<BabylonScene | null>(null);

  useEffect(() => {
    // Initialize Babylon.js scene
    const scene = new BabylonScene(canvasRef.current);
    setBabylonScene(scene);

    // Enable XR
    scene.enableXR(); // Async, requires user gesture

    // Start render loop
    scene.run();
  }, []);

  // Update graph data
  useEffect(() => {
    if (babylonScene && graphData) {
      babylonScene.setBotsData({
        graphData: graphData,
        nodePositions: nodePositions,
        nodes: graphData.nodes,
        edges: graphData.edges
      });
    }
  }, [babylonScene, graphData, nodePositions]);

  return <canvas ref={canvasRef} className="immersive-canvas" />;
};
```

**Why Babylon.js instead of Three.js for XR?**
- **Better WebXR support**: Native XR APIs integration
- **Controller handling**: Built-in hand tracking and controller input
- **Performance**: Optimized for VR rendering loops (foveated rendering, reprojection)

**⚠️ Technical Debt**: Maintaining two rendering engines (Three.js + Babylon.js) increases bundle size and complexity.

**Bundle Impact**:
- Three.js: 850KB (gzipped)
- Babylon.js: 1.2MB (gzipped)
- **Total**: 2.05MB just for 3D engines

**Future Consideration**: Migrate to unified engine (Three.js + WebXR polyfill) or separate XR-only build.

---

## Performance Benchmarks

### Rendering Performance (RTX 3080)

| Metric | 1,000 Nodes | 10,000 Nodes | 50,000 Nodes |
|--------|-------------|--------------|--------------|
| **Frame Rate** | 60 FPS | 60 FPS | 42 FPS |
| **Draw Calls** | 1 | 1 | 1 |
| **GPU VRAM** | 120MB | 450MB | 1.8GB |
| **CPU (Main Thread)** | 8% | 15% | 35% |
| **CPU (Worker)** | 5% | 12% | 28% |

### State Update Performance

| Operation | Essential Paths | Full Load | 10 Settings | Single Setting |
|-----------|----------------|-----------|-------------|----------------|
| **Initial Load** | 203ms | 1,650ms | - | - |
| **Lazy Section Load** | - | - | 89ms | - |
| **Update (Local)** | - | - | 18ms | 12ms |
| **Update (Server)** | - | - | 67ms | 45ms |

### WebSocket Bandwidth

| Data Type | JSON | Binary | Savings |
|-----------|------|--------|---------|
| **Node Update** | 340 bytes | 34 bytes | 90% |
| **10k Nodes** | 3.4MB | 340KB | 90% |
| **Update Rate (50/sec)** | 175KB/s | 17.5KB/s | 90% |

---

## Known Issues & Limitations

### ⚠️ Quest 3 Detection Fragility

**Problem**: User-agent string varies across firmware versions.

**Impact**: Some users land on desktop client instead of immersive mode.

**Workaround**: URL parameter `?force=quest3`

**Proposed Fix**: Check for `navigator.xr.isSessionSupported('immersive-vr')` instead of user-agent.

### ⚠️ Dual Rendering Engine Complexity

**Problem**: Three.js (desktop) + Babylon.js (XR) increases bundle size and maintenance.

**Impact**:
- 2.05MB 3D engine code in bundle
- Duplicate scene graph implementations
- Different shader systems

**Future Improvement**: Migrate to Three.js + WebXR polyfill, or separate XR-only build with code-splitting.

### ⚠️ Binary Protocol Version Confusion

**Problem**: No protocol versioning in binary messages.

**Impact**: Server/client version mismatch causes silent data corruption.

**Proposed Fix**: Add 1-byte protocol version header to binary messages.

### ⚠️ Performance Bottleneck: Billboard Labels

**Problem**: 10,000 text billboards degrade performance to 30 FPS.

**Cause**: Each `<Text>` component from `@react-three/drei` triggers separate draw call.

**Solution**: Implement LOD system for labels (only show nearby nodes) or use texture atlas with instanced rendering.

### ⚠️ SSSP Coloring Performance

**Problem**: Recalculating 10,000 node colors every frame when SSSP result changes.

**Cause**: `getNodeColor()` runs in `visibleNodes.map()` which executes every render.

**Solution**: Cache SSSP color results, only update on path change.

---

## Technical Debt

### 1. Voice System Fragmentation

**Issue**: Legacy `useVoiceInteraction` hook coexists with new `VoiceProvider` pattern.

**Files**:
- `client/src/hooks/useVoiceInteraction.ts` (legacy)
- `client/src/features/voice/VoiceProvider.tsx` (new)

**Impact**: Duplicate audio processing pipelines, unclear ownership.

**Recommendation**: Migrate all voice features to `VoiceProvider` + specialized hooks.

### 2. Disabled Test Suite

**Issue**: 934+ tests disabled due to supply chain security concerns.

**Files**: `client/src/services/api/__tests__/*.test.ts`

**Impact**: No automated validation of API clients.

**Recommendation**: Re-enable tests with safe mocking strategy (no external dependencies).

### 3. Settings Store Complexity

**Issue**: 1,070 lines in single file, mixing lazy loading, auto-save, subscriptions, and persistence.

**Impact**: Hard to maintain, test, and extend.

**Recommendation**: Extract into separate modules:
- `settingsStore.ts` (core store)
- `lazyLoading.ts` (lazy loading logic)
- `autoSave.ts` (auto-save manager)
- `subscriptions.ts` (path-based subscriptions)

---

## Future Improvements

### WebGPU Migration

**Motivation**: WebGL is reaching performance limits with 50k+ nodes.

**Benefits**:
- Compute shaders for physics (10x faster than Web Worker)
- Better memory management (typed buffer pools)
- Modern GPU features (indirect rendering, mesh shaders)

**Compatibility**: Chrome 113+, Safari 18+, Firefox (coming 2025)

**Migration Path**:
1. Implement WebGPU renderer alongside WebGL (feature detection)
2. Migrate physics simulation to compute shaders
3. Deprecate Web Worker approach

### Unified XR Engine

**Motivation**: Eliminate dual rendering engine complexity.

**Approach**:
1. Use Three.js for both desktop and XR
2. Add WebXR polyfill for controller handling
3. Implement foveated rendering for VR performance

**Bundle Impact**: -1.2MB (remove Babylon.js)

### Edge Rendering Optimization

**Current**: 50,000 edges = 50,000 `<Line>` components (45 FPS)

**Proposed**: Single `BufferGeometry` with all edges batched (60 FPS)

```typescript
// Batched edge rendering
const edgeGeometry = useMemo(() => {
  const positions = new Float32Array(edges.length * 6); // 2 vertices × 3 coords
  edges.forEach((edge, i) => {
    const i6 = i * 6;
    positions[i6] = edge.source.x;
    positions[i6 + 1] = edge.source.y;
    positions[i6 + 2] = edge.source.z;
    positions[i6 + 3] = edge.target.x;
    positions[i6 + 4] = edge.target.y;
    positions[i6 + 5] = edge.target.z;
  });
  return new THREE.BufferGeometry().setAttribute(
    'position',
    new THREE.BufferAttribute(positions, 3)
  );
}, [edges]);

<lineSegments geometry={edgeGeometry}>
  <lineBasicMaterial color="#ffffff" opacity={0.3} transparent />
</lineSegments>
```

---

## Integration Points

### Server Communication

**REST APIs**:
- `GET /api/settings/{path}` - Fetch single setting
- `POST /api/settings/batch/get` - Fetch multiple settings (lazy loading)
- `POST /api/settings/batch/update` - Batch update settings (auto-save)
- `GET /api/analytics/sssp` - Run shortest path algorithm
- `GET /api/analytics/clustering` - Run semantic clustering

**WebSocket Messages**:
- `initialGraphLoad` - Filtered graph data from server
- `filter_confirmed` - Server confirms filter application
- `filter_update` - Client sends filter settings to server
- `binary_node_data` - Binary position updates (34 bytes/node)

### Authentication

**Nostr Protocol**:
- Client uses `nostrAuthService` for authentication
- Session token stored in `localStorage`
- Token included in all API requests: `Authorization: Bearer <token>`
- WebSocket connection authenticated via initial handshake

---

## Development & Debugging

### Debug Tools

```typescript
// Enable debug logging
localStorage.setItem('visionflow:debug', 'true');

// Access debug state
import { debugState } from '../utils/clientDebugState';
debugState.enablePerformanceDebug = true;

// Monitor WebSocket traffic
webSocketService.getStatistics();
// {
//   messagesReceived: 12847,
//   messagesSent: 234,
//   bytesReceived: 4237890,
//   bytesSent: 18293,
//   averageLatency: 42ms
// }
```

### Performance Profiling

```typescript
// Three.js renderer stats
import { StatsGl } from '@react-three/drei';

<Canvas>
  <StatsGl />  {/* FPS, MS, MB */}
  <GraphManager />
</Canvas>
```

### Browser DevTools

**Chrome Performance Tab**:
1. Record → Interact with graph → Stop
2. Look for:
   - Long tasks (>50ms) in main thread
   - GC pauses (yellow spikes)
   - Renderer frame drops (red triangles)

**Memory Profiling**:
1. Chrome DevTools → Memory → Heap Snapshot
2. Look for:
   - `THREE.BufferGeometry` instances (should be ~10)
   - `Float32Array` allocations (should be reused)
   - Detached DOM nodes (memory leaks)

---

---

---

## Related Documentation

- [VisionFlow Client Architecture - Deep Analysis](../../../archive/analysis/client-architecture-analysis-2025-12.md)
- [XR/VR System Architecture - Complete Documentation](../../../diagrams/client/xr/xr-architecture-complete.md)
- [ASCII Diagram Deprecation Audit](../../../audits/ascii-diagram-deprecation-audit.md)
- [VisionFlow Architecture Cross-Reference Matrix](../../../diagrams/cross-reference-matrix.md)
- [ComfyUI MCP Server Integration with Management API](../../../comfyui-integration-design.md)

## File Organization

```
client/src/
├── app/                      # Application bootstrap (190 LOC)
│   ├── App.tsx              # Auth + device detection
│   ├── main.tsx             # Vite entry point
│   └── MainLayout.tsx       # Desktop layout
│
├── features/                # Feature modules (77,549 LOC total)
│   ├── graph/              # Core graph visualization (1,047 LOC)
│   │   ├── components/     # GraphManager, GraphCanvas, FlowingEdges
│   │   ├── managers/       # graphDataManager, graphWorkerProxy
│   │   ├── workers/        # Web Worker physics simulation
│   │   └── hooks/          # useGraphEventHandlers, useExpansionState
│   │
│   ├── visualisation/      # Rendering effects (32 files)
│   │   ├── components/     # HolographicDataSphere, MetadataVisualizer
│   │   ├── effects/        # AtmosphericGlow, SelectiveBloom
│   │   └── controls/       # SpacePilotController, camera controls
│   │
│   ├── bots/              # Agent visualization (18 files)
│   ├── settings/          # Settings UI (15 files)
│   ├── analytics/         # Graph analytics (12 files)
│   └── design-system/     # Reusable UI (24 files)
│
├── services/              # Core services (3,000+ LOC)
│   ├── WebSocketService.ts           # WebSocket manager (1,407 LOC)
│   ├── BinaryWebSocketProtocol.ts    # Binary protocol handler
│   └── nostrAuthService.ts           # Nostr authentication
│
├── store/                 # Global state (1,200+ LOC)
│   ├── settingsStore.ts   # Zustand settings (1,070 LOC)
│   └── autoSaveManager.ts # Debounced persistence
│
├── rendering/             # Custom Three.js (500+ LOC)
│   ├── materials/         # HologramNodeMaterial.ts (313 LOC)
│   └── SelectiveBloom.tsx # Post-processing (187 LOC)
│
├── immersive/             # WebXR + Babylon.js
│   ├── babylon/           # Babylon.js scene management
│   └── components/        # ImmersiveApp (212 LOC)
│
└── types/                 # TypeScript definitions
    ├── binaryProtocol.ts  # 34-byte node format
    └── websocketTypes.ts  # WebSocket message types
```

---
