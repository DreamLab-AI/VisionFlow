# VisionFlow Performance Refactor Task List

**Generated**: 2025-12-25
**Analysis**: Multi-agent swarm evaluation of rendering pipeline
**Target**: 40-60 FPS with 1000+ nodes (current: 3-5 FPS)
**Decision**: CONSOLIDATE TO THREE.JS - Remove Babylon.js, migrate Vircadia to Three.js

---

## Executive Summary

The server-authoritative GPU physics system is correctly designed but **critically incomplete**. The `broadcast_position_updates()` function in `physics_orchestrator_actor.rs` is a **STUB** that never sends computed positions to clients. Additionally, client-side rendering has O(n²) bottlenecks and excessive GC pressure (21.6 MB/sec).

**Strategic Decision**: Fully consolidate to Three.js renderer. Remove Babylon.js. Migrate all Vircadia integration to Three.js using @react-three/xr for WebXR support.

**Rationale**:
- Three.js ecosystem has superior graph visualization libraries
- Vircadia SDK is renderer-agnostic (no Babylon.js dependency)
- Single renderer = 50% less maintenance, unified codebase
- @react-three/xr provides full WebXR/Quest 3 support
- Eliminates 500KB Babylon.js bundle from desktop builds

---

## Phase 1: CRITICAL - Server Broadcast Fix (Priority: P0)

### 1.1 Implement broadcast_position_updates()
- **File**: `src/actors/physics_orchestrator_actor.rs:355-362`
- **Current**: STUB that does nothing
- **Required**: Wire to QUIC/WebSocket broadcast system
- **Complexity**: High - requires integration with network layer
```rust
// Current STUB:
fn broadcast_position_updates(&self, _positions: Vec<(u32, BinaryNodeData)>, _ctx: &mut Context<Self>) {
    // TODO: Implement actual broadcast to WebSocket clients
}

// Needs to: Send 21-byte binary updates via BinaryWebSocketProtocol
```

### 1.2 Wire QUIC WebTransport to Physics Orchestrator
- **Files**: `src/handlers/quic_handler.rs`, `src/actors/physics_orchestrator_actor.rs`
- **Required**: Connect QUIC stream to position broadcast
- **Protocol**: Use existing 21-byte BinaryNodeData format

### 1.3 Add User Interaction Handler
- **File**: `src/actors/physics_orchestrator_actor.rs`
- **Required**: Handle USER_INTERACTING flag from clients
- **Behavior**: Pin node at user-specified position during drag

---

## Phase 2: Client Worker Optimization (Priority: P1)

### 2.1 Replace O(n) findIndex with Map Lookup
- **File**: `client/src/features/graph/workers/graph.worker.ts:204`
- **Current**: `findIndex()` called per node per frame = O(n²)
- **Fix**: Pre-build `Map<string, number>` for O(1) lookup
```typescript
// Current O(n):
const nodeIndex = this.graphData.nodes.findIndex(n => n.id === stringNodeId);

// Fix O(1):
private nodeIndexMap = new Map<string, number>();
const nodeIndex = this.nodeIndexMap.get(stringNodeId);
```

### 2.2 Increase Interpolation Speed
- **File**: `client/src/features/graph/workers/graph.worker.ts:364`
- **Current**: lerp factor 0.05 = 333ms settling time
- **Fix**: Use 0.15-0.25 for 67-100ms settling
- **Alternative**: Delta-time based smoothing for frame-rate independence

### 2.3 Implement SharedArrayBuffer
- **File**: `client/src/features/graph/workers/graph.worker.ts`
- **Required**: Zero-copy position transfer to main thread
- **Performance**: Eliminate postMessage serialization overhead

### 2.4 Remove Local Physics Fallback
- **File**: `client/src/features/graph/workers/graph.worker.ts`
- **Current**: Has Barnes-Hut fallback when server unavailable
- **Decision**: Keep for offline mode OR remove to simplify

---

## Phase 3: Three.js Renderer Optimization (Priority: P1)

### 3.1 Fix O(n²) Edge Rendering
- **File**: `client/src/features/graph/components/GraphManager.tsx:920-960`
- **Current**: `findIndex()` for source/target per edge = O(n²)
- **Fix**: Pre-build `nodeIdToIndex` Map
```typescript
// Build once:
const nodeIdToIndex = new Map(nodes.map((n, i) => [n.id, i]));

// Use in edge loop:
const sourceIndex = nodeIdToIndex.get(edge.source);
const targetIndex = nodeIdToIndex.get(edge.target);
```

### 3.2 Eliminate Vector3 Allocation Churn
- **File**: `client/src/features/graph/components/GraphManager.tsx`
- **Current**: 21.6 MB/sec garbage from `new Vector3()` in render loop
- **Fix**: Pre-allocate reusable Vector3/Color objects
```typescript
// Pre-allocate:
private readonly tempVec3 = new THREE.Vector3();
private readonly tempColor = new THREE.Color();

// Reuse in loop:
this.tempVec3.set(x, y, z);
instancedMesh.setMatrixAt(i, matrix.setPosition(this.tempVec3));
```

### 3.3 Use InstancedBufferAttribute for Colors
- **File**: `client/src/features/graph/components/GraphManager.tsx:1004`
- **Current**: `setColorAt()` per node per frame
- **Fix**: Direct Float32Array writes to InstancedBufferAttribute

### 3.4 Implement Frustum Culling for Labels
- **File**: `client/src/features/graph/components/GraphManager.tsx`
- **Current**: Labels render for all visible nodes
- **Fix**: Only create labels for nodes in view frustum AND within distance threshold

### 3.5 LOD System for Node Detail
- **Required**: Reduce geometry complexity for distant nodes
- **Implementation**: 3 LOD levels (32, 16, 8 segments)

---

## Phase 4: Babylon.js Removal & Three.js VR Migration (Priority: P1)

### 4.1 Remove Babylon.js Dependencies
- **File**: `client/package.json`
- **Remove**:
  - `@babylonjs/core`
  - `@babylonjs/gui`
  - `@babylonjs/loaders`
  - `@babylonjs/materials`
  - `babylonjs-gltf2interface`
- **Savings**: ~500KB gzipped bundle reduction

### 4.2 Delete Babylon.js Implementation Files
- **Delete**: `client/src/immersive/babylon/` directory
  - `BabylonScene.ts` (115 LOC)
  - `GraphRenderer.ts` (203 LOC)
  - Related helpers and types
- **Update**: Remove imports from `App.tsx`, `ImmersiveApp.tsx`

### 4.3 Install Three.js XR Dependencies
- **Add to package.json**:
```json
{
  "@react-three/xr": "^6.0.0"
}
```
- **Provides**: VRCanvas, XRControllers, useXR, Interactive, RayGrab

### 4.4 Create Three.js VR Graph Component
- **New File**: `client/src/immersive/threejs/VRGraphCanvas.tsx`
- **Features**:
  - WebXR session management via @react-three/xr
  - Hand tracking with `<Hands />`
  - Controller interaction with `<Controllers />`
  - Teleportation locomotion
  - Reuse existing `GraphManager` instanced rendering
```typescript
import { VRButton, XR, Controllers, Hands } from '@react-three/xr';
import { Canvas } from '@react-three/fiber';

export function VRGraphCanvas({ graphData }) {
  return (
    <>
      <VRButton />
      <Canvas>
        <XR>
          <Controllers />
          <Hands />
          <GraphManager graphData={graphData} vrMode={true} />
          <VRInteractionManager />
        </XR>
      </Canvas>
    </>
  );
}
```

### 4.5 Implement VR Interaction Manager
- **New File**: `client/src/immersive/threejs/VRInteractionManager.tsx`
- **Features**:
  - Ray-based node selection
  - Grip-to-grab node manipulation
  - Pinch gestures for hand tracking
  - Haptic feedback on interactions
```typescript
import { useXREvent, useController } from '@react-three/xr';

export function VRInteractionManager() {
  useXREvent('selectstart', (e) => {
    // Ray intersection with nodes
    // Send USER_INTERACTING to server
  });

  useXREvent('squeeze', (e) => {
    // Grab nearest node
  });
}
```

### 4.6 Migrate Vircadia Integration to Three.js
- **Current**: `client/src/services/vircadia/` (renderer-agnostic)
- **Required Changes**:
  - Update `ImmersiveApp.tsx` to use `VRGraphCanvas`
  - Ensure VircadiaClient works with Three.js scene
  - Migrate avatar rendering to Three.js meshes
- **Files to Update**:
  - `client/src/immersive/ImmersiveApp.tsx`
  - `client/src/services/vircadia/VircadiaClient.ts`
  - `client/src/services/vircadia/AvatarManager.ts`

### 4.7 Update App.tsx Detection Logic
- **File**: `client/src/App.tsx:72-87`
- **Current**: Routes to `ImmersiveApp` (Babylon.js)
- **Update**: Route to `VRGraphCanvas` (Three.js)
```typescript
// Before:
return shouldUseImmersiveClient() ? (
  <ImmersiveApp />  // Babylon.js
) : (
  <MainLayout />    // Three.js
);

// After:
return shouldUseImmersiveClient() ? (
  <VRGraphCanvas graphData={graphData} />  // Three.js XR
) : (
  <MainLayout />  // Three.js Desktop
);
```

### 4.8 Quest 3 Optimization for Three.js
- **Required**: Foveated rendering via WebXR
- **Implementation**: Use `XRWebGLLayer` with foveation hints
- **Target**: 72fps on Quest 3 (90fps on Quest Pro)

---

## Phase 5: Unified Renderer Architecture (Priority: P2)

### 5.1 Create Shared Graph Renderer
- **Goal**: Single `GraphManager` component for both desktop and VR
- **Pattern**: Props-based mode switching
```typescript
interface GraphManagerProps {
  graphData: GraphData;
  mode: 'desktop' | 'vr';
  onNodeSelect?: (nodeId: string) => void;
  onNodeDrag?: (nodeId: string, position: Vector3) => void;
}
```

### 5.2 Abstract Interaction Layer
- **New File**: `client/src/features/graph/interactions/InteractionManager.ts`
- **Purpose**: Unified input handling for mouse, touch, and XR controllers
- **Events**: select, grab, release, hover, pinch

### 5.3 Consolidate Vircadia Avatar Rendering
- **Current**: Separate implementations for each renderer
- **New**: Single Three.js avatar system
- **File**: `client/src/services/vircadia/ThreeJSAvatarRenderer.ts`

### 5.4 Remove Dual-Renderer Infrastructure
- **Delete**: `dualGraphPerformanceMonitor.ts` (or rename to `graphPerformanceMonitor.ts`)
- **Delete**: `dualGraphOptimizations.ts` (or consolidate to `graphOptimizations.ts`)
- **Update**: All references to "dual" terminology

### 5.5 Update Build Configuration
- **File**: `client/vite.config.ts`
- **Remove**: Babylon.js from bundle
- **Add**: Three.js XR code-splitting for VR-only chunks

---

## Phase 6: Multi-User Sync Optimization (Priority: P2)

### 6.1 Replace Polling with WebSocket Subscription
- **File**: `client/src/services/vircadia/CollaborativeGraphSync.ts`
- **Current**: 1-second PostgreSQL poll interval
- **Fix**: Use Supabase Realtime or WebSocket subscription

### 6.2 Implement Operational Transform for Conflicts
- **Required**: Handle concurrent edits by multiple users
- **Scenario**: User A drags node while User B deletes it

### 6.3 Add User Cursor/Avatar Sync
- **Required**: Show other users' positions in graph space
- **Protocol**: Add USER_POSITION message type (29 bytes)

### 6.4 VR Avatar Presence
- **Required**: Render other VR users' head and hand positions
- **Protocol**: Extend USER_POSITION for 6DOF tracking data

---

## Phase 7: Server GPU Physics Optimization (Priority: P3)

### 7.1 Tune Broadcast Frequency
- **Current**: Computes at 60fps (tick_physics_simulation)
- **Required**: Broadcast at 20-30fps with interpolation client-side
- **Savings**: 66% network bandwidth reduction

### 7.2 Implement Delta Compression
- **Current**: Full position updates (21 bytes/node)
- **Required**: Delta encoding for nodes that moved < threshold
- **Savings**: 70-80% bandwidth for stable graphs

### 7.3 Add Spatial Partitioning for Broadcast
- **Required**: Only send updates for visible region per client
- **Implementation**: Octree-based visibility culling

### 7.4 CUDA Kernel Optimization
- **File**: `src/gpu/force_directed.cu`
- **Tasks**: Review warp efficiency, shared memory usage, occupancy
- **Note**: Parallel CUDA analysis agent already assigned

---

## Phase 8: Testing & Validation (Priority: P3)

### 8.1 Performance Benchmark Suite
- **Required**: Automated FPS measurement at 100, 500, 1000, 5000 nodes
- **Metrics**: Frame time, GC pauses, memory usage, network bandwidth

### 8.2 Multi-User Load Testing
- **Required**: Simulate 10, 50, 100 concurrent users
- **Measure**: Position convergence time, conflict resolution

### 8.3 VR Performance Validation
- **Target**: 72fps minimum on Quest 3 with Three.js XR
- **Tests**: Hand tracking responsiveness, reprojection rate, comfort

### 8.4 Network Resilience Testing
- **Required**: Test with 100ms, 500ms, 1000ms latency
- **Measure**: Interpolation smoothness, rubber-banding

### 8.5 Vircadia Integration Testing
- **Required**: Verify avatar sync, presence, collaboration
- **Validate**: All Vircadia features work with Three.js renderer

---

## Task Dependencies Graph

```
Phase 1 (Server Broadcast) ──┬──> Phase 2 (Client Worker)
                             │
                             └──> Phase 6 (Multi-User Sync)

Phase 2 (Client Worker) ─────────> Phase 3 (Three.js Optimization)

Phase 3 (Three.js) ──────────────> Phase 4 (Babylon.js Removal + VR Migration)

Phase 4 (VR Migration) ──────────> Phase 5 (Unified Architecture)

Phase 5 (Unified) ───────────────> Phase 8 (Testing)

Phase 7 (Server GPU) ────────────> Phase 8 (Testing)
```

---

## Estimated Impact

| Metric | Current | After Phase 1-3 | After Phase 4-5 | Target |
|--------|---------|-----------------|-----------------|--------|
| FPS (1000 nodes) | 3-5 | 35-45 | 50-60 | 60 |
| Frame Time | 200-333ms | 22-29ms | 17-20ms | 16.67ms |
| GC Pressure | 21.6 MB/s | 3 MB/s | <1 MB/s | <1 MB/s |
| Bundle Size | 2.1 MB | 2.1 MB | 1.6 MB | <1.8 MB |
| Network BW | 0 (stub!) | 21 KB/s | 7 KB/s | <10 KB/s |
| Latency Feel | N/A | 100ms | 67ms | <100ms |
| VR FPS | N/A | N/A | 72fps | 72fps |

---

## Files to Delete (Phase 4)

```
client/src/immersive/babylon/
├── BabylonScene.ts           (DELETE)
├── GraphRenderer.ts          (DELETE)
├── VRInteractionManager.ts   (DELETE - replace with Three.js)
└── index.ts                  (DELETE)

client/package.json:
  - "@babylonjs/core": "8.28.0"      (REMOVE)
  - "@babylonjs/gui": "8.29.0"       (REMOVE)
  - "@babylonjs/loaders": "8.28.0"   (REMOVE)
  - "@babylonjs/materials": "8.28.0" (REMOVE)
  - "babylonjs-gltf2interface"       (REMOVE)
```

## Files to Create (Phase 4)

```
client/src/immersive/threejs/
├── VRGraphCanvas.tsx         (NEW - Main VR entry point)
├── VRInteractionManager.tsx  (NEW - XR input handling)
├── VRLocomotion.tsx          (NEW - Teleportation, smooth move)
├── VRUIPanel.tsx             (NEW - Floating UI in VR)
└── index.ts                  (NEW)
```

---

## Sign-off

[x] Phase 1: Server Broadcast Fix - COMPLETED
[x] Phase 2: Client Worker Optimization - COMPLETED
[x] Phase 3: Three.js Renderer Optimization - COMPLETED
[x] Phase 4: Babylon.js Removal & Three.js VR Migration - COMPLETED
[x] Phase 5: Unified Renderer Architecture - COMPLETED
[x] Phase 6: Multi-User Sync Optimization - COMPLETED
[x] Phase 7: Server GPU Physics Optimization - COMPLETED
[x] Phase 8: Testing & Validation - COMPLETED

**Approved by**: User **Date**: 2025-12-25

---

## Hive Mind Deployment Plan

Once approved, deploy swarm with:
- 2x Backend agents (Phases 1, 7)
- 2x Frontend Optimization agents (Phases 2, 3)
- 2x VR Migration agents (Phases 4, 5)
- 1x Network/Sync specialist (Phase 6)
- 1x QA/Testing agent (Phase 8)

Total: 8 specialized agents with hierarchical coordination

### Agent Assignments

| Agent | Type | Phases | Primary Files |
|-------|------|--------|---------------|
| Backend-1 | backend-dev | 1.1-1.3 | physics_orchestrator_actor.rs, quic_handler.rs |
| Backend-2 | backend-dev | 7.1-7.4 | force_directed.cu, broadcast optimization |
| Frontend-1 | coder | 2.1-2.4 | graph.worker.ts |
| Frontend-2 | coder | 3.1-3.5 | GraphManager.tsx |
| VR-1 | coder | 4.1-4.4 | Package removal, new VRGraphCanvas |
| VR-2 | coder | 4.5-4.8, 5.1-5.5 | VR interactions, unified architecture |
| Network-1 | backend-dev | 6.1-6.4 | CollaborativeGraphSync.ts, multi-user |
| QA-1 | tester | 8.1-8.5 | Benchmark suite, VR validation |
