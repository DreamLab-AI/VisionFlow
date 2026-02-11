# AFD: Server-Authoritative Layout with Client Optimistic Tweening

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                      RUST SERVER (Single Source of Truth)            │
│                                                                      │
│  PhysicsSupervisor → ForceComputeActor → StressMajorizationActor   │
│                   → SemanticForcesActor → ClusteringActor           │
│                   → OntologyConstraintActor → PagerankActor         │
│                                                                      │
│  Output: BinaryNodeData[] → Binary Protocol V3/V4                   │
│  Rate: 10-60 Hz (adaptive, motion-gated)                            │
└─────────────────────────┬───────────────────────────────────────────┘
                          │ WebSocket /wss (binary frames)
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      CLIENT (Optimistic Tweening Only)              │
│                                                                      │
│  GraphDataManager → graph.worker.ts (Web Worker via Comlink)       │
│                                                                      │
│  Worker responsibilities (AFTER refactor):                          │
│    ✅ Receive binary position updates from server                   │
│    ✅ Set targetPositions[] from server data                        │
│    ✅ Interpolate currentPositions[] → targetPositions[] (tweening) │
│    ✅ Sync to SharedArrayBuffer for main-thread reads               │
│    ✅ Handle pinned nodes (user drag)                               │
│    ❌ NO force computation (computeForces removed)                  │
│    ❌ NO physics simulation (applyForces removed)                   │
│    ❌ NO alpha decay / temperature model                            │
│                                                                      │
│  Interpolation formula (existing, proven):                          │
│    lerpFactor = 1 - Math.pow(0.001, deltaTime)                     │
│    currentPos += (targetPos - currentPos) * lerpFactor             │
│    snap when distance < 5.0 units                                   │
└─────────────────────────────────────────────────────────────────────┘
```

## 2. Functional Changes

### 2.1 graph.worker.ts Refactoring

**Remove**: `computeForces()`, `applyForces()`, `forcePhysics` settings object, alpha/temperature model, domain clustering force computation, spatial grid repulsion, edge spring forces, center gravity forces.

**Keep**: `tick()` interpolation path (the `useServerPhysics` branch), `processBinaryData()`, `setGraphData()`, `updateUserDrivenNodePosition()`, `pinNode()`/`unpinNode()`.

**Change**: `setGraphType()` no longer toggles `useServerPhysics`. All graph types use server physics. The `visionflow` branch that sets `this.useServerPhysics = false` is removed.

### 2.2 User Drag Interaction Flow

```
User drags node (client)
    → updateUserDrivenNodePosition() sets local position immediately (optimistic)
    → Client sends drag position to server via REST/WebSocket
    → Server applies position as constraint, runs physics tick
    → Server broadcasts updated positions (including dragged node's final position)
    → Client receives and merges (snap or interpolate)
```

### 2.3 Settings Flow

Physics settings in the control panel (`GraphOptimisationTab`) send parameter changes to the server via `PUT /api/physics/parameters`. The server applies them to the GPU physics pipeline. Client settings only control:
- Interpolation speed (lerpFactor base)
- Snap threshold distance
- Visual quality (material parameters)
- LOD thresholds (for VR/Quest 3)

### 2.4 Vircadia Parity

Vircadia clients connect to the same WebSocket `/wss` endpoint. The `CollaborativeGraphSync` maps Vircadia entities to graph nodes using `GraphEntityMapper`. Position updates flow:

```
Server physics → /wss binary broadcast → Desktop client (tweening)
                                        → Vircadia client (entity sync)
                                        → Quest 3 client (VR tweening)
```

All clients receive identical position data. Each applies its own tweening for smooth local display.

## 3. File Impact Matrix

| File | Action | Scope |
|------|--------|-------|
| `client/src/features/graph/workers/graph.worker.ts` | MODIFY | Remove force physics, unify on server physics |
| `client/src/features/graph/managers/graphDataManager.ts` | MODIFY | Remove visionflow/logseq graph type branching for physics |
| `client/src/features/graph/managers/graphWorkerProxy.ts` | MINOR | Remove exposed force physics settings methods |
| `client/src/features/settings/config/settings.ts` | MODIFY | Route physics settings to server API |
| `client/src/features/visualisation/components/ControlPanel/GraphOptimisationTab.tsx` | MODIFY | Connect to server physics API |
| `client/src/services/vircadia/CollaborativeGraphSync.ts` | AUDIT | Verify position sync uses server data |
| `client/src/features/visualisation/WebXRScene.tsx` | AUDIT | Verify Quest 3 uses server positions |

## 4. Risk Analysis

| Risk | Mitigation |
|------|-----------|
| Server unavailable → no physics | Keep minimal client-side fallback (simple spring interpolation only) |
| High latency → jerky motion | Adaptive lerpFactor based on ping time |
| Drag feels laggy | Optimistic local update + server reconciliation |
| Breaking existing logseq flow | logseq already uses server physics — no change needed |
