# PRD: WebGPU TSL Graph Interface Audit & Hardening

## 1. Problem Statement

VisionFlow's WebGPU/TSL graphical interface requires a comprehensive audit to ensure:
- High-end experimental WebGPU shader features are the default rendering path
- Graceful fallback to WebGL when WebGPU is unavailable
- Server-side position calculation is the single source of truth
- Clients perform optimistic tweening only (no authoritative physics)
- Quest 3 / Vircadia parity with the desktop interface

## 2. Current State Analysis

### Architecture Dual-Physics Problem
The codebase currently has **two independent physics engines**:
1. **Server-side (Rust/CUDA)**: GPU-accelerated force-directed layout with stress majorization, semantic forces, ontology constraints, and clustering (`src/actors/gpu/`, `src/physics/`)
2. **Client-side (Web Worker)**: Full force-directed simulation with repulsion, attraction, gravity, and clustering (`client/src/features/graph/workers/graph.worker.ts`)

The client worker conditionally uses its own physics for `visionflow` graph type (line 179: `this.useServerPhysics = false`) and server physics for `logseq` type. This creates two sources of truth and divergent layouts.

### Rendering Pipeline Status
- **WebGPU path**: TSL metadata-driven materials with per-instance data textures, Fresnel rim lighting, authority-driven pulses, quality-driven emissive glow
- **WebGL path**: MeshPhysicalMaterial with transmission, standard uniforms
- **Fallback**: Clean WebGLRenderer when `navigator.gpu` absent; hybrid WebGPU+WebGLBackend correctly detected and rejected
- **Post-processing**: Dual bloom paths (WebGPU node-based vs WebGL EffectComposer)

### Quest 3 / Vircadia Status
- WebXR scene with LOD thresholds and reduced detail for VR
- Vircadia integration via WebSocket with entity sync, but graph sync is incomplete
- No shared position authority between Vircadia and desktop clients

## 3. Requirements

### R1: Server-Authoritative Layout (Single Source of Truth)
- **R1.1**: All node position calculations MUST happen on the server (Rust GPU physics)
- **R1.2**: Client graph worker MUST only perform optimistic interpolation/tweening toward server positions
- **R1.3**: Remove or gate client-side force-directed physics (currently in `computeForces()`, `applyForces()`)
- **R1.4**: User drag interactions send position deltas to server; server applies and rebroadcasts
- **R1.5**: Binary protocol V3/V4 remains the transport for position updates

### R2: WebGPU High-End Feature Set (Default Path)
- **R2.1**: TSL metadata-driven materials are the default when WebGPU is available
- **R2.2**: Per-instance metadata texture pipeline for quality/authority/connections/recency
- **R2.3**: Fresnel rim lighting with authority-driven pulse animation
- **R2.4**: Node-based PostProcessing bloom (not EffectComposer)
- **R2.5**: Environment-mapped reflections with HDR IBL
- **R2.6**: Iridescence, clearcoat, and sheen on all material types
- **R2.7**: Animated edge flow effects via TSL time-driven nodes

### R3: Graceful WebGL Fallback
- **R3.1**: Feature detection via `navigator.gpu` + backend verification (already working)
- **R3.2**: WebGL materials use MeshPhysicalMaterial with transmission (already working)
- **R3.3**: Bloom falls back to UnrealBloomPass (already working)
- **R3.4**: No visual regression -- WebGL must still render all node/edge types
- **R3.5**: Performance parity -- WebGL path must not introduce extra draw calls

### R4: Quest 3 / Vircadia Parity
- **R4.1**: Vircadia clients receive the same server-authoritative positions
- **R4.2**: Quest 3 LOD system mirrors desktop quality presets
- **R4.3**: Entity sync maps 1:1 with graph nodes/edges
- **R4.4**: Avatar rendering integrates with graph visualization
- **R4.5**: Optimistic tweening on Quest 3 matches desktop interpolation

### R5: Settings & Control Panel
- **R5.1**: Unified settings panel exposes all WebGPU/WebGL material parameters
- **R5.2**: Physics settings control server-side parameters (not client-side)
- **R5.3**: Quality presets (low/medium/high) map to concrete shader configurations
- **R5.4**: Real-time preview of material changes

## 4. Success Criteria

| Metric | Target |
|--------|--------|
| Node position divergence (server vs client) | < 1 unit after 1s |
| WebGPU material activation rate | 100% on supported browsers |
| WebGL fallback activation | < 500ms, no visual glitch |
| Quest 3 frame rate | >= 72 fps with 1000 nodes |
| Desktop frame rate | >= 60 fps with 10000 nodes |
| Vircadia position sync latency | < 100ms |

## 5. Out of Scope
- Changing the binary protocol format
- Modifying the Rust physics engine internals
- Adding new graph layout algorithms
- Changing the Neo4j data model
