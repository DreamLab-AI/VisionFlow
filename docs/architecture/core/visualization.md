# VisionFlow Visualisation Architecture

## Overview

VisionFlow supports dual-graph visualisation where both **Knowledge Graphs** (Logseq) and **Agent Graphs** (VisionFlow bots) coexist in the same 3D scene. This document outlines the existing architecture showing how both graph types are rendered and what infrastructure is shared vs distinct.

## Core Architecture

### 1. Main Canvas Component (`GraphCanvas.tsx`)

**Location**: `/workspace/ext/client/src/features/graph/components/GraphCanvas.tsx`

**Purpose**: Central rendering container that manages both graph types in the same Three.js scene.

**Key Features**:
- Single R3F Canvas component with unified camera and lighting
- Contains both `GraphManager` (knowledge) and `BotsVisualization` (agents)
- Shared post-processing effects (SelectiveBloom)
- Common controls (OrbitControls, SpacePilot integration)
- Holographic environment shared by both graphs

```tsx
// Both graphs render in the same scene
<GraphManager graphData={graphData} />           // Knowledge graph
<BotsVisualization />                            // Agent graph
<SelectiveBloom enabled={enableBloom} />         // Shared effects
```

## 2. Knowledge Graph Visualisation

### Components
- **GraphManager** (`features/graph/components/GraphManager.tsx`)
- **MetadataShapes** (`features/graph/components/MetadataShapes.tsx`)
- **FlowingEdges** (`features/graph/components/FlowingEdges.tsx`)

### Data Structure
```typescript
interface GraphNode {
  id: string;
  label: string;
  position: { x: number; y: number; z: number };
  metadata?: Record<string, any>; // Type, size, description, etc.
  graphType?: 'logseq' | 'visionflow';
}
```

### Visual Characteristics
- **Node Types**: File, folder, concept, todo, reference (different geometries)
- **Colours**: Type-based (gold folders, turquoise files, etc.)
- **Shapes**: Spheres, cubes, octahedrons, cones, torus based on metadata
- **Materials**: HologramNodeMaterial with emissive glow
- **Scaling**: Connection-based and type-importance scaling
- **Labels**: Billboard text with metadata display
- **SSSP Visualisation**: Colour gradient from green (close) to red (far) from source node

### Data Flow
1. `graphDataManager` → fetches from `/api/graph/data`
2. `graphWorkerProxy` → handles physics simulation
3. Binary position updates via WebSocket
4. Real-time updates to Three.js meshes

## 3. Agent Graph Visualisation

### Components
- **BotsVisualization** (`features/bots/components/BotsVisualizationFixed.tsx`)
- **BotsNode** (internal component)
- **BotsEdgeComponent** (internal component)

### Data Structure
```typescript
interface BotsAgent {
  id: string;
  type: 'coordinator' | 'researcher' | 'coder' | 'analyst' | /* +15 more types */;
  status: 'idle' | 'busy' | 'active' | 'error' | 'initialising' | 'terminating' | 'offline';
  health: number;        // 0-100
  cpuUsage: number;      // percentage
  memoryUsage: number;   // percentage
  tokens?: number;       // token usage
  tokenRate?: number;    // tokens per minute
  position?: { x: number; y: number; z: number };
  capabilities?: string[];
  currentTask?: string;
  swarmId?: string;
}
```

### Visual Characteristics
- **Dynamic Sizing**: Based on workload, CPU usage, token rate, activity
- **Health-based Colours**: Green (healthy) to red (critical)
- **Status-based Geometry**:
  - Error: Tetrahedron (sharp edges)
  - Busy: Complex shapes (Queen = Icosahedron, Coordinator = Dodecahedron)
  - Active: High-poly spheres
  - Idle: Low-poly spheres
- **Agent Type Colours**:
  - Coordination agents: Gold/orange palette
  - Development agents: Green palette
  - Special roles: Purple/blue/red palette
- **Animations**:
  - Pulsing based on token rate and health
  - Rotation for busy agents
  - High-activity floating/vibration
  - Memory pressure shake effects
- **Status Badges**: HTML overlays with comprehensive metrics
- **Display Modes**: Overview, Performance, Tasks, Network, Resources (clickable cycling)

### Data Flow
1. `BotsDataContext` → aggregates from polling service
2. `useBotsData()` → provides context to visualisation
3. Server-authoritative positions via context updates
4. Real-time metrics from WebSocket integration

## 4. Shared Visualisation Infrastructure

### Three.js/R3F Framework
- **Canvas**: Single R3F Canvas component
- **Camera**: Shared perspective camera (fov: 75, position: [20, 15, 20])
- **Lighting**: Ambient + directional lighting
- **Controls**: OrbitControls + SpacePilot integration

### Post-Processing Pipeline (`rendering/SelectiveBloom.tsx`)
- **Selective Bloom**: Layer-based bloom effects
- **Layer Configuration**:
  - Layer 0: Default rendering and raycasting
  - Layer 1: Node bloom effects
  - Layer 2: Environment/hologram effects
- **Bloom Strength**: Configurable via settings for both graph types

### Materials System
- **HologramNodeMaterial**: Custom shader material with:
  - Emissive glow effects
  - Time-based animations
  - Instance colour support
  - Hologram rim lighting
- **Standard Materials**: For non-holographic elements

### Binary Protocol Support (`types/binaryProtocol.ts`)
```typescript
interface BinaryNodeData {
  nodeId: number;        // u16 with flags
  position: Vec3;        // 3x f32
  velocity: Vec3;        // 3x f32
  ssspDistance: number;  // f32
  ssspParent: number;    // i32
}
// Total: 34 bytes per node
```

**Node Type Flags**:
- `AGENT_NODE_FLAG = 0x8000`: Bit 15 for agent nodes
- `KNOWLEDGE_NODE_FLAG = 0x4000`: Bit 14 for knowledge nodes
- `NODE_ID_MASK = 0x3FFF`: Actual ID in bits 0-13

## 5. Graph Coexistence Strategy

### Position Space Separation
- **Knowledge Graph**: Physics-simulated positions via worker
- **Agent Graph**: Server-calculated positions via context
- **Coordinate System**: Shared 3D space, different regions

### Data Management
- **GraphDataManager**: Handles knowledge graph data and binary updates
- **BotsDataContext**: Handles agent graph data via polling
- **Isolation**: Separate data flows prevent mixing

### Performance Optimizations
- **Instanced Meshes**: Knowledge graph uses InstancedMesh for nodes
- **Individual Meshes**: Agent graph uses separate meshes for complex animations
- **Frustum Culling**: Disabled for knowledge, enabled for agents
- **Layer-based Rendering**: Separate bloom layers for different effects

## 6. Control Center Options (Analytics Components)

### Semantic Clustering Controls (`features/analytics/components/SemanticClusteringControls.tsx`)

**Existing Capabilities**:
- **Clustering Methods**: Spectral, Hierarchical, DBSCAN, K-Means++, Louvain, Affinity Propagation
- **GPU Acceleration**: Most algorithms GPU-accelerated
- **Parameters**: Configurable similarity metrics, cluster counts, convergence thresholds
- **Results View**: Cluster visualisation with coherence metrics and keywords
- **Anomaly Detection**: Real-time outlier identification with multiple methods
- **Advanced Analytics**: UMAP reduction, Graph Wavelets, Hyperbolic Embedding, Persistent Homology

### Shortest Path Controls (`features/analytics/components/ShortestPathControls.tsx`)

**Existing Capabilities**:
- **SSSP Algorithms**: Dijkstra, Bellman-Ford, Floyd-Warshall
- **Visualisation**: Colour-coded distance visualisation (green=close, red=far, grey=unreachable)
- **Source Selection**: Interactive source node selection
- **Distance Normalization**: 0-1 normalised distance display
- **Path Reconstruction**: Via predecessor tracking
- **Performance Metrics**: Computation time, cache hit rates
- **Real-time Updates**: Live shortest path visualisation on graph

### Node Distribution Options

**Current Analytics Options**:
1. **Clustering-based Distribution**:
   - Spectral clustering for community detection
   - Hierarchical clustering for tree-like organisation
   - DBSCAN for density-based grouping

2. **Path-based Distribution**:
   - SSSP visualisation with distance-based colouring
   - Source node selection for path analysis
   - Reachability analysis

3. **Advanced Spatial Layouts**:
   - UMAP dimensionality reduction for 2D/3D projection
   - Hyperbolic embedding for hierarchical structures
   - Graph wavelet analysis for multi-scale organisation

4. **Anomaly-based Highlighting**:
   - Isolation Forest for structural anomalies
   - Local Outlier Factor for density-based outliers
   - Statistical methods for metric-based outliers

## 7. Key Differences: Knowledge vs Agent Graphs

| Aspect | Knowledge Graph | Agent Graph |
|--------|----------------|-------------|
| **Data Source** | `/api/graph/data` + WebSocket binary | Polling service + context |
| **Node Types** | File, folder, concept, todo, reference | 17+ agent types (coordinator, coder, etc.) |
| **Visual Basis** | Metadata type and connections | Real-time metrics (health, CPU, tokens) |
| **Positioning** | Physics simulation in worker | Server-authoritative |
| **Materials** | HologramNodeMaterial with type colours | Dynamic materials with status colours |
| **Animations** | Physics-based movement, SSSP pulsing | Status-based (pulsing, rotation, shake) |
| **Interactivity** | Drag & drop, SSSP source selection | Click for mode cycling, hover for details |
| **Labels** | Type-based with metadata | Dynamic with performance metrics |
| **Clustering** | Semantic clustering via analytics | Health/performance-based grouping |

## 8. Shared vs Distinct Infrastructure

### Shared Infrastructure
- ✅ Three.js Canvas and scene
- ✅ Camera and lighting setup
- ✅ Post-processing effects (SelectiveBloom)
- ✅ Control systems (OrbitControls, SpacePilot)
- ✅ Settings management
- ✅ WebSocket service (different message types)
- ✅ Performance monitoring
- ✅ Holographic environment

### Distinct Infrastructure
- ❌ Data managers (GraphDataManager vs BotsDataContext)
- ❌ Material systems (hologram vs status-based)
- ❌ Animation systems (physics vs metrics-based)
- ❌ Layout algorithms (force-directed vs server-computed)
- ❌ Interaction patterns (drag vs click-cycle)
- ❌ Binary protocol usage (knowledge only)
- ❌ Analytics integration (knowledge graph only)

## 9. Future Enhancement Opportunities

Based on the existing architecture, potential enhancements could include:

1. **Agent Graph Analytics**: Extend SSSP and clustering to agent relationships
2. **Cross-Graph Interactions**: Visual connections between knowledge and agent nodes
3. **Unified Binary Protocol**: Extend binary updates to agent positions
4. **Agent Distribution Controls**: Apply semantic clustering to agent swarms
5. **Performance Correlation**: Visual links between code nodes and executing agents
6. **Unified Material System**: Shared holographic effects for both graph types

## Conclusion

VisionFlow successfully implements a sophisticated dual-graph visualisation system where both knowledge and agent graphs coexist in the same 3D space. The architecture cleanly separates concerns while sharing core Three.js infrastructure, enabling rich analytics for knowledge graphs while providing real-time monitoring for agent systems. The existing control center provides extensive analytics capabilities for knowledge graphs, with opportunities to extend similar capabilities to agent graphs.