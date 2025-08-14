# Parallel Graph Architecture

## Overview

VisionFlow's parallel graph architecture enables simultaneous visualisation and processing of two distinct graph types: Knowledge Graphs and Agent Graphs. This approach allows users to see both static knowledge structures and dynamic AI agent interactions in a unified 3D space using the unified CUDA kernel with parallel processing modes.

## Core Architecture

```mermaid
graph TB
    subgraph "Parallel Graph System"
        subgraph "Knowledge Graph"
            KNodes[Knowledge Nodes<br/>Logseq Data]
            KEdges[Semantic Edges]
            KMeta[Metadata]
            KLayout[Force-Directed Layout]
        end

        subgraph "Agent Graph"
            ANodes[Agent Nodes<br/>Claude Flow Agents]
            AEdges[Communication Edges]
            AState[Agent State]
            ALayout[multi-agent Layout]
        end

        subgraph "Unified Processing"
            Coordinator[ParallelGraphCoordinator]
            UnifiedKernel[Unified CUDA Kernel]
            Physics[Parallel Physics]
            Render[WebGL Renderer]
        end
    end

    KNodes --> Coordinator
    ANodes --> Coordinator
    Coordinator --> UnifiedKernel
    UnifiedKernel --> Physics
    Physics --> Render
```

## Graph Type Identification

The system maintains separate data structures and processing pipelines for each graph type:

```typescript
type GraphType = 'logseq' | 'visionflow';

// Each node knows its graph context
interface GraphNode {
  id: string;
  graphType: GraphType;
  position: Vector3;
  // ... other properties
}
```

## Graph Types

### Knowledge Graph

The Knowledge Graph represents:
- **Nodes**: Markdown documents, concepts, and knowledge entities
- **Edges**: Semantic relationships, references, and links
- **Source**: Logseq markdown files and metadata
- **Updates**: File system changes and manual edits
- **Physics**: Stable, slowly evolving layout

```typescript
interface KnowledgeNode {
  id: number;        // Bit 30 set (0x40000000)
  label: string;     // Document title or concept name
  content: string;   // Markdown content
  metadata: {
    tags: string[];
    created: Date;
    modified: Date;
    references: string[];
  };
}
```

### Agent Graph

The Agent Graph represents:
- **Nodes**: AI agents, coordinators, and specialists
- **Edges**: Communication channels and task dependencies
- **Source**: Claude Flow MCP orchestrator
- **Updates**: Real-time via WebSocket (100ms intervals)
- **Physics**: Dynamic, rapidly changing layout

```typescript
interface AgentNode {
  id: number;        // Bit 31 set (0x80000000)
  label: string;     // Agent name and role
  type: AgentType;   // coordinator, coder, researcher, etc.
  status: 'active' | 'idle' | 'busy';
  metrics: {
    cpu: number;
    memory: number;
    tasks: number;
    tokens: number;
  };
}
```

## Binary Protocol Type Flags

The system uses the high bits of the 32-bit node ID to distinguish between graph types:

```rust
// Node ID structure (32 bits)
// Bit 31: Agent flag (1 = agent node)
// Bit 30: Knowledge flag (1 = knowledge node)
// Bits 0-29: Actual node ID

const AGENT_NODE_FLAG: u32 = 0x80000000;     // Bit 31
const KNOWLEDGE_NODE_FLAG: u32 = 0x40000000; // Bit 30

fn encode_node_id(id: u32, is_agent: bool) -> u32 {
    if is_agent {
        id | AGENT_NODE_FLAG
    } else {
        id | KNOWLEDGE_NODE_FLAG
    }
}

fn decode_node_type(encoded_id: u32) -> NodeType {
    if encoded_id & AGENT_NODE_FLAG != 0 {
        NodeType::Agent
    } else if encoded_id & KNOWLEDGE_NODE_FLAG != 0 {
        NodeType::Knowledge
    } else {
        NodeType::Unknown
    }
}
```

## Data Flow Architecture

```mermaid
graph TB
    subgraph "Knowledge Source"
        LG[Logseq Files] --> FS[File Service]
        FS --> KP[Knowledge Processor]
    end

    subgraph "Agent Source"
        CF[Claude Flow MCP] --> WS[WebSocket]
        WS --> CFA[ClaudeFlowActor]
    end

    subgraph "Graph Service"
        KP --> GSA[GraphServiceActor]
        CFA --> GSA
        GSA --> KG[Knowledge Graph Buffer]
        GSA --> AG[Agent Graph Buffer]
    end

    subgraph "GPU Compute"
        KG --> GPU[GPUComputeActor]
        AG --> GPU
        GPU --> PH[Physics Simulation]
        PH --> VP[Vertex Positions]
    end

    subgraph "Client"
        VP --> BR[Binary Protocol]
        BR --> WC[WebSocket Client]
        WC --> R3F[React Three Fiber]
        R3F --> V3D[3D Visualization]
    end
```

## Separation Strategies

### 1. Data Separation

Each graph maintains its own:
- Node buffer in GPU memory
- Edge buffer in GPU memory
- Metadata store in CPU memory
- Update queue for changes

### 2. Physics Separation

Different physics parameters for each graph:
- **Knowledge Graph**: High damping, low spring strength, stable layout
- **Agent Graph**: Low damping, high spring strength, dynamic layout

### 3. Rendering Separation

Visual differentiation:
- **Knowledge Nodes**: Blue spectrum, static icons, document previews
- **Agent Nodes**: Green spectrum, animated status, activity indicators
- **Mixed Edges**: Purple when connecting across graphs

## Update Mechanisms

### Knowledge Graph Updates

1. **File Watcher** detects markdown changes
2. **Semantic Analyzer** processes content
3. **Graph Builder** creates/updates nodes and edges
4. **Binary Protocol** streams position updates via WebSocket
5. **Unified Kernel** processes with DualGraph mode

### Agent Graph Updates

1. **EnhancedClaudeFlowActor** receives MCP data via direct WebSocket
2. **REST API** provides agent data to frontend (/api/bots/agents)
3. **ParallelGraphCoordinator** manages agent positions
4. **Binary Protocol** streams positions via WebSocket (60 FPS)
5. **Unified Kernel** processes with parallel physics simulation

## Performance Optimizations

### Unified Kernel with Parallel Processing

```rust
pub struct UnifiedGPUCompute {
    // Structure of Arrays for maximum performance
    positions_x: CudaSlice<f32>,
    positions_y: CudaSlice<f32>,
    positions_z: CudaSlice<f32>,
    velocities_x: CudaSlice<f32>,
    velocities_y: CudaSlice<f32>,
    velocities_z: CudaSlice<f32>,

    // Edge data
    edge_sources: CudaSlice<i32>,
    edge_targets: CudaSlice<i32>,
    edge_weights: CudaSlice<f32>,

    // Unified kernel function
    unified_kernel: CudaFunction,

    // Compute modes
    compute_mode: ComputeMode, // Basic, DualGraph, Constraints, Analytics
}
```

### Parallel Graph Coordination

Coordinate updates through the parallel graph system:
```typescript
// Frontend coordination
const coordinator = parallelGraphCoordinator;
coordinator.enableLogseq(true);     // Enable knowledge graph
coordinator.enableVisionFlow(true); // Enable agent graph

// Backend processes both in unified kernel
let sim_params = SimParams::default();
sim_params.compute_mode = ComputeMode::DualGraph as i32;
gpu_actor.execute_unified_kernel(sim_params)?;
```

### Level-of-Detail (LOD)

Reduce detail for distant or inactive portions:
- Cull edges below threshold weight
- Simplify node representations
- Aggregate cluster details

## Cross-Graph Interactions

While graphs are separate, they can interact:

1. **Semantic Linking**: Agents reference knowledge nodes
2. **Task Assignment**: Knowledge triggers agent actions
3. **Result Integration**: Agent outputs update knowledge
4. **Unified Search**: Query across both graphs

## Configuration

Configure dual-graph behavior in `settings.yaml`:

```yaml
visualisation:
  graphs:
    knowledge:
      enabled: true
      physics:
        spring_strength: 0.001
        repulsion: 1000.0
        damping: 0.95
      rendering:
        node_size: 1.0
        edge_opacity: 0.6

    agents:
      enabled: true
      physics:
        spring_strength: 0.01
        repulsion: 500.0
        damping: 0.7
      rendering:
        node_size: 1.5
        edge_opacity: 0.8

    cross_graph:
      enabled: false
      max_edges: 100
      weight_threshold: 0.5
```

## Benefits

1. **Performance**: Independent update cycles
2. **Clarity**: Visual separation of concerns
3. **Flexibility**: Different physics per graph type
4. **Scalability**: Parallel GPU processing
5. **Maintainability**: Clean code separation

## Implementation Details

See also:
- [Binary Protocol](../api/binary-protocol.md) for wire format
- [GPU Compute](gpu-compute.md) for physics implementation
- [System Overview](system-overview.md) for overall architecture