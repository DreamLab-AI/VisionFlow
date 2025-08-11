# Parallel Graph Architecture

This document describes the parallel graph architecture that enables VisionFlow to run multiple independent graph visualizations simultaneously using the unified GPU kernel and parallel graph coordinator.

## Overview

The parallel graph system allows two graph types to coexist with independent processing:
- **Logseq Graph**: Knowledge graph data from Logseq markdown files (binary protocol)
- **VisionFlow Graph**: AI agent visualization data from Claude Flow MCP integration (REST API)

## Architecture Components

### 1. Parallel Graph Coordinator (`parallelGraphCoordinator.ts`)

The central orchestrator managing multiple graph systems:

```typescript
class ParallelGraphCoordinator {
  private state: ParallelGraphState = {
    logseq: {
      enabled: boolean,
      data: GraphData | null,
      lastUpdate: number
    },
    visionflow: {
      enabled: boolean,
      agents: BotsAgent[],
      edges: BotsEdge[],
      tokenUsage: TokenUsage | null,
      lastUpdate: number
    }
  };
  private listeners: Set<(state: ParallelGraphState) => void>;
}
```

**Key Responsibilities:**
- Singleton pattern for centralized graph state management
- Independent enable/disable for each graph type
- Event-driven updates to React components
- Position map management for both graph types
- Integration with both binary protocol and REST APIs

### 2. Graph Type System

Each component in the system has a `graphType` property to prevent data mixing:

```typescript
type GraphType = 'logseq' | 'visionflow';

// Data managers check graph type before processing
if (graphDataManager.getGraphType() === 'logseq') {
  // Process Logseq-specific data
}
```

### 3. Unified Backend with Parallel Processing

Both graphs are processed by the unified CUDA kernel on the backend:

- **Backend**: Unified GPU kernel processes both graphs with DualGraph mode
- **Frontend**: `parallelGraphCoordinator` manages separate position maps
- **Physics**: Different physics parameters applied per graph type
- **Updates**: Binary protocol streams positions for both graphs

This architecture ensures:
- Optimal GPU utilization with single kernel
- Independent graph lifecycle management
- Efficient memory coalescing with Structure of Arrays
- Real-time updates via WebSocket binary protocol

### 4. Data Flow Architecture

#### Logseq Data Flow
```
File System Changes (markdown files)
    ↓
Backend GraphServiceActor
    ↓
Unified GPU Kernel (DualGraph mode)
    ↓
WebSocketService (binary protocol position updates)
    ↓
parallelGraphCoordinator.logseqPositions
    ↓
3D Rendering
```

#### VisionFlow Data Flow
```
Claude Flow MCP (port 3002)
    ↓
EnhancedClaudeFlowActor (direct WebSocket)
    ↓
REST API (/api/bots/agents) + Binary position updates
    ↓
parallelGraphCoordinator.visionFlowPositions
    ↓
3D Rendering
```

## React Hook API

The `useParallelGraphs` hook provides a clean interface for components:

```typescript
const {
  state,                    // Current state of both graphs
  isLogseqEnabled,         // Logseq graph status
  isVisionFlowEnabled,     // VisionFlow graph status
  enableLogseq,            // Toggle Logseq graph
  enableVisionFlow,        // Toggle VisionFlow graph
  logseqPositions,         // Map<nodeId, {x,y,z}>
  visionFlowPositions,     // Map<agentId, {x,y,z}>
  refreshPositions         // Manual position refresh
} = useParallelGraphs({
  enableLogseq: true,
  enableVisionFlow: true,
  autoConnect: true
});
```

## Configuration

Each graph type has independent physics configuration:

```typescript
const DEFAULT_GRAPH_CONFIG = {
  logseq: {
    physics: {
      springStrength: 0.2,
      damping: 0.95,
      maxVelocity: 0.02,
      centerForce: 0.001,
      repulsionStrength: 0.5,
      linkDistance: 2.0
    }
  },
  visionflow: {
    physics: {
      springStrength: 0.3,
      damping: 0.95,
      maxVelocity: 0.5,
      centerForce: 0.002,
      repulsionStrength: 0.8,
      linkDistance: 3.0
    }
  }
};
```

## Key Features

### 1. Independent Operation
- Each graph can be enabled/disabled without affecting the other
- Separate data sources and update mechanisms
- No shared state between graph types

### 2. Type Safety
- TypeScript interfaces ensure type safety at compile time
- Runtime graph type checking prevents data corruption
- Strongly typed position maps for each graph

### 3. Performance Optimizations
- Web Workers keep physics calculations off the main thread
- Binary protocol for efficient Logseq position updates
- Differential updates only send changed positions
- Independent update rates for each graph

### 4. Extensibility
The architecture supports adding new graph types:

```typescript
// Future graph types could include:
type GraphType = 'logseq' | 'visionflow' | 'dependency' | 'flowchart';
```

## Implementation Details

### Data Protocol Integration

The system uses multiple data channels:
- **Binary Protocol WebSocket** (`/wss`): Position/velocity updates for both graphs
- **REST API** (`/api/bots/*`): Agent metadata and status from MCP
- **Direct MCP WebSocket**: Backend-only connection to Claude Flow

Note: Frontend never connects directly to MCP - all agent data flows through backend.

### Binary Protocol Integration

Both graph types use the unified binary protocol:
- 28 bytes per node (ID, position, velocity) for both Logseq and agent nodes
- Graph type differentiation handled by parallel coordinator
- Compression for messages > 1KB
- Single WebSocket connection handles both graph streams
- GPU processes both graphs in unified kernel with DualGraph mode

### Memory Management

- Position maps are updated incrementally
- Old positions are garbage collected
- Workers manage their own memory pools

## Testing

The parallel graph system includes comprehensive tests:
- Graph type isolation tests
- Independent enable/disable verification
- State change notification tests
- Position map independence checks

## Migration Notes

When migrating from single-graph to parallel-graph architecture:

1. Update components to use `useParallelGraphs` hook
2. Replace direct `graphDataManager` calls with coordinator API
3. Ensure graph type is specified for all data operations
4. Update physics configuration for each graph type

## Future Enhancements

1. **Dynamic Graph Types**: Runtime registration of new graph types
2. **Cross-Graph Relationships**: Visual links between related nodes
3. **Unified Search**: Search across all graph types
4. **Graph Persistence**: Save/restore graph states
5. **Performance Monitoring**: Per-graph performance metrics