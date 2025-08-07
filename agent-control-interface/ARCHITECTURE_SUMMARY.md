# Agent Control Interface - Architecture Summary

## Core Insight: Separation of Concerns

The VisionFlow system implements a clean separation between **telemetry collection** and **spatial visualization**:

### Our Responsibility (This Container)
- **Agent Telemetry Collection**
  - Agent states (active, idle, busy)
  - Performance metrics (CPU, memory, task completion)
  - Agent types and capabilities
  - Connection relationships
  - Message flow statistics
- **Data Aggregation**
  - Merge data from multiple MCP sources
  - Normalize to consistent format
  - Cache for performance
- **API Service**
  - TCP server on port 9500
  - JSON-RPC 2.0 protocol
  - Session management

### Remote Rust Service Responsibility
- **Spatial Topology Calculation**
  - GPU-accelerated physics (compute_forces.cu)
  - Force-directed graph layout
  - 3D position computation
  - Real-time position updates
- **Visualization Rendering**
  - 3D scene management
  - Camera controls
  - Visual effects

## Simplified Architecture

```
┌────────────────────┐
│  VisionFlow Client │
└─────────┬──────────┘
          │ TCP:9500
┌─────────▼──────────┐
│   Our Container    │
│ ┌────────────────┐ │
│ │ Telemetry API  │ │ ◄── No position calculations!
│ └────────────────┘ │
│ ┌────────────────┐ │
│ │ MCP Tools:     │ │
│ │ • claude-flow  │ │
│ │ • ruv-swarm    │ │
│ │ • observability│ │
│ └────────────────┘ │
└────────────────────┘
          │
    Raw Telemetry
          │
┌─────────▼──────────┐
│  Rust GPU Service  │
│ ┌────────────────┐ │
│ │ Physics Engine │ │ ◄── GPU-accelerated positioning
│ └────────────────┘ │
│ ┌────────────────┐ │
│ │ 3D Viz Render  │ │
│ └────────────────┘ │
└────────────────────┘
```

## Key Advantages

1. **Simplified Implementation**: No complex physics calculations needed
2. **Better Performance**: GPU handles heavy computation
3. **Clean Interface**: We provide data, Rust provides visualization
4. **Existing Tools**: Leverage mcp-observability's telemetry features
5. **Scalability**: Can handle thousands of agents efficiently

## Data Flow Example

When VisionFlow requests a visualization snapshot:

1. **Request**: `{"method": "tools/call", "params": {"name": "visualization.snapshot"}}`
2. **Our Processing**:
   - Query MCP tools for agent states
   - Aggregate telemetry data
   - Format as standardized response
3. **Response**: Agent telemetry WITHOUT positions
4. **Rust Processing**: 
   - Receives telemetry
   - Computes positions using GPU
   - Renders 3D visualization

## Implementation Strategy

Rather than building everything from scratch, we can:

1. **Leverage mcp-observability** - Already tracks agents, messages, and metrics
2. **Add TCP wrapper** - Simple TCP→stdio bridge for mcp-observability
3. **Format translation** - Convert MCP responses to VisionFlow format
4. **State aggregation** - Combine data from multiple MCP sources

This approach reduces complexity from ~2000 lines of code to ~500 lines.