# Dual Graph Architecture Documentation

## Overview

The system implements a dual graph architecture supporting two distinct graph types:
1. **Knowledge Graph** (Logseq documents and metadata)
2. **Agent Graph** (AI agent swarm from Claude Flow)

## Binary Protocol (28-byte structure per node)

### Wire Format
```
[0-3]   Node ID (u32) with type flags in high bits
[4-15]  Position (3x f32: x, y, z)  
[16-27] Velocity (3x f32: vx, vy, vz)
Total: 28 bytes per node
```

### Node Type Identification
- **Bit 31 (0x80000000)**: Agent node flag
- **Bit 30 (0x40000000)**: Knowledge node flag
- **Bits 0-29 (0x3FFFFFFF)**: Actual node ID

### Bidirectional Communication
- **Server → Client**: Position/velocity updates via binary WebSocket
- **Client → Server**: User interaction updates (drag, pin, etc.)
- Both directions use the same 28-byte format
- Client-side: `WebSocketService.sendNodePositionUpdates()`
- Server-side: `socket_flow_handler` handles binary messages

## GPU Compute Architecture

### Kernel Selection (Automatic)
```rust
enum KernelMode {
    Legacy,          // < 1000 nodes
    Advanced,        // 1000-10000 nodes  
    VisualAnalytics, // > 10000 nodes or isolation layers
}
```

### Dual Graph Kernel (`compute_dual_graphs.cu`)
- Processes knowledge and agent nodes separately
- Different physics parameters per graph type:
  - Knowledge: Gentler forces (spring=0.15, damping=0.85)
  - Agent: Dynamic forces (spring=0.25, damping=0.75)
- Shared GPU buffers with type-based filtering

### GPU Data Structures
```rust
// Separate buffers for each graph
knowledge_node_data: CudaSlice<BinaryNodeData>
agent_node_data: CudaSlice<BinaryNodeData>
edge_data: CudaSlice<EdgeData>  // Shared, filtered by node type

// Index mappings
knowledge_node_indices: HashMap<u32, usize>
agent_node_indices: HashMap<u32, usize>
graph_type_map: HashMap<u32, GraphType>
```

## REST API Endpoints

### Knowledge Graph Data
```
GET /api/graph/data
Response: {
  nodes: [...],      // Knowledge nodes with metadata
  edges: [...],      // Document relationships
  metadata: {...}    // Full metadata store
}
```

### Agent Graph Data
```
GET /api/bots/data
Response: {
  nodes: [...],      // Agent nodes with capabilities
  edges: [...],      // Communication channels
  metadata: {...}    // Agent status, tasks, performance
}
```

### Quest 3 Optimized Settings
```
GET /api/quest3/defaults
POST /api/quest3/calibrate
```

### Visual Analytics Control
```
GET/POST /api/analytics/params
GET/POST /api/analytics/constraints
POST /api/analytics/focus
GET /api/analytics/stats
```

## Metadata Population

### Knowledge Node Metadata
- Document title, content, tags
- File size, creation date
- Relationships, references
- User annotations

### Agent Node Metadata
```json
{
  "type": "coordinator|researcher|coder|...",
  "status": "active|idle|processing",
  "capabilities": ["code_generation", "analysis"],
  "current_task": "Implementing authentication",
  "tasks_active": 3,
  "tasks_completed": 127,
  "success_rate": 0.94,
  "tokens": 45230,
  "swarm_id": "hive-mind-001",
  "parent_queen_id": "queen-alpha"
}
```

## WebSocket Binary Stream Management

### Server-Side Broadcasting
```rust
// Graph actor broadcasts positions
let positions: Vec<(u32, BinaryNodeData)> = ...;

// Set type flags
for (id, data) in positions {
    let flagged_id = if is_agent { 
        set_agent_flag(id) 
    } else { 
        set_knowledge_flag(id) 
    };
    // Encode with flags
}

// Broadcast via ClientManager
let binary_data = binary_protocol::encode_node_data(&positions);
client_manager.broadcast_binary(binary_data);
```

### Client-Side Processing
```typescript
// Parse incoming binary data
const nodes = parseBinaryNodeData(buffer);

// Identify node types
nodes.forEach(node => {
    const nodeType = getNodeType(node.nodeId);
    const actualId = getActualNodeId(node.nodeId);
    
    if (nodeType === NodeType.Agent) {
        // Update agent visualization
    } else if (nodeType === NodeType.Knowledge) {
        // Update knowledge visualization
    }
});
```

### Client-Side Position Updates
```typescript
// Send user interactions back to server
webSocketService.sendNodePositionUpdates([{
    nodeId: actualNodeId,
    position: { x, y, z },
    velocity: { x: 0, y: 0, z: 0 }
}]);
```

## Index Management

### Server-Side Indexing
1. **Global index**: All nodes across both graphs
2. **Graph-specific indices**: Separate for knowledge and agents
3. **GPU buffer indices**: Direct memory offsets

### Client-Side Indexing
1. **Display ID**: What users see (without flags)
2. **Wire ID**: What's transmitted (with type flags)
3. **Actual ID**: Extracted from wire format

## Performance Optimizations

### GPU Optimizations
- Dual graph kernel processes both types in parallel
- Shared memory for block-level optimization
- Coalesced memory access patterns
- Warp-level primitives for reductions

### Network Optimizations
- Binary protocol (28 bytes vs JSON ~200 bytes)
- Type flags in ID (no separate type field)
- Delta compression for position updates
- Adaptive update rates based on motion

### Quest 3 Specific
- LOD based on distance
- Instanced rendering
- Immediate AR mode entry
- No UI chrome overhead

## Testing Endpoints

### Verify Dual Graph Separation
```bash
# Get knowledge graph
curl http://localhost:8080/api/graph/data | jq '.nodes[0]'

# Get agent graph  
curl http://localhost:8080/api/bots/data | jq '.nodes[0]'

# Check binary protocol
wscat -c ws://localhost:8080/wss
> {"type": "request_full_snapshot", "graphs": ["knowledge", "agent"]}
```

### Monitor GPU Kernel Mode
```bash
curl http://localhost:8080/api/analytics/stats | jq '.kernel_mode'
```

## Common Issues & Solutions

### Issue: Nodes mixing between graphs
**Solution**: Check node type flags in binary protocol

### Issue: Wrong physics applied
**Solution**: Verify graph_type_map in GPU compute actor

### Issue: Missing metadata
**Solution**: Ensure REST endpoints include full metadata store

### Issue: Position updates not working
**Solution**: Check bidirectional WebSocket handlers

## Future Enhancements

1. **Separate edge buffers** for each graph type
2. **Graph-specific physics kernels** for specialized behaviors
3. **Cross-graph interactions** (agents affecting documents)
4. **Hierarchical graph support** (sub-graphs within graphs)
5. **Temporal graph evolution** (4D visualization)