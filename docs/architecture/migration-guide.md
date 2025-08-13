# Architecture Migration Guide

This guide helps developers understand the current VisionFlow architecture with unified GPU compute and parallel graphs.

## Current Architecture (v2.0)

### Unified System Features
- **Unified GPU Kernel**: Single CUDA kernel (`visionflow_unified.cu`) with 4 compute modes
- **Parallel Graphs**: Independent Logseq and VisionFlow graph processing
- **Direct MCP Integration**: Backend-only WebSocket connection to Claude Flow
- **Enhanced Performance**: Structure of Arrays memory layout, optimised binary protocol
- **Comprehensive Agent Visualization**: Real-time Multi Agent Visualisation

### Key Components
- **EnhancedClaudeFlowActor**: Direct WebSocket MCP connection with differential updates
- **ParallelGraphCoordinator**: Frontend coordinator managing both graph types
- **Unified GPU Compute**: Single kernel handling all physics modes
- **Binary Protocol**: Efficient position/velocity streaming for both graphs

## Current Implementation Patterns

### 1. Graph Component Usage

**Recommended Pattern (Current):**
```typescript
// Use parallel graphs hook
import { useParallelGraphs } from '@/features/graph/hooks/useParallelGraphs';

function GraphVisualization() {
  const {
    logseqPositions,
    visionFlowPositions,
    enableLogseq,
    enableVisionFlow,
    state
  } = useParallelGraphs({
    enableLogseq: true,
    enableVisionFlow: true,
    autoConnect: true
  });

  // Use positions from both graphs in rendering
  return <Canvas positions={new Map([...logseqPositions, ...visionFlowPositions])} />;
}
```

### 2. Agent Data Integration

**Current Implementation:**
```typescript
// Backend handles MCP via EnhancedClaudeFlowActor
// Frontend uses REST API for agent metadata
const response = await fetch('/api/bots/agents');
const { agents, communications } = await response.json();

// Position updates come via binary protocol WebSocket
// ParallelGraphCoordinator automatically handles both graph types
const coordinator = parallelGraphCoordinator;
coordinator.enableVisionFlow(true);

// Positions are updated automatically via WebSocket binary protocol
const agentPositions = coordinator.getVisionFlowPositions();
```

### 3. Physics Processing

**Current Implementation:**
```typescript
// Physics is handled by unified backend kernel
// No frontend physics workers needed
// ParallelGraphCoordinator manages position updates from binary protocol

// Configuration via settings
const config = {
  gpu: {
    compute_mode: 'dual_graph', // Use DualGraph mode for parallel processing
    unified_kernel: {
      block_size: 256,
      ptx_path: 'src/utils/ptx/visionflow_unified.ptx'
    }
  }
};

// Backend automatically processes both graphs in unified kernel
// Frontend receives position updates via WebSocket
```

### 4. Backend Integration

**Current Implementation:**
```rust
// EnhancedClaudeFlowActor with direct WebSocket to MCP
let enhanced_actor = EnhancedClaudeFlowActor::new(
    claude_flow_client,
    graph_service_addr.clone()
);
let enhanced_addr = enhanced_actor.start();

// Actor handles WebSocket connection automatically
// Differential updates with pending changes
enhanced_addr.send(initializeMultiAgent {
    topology: multi-agentTopology::Hierarchical,
    max_agents: 8,
    agent_types: vec![
        AgentType::Coordinator,
        AgentType::Researcher,
        AgentType::Coder
    ],
    enable_neural: true,
}).await?;

// Unified GPU kernel processes both graphs
let gpu_actor = GPUComputeActor::new_with_unified_kernel()?;
let sim_params = SimParams {
    compute_mode: ComputeMode::DualGraph as i32,
    // ... other physics parameters
};
```

### 5. Update Configuration

**Old Pattern:**
```typescript
// Single graph config
const graphConfig = {
  physics: {
    springStrength: 0.2,
    damping: 0.95
  }
};
```

**New Pattern:**
```typescript
// Per-graph-type config
const DEFAULT_GRAPH_CONFIG = {
  logseq: {
    physics: {
      springStrength: 0.2,
      damping: 0.95,
      // ... Logseq specific
    }
  },
  visionflow: {
    physics: {
      springStrength: 0.3,
      damping: 0.95,
      // ... VisionFlow specific
    }
  }
};
```

## API Changes

### Deprecated Endpoints
- `WS /mcp` - Direct MCP WebSocket (removed)
- `POST /api/claude-flow/*` - Direct Claude Flow proxy (removed)

### New Endpoints
- `GET /api/bots/agents` - Get agent list
- `POST /api/bots/spawn` - Spawn new agent
- `POST /api/bots/multi-agent/init` - Initialize multi-agent
- `DELETE /api/bots/agent/:id` - Terminate agent

## Component Updates

### Affected Components

1. **Graph Visualization Components**
   - Update to use `useParallelGraphs` hook
   - Add graph type selector UI

2. **Settings Panel**
   - Add VisionFlow enable/disable toggle
   - Add agent visualisation settings

3. **Performance Monitoring**
   - Update to track both graph types
   - Add agent-specific metrics

### New Components

1. **Agent Visualization**
   - `BotsRenderer` - 3D agent rendering
   - `AgentInspector` - Agent details panel
   - `CommunicationFlow` - Message visualisation

2. **multi-agent Controls**
   - `multi-agentInitializer` - UI for multi-agent setup
   - `AgentSpawner` - UI for adding agents
   - `TopologySelector` - Choose multi-agent topology

## Testing Updates

### Unit Tests
```typescript
// Test parallel graphs isolation
it('should maintain separate state for each graph type', () => {
  const { result } = renderHook(() => useParallelGraphs());

  act(() => {
    result.current.enableLogseq(true);
    result.current.enableVisionFlow(false);
  });

  expect(result.current.isLogseqEnabled).toBe(true);
  expect(result.current.isVisionFlowEnabled).toBe(false);
});
```

### Integration Tests
```typescript
// Test backend MCP relay
it('should relay agent data from backend', async () => {
  const response = await fetch('/api/bots/agents');
  const data = await response.json();

  expect(data.agents).toBeDefined();
  expect(data.agents.length).toBeGreaterThan(0);
});
```

## Performance Considerations

1. **Memory Usage**: Two physics simulations run in parallel
   - Monitor total memory usage
   - Implement cleanup for inactive graphs

2. **CPU Usage**: Multiple web workers
   - Use worker pooling if needed
   - Implement LOD for distant nodes

3. **Network Traffic**: Polling vs WebSocket
   - Current: REST polling every 10s
   - Future: WebSocket push updates

## Rollback Plan

If issues arise, you can disable VisionFlow:

```typescript
// Disable in settings
const settings = {
  features: {
    parallelGraphs: false,  // Disable parallel graphs
    agentVisualization: false  // Disable agent viz
  }
};
```

Or via environment:
```bash
DISABLE_VISIONFLOW=true npm start
```

## Common Issues

### 1. "Cannot find module" Errors
Update import paths:
```typescript
// Old
import { mcpService } from '@/services/mcpService';

// New
import { mcpWebSocketService } from '@/features/bots/services/MCPWebSocketService';
```

### 2. WebSocket Connection Failures
Frontend no longer connects to MCP directly. Check backend logs:
```bash
RUST_LOG=debug cargo run
```

### 3. Missing Agent Data
Ensure ClaudeFlowActor is running:
```bash
# Check actor logs
grep "ClaudeFlowActor" /app/logs/visionflow.log
```

## Support Resources

- Architecture Docs: `/docs/architecture/`
- Example Code: `/examples/parallel-graphs/`
- Issue Tracker: Create issues with tag `migration`

## Timeline

1. **Phase 1** (Completed): Core parallel graphs implementation
2. **Phase 2** (Completed): MCP backend relay
3. **Phase 3** (Current): Documentation and migration support
4. **Phase 4** (Future): WebSocket push updates