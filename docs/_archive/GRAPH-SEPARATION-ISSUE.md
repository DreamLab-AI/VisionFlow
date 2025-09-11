# Graph Separation Issue - Logseq vs Agent Telemetry

## Current State Analysis

### ✅ What's Working (Separated)

1. **Settings Structure** (`/data/settings.yaml`):
   - `visualisation.graphs.logseq` - Logseq knowledge graph settings
   - `visualisation.graphs.visionflow` - Agent telemetry graph settings
   - Each has separate physics, nodes, edges, and labels configurations

2. **Graph Data Storage** (`GraphServiceActor`):
   - `self.graph_data` - Stores Logseq knowledge graph nodes/edges
   - `self.bots_graph_data` - Stores agent telemetry graph nodes/edges
   - Separate data structures for each graph

3. **Graph Type Enum** (`GPUComputeActor`):
   ```rust
   pub enum GraphType {
       Knowledge,  // Logseq knowledge graph
       Agent,      // AI agent swarm
   }
   ```

4. **Physics Propagation Function**:
   - `propagate_physics_to_gpu()` accepts graph name parameter ("logseq" or "visionflow")
   - Correctly extracts physics for the specified graph from settings

### ❌ What's NOT Separated (The Problem)

1. **Single Simulation Parameters**:
   - GraphServiceActor has only ONE `simulation_params: SimulationParams`
   - Both graphs share the same physics parameters at runtime
   - When you update Logseq physics, it affects the agent graph too
   - When you update VisionFlow physics, it overwrites Logseq physics

2. **UpdateSimulationParams Message**:
   - Doesn't include which graph it's for
   - Handler blindly updates the single `simulation_params`
   - No way to target specific graph

3. **GPU Compute**:
   - Single unified compute context
   - No separation between graph physics simulations
   - Both graphs would run with same physics parameters

## The Impact

When you adjust physics controls in the UI:
- If you're viewing the Logseq graph and change physics, it works
- If you switch to the agent graph, it uses the SAME physics
- Changing agent graph physics overwrites Logseq settings
- Both graphs can't have different physics running simultaneously

## Code Evidence

**GraphServiceActor struct** (line 90):
```rust
simulation_params: SimulationParams,  // Single params for both graphs!
target_params: SimulationParams,      // Single target for smooth transitions
```

**UpdateSimulationParams handler** (line 2013):
```rust
self.simulation_params = msg.params.clone();  // Overwrites for both graphs
self.target_params = msg.params.clone();      // No graph distinction
```

**Physics propagation** (`settings_handler.rs`):
```rust
// Correctly gets graph-specific physics
let physics = settings.get_physics(graph);  // "logseq" or "visionflow"

// But sends generic message without graph info
let update_msg = UpdateSimulationParams { params: sim_params.clone() };
```

## Required Fix

### Option 1: Separate Simulation Parameters (Recommended)
1. Add to GraphServiceActor:
   ```rust
   logseq_simulation_params: SimulationParams,
   logseq_target_params: SimulationParams,
   visionflow_simulation_params: SimulationParams,
   visionflow_target_params: SimulationParams,
   current_graph: GraphType,  // Which graph is active
   ```

2. Update message to include graph:
   ```rust
   pub struct UpdateSimulationParams {
       pub params: SimulationParams,
       pub graph: String,  // "logseq" or "visionflow"
   }
   ```

3. Update handler to target correct params:
   ```rust
   match msg.graph.as_str() {
       "logseq" => {
           self.logseq_simulation_params = msg.params.clone();
           self.logseq_target_params = msg.params.clone();
       }
       "visionflow" => {
           self.visionflow_simulation_params = msg.params.clone();
           self.visionflow_target_params = msg.params.clone();
       }
       _ => // handle error
   }
   ```

### Option 2: Active Graph Switching (Simpler but Limited)
1. Add current graph tracking:
   ```rust
   current_graph: GraphType,
   ```

2. Only update physics for the active graph
3. Switch physics when switching graphs
4. Limitation: Can't run both graphs with different physics simultaneously

## Testing After Fix

1. Set different physics for each graph in settings.yaml
2. Load Logseq graph, verify it uses Logseq physics
3. Switch to agent graph, verify it uses VisionFlow physics
4. Adjust Logseq physics, verify agent physics unchanged
5. Adjust agent physics, verify Logseq physics unchanged

## Files That Need Changes

1. `/src/actors/graph_actor.rs`:
   - Add separate simulation params
   - Update handler logic
   - Update physics step to use correct params

2. `/src/actors/messages.rs`:
   - Add graph field to UpdateSimulationParams

3. `/src/handlers/settings_handler.rs`:
   - Include graph in UpdateSimulationParams message

4. `/src/actors/gpu_compute_actor.rs`:
   - May need to handle dual graph physics separately

## Current Workaround

Currently, the physics settings will apply to whichever graph is active. To work around:
1. Set physics appropriate for the graph you're viewing
2. Understand that switching graphs keeps the same physics
3. Manually adjust physics when switching between graphs