# Corrected: Settings to GPU Flow Documentation

## Overview
You're absolutely right - WebSockets are for bi-directional position/velocity streaming for the force-directed graphs. Settings management uses JSON REST API endpoints. Here's the correct flow:

## Architecture Components

### 1. WebSocket (`/wss`) - Position & Velocity Streaming
- **Purpose**: Real-time bi-directional streaming of node positions and velocities
- **Binary Protocol**: Efficient binary encoding for performance
- **Direction**: Server → Client (computed positions), Client → Server (user interactions)

```rust
// From socket_flow_handler.rs
impl Handler<BroadcastPositionUpdate> for SocketFlowServer {
    fn handle(&mut self, msg: BroadcastPositionUpdate, ctx: &mut Self::Context) {
        // Encode positions/velocities in binary format
        let binary_data = binary_protocol::encode_node_data(&msg.0);
        ctx.binary(binary_data);  // Send to client
    }
}
```

### 2. JSON REST API - Settings Management

#### Main Settings Endpoint: `/api/settings`
From `settings_handler.rs`:
```rust
cfg.service(
    web::scope("/api/settings")
        .route("", web::get().to(get_settings))      // GET current settings
        .route("", web::post().to(update_settings))  // POST updates
        .route("/reset", web::post().to(reset_settings))
);
```

#### Settings Update Flow:
1. **Client sends JSON POST to `/api/settings`**
2. **Server validates and merges settings**
3. **If physics updated, propagates to GPU**
4. **GPU applies new parameters immediately**

## Detailed Flow: Settings → GPU

### Step 1: Client Updates Settings (JSON API)
```javascript
// Client sends settings update
POST /api/settings
{
  "visualisation": {
    "graphs": {
      "logseq": {
        "physics": {
          "springStrength": 0.01,
          "repulsion": 100.0,
          "damping": 0.8
        }
      }
    }
  }
}
```

### Step 2: Server Processes Update
```rust
// settings_handler.rs
async fn update_settings() {
    // 1. Validate settings
    validate_settings_update(&update)?;
    
    // 2. Merge with current settings
    app_settings.merge_update(update)?;
    
    // 3. Check if physics was updated
    let physics_updated = update.get("visualisation")
        .and_then(|v| v.get("graphs"))
        .map(|graphs| graphs.contains_key("logseq"));
    
    // 4. If physics changed, propagate to GPU
    if physics_updated {
        propagate_physics_to_gpu(&state, &app_settings, "logseq").await;
    }
}
```

### Step 3: Propagate to GPU
```rust
async fn propagate_physics_to_gpu(
    state: &AppState,
    settings: &AppFullSettings,
    graph: &str
) {
    let physics = settings.get_physics(graph);
    let sim_params = SimulationParams::from(physics);
    
    // Send to GPU compute actor
    if let Some(gpu_addr) = &state.gpu_compute_addr {
        gpu_addr.send(UpdateSimulationParams { 
            params: sim_params 
        }).await;
    }
}
```

### Step 4: GPU Applies Parameters
```rust
impl Handler<UpdateSimulationParams> for GPUComputeActor {
    fn handle(&mut self, msg: UpdateSimulationParams) {
        // Update simulation parameters
        self.simulation_params = msg.params.clone();
        self.unified_params = SimParams::from(&msg.params);
        
        // Push to GPU immediately
        if let Some(ref mut unified_compute) = self.unified_compute {
            unified_compute.set_params(self.unified_params);
        }
    }
}
```

## The Physics Loop (Separate from Settings)

The physics simulation runs continuously at 60 FPS:
```rust
// graph_actor.rs
fn start_simulation_loop(&mut self, ctx: &mut Context<Self>) {
    ctx.run_interval(Duration::from_millis(16), |actor, _| {
        // Compute forces with current parameters
        gpu_addr.do_send(ComputeForces);
    });
}
```

After each computation:
1. GPU calculates new positions/velocities
2. Results sent to ClientManager
3. ClientManager broadcasts via WebSocket (binary protocol)
4. Client receives and renders updated positions

## Summary of Correct Architecture

### REST API (JSON)
- **Endpoint**: `/api/settings`
- **Purpose**: Settings and metadata management
- **Format**: JSON request/response
- **Updates**: Persistent configuration changes

### WebSocket (Binary)
- **Endpoint**: `/wss`
- **Purpose**: Real-time position/velocity streaming
- **Format**: Binary protocol for efficiency
- **Updates**: 60 FPS physics simulation results

### Key Points:
1. **Settings changes go through REST API**, not WebSocket
2. **WebSocket is for physics data streaming**, not configuration
3. **Settings updates trigger GPU parameter updates** via actor messages
4. **Physics loop runs continuously** regardless of settings changes
5. **Binary protocol optimizes bandwidth** for position/velocity data

## File Watching
- **settings.yaml is NOT watched** - requires restart to load from file
- **REST API updates are immediate** - no file write required
- **GPU receives updates instantly** when settings change via API