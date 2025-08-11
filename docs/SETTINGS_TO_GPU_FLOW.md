# Settings Changes to GPU Flow Documentation

## Overview
Settings changes flow from the client to the GPU through multiple paths, but **the scene is NOT automatically updated when settings.yaml changes on disk**. Updates happen through:
1. **Control Center UI** (real-time updates via WebSocket)
2. **Settings API** (programmatic updates)
3. **Application restart** (loads from settings.yaml)

## Flow Paths

### 1. Settings.yaml at Startup
```
settings.yaml → AppFullSettings::load() → AppState::new() → UpdateSimulationParams → GPUComputeActor
```
- Loaded once at application startup
- Values converted through: PhysicsSettings → SimulationParams → SimParams
- Sent to GPU during initialization

### 2. Control Center Real-time Updates (WebSocket)
```
Client UI → WebSocket → socket_flow_handler → UpdatePhysicsParams → GPUComputeActor → GPU
```

**Key code from socket_flow_handler.rs:**
```rust
Some("update_physics_params") => {
    // Parse parameters from client
    let mut params = SimulationParams::default();
    params.spring_strength = msg["spring_strength"];
    params.damping = msg["damping"];
    params.repulsion = msg["repulsion"];
    
    // Send directly to GPU compute actor
    gpu_addr.send(UpdatePhysicsParams {
        graph_type,
        params,
    }).await
}
```

**GPU receives and applies immediately:**
```rust
impl Handler<UpdatePhysicsParams> for GPUComputeActor {
    fn handle(&mut self, msg: UpdatePhysicsParams) {
        // Update internal params
        self.simulation_params = msg.params.clone();
        self.unified_params = SimParams::from(&msg.params);
        
        // Push to GPU immediately
        if let Some(ref mut unified_compute) = self.unified_compute {
            unified_compute.set_params(self.unified_params);
        }
    }
}
```

### 3. Settings API Updates
```
HTTP POST /api/settings → settings_handler → UpdateSettings → propagate_physics_to_gpu → GPUComputeActor
```

**From settings_handler.rs:**
```rust
async fn update_settings() {
    // Check if physics was updated
    if physics_updated {
        propagate_physics_to_gpu(&state, &app_settings, "logseq").await;
    }
}

async fn propagate_physics_to_gpu() {
    let physics = settings.get_physics(graph);
    let sim_params = physics.into();
    
    // Send to both GraphService and GPU
    gpu_addr.send(UpdateSimulationParams { params: sim_params }).await;
}
```

## Physics Simulation Loop

The physics simulation runs continuously at 60 FPS:

**From graph_actor.rs:**
```rust
fn start_simulation_loop(&mut self, ctx: &mut Context<Self>) {
    ctx.run_interval(Duration::from_millis(16), |act, _ctx| {
        // Every 16ms (60 FPS)
        if let Some(gpu_addr) = &act.gpu_compute_addr {
            gpu_addr.do_send(ComputeForces);
        }
    });
}
```

## Key Points

### Real-time Updates
- **Control Center sliders** → Immediate GPU update via WebSocket
- **No file watching** - settings.yaml changes require restart
- **60 FPS simulation** - Forces computed every 16ms

### Update Mechanisms
1. **WebSocket** (fastest) - Direct from UI to GPU
2. **Settings API** - Programmatic updates with validation
3. **Restart** - Loads from settings.yaml

### Parameter Flow
```
Client (camelCase) → Server (snake_case) → GPU (SimParams struct)
```

Example conversion:
- Client: `springStrength: 0.005`
- Server: `spring_strength: 0.005`
- GPU: `spring_k: 0.005`

### No Automatic File Watching
- **settings.yaml is NOT watched for changes**
- Must restart application to load new values from file
- Or use Control Center/API for live updates

## Testing Physics Updates

### Via Control Center:
1. Open Control Center in UI
2. Adjust physics sliders
3. Changes apply immediately (60 FPS)

### Via Settings API:
```bash
curl -X POST http://localhost:3001/api/settings \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### Via WebSocket (direct):
```javascript
ws.send(JSON.stringify({
  type: "update_physics_params",
  graph_type: "knowledge",
  params: {
    spring_strength: 0.01,
    repulsion: 100.0,
    damping: 0.8
  }
}));
```

## Performance Characteristics

- **GPU update latency**: < 1ms (params are just copied to GPU memory)
- **Simulation rate**: 60 FPS (16ms intervals)
- **WebSocket latency**: ~5-10ms
- **Settings API latency**: ~20-50ms (includes validation)

## Important Notes

1. **Settings.yaml is NOT hot-reloaded** - requires restart
2. **Control Center provides real-time updates** without file changes
3. **GPU simulation runs continuously** at 60 FPS regardless of updates
4. **Parameters are immediately applied** when received by GPU
5. **Multiple update paths** exist for different use cases