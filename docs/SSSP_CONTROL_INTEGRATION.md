# SSSP Control Integration

## Current Control Flow

1. **Control Center** → REST API (`/api/settings`)
2. **Settings Handler** → SettingsActor
3. **SettingsActor** → GraphServiceActor (UpdateSimulationParams)
4. **GraphServiceActor** → ForceComputeActor
5. **ForceComputeActor** → GPU execution with physics parameters

## Missing SSSP Controls

The `sssp_alpha` field exists in `SimParams` (line 72 of simulation_params.rs) but is NOT exposed through the REST API. We need to add:

### 1. Add to PhysicsSettingsDTO

```rust
// src/handlers/settings_handler.rs
pub struct PhysicsSettingsDTO {
    // ... existing fields ...
    pub sssp_enabled: bool,           // Enable SSSP computation
    pub sssp_alpha: f32,               // SSSP influence on layout (0.0-1.0)
    pub sssp_source_nodes: Vec<u32>,  // Source nodes for SSSP
    pub sssp_update_interval: u32,    // How often to recompute SSSP (ms)
}
```

### 2. Add to PhysicsSettings Config

```rust
// src/config/mod.rs (or wherever PhysicsSettings is defined)
pub struct PhysicsSettings {
    // ... existing fields ...
    pub sssp_enabled: bool,
    pub sssp_alpha: f32,
    pub sssp_source_nodes: Vec<u32>,
    pub sssp_update_interval: u32,
}
```

### 3. Add Validation in settings_handler.rs

```rust
fn validate_physics_settings(physics: &Value) -> Result<(), String> {
    // ... existing validations ...

    // SSSP alpha validation
    if let Some(sssp_alpha) = physics.get("ssspAlpha") {
        let val = sssp_alpha.as_f64().ok_or("ssspAlpha must be a number")?;
        if !(0.0..=1.0).contains(&val) {
            return Err("ssspAlpha must be between 0.0 and 1.0".to_string());
        }
    }

    // SSSP source nodes validation
    if let Some(source_nodes) = physics.get("ssspSourceNodes") {
        let nodes = source_nodes.as_array()
            .ok_or("ssspSourceNodes must be an array")?;
        for node in nodes {
            node.as_u64().ok_or("ssspSourceNodes must contain integers")?;
        }
    }

    Ok(())
}
```

### 4. Update SimulationParams Conversion

```rust
impl From<PhysicsSettings> for SimulationParams {
    fn from(physics: PhysicsSettings) -> Self {
        SimulationParams {
            // ... existing fields ...
            sssp_alpha: physics.sssp_alpha,
            // Note: sssp_enabled and source_nodes would need separate handling
            // as they're not part of SimParams but control ForceComputeActor
        }
    }
}
```

### 5. Handle in ForceComputeActor

```rust
impl Handler<UpdateSimulationParams> for ForceComputeActor {
    fn handle(&mut self, msg: UpdateSimulationParams) -> Self::Result {
        // Update physics params including sssp_alpha
        self.params.sssp_alpha = msg.params.sssp_alpha;

        // Enable/disable SSSP based on settings
        if msg.params.sssp_enabled != self.sssp_enabled {
            self.sssp_enabled = msg.params.sssp_enabled;
            if self.sssp_enabled {
                // Initialize SSSP computation
                self.init_sssp(msg.params.sssp_source_nodes);
            }
        }

        Ok(())
    }
}
```

## How SSSP Alpha Works

The `sssp_alpha` parameter controls how much SSSP distances influence the force-directed layout:

```rust
// In GPU kernel or force computation
fn apply_spring_force(node_a, node_b, sssp_distance_ab, sssp_alpha) {
    // Base spring rest length
    let base_rest_length = 50.0;

    // Adjust rest length based on SSSP distance
    // If nodes are far in SSSP, increase rest length
    let sssp_factor = 1.0 + sssp_distance_ab * 0.1;

    // Blend base and SSSP-adjusted rest length
    let rest_length = lerp(base_rest_length, base_rest_length * sssp_factor, sssp_alpha);

    // Apply spring force with adjusted rest length
    let force = spring_k * (distance - rest_length);
}
```

When `sssp_alpha = 0.0`: Pure force-directed layout
When `sssp_alpha = 1.0`: Full hierarchical layout based on shortest paths

## Control Center UI

The control center would add these controls:

```javascript
// In the physics settings panel
{
    label: "SSSP Layout",
    controls: [
        {
            type: "toggle",
            key: "ssspEnabled",
            label: "Enable SSSP",
            default: false
        },
        {
            type: "slider",
            key: "ssspAlpha",
            label: "SSSP Influence",
            min: 0,
            max: 1,
            step: 0.1,
            default: 0.5,
            disabled: !ssspEnabled
        },
        {
            type: "multiselect",
            key: "ssspSourceNodes",
            label: "Source Nodes",
            placeholder: "Select or enter node IDs",
            disabled: !ssspEnabled
        }
    ]
}
```

## Summary

The control flow is properly set up through REST API → Settings → Actors → GPU. We just need to:

1. Add SSSP fields to PhysicsSettingsDTO
2. Add validation for SSSP parameters
3. Pass through to SimulationParams
4. Handle in ForceComputeActor to enable/disable SSSP computation
5. Use sssp_alpha in force calculations to blend layout modes

This gives full control over SSSP from the control center without needing a separate protocol.