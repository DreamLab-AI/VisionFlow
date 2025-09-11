# Case Conversion Architecture

*[Architecture](../index.md)*

## Overview

VisionFlow maintains strict case conventions across different layers of the application to ensure consistency and prevent naming conflicts. This document outlines the naming conventions and automatic conversion mechanisms.

## Naming Conventions by Layer

```mermaid
flowchart LR
    subgraph "TypeScript/React"
        Client[camelCase<br/>springK, repelK<br/>autoBalance]
    end
    
    subgraph "REST API"
        API[Conversion Layer<br/>↓ camelCase → snake_case<br/>↑ snake_case → camelCase]
    end
    
    subgraph "Rust Backend"
        Server[snake_case<br/>spring_k, repel_k<br/>auto_balance]
    end
    
    subgraph "GPU/CUDA"
        GPU[snake_case<br/>Matches Rust]
    end
    
    Client < --> API
    API < --> Server
    Server < --> GPU
```

## Case Convention Rules

### TypeScript/React Client
All client-side code uses **camelCase**:
```typescript
interface PhysicsSettings {
    autoBalance: boolean;
    autoBalanceIntervalMs: number;
    springK: number;
    repelK: number;
    attractionK: number;
    maxVelocity: number;
    enableBounds: boolean;
    boundsSize: number;
}
```

### Rust Backend
All server-side code uses **snake_case**:
```rust
pub struct PhysicsSettings {
    pub auto_balance: bool,
    pub auto_balance_interval_ms: u32,
    pub spring_k: f32,
    pub repel_k: f32,
    pub attraction_k: f32,
    pub max_velocity: f32,
    pub enable_bounds: bool,
    pub bounds_size: f32,
}
```

### Settings YAML
Configuration files use **snake_case**:
```yaml
physics:
  auto_balance: true
  auto_balance_interval_ms: 500
  spring_k: 0.5
  repel_k: 0.1
  max_velocity: 0.5
```

## Conversion Implementation

### Server → Client (GET /api/settings)

```mermasequenceDiagram
    participant Client
    participant API
    participant Settings
    
    Client->>API: GET /api/settings
    API->>Settings: GetSettings
    Settings --> >API: AppFullSettings (snake_case)
    API->>API: to_camel_case_json()
    API --> >Client: JSON (camelCase)se)
```

The conversion happens in `config/mod.rs`:
```rust
pub fn to_camel_case_json(&self) -> Result<Value, String> {
    let snake_json = serde_json::to_value(&self)?;
    Ok(keys_to_camel_case(snake_json))
}
```

### Client → Server (POST /api/settings)

```msequenceDiagram
    participant Client
    participant API
    participant Validator
    participant Settings
    
    Client->>API: POST /api/settings (camelCase)
    API->>API: keys_to_snake_case()
    API->>Validator: validate_settings_update()
    Validator --> >API: Valid ✓
    API->>Settings: UpdateSettings (snake_case)
    Settings --> >API: Success
    API --> >Client: 200 OK 200 OK
```

The conversion happens in `handlers/settings_handler.rs`:
```rust
let snake_update = keys_to_snake_case(update.clone());
app_settings.merge_update(snake_update)?;
```

## Common Field Mappings

| Client (camelCase) | Server (snake_case) | Description |
|--------------------|---------------------|-------------|
| autoBalance | auto_balance | Neural auto-tuning toggle |
| autoBalanceIntervalMs | auto_balance_interval_ms | Check interval |
| springK | spring_k | Spring force constant |
| repelK | repel_k | Repulsion force constant |
| attractionK | attraction_k | Attraction force |
| maxVelocity | max_velocity | Velocity limit |
| enableBounds | enable_bounds | Boundary toggle |
| boundsSize | bounds_size | Boundary dimensions |
| separationRadius | separation_radius | Node separation |
| stressWeight | stress_weight | Stress optimisation |
| minDistance | min_distance | Minimum node distance |
| maxRepulsionDist | max_repulsion_dist | Repulsion cutoff |
| warmupIterations | warmup_iterations | Warmup cycles |
| coolingRate | cooling_rate | Annealing rate |

## UI Component Configuration

The `IntegratedControlPanel.tsx` must use camelCase paths:
```typescript
// ✅ Correct
{ key: 'springK', path: 'visualisation.graphs.logseq.physics.springK' }

// ❌ Wrong - will result in undefined values
{ key: 'spring_k', path: 'visualisation.graphs.logseq.physics.spring_k' }
```

## Validation Considerations

### Server-Side Validation
Validation happens **after** conversion to snake_case:
```rust
fn validate_physics_settings(physics: &Value) -> Result<(), String> {
    // Expects snake_case fields
    if let Some(auto_balance) = physics.get("autoBalance") {
        // This validates the camelCase field from client
    }
}
```

### Type Safety
The Rust backend uses strongly-typed structs with serde attributes:
```rust
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]  // Forces snake_case in JSON
pub struct PhysicsSettings {
    pub auto_balance: bool,
    // ...
}
```

## Debugging Case Issues

### Common Symptoms
1. **Undefined values in UI**: Field names don't match camelCase convention
2. **Validation errors**: Server expects different case than provided
3. **Settings not persisting**: Conversion failure during save

### Debug Checklist
1. Check browser console for undefined physics values
2. Verify REST API response has camelCase fields
3. Check server logs for conversion errors
4. Ensure UI paths use correct case

### Logging
Enable debug logging to trace conversions:
```rust
debug!("Before conversion: {:?}", update);
let converted = keys_to_snake_case(update);
debug!("After conversion: {:?}", converted);
```

## Best Practices

1. **Always use camelCase in TypeScript/React**
2. **Always use snake_case in Rust**
3. **Let the API layer handle conversion**
4. **Never mix cases within a layer**
5. **Document any exceptions clearly**
6. **Test both directions of conversion**

## Testing Conversion

### Unit Test Example
```rust
#[test]
fn test_case_conversion() {
    let camel = json!({
        "autoBalance": true,
        "springK": 0.5
    });
    
    let snake = keys_to_snake_case(camel);
    assert_eq!(snake["auto_balance"], true);
    assert_eq!(snake["spring_k"], 0.5);
}
```

### Integration Test
```typescript
// Client test
expect(settings.physics.autoBalance).toBeDefined();
expect(settings.physics.springK).toBeGreaterThan(0);
```

## Migration Notes

When adding new fields:
1. Add snake_case field to Rust struct
2. Update validation to handle both cases during transition
3. Add camelCase field to TypeScript interface
4. Update UI component paths to use camelCase
5. Test round-trip conversion



## See Also

- [Configuration Architecture](../server/config.md)
- [Feature Access Control](../server/feature-access.md)
- [GPU Compute Architecture](../server/gpu-compute.md)

## Related Topics

- [Agent Visualisation Architecture](../agent-visualization-architecture.md)
- [Architecture Documentation](../architecture/README.md)
- [Architecture Migration Guide](../architecture/migration-guide.md)
- [Bots Visualisation Architecture](../architecture/bots-visualization.md)
- [Bots/VisionFlow System Architecture](../architecture/bots-visionflow-system.md)
- [ClaudeFlowActor Architecture](../architecture/claude-flow-actor.md)
- [Client Architecture](../client/architecture.md)
- [Decoupled Graph Architecture](../technical/decoupled-graph-architecture.md)
- [Dynamic Agent Architecture (DAA) Setup Guide](../architecture/daa-setup-guide.md)
- [GPU Compute Improvements & Troubleshooting Guide](../architecture/gpu-compute-improvements.md)
- [MCP Connection Architecture](../architecture/mcp-connection.md)
- [MCP Integration Architecture](../architecture/mcp-integration.md)
- [MCP WebSocket Relay Architecture](../architecture/mcp-websocket-relay.md)
- [Managing the Claude-Flow System](../architecture/managing-claude-flow.md)
- [Parallel Graph Architecture](../architecture/parallel-graphs.md)
- [Server Architecture](../server/architecture.md)
- [Settings Architecture Analysis Report](../architecture_analysis_report.md)
- [VisionFlow Component Architecture](../architecture/components.md)
- [VisionFlow Data Flow Architecture](../architecture/data-flow.md)
- [VisionFlow GPU Compute Integration](../architecture/gpu-compute.md)
- [VisionFlow GPU Migration Architecture](../architecture/visionflow-gpu-migration.md)
- [VisionFlow System Architecture Overview](../architecture/index.md)
- [VisionFlow System Architecture](../architecture/system-overview.md)
- [arch-system-design](../reference/agents/architecture/system-design/arch-system-design.md)
- [architecture](../reference/agents/sparc/architecture.md)
