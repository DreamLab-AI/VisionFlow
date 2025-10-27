# Actor Integration with Hexagonal Architecture

## Overview

VisionFlow v1.0.0 integrates the Actix actor system with hexagonal architecture, providing a clean separation between business logic and actor implementation. This document describes the integration patterns, usage examples, and migration guide.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     API Layer (HTTP Handlers)               │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│              Application Services Layer                      │
│  - PhysicsService: GPU-accelerated physics simulation       │
│  - SemanticService: GPU-accelerated semantic analysis       │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    Hexagonal Ports                           │
│  - GpuPhysicsAdapter (trait)                                │
│  - GpuSemanticAnalyzer (trait)                              │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                  Actor Adapters Layer                        │
│  - ActixPhysicsAdapter: Actor → Port adapter                │
│  - ActixSemanticAdapter: Actor → Port adapter               │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    Actix Actor System                        │
│  - PhysicsActor: GPU physics computations                   │
│  - SemanticActor: GPU semantic algorithms                   │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Application Services

#### PhysicsService

Manages GPU-accelerated physics simulation through the `GpuPhysicsAdapter` port.

**Location**: `src/application/physics_service.rs`

**Key Methods**:
- `start_simulation()` - Initialize and start physics simulation
- `stop_simulation()` - Stop running simulation
- `compute_layout()` - Compute force-directed layout
- `optimize_layout()` - Optimize layout with specific algorithm
- `step()` - Perform single simulation step
- `get_gpu_status()` - Get GPU device information
- `get_statistics()` - Get physics statistics

**Example**:
```rust
use visionflow::application::physics_service::{PhysicsService, SimulationParams};
use std::sync::Arc;

let physics_service = Arc::new(PhysicsService::new(adapter, event_bus));

// Start simulation
let params = SimulationParams {
    profile_name: "force-directed".to_string(),
    physics_params: PhysicsParameters::default(),
    auto_stop_on_convergence: true,
};

let sim_id = physics_service
    .start_simulation(graph, params)
    .await?;

// Get status
let status = physics_service.get_gpu_status().await?;
println!("GPU: {}", status.device_name);
```

#### SemanticService

Manages GPU-accelerated semantic analysis through the `GpuSemanticAnalyzer` port.

**Location**: `src/application/semantic_service.rs`

**Key Methods**:
- `detect_communities()` - Detect graph communities
- `compute_centrality()` - Compute node centrality scores
- `compute_shortest_paths()` - Find shortest paths
- `compute_pagerank()` - Compute PageRank scores
- `generate_semantic_constraints()` - Generate layout constraints
- `get_statistics()` - Get analysis statistics

**Example**:
```rust
use visionflow::application::semantic_service::SemanticService;

let semantic_service = Arc::new(SemanticService::new(adapter, event_bus));

// Detect communities
semantic_service.initialize(graph).await?;
let communities = semantic_service.detect_communities_louvain().await?;

println!("Found {} communities", communities.clusters.len());
println!("Modularity: {}", communities.modularity);

// Compute PageRank
let pagerank = semantic_service.compute_pagerank(0.85, 100).await?;
```

### 2. Actor Lifecycle Management

**Location**: `src/actors/lifecycle.rs`

Manages actor startup, shutdown, health monitoring, and supervision.

**Features**:
- Automatic actor initialization
- Graceful shutdown sequence
- Health monitoring with configurable intervals
- Actor supervision strategies
- Automatic restart on failure

**Example**:
```rust
use visionflow::actors::lifecycle::{ActorLifecycleManager, SupervisionStrategy};

let mut manager = ActorLifecycleManager::new();

// Initialize all actors
manager.initialize().await?;

// Check health
if manager.is_healthy() {
    println!("All actors running");
}

// Graceful shutdown
manager.shutdown().await?;
```

### 3. Event-Driven Coordination

**Location**: `src/actors/event_coordination.rs`

Coordinates actors through domain events for reactive behavior.

**Event Handlers**:
- `GraphSavedEvent` → Trigger physics update
- `OntologyImportedEvent` → Trigger semantic inference
- `PositionsUpdatedEvent` → Broadcast to WebSocket clients
- `NodeAddedEvent` → Invalidate semantic cache
- `EdgeAddedEvent` → Invalidate pathfinding cache

**Example**:
```rust
use visionflow::actors::event_coordination::EventCoordinator;

let coordinator = EventCoordinator::new(
    physics_service,
    semantic_service,
    event_bus,
    graph_data,
);

coordinator.initialize().await;

// Events are automatically handled
// e.g., when graph is saved, physics is updated automatically
```

### 4. HTTP Handlers

#### Physics Handlers

**Location**: `src/handlers/physics_handler.rs`

**Endpoints**:
- `POST /api/physics/start` - Start simulation
- `POST /api/physics/stop` - Stop simulation
- `GET /api/physics/status` - Get simulation status
- `POST /api/physics/optimize` - Optimize layout
- `POST /api/physics/step` - Perform single step
- `POST /api/physics/forces/apply` - Apply external forces
- `POST /api/physics/nodes/pin` - Pin nodes
- `POST /api/physics/nodes/unpin` - Unpin nodes
- `POST /api/physics/parameters` - Update parameters
- `POST /api/physics/reset` - Reset simulation

**Example Request**:
```bash
curl -X POST http://localhost:8080/api/physics/start \
  -H "Content-Type: application/json" \
  -d '{
    "profile_name": "force-directed",
    "time_step": 0.016,
    "damping": 0.8,
    "max_iterations": 1000
  }'
```

#### Semantic Handlers

**Location**: `src/handlers/semantic_handler.rs`

**Endpoints**:
- `POST /api/semantic/communities` - Detect communities
- `POST /api/semantic/centrality` - Compute centrality
- `POST /api/semantic/shortest-path` - Find shortest paths
- `POST /api/semantic/constraints/generate` - Generate constraints
- `GET /api/semantic/statistics` - Get statistics
- `POST /api/semantic/cache/invalidate` - Invalidate cache

**Example Request**:
```bash
curl -X POST http://localhost:8080/api/semantic/communities \
  -H "Content-Type: application/json" \
  -d '{
    "algorithm": "louvain"
  }'
```

## Backward Compatibility

**Location**: `src/actors/backward_compat.rs`

Provides compatibility layer for legacy actor code with deprecation warnings.

**Features**:
- Wrapper classes for legacy message types
- Automatic routing to new service layer
- Deprecation warnings
- Migration helper utilities

**Example**:
```rust
use visionflow::actors::backward_compat::LegacyActorCompat;

// Enable legacy mode (suppress warnings)
std::env::set_var("VISIONFLOW_LEGACY_ACTORS", "true");

// Check if legacy mode is enabled
if LegacyActorCompat::legacy_mode_enabled() {
    // Use legacy actor messages (deprecated)
}

// Print migration guide
MigrationHelper::print_migration_guide();
```

## Migration Guide

### From Direct Actors to Services

#### Before (Deprecated):
```rust
// Direct actor messaging (deprecated)
use actix::prelude::*;

let physics_actor = PhysicsActor::default().start();
let msg = StartPhysicsMessage { ... };
physics_actor.send(msg).await?;
```

#### After (Recommended):
```rust
// Service through hexagonal ports
use visionflow::application::physics_service::PhysicsService;

let physics_service = PhysicsService::new(adapter, event_bus);
physics_service.start_simulation(graph, params).await?;
```

### Migration Checklist

1. **Replace Direct Actor Calls**
   - [ ] Replace `PhysicsActor::send()` with `PhysicsService` methods
   - [ ] Replace `SemanticActor::send()` with `SemanticService` methods
   - [ ] Update message types to service parameters

2. **Update Dependency Injection**
   - [ ] Inject `PhysicsService` instead of `Addr<PhysicsActor>`
   - [ ] Inject `SemanticService` instead of `Addr<SemanticActor>`
   - [ ] Configure services in application startup

3. **Event Handling**
   - [ ] Subscribe to domain events instead of actor messages
   - [ ] Use `EventCoordinator` for reactive behavior
   - [ ] Update event listeners

4. **Testing**
   - [ ] Replace actor test harnesses with service mocks
   - [ ] Use port traits for test doubles
   - [ ] Update integration tests

## Configuration

### Actor System Initialization

```rust
// In main.rs or application startup

use visionflow::actors::lifecycle::initialize_actor_system;
use visionflow::actors::event_coordination::initialize_event_coordinator;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Initialize actor system
    initialize_actor_system().await?;

    // Initialize event coordination
    initialize_event_coordinator(
        physics_service,
        semantic_service,
        event_bus,
        graph_data,
    ).await;

    // Start HTTP server
    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(physics_service.clone()))
            .app_data(web::Data::new(semantic_service.clone()))
            .configure(physics_handler::configure_routes)
            .configure(semantic_handler::configure_routes)
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}
```

### Environment Variables

- `VISIONFLOW_LEGACY_ACTORS=true` - Enable legacy actor mode (suppress warnings)
- `ACTOR_HEALTH_CHECK_INTERVAL=30` - Health check interval in seconds

## Best Practices

1. **Always Use Services**
   - Access actors through `PhysicsService` and `SemanticService`
   - Never call actors directly from handlers
   - Use dependency injection for testability

2. **Event-Driven Architecture**
   - Publish domain events for state changes
   - Subscribe to events for reactive behavior
   - Keep event handlers idempotent

3. **Error Handling**
   - Handle adapter errors at service layer
   - Convert adapter errors to HTTP responses
   - Log errors with proper context

4. **Testing**
   - Mock port traits, not actors
   - Test services independently
   - Use integration tests for end-to-end flows

5. **Performance**
   - Reuse service instances (Arc)
   - Batch operations when possible
   - Monitor GPU memory usage

## Troubleshooting

### Common Issues

1. **Actor Not Responding**
   - Check actor lifecycle manager health
   - Review supervision logs
   - Verify actor is initialized

2. **GPU Memory Errors**
   - Check GPU status endpoint
   - Monitor memory usage
   - Reduce batch sizes

3. **Event Not Triggering**
   - Verify event coordinator is initialized
   - Check event bus subscriptions
   - Review event handler logs

### Debugging

Enable debug logging:
```rust
env_logger::Builder::from_env(
    env_logger::Env::default().default_filter_or("debug")
).init();
```

Check actor health:
```bash
curl http://localhost:8080/api/physics/status
curl http://localhost:8080/api/semantic/statistics
```

## Performance Considerations

### GPU Acceleration

- Physics and semantic operations run on GPU
- Monitor GPU utilization and memory
- Use batch operations for better throughput

### Actor Concurrency

- Actors process messages sequentially
- Use multiple actor instances for parallelism
- Consider workload distribution

### Caching

- Semantic analyzer caches pathfinding results
- Invalidate cache after graph modifications
- Monitor cache hit rates

## Future Improvements

1. **Dynamic Actor Scaling**
   - Auto-scale actors based on load
   - Load balancing across actor instances
   - Resource-aware scheduling

2. **Distributed Actors**
   - Multi-node actor deployment
   - Remote actor communication
   - Fault tolerance across nodes

3. **Advanced Monitoring**
   - Prometheus metrics export
   - Performance profiling
   - Distributed tracing

## References

- [Hexagonal Architecture](https://alistair.cockburn.us/hexagonal-architecture/)
- [Actix Documentation](https://actix.rs/)
- [Domain-Driven Design](https://www.domainlanguage.com/ddd/)
- [Event-Driven Architecture](https://martinfowler.com/articles/201701-event-driven.html)

## Support

For questions or issues:
- GitHub Issues: https://github.com/visionflow/visionflow/issues
- Documentation: https://docs.visionflow.dev
- Migration Guide: https://docs.visionflow.dev/migration/actors-to-adapters
