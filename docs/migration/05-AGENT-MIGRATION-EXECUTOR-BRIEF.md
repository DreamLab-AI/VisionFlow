# ⚙️ Agent 5: Migration Executor - Mission Brief

**Agent ID:** migration-executor
**Type:** Implementation Engineer
**Priority:** CRITICAL
**Compute Units:** 30
**Memory Quota:** 1024 MB

## Mission Statement

Execute hexagonal migration in 4 phases. Systematically extract functionality from GraphServiceActor into CQRS command/query handlers, implement event sourcing for real-time updates, and migrate physics to domain services. This is the core implementation work.

## Migration Phases

### Phase 1: Extract Read Operations (Days 1-2)
**Goal:** All GET operations through query handlers

**Implementation Steps:**

1. **Create Query Handlers**
   - File: `src/application/knowledge_graph/queries.rs`
   - Add query handlers for all read operations:

```rust
// src/application/knowledge_graph/queries.rs

use crate::ports::knowledge_graph_repository::KnowledgeGraphRepository;
use crate::models::node::Node;
use std::sync::Arc;

pub struct GetNodeQuery {
    pub node_id: u32,
}

pub struct ListNodesQuery {
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

pub struct GetGraphStatsQuery;

pub struct KnowledgeGraphQueryHandler {
    repository: Arc<dyn KnowledgeGraphRepository>,
}

impl KnowledgeGraphQueryHandler {
    pub fn new(repository: Arc<dyn KnowledgeGraphRepository>) -> Self {
        Self { repository }
    }

    pub async fn handle_get_node(&self, query: GetNodeQuery) -> Result<Node, VisionFlowError> {
        self.repository.get_node(query.node_id).await
    }

    pub async fn handle_list_nodes(&self, query: ListNodesQuery) -> Result<Vec<Node>, VisionFlowError> {
        self.repository.list_nodes(query.limit, query.offset).await
    }

    pub async fn handle_get_stats(&self, _query: GetGraphStatsQuery) -> Result<GraphStats, VisionFlowError> {
        self.repository.get_graph_statistics().await
    }
}
```

2. **Update API Handlers**
   - File: `src/handlers/api_handler/graph/mod.rs`
   - Route GET requests through query handlers:

```rust
// Before (using actor):
pub async fn get_nodes(
    state: web::Data<AppState>,
) -> Result<HttpResponse, VisionFlowError> {
    let result = state.graph_actor.send(GetNodes).await??;
    Ok(HttpResponse::Ok().json(result))
}

// After (using query handler):
pub async fn get_nodes(
    state: web::Data<AppState>,
) -> Result<HttpResponse, VisionFlowError> {
    let query = ListNodesQuery { limit: None, offset: None };
    let result = state.query_handler.handle_list_nodes(query).await?;
    Ok(HttpResponse::Ok().json(result))
}
```

3. **Maintain Backward Compatibility**
   - Keep GraphServiceActor running
   - Add feature flag: `--features legacy-actors`
   - Dual read path: Try new handler, fall back to actor

4. **Validation**
```bash
# Test all GET endpoints
curl http://localhost:8080/api/graph/nodes
curl http://localhost:8080/api/graph/nodes/123
curl http://localhost:8080/api/graph/stats

# Verify responses match previous behavior
```

**Phase 1 Success Criteria:**
✅ All GET endpoints use query handlers
✅ Tests pass with new handlers
✅ GraphServiceActor still available as fallback
✅ No performance regression

---

### Phase 2: Extract Write Operations (Days 3-4)
**Goal:** All POST/PUT/DELETE through command handlers

**Implementation Steps:**

1. **Create Command Handlers**
   - File: `src/application/knowledge_graph/directives.rs`
   - Implement commands for all mutations:

```rust
// src/application/knowledge_graph/directives.rs

use crate::ports::knowledge_graph_repository::KnowledgeGraphRepository;
use crate::models::node::Node;
use std::sync::Arc;

pub struct AddNodeCommand {
    pub node_data: NodeCreationData,
}

pub struct UpdateNodeCommand {
    pub node_id: u32,
    pub updates: NodeUpdateData,
}

pub struct RemoveNodeCommand {
    pub node_id: u32,
}

pub struct KnowledgeGraphCommandHandler {
    repository: Arc<dyn KnowledgeGraphRepository>,
    event_publisher: Arc<dyn EventPublisher>,
}

impl KnowledgeGraphCommandHandler {
    pub async fn handle_add_node(&self, command: AddNodeCommand) -> Result<Node, VisionFlowError> {
        // Validate command
        self.validate_node_data(&command.node_data)?;

        // Execute command
        let node = self.repository.create_node(command.node_data).await?;

        // Publish event
        self.event_publisher.publish(NodeAddedEvent {
            node_id: node.id,
            node_data: node.clone(),
        }).await?;

        Ok(node)
    }

    pub async fn handle_update_node(&self, command: UpdateNodeCommand) -> Result<Node, VisionFlowError> {
        let node = self.repository.update_node(command.node_id, command.updates).await?;

        self.event_publisher.publish(NodeUpdatedEvent {
            node_id: node.id,
            changes: command.updates,
        }).await?;

        Ok(node)
    }

    pub async fn handle_remove_node(&self, command: RemoveNodeCommand) -> Result<(), VisionFlowError> {
        self.repository.delete_node(command.node_id).await?;

        self.event_publisher.publish(NodeRemovedEvent {
            node_id: command.node_id,
        }).await?;

        Ok(())
    }
}
```

2. **Update API Handlers**
```rust
// src/handlers/api_handler/graph/mod.rs

pub async fn create_node(
    state: web::Data<AppState>,
    body: web::Json<NodeCreationData>,
) -> Result<HttpResponse, VisionFlowError> {
    let command = AddNodeCommand { node_data: body.into_inner() };
    let node = state.command_handler.handle_add_node(command).await?;
    Ok(HttpResponse::Created().json(node))
}
```

3. **Transaction Management**
   - Implement unit of work pattern
   - Ensure atomicity of command + event
   - Rollback on failure

**Phase 2 Success Criteria:**
✅ All POST/PUT/DELETE use command handlers
✅ Events published for all state changes
✅ Transaction guarantees maintained
✅ Tests verify behavior unchanged

---

### Phase 3: Event Sourcing for WebSocket (Days 5-6)
**Goal:** Real-time updates via domain events

**Implementation Steps:**

1. **Create Event Publisher**
   - File: `src/infrastructure/event_publisher.rs`

```rust
use tokio::sync::broadcast;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub enum DomainEvent {
    NodeAdded(NodeAddedEvent),
    NodeUpdated(NodeUpdatedEvent),
    NodeRemoved(NodeRemovedEvent),
    PhysicsStateUpdated(PhysicsStateUpdatedEvent),
    // ... more events
}

pub trait EventPublisher: Send + Sync {
    async fn publish(&self, event: DomainEvent) -> Result<(), VisionFlowError>;
    fn subscribe(&self) -> broadcast::Receiver<DomainEvent>;
}

pub struct BroadcastEventPublisher {
    sender: broadcast::Sender<DomainEvent>,
}

impl BroadcastEventPublisher {
    pub fn new(capacity: usize) -> Self {
        let (sender, _) = broadcast::channel(capacity);
        Self { sender }
    }
}

impl EventPublisher for BroadcastEventPublisher {
    async fn publish(&self, event: DomainEvent) -> Result<(), VisionFlowError> {
        self.sender.send(event)
            .map_err(|_| VisionFlowError::EventPublishFailed)?;
        Ok(())
    }

    fn subscribe(&self) -> broadcast::Receiver<DomainEvent> {
        self.sender.subscribe()
    }
}
```

2. **Update WebSocket Handler**
   - File: `src/handlers/socket_flow_handler.rs`
   - Subscribe to events instead of actor messages:

```rust
// WebSocket handler subscribes to events
let mut event_receiver = event_publisher.subscribe();

tokio::spawn(async move {
    while let Ok(event) = event_receiver.recv().await {
        match event {
            DomainEvent::NodeAdded(e) => {
                // Convert to binary protocol
                let binary_frame = create_binary_frame_for_node(&e.node_data);
                // Send to WebSocket client
                ws_sender.send(binary_frame).await;
            },
            DomainEvent::PhysicsStateUpdated(e) => {
                let positions_frame = create_positions_update(&e.positions);
                ws_sender.send(positions_frame).await;
            },
            // ... handle other events
        }
    }
});
```

3. **Remove Direct Actor Communication**
   - Delete actor → WebSocket message passing
   - All updates flow through events
   - Decouple components

**Phase 3 Success Criteria:**
✅ WebSocket receives updates via events
✅ Binary protocol still works
✅ Real-time updates have < 50ms latency
✅ No direct actor dependencies in WebSocket handler

---

### Phase 4: Domain Services for Physics (Days 7-8)
**Goal:** Physics as pluggable service

**Implementation Steps:**

1. **Implement Physics Simulator Port**
   - File already exists: `src/ports/physics_simulator.rs`
   - Create concrete implementation:

```rust
// src/infrastructure/gpu_physics_service.rs

use crate::ports::physics_simulator::PhysicsSimulator;
use crate::ports::gpu_physics_adapter::GPUPhysicsAdapter;
use std::sync::Arc;

pub struct GPUPhysicsService {
    gpu_adapter: Arc<dyn GPUPhysicsAdapter>,
    state: Arc<Mutex<PhysicsState>>,
}

impl PhysicsSimulator for GPUPhysicsService {
    async fn step(&self, delta_time: f32) -> Result<PhysicsStepResult, VisionFlowError> {
        let mut state = self.state.lock().await;

        // Use GPU adapter for computation
        let forces = self.gpu_adapter.compute_forces(&state.positions).await?;
        let new_positions = self.apply_forces(&state.positions, &forces, delta_time);

        state.positions = new_positions.clone();

        // Publish event
        self.event_publisher.publish(DomainEvent::PhysicsStateUpdated(
            PhysicsStateUpdatedEvent {
                positions: new_positions.clone(),
                timestamp: Instant::now(),
            }
        )).await?;

        Ok(PhysicsStepResult { positions: new_positions })
    }

    async fn update_params(&self, params: SimulationParams) -> Result<(), VisionFlowError> {
        self.gpu_adapter.update_parameters(params).await
    }
}
```

2. **Inject Service into Application Layer**
   - Update `AppState` to include physics service
   - Remove physics orchestrator actor dependency

3. **Create Physics Command Handlers**
```rust
// src/application/physics/directives.rs

pub struct UpdatePhysicsParamsCommand {
    pub params: SimulationParams,
}

pub struct StepSimulationCommand {
    pub delta_time: f32,
}

pub struct PhysicsCommandHandler {
    simulator: Arc<dyn PhysicsSimulator>,
}

impl PhysicsCommandHandler {
    pub async fn handle_update_params(&self, cmd: UpdatePhysicsParamsCommand) -> Result<(), VisionFlowError> {
        self.simulator.update_params(cmd.params).await
    }

    pub async fn handle_step(&self, cmd: StepSimulationCommand) -> Result<PhysicsStepResult, VisionFlowError> {
        self.simulator.step(cmd.delta_time).await
    }
}
```

**Phase 4 Success Criteria:**
✅ Physics runs via domain service
✅ GPU adapter properly isolated
✅ Physics orchestrator actor removed
✅ Simulation performance unchanged

---

## Validation After Each Phase

**Automated Tests:**
```bash
# Run full test suite
cargo test

# Run integration tests
cargo test --test integration_tests

# Performance benchmarks
cargo bench
```

**Manual Validation:**
1. Start application
2. Sync GitHub repository
3. Verify 316 nodes created
4. Check WebSocket real-time updates
5. Verify physics simulation running
6. Test all API endpoints

## Deliverables

Create: `/home/devuser/workspace/project/docs/migration/migration-execution-log.md`

**Required Sections:**
1. **Phase-by-Phase Execution Log**
   - What was changed
   - Files modified
   - Tests added/updated
   - Validation results
2. **Code Changes Summary**
   - Lines added/removed
   - New files created
   - Deprecated code paths
3. **Performance Metrics**
   - Before/after benchmarks
   - Latency measurements
   - Memory usage comparison
4. **Issues Encountered**
   - Problems found
   - Solutions applied
   - Technical debt created

## Memory Storage

Store progress under: `hive-coordination/migration/phase_status`

**JSON Structure:**
```json
{
  "current_phase": 3,
  "phases": [
    {
      "id": 1,
      "name": "Extract Read Operations",
      "status": "completed",
      "files_modified": 12,
      "tests_added": 8,
      "duration_hours": 16,
      "validation_passed": true
    },
    ...
  ],
  "total_files_modified": 45,
  "total_tests_added": 32,
  "blockers": [],
  "next_steps": "Begin Phase 4: Domain Services for Physics"
}
```

## Success Criteria

✅ All 4 phases completed
✅ GraphServiceActor no longer required
✅ All functionality migrated to hexagonal layer
✅ Tests pass with new architecture
✅ Performance meets or exceeds baseline
✅ GitHub sync works (316 nodes)
✅ WebSocket real-time updates functional
✅ Physics simulation operational

---
*Assigned by Queen Coordinator - Highest Priority*
