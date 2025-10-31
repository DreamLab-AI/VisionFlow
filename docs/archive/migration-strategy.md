# Migration Strategy: Actor to Hexagonal/CQRS
**4-Phase Migration Plan with Zero Downtime**

---

## Overview

This document provides a detailed, actionable migration plan for transitioning from the monolithic `GraphServiceActor` to a clean hexagonal/CQRS architecture.

### Goals
- ✅ Zero downtime during migration
- ✅ No data loss
- ✅ Maintain backward compatibility during transition
- ✅ Fix GitHub sync cache bug (316 nodes issue)
- ✅ Improve testability and maintainability

### Timeline
- **Total Duration**: 6 weeks
- **Phase 1**: 1 week (Read operations)
- **Phase 2**: 2 weeks (Write operations + events)
- **Phase 3**: 2 weeks (Real-time features)
- **Phase 4**: 1 week (Legacy cleanup)

---

## Phase 1: Read Operations (SAFEST)
**Duration**: 1 week
**Risk**: Low
**Goal**: Migrate all query operations from actor to CQRS

### Why Start with Reads?
1. **No state changes** - safest operations
2. **Easy rollback** - can run actor in parallel
3. **Performance testing** - validate query performance
4. **Team familiarity** - learn CQRS patterns safely

### Step 1.1: Create Query Infrastructure (Day 1)
**Estimated Time**: 4 hours

**Files to Create**:
```bash
src/
├── application/
│   └── graph/
│       ├── mod.rs                     # Module exports
│       ├── queries.rs                 # Query DTOs
│       └── query_handlers.rs          # Query handlers
├── ports/
│   └── graph_repository.rs            # Repository trait
└── adapters/
    └── sqlite_graph_repository.rs     # SQLite implementation
```

**Code to Write**:

```rust
// src/application/graph/queries.rs
pub struct GetGraphDataQuery {
    pub include_edges: bool,
    pub filter: Option<GraphFilter>,
}

pub struct GetNodeByIdQuery {
    pub node_id: u32,
}

pub struct GetGraphStatisticsQuery {
    pub include_detailed: bool,
}
```

```rust
// src/application/graph/query_handlers.rs
pub struct GetGraphDataQueryHandler {
    graph_repo: Arc<dyn GraphRepository>,
}

impl GetGraphDataQueryHandler {
    pub async fn handle(&self, query: GetGraphDataQuery) -> Result<GraphData, String> {
        let graph_data = self.graph_repo.get_graph().await?;
        // Apply filters if needed
        Ok(graph_data)
    }
}
```

**Validation**:
```bash
cargo build --release
cargo test application::graph::query_handlers
```

---

### Step 1.2: Implement Repository Port (Day 2)
**Estimated Time**: 6 hours

**Extend existing `SqliteKnowledgeGraphRepository`**:

```rust
// src/ports/graph_repository.rs
#[async_trait]
pub trait GraphRepository: Send + Sync {
    async fn get_graph(&self) -> Result<GraphData, String>;
    async fn get_node(&self, node_id: u32) -> Result<Option<Node>, String>;
    async fn get_edges(&self) -> Result<Vec<Edge>, String>;
    async fn get_graph_statistics(&self) -> Result<GraphStatistics, String>;
}
```

```rust
// src/adapters/sqlite_graph_repository.rs
pub struct SqliteGraphRepository {
    db_path: String,
}

#[async_trait]
impl GraphRepository for SqliteGraphRepository {
    async fn get_graph(&self) -> Result<GraphData, String> {
        // Use existing SqliteKnowledgeGraphRepository code
        // This already exists! Just wrap it with new trait
        let conn = Connection::open(&self.db_path)
            .map_err(|e| format!("Failed to open database: {}", e))?;

        let nodes = self.load_nodes(&conn)?;
        let edges = self.load_edges(&conn)?;

        Ok(GraphData { nodes, edges })
    }
}
```

**Validation**:
```bash
cargo test adapters::sqlite_graph_repository
```

---

### Step 1.3: Migrate API Handlers (Day 3)
**Estimated Time**: 8 hours

**Before**:
```rust
// src/handlers/api_handler/graph_data.rs (OLD)
pub async fn get_graph_data(
    state: web::Data<AppState>,
) -> Result<HttpResponse, Error> {
    let graph_data = state.graph_service_actor
        .send(GetGraphData)
        .await??;
    Ok(HttpResponse::Ok().json(graph_data))
}
```

**After**:
```rust
// src/handlers/api_handler/graph_data.rs (NEW)
pub async fn get_graph_data_v2(
    query_handler: web::Data<Arc<GetGraphDataQueryHandler>>,
) -> Result<HttpResponse, Error> {
    let query = GetGraphDataQuery {
        include_edges: true,
        filter: None,
    };
    let graph_data = query_handler.handle(query).await
        .map_err(actix_web::error::ErrorInternalServerError)?;
    Ok(HttpResponse::Ok().json(graph_data))
}
```

**Migration Strategy**:
1. Keep old endpoint: `GET /api/graph/data` (uses actor)
2. Add new endpoint: `GET /api/graph/data/v2` (uses CQRS)
3. Run A/B testing to compare results
4. Once validated, switch old endpoint to CQRS

**Validation**:
```bash
# Test both endpoints return same data
curl http://localhost:8080/api/graph/data > old.json
curl http://localhost:8080/api/graph/data/v2 > new.json
diff old.json new.json  # Should be identical!
```

---

### Step 1.4: Performance Testing (Day 4)
**Estimated Time**: 6 hours

**Load Testing Script**:
```bash
# test/load_test.sh
#!/bin/bash
echo "Testing old actor endpoint..."
ab -n 1000 -c 10 http://localhost:8080/api/graph/data > old_perf.txt

echo "Testing new CQRS endpoint..."
ab -n 1000 -c 10 http://localhost:8080/api/graph/data/v2 > new_perf.txt

echo "Comparing results..."
grep "Requests per second" old_perf.txt new_perf.txt
grep "Time per request" old_perf.txt new_perf.txt
```

**Expected Results**:
- CQRS should be **faster** (no actor overhead)
- Latency p95 < 50ms
- No errors

**If CQRS is slower**:
- Add SQLite indexing
- Implement caching layer
- Use connection pooling

---

### Step 1.5: Switch Production Traffic (Day 5)
**Estimated Time**: 4 hours

**Gradual Rollout**:
```rust
// Feature flag approach
pub async fn get_graph_data(
    state: web::Data<AppState>,
    query_handler: web::Data<Arc<GetGraphDataQueryHandler>>,
) -> Result<HttpResponse, Error> {
    // Use feature flag to gradually roll out
    if state.config.use_cqrs_queries {
        get_graph_data_cqrs(query_handler).await
    } else {
        get_graph_data_actor(state).await
    }
}
```

**Rollout Plan**:
1. 5% traffic → CQRS (monitor for 1 hour)
2. 25% traffic → CQRS (monitor for 2 hours)
3. 50% traffic → CQRS (monitor for 4 hours)
4. 100% traffic → CQRS (full migration)

**Monitoring**:
- Track error rates
- Track latency (p50, p95, p99)
- Track memory usage
- Track database connection pool

---

## Phase 2: Write Operations (REQUIRES EVENTS)
**Duration**: 2 weeks
**Risk**: Medium
**Goal**: Migrate all command operations with event sourcing

### Step 2.1: Implement Event Bus (Week 2, Day 1-2)
**Estimated Time**: 16 hours

**Files to Create**:
```bash
src/
├── domain/
│   └── events.rs              # Domain events
├── infrastructure/
│   ├── event_bus.rs          # Event bus trait + impl
│   ├── event_store.rs        # Event store (optional)
│   └── mod.rs
```

**Event Bus Implementation**:
```rust
// src/infrastructure/event_bus.rs
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use async_trait::async_trait;
use crate::domain::events::GraphEvent;

#[async_trait]
pub trait EventBus: Send + Sync {
    async fn publish(&self, event: GraphEvent) -> Result<(), String>;
    async fn subscribe(&self, event_type: &str, handler: Arc<dyn EventHandler>) -> Result<(), String>;
}

#[async_trait]
pub trait EventHandler: Send + Sync {
    async fn handle(&self, event: &GraphEvent) -> Result<(), String>;
}

pub struct InMemoryEventBus {
    subscribers: Arc<RwLock<HashMap<String, Vec<Arc<dyn EventHandler>>>>>,
}

impl InMemoryEventBus {
    pub fn new() -> Self {
        Self {
            subscribers: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

#[async_trait]
impl EventBus for InMemoryEventBus {
    async fn publish(&self, event: GraphEvent) -> Result<(), String> {
        let event_type = event.event_type().to_string();
        let subscribers = self.subscribers.read().unwrap();

        if let Some(handlers) = subscribers.get(&event_type) {
            for handler in handlers {
                tokio::spawn({
                    let handler = handler.clone();
                    let event = event.clone();
                    async move {
                        if let Err(e) = handler.handle(&event).await {
                            log::error!("Event handler failed: {}", e);
                        }
                    }
                });
            }
        }
        Ok(())
    }

    async fn subscribe(&self, event_type: &str, handler: Arc<dyn EventHandler>) -> Result<(), String> {
        let mut subscribers = self.subscribers.write().unwrap();
        subscribers.entry(event_type.to_string())
            .or_insert_with(Vec::new)
            .push(handler);
        Ok(())
    }
}
```

**Domain Events**:
```rust
// src/domain/events.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GraphEvent {
    NodeCreated {
        node_id: u32,
        label: String,
        timestamp: DateTime<Utc>,
        source: UpdateSource,
    },
    NodePositionChanged {
        node_id: u32,
        old_position: (f32, f32, f32),
        new_position: (f32, f32, f32),
        timestamp: DateTime<Utc>,
        source: UpdateSource,
    },
    GitHubSyncCompleted {
        total_nodes: usize,
        total_edges: usize,
        timestamp: DateTime<Utc>,
    },
    // ... more events
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UpdateSource {
    UserInteraction,
    PhysicsSimulation,
    GitHubSync,
    SemanticAnalysis,
}

impl GraphEvent {
    pub fn event_type(&self) -> &str {
        match self {
            GraphEvent::NodeCreated { .. } => "NodeCreated",
            GraphEvent::NodePositionChanged { .. } => "NodePositionChanged",
            GraphEvent::GitHubSyncCompleted { .. } => "GitHubSyncCompleted",
        }
    }
}
```

**Validation**:
```rust
#[tokio::test]
async fn test_event_bus() {
    let event_bus = InMemoryEventBus::new();
    let mock_handler = Arc::new(MockEventHandler::new());

    event_bus.subscribe("NodeCreated", mock_handler.clone()).await.unwrap();

    let event = GraphEvent::NodeCreated {
        node_id: 1,
        label: "Test".to_string(),
        timestamp: Utc::now(),
        source: UpdateSource::UserInteraction,
    };

    event_bus.publish(event).await.unwrap();

    // Wait for async handler
    tokio::time::sleep(Duration::from_millis(100)).await;

    assert_eq!(mock_handler.handled_count(), 1);
}
```

---

### Step 2.2: Implement Command Handlers (Week 2, Day 3-5)
**Estimated Time**: 24 hours

**Files to Create**:
```bash
src/application/graph/
├── commands.rs               # Command DTOs
└── command_handlers.rs       # Command handlers
```

**Command Handlers**:
```rust
// src/application/graph/command_handlers.rs
pub struct CreateNodeCommandHandler {
    graph_repo: Arc<dyn GraphRepository>,
    event_bus: Arc<dyn EventBus>,
}

impl CreateNodeCommandHandler {
    pub async fn handle(&self, cmd: CreateNodeCommand) -> Result<(), String> {
        // 1. Validate
        self.validate(&cmd)?;

        // 2. Create entity
        let node = Node::new(cmd.node_id, cmd.label, cmd.position);

        // 3. Persist
        self.graph_repo.add_node(node.clone()).await?;

        // 4. Emit event
        let event = GraphEvent::NodeCreated {
            node_id: node.id,
            label: node.label.clone(),
            timestamp: Utc::now(),
            source: UpdateSource::UserInteraction,
        };
        self.event_bus.publish(event).await?;

        Ok(())
    }
}
```

**API Handler Update**:
```rust
// src/handlers/api_handler/nodes.rs
pub async fn create_node(
    body: web::Json<CreateNodeRequest>,
    cmd_handler: web::Data<Arc<CreateNodeCommandHandler>>,
) -> Result<HttpResponse, Error> {
    let cmd = CreateNodeCommand {
        node_id: generate_id(),
        label: body.label.clone(),
        position: body.position,
        metadata_id: body.metadata_id.clone(),
    };

    cmd_handler.handle(cmd).await
        .map_err(actix_web::error::ErrorInternalServerError)?;

    Ok(HttpResponse::Created().json(json!({
        "status": "success",
        "message": "Node created"
    })))
}
```

---

### Step 2.3: WebSocket Event Subscriber (Week 3, Day 1-2)
**Estimated Time**: 16 hours

**Files to Create**:
```bash
src/infrastructure/
├── websocket_event_subscriber.rs
└── websocket_gateway.rs
```

**WebSocket Gateway Port**:
```rust
// src/ports/websocket_gateway.rs
#[async_trait]
pub trait WebSocketGateway: Send + Sync {
    async fn broadcast(&self, message: serde_json::Value) -> Result<(), String>;
    async fn send_to_client(&self, client_id: &str, message: serde_json::Value) -> Result<(), String>;
}
```

**Actix WebSocket Adapter**:
```rust
// src/adapters/actix_websocket_adapter.rs
pub struct ActixWebSocketAdapter {
    ws_server: Option<Addr<WebSocketServer>>,
}

#[async_trait]
impl WebSocketGateway for ActixWebSocketAdapter {
    async fn broadcast(&self, message: serde_json::Value) -> Result<(), String> {
        if let Some(server) = &self.ws_server {
            server.do_send(BroadcastMessage {
                data: message.to_string(),
            });
        }
        Ok(())
    }
}
```

**Event Subscriber**:
```rust
// src/infrastructure/websocket_event_subscriber.rs
pub struct WebSocketEventSubscriber {
    ws_gateway: Arc<dyn WebSocketGateway>,
}

#[async_trait]
impl EventHandler for WebSocketEventSubscriber {
    async fn handle(&self, event: &GraphEvent) -> Result<(), String> {
        match event {
            GraphEvent::NodeCreated { node_id, label, .. } => {
                self.ws_gateway.broadcast(json!({
                    "type": "nodeCreated",
                    "nodeId": node_id,
                    "label": label,
                })).await?;
            },
            GraphEvent::NodePositionChanged { node_id, new_position, .. } => {
                self.ws_gateway.broadcast(json!({
                    "type": "nodePositionUpdate",
                    "nodeId": node_id,
                    "position": new_position,
                })).await?;
            },
            _ => {}
        }
        Ok(())
    }
}
```

**Wire Up in main.rs**:
```rust
// src/main.rs
#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Create event bus
    let event_bus = Arc::new(InMemoryEventBus::new());

    // Create WebSocket gateway
    let ws_gateway = Arc::new(ActixWebSocketAdapter::new(ws_server.clone()));

    // Subscribe WebSocket to events
    let ws_subscriber = Arc::new(WebSocketEventSubscriber::new(ws_gateway));
    event_bus.subscribe("NodeCreated", ws_subscriber.clone()).await.unwrap();
    event_bus.subscribe("NodePositionChanged", ws_subscriber.clone()).await.unwrap();

    // ... rest of setup
}
```

---

## Phase 3: Real-Time Features (COMPLEX)
**Duration**: 2 weeks
**Risk**: High
**Goal**: Physics simulation and GitHub sync via events

### Step 3.1: GitHub Sync Event Integration (Week 4, Day 1-3)
**Estimated Time**: 24 hours

**This is THE FIX for the 316 nodes bug!**

**Modify GitHub Sync Service**:
```rust
// src/services/github_sync_service.rs
pub struct GitHubSyncService {
    content_api: Arc<EnhancedContentAPI>,
    kg_repo: Arc<dyn GraphRepository>,
    event_bus: Arc<dyn EventBus>,  // ← ADD THIS!
}

impl GitHubSyncService {
    pub async fn sync_graphs(&self) -> Result<SyncStatistics, String> {
        info!("🔄 Starting GitHub sync...");

        // 1. Fetch files
        let files = self.content_api.fetch_all_files().await?;

        // 2. Parse into nodes/edges
        let (nodes, edges) = self.parse_files(&files)?;

        // 3. Save to database
        self.kg_repo.save_graph(GraphData {
            nodes: nodes.clone(),
            edges: edges.clone(),
        }).await?;

        // 4. ✅ EMIT EVENT - This fixes the cache bug!
        let event = GraphEvent::GitHubSyncCompleted {
            total_nodes: nodes.len(),
            total_edges: edges.len(),
            timestamp: Utc::now(),
        };
        self.event_bus.publish(event).await?;

        info!("✅ GitHub sync completed: {} nodes, {} edges", nodes.len(), edges.len());

        Ok(SyncStatistics {
            total_nodes: nodes.len(),
            total_edges: edges.len(),
            duration: start.elapsed(),
            ..Default::default()
        })
    }
}
```

**Cache Invalidation Subscriber**:
```rust
// src/infrastructure/cache_invalidation_subscriber.rs
pub struct CacheInvalidationSubscriber {
    cache_service: Arc<dyn CacheService>,
}

#[async_trait]
impl EventHandler for CacheInvalidationSubscriber {
    async fn handle(&self, event: &GraphEvent) -> Result<(), String> {
        match event {
            GraphEvent::GitHubSyncCompleted { total_nodes, .. } => {
                log::info!("🔄 Invalidating all caches after GitHub sync ({} nodes)", total_nodes);
                self.cache_service.invalidate_all().await?;
            },
            GraphEvent::NodeCreated { .. } |
            GraphEvent::NodePositionChanged { .. } => {
                self.cache_service.invalidate_graph_data().await?;
            },
            _ => {}
        }
        Ok(())
    }
}
```

**WebSocket Notification**:
```rust
// In WebSocketEventSubscriber, add:
GraphEvent::GitHubSyncCompleted { total_nodes, total_edges, .. } => {
    self.ws_gateway.broadcast(json!({
        "type": "graphReloaded",
        "totalNodes": total_nodes,
        "totalEdges": total_edges,
        "message": "Graph synchronized from GitHub",
    })).await?;
}
```

**Validation**:
```bash
# Run GitHub sync
curl -X POST http://localhost:8080/api/sync/github

# Check database
sqlite3 data/knowledge_graph.db "SELECT COUNT(*) FROM nodes;"
# Expected: 316

# Check API
curl http://localhost:8080/api/graph/data | jq '.nodes | length'
# Expected: 316 ✅

# Check WebSocket clients receive notification
# (monitor browser console for "graphReloaded" message)
```

---

### Step 3.2: Physics Simulation Events (Week 4-5)
**Estimated Time**: 40 hours

**Physics Service as Domain Service**:
```rust
// src/domain/services/physics_service.rs
pub struct PhysicsService {
    graph_repo: Arc<dyn GraphRepository>,
    physics_adapter: Arc<dyn PhysicsSimulator>,
    event_bus: Arc<dyn EventBus>,
}

impl PhysicsService {
    pub async fn simulate_step(&self, params: SimulationParams) -> Result<(), String> {
        // 1. Load graph
        let graph = self.graph_repo.get_graph().await?;

        // 2. Compute physics on GPU
        let updated_positions = self.physics_adapter
            .simulate_step(graph.nodes, graph.edges, params)
            .await?;

        // 3. Save new positions
        self.graph_repo.batch_update_positions(updated_positions.clone()).await?;

        // 4. Emit event
        let event = GraphEvent::PhysicsStepCompleted {
            iteration: params.iteration,
            nodes_updated: updated_positions.len(),
            timestamp: Utc::now(),
        };
        self.event_bus.publish(event).await?;

        Ok(())
    }
}
```

**Physics Command Handler**:
```rust
// src/application/graph/command_handlers.rs
pub struct TriggerPhysicsStepCommandHandler {
    physics_service: Arc<PhysicsService>,
}

impl TriggerPhysicsStepCommandHandler {
    pub async fn handle(&self, cmd: TriggerPhysicsStepCommand) -> Result<(), String> {
        for i in 0..cmd.iterations {
            let params = SimulationParams {
                iteration: i,
                ..cmd.params.clone()
            };
            self.physics_service.simulate_step(params).await?;
        }
        Ok(())
    }
}
```

**WebSocket Broadcasting with Batching** (for smooth 60 FPS):
```rust
// src/infrastructure/websocket_event_subscriber.rs
pub struct WebSocketEventSubscriber {
    ws_gateway: Arc<dyn WebSocketGateway>,
    position_buffer: Arc<RwLock<Vec<(u32, (f32, f32, f32))>>>,
}

impl WebSocketEventSubscriber {
    pub fn new(ws_gateway: Arc<dyn WebSocketGateway>) -> Self {
        let subscriber = Self {
            ws_gateway,
            position_buffer: Arc::new(RwLock::new(Vec::new())),
        };

        // Flush buffer every 16ms (60 FPS)
        let buffer_clone = subscriber.position_buffer.clone();
        let ws_clone = subscriber.ws_gateway.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(16));
            loop {
                interval.tick().await;
                let positions = {
                    let mut buffer = buffer_clone.write().unwrap();
                    std::mem::take(&mut *buffer)
                };

                if !positions.is_empty() {
                    let _ = ws_clone.broadcast(json!({
                        "type": "physicsUpdate",
                        "positions": positions,
                    })).await;
                }
            }
        });

        subscriber
    }
}

#[async_trait]
impl EventHandler for WebSocketEventSubscriber {
    async fn handle(&self, event: &GraphEvent) -> Result<(), String> {
        match event {
            GraphEvent::NodePositionChanged { node_id, new_position, .. } => {
                // Buffer position updates instead of immediate send
                self.position_buffer.write().unwrap().push((*node_id, *new_position));
            },
            _ => {}
        }
        Ok(())
    }
}
```

---

## Phase 4: Legacy Removal (CLEANUP)
**Duration**: 1 week
**Risk**: Low
**Goal**: Delete old actor code

### Step 4.1: Remove Actor Dependencies (Day 1-2)
**Estimated Time**: 16 hours

**Files to Delete**:
```bash
rm src/actors/graph_actor.rs
rm src/actors/graph_messages.rs
rm src/actors/graph_service_supervisor.rs
```

**Update AppState**:
```rust
// src/app_state.rs (BEFORE)
pub struct AppState {
    pub graph_service_actor: Addr<GraphServiceActor>,  // ❌ DELETE
    pub physics_actor: Addr<PhysicsActor>,
    // ...
}

// src/app_state.rs (AFTER)
pub struct AppState {
    pub graph_query_handler: Arc<GetGraphDataQueryHandler>,  // ✅ NEW
    pub create_node_handler: Arc<CreateNodeCommandHandler>,  // ✅ NEW
    pub event_bus: Arc<dyn EventBus>,  // ✅ NEW
    // ...
}
```

---

### Step 4.2: Update Tests (Day 3-4)
**Estimated Time**: 16 hours

**Before (Actor Tests)**:
```rust
#[actix_rt::test]
async fn test_create_node() {
    let actor = GraphServiceActor::new(...).start();
    let result = actor.send(CreateNode { ... }).await;
    assert!(result.is_ok());
}
```

**After (CQRS Tests)**:
```rust
#[tokio::test]
async fn test_create_node() {
    let mock_repo = Arc::new(MockGraphRepository::new());
    let mock_bus = Arc::new(MockEventBus::new());
    let handler = CreateNodeCommandHandler::new(mock_repo, mock_bus);

    let cmd = CreateNodeCommand { ... };
    let result = handler.handle(cmd).await;

    assert!(result.is_ok());
}
```

---

### Step 4.3: Documentation Update (Day 5)
**Estimated Time**: 8 hours

**Update**:
- README.md
- API documentation
- Architecture diagrams
- Developer onboarding guide

---

## Rollback Strategy

### If Phase 1 Fails
**Symptom**: CQRS queries slower than actor
**Action**:
1. Revert API handlers to use actor
2. Keep CQRS code for optimization
3. Add database indexing
4. Retry Phase 1

### If Phase 2 Fails
**Symptom**: Events not delivered, WebSocket disconnects
**Action**:
1. Keep command handlers, disable event publishing
2. Use actor for WebSocket broadcasting
3. Debug event bus
4. Add event retry logic

### If Phase 3 Fails
**Symptom**: Physics simulation breaks, GitHub sync fails
**Action**:
1. Keep old physics actor
2. Use CQRS for non-physics operations
3. Debug physics service
4. Add more logging

---

## Success Metrics

### Phase 1 Success
- [ ] All GET endpoints use CQRS
- [ ] Query latency p95 < 50ms
- [ ] Zero regression in functionality
- [ ] Test coverage > 80%

### Phase 2 Success
- [ ] All POST/PUT/DELETE endpoints use CQRS
- [ ] Events emitted for all state changes
- [ ] WebSocket clients receive updates
- [ ] Zero data loss

### Phase 3 Success
- [ ] Physics simulation works via events
- [ ] GitHub sync emits `GitHubSyncCompletedEvent`
- [ ] API returns 316 nodes after sync ✅
- [ ] Real-time updates smooth (60 FPS)

### Phase 4 Success
- [ ] Zero actor references in codebase
- [ ] All tests passing
- [ ] Documentation updated
- [ ] Code review approved

---

## Risk Matrix

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Performance regression | Medium | High | Benchmark before/after, add caching |
| Data loss | Low | Critical | Run dual-write during transition |
| WebSocket disconnect | Medium | Medium | Implement reconnection logic |
| Event bus failure | Low | High | Add event persistence and retry |
| Team unfamiliarity | Medium | Low | Conduct training sessions |

---

## Team Coordination

### Roles
- **Architecture Lead**: Review design decisions
- **Backend Developer**: Implement command/query handlers
- **Infrastructure Engineer**: Set up event bus and monitoring
- **QA Engineer**: Test migration phases
- **DevOps**: Monitor production rollout

### Communication
- Daily standups during migration
- Weekly architecture reviews
- Incident response plan for rollbacks

---

**Migration strategy designed by**: Hive Mind Architecture Planner
**Date**: 2025-10-26
**Ready for Queen's approval**: 👑
