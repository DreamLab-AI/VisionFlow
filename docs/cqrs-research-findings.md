# CQRS Migration Research Findings

## Research Summary

**Date**: 2025-10-26
**Objective**: Analyze existing CQRS implementations and plan Phase 1 migration for GraphServiceActor
**Status**: ✅ Complete

## Codebase Analysis

### 1. Existing CQRS Implementations

#### Settings Domain (Complete Reference)

**Location**: `src/application/settings/`

**Files Analyzed**:
- `queries.rs` (204 lines) - 5 query handlers
- `directives.rs` (340 lines) - 6 directive handlers
- `mod.rs` - Module exports

**Query Handlers Implemented**:
1. `GetSetting` - Single setting retrieval
2. `GetSettingsBatch` - Batch retrieval (multiple keys)
3. `LoadAllSettings` - Complete settings load
4. `GetPhysicsSettings` - Profile-specific physics config
5. `ListPhysicsProfiles` - Available profile enumeration

**Key Patterns Observed**:
```rust
// Pattern 1: Simple query struct
#[derive(Debug, Clone)]
pub struct GetSetting {
    pub key: String,
}

// Pattern 2: Handler with repository injection
pub struct GetSettingHandler {
    repository: Arc<dyn SettingsRepository>,
}

// Pattern 3: QueryHandler trait implementation
impl QueryHandler<GetSetting, Option<SettingValue>> for GetSettingHandler {
    fn handle(&self, query: GetSetting) -> HexResult<Option<SettingValue>> {
        // Use tokio runtime for async repository access
        tokio::runtime::Handle::current().block_on(async move {
            self.repository.get_setting(&query.key).await
                .map_err(|e| Hexserror::adapter("E_HEX_200", &format!("{}", e)))
        })
    }
}
```

**Directive Patterns** (for Phase 2):
```rust
// Pattern 1: Directive with validation
impl Directive for UpdateSetting {
    fn validate(&self) -> HexResult<()> {
        if self.key.is_empty() {
            return Err(Hexserror::validation("Setting key cannot be empty"));
        }
        Ok(())
    }
}

// Pattern 2: DirectiveHandler implementation
impl DirectiveHandler<UpdateSetting> for UpdateSettingHandler {
    fn handle(&self, directive: UpdateSetting) -> HexResult<()> {
        log::info!("Executing UpdateSetting directive: key={}", directive.key);
        // Execute mutation...
    }
}
```

#### Ontology Domain (Complex Queries)

**Location**: `src/application/ontology/`

**Files Analyzed**:
- `queries.rs` (326 lines) - 10 query handlers
- `directives.rs` (482 lines) - 9 directive handlers

**Complex Query Examples**:
1. `LoadOntologyGraph` - Returns `Arc<GraphData>` (same as our target)
2. `QueryOntology` - SPARQL-like query with dynamic results
3. `ValidateOntology` - Validation with detailed report
4. `GetInferenceResults` - Reasoning results

**Patterns Applicable to Graph Migration**:
```rust
// Arc-based returns (avoid cloning)
impl QueryHandler<LoadOntologyGraph, Arc<GraphData>> for LoadOntologyGraphHandler {
    fn handle(&self, _query: LoadOntologyGraph) -> HexResult<Arc<GraphData>> {
        tokio::runtime::Handle::current().block_on(async {
            self.repository.load_ontology_graph().await
                .map_err(|e| Hexserror::port("E_REPO_001", &format!("{}", e)))
        })
    }
}
```

### 2. Repository Port Patterns

#### SettingsRepository Port

**Location**: `src/ports/settings_repository.rs`

**Interface Design**:
```rust
#[async_trait]
pub trait SettingsRepository: Send + Sync {
    async fn get_setting(&self, key: &str) -> Result<Option<SettingValue>>;
    async fn get_settings_batch(&self, keys: &[String]) -> Result<HashMap<String, SettingValue>>;
    async fn load_all_settings(&self) -> Result<Option<AppFullSettings>>;
    // ... more methods
}
```

**Key Observations**:
- All methods are `async`
- Return custom `Result<T>` type (not `HexResult`)
- Use references (`&str`, `&[String]`) to avoid clones
- Clear separation of concerns

#### GraphRepository Port (Existing)

**Location**: `src/ports/graph_repository.rs`

**Current State**:
```rust
#[async_trait]
pub trait GraphRepository: Send + Sync {
    async fn get_graph(&self) -> Result<Arc<GraphData>>;
    async fn add_nodes(&self, nodes: Vec<Node>) -> Result<Vec<u32>>;
    async fn add_edges(&self, edges: Vec<Edge>) -> Result<Vec<String>>;
    async fn update_positions(&self, updates: Vec<(u32, BinaryNodeData)>) -> Result<()>;
    async fn get_dirty_nodes(&self) -> Result<HashSet<u32>>;
    async fn clear_dirty_nodes(&self) -> Result<()>;
}
```

**Analysis**:
- ✅ Already has `get_graph()` method
- ✅ Returns `Arc<GraphData>` (no clone needed)
- ⚠️ Missing methods for other read operations
- ⚠️ Mixes reads and writes (needs separation)

### 3. Adapter Implementation Patterns

#### SqliteSettingsRepository

**Location**: `src/adapters/sqlite_settings_repository.rs`

**Key Features**:
1. **Caching Layer**:
```rust
struct SettingsCache {
    settings: HashMap<String, CachedSetting>,
    last_updated: std::time::Instant,
    ttl_seconds: u64,
}

async fn get_from_cache(&self, key: &str) -> Option<SettingValue> {
    let cache = self.cache.read().await;
    if let Some(cached) = cache.settings.get(key) {
        if cached.timestamp.elapsed().as_secs() < cache.ttl_seconds {
            return Some(cached.value.clone());
        }
    }
    None
}
```

2. **Thread Pool for Blocking I/O**:
```rust
async fn get_setting(&self, key: &str) -> Result<Option<SettingValue>> {
    // Check cache first
    if let Some(cached_value) = self.get_from_cache(key).await {
        return Ok(Some(cached_value));
    }

    // Query database in blocking thread pool
    let result = tokio::task::spawn_blocking(move || {
        db.get_setting(&key_owned)
    }).await?;

    // Update cache on success
    if let Some(ref value) = result {
        self.update_cache(key.to_string(), value.clone()).await;
    }

    Ok(result)
}
```

**Patterns to Adopt**:
- ✅ Cache-first strategy for performance
- ✅ Use `spawn_blocking` for sync database calls
- ✅ Update cache after successful queries
- ✅ Instrumentation with `tracing` crate

### 4. GraphServiceActor Analysis

**Location**: `src/actors/graph_actor.rs`
**Size**: 2,501 lines (4,343 total with comments)

#### Read Operations Identified

**Tier 1 - Simple Arc Returns** (Highest Priority):
1. `GetGraphData` → `Arc<GraphData>` (Line 3144-3150)
2. `GetNodeMap` → `Arc<HashMap<u32, Node>>` (Line 3340-3345)
3. `GetBotsGraphData` → `Arc<GraphData>` (Line 3815+)
4. `GetConstraints` → `ConstraintSet` (Line 4038+)

**Tier 2 - Computed State** (Medium Priority):
5. `GetPhysicsState` → `PhysicsState` (Line 3349-3377)
   - Calculates average kinetic energy
   - Analyzes current auto-balance state
   - Returns settlement information
6. `GetAutoBalanceNotifications` → `Vec<AutoBalanceNotification>` (Line 3517+)
7. `GetEquilibriumStatus` → `EquilibriumStatus` (Line 4294+)

**Tier 3 - Complex Computations** (Lower Priority):
8. `ComputeShortestPaths` → `PathfindingResult` (Line 4115+)
   - GPU-based SSSP algorithm
   - Should remain in actor for now

#### Current API Routes

**Location**: `src/handlers/api_handler/graph/mod.rs`

**Primary Endpoint**:
```rust
pub async fn get_graph_data(state: web::Data<AppState>) -> impl Responder {
    // Parallel fetching of three queries
    let graph_data_future = state.graph_service_addr.send(GetGraphData);
    let node_map_future = state.graph_service_addr.send(GetNodeMap);
    let physics_state_future = state.graph_service_addr.send(GetPhysicsState);

    let (graph_result, node_map_result, physics_result) =
        tokio::join!(graph_data_future, node_map_future, physics_state_future);

    // Build response with physics positions
    let nodes_with_positions: Vec<NodeWithPosition> = // ...
}
```

**Observations**:
- Already doing parallel queries
- Perfect candidate for CQRS conversion
- No mutations in this route
- High traffic endpoint

## Pattern Recommendations

### For Graph Read Operations

#### 1. Query Handler Template
```rust
#[derive(Debug, Clone)]
pub struct GetGraphData;

pub struct GetGraphDataHandler {
    repository: Arc<dyn GraphRepository>,
}

impl GetGraphDataHandler {
    pub fn new(repository: Arc<dyn GraphRepository>) -> Self {
        Self { repository }
    }
}

impl QueryHandler<GetGraphData, Arc<GraphData>> for GetGraphDataHandler {
    fn handle(&self, _query: GetGraphData) -> HexResult<Arc<GraphData>> {
        log::debug!("Executing GetGraphData query");

        let repository = self.repository.clone();

        tokio::runtime::Handle::current()
            .block_on(async move {
                repository.get_graph()
                    .await
                    .map_err(|e| Hexserror::port("E_GRAPH_001", &format!("Failed to get graph: {}", e)))
            })
    }
}
```

#### 2. Actor Adapter Pattern
```rust
pub struct ActorGraphRepository {
    actor_addr: Addr<GraphServiceActor>,
}

#[async_trait]
impl GraphRepository for ActorGraphRepository {
    async fn get_graph(&self) -> Result<Arc<GraphData>> {
        self.actor_addr
            .send(actor_msgs::GetGraphData)
            .await
            .map_err(|e| GraphRepositoryError::AccessError(format!("Mailbox error: {}", e)))?
            .map_err(|e| GraphRepositoryError::AccessError(e))
    }
}
```

#### 3. API Handler Update
```rust
pub async fn get_graph_data(state: web::Data<AppState>) -> impl Responder {
    // Use query handlers instead of direct actor calls
    let graph_handler = state.graph_query_handlers.get_graph_data.clone();
    let node_map_handler = state.graph_query_handlers.get_node_map.clone();
    let physics_handler = state.graph_query_handlers.get_physics_state.clone();

    // Execute in parallel with spawn_blocking
    let (graph_result, node_map_result, physics_result) = tokio::join!(
        tokio::task::spawn_blocking(move || graph_handler.handle(GetGraphData)),
        tokio::task::spawn_blocking(move || node_map_handler.handle(GetNodeMap)),
        tokio::task::spawn_blocking(move || physics_handler.handle(GetPhysicsState))
    );

    // Process results...
}
```

## Dependencies and Integration Points

### 1. Hexser Framework
**Package**: `hexser = "0.1.0"` (or similar)

**Traits Used**:
- `QueryHandler<Q, R>` - Read operations
- `DirectiveHandler<D>` - Write operations (Phase 2)
- `Directive` - Validation trait for directives
- `HexResult<T>` - Unified result type
- `Hexserror` - Error construction

### 2. Actor System Integration
**Package**: `actix = "0.13"`

**Current Usage**:
- `Addr<GraphServiceActor>` - Actor address for messaging
- `Handler<M>` - Message handler trait
- Message passing for all operations

**Migration Path**:
- Phase 1: Keep actor, add repository adapter
- Phase 2: Gradually replace actor calls
- Phase 3: Remove actor dependency

### 3. Database Layer
**Current**: Actor manages in-memory state
**Future**: Direct repository access to database

**Options**:
1. `SqliteGraphRepository` - SQLite backend
2. `PostgresGraphRepository` - PostgreSQL backend
3. `InMemoryGraphRepository` - Testing/development

### 4. Application State
**Location**: `src/app_state.rs`

**Current Structure**:
```rust
pub struct AppState {
    pub graph_service_addr: Addr<GraphServiceActor>,
    // ... other actors
}
```

**After Migration**:
```rust
pub struct GraphQueryHandlers {
    pub get_graph_data: Arc<GetGraphDataHandler>,
    pub get_node_map: Arc<GetNodeMapHandler>,
    pub get_physics_state: Arc<GetPhysicsStateHandler>,
    // ... more handlers
}

pub struct AppState {
    pub graph_service_addr: Addr<GraphServiceActor>, // Keep for Phase 1
    pub graph_query_handlers: GraphQueryHandlers,    // NEW
    // ... other fields
}
```

## Performance Analysis

### Current Performance Characteristics

**GetGraphData** (most critical):
- Returns `Arc<GraphData>` - no clone needed ✅
- Direct memory access - minimal latency ✅
- No database I/O - fast ✅

**GetNodeMap**:
- Returns `Arc<HashMap<u32, Node>>` - no clone ✅
- In-memory lookup - fast ✅

**GetPhysicsState**:
- Computes average from history array
- Small computation overhead
- 30-50 frame history typical

### Expected Impact of CQRS

**Positive**:
- Caching layer can reduce repeated lookups
- Better monitoring/metrics per query
- Easier to optimize individual queries
- Testability improves dramatically

**Negative**:
- Additional function call overhead (negligible)
- `tokio::runtime::Handle::current().block_on()` overhead
- Handler initialization in AppState (one-time)

**Mitigation**:
- Keep Arc semantics - no new clones
- Repository adapter is thin wrapper
- Profile before/after with real traffic

### Benchmarking Recommendations

```rust
#[cfg(test)]
mod benchmarks {
    use criterion::{black_box, criterion_group, criterion_main, Criterion};

    fn bench_get_graph_actor(c: &mut Criterion) {
        // Benchmark current actor approach
    }

    fn bench_get_graph_cqrs(c: &mut Criterion) {
        // Benchmark CQRS query handler
    }

    criterion_group!(benches, bench_get_graph_actor, bench_get_graph_cqrs);
    criterion_main!(benches);
}
```

## Testing Strategy

### Unit Tests (Query Handlers)
```rust
#[cfg(test)]
mod tests {
    use super::*;

    struct MockGraphRepository {
        graph: Arc<GraphData>,
    }

    #[async_trait]
    impl GraphRepository for MockGraphRepository {
        async fn get_graph(&self) -> Result<Arc<GraphData>> {
            Ok(Arc::clone(&self.graph))
        }
    }

    #[test]
    fn test_get_graph_data_handler() {
        let mock_repo = Arc::new(MockGraphRepository {
            graph: Arc::new(GraphData::default()),
        });

        let handler = GetGraphDataHandler::new(mock_repo);
        let result = handler.handle(GetGraphData);

        assert!(result.is_ok());
    }
}
```

### Integration Tests (API Routes)
```rust
#[actix_web::test]
async fn test_get_graph_data_endpoint() {
    let app = test::init_service(App::new().configure(routes)).await;
    let req = test::TestRequest::get()
        .uri("/api/graph/data")
        .to_request();

    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), StatusCode::OK);
}
```

## Risk Assessment

### Low Risk (Safe to Migrate)
- ✅ `GetGraphData` - Pure read, Arc return
- ✅ `GetNodeMap` - Pure read, Arc return
- ✅ `GetConstraints` - Pure read, owned return
- ✅ `GetBotsGraphData` - Pure read, Arc return

### Medium Risk (Needs Care)
- ⚠️ `GetPhysicsState` - Computed state, verify calculation logic
- ⚠️ `GetAutoBalanceNotifications` - Mutex access, verify thread safety
- ⚠️ `GetEquilibriumStatus` - Complex analysis, verify algorithm

### High Risk (Phase 2+)
- ❌ `ComputeShortestPaths` - GPU computation, keep in actor
- ❌ Write operations - Need directive handlers (Phase 2)
- ❌ Position updates - Real-time, keep in actor

## Dependencies to Add

### Cargo.toml Updates
```toml
[dependencies]
# Existing
actix = "0.13"
actix-web = "4.0"
tokio = { version = "1.0", features = ["full"] }

# May need to add/verify
hexser = "0.1"  # CQRS framework
async-trait = "0.1"
thiserror = "1.0"
```

## Conclusion

### Key Findings

1. **Existing Patterns are Solid**: Settings and Ontology domains provide excellent reference implementations
2. **GraphServiceActor is Ready**: Already returns Arc for most reads, minimal changes needed
3. **Risk is Low**: Read operations are side-effect free, perfect for Phase 1
4. **Performance Should Improve**: Caching opportunities, better monitoring
5. **Migration Path is Clear**: Actor adapter allows gradual transition

### Recommended Next Steps

1. ✅ **Create** `src/application/graph/queries.rs` with 7 query handlers
2. ✅ **Extend** `src/ports/graph_repository.rs` with read methods
3. ✅ **Implement** `src/adapters/actor_graph_repository.rs` adapter
4. ✅ **Update** API routes in `src/handlers/api_handler/graph/mod.rs`
5. ✅ **Add** integration tests
6. ✅ **Deploy** with feature flag for gradual rollout

### Timeline Estimate

- **Research**: ✅ Complete (this document)
- **Implementation**: 2-3 days (7 query handlers + adapter)
- **Testing**: 1 day (unit + integration tests)
- **Deployment**: 1 day (feature flag + monitoring)
- **Total**: ~5 days for Phase 1

### Success Metrics

- [ ] All API routes use query handlers
- [ ] Response times within 5% of baseline
- [ ] 100% test coverage for query handlers
- [ ] Zero behavioral changes for clients
- [ ] Clean separation of read/write operations

---

**Research Completed**: 2025-10-26
**Blueprint Created**: `/home/devuser/workspace/project/docs/cqrs-phase1-read-operations.md`
