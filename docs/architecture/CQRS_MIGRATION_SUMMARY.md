# CQRS Migration Summary - Quick Reference

**Date**: November 3, 2025
**Status**: 🔄 15% Complete (Phase 1 Done, Phase 2-4 Pending)

---

## TL;DR - What Needs To Happen

### The Problem
- **GraphServiceActor**: 156KB, 4,614 lines, 46 message handlers
- **ActorGraphRepository**: Technical debt adapter wrapping actor
- **Cache Bug**: GitHub sync writes 316 nodes to DB, but actor cache shows 63 (stale)
- **Mixed Architecture**: Queries use CQRS, writes still use actor messages

### The Solution
1. ✅ **Phase 1 DONE**: Query operations use CQRS handlers
2. 🔄 **Phase 2 NEEDED**: Create directive handlers for write operations
3. 🎯 **Phase 3 CRITICAL**: Add event bus for cache invalidation (fixes 316 nodes bug)
4. 🎯 **Phase 4 FINAL**: Remove GraphServiceActor (4,614 lines deleted!)

---

## Architecture Comparison

### BEFORE (Current Hybrid)
```
HTTP Handler
    ↓
GraphServiceActor (48K tokens!)
    ↓
In-Memory Cache (STALE!)
    ↓
Database (unified.db)
```

**Problems**:
- ❌ Actor holds stale cache
- ❌ GitHub sync → DB write → cache not updated
- ❌ API returns 63 nodes instead of 316
- ❌ Monolithic 4,614 line actor

### AFTER (Pure CQRS)
```
HTTP Handler
    ↓
CQRS Handlers (Queries + Directives)
    ↓
UnifiedGraphRepository
    ↓
Database (unified.db)
    ⇅
Event Bus → Cache Invalidator + WebSocket Broadcaster
```

**Benefits**:
- ✅ Database is source of truth (no stale cache)
- ✅ GitHub sync emits event → cache invalidated
- ✅ API always returns fresh data (316 nodes!)
- ✅ Small, focused handlers instead of monolith

---

## What's Implemented vs Missing

### ✅ Implemented (Phase 1 Complete)

**Query Handlers** (`src/application/graph/queries.rs`):
- `GetGraphDataHandler`
- `GetNodeMapHandler`
- `GetPhysicsStateHandler`
- `GetNodePositionsHandler`
- `GetConstraintsHandler`
- `GetBotsGraphDataHandler`
- `GetAutoBalanceNotificationsHandler`
- `GetEquilibriumStatusHandler`
- `ComputeShortestPathsHandler`

**Repositories**:
- `UnifiedGraphRepository` - Direct database access ✅

### ❌ Missing (Phase 2-4 Needed)

**Graph Directive Handlers** (`src/application/graph/directives.rs` **DOES NOT EXIST**):
- `CreateNodeHandler` - ❌ Missing
- `CreateEdgeHandler` - ❌ Missing
- `UpdateNodePositionHandler` - ❌ Missing
- `BatchUpdatePositionsHandler` - ❌ Missing
- `DeleteNodeHandler` - ❌ Missing
- `DeleteEdgeHandler` - ❌ Missing

**Event Infrastructure**:
- Event bus implementation - ❌ Missing
- Cache invalidation subscriber - ❌ Missing
- WebSocket broadcaster subscriber - ❌ Missing
- GitHub sync event emission - ❌ Missing

**Technical Debt to Remove**:
- `ActorGraphRepository` adapter - ⚠️ Deprecated
- Direct actor message usage in HTTP handlers - ⚠️ Mixed
- GraphServiceActor CRUD logic - 🎯 To be deleted

---

## Message Type Mapping

### Queries (✅ DONE)
| Actor Message | CQRS Handler | Status |
|--------------|--------------|--------|
| `GetGraphData` | `GetGraphDataHandler` | ✅ |
| `GetNodeMap` | `GetNodeMapHandler` | ✅ |
| `GetPhysicsState` | `GetPhysicsStateHandler` | ✅ |
| `GetConstraints` | `GetConstraintsHandler` | ✅ |

### Commands (❌ TODO)
| Actor Message | CQRS Directive | Status |
|--------------|----------------|--------|
| `AddNode` | `CreateNode` | ❌ Missing |
| `AddEdge` | `CreateEdge` | ❌ Missing |
| `UpdateNodePositions` | `BatchUpdatePositions` | ❌ Missing |
| `RemoveNode` | `DeleteNode` | ❌ Missing |
| `RemoveEdge` | `DeleteEdge` | ❌ Missing |

---

## Critical Path to Fix Cache Bug

### The Bug
1. GitHub sync writes 316 nodes to `unified.db`
2. `GraphServiceActor` in-memory cache still has 63 nodes (old data)
3. API call → actor returns cache → user sees 63 nodes ❌

### The Fix (3-Step Process)

**Step 1: Create Directive Handlers**
```rust
// src/application/graph/directives.rs (CREATE THIS FILE)

pub struct CreateNodeHandler {
    repository: Arc<UnifiedGraphRepository>,
    event_publisher: Arc<dyn DomainEventPublisher>,
}

impl DirectiveHandler<CreateNode> for CreateNodeHandler {
    fn handle(&self, directive: CreateNode) -> HexResult<()> {
        // 1. Persist to database
        self.repository.add_nodes(vec![directive.node]).await?;

        // 2. Emit event
        self.event_publisher.publish(GraphEvent::NodeCreated { ... })?;

        Ok(())
    }
}
```

**Step 2: Add Event Bus + Subscribers**
```rust
// src/application/graph/cache_invalidator.rs (CREATE THIS FILE)

pub struct CacheInvalidationSubscriber;

impl DomainEventSubscriber for CacheInvalidationSubscriber {
    fn on_event(&self, event: &GraphEvent) -> Result<(), String> {
        match event {
            GraphEvent::GraphSyncCompleted { .. } => {
                log::info!("🔄 Cache invalidated after GitHub sync");
                // Clear any cached data
            }
            _ => {}
        }
        Ok(())
    }
}
```

**Step 3: GitHub Sync Emits Event**
```rust
// src/services/github_sync_service.rs (UPDATE THIS FILE)

pub async fn sync_graphs(&self) -> Result<SyncStatistics> {
    // Write to database
    self.unified_repo.save_graph(graph_data).await?;

    // ✅ EMIT EVENT (this is the fix!)
    let event = GraphEvent::GraphSyncCompleted {
        total_nodes: nodes.len(),  // 316
        total_edges: edges.len(),
        timestamp: Utc::now(),
    };
    self.event_publisher.publish(event)?;

    Ok(stats)
}
```

**Result**:
- GitHub sync → writes to DB → emits event
- Event → Cache invalidator clears cache
- Event → WebSocket broadcaster notifies clients
- Next API call → reads from DB → returns 316 nodes ✅

---

## Immediate Action Items

### Week 1-2: Phase 2 (Directive Handlers)

**Files to Create**:
1. `src/application/graph/directives.rs`
   - `CreateNode` + `CreateNodeHandler`
   - `CreateEdge` + `CreateEdgeHandler`
   - `UpdateNodePosition` + `UpdateNodePositionHandler`
   - `BatchUpdatePositions` + `BatchUpdatePositionsHandler`

**Files to Update**:
2. `src/application/graph/mod.rs`
   - Export directive handlers

3. `src/handlers/api_handler/graph/mod.rs`
   - Replace `AddNode` actor message with `CreateNode` directive
   - Replace `UpdateNodePositions` with `BatchUpdatePositions` directive

4. `src/app_state.rs`
   - Add directive handlers to AppState
   - Wire UnifiedGraphRepository to handlers

**Test**:
5. Create unit tests for each directive handler
6. Integration test: POST /api/graph/nodes → verify DB write → verify response

---

### Week 3-4: Phase 3 (Event Bus - CRITICAL FOR BUG FIX)

**Files to Create**:
1. `src/application/events.rs` (enhance existing)
   - Add `GraphEvent::GraphSyncCompleted`
   - Add `DomainEventPublisher` trait
   - Add `InMemoryEventBus` implementation

2. `src/application/graph/cache_invalidator.rs`
   - Implement `CacheInvalidationSubscriber`

3. `src/application/graph/websocket_broadcaster.rs`
   - Implement `WebSocketBroadcasterSubscriber`

**Files to Update**:
4. `src/services/github_sync_service.rs`
   - Add `event_publisher` dependency
   - Emit `GraphSyncCompleted` event after DB write

5. `src/application/graph/directives.rs`
   - Update all handlers to emit events after persistence

**Test**:
6. Integration test: GitHub sync → verify event emitted → verify cache cleared → verify API returns 316 nodes ✅

---

### Week 5-6: Phase 4 (Actor Removal)

**Extract Physics Simulation**:
1. Create `src/application/graph/physics_service.rs`
2. Move physics logic from actor to service
3. Update physics directives to use service

**Extract WebSocket**:
4. Create `src/adapters/websocket_gateway.rs`
5. Wire event subscribers to gateway

**Delete Actor**:
6. Remove `src/actors/graph_actor.rs` (4,614 lines!)
7. Remove `src/adapters/actor_graph_repository.rs`
8. Update all references

---

## File Checklist

### Create These Files
- [ ] `src/application/graph/directives.rs` (directive handlers)
- [ ] `src/application/graph/cache_invalidator.rs` (event subscriber)
- [ ] `src/application/graph/websocket_broadcaster.rs` (event subscriber)
- [ ] `src/application/graph/physics_service.rs` (domain service)
- [ ] `src/adapters/websocket_gateway.rs` (WebSocket adapter)
- [ ] `src/ports/event_publisher.rs` (event bus trait)

### Update These Files
- [ ] `src/application/events.rs` (add graph events)
- [ ] `src/application/graph/mod.rs` (export directives)
- [ ] `src/handlers/api_handler/graph/mod.rs` (use directives)
- [ ] `src/services/github_sync_service.rs` (emit events)
- [ ] `src/app_state.rs` (wire directive handlers)

### Delete These Files (Phase 4)
- [ ] `src/actors/graph_actor.rs` (4,614 lines)
- [ ] `src/adapters/actor_graph_repository.rs` (technical debt)

---

## Success Metrics

### Phase 1 ✅
- [x] All GET endpoints use query handlers
- [x] Query latency <50ms (p95)
- [x] Test coverage >80%

### Phase 2 🎯
- [ ] All POST/PUT/DELETE endpoints use directive handlers
- [ ] ActorGraphRepository usage removed for writes
- [ ] Zero data loss during migration

### Phase 3 🎯 CRITICAL
- [ ] GitHub sync emits `GraphSyncCompleted` event
- [ ] Cache invalidation subscriber receives event
- [ ] API returns 316 nodes after sync ✅ BUG FIXED!

### Phase 4 🎯
- [ ] GraphServiceActor deleted
- [ ] All tests passing
- [ ] No actor references in HTTP handlers

---

## Frequently Asked Questions

### Q: Why not just fix the actor's cache invalidation?
**A**: The actor is 4,614 lines and handles too many responsibilities. Fixing cache invalidation is a band-aid. The real solution is proper architecture (CQRS + events) that prevents cache coherency issues by design.

### Q: Can we keep the actor for some things?
**A**: Yes! Physics simulation and WebSocket coordination can stay in actors initially. We're only removing the CRUD/cache logic that belongs in repositories and handlers.

### Q: What's the risk of this migration?
**A**: Medium. Phase 1 (queries) is done with zero issues. Phase 2 (directives) is low risk since we can run both systems in parallel. Phase 3 (events) is the critical path for the cache bug fix.

### Q: How long will this take?
**A**: 4-7 weeks total:
- Phase 1: ✅ Done (1 week)
- Phase 2: 1-2 weeks (directive handlers)
- Phase 3: 1-2 weeks (event bus + cache fix)
- Phase 4: 1-2 weeks (actor removal)

### Q: What if we don't do this migration?
**A**: The cache coherency bug will remain (63 nodes vs 316). The 4,614 line actor will keep growing. Testing and maintenance will get harder. Technical debt will accumulate.

---

## References

**Full Documentation**:
- `docs/architecture/CQRS_MIGRATION_COMPLETE.md` - Complete migration plan
- `docs/architecture/hexagonal-cqrs-architecture.md` - Architecture patterns
- `docs/architecture/00-ARCHITECTURE-OVERVIEW.md` - Current implementation

**Source Code**:
- `src/application/graph/queries.rs` - Query handlers (✅ implemented)
- `src/application/graph/directives.rs` - Directive handlers (❌ missing)
- `src/adapters/actor_graph_repository.rs` - Technical debt
- `src/actors/graph_actor.rs` - 4,614 line monolith to decompose

**Related Issues**:
- Cache coherency bug (63 vs 316 nodes)
- Actor message explosion (129 message types)
- Mixed architecture patterns (CQRS queries + actor commands)

---

**Prepared by**: System Architecture Designer
**Next review**: After Phase 2 completion
