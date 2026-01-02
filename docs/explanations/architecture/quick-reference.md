---
layout: default
title: "CQRS Migration - Quick Reference Card"
parent: Architecture
grand_parent: Explanations
nav_order: 31
---

# CQRS Migration - Quick Reference Card

> ⚠️ **PARTIAL DEPRECATION** ⚠️
> Some examples in this quick reference reference the **deprecated GraphServiceActor**. For current patterns, see `/docs/guides/graphserviceactor-migration.md`.

## The Problem

**GraphServiceActor Monolith Issue** (156KB, 4,614 lines - DEPRECATED Nov 2025)

The old monolithic architecture suffered from:
- **Cache Coherency Bug**: In-memory cache showed 63 nodes, actual database had 316
- **Mixed Concerns**: Single actor handling physics, WebSocket, caching, and database operations
- **46 Message Handlers**: Difficult to extend and maintain
- **No Invalidation**: Cache wasn't updated after GitHub sync operations

**See detailed analysis**: [Actor System Architecture](../../diagrams/server/actors/actor-system-complete.md#actor-lifecycle-and-supervision-strategies)

## The Solution - CQRS Architecture

The new architecture separates concerns:

1. **HTTP Handler** → Receives request
2. **CQRS Handler** → Processes command/query
3. **Repository** → Reads/writes to unified.db (always fresh)
4. **Event Bus** → Broadcasts changes
5. **Subscribers** → Cache invalidation, WebSocket broadcast, metrics tracking

**Result**: ✅ Always fresh data from database, no stale cache

**See detailed architecture**:
- [REST API Architecture](../../diagrams/server/api/rest-api-architecture.md)
- [Actor System with CQRS](../../diagrams/server/actors/actor-system-complete.md#message-flow-patterns)

## Migration Status

| Phase | Description | Status | Time |
|-------|-------------|--------|------|
| 1     | Query handlers (reads) | ✅ DONE | 1 week |
| 2     | Directive handlers (writes) | ❌ TODO | 1-2 weeks |
| 3     | Event bus (cache fix) | ❌ TODO | 1-2 weeks |
| 4     | Actor removal | ⚠️ BLOCKED | 1-2 weeks |

**Progress**: 15% complete (Phase 1 done)

## Files to Create

### Priority 1: Directive Handlers (Week 1-2)
```
src/application/graph/directives.rs          ❌ Create this!
  ├─ CreateNodeHandler
  ├─ CreateEdgeHandler
  ├─ UpdateNodePositionHandler
  ├─ BatchUpdatePositionsHandler
  ├─ DeleteNodeHandler
  └─ DeleteEdgeHandler
```

### Priority 2: Event Infrastructure (Week 3-4)

**Files to create/enhance:**
- `src/application/events.rs` - Add GraphEvent variants for all domain events
- `src/application/graph/cache-invalidator.rs` - Implement CacheInvalidationSubscriber
- `src/application/graph/websocket-broadcaster.rs` - Implement WebSocketBroadcasterSubscriber

These subscribers listen to domain events and handle cache invalidation and client notifications.

### Priority 3: Integration (Week 3-4)

**Files to update:**
- `src/services/github-sync-service.rs` - Emit `GraphSyncCompleted` event after sync
- `src/handlers/api-handler/graph/mod.rs` - Use directive handlers instead of actor messages
- `src/app-state.rs` - Wire directive handlers into application state

## Code Patterns

### Actor Message (OLD ❌)
```rust
// HTTP Handler sends actor message
state.graph-service-actor
    .send(AddNode { node })
    .await??;
```

### CQRS Directive (NEW ✅)
```rust
// HTTP Handler uses directive handler
let handler = state.graph-directive-handlers.create-node;
let directive = CreateNode { node };

handler.handle(directive)?;
// Handler:
//  1. Validates
//  2. Persists to DB
//  3. Emits event
```

## Event Flow (Critical for Cache Fix)

**GitHub Sync Operation**:
1. GitHub Sync reads markdown files
2. Write to unified.db with new graph data
3. Emit `GraphSyncCompleted` event
4. Subscribers act automatically:
   - **CacheInvalidationSubscriber** - Clears in-memory caches
   - **WebSocketBroadcasterSubscriber** - Notifies all connected clients
5. Next API call uses Query Handler
6. Query Handler reads fresh data from database (all 316 nodes!) ✅

**See**: [Complete Data Flows - GitHub Sync](../../diagrams/data-flow/complete-data-flows.md)

## Message Mapping

### Queries (✅ DONE)
| Actor | CQRS |
|-------|------|
| GetGraphData | GetGraphDataHandler ✅ |
| GetNodeMap | GetNodeMapHandler ✅ |
| GetPhysicsState | GetPhysicsStateHandler ✅ |

### Commands (❌ TODO)
| Actor | CQRS | Priority |
|-------|------|----------|
| AddNode | CreateNode ❌ | HIGH |
| AddEdge | CreateEdge ❌ | HIGH |
| UpdateNodePositions | BatchUpdatePositions ❌ | HIGH |
| RemoveNode | DeleteNode ❌ | MEDIUM |
| RemoveEdge | DeleteEdge ❌ | MEDIUM |

## Validation Checklist

### Phase 2 Complete When:
- [ ] `src/application/graph/directives.rs` exists
- [ ] All 6 directive handlers implemented
- [ ] HTTP handlers use directives, not actor messages
- [ ] Unit tests for each handler
- [ ] Integration test: POST → DB write → verify

### Phase 3 Complete When:
- [ ] Events enhanced with GraphEvent variants
- [ ] InMemoryEventBus implemented
- [ ] CacheInvalidationSubscriber implemented
- [ ] WebSocketBroadcasterSubscriber implemented
- [ ] GitHub sync emits GraphSyncCompleted
- [ ] Integration test: sync → event → cache clear → API returns 316 nodes ✅

## Common Pitfalls

### ❌ Don't
- Send actor messages from HTTP handlers
- Update ActorGraphRepository (it's deprecated)
- Add more message types to actor
- Keep cache in actor

### ✅ Do
- Use directive handlers for writes
- Use query handlers for reads
- Use UnifiedGraphRepository directly
- Emit events after persistence
- Let event subscribers handle side effects

## Key Metrics

| Metric | Current | Target |
|--------|---------|--------|
| GraphServiceActor size ❌ DEPRECATED (Nov 2025) | 156KB | 0 (deleted) |
| Message handlers | 46 | 0 |
| HTTP handlers using CQRS | ~30% | 100% |
| Cache coherency bug | Exists | Fixed |
| Test coverage | ~60% | >80% |

**Current Pattern**: Use `UnifiedGraphRepository` + CQRS handlers instead of GraphServiceActor. See `/docs/guides/graphserviceactor-migration.md` for migration guide.

## Next Steps

### This Week
1. Create `src/application/graph/directives.rs` from template
2. Implement CreateNode + CreateEdge handlers
3. Update 1-2 HTTP handlers to use directives
4. Write unit tests

### Next Week
1. Implement remaining directive handlers
2. Update all HTTP handlers
3. Remove ActorGraphRepository usage
4. Integration tests

### Week 3-4
1. Add event bus integration
2. Implement event subscribers
3. Update GitHub sync to emit events
4. ✅ Fix cache bug (316 nodes!)

## Help & References

**Full Docs**:
- `cqrs-migration-complete.md` - Complete analysis
- `cqrs-directive-template.md` - Copy-paste template
- `hexagonal-cqrs.md` - Architecture patterns

**Source Files**:
- `src/application/graph/queries.rs` - Query examples
- `src/application/knowledge-graph/directives.rs` - Directive examples
- `src/adapters/actor-graph-repository.rs` - What NOT to do

**Ask Questions**:
- Which handler do I update first? → Start with CreateNode
- How do I test directives? → See template testing section
- What about physics? → Leave in actor for now (Phase 4)
