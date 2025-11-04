# CQRS Migration - Quick Reference Card

> ⚠️ **PARTIAL DEPRECATION** ⚠️
> Some examples in this quick reference reference the **deprecated GraphServiceActor**. For current patterns, see `/docs/guides/graphserviceactor-migration.md`.

## The Problem

```
┌─────────────────────────────────────────┐
│  GraphServiceActor (156KB, 4614 lines)  │
│  ❌ DEPRECATED (Nov 2025)               │
│  ┌───────────────────────────────────┐  │
│  │  In-Memory Cache (STALE!)         │  │
│  │  • Shows 63 nodes                 │  │
│  │  • Should show 316 nodes          │  │
│  │  • No invalidation after sync     │  │
│  └───────────────────────────────────┘  │
│                                          │
│  46 Message Handlers                    │
│  129 Message Types                      │
│  Mixed Concerns (physics+WS+cache+DB)   │
└─────────────────────────────────────────┘
                    ↓
         ❌ CACHE COHERENCY BUG

CURRENT PATTERN: Use CQRS query/directive handlers
See: /docs/guides/graphserviceactor-migration.md
```

## The Solution

```
┌──────────────┐     ┌──────────────┐     ┌─────────────┐
│ HTTP Handler │────▶│ CQRS Handler │────▶│ Repository  │
└──────────────┘     └──────┬───────┘     └──────┬──────┘
                            │                     │
                            │ emit                │ read/write
                            ▼                     ▼
                     ┌──────────────┐     ┌─────────────┐
                     │  Event Bus   │     │ unified.db  │
                     └──────┬───────┘     └─────────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
       ┌──────────┐  ┌──────────┐  ┌──────────┐
       │  Cache   │  │WebSocket │  │ Metrics  │
       │Invalidate│  │Broadcast │  │ Tracker  │
       └──────────┘  └──────────┘  └──────────┘

         ✅ ALWAYS FRESH DATA FROM DB
```

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
```
src/application/events.rs                    ⚠️ Enhance
  └─ Add GraphEvent variants

src/application/graph/cache_invalidator.rs   ❌ Create
  └─ CacheInvalidationSubscriber

src/application/graph/websocket_broadcaster.rs ❌ Create
  └─ WebSocketBroadcasterSubscriber
```

### Priority 3: Integration (Week 3-4)
```
src/services/github_sync_service.rs          ⚠️ Update
  └─ Emit GraphSyncCompleted event

src/handlers/api_handler/graph/mod.rs        ⚠️ Update
  └─ Use directive handlers, not actor messages

src/app_state.rs                             ⚠️ Update
  └─ Wire directive handlers
```

## Code Patterns

### Actor Message (OLD ❌)
```rust
// HTTP Handler sends actor message
state.graph_service_actor
    .send(AddNode { node })
    .await??;
```

### CQRS Directive (NEW ✅)
```rust
// HTTP Handler uses directive handler
let handler = state.graph_directive_handlers.create_node;
let directive = CreateNode { node };

handler.handle(directive)?;
// Handler:
//  1. Validates
//  2. Persists to DB
//  3. Emits event
```

## Event Flow (Critical for Cache Fix)

```
GitHub Sync
    │
    ├─▶ Write to unified.db
    │
    └─▶ Emit GraphSyncCompleted event
            │
            ├─▶ CacheInvalidationSubscriber
            │   └─▶ Clear caches
            │
            └─▶ WebSocketBroadcasterSubscriber
                └─▶ Notify clients

Next API Call
    │
    └─▶ Query Handler
        └─▶ Read from DB (fresh 316 nodes!) ✅
```

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
- `CQRS_MIGRATION_COMPLETE.md` - Complete analysis
- `CQRS_DIRECTIVE_TEMPLATE.md` - Copy-paste template
- `hexagonal-cqrs-architecture.md` - Architecture patterns

**Source Files**:
- `src/application/graph/queries.rs` - Query examples
- `src/application/knowledge_graph/directives.rs` - Directive examples
- `src/adapters/actor_graph_repository.rs` - What NOT to do

**Ask Questions**:
- Which handler do I update first? → Start with CreateNode
- How do I test directives? → See template testing section
- What about physics? → Leave in actor for now (Phase 4)
