# CQRS Migration Documentation - READ ME FIRST

**Date**: November 3, 2025
**System Architecture Analysis by**: Agent 3 (CQRS Migration Specialist)

---

## 📚 Documentation Structure

This directory contains comprehensive CQRS migration documentation for the VisionFlow Graph Service actor decomposition project.

### Start Here

1. **QUICK_REFERENCE.md** (7.6KB, 233 lines) ⭐ START HERE
   - Visual diagrams showing the problem and solution
   - Current status (15% complete)
   - Files to create with priorities
   - Code pattern examples (old vs new)
   - Week-by-week action plan
   - Common pitfalls and best practices

2. **CQRS_MIGRATION_SUMMARY.md** (12KB, 370 lines) 📋 EXECUTIVE SUMMARY
   - TL;DR - What needs to happen
   - Architecture comparison (before/after)
   - What's implemented vs missing
   - Message type mapping (129 messages analyzed)
   - Critical path to fix cache bug
   - Immediate action items (week by week)
   - FAQ section

3. **CQRS_MIGRATION_COMPLETE.md** (39KB, 1,256 lines) 📖 DETAILED ANALYSIS
   - Complete architecture analysis
   - Current state assessment (156KB actor, 46 handlers, 129 messages)
   - Problems identified (ActorGraphRepository, missing directives, no events)
   - Target architecture with mermaid diagrams
   - Message mapping for all 129 types
   - Critical missing components
   - Migration strategy (4 phases)
   - Event flow examples with sequence diagrams
   - File structure after migration
   - Testing strategy
   - Success criteria and risk assessment
   - Timeline estimate (4-7 weeks)

4. **CQRS_DIRECTIVE_TEMPLATE.md** (25KB, 855 lines) 🔧 IMPLEMENTATION TEMPLATE
   - Copy-paste directive handler implementations
   - Complete code for all 6 directive handlers:
     - CreateNodeHandler
     - CreateEdgeHandler
     - UpdateNodePositionHandler
     - BatchUpdatePositionsHandler
     - DeleteNodeHandler
     - DeleteEdgeHandler
   - Domain event definitions
   - HTTP handler integration examples
   - AppState wiring code
   - Testing templates
   - Ready to use - just copy and adapt!

---

## 🎯 The Mission

**Goal**: Decompose the monolithic GraphServiceActor (156KB, 4,614 lines) into pure CQRS handlers following hexagonal architecture.

**Why**:
- Fix cache coherency bug (GitHub sync writes 316 nodes, but stale cache shows 63)
- Remove ActorGraphRepository technical debt
- Enable event-driven cache invalidation
- Reduce 4,614 line monolith to small, focused handlers

**Current Progress**:
- ✅ Phase 1 Complete: Query operations migrated (8 query handlers)
- ❌ Phase 2 Needed: Directive handlers for write operations
- ❌ Phase 3 Critical: Event bus for cache invalidation (fixes bug!)
- ⚠️ Phase 4 Blocked: Actor removal (pending Phase 2-3)

---

## 📊 Quick Stats

### GraphServiceActor Analysis

| Metric | Value |
|--------|-------|
| File size | 156,158 bytes (152KB) |
| Line count | 4,614 lines |
| Token estimate | ~48,000 tokens |
| Message handlers | 46 implemented |
| Message types | 129 defined |
| Dependencies | GPU, WebSocket, semantic, physics, settings |

### Migration Status

| Component | Status | Progress |
|-----------|--------|----------|
| Query handlers | ✅ Complete | 8/8 handlers |
| Directive handlers | ❌ Missing | 0/6 handlers |
| Event bus | ❌ Not integrated | 0% |
| HTTP handlers | ⚠️ Mixed | ~30% CQRS |
| ActorGraphRepository | ⚠️ Deprecated | Still used |

**Overall Progress**: 15% complete (Phase 1 done)

---

## 🚀 Immediate Next Steps (This Week)

### Priority 1: Create Directive Handlers (1-2 days)

**File to create**: `src/application/graph/directives.rs`

**Use template from**: `CQRS_DIRECTIVE_TEMPLATE.md`

**Start with**:
1. CreateNodeHandler (replaces AddNode actor message)
2. CreateEdgeHandler (replaces AddEdge actor message)

**Test**:
- Unit test each handler (see template)
- Integration test: POST → verify DB write → verify response

### Priority 2: Update HTTP Handlers (1-2 days)

**File to update**: `src/handlers/api_handler/graph/mod.rs`

**Changes**:
- Replace `state.graph_service_actor.send(AddNode { node })`
- With `state.graph_directive_handlers.create_node.handle(CreateNode { node })`

**Test**:
- Manual testing via API calls
- Integration test: End-to-end flow

### Priority 3: Wire AppState (1 day)

**File to update**: `src/app_state.rs`

**Add**:
```rust
pub struct GraphDirectiveHandlers {
    pub create_node: Arc<CreateNodeHandler>,
    pub create_edge: Arc<CreateEdgeHandler>,
    // ... others
}
```

---

## 🔍 Architecture Comparison

### BEFORE (Current Hybrid - THE PROBLEM)

```
┌─────────────────────────────────────────┐
│  GraphServiceActor (156KB, 4614 lines)  │
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
```

**Problems**:
- Actor holds stale in-memory cache
- GitHub sync writes to DB but doesn't invalidate cache
- API returns 63 nodes instead of 316
- 4,614 line monolith unmaintainable
- Mixed concerns (CRUD + physics + WebSocket + cache)

### AFTER (Pure CQRS - THE SOLUTION)

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

**Benefits**:
- Database is single source of truth (no stale cache)
- GitHub sync emits event → cache invalidated
- API always returns fresh data (316 nodes!)
- Small, focused handlers instead of monolith
- Event-driven architecture for loose coupling

---

## 📁 Files Created by This Analysis

| File | Size | Lines | Purpose |
|------|------|-------|---------|
| QUICK_REFERENCE.md | 7.6KB | 233 | Quick visual guide, start here |
| CQRS_MIGRATION_SUMMARY.md | 12KB | 370 | Executive summary, action plan |
| CQRS_MIGRATION_COMPLETE.md | 39KB | 1,256 | Complete analysis, architecture |
| CQRS_DIRECTIVE_TEMPLATE.md | 25KB | 855 | Copy-paste implementation code |
| **Total** | **84KB** | **2,714** | Complete migration documentation |

---

## ✅ Success Criteria

### Phase 1: Queries ✅ COMPLETE
- [x] All GET endpoints use query handlers
- [x] Query latency <50ms (p95)
- [x] Test coverage >80%
- [x] Zero performance regression

### Phase 2: Commands 🎯 IN PROGRESS (You Are Here)
- [ ] `src/application/graph/directives.rs` created
- [ ] All 6 directive handlers implemented
- [ ] HTTP handlers use directives, not actor messages
- [ ] ActorGraphRepository deprecated for writes
- [ ] Unit tests for each handler
- [ ] Integration test: POST → DB write → verify
- [ ] Zero data loss during migration

### Phase 3: Events 🎯 CRITICAL FOR BUG FIX
- [ ] Domain events enhanced with graph event types
- [ ] Event bus implemented (in-memory pub/sub)
- [ ] Cache invalidation subscriber implemented
- [ ] WebSocket broadcaster subscriber implemented
- [ ] GitHub sync emits `GraphSyncCompleted` event
- [ ] Integration test: sync → event → cache clear
- [ ] ⭐ **API returns 316 nodes after sync** ✅ BUG FIXED!

### Phase 4: Actor Removal 🎯 FINAL CLEANUP
- [ ] Physics simulation extracted to domain service
- [ ] WebSocket coordination extracted to adapter
- [ ] GraphServiceActor deleted (4,614 lines!)
- [ ] ActorGraphRepository deleted
- [ ] All tests passing
- [ ] No actor references in HTTP handlers
- [ ] Documentation updated

---

## ⚠️ Critical Warnings

### Don't Do These
- ❌ Send actor messages from HTTP handlers (use directive handlers)
- ❌ Update ActorGraphRepository (it's deprecated, use UnifiedGraphRepository)
- ❌ Add more message types to GraphServiceActor (we're removing it!)
- ❌ Keep cache in actor (database is source of truth)

### Do These
- ✅ Use directive handlers for all write operations
- ✅ Use query handlers for all read operations
- ✅ Use UnifiedGraphRepository directly
- ✅ Emit events after persistence
- ✅ Let event subscribers handle side effects (cache, WebSocket)

---

## 📞 Getting Help

### Questions About Implementation?
- **Which file do I start with?** → Create `src/application/graph/directives.rs` from template
- **How do I test directives?** → See testing section in CQRS_DIRECTIVE_TEMPLATE.md
- **What about physics simulation?** → Leave in actor for now (Phase 4)
- **How do I wire handlers?** → See AppState section in template
- **When do I emit events?** → After successful persistence in directive handlers

### Questions About Architecture?
- **Why not just fix actor cache?** → Band-aid solution, proper architecture prevents issues by design
- **Can we keep actor for some things?** → Yes! Physics and WebSocket can stay initially (Phase 4)
- **What's the migration risk?** → Medium. Phase 1 done with zero issues. Can run both systems in parallel.
- **How long will this take?** → 4-7 weeks total (1 week done, 3-6 weeks remaining)

### References
- **Full Analysis**: CQRS_MIGRATION_COMPLETE.md
- **Code Template**: CQRS_DIRECTIVE_TEMPLATE.md
- **Quick Guide**: QUICK_REFERENCE.md
- **Action Plan**: CQRS_MIGRATION_SUMMARY.md

---

## 🎓 Key Takeaways

### The Problem
GraphServiceActor is a 156KB, 4,614 line monolith with 46 message handlers managing an in-memory cache that goes stale after GitHub sync (63 nodes shown instead of 316).

### The Solution
Decompose actor into CQRS directive/query handlers that use UnifiedGraphRepository directly, emit domain events for side effects (cache invalidation, WebSocket broadcasting), and use database as single source of truth.

### The Plan
- ✅ Phase 1 (1 week): Query handlers DONE
- 🔄 Phase 2 (1-2 weeks): Directive handlers IN PROGRESS
- 🎯 Phase 3 (1-2 weeks): Event bus CRITICAL FOR BUG FIX
- 🎯 Phase 4 (1-2 weeks): Actor removal FINAL CLEANUP

### The Impact
- Fixes cache coherency bug (316 nodes displayed correctly)
- Reduces 4,614 line monolith to small, focused handlers
- Enables event-driven architecture
- Improves testability and maintainability
- Removes ActorGraphRepository technical debt

---

## 📈 Timeline

| Week | Phase | Tasks | Deliverable |
|------|-------|-------|-------------|
| 1 (Done) | Phase 1 | Query handlers | ✅ 8 query handlers |
| 2-3 | Phase 2 | Directive handlers | 6 directive handlers + HTTP updates |
| 4-5 | Phase 3 | Event bus | Event-driven cache invalidation ⭐ |
| 6-7 | Phase 4 | Actor removal | GraphServiceActor deleted |

**Total**: 4-7 weeks (1 week complete, 15% done)

---

## 🚀 Start Coding!

**Next file to create**: `src/application/graph/directives.rs`

**Copy from**: `CQRS_DIRECTIVE_TEMPLATE.md` (lines 1-855)

**Start with**: CreateNodeHandler and CreateEdgeHandler

**Test**: Unit tests from template, then integration test POST /api/graph/nodes

**Good luck!** 🎯

---

**Prepared by**: System Architecture Designer (Agent 3)
**Date**: November 3, 2025
**Analysis based on**:
- GraphServiceActor (156KB, 4,614 lines, 46 handlers, 129 message types)
- Existing CQRS implementation (8 query handlers)
- ActorGraphRepository adapter analysis
- Hexagonal/CQRS architecture patterns
