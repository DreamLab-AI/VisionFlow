# VisionFlow Data Flow - Root Cause Analysis

**Status:** IDENTIFIED ✓
**Date:** 2025-11-01

---

## Executive Summary

The client displays 0 nodes because the **GraphServiceActor starts with empty in-memory state** and never loads from the database. The system is designed to populate data via GitHub sync, but the sync endpoint is unreachable due to a routing misconfiguration.

---

## Complete Data Flow (As Designed)

```
GitHub (logseq repo)
    ↓
POST /api/admin/sync → GitHubSyncService
    ↓
Parse markdown files (look for "public:: true")
    ↓
UnifiedGraphRepository.add_nodes()
    ↓
SQLite knowledge_graph.db (PERSISTENT)
    ↓
??? (MISSING STEP) ???
    ↓
GraphServiceActor.graph_data (IN-MEMORY)
    ↓
ActorGraphRepository.get_graph()
    ↓
GetGraphDataHandler (CQRS)
    ↓
GET /api/graph/data
    ↓
Client GraphDataManager
    ↓
WebGL Rendering (Three.js)
```

---

## The Missing Link

###  Database → Actor Loading

**What We Confirmed:**
1. ✅ Database has 5 test nodes (verified with sqlite3)
2. ✅ Database schema is correct
3. ✅ API endpoint `/api/graph/data` works (returns 200)
4. ❌ API returns 0 nodes despite database having 5

**Root Cause:**
```rust
// src/main.rs:386-388
// Database is empty - actor will remain empty until GitHub sync completes
info!("⏳ GraphServiceActor will remain empty until GitHub sync finishes");
info!("ℹ️  You can manually trigger sync via /api/admin/sync endpoint");
```

**The Problem:**
- GraphServiceActor initializes with `Arc::new(GraphData::default())` (empty)
- Actor `started()` method does NOT load from database
- Actor `GetGraphData` handler returns `Arc::clone(&self.graph_data)` (empty!)
- System expects `/api/admin/sync` to populate the actor
- Sync endpoint has routing bug → Never gets called → Actor stays empty forever

---

## Why Sync Endpoint Fails

### Routing Bug

**File:** `src/handlers/admin_sync_handler.rs:77-82`

**Before Fix:**
```rust
pub fn configure_routes(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/admin")              // Creates /admin scope
            .route("/sync", ...)           // Adds /sync inside scope
    );
}
```

**Result:** Route becomes `/api/admin/admin/sync` (404)

**After Fix:**
```rust
pub fn configure_routes(cfg: &mut web::ServiceConfig) {
    cfg.route("/admin/sync", web::post().to(trigger_sync));
}
```

**Result:** Route should be `/api/admin/sync` ✓

---

## Data Flow States

### Current State (Broken)

| Component | Status | Data Count |
|-----------|--------|------------|
| knowledge_graph.db | ✅ Working | 5 nodes, 5 edges |
| GitHubSyncService | ⚠️ Unreachable | N/A |
| GraphServiceActor | ❌ Empty | 0 nodes, 0 edges |
| API Response | ❌ Empty | 0 nodes, 0 edges |
| Client Display | ❌ Empty | 0 nodes rendered |
| GPU Pipeline | ❌ Inactive | No data to process |

### Expected State (After Fix)

| Component | Status | Data Count |
|-----------|--------|------------|
| GitHub (logseq) | ✅ Source | ~100+ markdown files |
| POST /api/admin/sync | ✅ Accessible | Triggers ingestion |
| GitHubSyncService | ✅ Running | Fetches & parses files |
| knowledge_graph.db | ✅ Populated | 100+ nodes, edges |
| GraphServiceActor | ✅ Loaded | 100+ nodes in memory |
| API Response | ✅ Working | Returns full graph |
| Client Display | ✅ Rendering | Nodes visible in 3D |
| GPU Pipeline | ✅ Active | Computing physics |

---

## Fix Strategy

### Option 1: Fix Sync Endpoint (Primary)

1. ✅ Route configuration fixed
2. ⏳ Rebuild backend with fix
3. ⏳ Trigger `POST /api/admin/sync`
4. ⏳ Verify data flows to actor
5. ⏳ Confirm client rendering

### Option 2: Load from Database on Startup (Alternative)

Modify `GraphServiceActor::started()` to:
```rust
fn started(&mut self, ctx: &mut Self::Context) {
    info!("GraphServiceActor started");

    // Load existing data from database
    ctx.address().do_send(LoadFromDatabase);

    // Then initialize simulation
    ctx.address().do_send(InitializeActor);
}
```

This would make the system resilient to sync failures.

---

## GPU Pipeline Dependencies

The GPU pipeline is **correctly implemented** but **waiting for data**:

```
GraphServiceActor (0 nodes)
    ↓
NO DATA TO PROCESS
    ↓
ForceComputeActor (idle)
StressMajorizationActor (idle)
OntologyConstraintActor (idle)
    ↓
Client receives empty positions
    ↓
WebGL renders nothing
```

Once data flows into GraphServiceActor, the GPU pipeline will automatically activate.

---

## Verification Steps

### 1. Database Verification ✅
```bash
docker cp visionflow_container:/app/data/knowledge_graph.db /tmp/kg.db
sqlite3 /tmp/kg.db "SELECT COUNT(*) FROM nodes; SELECT COUNT(*) FROM edges;"
# Result: 5 nodes, 5 edges ✅
```

### 2. API Verification ✅
```bash
curl http://localhost:4000/api/graph/data | jq '.nodes | length'
# Result: 0 (actor is empty)
```

### 3. Actor State Verification
```bash
# Check logs for actor initialization
docker logs visionflow_container | grep "GraphServiceActor"
# Shows: "will remain empty until GitHub sync finishes"
```

### 4. Sync Endpoint Verification ❌
```bash
curl -X POST http://localhost:4000/api/admin/sync
# Result: 404 Not Found (routing bug)
```

---

## Next Steps

1. **Wait for backend rebuild** (cargo build in progress)
2. **Restart backend** with fixed route
3. **Trigger sync:** `curl -X POST http://localhost:4000/api/admin/sync`
4. **Monitor logs** for GitHub sync progress
5. **Verify data flow:**
   - Database gets populated
   - Actor loads data
   - API returns nodes
   - Client renders graph
6. **Confirm GPU activation** in logs

---

## Technical Insights

### Why In-Memory?
GraphServiceActor uses in-memory state for performance:
- ✅ Fast physics simulation (60 FPS updates)
- ✅ No database I/O in render loop
- ❌ Requires explicit loading mechanism

### Why GitHub Sync?
System designed for "knowledge graph as code":
- Markdown files in Git → Version control
- `public:: true` marker → Selective publishing
- Continuous sync → Real-time updates

### Why Not Load on Startup?
Current design assumes:
- Fresh deployment = empty database
- First sync populates everything
- **Gap:** Existing data in database gets ignored!

---

## Recommendation

**Implement Both:**
1. ✅ Fix sync endpoint (immediate)
2. 📋 Add database loading on startup (future resilience)

This ensures:
- Sync works for initial population
- Restarts preserve existing data
- System is fault-tolerant
