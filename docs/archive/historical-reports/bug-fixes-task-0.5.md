# Bug Fixes for Task 0.5 Integration Testing
**Date:** November 2, 2025
**Session:** Post-Integration Testing Bug Fixes

---

## Critical Bugs Identified

### 1. GPU Physics Initialization Bug ❌ CRITICAL
**File:** `src/actors/physics_orchestrator_actor.rs`
**Lines:** 273-299 (function `initialize_gpu_if_needed`)
**Severity:** CRITICAL - Prevents GPU physics from ever running

#### Problem:
The fix from commit 5e64e700 was **only applied to `GraphServiceActor`** but **NOT to `PhysicsOrchestratorActor`**.

```rust
// CURRENT CODE (BUGGY) - Lines 295-296
self.gpu_initialized = true;  // ❌ Sets flag IMMEDIATELY
self.gpu_init_in_progress = false;
```

The code sets `gpu_initialized = true` immediately after sending the `InitializeGPU` message, without waiting for confirmation from the GPU actor. This means:
1. Flag gets set to `true` prematurely
2. GPU hardware never actually initializes
3. Physics simulation thinks GPU is ready when it's not
4. All node velocities remain at 0
5. Kinetic energy stays at 0

#### Correct Pattern (from GraphServiceActor):
```rust
// graph_actor.rs lines 3893-3900 (CORRECT)
impl Handler<GPUInitialized> for GraphServiceActor {
    fn handle(&mut self, _msg: GPUInitialized, _ctx: &mut Self::Context) {
        info!("✅ GPU initialization CONFIRMED - GPUInitialized message received");
        self.gpu_initialized = true;  // ✅ Only sets flag AFTER receiving message
        self.gpu_init_in_progress = false;
    }
}
```

#### Root Cause:
`PhysicsOrchestratorActor` is **missing the `Handler<GPUInitialized>` implementation** entirely.

#### Fix Required:

**Step 1:** Remove lines 295-296 from `initialize_gpu_if_needed`:
```rust
// DELETE THESE LINES:
// self.gpu_initialized = true;
// self.gpu_init_in_progress = false;
```

**Step 2:** Add `GPUInitialized` handler at the end of the file (after line 1105):
```rust
/// Handler for GPU initialization confirmation
#[cfg(feature = "gpu")]
impl Handler<crate::actors::messages::GPUInitialized> for PhysicsOrchestratorActor {
    type Result = ();

    fn handle(&mut self, _msg: crate::actors::messages::GPUInitialized, _ctx: &mut Self::Context) -> Self::Result {
        info!("✅ GPU initialization CONFIRMED for PhysicsOrchestrator - GPUInitialized message received");
        self.gpu_initialized = true;
        self.gpu_init_in_progress = false;

        // Log that physics is now ready to start
        info!("Physics simulation GPU initialization complete - ready for simulation");
    }
}
```

**Step 3:** Update `InitializeGPU` message to include PhysicsOrchestratorActor address:

In `initialize_gpu_if_needed` (line 284), change:
```rust
// CURRENT:
gpu_addr.do_send(InitializeGPU {
    graph: Arc::clone(graph_data),
    graph_service_addr: None,  // ❌ No address to send GPUInitialized back to
    gpu_manager_addr: None,
});

// FIXED:
gpu_addr.do_send(InitializeGPU {
    graph: Arc::clone(graph_data),
    graph_service_addr: Some(ctx.address()),  // ✅ Provide our address
    gpu_manager_addr: None,
});
```

---

### 2. WebSocket Route Configuration Issue ⚠️ IMPORTANT
**Files:**
- `src/main.rs` line 404
- `client/vite.config.ts` lines 79-88
**Severity:** HIGH - Client cannot connect to WebSocket

#### Problem:
The backend route is configured as `/wss` but the client (according to Chrome DevTools error logs) is trying to connect to `ws://192.168.0.51:3001/ws`.

**Backend route (main.rs:404):**
```rust
.route("/wss", web::get().to(socket_flow_handler))  // ❌ Route is /wss
```

**Client proxy (vite.config.ts:79-87):**
```typescript
'/ws': {
  target: process.env.VITE_WS_URL || 'ws://visionflow_container:4000',
  ws: true,
  changeOrigin: true
},
'/wss': {
  target: process.env.VITE_WS_URL || 'ws://visionflow_container:4000',
  ws: true,
  changeOrigin: true
}
```

**Client error from integration testing:**
```
Error during WebSocket handshake: Unexpected response code: 200
Connection to Backend Failed - Running in offline mode
```

#### Root Cause Analysis:

The issue is that **Nginx is not configured to proxy WebSocket connections**. When the client tries to connect to `ws://192.168.0.51:3001/ws`, Nginx returns HTTP 200 instead of upgrading the connection to WebSocket protocol.

#### Fix Required:

**Option 1 (Recommended): Fix Nginx Configuration**

Check the Nginx configuration file in the container (likely `/etc/nginx/sites-available/visionflow` or similar) and ensure WebSocket proxy is configured:

```nginx
location /ws {
    proxy_pass http://localhost:4000/wss;  # Forward to backend /wss route
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_read_timeout 86400;
}
```

**Option 2: Standardize on /wss everywhere**

Update client code to connect to `/wss` instead of `/ws`, or update backend to use `/ws`.

---

### 3. Node Count Discrepancy (48 vs 900+) ⚠️ IMPORTANT
**Expected:** 900+ nodes from GitHub sync
**Actual:** 48 nodes
**Severity:** HIGH - Missing data

#### Problem:
API returns only 48 nodes instead of the expected 900+ nodes from GitHub sync.

**API Response:**
```json
{
  "nodes": 48,
  "edges": 47,
  "settlement": { "settled": false, "frames": 0, "KE": 0 }
}
```

#### Possible Causes:

1. **GitHub sync incomplete or failed**
   - Sync may not have completed
   - Sync errors were not logged
   - Database has more nodes but graph actor not reloaded

2. **Database has data but graph not reloaded**
   - Database size is 532KB (suggests more than 48 nodes)
   - Graph actor may be serving stale/cached data
   - `ReloadGraphFromDatabase` message not triggered after sync

3. **API returning wrong graph**
   - API might be returning ontology graph instead of KG nodes
   - Graph filtering removing most nodes
   - Wrong query being used

#### Investigation Required:

1. **Check database content directly:**
```bash
docker exec visionflow_container sh -c "apt-get update && apt-get install -y sqlite3"
docker exec visionflow_container sqlite3 /app/data/unified.db "SELECT COUNT(*) FROM nodes;"
docker exec visionflow_container sqlite3 /app/data/unified.db "SELECT COUNT(*) FROM edges;"
```

2. **Check GitHub sync logs:**
```bash
docker logs visionflow_container 2>&1 | grep -i "github\|sync" | tail -50
```

3. **Trigger graph reload:**
```bash
# Check if endpoint exists
curl -X POST http://192.168.0.51:4000/api/graph/reload

# Or check internal endpoint
docker exec visionflow_container curl -X POST http://localhost:4000/api/graph/reload
```

4. **Check graph actor state:**
Add logging to see what the GraphServiceActor thinks is loaded:
- Check `graph_data.nodes.len()`
- Check when last reload happened
- Verify GitHub sync completion status

---

## Summary of Required Changes

### Files to Modify:

1. **`src/actors/physics_orchestrator_actor.rs`**
   - Remove lines 295-296 (immediate flag setting)
   - Modify line 284-288 (add ctx.address())
   - Add `Handler<GPUInitialized>` implementation (after line 1105)

2. **Nginx configuration** (location TBD in container)
   - Add WebSocket proxy configuration for `/ws` → `/wss`
   - Or update route to use `/ws` consistently

3. **Investigation needed:**
   - Database node count verification
   - GitHub sync status check
   - Graph reload mechanism testing

---

## Testing Plan

### After GPU Physics Fix:
1. Rebuild container with updated code
2. Start application and monitor logs for: `✅ GPU initialization CONFIRMED for PhysicsOrchestrator`
3. Check API for non-zero velocities:
```bash
curl http://192.168.0.51:4000/api/graph/data | jq '.nodes[] | select(.data.velocity.x != 0 or .data.velocity.y != 0 or .data.velocity.z != 0) | {id, velocity: .data.velocity}'
```
4. Verify kinetic energy > 0 in settlement status

### After WebSocket Fix:
1. Connect Chrome DevTools to http://192.168.0.51:3001
2. Check console - should see: `✅ WebSocket connected`
3. Should receive `InitialGraphLoad` message
4. Should receive `PositionUpdate` messages streaming
5. Status should change from "WAITING FOR TELEMETRY" to active

### After Node Count Investigation:
1. Verify database contains 900+ nodes
2. Trigger graph reload if needed
3. Verify API returns all nodes
4. Verify client renders all nodes

---

## Priority Order:
1. **Priority 1:** GPU Physics Fix (blocks all physics simulation)
2. **Priority 2:** Node Count Investigation (missing data)
3. **Priority 3:** WebSocket Fix (blocks real-time updates)

All three must be fixed for Task 0.5 to pass.
