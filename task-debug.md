# Debugging Summary: GitHub Sync Service

## 1. Initial Problem

The GitHub sync service was failing to populate the `knowledge_graph.db`. The API returned stale data (4 nodes), and logs showed `UNIQUE constraint failed` and `nested transaction` errors.

## 2. Investigation and Fixes

### Round 1: Code Logic and Race Conditions

-   **Hypothesis**: The application was crashing on startup due to a race condition in the database connection pool.
-   **Evidence**: Logs showed an "Address already in use" error, indicating a zombie process was holding the port. A subsequent restart revealed a "timed out waiting for connection" error during the settings migration.
-   **Analysis**: The `DatabaseService` was configured with a connection pool of size 1. The `AppState::new` function was attempting multiple concurrent database operations, leading to a deadlock.
-   **Fix 1**: Increased the connection pool size to 10 in `database_service.rs`.
-   **Fix 2**: Moved the settings migration to a `spawn_blocking` task in `app_state.rs` to prevent blocking the main thread.
-   **Fix 3**: Corrected a compile error by adding a missing `log::error` import in `sqlite_knowledge_graph_repository.rs`.

### Round 2: Silent Transaction Failure

-   **Hypothesis**: After fixing the startup crashes, the sync runs but fails silently within the `save_graph` database transaction.
-   **Evidence**:
    -   The application now runs without crashing.
    -   Logs show the sync process starting and parsing files.
    -   Direct database queries show that the `kg_nodes` table is cleared (`DELETE` works) but is then left with 0 nodes (the `INSERT` fails).
-   **Analysis**: The transaction is being rolled back. The error is not being logged because it's a database-level constraint violation that is not being caught by the existing Rust error handling. The most likely cause is a foreign key constraint violation when inserting edges.

## 3. Current Status

-   The application is stable and runs without crashing.
-   The GitHub sync process starts and runs.
-   The root cause has been narrowed down to a silent transaction failure in the `save_graph` function.

### Round 3: Enhanced Error Logging (In Progress)

-   **Action**: Added detailed `rusqlite::Error` logging to `save_graph` function in `sqlite_knowledge_graph_repository.rs`
-   **Changes Made**:
    -   **Node insertion** (lines 261-307): Enhanced error handling with SQLite error code extraction, UNIQUE constraint detection
    -   **Edge insertion** (lines 305-355): Enhanced error handling with SQLite error code extraction, FOREIGN KEY constraint detection
    -   Both now log: error message, SQLite error code, extended code, and constraint type
-   **File**: `/home/devuser/workspace/project/src/adapters/sqlite_knowledge_graph_repository.rs:261-355`
-   **Rebuild Status**: ✅ Backend recompiled successfully (0.27s build time)
-   **Container Status**: ✅ Running with fresh databases (cleared at 21:37:50Z)
-   **Monitoring**: Background tasks capturing logs for 210 seconds (GitHub sync duration)

## 4. Current Status

-   **Application**: Stable, running with enhanced error logging
-   **Databases**: Cleared and ready for fresh sync
-   **GitHub sync**: Starting (typically takes ~3.5 minutes)
-   **Monitoring tasks**:
    -   Task 3a4444: Capturing first 60 seconds of logs
    -   Task 60cd37: Checking for save_graph errors after 90 seconds
    -   Task fa4d4d: Checking node count after 210 seconds
    -   Task aab791: Getting recent container logs after 30 seconds

## 5. Container Communication Fix (2025-10-25 22:25)

**Issue**: agentic-workstation Management API was not running, causing connection failures.

**Fix Applied**:
```bash
docker exec agentic-workstation bash -c "cd /opt/management-api && node server.js" &
```

**Verification**:
- ✅ Management API responding on http://agentic-workstation:9090
- ✅ AgentMonitorActor successfully polling every 3 seconds
- ✅ Logs show: `reuse idle connection for ("http", agentic-workstation:9090)`
- ✅ No more network errors

**Documentation**: `/home/devuser/workspace/project/CONTAINER_COMMUNICATION_FIX.md`

## 6. Current Status (2025-10-25 22:28)

**Database Status**:
- Files exist: `/app/data/knowledge_graph.db` (4KB), WAL file (840KB)
- **EMPTY**: No tables (`no such table: kg_nodes`)
- Database created at: Oct 25 21:36 (container start time)

**GitHub Sync Status**:
- ❌ NO sync has run since container start
- ❌ No "GitHub sync" messages in logs
- API returns 529 nodes (source unknown - possibly in-memory defaults)

**Application Status**:
- ✅ webxr process running (PID 23)
- ✅ API responding on port 4000
- ✅ AgentMonitorActor working correctly

## 7. Root Cause Identified (2025-10-25 22:35)

**Issue**: UNIQUE constraint violation on `kg_edges.id`

**Error Message**:
```
Failed to save accumulated knowledge graph: Database error: Failed to insert edge 3: UNIQUE constraint failed: kg_edges.id
```

**Root Cause**:
- Edge IDs are generated as `format!("{}_{}", source_id, target_id)` in `knowledge_graph_parser.rs:154`
- When multiple markdown files link to the same target from the same source, they create duplicate edge IDs
- Example: If files A and B both link from node 1 to node 2, both create edge ID "1_2"
- Database has UNIQUE constraint on `kg_edges.id` (schema shows `id TEXT PRIMARY KEY`)
- Nodes are deduplicated using HashMap, but edges are accumulated in a Vec without deduplication

**Sync Statistics**:
- Duration: 201 seconds (~3.5 minutes)
- Files scanned: 731 total (188 knowledge graph, 241 ontology, 300 skipped)
- Nodes found: 529 unique nodes (deduplicated successfully)
- Edges found: 1263 edges (includes duplicates - PROBLEM!)
- Failed at: Database save step

**Fix Required**: Deduplicate edges using HashMap by edge ID before saving to database

## 8. Fix Implemented and Verified (2025-10-25 22:44)

**Changes Made**:
1. ✅ Modified `src/services/github_sync_service.rs:93-94` to use HashMap for edge deduplication
2. ✅ Updated edge accumulation logic to use `HashMap<String, Edge>` instead of `Vec<Edge>`
3. ✅ Converted HashMap to Vec before saving: `let edge_vec = accumulated_edges.into_values().collect()`
4. ✅ Updated log message from "edges" to "unique edges" for clarity

**Build and Deploy**:
- Backend rebuilt successfully in 37.64 seconds
- Database cleared and container restarted
- Fresh GitHub sync completed at 22:44:15 UTC

**Results**:
- ✅ **Before fix**: 1263 edges (with duplicates) → UNIQUE constraint error
- ✅ **After fix**: 839 unique edges → Saved successfully
- ✅ **Duplicates removed**: 424 edges (33% reduction)
- ✅ **Database**: knowledge_graph.db contains all 529 nodes and 839 edges
- ✅ **API**: Returns 529 nodes and 839 edges correctly

**File**: `src/services/github_sync_service.rs:90-278`

## 9. Permanent Fix Applied (2025-10-25 23:00)

**Management API Autostart** - ✅ COMPLETED:
- **Status**: Permanent fix applied to multi-agent-docker source
- **Changes**:
  1. Created health check script: `unified-config/scripts/verify-management-api.sh`
  2. Updated supervisord config: Added `management-api-healthcheck` program
  3. Updated entrypoint: Added Phase 7.5 to install health check script
- **How it works**:
  - Supervisord starts Management API (priority 300)
  - Health check runs after all services start (priority 950)
  - Verifies API responds on port 9090 within 60 seconds
  - Automatically restarts service if not running
  - Logs detailed diagnostics on failure
- **Documentation**: See `CONTAINER_COMMUNICATION_FIX.md` for complete details
- **Testing**: Next agentic-workstation container rebuild will test the fix

**Files Modified**:
- `multi-agent-docker/unified-config/scripts/verify-management-api.sh` (created)
- `multi-agent-docker/unified-config/supervisord.unified.conf` (lines 226-240 added)
- `multi-agent-docker/unified-config/entrypoint-unified.sh` (lines 314-363 added)
- `CONTAINER_COMMUNICATION_FIX.md` (documented permanent fix)
- `task-debug.md` (this file - marked task complete)