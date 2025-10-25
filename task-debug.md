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

## 5. Next Steps

**Immediate** (automated): Wait for GitHub sync to trigger save_graph and capture the detailed error messages showing:
-   Exact SQLite error code and extended code
-   Which constraint failed (FOREIGN KEY or UNIQUE)
-   Which specific node IDs or edge IDs caused the failure

**After Error Identified**: Implement fix based on the specific constraint violation captured in logs.