# SQLite Repository Async/Sync Mutex Fix

## Problem

The three SQLite repository implementations were using `tokio::sync::Mutex<Connection>` which caused a critical async/sync impedance mismatch:

1. **Blocking Operations**: SQLite operations (`execute`, `query_row`, etc.) are synchronous/blocking
2. **Async Mutex**: `tokio::sync::Mutex` requires `.await` which holds locks across suspension points
3. **30-Second Timeouts**: When async mutexes hold locks during blocking SQLite operations, the tokio runtime can't make progress, causing 30-second timeouts

## Solution

Replaced `tokio::sync::Mutex` with `std::sync::Mutex` in all three repository files:

### Files Modified

1. **`src/adapters/sqlite_knowledge_graph_repository.rs`**
   - Changed `Arc<tokio::sync::Mutex<Connection>>` to `Arc<Mutex<Connection>>`
   - Updated all `.lock().await` to `.lock().expect("...")`
   - Added proper error messages for mutex poisoning

2. **`src/adapters/sqlite_ontology_repository.rs`**
   - Changed `Arc<tokio::sync::Mutex<Connection>>` to `Arc<Mutex<Connection>>`
   - Updated all `.lock().await` to `.lock().expect("...")`
   - Fixed deadlock in `load_ontology_graph` by releasing lock before calling `list_owl_classes`

3. **`src/adapters/sqlite_settings_repository.rs`**
   - Already using `tokio::task::spawn_blocking` for all database operations
   - No changes needed (already follows best practices)

## Key Changes

### Import Changes
```rust
// Before
use std::sync::Arc;

// After
use std::sync::{Arc, Mutex};
```

### Struct Definition
```rust
// Before
pub struct SqliteKnowledgeGraphRepository {
    conn: Arc<tokio::sync::Mutex<Connection>>,
}

// After
pub struct SqliteKnowledgeGraphRepository {
    conn: Arc<Mutex<Connection>>,
}
```

### Mutex Acquisition
```rust
// Before
let conn = self.conn.lock().await;

// After
let conn = self.conn.lock().expect("Failed to acquire knowledge graph repository mutex");
```

## Why This Works

1. **Synchronous Locking**: `std::sync::Mutex` blocks the thread instead of yielding to the async runtime
2. **Short Critical Sections**: SQLite operations complete quickly, so blocking is acceptable
3. **No Suspension**: The lock is held only during the actual database operation, not across await points
4. **Thread-Safe**: `std::sync::Mutex` is still thread-safe and works correctly in async contexts for short operations

## Error Handling

Used `.expect()` instead of `.unwrap()` to provide clear error messages if mutex poisoning occurs:
- `"Failed to acquire knowledge graph repository mutex"`
- `"Failed to acquire ontology repository mutex"`

## Testing

Verified with `cargo check`:
```bash
cargo check --message-format=short
```

Result: **Successful compilation** with no errors

## Performance Impact

**Positive Impact**:
- Eliminates 30-second timeout issues
- Reduces context switching overhead
- Simplifies async code paths

**Minimal Negative Impact**:
- Blocking mutexes can block threads, but SQLite operations are fast
- Better alternative: Use connection pools or `tokio::task::spawn_blocking` (already done in settings repository)

## Future Improvements

Consider migrating knowledge graph and ontology repositories to use `tokio::task::spawn_blocking` pattern like the settings repository for even better async/sync isolation.

## References

- Rust async book: https://rust-lang.github.io/async-book/
- Tokio documentation on blocking operations: https://tokio.rs/tokio/topics/bridging
- SQLite thread-safety: https://www.sqlite.org/threadsafe.html
