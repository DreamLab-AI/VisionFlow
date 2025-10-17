# Rust Compilation Error - Missing schema/ontology_db.sql

## Problem
The Rust backend is failing to compile in the Docker container with:
```
error: couldn't read `src/services/../../schema/ontology_db.sql`: No such file or directory (os error 2)
```

## Root Cause
The `schema/` directory is not being copied into the Docker container during build. The code in `src/services/database_service.rs` uses `include_str!()` to embed the SQL schema at compile time:

```rust
// Line 63
const SCHEMA_SQL: &str = include_str!("../../schema/ontology_db.sql");

// Line 620
let schema = include_str!("../../schema/ontology_db.sql");
```

### Current State
- ✅ File exists on host: `/home/devuser/workspace/project/schema/ontology_db.sql`
- ❌ File missing in container: `/app/schema/ontology_db.sql` (doesn't exist)
- ❌ Build fails because `include_str!()` runs at compile time

## Solution
Add the `schema/` directory to the Dockerfile so it's available during compilation.

### Files to Modify

Check your Dockerfile (likely `Dockerfile.dev` or `Dockerfile.production`) and add:

```dockerfile
# Copy schema files (needed for include_str! macro at compile time)
COPY schema/ /app/schema/
```

This line should be added **before** the Rust compilation step (before `cargo build`).

### Example Dockerfile Section
```dockerfile
# Copy application source
COPY src/ /app/src/
COPY Cargo.toml Cargo.lock /app/
COPY schema/ /app/schema/  # <-- ADD THIS LINE

# Build application
RUN cargo build --release
```

## Affected Files
- `src/services/database_service.rs:63` - Uses `include_str!("../../schema/ontology_db.sql")`
- `src/services/database_service.rs:620` - Uses `include_str!("../../schema/ontology_db.sql")`

## Current Impact
- Rust backend fails to compile
- Server restarts continuously (exit status 1)
- Client shows HTTP 502 errors
- Supervisor keeps respawning rust-backend process

## Verification After Fix
After rebuilding the container, check:
```bash
# Inside container
ls -la /app/schema/ontology_db.sql

# Should show the file exists
```

## Note
SQLite doesn't need to be installed - the `rusqlite` crate is already in Cargo.toml. This is purely a build-time file inclusion issue.
