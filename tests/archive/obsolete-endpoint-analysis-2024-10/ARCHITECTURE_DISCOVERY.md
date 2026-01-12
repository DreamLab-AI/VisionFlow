# 🎯 CRITICAL ARCHITECTURE DISCOVERY

## Actual Backend Found!

### ⚠️ CORRECTION TO PREVIOUS ANALYSIS

**WRONG**: Backend is Vite dev server (Node.js)
**RIGHT**: Backend is **Rust Actix-Web server**!

---

## Running Processes

```
PID   | Process                                    | Role
------|--------------------------------------------|-----------------------
1     | supervisord                                | Process supervisor
19    | nginx master                               | Reverse proxy
21    | npm run dev                                | Package manager
24    | nginx worker                               | Proxy worker
107   | sh -c vite                                 | Shell for Vite
108   | node /app/client/.../vite                  | Frontend dev server
119   | esbuild                                    | Build tool
1870  | /app/target/debug/webxr ⭐                 | RUST BACKEND (Actix-Web)
```

---

## 🦀 Rust Backend Details

### Binary
- **Path**: `/app/target/debug/webxr`
- **PID**: 1870
- **Framework**: Actix-Web 4.11.0
- **Language**: Rust
- **Type**: Debug build
- **Memory**: 647MB

### Source Code Location
```
/app/src/               - Rust source root
/app/src/handlers/      - API endpoint handlers ⭐
/app/src/main.rs        - Entry point
/app/Cargo.toml         - Dependencies
```

### Dependencies (from Cargo.toml)
```toml
[dependencies]
actix-web = { version = "4.11.0", features = ["compress-zstd", "macros"] }
actix-cors = "0.7.1"
actix-files = "0.6"
actix = "0.13"
```

---

## Architecture Layers

```
User Request
    ↓
nginx (port 3001) → reverse proxy
    ↓
Rust Backend (port 4000) → /app/target/debug/webxr
    ↓
SQLite Databases → /app/data/*.db
```

### Frontend
```
Vite Dev Server (port 5173)
    ↓
React + TypeScript client
    ↓
Proxies API requests to Rust backend
```

---

## Why Endpoints Crash (Rust-Specific)

### Rust Panic Behavior
When Rust code panics:
1. Stack unwinding begins
2. Actix-Web catches panic
3. **TCP connection closes immediately**
4. No HTTP response sent
5. Result: curl exit code 52 (empty reply)

### Likely Crash Causes in Rust

#### 1. Unwrap on None/Error
```rust
// BAD - panics if query fails
let settings = db.query("SELECT * FROM settings").unwrap(); // ❌ PANIC!

// GOOD - handles error
let settings = db.query("SELECT * FROM settings")
    .map_err(|e| actix_web::error::ErrorInternalServerError(e))?;
```

#### 2. Index Out of Bounds
```rust
// BAD - panics if no results
let first = results[0]; // ❌ PANIC if empty!

// GOOD - safe access
let first = results.get(0).ok_or_else(|| error)?;
```

#### 3. Type Conversion Errors
```rust
// BAD - panics on invalid UTF-8
let text = String::from_utf8(bytes).unwrap(); // ❌ PANIC!

// GOOD - handles error
let text = String::from_utf8(bytes)
    .map_err(|e| actix_web::error::ErrorBadRequest(e))?;
```

---

## Why /api/config Works

**Hypothesis**: Config endpoint has proper error handling:

```rust
// /app/src/handlers/config.rs (hypothetical)
#[get("/api/config")]
async fn get_config() -> Result<HttpResponse, Error> {
    match load_config_from_db() {
        Ok(config) => Ok(HttpResponse::Ok().json(config)),
        Err(e) => {
            log::error!("Config load failed: {}", e);
            Ok(HttpResponse::InternalServerError().finish())
        }
    }
}
```

---

## Why /api/settings Crashes

**Hypothesis**: Settings endpoint has `.unwrap()` or similar:

```rust
// /app/src/handlers/settings.rs (hypothetical)
#[get("/api/settings")]
async fn get_settings() -> Result<HttpResponse, Error> {
    let db = Database::open("/app/data/settings.db").unwrap(); // ❌ PANIC if DB missing!
    let settings = db.query("SELECT * FROM settings").unwrap(); // ❌ PANIC if query fails!
    Ok(HttpResponse::Ok().json(settings))
}
```

**Fix**:
```rust
#[get("/api/settings")]
async fn get_settings() -> Result<HttpResponse, Error> {
    let db = Database::open("/app/data/settings.db")
        .map_err(|e| actix_web::error::ErrorInternalServerError(e))?;

    let settings = db.query("SELECT * FROM settings")
        .map_err(|e| actix_web::error::ErrorInternalServerError(e))?;

    Ok(HttpResponse::Ok().json(settings))
}
```

---

## Debugging Strategy for Rust Backend

### 1. Check Handler Files
```bash
docker exec visionflow_container ls /app/src/handlers/
```

Expected files:
- `config.rs` (working)
- `settings.rs` (crashing)
- `ontology.rs` (crashing)
- `graph.rs` (timeout)

### 2. Search for Panic Sources
```bash
docker exec visionflow_container grep -r "unwrap()" /app/src/handlers/
docker exec visionflow_container grep -r "expect(" /app/src/handlers/
docker exec visionflow_container grep -r "\[0\]" /app/src/handlers/
```

### 3. Check Logs
```bash
docker exec visionflow_container cat /app/logs/*.log | grep -i panic
docker exec visionflow_container cat /app/logs/*.log | grep -i error
```

### 4. Enable Rust Backtrace
```bash
docker exec visionflow_container env RUST_BACKTRACE=1 /app/target/debug/webxr
```

---

## Performance Issue (Timeout)

### /api/graph/data Hypothesis

**Rust-specific causes**:
1. Synchronous blocking query in async handler
2. Missing `.await` on async operation
3. Deadlock in async runtime
4. Database locked by another thread

**Example Bug**:
```rust
#[get("/api/graph/data")]
async fn get_graph_data() -> Result<HttpResponse, Error> {
    // BAD - blocks entire async runtime!
    std::thread::sleep(Duration::from_secs(10)); // ❌

    // Or: database query with no timeout
    let data = db.query_all_graph_data(); // ❌ Slow query, no limit

    Ok(HttpResponse::Ok().json(data))
}
```

---

## Next Steps for Debugging Agent

### Priority 1: Find Crash Sources
```bash
# List handler files
docker exec visionflow_container ls -la /app/src/handlers/

# Read settings handler
docker exec visionflow_container cat /app/src/handlers/settings.rs

# Read ontology handler
docker exec visionflow_container cat /app/src/handlers/ontology.rs

# Compare with working config handler
docker exec visionflow_container cat /app/src/handlers/config.rs
```

### Priority 2: Check Database Schema
```bash
# Install sqlite3 if needed
docker exec visionflow_container apk add sqlite || \
docker exec visionflow_container apt-get install -y sqlite3

# Check tables
docker exec visionflow_container sqlite3 /app/data/settings.db ".tables"
docker exec visionflow_container sqlite3 /app/data/ontology.db ".schema classes"
```

### Priority 3: Enable Debug Logging
```bash
# Set Rust log level
docker exec visionflow_container env RUST_LOG=debug /app/target/debug/webxr

# Or check existing logs
docker exec visionflow_container tail -f /app/logs/*.log
```

---

## Summary

### Corrected Architecture
- **Backend**: Rust Actix-Web (NOT Node.js Vite)
- **Frontend**: Vite dev server (React + TypeScript)
- **Database**: SQLite3 with Rust bindings
- **Crash Pattern**: Rust panics → TCP close (exit 52)
- **Timeout Pattern**: Blocking operation in async handler

### Critical Files to Examine
1. `/app/src/handlers/settings.rs` - Crashing endpoint
2. `/app/src/handlers/ontology.rs` - Crashing endpoints (classes, properties)
3. `/app/src/handlers/graph.rs` - Timeout endpoint
4. `/app/src/handlers/config.rs` - Working endpoint (reference)

### Expected Fixes
1. Replace `.unwrap()` with `.map_err()` or `?`
2. Add error handling for database operations
3. Add timeouts for slow queries
4. Use proper async/await patterns
5. Add database connection pooling

---

**Discovery Status**: ✅ COMPLETE
**Backend Type**: 🦀 Rust Actix-Web
**Next Agent**: Debugging Agent (Rust expertise required)
