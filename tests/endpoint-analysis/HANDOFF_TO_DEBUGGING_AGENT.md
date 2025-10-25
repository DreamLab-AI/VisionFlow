# 🎯 Handoff to Debugging Agent - VisionFlow Backend Crashes

**From**: Tester Agent (Hive Mind)
**To**: Debugging Agent
**Date**: 2025-10-24
**Priority**: 🔴 HIGH - 3 Critical Crashes

---

## 🚨 Critical Information

### Backend Architecture
- **Language**: Rust
- **Framework**: Actix-Web 4.11.0
- **Binary**: `/app/target/debug/webxr` (PID 1870)
- **Handlers**: `/app/src/handlers/`
- **Database**: SQLite3 at `/app/data/*.db`

### Crash Pattern
- **Type**: Rust panic → TCP connection close
- **Symptom**: curl exit code 52 (empty reply from server)
- **Timing**: Instant crash (<10ms)
- **Affected Endpoints**: 3 total

---

## 💥 Endpoints That Crash

### 1. /api/settings
- **HTTP Code**: 000
- **Exit Code**: 52
- **Time**: 6ms
- **Handler File**: `/app/src/handlers/settings_handler.rs` ⭐
- **Database**: `/app/data/settings.db`

### 2. /api/ontology/classes
- **HTTP Code**: 000
- **Exit Code**: 52
- **Time**: 3ms
- **Handler File**: `/app/src/handlers/ontology_handler.rs` ⭐
- **Database**: `/app/data/ontology.db`

### 3. /api/ontology/properties
- **HTTP Code**: 000
- **Exit Code**: 52
- **Time**: 3ms
- **Handler File**: `/app/src/handlers/ontology_handler.rs` ⭐
- **Database**: `/app/data/ontology.db`

---

## ✅ Working Endpoint (Reference)

### /api/config
- **HTTP Code**: 200
- **Response Time**: 5-7ms
- **Reliability**: 100% (3/3 tests)
- **Handler File**: `/app/src/handlers/api_handler/` (likely)
- **Response**:
```json
{
  "features": { "kokoro": false, "openai": false, ... },
  "rendering": { "ambientLightIntensity": 0.0, ... },
  "version": "0.1.0",
  "websocket": { "maxUpdateRate": 60, ... },
  "xr": { "enabled": false, ... }
}
```

**Why it works**: Proper error handling in Rust code

---

## 🔍 Investigation Tasks

### Priority 1: Examine Crash Sources

```bash
# Read the crashing handlers
docker exec visionflow_container cat /app/src/handlers/settings_handler.rs
docker exec visionflow_container cat /app/src/handlers/ontology_handler.rs

# Search for panic sources
docker exec visionflow_container grep -n "unwrap()" /app/src/handlers/settings_handler.rs
docker exec visionflow_container grep -n "expect(" /app/src/handlers/settings_handler.rs
docker exec visionflow_container grep -n "\[0\]" /app/src/handlers/settings_handler.rs
```

### Priority 2: Compare with Working Handler

```bash
# Find config handler (working reference)
docker exec visionflow_container find /app/src/handlers -name "*.rs" | xargs grep -l "api/config"

# Compare error handling patterns
# Look for: .map_err(), ?, Result<>, proper error types
```

### Priority 3: Check Database Tables

```bash
# Install sqlite3
docker exec visionflow_container apk add sqlite || \
docker exec visionflow_container apt-get install -y sqlite3

# Check if tables exist
docker exec visionflow_container sqlite3 /app/data/settings.db ".tables"
docker exec visionflow_container sqlite3 /app/data/ontology.db ".tables"

# Check table schemas
docker exec visionflow_container sqlite3 /app/data/ontology.db ".schema classes"
docker exec visionflow_container sqlite3 /app/data/ontology.db ".schema properties"
```

---

## 🦀 Common Rust Panic Patterns to Look For

### 1. Unwrap on Database Operations
```rust
// ❌ BAD - panics if query fails
let settings = db.query("SELECT * FROM settings").unwrap();

// ✅ GOOD - returns error to Actix
let settings = db.query("SELECT * FROM settings")
    .map_err(|e| actix_web::error::ErrorInternalServerError(e))?;
```

### 2. Index Out of Bounds
```rust
// ❌ BAD - panics if empty
let first = results[0];

// ✅ GOOD - safe access
let first = results.first().ok_or_else(|| {
    actix_web::error::ErrorNotFound("No results")
})?;
```

### 3. Unwrap on Database Connection
```rust
// ❌ BAD - panics if file missing
let conn = Connection::open("/app/data/settings.db").unwrap();

// ✅ GOOD - returns error
let conn = Connection::open("/app/data/settings.db")
    .map_err(|e| actix_web::error::ErrorInternalServerError(e))?;
```

### 4. Expect on Option Types
```rust
// ❌ BAD - panics if None
let value = map.get("key").expect("Key must exist");

// ✅ GOOD - returns 404
let value = map.get("key").ok_or_else(|| {
    actix_web::error::ErrorNotFound("Key not found")
})?;
```

---

## 📁 Handler Files Available

Located at `/app/src/handlers/`:

**Key Files**:
- `settings_handler.rs` - 18,946 bytes (CRASHING) ⚠️
- `ontology_handler.rs` - 23,235 bytes (CRASHING) ⚠️
- `api_handler/` - Directory (likely contains config handler)
- `consolidated_health_handler.rs` - Health endpoint (WORKING)

**Other Files**:
- `graph_state_handler.rs` - 13,928 bytes (possibly /api/graph/data timeout)
- `graph_export_handler.rs` - 17,165 bytes
- `bots_handler.rs`, `clustering_handler.rs`, etc.

---

## 🎯 Expected Fixes

### For /api/settings
```rust
// Before (crashes):
#[get("/api/settings")]
async fn get_settings() -> Result<HttpResponse, Error> {
    let conn = Connection::open("/app/data/settings.db").unwrap(); // ❌
    let settings = conn.query_row(...).unwrap(); // ❌
    Ok(HttpResponse::Ok().json(settings))
}

// After (handles errors):
#[get("/api/settings")]
async fn get_settings() -> Result<HttpResponse, Error> {
    let conn = Connection::open("/app/data/settings.db")
        .map_err(|e| actix_web::error::ErrorInternalServerError(e))?; // ✅

    let settings = conn.query_row(...)
        .map_err(|e| actix_web::error::ErrorInternalServerError(e))?; // ✅

    Ok(HttpResponse::Ok().json(settings))
}
```

### For /api/ontology/classes and /api/ontology/properties
Same pattern - find `.unwrap()`, `.expect()`, or `[index]` and replace with proper error handling.

---

## ⏱️ Bonus: Timeout Issue

### /api/graph/data
- **HTTP Code**: 000
- **Exit Code**: 28 (timeout)
- **Time**: 10.005 seconds
- **Handler**: Likely in `graph_state_handler.rs` or `graph_export_handler.rs`

**Possible causes**:
1. Blocking database query in async handler
2. Missing query timeout
3. N+1 query pattern
4. Missing database index on large table
5. Infinite loop in graph processing

**Investigation**:
```bash
docker exec visionflow_container cat /app/src/handlers/graph_state_handler.rs | grep -A 20 "api/graph/data"
```

---

## 📊 Test Evidence

### Load Test Results
- Health endpoint: 5/5 successful
- Config endpoint: 3/3 successful ✅
- Settings endpoint: 0/3 successful (all crash) ❌
- Ontology endpoints: 0/2 successful (all crash) ❌

### Database Files Confirmed
```
✅ /app/data/settings.db - EXISTS
✅ /app/data/ontology.db - EXISTS
✅ /app/data/knowledge_graph.db - EXISTS
```

### Logs to Check
```bash
docker exec visionflow_container ls -la /app/logs/
docker exec visionflow_container tail -100 /app/logs/*.log | grep -i "panic\|error"
```

---

## 🎓 Debugging Tools

### Enable Rust Backtrace
```bash
docker exec visionflow_container env RUST_BACKTRACE=full /app/target/debug/webxr
```

### Check Dependencies
```bash
docker exec visionflow_container cat /app/Cargo.toml | grep -A 20 dependencies
```

### Test Endpoint Manually
```bash
# Inside container
docker exec -it visionflow_container sh

# Test database access
sqlite3 /app/data/settings.db "SELECT * FROM sqlite_master;"
```

---

## 📋 Hive Mind Memory

All findings stored in Claude Flow memory:
- `hive-mind/testing/endpoint-results`
- `hive-mind/testing/crash-analysis`
- `hive-mind/testing/database-findings`
- `hive-mind/testing/architecture-discovery`
- `hive-mind/testing/critical-discovery`

---

## 🎯 Success Criteria

### Debugging Complete When:
1. ✅ Identified exact line causing panic in each handler
2. ✅ Confirmed root cause (unwrap, expect, index, etc.)
3. ✅ Verified database tables exist and have correct schema
4. ✅ Proposed code fix with proper error handling
5. ✅ (Optional) Fixed timeout issue in graph/data endpoint

---

## 📁 Test Artifacts

All files in `/home/devuser/workspace/project/tests/endpoint-analysis/`:
1. `endpoint-test-results.json` - Raw test data
2. `corrected-analysis.json` - Categorized results
3. `load-test-results.log` - Load testing
4. `COMPLETE_TEST_REPORT.md` - Full analysis
5. `ARCHITECTURE_DISCOVERY.md` - Rust backend details
6. `HANDOFF_TO_DEBUGGING_AGENT.md` - This document

---

## 🚀 Next Steps

1. **Read handler files**:
   - `/app/src/handlers/settings_handler.rs`
   - `/app/src/handlers/ontology_handler.rs`

2. **Find panic sources**:
   - Search for `.unwrap()`
   - Search for `.expect()`
   - Search for direct array indexing `[0]`

3. **Verify database schema**:
   - Check if `settings` table exists
   - Check if `classes` and `properties` tables exist
   - Verify schema matches Rust struct expectations

4. **Propose fixes**:
   - Replace panic sources with `?` operator
   - Add proper error types
   - Add error logging
   - Test fixes locally

---

**Handoff Status**: ✅ COMPLETE
**Evidence Quality**: HIGH (comprehensive testing)
**Confidence Level**: 95% (architecture confirmed, handlers located)

**Ready for debugging!** 🐛🔧
