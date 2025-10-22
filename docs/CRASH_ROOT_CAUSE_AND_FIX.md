# WebXR Backend Crash - Root Cause Analysis and Fix

**Date**: 2025-10-22
**Severity**: CRITICAL
**Impact**: Backend crashes every ~30 seconds with SIGSEGV

## Executive Summary

The Rust backend crashes during HTTP server startup due to invalid network configuration defaults (`bind_address=""`, `port=0`), causing the server to fail binding and triggering a cascade shutdown with SIGSEGV.

## Investigation Timeline

### Phase 1: Initial Symptoms
- Backend crashes every 30-35 seconds with SIGSEGV
- Supervisor automatically restarts the process
- Log showed infinite "Skipping physics simulation - waiting for GPU initialization" warnings

### Phase 2: GPU Investigation
**Initial Hypothesis**: GPU initialization failure causing crash
**Result**: ❌ FALSE - GPU initialization completes successfully!

Evidence from logs (`rust-error.log:97961,111530`):
```
[2025-10-22T15:23:46Z INFO] GPUInitialized message sent successfully
[2025-10-22T15:23:46Z INFO] GPU has been successfully initialized
[2025-10-22T15:23:46Z INFO] Physics simulation is now ready:
[2025-10-22T15:23:46Z INFO]   - GPU initialized: true
[2025-10-22T15:23:46Z INFO]   - Physics enabled: true
```

### Phase 3: Crash Location Discovery
**True Root Cause**: HTTP server bind failure

Evidence from logs (`rust-error.log:117236-117248`):
```
[2025-10-22T15:23:47Z INFO] Starting HTTP server on :0  ← INVALID ADDRESS!
[2025-10-22T15:23:47Z INFO] ClientCoordinatorActor stopped
[2025-10-22T15:23:47Z INFO] MetadataActor stopped
[2025-10-22T15:23:47Z INFO] GPU Manager Actor stopped
[2025-10-22T15:23:47Z INFO] GraphServiceActor stopped
```

The server tries to bind to `:0` (invalid), fails, and all actors shut down in cascade.

## Root Cause

### Location
- **File**: `src/config/mod.rs:1200-1218`
- **Struct**: `NetworkSettings`

### The Problem

```rust
#[derive(Debug, Serialize, Deserialize, Clone, Default, Type, Validate)]
#[serde(rename_all = "camelCase")]
pub struct NetworkSettings {
    pub bind_address: String,  // ❌ Defaults to "" (empty string)
    pub port: u16,              // ❌ Defaults to 0
    // ... other fields
}
```

**Issue**: Using `#[derive(Default)]` causes Rust to generate:
- `bind_address: String` → `""` (empty string)
- `port: u16` → `0` (invalid port)

### Crash Sequence

```
1. Settings loaded with Default::default() values
   └─ bind_address = ""
   └─ port = 0

2. main.rs:615-618 formats bind address
   └─ format!("{}:{}", "", 0) = ":0"

3. main.rs:686 attempts to bind
   └─ HttpServer.bind(&":0")?  ← FAILS

4. Error propagated by ? operator
   └─ Main function panics/errors

5. Tokio runtime begins shutdown
   └─ All actors stop gracefully

6. Process exits with SIGSEGV
   └─ Supervisor detects crash

7. Supervisor restarts backend after 2s
   └─ Cycle repeats (~30s total)
```

## Solutions

### **Solution 1: Custom Default Implementation (RECOMMENDED)**

Remove `Default` derive and implement proper defaults:

```rust
// src/config/mod.rs:1200
#[derive(Debug, Serialize, Deserialize, Clone, Type, Validate)]
#[serde(rename_all = "camelCase")]
pub struct NetworkSettings {
    #[serde(alias = "bind_address")]
    pub bind_address: String,
    #[serde(alias = "port")]
    pub port: u16,
    // ... other fields
}

impl Default for NetworkSettings {
    fn default() -> Self {
        Self {
            bind_address: "0.0.0.0".to_string(),  // ✅ Valid bind address
            port: 8080,                            // ✅ Valid default port
            domain: String::new(),
            enable_http2: false,
            enable_rate_limiting: false,
            enable_tls: false,
            max_request_size: 10485760, // 10MB
            min_tls_version: "1.2".to_string(),
            rate_limit_requests: 100,
            rate_limit_window: 60,
            tunnel_id: String::new(),
            api_client_timeout: 30,
            enable_metrics: true,
            max_concurrent_requests: 1000,
        }
    }
}
```

**Pros**:
- ✅ Prevents invalid defaults at compile time
- ✅ Self-documenting (shows intended defaults)
- ✅ Type-safe

**Cons**:
- ❌ Must maintain Default impl when adding fields

---

### **Solution 2: Database Loading Validation**

Ensure settings are loaded from database before HTTP server starts:

```rust
// src/main.rs (around line 610)
let bind_address = {
    let settings_read = settings.read().await;

    // Validate settings before use
    let addr = &settings_read.system.network.bind_address;
    let port = settings_read.system.network.port;

    // Fallback to defaults if invalid
    let final_addr = if addr.is_empty() || port == 0 {
        warn!("Invalid network settings detected (addr='{}', port={}), using defaults", addr, port);
        "0.0.0.0:8080".to_string()
    } else {
        format!("{}:{}", addr, port)
    };

    final_addr
};

info!("Starting HTTP server on {}", bind_address);
```

**Pros**:
- ✅ Runtime validation and fallback
- ✅ Logs warning when defaults are used
- ✅ Minimal code changes

**Cons**:
- ❌ Hides underlying configuration issue
- ❌ Runtime overhead

---

### **Solution 3: Environment Variable Fallback**

Use environment variables as primary source:

```rust
// src/main.rs (around line 610)
use std::env;

let bind_address = {
    // Try environment first
    let bind_from_env = env::var("BIND_ADDRESS").unwrap_or_else(|_| "0.0.0.0".to_string());
    let port_from_env = env::var("PORT")
        .ok()
        .and_then(|s| s.parse::<u16>().ok())
        .unwrap_or(8080);

    format!("{}:{}", bind_from_env, port_from_env)
};
```

**Pros**:
- ✅ Standard 12-factor app pattern
- ✅ Easy container configuration
- ✅ No code changes to config structs

**Cons**:
- ❌ Bypasses settings system
- ❌ Less discoverable

---

## Recommended Fix

**Use Solution 1 (Custom Default Implementation)** because:
1. Prevents the issue at compile time
2. Makes intended defaults explicit
3. Type-safe and self-documenting
4. Aligns with Rust best practices

**Backup**: Combine with Solution 2 (validation) for defense-in-depth.

## Testing Steps

After implementing the fix:

1. **Verify build**:
   ```bash
   cargo build
   ```

2. **Check defaults**:
   ```bash
   cargo test network_settings_defaults
   ```

3. **Monitor logs**:
   ```bash
   docker exec visionflow_container tail -f /app/logs/rust.log | grep "Starting HTTP server"
   ```
   Should show: `Starting HTTP server on 0.0.0.0:8080`

4. **Verify stability**:
   ```bash
   # Check process doesn't crash after 30s
   docker exec visionflow_container ps aux | grep webxr
   sleep 35
   docker exec visionflow_container ps aux | grep webxr
   ```

5. **Test HTTP connectivity**:
   ```bash
   curl -v http://localhost:8080/api/health
   ```

## Prevention

Add to CI/CD pipeline:

```rust
// tests/config_validation.rs
#[test]
fn test_network_settings_defaults_are_valid() {
    let settings = NetworkSettings::default();

    assert!(!settings.bind_address.is_empty(), "bind_address must not be empty");
    assert_ne!(settings.port, 0, "port must not be 0");
    assert!(settings.port >= 1024, "port should be >= 1024 for non-privileged");
}
```

## Related Files

- `src/config/mod.rs:1200-1230` - NetworkSettings struct
- `src/main.rs:610-620` - Bind address formatting
- `src/main.rs:686` - HTTP server binding
- `logs/rust-error.log` - Crash evidence
- `logs/supervisord.log` - Restart evidence

## Impact Assessment

**Before Fix**:
- ❌ Backend crashes every ~30 seconds
- ❌ No HTTP API available
- ❌ GPU initialization wasted (succeeds but then crashes)
- ❌ Continuous supervisor restart cycle

**After Fix**:
- ✅ Stable HTTP server on configured port
- ✅ GPU-accelerated physics simulation running
- ✅ WebSocket connections stable
- ✅ API endpoints accessible
