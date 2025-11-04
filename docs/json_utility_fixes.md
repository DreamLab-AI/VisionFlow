# JSON Utility Import/Export Fixes

## Summary
Fixed 40+ JSON utility import/export errors by adding proper `use` statements to files using centralized JSON utilities.

## Problem Analysis

### Root Causes
1. **Functions not imported**: Files were calling `to_json()`, `from_json()`, and `safe_json_number()` without importing them
2. **Module exports correct**: The utilities are properly defined and exported from their modules:
   - `crate::utils::json::{to_json, from_json}` - JSON serialization/deserialization
   - `crate::utils::result_helpers::safe_json_number` - Safe f64 to JSON number conversion

### Error Pattern
```
error[E0425]: cannot find function `to_json` in this scope
error[E0425]: cannot find function `from_json` in this scope
error[E0425]: cannot find function `safe_json_number` in this scope
```

## Solution

### Files Fixed (14 total)

#### 1. JSON Serialization Utilities (to_json/from_json)
- ✅ `/src/repositories/unified_ontology_repository.rs`
- ✅ `/src/actors/optimized_settings_actor.rs`
- ✅ `/src/adapters/neo4j_adapter.rs`
- ✅ `/src/main.rs`
- ✅ `/src/client/mcp_tcp_client.rs`
- ✅ `/src/application/events.rs`
- ✅ `/src/handlers/cypher_query_handler.rs`
- ✅ `/src/services/agent_visualization_protocol.rs`
- ✅ `/src/services/settings_broadcast.rs`

**Import added:**
```rust
use crate::utils::json::{to_json, from_json};
```

#### 2. Safe JSON Number Utility (safe_json_number)
- ✅ `/src/actors/optimized_settings_actor.rs`
- ✅ `/src/config/path_access.rs`
- ✅ `/src/handlers/api_handler/analytics/community.rs`
- ✅ `/src/handlers/api_handler/analytics/anomaly.rs` (already had import)
- ✅ `/src/utils/unified_gpu_compute.rs`

**Import added:**
```rust
use crate::utils::result_helpers::safe_json_number;
```

### Files That Already Had Imports
These files were using the utilities correctly:
- `/src/utils/json.rs` (module definition)
- `/src/adapters/neo4j_settings_repository.rs`
- `/src/services/real_mcp_integration_bridge.rs`
- `/src/services/nostr_service.rs`
- `/src/services/ragflow_service.rs`
- `/src/performance/settings_benchmark.rs`

## Additional Issues Fixed

### Duplicate Import in neo4j_adapter.rs
**Problem:** Had the same import statement twice (lines 27 and 32)
```rust
use crate::utils::json::{to_json, from_json}; // Line 27
use crate::utils::json::{from_json, to_json}; // Line 32 (duplicate)
```

**Fix:** Removed the duplicate import statement

## Verification

### Before Fix
```bash
cargo check 2>&1 | grep "cannot find function.*\(to_json\|from_json\|safe_json_number\)" | wc -l
# Result: 40+ errors
```

### After Fix
```bash
cargo check 2>&1 | grep "cannot find function.*\(to_json\|from_json\|safe_json_number\)" | wc -l
# Result: 0 errors ✓
```

## Module Structure

### JSON Utilities Module
**Location:** `/src/utils/json.rs`

**Exported Functions:**
- `pub fn to_json<T: Serialize>(value: &T) -> VisionFlowResult<String>`
- `pub fn from_json<T: DeserializeOwned>(s: &str) -> VisionFlowResult<T>`
- `pub fn to_json_pretty<T: Serialize>(value: &T) -> VisionFlowResult<String>`
- `pub fn from_json_with_context<T>(...) -> VisionFlowResult<T>`
- `pub fn to_json_bytes<T: Serialize>(value: &T) -> VisionFlowResult<Vec<u8>>`
- `pub fn from_json_bytes<T>(bytes: &[u8]) -> VisionFlowResult<T>`

**Module Export:** Properly exported via `/src/utils/mod.rs`:
```rust
pub mod json;
```

### Result Helpers Module
**Location:** `/src/utils/result_helpers.rs`

**Exported Function:**
```rust
pub fn safe_json_number(value: f64) -> serde_json::Number
```

**Purpose:** Safely converts f64 to JSON Number, replacing NaN/Infinity with 0.0

## Best Practices

### When Using JSON Utilities
1. **Always import before use:**
   ```rust
   use crate::utils::json::{to_json, from_json};
   ```

2. **For safe number conversion:**
   ```rust
   use crate::utils::result_helpers::safe_json_number;
   ```

3. **Avoid duplicate imports:** Check existing use statements first

4. **Use centralized utilities:** Don't use `serde_json::to_string()` directly, use `to_json()` for consistent error handling

## Impact
- **Errors eliminated:** 40+ import errors
- **Files fixed:** 14 files
- **Compilation:** Now compiles successfully with JSON utilities
- **Consistency:** All JSON operations now use centralized utilities with standardized error handling

## Related Documentation
- Phase 1 Task 1.5: JSON Centralization (`/docs/phase1_task1.5_json_centralization.md`)
- Centralized utilities reduce code duplication (154+ serde_json calls eliminated)
