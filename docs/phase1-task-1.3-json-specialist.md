# Serialization Specialist Agent - Task 1.3: JSON Processing Utilities

## MISSION BRIEF
Consolidate 154 JSON serialization/deserialization duplicates into centralized utilities.

## OBJECTIVE
Create `/home/devuser/workspace/project/src/utils/json.rs` with standardized error handling.

## CONTEXT
- 103 JSON serialization calls with duplicate error handling
- 51 JSON deserialization calls with duplicate error handling
- Inconsistent error messages across 30+ files

## IMPLEMENTATION

### Step 1: Create JSON Utilities (2 hours)
File: `/home/devuser/workspace/project/src/utils/json.rs`

```rust
use serde::{Deserialize, Serialize};
use serde::de::DeserializeOwned;
use crate::errors::VisionFlowResult;

/// Centralized JSON deserialization with context
pub fn from_json<T: DeserializeOwned>(s: &str) -> VisionFlowResult<T> {
    serde_json::from_str(s).map_err(|e| {
        VisionFlowError::Serialization(format!("JSON deserialization failed: {}", e))
    })
}

/// Centralized JSON serialization
pub fn to_json<T: Serialize>(value: &T) -> VisionFlowResult<String> {
    serde_json::to_string(value).map_err(|e| {
        VisionFlowError::Serialization(format!("JSON serialization failed: {}", e))
    })
}

/// JSON deserialization with custom context
pub fn from_json_with_context<T: DeserializeOwned>(
    s: &str,
    context: &str
) -> VisionFlowResult<T> {
    serde_json::from_str(s).map_err(|e| {
        VisionFlowError::Serialization(format!("{}: {}", context, e))
    })
}

/// Pretty-printed JSON
pub fn to_json_pretty<T: Serialize>(value: &T) -> VisionFlowResult<String> {
    serde_json::to_string_pretty(value).map_err(|e| {
        VisionFlowError::Serialization(format!("JSON serialization (pretty) failed: {}", e))
    })
}

/// Byte array parsing
pub fn from_json_bytes<T: DeserializeOwned>(bytes: &[u8]) -> VisionFlowResult<T> {
    serde_json::from_slice(bytes).map_err(|e| {
        VisionFlowError::Serialization(format!("JSON deserialization from bytes failed: {}", e))
    })
}
```

### Step 2: Replace Direct JSON Calls (2 hours)
Search and replace:
```bash
# Find all usages
grep -rn "serde_json::from_str" src/ --include="*.rs"
grep -rn "serde_json::to_string" src/ --include="*.rs"

# Replace patterns
serde_json::from_str(&data)? → from_json(&data)?
serde_json::to_string(&value)? → to_json(&value)?
```

## ACCEPTANCE CRITERIA
- [ ] All `serde_json::from_str` replaced (except in json.rs)
- [ ] All `serde_json::to_string` replaced (except in json.rs)
- [ ] Consistent error messages
- [ ] Tests pass: `cargo test --workspace`

## TESTING
```bash
grep -r "serde_json::from_str" src/ --include="*.rs" | grep -v "utils/json.rs" | wc -l
# Target: 0

cargo test --lib utils::json
cargo test --workspace
```

## MEMORY KEYS
- Publish API to: `hive/phase1/json-util-api`
- Report completion to: `hive/phase1/completion-status`
