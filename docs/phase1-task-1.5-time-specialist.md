# Infrastructure Specialist Agent - Task 1.5: Time Utilities Module

## MISSION BRIEF
Centralize 305 scattered Utc::now() calls with consistent timestamp formatting.

## OBJECTIVE
Create `/home/devuser/workspace/project/src/utils/time.rs` for all time operations.

## CONTEXT
- 305 Utc::now() calls scattered across codebase
- 50+ duplicate DateTime formatting patterns
- 40+ duplicate Duration calculations
- No centralized time management

## IMPLEMENTATION

### Step 1: Create Time Utilities (2 hours)
File: `/home/devuser/workspace/project/src/utils/time.rs`

```rust
use chrono::{DateTime, Duration, Utc};
use crate::errors::VisionFlowResult;

/// Get current UTC timestamp
pub fn now() -> DateTime<Utc> {
    Utc::now()
}

/// Format timestamp for logging (YYYY-MM-DD HH:MM:SS.mmm)
pub fn format_log_time(dt: &DateTime<Utc>) -> String {
    dt.format("%Y-%m-%d %H:%M:%S%.3f").to_string()
}

/// Format timestamp for API responses (RFC3339)
pub fn format_api_time(dt: &DateTime<Utc>) -> String {
    dt.to_rfc3339()
}

/// Parse RFC3339 timestamp
pub fn parse_api_time(s: &str) -> VisionFlowResult<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(s)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|e| VisionFlowError::from(
            format!("Invalid timestamp: {}", e)
        ))
}

/// Calculate duration since timestamp
pub fn duration_since(start: &DateTime<Utc>) -> Duration {
    Utc::now().signed_duration_since(*start)
}

/// Format duration in human-readable form
pub fn format_duration(duration: &Duration) -> String {
    if duration.num_seconds() < 60 {
        format!("{}s", duration.num_seconds())
    } else if duration.num_minutes() < 60 {
        format!("{}m {}s", duration.num_minutes(), duration.num_seconds() % 60)
    } else {
        format!("{}h {}m", duration.num_hours(), duration.num_minutes() % 60)
    }
}

/// Get timestamp in milliseconds (Unix epoch)
pub fn timestamp_millis() -> i64 {
    Utc::now().timestamp_millis()
}

/// Get current timestamp for StandardResponse
pub fn response_timestamp() -> DateTime<Utc> {
    now()
}
```

### Step 2: Replace Utc::now() Calls (2 hours)
Search and replace:

```bash
# Find all Utc::now() calls
grep -rn "Utc::now()" src/ --include="*.rs"

# Replace patterns:
Utc::now() → time::now()
Utc::now().format("%Y-%m-%d %H:%M:%S") → time::format_log_time(&time::now())
Utc::now().to_rfc3339() → time::format_api_time(&time::now())
Utc::now().timestamp_millis() → time::timestamp_millis()
```

Remove direct chrono imports where appropriate (keep only in time.rs).

## ACCEPTANCE CRITERIA
- [ ] All `Utc::now()` replaced with `time::now()` (except time.rs)
- [ ] Consistent timestamp format across system
- [ ] No direct chrono imports outside utils (except time.rs)
- [ ] Tests pass: `cargo test --lib utils::time`
- [ ] Tests pass: `cargo test --workspace`

## TESTING
```bash
# Verify centralized usage
grep -r "Utc::now()" src/ --include="*.rs" | grep -v "utils/time.rs" | wc -l
# Target: <10 (some edge cases allowed)

cargo test --lib utils::time
cargo test --workspace
```

## MEMORY KEYS
- Publish API to: `hive/phase1/time-util-api`
- Report completion to: `hive/phase1/completion-status`
