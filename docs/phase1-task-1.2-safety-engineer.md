# Safety Engineer Agent - Task 1.2: Result/Error Helper Utilities

## MISSION BRIEF
You are the Safety Engineer, responsible for eliminating 432 unsafe .unwrap() calls and standardizing 1,544 error handling patterns.

## OBJECTIVE
Create `/home/devuser/workspace/project/src/utils/result_helpers.rs` and replace unsafe error handling across the codebase.

## CONTEXT FROM AUDIT
- 1,544 Result transformation patterns (.map_err, .ok_or, .unwrap_or)
- 432 .unwrap() calls (UNSAFE - can panic in production)
- 180+ identical error conversions with string formatting
- 51 JSON deserialization errors with duplicate handling
- 103 JSON serialization errors with duplicate handling

## HIGH-RISK LOCATIONS (PRIORITY ORDER)
1. `src/handlers/*.rs` - 150+ .unwrap() calls
2. `src/services/*.rs` - 120+ .unwrap() calls
3. `src/actors/*.rs` - 80+ .unwrap() calls
4. `src/adapters/*.rs` - 82+ .unwrap() calls

## IMPLEMENTATION STEPS

### Step 1: Create Result Helpers Module (3 hours)
Create `/home/devuser/workspace/project/src/utils/result_helpers.rs`

Implement:
- `safe_unwrap<T>(option: Option<T>, default: T, context: &str) -> T` - Logs warning instead of panic
- `map_err_context<T, E, F>(result, context_fn) -> Result<T, String>` - Adds context to errors
- `to_vf_error<T, E>(result, context) -> VisionFlowResult<T>` - Converts to VisionFlowError
- Macros:
  - `try_with_context!(result, "context")` - Error handling with context
  - `unwrap_or_default!(option)` - Safe unwrap with Default trait
  - `unwrap_or_log!(option, "context")` - Unwrap with logging

Reference: `/home/devuser/workspace/project/docs/UTILITY_FUNCTION_DUPLICATION.md` lines 150-250

### Step 2: Replace Unsafe .unwrap() in Handlers (3 hours)
Priority: HTTP handlers (highest risk - 150 calls)

Search pattern:
```bash
grep -rn "\.unwrap()" src/handlers/ --include="*.rs" > unwrap_audit_handlers.txt
```

Replace with safe alternatives:
- `.unwrap()` → `safe_unwrap(value, default, "context")`
- `.expect("msg")` → `try_with_context!(value, "msg")?`

### Step 3: Replace in Services, Actors, Adapters (2 hours)
Same pattern for services (120 calls), actors (80 calls), adapters (82 calls)

## ACCEPTANCE CRITERIA
- [ ] All unsafe .unwrap() in handlers replaced (0 remaining in `src/handlers/`)
- [ ] All unsafe .unwrap() in services replaced (0 remaining in `src/services/`)
- [ ] All error messages include context (no bare `.to_string()`)
- [ ] No panics in production code paths
- [ ] Code reduction: Minimum 500 lines eliminated
- [ ] All tests pass: `cargo test --workspace`

## TESTING COMMANDS
```bash
# Verify no .unwrap() in production code
grep -r "\.unwrap()" src/ --include="*.rs" | grep -v "test" | grep -v "examples" | wc -l
# Target: 0

# Test error handling
cargo test --lib utils::result_helpers
cargo test --workspace
```

## COORDINATION PROTOCOL
BEFORE:
```bash
npx claude-flow@alpha hooks pre-task --description "Error Helper Utilities Implementation"
```

AFTER:
```bash
npx claude-flow@alpha hooks post-task --task-id "task-1.2-error-helpers"
npx claude-flow@alpha hooks notify --message "Error helpers complete: 432 unsafe calls eliminated"
```

Report completion to memory key: `hive/phase1/completion-status`
Report API signature to memory key: `hive/phase1/error-helper-api`
