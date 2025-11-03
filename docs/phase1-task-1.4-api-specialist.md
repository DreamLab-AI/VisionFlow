# API Specialist Agent - Task 1.4: HTTP Response Standardization

## MISSION BRIEF
Fix 370 non-standard HTTP response constructions to use HandlerResponse trait consistently.

## OBJECTIVE
Create `/home/devuser/workspace/project/src/utils/response_macros.rs` and enforce trait usage.

## CONTEXT
- HandlerResponse trait exists in `src/utils/handler_commons.rs`
- Only ~300 of 673 responses use the trait
- 370 direct HttpResponse constructions bypass standardization
- Affected files: analytics handlers (80+), ontology handlers (60+), settings (45+)

## IMPLEMENTATION

### Step 1: Create Response Macros (2 hours)
File: `/home/devuser/workspace/project/src/utils/response_macros.rs`

```rust
/// Macro for success JSON response
#[macro_export]
macro_rules! ok_json {
    ($data:expr) => {
        <_>::success($data)
    };
}

/// Macro for error JSON response
#[macro_export]
macro_rules! error_json {
    ($msg:expr) => {
        <()>::internal_error($msg.to_string())
    };
}

/// Macro for bad request response
#[macro_export]
macro_rules! bad_request {
    ($msg:expr) => {
        <()>::bad_request($msg.to_string())
    };
}

/// Macro for not found response
#[macro_export]
macro_rules! not_found {
    ($msg:expr) => {
        <()>::not_found($msg.to_string())
    };
}

/// Macro for success with custom message
#[macro_export]
macro_rules! success_msg {
    ($data:expr, $msg:expr) => {
        <_>::success_with_message($data, $msg.to_string())
    };
}
```

### Step 2: Refactor Handlers (4 hours)
Search and replace across handler files:

```bash
# Find all non-trait responses
grep -r "HttpResponse::" src/handlers/ --include="*.rs" | grep -v "use actix_web"

# Replace patterns:
HttpResponse::Ok().json(...) → ok_json!(...)
HttpResponse::BadRequest().json(...) → bad_request!("message")
HttpResponse::InternalServerError().json(...) → error_json!("message")
HttpResponse::NotFound().json(...) → not_found!("message")
```

Priority files:
1. `src/handlers/api_handler/analytics/*.rs` (80+ occurrences)
2. `src/handlers/api_handler/ontology/*.rs` (60+ occurrences)
3. `src/handlers/settings_handler.rs` (45 occurrences)
4. `src/handlers/graph_state_handler.rs` (30 occurrences)

## ACCEPTANCE CRITERIA
- [ ] Zero direct `HttpResponse::` in handlers (except imports)
- [ ] All responses use HandlerResponse trait
- [ ] Consistent format across endpoints
- [ ] Tests pass: `cargo test --lib handlers`

## TESTING
```bash
# Verify no direct HttpResponse
grep -r "HttpResponse::" src/handlers/ --include="*.rs" | grep -v "use actix_web" | wc -l
# Target: 0

cargo test --lib handlers
cargo test --workspace
```

## MEMORY KEYS
- Check for conflicts: `hive/phase1/conflicts`
- Publish macros to: `hive/phase1/response-macro-api`
- Report completion to: `hive/phase1/completion-status`
