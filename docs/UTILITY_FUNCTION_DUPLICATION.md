# Utility Function Duplication Analysis

**Analysis Date:** 2025-11-03
**Codebase:** VisionFlow Project
**Total Source Files Analyzed:** 200+ Rust files
**Focus:** Utility functions, helpers, and common patterns

---

## Executive Summary

### Key Findings

- **1,544 Result unwrapping patterns** (.map_err, .ok_or, .unwrap_or) indicating significant opportunity for error handling utilities
- **673 HTTP response constructions** with duplicated patterns across handlers
- **432 .unwrap() calls** that could be replaced with safer utility patterns
- **305 timestamp generation calls** (Utc::now()) with no centralized utility
- **103 JSON serialization** and **51 JSON deserialization** calls with repeated error handling
- **2,779 .to_string() conversions** indicating need for string utility helpers
- **1,510 format! macro calls** with duplicated formatting patterns

### Impact Assessment

- **High Duplication Areas:** Error handling, HTTP responses, JSON processing, validation
- **Medium Duplication Areas:** String manipulation, parsing, type conversion
- **Consolidation Potential:** 40-50% reduction in duplicated code through shared utilities
- **Technical Debt:** Estimated 80-120 hours to refactor without utilities, 20-30 hours with proper abstraction

---

## 1. Utility Function Categories

### 1.1 String Utilities (HIGH DUPLICATION)

**Current State:**
- 2,779 `.to_string()` calls scattered across codebase
- 1,510 `format!` macro calls
- Multiple string splitting patterns with repeated `.split().next().unwrap_or("")`
- String sanitization duplicated in multiple locations

**Files with String Utilities:**
```
src/utils/validation/sanitization.rs
  - sanitize_string()
  - sanitize_filename()
  - sanitize_email()
  - sanitize_url()
  - escape_html()
  - remove_script_tags()
  - remove_sql_injection_patterns()
  - remove_path_traversal()
```

**Duplicate Patterns Found:**
- **Split and Extract (20+ occurrences):**
  ```rust
  // Pattern 1: Split with default
  agent_id.split(':').next().unwrap_or("unknown")

  // Pattern 2: Split into parts
  let parts: Vec<&str> = path.split('.').collect();

  // Pattern 3: Email/URL splitting
  sanitized.split('@').collect()
  ```

- **IRI/URL Parsing (15+ occurrences):**
  ```rust
  // Extracting last segment from IRI
  class.iri.split('#').last()
      .or_else(|| class.iri.split('/').last())
  ```

### 1.2 Validation Utilities (WELL ORGANIZED)

**Current State:** ✅ Good consolidation in `src/utils/validation/`

**Files:**
```
src/utils/validation/mod.rs
  - validate_string_length()
  - validate_numeric_range()
  - validate_required()
  - validate_array_size()
  - validate_email()
  - validate_url()
  - validate_hex_color()
  - validate_uuid()

src/utils/validation/sanitization.rs
  - validate_numeric()
  - sanitize_json()
  - All sanitization functions

src/utils/validation/schemas.rs
  - validate_type()
  - validate_min_length()
  - validate_max_length()
  - validate_min_value()
  - validate_max_value()
  - validate_pattern()

src/utils/validation/rate_limit.rs
  - extract_client_id()
  - extract_client_id_from_service_request()
```

**Recommendation:** Validation is already well-organized. Consider extracting common patterns used in 3+ locations.

### 1.3 Error Handling Utilities (CRITICAL NEED)

**Current State:**
- 1,544 Result transformation patterns (.map_err, .ok_or, .unwrap_or)
- 432 .unwrap() calls (unsafe pattern)
- Custom error conversions repeated across files
- ErrorContext trait exists but underutilized

**Centralized Error Module:**
```
src/errors/mod.rs
  - VisionFlowError (comprehensive enum)
  - ErrorContext trait
  - with_context(), with_actor_context(), with_gpu_context()
```

**Duplicate Error Handling Patterns:**

1. **JSON Deserialization (51 occurrences):**
   ```rust
   serde_json::from_str(&event.data).map_err(|e| {
       EventError::Deserialization(format!("Invalid JSON: {}", e))
   })
   ```

2. **JSON Serialization (103 occurrences):**
   ```rust
   event.to_json_string().map_err(|e|
       EventError::Serialization(e.to_string())
   )?;
   ```

3. **Error Conversion (multiple patterns):**
   ```rust
   // Pattern 1: String conversion
   .map_err(|e| e.to_string())?

   // Pattern 2: Format wrapper
   .map_err(|e| format!("Failed to X: {}", e))?

   // Pattern 3: Custom error type
   .map_err(|e| VisionFlowError::from(e))?
   ```

**Missing Utilities:**
- `json_deserialize_or<T>(str, context)` - Centralized JSON error handling
- `json_serialize_or(value, context)` - Centralized JSON serialization
- `error_context!(result, "context")` - Macro for adding context
- `safe_unwrap(option, default, log_msg)` - Logging unwrap alternative

### 1.4 HTTP Response Utilities (MODERATE ORGANIZATION)

**Current State:**
- 673 HTTP response constructions
- HandlerResponse trait in `src/utils/handler_commons.rs`
- Partially standardized but not universally used

**Existing Utilities:**
```
src/utils/handler_commons.rs
  - HandlerResponse trait:
    - success()
    - success_with_message()
    - internal_error()
    - bad_request()
    - not_found()
    - from_error()
    - from_str_error()

  - StandardResponse<T>
  - ErrorResponse
  - SuccessResponse<T>
  - PaginatedResponse<T>
  - HealthCheckResponse
```

**Duplicate Patterns Not Using Trait:**
```rust
// Direct HttpResponse construction (300+ occurrences)
HttpResponse::Ok().json(...)
HttpResponse::BadRequest().json(...)
HttpResponse::InternalServerError().json(...)
HttpResponse::NotFound().json(...)
```

**Recommendation:** Enforce HandlerResponse trait usage across all handlers.

### 1.5 Data Conversion Utilities (HIGH DUPLICATION)

**Current State:**
- 20+ conversion functions scattered across actors, adapters, handlers
- No centralized conversion module
- Type conversion patterns repeated

**Duplicate Conversion Functions:**

1. **GPU Data Conversion (5+ similar functions):**
   ```rust
   convert_gpu_kmeans_result_to_clusters()
   convert_gpu_community_result_to_communities()
   convert_gpu_clusters_to_response()
   convert_gpu_anomaly_result_to_anomalies()
   convert_constraints_to_gpu_format()
   ```

2. **Graph/Node Conversion (4+ functions):**
   ```rust
   convert_agents_to_nodes()
   convert_subsumptions_to_axioms()
   map_graph_to_rdf()
   ```

3. **Case Conversion (2+ implementations):**
   ```rust
   convert_to_camel_case()
   convert_to_snake_case_recursive()
   ```

4. **Parameter/Config Conversion (3+ functions):**
   ```rust
   convert_physics_params()
   convert_params_to_actor()
   convert_constraint_to_actor()
   ```

**Missing Utilities:**
- `src/utils/conversion/` module with:
  - `case_conversion.rs` - snake_case ↔ camelCase ↔ PascalCase
  - `type_conversion.rs` - Common type transformations
  - `gpu_conversion.rs` - GPU-specific conversions
  - `graph_conversion.rs` - Graph/RDF conversions

### 1.6 Parsing Utilities (MODERATE DUPLICATION)

**Current State:**
- 23 parse_ functions across codebase
- OWL/RDF parsing in `src/inference/owl_parser.rs`
- Multiple format-specific parsers

**Parse Functions:**
```
Inference/Ontology Parsing:
  - parse_with_format()
  - parse_owl_xml()
  - parse_rdf_xml()
  - parse_turtle()
  - parse_functional_syntax()
  - parse_owx()
  - parse_ontology()
  - parse_owl_blocks()
  - parse_with_horned_owl()

Configuration Parsing:
  - parse_path()
  - parse_axiom_type()
  - parse_property_type()
  - parse_sql()

MCP/Agent Parsing:
  - parse_agent_list_response()
  - parse_single_agent()
  - parse_performance_data()
  - parse_agent_metadata()
  - parse_neural_data()
  - parse_swarm_topology_response()
  - parse_server_info_response()
  - parse_agent_status()

File Parsing:
  - parse_logseq_file()
```

**Duplicate Patterns:**
1. **Format Detection and Parsing:**
   ```rust
   match format {
       OWLFormat::XML => parse_owl_xml(content),
       OWLFormat::RDF => parse_rdf_xml(content),
       OWLFormat::Turtle => parse_turtle(content),
   }
   ```

2. **JSON Response Parsing (MCP):**
   Similar error handling across 8 parse_ functions in mcp_tcp_client.rs

**Recommendation:** Consolidate format-specific parsers under parser modules.

### 1.7 Extraction Utilities (HIGH DUPLICATION)

**Current State:**
- 38 extract_ functions across codebase
- Many similar patterns for extracting data from structures
- No centralized extraction module

**Extract Functions by Category:**

**Voice/NLP Extraction (6 functions):**
```rust
extract_agent_type()
extract_target()
extract_agent_id()
extract_label()
extract_ai_features()
extract_conceptual_relationships()
extract_named_entities()
```

**Data Structure Extraction (10+ functions):**
```rust
extract_client_messages()
extract_client_id()
extract_client_id_from_service_request()
extract_positions()
extract_ontology_components()
extract_data<T>()
extract_all_data<T>()
```

**Ontology/Graph Extraction (12+ functions):**
```rust
extract_ontology_section()
extract_classes()
extract_properties()
extract_axioms()
extract_class_hierarchy()
extract_links()
extract_metadata_store()
extract_tags()
extract_label_from_iri()
```

**Feature Extraction (8 functions):**
```rust
extract_topics()
extract_temporal_features()
extract_structural_features()
extract_content_features()
extract_references()
extract_commit_date()
extract_gpu_metrics()
```

**Settings Extraction (2+ functions):**
```rust
extract_physics_updates()
extract_failed_field()
```

**Duplicate Pattern:**
```rust
// Repeated pattern: Extract from Option or provide default
connection.source.split(':').next().unwrap_or("")
```

**Recommendation:** Create extraction utility module with common patterns.

### 1.8 Timestamp/Time Utilities (MISSING)

**Current State:**
- 305 `Utc::now()` calls scattered across codebase
- No centralized time utility
- DateTime formatting repeated

**Duplicate Patterns:**
```rust
// Pattern 1: Simple timestamp
timestamp: Utc::now()

// Pattern 2: Format for logging
format!("{}", Utc::now().format("%Y-%m-%d %H:%M:%S"))

// Pattern 3: Timestamp in responses
StandardResponse {
    timestamp: Utc::now(),
    ...
}
```

**Missing Utilities:**
```rust
// Proposed: src/utils/time.rs
pub fn now() -> DateTime<Utc>
pub fn format_timestamp(dt: &DateTime<Utc>) -> String
pub fn parse_timestamp(s: &str) -> Result<DateTime<Utc>>
pub fn timestamp_millis() -> i64
pub fn duration_since(start: &DateTime<Utc>) -> Duration
```

---

## 2. Duplicate Utility Groups

### Group 1: JSON Processing Utilities (154 DUPLICATES)

**Priority:** HIGH
**Impact:** High (used in 30+ files)
**Effort:** Low (2-3 hours)

**Locations:**
- Event handlers: 10+ duplicate patterns
- API handlers: 40+ duplicate patterns
- Services: 30+ duplicate patterns
- Protocols: 20+ duplicate patterns
- Misc: 54+ duplicate patterns

**Current Implementation:**
```rust
// Deserialization (51 occurrences)
serde_json::from_str(&data).map_err(|e| Error::Parse(e.to_string()))

// Serialization (103 occurrences)
serde_json::to_string(&value).map_err(|e| Error::Serialize(e.to_string()))
```

**Proposed Consolidation:**
```rust
// src/utils/json.rs
pub fn from_json<T: DeserializeOwned>(s: &str) -> VisionFlowResult<T> {
    serde_json::from_str(s)
        .map_err(|e| VisionFlowError::Serialization(
            format!("JSON deserialization failed: {}", e)
        ))
}

pub fn to_json<T: Serialize>(value: &T) -> VisionFlowResult<String> {
    serde_json::to_string(value)
        .map_err(|e| VisionFlowError::Serialization(
            format!("JSON serialization failed: {}", e)
        ))
}

pub fn from_json_with_context<T: DeserializeOwned>(
    s: &str,
    context: &str
) -> VisionFlowResult<T> {
    serde_json::from_str(s)
        .map_err(|e| VisionFlowError::Serialization(
            format!("{}: {}", context, e)
        ))
}
```

**Estimated Savings:** 150+ lines of duplicated error handling code

---

### Group 2: HTTP Response Construction (370+ DUPLICATES)

**Priority:** HIGH
**Impact:** High (used in all handlers)
**Effort:** Medium (4-6 hours to refactor handlers)

**Problem:**
HandlerResponse trait exists but only ~300 of 673 HTTP responses use it.

**Non-trait Patterns (370+ occurrences):**
```rust
HttpResponse::Ok().json(serde_json::json!({...}))
HttpResponse::BadRequest().json(serde_json::json!({...}))
HttpResponse::InternalServerError().json(serde_json::json!({...}))
```

**Recommendation:**
1. Create helper macros to enforce trait usage
2. Add linting rule to detect direct HttpResponse construction
3. Refactor existing handlers to use HandlerResponse trait

**Macro Proposal:**
```rust
#[macro_export]
macro_rules! ok_json {
    ($data:expr) => {
        <_>::success($data)
    };
}

#[macro_export]
macro_rules! error_json {
    ($msg:expr) => {
        <()>::internal_error($msg.to_string())
    };
}
```

---

### Group 3: String Splitting/Extraction (40+ DUPLICATES)

**Priority:** MEDIUM
**Impact:** Medium (reduces boilerplate)
**Effort:** Low (2 hours)

**Current Patterns:**
```rust
// Pattern 1: Split with default (20+ occurrences)
agent_id.split(':').next().unwrap_or("unknown")

// Pattern 2: Split email/URL (5+ occurrences)
sanitized.split('@').collect()

// Pattern 3: IRI extraction (10+ occurrences)
class.iri.split('#').last()
    .or_else(|| class.iri.split('/').last())

// Pattern 4: Path splitting (5+ occurrences)
path.split('.').map(|s| s.to_string()).collect()
```

**Proposed Utility:**
```rust
// src/utils/string_helpers.rs
pub fn split_first<'a>(s: &'a str, delimiter: char, default: &'a str) -> &'a str {
    s.split(delimiter).next().unwrap_or(default)
}

pub fn split_last<'a>(s: &'a str, delimiter: char) -> Option<&'a str> {
    s.split(delimiter).last()
}

pub fn extract_iri_fragment(iri: &str) -> Option<&str> {
    iri.split('#').last()
        .or_else(|| iri.split('/').last())
}

pub fn split_to_vec(s: &str, delimiter: char) -> Vec<String> {
    s.split(delimiter).map(|s| s.to_string()).collect()
}
```

---

### Group 4: GPU Data Conversion (8 DUPLICATES)

**Priority:** MEDIUM
**Impact:** Medium (GPU-specific)
**Effort:** Low (3-4 hours)

**Similar Functions:**
```
src/actors/gpu/clustering_actor.rs:
  - convert_gpu_kmeans_result_to_clusters()
  - convert_gpu_community_result_to_communities()

src/actors/gpu/constraint_actor.rs:
  - convert_constraints_to_gpu_format()

src/handlers/api_handler/analytics/:
  - convert_gpu_anomaly_result_to_anomalies()
  - convert_gpu_result_to_communities()
  - convert_gpu_clusters_to_response()
```

**Recommendation:**
Create `src/utils/gpu_conversion.rs` to consolidate GPU-specific conversions.

---

### Group 5: Type/Format Conversion (10+ DUPLICATES)

**Priority:** MEDIUM
**Impact:** Medium
**Effort:** Low (2-3 hours)

**Duplicate Functions:**
```rust
// Case conversion
convert_to_camel_case() - bin/generate_types.rs
convert_to_snake_case_recursive() - handlers/settings_validation_fix.rs

// Parameter conversion
convert_physics_params() - actors/backward_compat.rs
convert_params_to_actor() - adapters/physics_orchestrator_adapter.rs

// Graph conversion
convert_agents_to_nodes() - handlers/bots_handler.rs
```

**Recommendation:**
```rust
// src/utils/conversion/case.rs
pub fn to_snake_case(s: &str) -> String
pub fn to_camel_case(s: &str) -> String
pub fn to_pascal_case(s: &str) -> String
pub fn convert_keys_recursive(value: &mut Value, converter: fn(&str) -> String)
```

---

### Group 6: Validation Helper Duplicates (12+ DUPLICATES)

**Priority:** LOW (Already well-organized)
**Impact:** Low
**Effort:** Low (1-2 hours)

**Findings:**
Validation is already well-consolidated in `src/utils/validation/`.

**Minor improvements:**
1. Extract repeated regex patterns to constants
2. Create validation macros for common cases
3. Add more specialized validators (phone, IP, etc.)

---

## 3. Missing Utility Abstractions

### 3.1 High-Impact Missing Utilities

#### A. Result/Option Helper Utilities
**Occurrences:** 1,544+ Result transformations, 432 .unwrap() calls
**Recommended Module:** `src/utils/result_helpers.rs`

```rust
/// Safe unwrap with logging
pub fn safe_unwrap<T>(option: Option<T>, default: T, context: &str) -> T {
    match option {
        Some(v) => v,
        None => {
            log::warn!("Using default value for {}", context);
            default
        }
    }
}

/// Map error with context
pub fn map_err_context<T, E, F>(
    result: Result<T, E>,
    context: F
) -> Result<T, String>
where
    E: std::fmt::Display,
    F: FnOnce() -> String,
{
    result.map_err(|e| format!("{}: {}", context(), e))
}

/// Convert to VisionFlowError with context
pub fn to_vf_error<T, E>(result: Result<T, E>, context: &str) -> VisionFlowResult<T>
where
    E: std::error::Error + Send + Sync + 'static,
{
    result.map_err(|e| VisionFlowError::Generic {
        message: format!("{}: {}", context, e),
        source: Some(std::sync::Arc::new(e)),
    })
}
```

**Impact:** Eliminates 400+ unsafe .unwrap() calls, standardizes error handling

---

#### B. Collection Initialization Helpers
**Occurrences:** 390 HashMap::new(), 419 Vec::new()
**Recommended Module:** `src/utils/collections.rs`

```rust
/// Create HashMap with initial capacity
pub fn new_map<K, V>(capacity: usize) -> HashMap<K, V> {
    HashMap::with_capacity(capacity)
}

/// Create HashMap from key-value pairs
pub fn hash_map<K, V, const N: usize>(pairs: [(K, V); N]) -> HashMap<K, V>
where
    K: Eq + std::hash::Hash,
{
    pairs.into_iter().collect()
}

/// Create Vec with initial capacity
pub fn new_vec<T>(capacity: usize) -> Vec<T> {
    Vec::with_capacity(capacity)
}
```

**Impact:** More semantic collection initialization, prevents reallocations

---

#### C. Logging Utilities
**Occurrences:** 200+ repeated logging patterns
**Recommended Module:** `src/utils/logging_helpers.rs`

```rust
/// Log with timestamp and context
pub fn log_with_context(level: log::Level, context: &str, message: &str) {
    log::log!(level, "[{}] {}: {}", Utc::now().format("%H:%M:%S"), context, message);
}

/// Log error with full context
pub fn log_error_with_context<E: std::fmt::Display>(
    error: &E,
    operation: &str
) {
    log::error!("[{}] Operation '{}' failed: {}",
        Utc::now().format("%H:%M:%S"),
        operation,
        error
    );
}

/// Measure and log execution time
pub async fn log_execution_time<F, T>(
    operation: &str,
    f: F
) -> T
where
    F: Future<Output = T>,
{
    let start = Instant::now();
    let result = f.await;
    log::info!("{} took {:?}", operation, start.elapsed());
    result
}
```

---

#### D. Pagination Helpers
**Current:** PaginationParams exists but underutilized
**Recommended Enhancement:** `src/utils/pagination.rs`

```rust
/// Calculate pagination metadata
pub fn paginate<T>(
    items: Vec<T>,
    params: &PaginationParams
) -> PaginatedResponse<T> {
    PaginatedResponse::new(items, total_count, params)
}

/// Apply pagination to database query
pub fn apply_pagination(
    query: &str,
    params: &PaginationParams
) -> (String, Vec<i64>) {
    let offset = params.get_offset() as i64;
    let limit = params.get_limit() as i64;
    (
        format!("{} LIMIT ? OFFSET ?", query),
        vec![limit, offset]
    )
}
```

---

#### E. Time/Duration Utilities
**Occurrences:** 305 Utc::now() calls, repeated duration calculations
**Recommended Module:** `src/utils/time.rs`

```rust
use chrono::{DateTime, Duration, Utc};

/// Get current UTC timestamp
pub fn now() -> DateTime<Utc> {
    Utc::now()
}

/// Format timestamp for logging
pub fn format_log_time(dt: &DateTime<Utc>) -> String {
    dt.format("%Y-%m-%d %H:%M:%S%.3f").to_string()
}

/// Format timestamp for API responses
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
```

---

#### F. Path/IRI Utilities
**Occurrences:** 20+ IRI parsing patterns
**Recommended Module:** `src/utils/iri_helpers.rs`

```rust
/// Extract fragment from IRI (after # or last /)
pub fn extract_iri_fragment(iri: &str) -> Option<&str> {
    iri.split('#')
        .last()
        .or_else(|| iri.split('/').last())
        .filter(|s| !s.is_empty())
}

/// Extract namespace from IRI
pub fn extract_iri_namespace(iri: &str) -> Option<&str> {
    if let Some(pos) = iri.rfind('#') {
        Some(&iri[..pos])
    } else if let Some(pos) = iri.rfind('/') {
        Some(&iri[..pos])
    } else {
        None
    }
}

/// Normalize IRI (ensure proper format)
pub fn normalize_iri(iri: &str) -> String {
    iri.trim().to_string()
}

/// Join IRI parts
pub fn join_iri(namespace: &str, fragment: &str) -> String {
    if namespace.ends_with('#') || namespace.ends_with('/') {
        format!("{}{}", namespace, fragment)
    } else {
        format!("{}#{}", namespace, fragment)
    }
}
```

---

### 3.2 Medium-Impact Missing Utilities

#### G. Retry Logic Utilities
**Occurrences:** Multiple retry patterns in network code
**Recommended Module:** `src/utils/network/retry.rs` (already exists, needs expansion)

```rust
/// Retry with exponential backoff
pub async fn retry_with_backoff<F, T, E>(
    mut operation: F,
    max_retries: u32,
    initial_delay_ms: u64,
) -> Result<T, E>
where
    F: FnMut() -> Future<Output = Result<T, E>>,
{
    // Implementation with exponential backoff
}
```

---

#### H. Cache Utilities
**Occurrences:** Custom caching in multiple services
**Recommended Module:** `src/utils/cache.rs`

```rust
/// Simple in-memory cache with TTL
pub struct TtlCache<K, V> {
    store: HashMap<K, (V, Instant)>,
    ttl: Duration,
}

impl<K, V> TtlCache<K, V> {
    pub fn new(ttl: Duration) -> Self
    pub fn get(&self, key: &K) -> Option<&V>
    pub fn insert(&mut self, key: K, value: V)
    pub fn cleanup_expired(&mut self)
}
```

---

## 4. Consolidation Plan

### Phase 1: Critical Utilities (HIGH PRIORITY)
**Timeline:** Week 1-2
**Effort:** 20-25 hours
**ROI:** High

1. **JSON Processing Utilities** (2 hours)
   - Create `src/utils/json.rs`
   - Implement from_json, to_json, from_json_with_context
   - Replace 154 duplicates in event handlers, services

2. **HTTP Response Standardization** (6 hours)
   - Enforce HandlerResponse trait usage
   - Create helper macros
   - Refactor 370 non-trait HTTP responses in handlers

3. **Result/Error Helpers** (4 hours)
   - Create `src/utils/result_helpers.rs`
   - Implement safe_unwrap, map_err_context, to_vf_error
   - Replace 200 highest-risk .unwrap() calls

4. **Time Utilities** (3 hours)
   - Create `src/utils/time.rs`
   - Centralize timestamp generation and formatting
   - Replace 305 Utc::now() calls

5. **String Helpers** (3 hours)
   - Create `src/utils/string_helpers.rs`
   - Implement split_first, split_last, extract_iri_fragment
   - Replace 40 duplicate string operations

### Phase 2: Important Utilities (MEDIUM PRIORITY)
**Timeline:** Week 3-4
**Effort:** 15-20 hours
**ROI:** Medium-High

6. **Collection Helpers** (2 hours)
   - Create `src/utils/collections.rs`
   - Semantic initialization functions
   - Replace 100+ HashMap/Vec::new() calls

7. **IRI/Path Utilities** (3 hours)
   - Create `src/utils/iri_helpers.rs`
   - Consolidate IRI parsing
   - Replace 20 duplicate IRI operations

8. **GPU Conversion Utilities** (4 hours)
   - Create `src/utils/gpu_conversion.rs`
   - Consolidate 8 GPU conversion functions
   - Standardize GPU data transformations

9. **Type Conversion Utilities** (3 hours)
   - Create `src/utils/conversion/case.rs`
   - Implement snake_case, camelCase conversions
   - Replace 10 duplicate conversion functions

10. **Logging Helpers** (3 hours)
    - Create `src/utils/logging_helpers.rs`
    - Implement log_with_context, log_execution_time
    - Replace 50+ repeated logging patterns

### Phase 3: Nice-to-Have Utilities (LOW PRIORITY)
**Timeline:** Week 5-6
**Effort:** 10-12 hours
**ROI:** Medium

11. **Pagination Enhancement** (2 hours)
    - Enhance existing pagination utilities
    - Add database query helpers

12. **Cache Utilities** (3 hours)
    - Create `src/utils/cache.rs`
    - Implement TtlCache
    - Refactor custom caching

13. **Retry Logic Enhancement** (2 hours)
    - Expand `src/utils/network/retry.rs`
    - Add exponential backoff variants

14. **Validation Enhancements** (3 hours)
    - Add specialized validators
    - Extract regex constants
    - Create validation macros

---

## 5. Implementation Recommendations

### 5.1 Development Guidelines

**Before Creating New Utility:**
1. Search codebase for similar patterns (grep, ripgrep)
2. If pattern appears 3+ times, create utility function
3. If pattern appears 5+ times, create utility module
4. Document usage examples in doc comments

**Utility Design Principles:**
1. **Single Responsibility:** Each utility does one thing well
2. **Type Safety:** Use generics and traits for flexibility
3. **Error Handling:** Always return Result for fallible operations
4. **Documentation:** Include usage examples and edge cases
5. **Testing:** Unit tests for all utility functions

### 5.2 Module Organization

**Recommended Structure:**
```
src/utils/
├── mod.rs                    # Public exports
├── json.rs                   # JSON processing
├── result_helpers.rs         # Result/Option utilities
├── string_helpers.rs         # String operations
├── iri_helpers.rs           # IRI/URL parsing
├── time.rs                   # Time/timestamp utilities
├── logging_helpers.rs        # Logging utilities
├── collections.rs            # Collection helpers
├── cache.rs                  # Caching utilities
├── conversion/
│   ├── mod.rs
│   ├── case.rs              # Case conversion
│   ├── type.rs              # Type conversion
│   └── gpu.rs               # GPU conversions
├── validation/              # (existing)
│   ├── mod.rs
│   ├── sanitization.rs
│   ├── schemas.rs
│   └── rate_limit.rs
└── network/                 # (existing)
    └── retry.rs
```

### 5.3 Migration Strategy

**Incremental Adoption:**
1. Create utility module with comprehensive tests
2. Use in new code immediately
3. Refactor existing code module-by-module
4. Track migration progress with grep metrics
5. Remove deprecated patterns after migration

**Validation:**
- Run full test suite after each utility introduction
- Verify no behavioral changes
- Check performance impact (should be neutral or positive)
- Update documentation and examples

### 5.4 Metrics and Success Criteria

**Key Metrics:**
- Lines of duplicated code reduced
- Number of .unwrap() calls eliminated
- Test coverage of utility functions
- Compilation time impact
- Runtime performance impact

**Success Criteria:**
- ✅ 40%+ reduction in duplicated utility code
- ✅ 80%+ of .unwrap() calls replaced with safe alternatives
- ✅ 100% test coverage for new utilities
- ✅ No performance regression
- ✅ Improved code readability (subjective, team review)

---

## 6. Priority Summary

### Immediate Action (This Sprint)
1. JSON processing utilities - 154 duplicates
2. HTTP response standardization - 370 duplicates
3. Result/error helpers - 432 unsafe .unwrap() calls

**Estimated Effort:** 12-14 hours
**Estimated Impact:** Eliminate 400+ unsafe patterns, standardize 524 operations

### Next Sprint
4. Time utilities - 305 duplicates
5. String helpers - 40 duplicates
6. Collection helpers - 800+ initialization calls
7. IRI/Path utilities - 20 duplicates

**Estimated Effort:** 11-13 hours
**Estimated Impact:** Centralize 1,175+ operations

### Future Sprints
8. GPU conversion utilities - 8 duplicates
9. Type conversion utilities - 10 duplicates
10. Logging helpers - 50+ duplicates
11. Cache utilities - custom implementations
12. Enhanced validation - minor improvements

**Estimated Effort:** 15-18 hours
**Estimated Impact:** Consolidate 78+ specialized operations

---

## 7. Appendix: Detailed Analysis

### A. Top Files with Utility Duplication

1. **Event Handlers** (30+ JSON duplicates)
   - `src/events/handlers/graph_handler.rs`
   - `src/events/handlers/ontology_handler.rs`
   - `src/events/handlers/notification_handler.rs`
   - `src/events/handlers/audit_handler.rs`

2. **API Handlers** (100+ HTTP response duplicates)
   - `src/handlers/api_handler/analytics/*.rs`
   - `src/handlers/api_handler/ontology/*.rs`
   - `src/handlers/settings_handler.rs`

3. **Services** (50+ conversion duplicates)
   - `src/services/file_service.rs`
   - `src/services/graph_serialization.rs`
   - `src/services/ontology_reasoner.rs`

4. **Actors** (40+ error handling duplicates)
   - `src/actors/gpu/*.rs`
   - `src/actors/semantic_processor_actor.rs`
   - `src/actors/optimized_settings_actor.rs`

### B. Function Occurrence Statistics

| Pattern | Occurrences | Priority |
|---------|-------------|----------|
| `.to_string()` | 2,779 | Medium |
| `format!` | 1,510 | Low |
| Result transformations | 1,544 | HIGH |
| HTTP responses | 673 | HIGH |
| `.unwrap()` | 432 | HIGH |
| `HashMap::new()` | 390 | Medium |
| `Vec::new()` | 419 | Medium |
| `Utc::now()` | 305 | HIGH |
| JSON serialization | 103 | HIGH |
| JSON deserialization | 51 | HIGH |

### C. Grep Commands for Tracking Progress

```bash
# Track JSON duplicates
grep -rn "serde_json::from_str" src/ --include="*.rs" | wc -l

# Track HTTP response patterns
grep -rn "HttpResponse::" src/ --include="*.rs" | grep -v "HandlerResponse" | wc -l

# Track unsafe unwrap calls
grep -rn "\.unwrap()" src/ --include="*.rs" | wc -l

# Track timestamp generation
grep -rn "Utc::now()" src/ --include="*.rs" | wc -l
```

---

## Conclusion

This analysis identified **significant opportunities for consolidation** across 8 major utility categories. The codebase has **excellent validation organization** but suffers from **duplicated error handling, JSON processing, and HTTP response construction**.

**Recommended Focus Areas:**
1. **Error handling standardization** (1,544 patterns, 432 unsafe calls)
2. **JSON processing utilities** (154 duplicates)
3. **HTTP response enforcement** (370 non-standard constructions)
4. **Time/timestamp centralization** (305 scattered calls)

**Expected Outcomes:**
- 40-50% reduction in duplicated utility code
- Elimination of 400+ unsafe .unwrap() calls
- Standardized error handling across codebase
- Improved maintainability and consistency

**Total Estimated Effort:** 45-55 hours over 6 weeks
**Total Lines Reduced:** ~2,000+ duplicated lines
**Risk:** Low (incremental refactoring, comprehensive testing)
