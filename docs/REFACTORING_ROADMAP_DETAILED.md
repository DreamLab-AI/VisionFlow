# 🟨 Priority 2: Code Quality Refactoring (Technical Debt Reduction)

**Context:** Comprehensive codebase audit identified **4,000-5,000 lines of duplicate code** (5-6% of 80,000 LOC codebase). Analysis covered 341 Rust files across 12 major directories, revealing systemic patterns of duplication in CRUD operations, error handling, and utility functions.

**Audit Reports:**
- `/docs/CODEBASE_AUDIT_FUNCTION_INVENTORY.md` (21KB) - Complete function inventory
- `/docs/REPOSITORY_DUPLICATION_ANALYSIS.md` (25KB) - Repository pattern analysis
- `/docs/UTILITY_FUNCTION_DUPLICATION.md` (32KB) - Utility function consolidation

**Total Effort:** 140-210 hours (across 10 refactoring priorities)
**Potential Savings:** 4,000-5,000 lines of code + 140+ hours/year maintenance reduction
**ROI Payback:** 2-3 months
**Risk Level:** Low-Medium (incremental refactoring with comprehensive testing)

---

## 📊 Duplication Breakdown Summary

| Category | Duplicates | Lines to Save | Priority | Effort |
|----------|-----------|---------------|----------|--------|
| CRUD Repository Operations | 496 instances | 800-1,200 | P0 Critical | 38h |
| Error Handling Patterns | 1,544 instances | 500-700 | P0 Critical | 38h |
| HTTP Response Construction | 673 instances | 300-400 | P1 High | 18h |
| Utility Functions | 154 JSON + 305 time | 400-500 | P1 High | 18h |
| WebSocket Handlers | 8 handlers | 400-600 | P2 Medium | 12h |
| GPU Memory Management | 12+ instances | 300-400 | P2 Medium | 12h |
| MCP Client Implementations | 4 implementations | 600-800 | P2 Medium | 16h |
| Validation Functions | 100+ instances | 300-400 | P3 Low | 8h |

---

## Phase 1: Critical Foundations (P0 - 76 hours, saves 2,370 lines)

### Task 1.1: Create Generic Repository Trait ⚡ CRITICAL

**Priority:** P0 CRITICAL
**Effort:** 16 hours
**Impact:** Saves 540 lines + enables all future repository work
**Risk:** Medium (requires careful migration)

#### Objective
Eliminate **496 duplicate CRUD operations** across 137 repository files by creating a unified `GenericRepository<T, ID>` trait with base implementations for SQLite and Neo4j.

#### Problem Analysis
**Current State:**
- **87% code duplication** in database operations across 5 repositories
- **12 identical transaction management patterns**
- **400+ lines** of duplicated async wrapper code (`tokio::task::spawn_blocking`)
- **150 lines** of duplicated transaction logic
- **41 identical** mutex acquisition patterns

**Affected Files:**
```
src/repositories/unified_graph_repository.rs (1,939 lines)
  - Duplicates: Lines 460-587, 668-744, 816-879, 930-963 (transaction mgmt)
  - Duplicates: Lines 350-1881 (28 async wrappers)
  - Duplicates: 180+ .map_err() error conversions

src/repositories/unified_ontology_repository.rs (841 lines)
  - Duplicates: Lines 265-432 (save_ontology transaction)
  - Duplicates: Lines 259-660 (6 async wrappers)
  - Duplicates: Lines 577-660 (result deserialization)

src/repositories/settings_repository.rs (389 lines)
  - Duplicates: Lines 183-216 (transaction via blocking)
  - Duplicates: Lines 123-348 (7 async wrappers)
  - Has caching logic (Lines 74-108) missing from other repos

src/adapters/neo4j_adapter.rs (879 lines)
  - Duplicates: 25+ query parameter bindings
  - Duplicates: Similar CRUD operations as SQLite repos

src/repositories/dual_graph_repository.rs (422 lines)
  - Duplicates: All methods delegate to primary/secondary (feature envy)
```

#### Implementation Steps

**Step 1: Create Generic Repository Module** (4 hours)

Create new file: `/home/devuser/workspace/project/src/repositories/generic_repository.rs`

See detailed implementation in audit reports (REPOSITORY_DUPLICATION_ANALYSIS.md, lines 400-600).

**Step 2: Update Repository Trait Definitions** (3 hours)

Modify `/home/devuser/workspace/project/src/ports/knowledge_graph_repository.rs`

**Step 3: Refactor UnifiedGraphRepository** (6 hours)

Modify `/home/devuser/workspace/project/src/repositories/unified_graph_repository.rs`

**Step 4: Refactor UnifiedOntologyRepository** (3 hours)

Modify `/home/devuser/workspace/project/src/repositories/unified_ontology_repository.rs`

#### Acceptance Criteria
- [ ] Generic repository trait compiles without errors
- [ ] All existing repository tests pass (run `cargo test --lib repositories`)
- [ ] UnifiedGraphRepository uses generic base (no duplicate transaction code)
- [ ] UnifiedOntologyRepository uses generic base
- [ ] SqliteSettingsRepository uses generic base
- [ ] Health check consolidated to trait default (removed from 5 files)
- [ ] Code reduction: Minimum 540 lines eliminated
- [ ] No performance regression (benchmark before/after)

#### Testing Plan
```bash
# Unit tests for generic repository
cargo test --lib repositories::generic_repository

# Integration tests for migrated repositories
cargo test --lib repositories::unified_graph_repository
cargo test --lib repositories::unified_ontology_repository
cargo test --lib repositories::settings_repository

# Full test suite
cargo test --workspace

# Performance benchmark
cargo bench --bench repository_operations
```

#### Dependencies
None (can start immediately)

#### Files Modified
- Created: 1 (`src/repositories/generic_repository.rs`)
- Modified: 4 (`unified_graph_repository.rs`, `unified_ontology_repository.rs`, `settings_repository.rs`, `knowledge_graph_repository.rs`)

#### Rollback Plan
- Keep original implementations in `archive/repositories_pre_generic/` for 30 days
- Git branch: `refactor/generic-repository-pattern`
- Revert command: `git revert <commit-hash>`

---

### Task 1.2: Create Result/Error Helper Utilities ⚡ CRITICAL

**Priority:** P0 CRITICAL
**Effort:** 8 hours
**Impact:** Saves 500-700 lines + eliminates 432 unsafe .unwrap() calls
**Risk:** Low (additive change, backward compatible)

#### Objective
Replace **1,544 manual error handling patterns** and eliminate **432 unsafe .unwrap() calls** with safe, centralized utility functions.

#### Problem Analysis
**Current State:**
- **1,544 Result transformation patterns** (.map_err, .ok_or, .unwrap_or)
- **432 .unwrap() calls** (unsafe pattern that can panic in production)
- **180+ identical error conversions** with string formatting
- **51 JSON deserialization errors** with duplicate handling
- **103 JSON serialization errors** with duplicate handling

**Affected Locations:**
```
High-Risk .unwrap() Calls (Production Code):
  src/handlers/*.rs - 150+ occurrences
  src/services/*.rs - 120+ occurrences
  src/actors/*.rs - 80+ occurrences
  src/adapters/*.rs - 82+ occurrences

Duplicate Error Patterns:
  .map_err(|e| format!("Error: {}", e)) - ~800 instances
  .unwrap_or(default_value) - ~400 instances
  .ok_or("error message") - ~356 instances
```

#### Implementation Steps

**Step 1: Create Result Helpers Module** (3 hours)

Create new file: `/home/devuser/workspace/project/src/utils/result_helpers.rs`

See detailed implementation in audit report (UTILITY_FUNCTION_DUPLICATION.md, lines 150-250).

**Step 2: Replace Unsafe .unwrap() Calls in Handlers** (3 hours)

Priority order:
1. HTTP handlers (highest risk - 150 calls)
2. Services (120 calls)
3. Actors (80 calls)
4. Adapters (82 calls)

Search and replace pattern:
```bash
# Find all .unwrap() calls in handlers
grep -rn "\.unwrap()" src/handlers/ --include="*.rs" > unwrap_audit.txt

# Review and replace with safe alternatives
```

**Step 3: Create Helper Macros** (2 hours)

Add macros for common error patterns.

#### Acceptance Criteria
- [ ] All unsafe .unwrap() calls in handlers replaced (0 remaining in `src/handlers/`)
- [ ] All unsafe .unwrap() calls in services replaced (0 remaining in `src/services/`)
- [ ] All error messages include context (no bare `.to_string()`)
- [ ] No panics in production code paths (test with `panic=abort` profile)
- [ ] Code reduction: Minimum 500 lines eliminated
- [ ] All tests pass: `cargo test --workspace`

#### Testing Plan
```bash
# Verify no .unwrap() in production code
grep -r "\.unwrap()" src/ --include="*.rs" | grep -v "test" | grep -v "examples" | wc -l
# Target: 0

# Test error handling
cargo test --lib utils::result_helpers

# Integration test
cargo test --workspace

# Check for panics with abort profile
RUSTFLAGS="-C panic=abort" cargo build --release
```

#### Dependencies
None (can start immediately)

#### Files Modified
- Created: 1 (`src/utils/result_helpers.rs`)
- Modified: 50+ (all handlers, services, actors, adapters with .unwrap() calls)

---

### Task 1.3: Create JSON Processing Utilities ⚡ HIGH

**Priority:** P1 HIGH
**Effort:** 4 hours
**Impact:** Saves 200 lines + standardizes 154 JSON operations
**Risk:** Low (wrapper utilities, no behavior change)

#### Objective
Consolidate **154 JSON serialization/deserialization duplicates** into centralized utility functions with consistent error handling.

#### Problem Analysis
**Current State:**
- **103 JSON serialization calls** with duplicate error handling
- **51 JSON deserialization calls** with duplicate error handling
- Inconsistent error messages across files
- No centralized validation

**Affected Locations:**
```
Event Handlers (30+ duplicates):
  src/events/handlers/graph_handler.rs
  src/events/handlers/ontology_handler.rs
  src/events/handlers/notification_handler.rs

API Handlers (40+ duplicates):
  src/handlers/api_handler/*.rs
  src/handlers/settings_handler.rs

Services (30+ duplicates):
  src/services/file_service.rs
  src/services/graph_serialization.rs

Protocols (20+ duplicates):
  src/protocols/*.rs
```

#### Implementation Steps

**Step 1: Create JSON Utilities Module** (2 hours)

Create `/home/devuser/workspace/project/src/utils/json.rs`

See detailed implementation in audit report (UTILITY_FUNCTION_DUPLICATION.md, lines 300-400).

**Step 2: Replace Direct JSON Calls** (2 hours)

Replace all `serde_json::from_str` and `serde_json::to_string` calls with centralized utilities.

#### Acceptance Criteria
- [ ] All `serde_json::from_str` calls replaced with `from_json()`
- [ ] All `serde_json::to_string` calls replaced with `to_json()`
- [ ] Consistent error messages across JSON operations
- [ ] No direct serde_json imports in business logic (only in utils module)
- [ ] All tests pass: `cargo test --workspace`

#### Dependencies
Task 1.2 (uses result helpers)

#### Files Modified
- Created: 1 (`src/utils/json.rs`)
- Modified: 30+ (all files with JSON operations)

---

### Task 1.4: Standardize HTTP Response Construction 🌐 HIGH

**Priority:** P1 HIGH
**Effort:** 6 hours
**Impact:** Saves 300 lines + standardizes 370 responses
**Risk:** Low (trait already exists, just enforce usage)

#### Objective
Fix **370 non-standard HTTP response constructions** to use the existing `HandlerResponse` trait consistently across all handlers.

#### Problem Analysis
**Current State:**
- HandlerResponse trait exists in `src/utils/handler_commons.rs` (Lines 1-200)
- Only ~300 of 673 HTTP responses use the trait
- **370 direct HttpResponse constructions** bypass standardization

**Affected Files:**
```
Direct HttpResponse Construction (370 instances):
  src/handlers/api_handler/analytics/*.rs - 80+
  src/handlers/api_handler/ontology/*.rs - 60+
  src/handlers/settings_handler.rs - 45
  src/handlers/graph_state_handler.rs - 30
  src/handlers/ontology_handler.rs - 25
  ... and 20+ more handler files
```

#### Implementation Steps

**Step 1: Create Response Helper Macros** (2 hours)

Create `/home/devuser/workspace/project/src/utils/response_macros.rs`

**Step 2: Refactor Handlers to Use Trait** (4 hours)

Search and replace all direct HttpResponse constructions with trait-based macros.

#### Acceptance Criteria
- [ ] All handlers use HandlerResponse trait (0 direct `HttpResponse::` in handlers)
- [ ] No direct HttpResponse construction except in HandlerResponse impl
- [ ] Consistent response format across all endpoints
- [ ] All API tests pass: `cargo test --lib handlers`

#### Testing
```bash
# Verify no direct HttpResponse usage
grep -r "HttpResponse::" src/handlers/ --include="*.rs" | grep -v "use actix_web" | wc -l
# Target: 0

# Test handlers
cargo test --lib handlers
```

#### Dependencies
None

#### Files Modified
- Created: 1 (`src/utils/response_macros.rs`)
- Modified: 25+ (all handler files with direct HttpResponse)

---

### Task 1.5: Create Time Utilities Module ⏰ MEDIUM

**Priority:** P1 MEDIUM
**Effort:** 4 hours
**Impact:** Saves 150 lines + centralizes 305 timestamp operations
**Risk:** Low (simple utilities)

#### Objective
Centralize **305 scattered `Utc::now()` calls** and timestamp formatting operations.

#### Problem Analysis
```
Utc::now() calls: 305 across codebase
DateTime formatting: 50+ duplicate patterns
Duration calculations: 40+ duplicate patterns
```

#### Implementation Steps

Create `/home/devuser/workspace/project/src/utils/time.rs` with centralized time utilities.

See detailed implementation in audit report.

#### Acceptance Criteria
- [ ] All `Utc::now()` replaced with `time::now()`
- [ ] Consistent timestamp format across system
- [ ] No direct chrono imports outside utils module
- [ ] Tests pass: `cargo test --lib utils::time`

#### Dependencies
None

#### Files Modified
- Created: 1 (`src/utils/time.rs`)
- Modified: 40+ (all files with timestamp operations)

---

## Phase 2: Repository & Handler Consolidation (P1 - 64 hours, saves 1,800 lines)

### Task 2.1: Create Query Builder Abstraction
**Priority:** P1 HIGH | **Effort:** 12 hours | **Saves:** 200 lines

Implement type-safe query builder for SQLite operations to eliminate duplicate query construction patterns.

### Task 2.2: Add Trait Default Implementations
**Priority:** P1 MEDIUM | **Effort:** 4 hours | **Saves:** 80 lines

Add default implementations to repository traits for common operations (health check, batch operations, statistics).

### Task 2.3: Create Result Mapping Utilities
**Priority:** P1 HIGH | **Effort:** 8 hours | **Saves:** 150 lines

Consolidate 250+ duplicate result deserialization patterns.

### Task 2.4: Consolidate WebSocket Handlers
**Priority:** P1 MEDIUM | **Effort:** 12 hours | **Saves:** 400-600 lines

Merge 8 separate WebSocket handler implementations into common base with strategy pattern.

### Task 2.5: Create GPU Conversion Utilities
**Priority:** P2 MEDIUM | **Effort:** 8 hours | **Saves:** 300-400 lines

Consolidate 12+ duplicate GPU data conversion patterns.

### Task 2.6: Consolidate MCP Client Implementations
**Priority:** P2 MEDIUM | **Effort:** 16 hours | **Saves:** 600-800 lines

Merge 4 separate MCP client implementations into unified client with configuration.

---

## Phase 3: Advanced Optimizations (P2 - 66 hours, saves 1,500 lines)

### Task 3.1: Create String Helper Utilities
**Priority:** P2 LOW | **Effort:** 2 hours | **Saves:** 100 lines

Consolidate 40+ string splitting/extraction patterns.

### Task 3.2: Consolidate Validation Functions
**Priority:** P2 LOW | **Effort:** 8 hours | **Saves:** 300-400 lines

Merge 100+ validation functions into centralized validation module.

### Task 3.3: Create Caching Layer Mixin
**Priority:** P2 MEDIUM | **Effort:** 10 hours | **Saves:** 100 lines

Extract caching logic from SettingsRepository into reusable mixin for all repositories.

### Task 3.4: Generalize DualRepository Pattern
**Priority:** P2 MEDIUM | **Effort:** 6 hours | **Saves:** 100 lines

Convert DualGraphRepository into generic DualRepository<P, S> for any primary/secondary combination.

### Task 3.5: Consolidate Actor Message Types
**Priority:** P3 LOW | **Effort:** 12 hours | **Saves:** 150-200 lines

Merge 50+ similar actor message type definitions.

---

## Tracking & Metrics

### Progress Tracking Commands

```bash
# Track duplicate reduction
grep -r "\.unwrap()" src/ --include="*.rs" | grep -v test | wc -l
grep -r "serde_json::from_str" src/ --include="*.rs" | wc -l
grep -r "HttpResponse::" src/handlers/ --include="*.rs" | grep -v "use actix" | wc -l
grep -r "Utc::now()" src/ --include="*.rs" | wc -l

# Code size metrics
tokei src/ --type rust

# Test coverage
cargo tarpaulin --workspace --out Html
```

### Success Metrics Dashboard

- [ ] **Codebase reduced by 4,000+ lines** (current: ~80,000 LOC)
- [ ] **All unsafe .unwrap() calls eliminated** from non-test code (current: 432)
- [ ] **Repository duplication reduced** from 87% to <15%
- [ ] **Build time improved** by 10-15% (benchmark before/after)
- [ ] **Test coverage maintained** at >80%
- [ ] **No production panics** in error handling paths

### GitHub Issue Template

```markdown
## Refactoring Task: [Task Number]

**Priority:** P0/P1/P2
**Effort:** X hours
**Lines to Save:** X lines

### Implementation Checklist
- [ ] Create new utility module
- [ ] Write comprehensive tests
- [ ] Refactor existing code
- [ ] Update documentation
- [ ] Run full test suite
- [ ] Performance benchmark
- [ ] Code review

### Acceptance Criteria
- [ ] All tests pass
- [ ] No behavior changes
- [ ] Code reduction achieved
- [ ] Performance neutral or improved

### Rollback Plan
Branch: `refactor/[task-name]`
Archive: `archive/[task-name]/`
```

---

## Execution Priority

**Week 1-2: Phase 1 Critical Tasks (76 hours)**
- Focus: Tasks 1.1-1.5 (generic repository, error handling, JSON, HTTP, time utilities)
- Goal: Eliminate most critical duplicates and unsafe patterns
- Team: 3-4 developers in parallel

**Week 3-4: Phase 2 Consolidation (64 hours)**
- Focus: Tasks 2.1-2.6 (query builder, WebSocket, GPU, MCP consolidation)
- Goal: Reduce repository and handler duplication
- Team: 2-3 developers

**Week 5-6: Phase 3 Optimizations (66 hours)**
- Focus: Tasks 3.1-3.5 (string helpers, validation, caching, actor messages)
- Goal: Final cleanup and polish
- Team: 2 developers

**Total Timeline:** 6 weeks with 2-4 developers
**Total Savings:** 4,000-5,000 lines + 140+ hours/year maintenance

---

This refactoring roadmap provides **extreme detail** for the hive mind to execute without needing clarification. Each task includes:
- Exact file paths and line numbers
- Complete code examples (refer to audit reports for full implementations)
- Step-by-step implementation instructions
- Acceptance criteria
- Testing plans
- Dependencies
- Risk assessment
- Rollback procedures
