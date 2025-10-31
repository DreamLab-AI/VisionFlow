# Legacy Code Removal Plan: Unified Database Architecture
**Status**: ANALYSIS COMPLETE - AWAITING EXECUTION APPROVAL
**Date**: 2025-10-31
**Critical Finding**: Current "unified" repositories are EXPERIMENTAL - Legacy is PRODUCTION

---

## ⚠️ CRITICAL DISCOVERY

**The UnifiedGraphRepository and UnifiedOntologyRepository are NOT production-ready replacements.**

After comprehensive codebase analysis:
- ✅ **SqliteKnowledgeGraphRepository**: Fully implemented (1017 lines), actively used in production
- ✅ **SqliteOntologyRepository**: Fully implemented (1083 lines), actively used in production
- ⚠️ **UnifiedGraphRepository**: Research/experimental code, not integrated
- ⚠️ **UnifiedOntologyRepository**: Research/experimental code, not integrated
- ✅ **OntologyGraphBridge**: Actively used for synchronization between databases

**Current Architecture**: Dual-database system (knowledge_graph.db + ontology.db) is PRODUCTION

**Recommendation**: DO NOT REMOVE legacy adapters until unified repositories are production-ready

---

## Phase 1: Safe to Remove Immediately (Documentation Only)

### 1.1 Legacy Documentation Files
**Risk**: NONE - Documentation only, no code dependencies

```bash
# Research documents (migration planning completed)
rm -f docs/research/research/Detailed\ Migration\ Roadmap.md
rm -f docs/research/research/Master-Architecture-Diagrams.md
rm -f docs/research/research/Migration_Strategy_Options.md
rm -f docs/research/research/Legacy-Knowledge-Graph-System-Analysis.md
rm -f docs/research/research/MIGRATION-CHECKLIST.md
rm -f docs/research/research/EXECUTIVE-SUMMARY.md

# Outdated migration docs
rm -f migration/COMPLETION_REPORT.md
rm -f docs/INTEGRATION_COMPLETE.md
rm -f docs/WEEK12_INTEGRATION.md
rm -f docs/WEEK5_UNIFIED_ADAPTERS.md
```

**Files**: 10+ markdown files
**Reason**: Migration research completed, information preserved in active docs
**Dependencies**: None
**Action**: Move to `/docs/archive/migration-research/` instead of deleting

---

## Phase 2: Remove After Dependency Updates (HIGH PRIORITY)

### 2.1 Test Files for Legacy Adapters
**Risk**: LOW - Tests can be recreated for unified repositories
**Blocker**: Need unified repository integration tests first

```bash
# Adapter-specific tests
tests/adapters/sqlite_knowledge_graph_repository_tests.rs
tests/adapters/sqlite_ontology_repository_tests.rs
tests/benchmarks/repository_benchmarks.rs (uses both legacy repos)
```

**Action Required**:
1. Create `tests/repositories/unified_graph_repository_tests.rs`
2. Create `tests/repositories/unified_ontology_repository_tests.rs`
3. Verify unified repositories pass all existing test scenarios
4. THEN remove legacy test files

---

### 2.2 Migration Scripts (One-Time Use)
**Risk**: LOW - Already executed or obsolete
**Blocker**: Verify no rollback scenarios need these

```bash
# One-time migration utilities
migration/src/export_knowledge_graph.rs
migration/src/export_ontology.rs
migration/src/import_to_unified.rs
migrations/001_fix_ontology_schema.sql
scripts/migrate_ontology_database.sql
```

**Action Required**:
1. Confirm unified database is stable in production
2. Archive migration scripts to `/archive/migration-tools/`
3. Document migration history in CHANGELOG.md
4. Remove from active codebase

---

## Phase 3: Critical Dependency Resolution (MUST COMPLETE FIRST)

### 3.1 Active Service Dependencies
**Risk**: CRITICAL - Breaking changes
**Files with Active Dependencies**:

```rust
// BLOCKING FILES - Cannot remove legacy repos until these are updated:

// src/services/streaming_sync_service.rs (Lines 40-41)
use crate::adapters::sqlite_knowledge_graph_repository::SqliteKnowledgeGraphRepository;
use crate::adapters::sqlite_ontology_repository::SqliteOntologyRepository;

// src/services/github_sync_service.rs (Lines 7-8)
use crate::adapters::sqlite_knowledge_graph_repository::SqliteKnowledgeGraphRepository;
use crate::adapters::sqlite_ontology_repository::SqliteOntologyRepository;

// src/services/ontology_graph_bridge.rs (Lines 10-11)
use crate::adapters::sqlite_knowledge_graph_repository::SqliteKnowledgeGraphRepository;
use crate::adapters::sqlite_ontology_repository::SqliteOntologyRepository;

// src/app_state.rs (Lines 56-57, 94-95)
use crate::adapters::sqlite_knowledge_graph_repository::SqliteKnowledgeGraphRepository;
use crate::adapters::sqlite_ontology_repository::SqliteOntologyRepository;
pub knowledge_graph_repository: Arc<SqliteKnowledgeGraphRepository>,
pub ontology_repository: Arc<SqliteOntologyRepository>,

// src/main.rs (Lines 7-8)
use webxr::adapters::sqlite_knowledge_graph_repository::SqliteKnowledgeGraphRepository;
use webxr::adapters::sqlite_ontology_repository::SqliteOntologyRepository;

// src/adapters/mod.rs (Lines 15-16, 37-38)
pub mod sqlite_knowledge_graph_repository;
pub mod sqlite_ontology_repository;
pub use sqlite_knowledge_graph_repository::SqliteKnowledgeGraphRepository;
pub use sqlite_ontology_repository::SqliteOntologyRepository;
```

**Update Strategy**:
```rust
// BEFORE (Legacy):
use crate::adapters::sqlite_knowledge_graph_repository::SqliteKnowledgeGraphRepository;
use crate::adapters::sqlite_ontology_repository::SqliteOntologyRepository;

// AFTER (Unified):
use crate::repositories::unified_graph_repository::UnifiedGraphRepository;
use crate::repositories::unified_ontology_repository::UnifiedOntologyRepository;
```

---

### 3.2 OntologyGraphBridge Service
**Risk**: HIGH - Active synchronization service
**File**: `src/services/ontology_graph_bridge.rs`

**Current Role**:
- Synchronizes ontology.db (OWL classes) → knowledge_graph.db (visualization nodes)
- Converts OWL class hierarchies to graph edges
- Used by admin sync endpoints

**Removal Strategy**:
```
Option A: Remove entirely (if unified DB eliminates need)
  - Unified schema already has both ontology + graph data
  - No cross-database sync needed

Option B: Refactor to UnifiedGraphBridge (if sync still needed)
  - Bridge between unified DB and external ontology sources
  - Maintain sync functionality with new architecture
```

**Decision Required**: Determine if bridge pattern still needed in unified architecture

---

### 3.3 Repository Tests Integration
**File**: `src/repositories/repository_tests.rs`

```rust
// Lines 10-11: Uses BOTH legacy and unified repos
use crate::adapters::sqlite_knowledge_graph_repository::SqliteKnowledgeGraphRepository;
use crate::adapters::sqlite_ontology_repository::SqliteOntologyRepository;
```

**Action**: Update tests to compare unified vs legacy, then remove legacy imports

---

## Phase 4: Core Legacy Files (FINAL REMOVAL)

### 4.1 Legacy Repository Implementations
**Risk**: CRITICAL - Production code
**Files**:
- `src/adapters/sqlite_knowledge_graph_repository.rs` (1017 lines)
- `src/adapters/sqlite_ontology_repository.rs` (1083 lines)

**Prerequisites**:
1. ✅ UnifiedGraphRepository fully implemented
2. ✅ UnifiedOntologyRepository fully implemented
3. ✅ All services updated (Phase 3.1)
4. ✅ All tests passing with unified repos
5. ✅ Production deployment validated
6. ✅ Performance benchmarks meet/exceed legacy
7. ✅ Rollback plan documented

**Removal Steps**:
```bash
# 1. Create feature flag for gradual rollout
[features]
use_unified_repositories = []

# 2. Deploy with both implementations side-by-side
# 3. Monitor unified repo performance in production
# 4. Switch default to unified repos
# 5. Deprecate legacy with warning logs
# 6. Remove after 2 release cycles
```

---

### 4.2 Dual Database Schema Files
**Risk**: MEDIUM - Historical reference value
**Action**: Archive, do not delete

```bash
# Archive instead of removing (historical value)
mkdir -p archive/legacy-schemas/
mv schema/knowledge_graph_db.sql archive/legacy-schemas/
mv schema/ontology_db.sql archive/legacy-schemas/
mv schema/ontology_metadata_db.sql archive/legacy-schemas/
```

---

### 4.3 Legacy Configuration References
**Risk**: LOW - Already migrated

```toml
# Remove from Cargo.toml after unified repos stable
[dependencies]
rusqlite = { version = "0.32", features = ["bundled"] }  # May still be needed
```

---

## Phase 5: Verification & Cleanup

### 5.1 Code Reference Audit
**Command**:
```bash
# Verify no remaining references
rg "SqliteKnowledgeGraphRepository|SqliteOntologyRepository" --type rust
rg "knowledge_graph\.db|ontology\.db" --type rust
rg "OntologyGraphBridge" --type rust
```

**Expected Result**: No matches (except in archived files)

---

### 5.2 Database Migration Validation
**Tests Required**:
1. Data integrity: All records migrated correctly
2. Performance: Query times ≤ legacy system
3. Concurrent access: No deadlocks or race conditions
4. Backup/restore: Unified backups work correctly
5. Schema versioning: Migration tracking in place

---

### 5.3 Documentation Updates
**Files to Update**:
```
README.md - Remove dual-database references
docs/architecture/04-database-schemas.md - Update to unified schema
docs/api/migration-guide.md - Add unified repository section
docs/ROADMAP.md - Mark migration complete
CHANGELOG.md - Document removal of legacy repos
```

---

## Current Status Summary

| Component | Status | Production Ready | Can Remove |
|-----------|--------|------------------|------------|
| SqliteKnowledgeGraphRepository | ✅ Active | ✅ Yes | ❌ No |
| SqliteOntologyRepository | ✅ Active | ✅ Yes | ❌ No |
| OntologyGraphBridge | ✅ Active | ✅ Yes | ❌ No |
| UnifiedGraphRepository | ⚠️ Experimental | ❌ No | N/A |
| UnifiedOntologyRepository | ⚠️ Experimental | ❌ No | N/A |
| Dual Database System | ✅ Production | ✅ Yes | ❌ No |
| Legacy Documentation | ✅ Complete | N/A | ✅ Yes |
| Migration Scripts | ✅ Complete | N/A | ⚠️ Archive Only |

---

## Recommended Execution Order

### ✅ Can Execute Now (Zero Risk)
1. Archive legacy documentation (Phase 1)
2. Archive migration scripts (Phase 2.2)
3. Update documentation to reflect current state

### ⚠️ High Priority (Blocks Everything)
1. Complete UnifiedGraphRepository implementation
2. Complete UnifiedOntologyRepository implementation
3. Comprehensive integration testing
4. Production validation with real workloads

### 🔴 Cannot Execute Yet (Dependencies)
1. Remove legacy adapters (blocked by unified repos)
2. Remove OntologyGraphBridge (architecture decision needed)
3. Remove dual database schemas (need unified schema stable)
4. Remove adapter tests (need unified repo tests)

---

## Risk Assessment

### Critical Risks
1. **Premature removal of legacy repos** → System failure
   - Mitigation: Feature flagging, gradual rollout, extensive testing

2. **Data loss during migration** → Business impact
   - Mitigation: Comprehensive backups, migration validation, rollback procedures

3. **Performance regression** → User experience degradation
   - Mitigation: Benchmarking, load testing, performance monitoring

### Medium Risks
1. **Service disruption during transition** → Downtime
   - Mitigation: Blue-green deployment, canary releases

2. **Incomplete test coverage** → Bugs in production
   - Mitigation: Increase unified repo test coverage to >90%

---

## Success Criteria

### Phase Completion Checklist
- [ ] All legacy documentation archived
- [ ] Migration scripts archived with documentation
- [ ] Unified repositories fully implemented
- [ ] All services updated to use unified repos
- [ ] OntologyGraphBridge decision made and implemented
- [ ] All tests passing with unified repos
- [ ] Performance benchmarks meet targets:
  - Query latency: <10ms p99
  - Throughput: ≥ legacy system
  - Memory usage: ≤ legacy system
- [ ] Production deployment successful
- [ ] Zero regressions in functionality
- [ ] Legacy code removed
- [ ] Documentation updated
- [ ] CHANGELOG.md reflects changes

---

## Timeline Estimate

**Phase 1** (Documentation): 1 day
**Phase 2** (Test Migration): 3 days
**Phase 3** (Service Updates): 5 days
**Phase 4** (Legacy Removal): 3 days
**Phase 5** (Verification): 3 days

**Total Estimated Effort**: 15 days (3 weeks)

**Prerequisites**: Unified repositories must be production-ready first (unknown timeline)

---

## Conclusion

**Current State**: Legacy adapters are PRODUCTION CODE, not deprecated code.

**Key Insight**: The "unified" repositories exist but are not integrated. This is a FUTURE architecture, not current reality.

**Immediate Action**: DO NOT REMOVE ANY CODE marked as "legacy" until:
1. Unified repositories are fully implemented
2. Production validation complete
3. All dependencies updated
4. Comprehensive testing passed

**Safe Actions**:
- Archive documentation (Phase 1)
- Plan migration strategy (this document)
- Begin unified repository implementation

**Unsafe Actions**:
- Removing SqliteKnowledgeGraphRepository
- Removing SqliteOntologyRepository
- Removing OntologyGraphBridge
- Deleting dual-database schemas

---

**Document Version**: 1.0
**Author**: Code Quality Analysis Agent
**Last Updated**: 2025-10-31
**Next Review**: After unified repositories implemented
**Status**: AWAITING ARCHITECTURAL DECISION ON UNIFIED REPO TIMELINE
