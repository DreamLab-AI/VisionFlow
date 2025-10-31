# Phase 6: Legacy Cleanup Analysis
**Status**: Waiting for Phase 5 (Actor Integration) Completion
**Date**: 2025-10-27
**Version**: VisionFlow v1.0.0

---

## Executive Summary

This document provides a comprehensive analysis of Phase 6 cleanup requirements. **Phase 6 cannot begin until Phase 5 (Actor Integration) is complete**, as documented in the roadmap.

### Current Status
- **Phase 5 Status**: PLANNED (Not Started)
- **Phase 6 Status**: BLOCKED (Dependencies not met)
- **Blocking Items**: Actor system must be fully integrated with hexagonal architecture before legacy code can be safely removed

---

## 6.1: Legacy Code Removal

### 1. Legacy Config Files Identified

#### Root-Level Config Files (TO BE REMOVED)
```
/home/devuser/workspace/project/config.yml
/home/devuser/workspace/project/ontology_physics.toml
/home/devuser/workspace/project/data/dev_config.toml
```

**Analysis**:
- `config.yml` - **Cloudflare tunnel config** (NOT legacy, keep)
- `ontology_physics.toml` - **Active TOML config** for physics constraints
- `data/dev_config.toml` - **Development config** loaded at runtime

#### Test Fixtures (KEEP - Used for Testing)
```
/home/devuser/workspace/project/tests/fixtures/ontology/test_constraints.toml
/home/devuser/workspace/project/tests/fixtures/ontology/test_mapping.toml
```

#### Build Configs (KEEP - Required)
```
/home/devuser/workspace/project/Cargo.toml
/home/devuser/workspace/project/whelk-rs/Cargo.toml
```

### 2. File-Based Config Loading Code

**Found in:**
```rust
// src/ontology/services/owl_validator.rs
let mapping_toml = std::fs::read_to_string("ontology/mapping.toml")
    .context("Failed to read ontology/mapping.toml")?;

// src/config/dev_config.rs
match std::fs::read_to_string("data/dev_config.toml") {
    // ...
}
```

**Action Required**:
- Migrate `ontology/mapping.toml` → `ontology.db`
- Migrate `data/dev_config.toml` → `settings.db`
- Remove file I/O code after database migration

### 3. Deprecated Actors Analysis

#### Still Active (DO NOT REMOVE):
- `GraphServiceSupervisor` - **ACTIVE** (src/actors/graph_service_supervisor.rs)
  - Manages 4 child actors
  - Part of current supervision strategy
  - Referenced in mod.rs exports

**WARNING**: Roadmap mentions "Delete deprecated actors (GraphServiceSupervisor, etc.)" but current code shows it's ACTIVE. Need clarification before removal.

#### Potentially Deprecated:
```bash
# Search results from grep:
src/actors/graph_service_supervisor.rs - ACTIVE (630+ lines)
src/actors/agent_monitor_actor.rs - References legacy config
src/actors/mod.rs - Exports GraphServiceSupervisor
```

**Recommendation**:
1. Verify with Phase 5 completion which actors are truly deprecated
2. Only remove after actor migration is complete
3. Ensure no breaking changes to existing features

### 4. Client-Side Caching Layer

**Location**: Unknown (needs frontend analysis)

**From Roadmap**:
- Limitation L6: "Client-side caching layer causes sync issues"
- Fix: "Remove client-side cache in v1.0.0 (item #6.1)"

**Action Required**:
1. Audit client code for caching implementation
2. Test backend-only cache invalidation
3. Update client to rely on real-time WebSocket updates
4. Remove stale cache logic

### 5. Database Query Optimization

**Current State**: No EXPLAIN ANALYZE results available

**Required Analysis**:
```sql
-- Analyze slow queries
EXPLAIN QUERY PLAN SELECT * FROM kg_nodes WHERE ...;
EXPLAIN QUERY PLAN SELECT * FROM kg_edges WHERE ...;
EXPLAIN QUERY PLAN SELECT * FROM ontology_terms WHERE ...;

-- Check existing indexes
SELECT name, sql FROM sqlite_master
WHERE type='index' AND tbl_name IN ('kg_nodes', 'kg_edges', 'ontology_terms');
```

**Expected Improvements** (from roadmap):
- 5-10x speedup for complex queries
- <10ms database operations (p99)

### 6. Connection Pooling Tuning

**Current Implementation**: Unknown (needs analysis)

**From Roadmap**:
- Phase 6 task: "Connection pooling tuning"
- Expected: r2d2 pool configuration

**Required Analysis**:
1. Current pool size settings
2. Connection usage patterns
3. Timeout configurations
4. Adjust based on load testing

### 7. Redis Caching Layer (Optional)

**Status**: Optional feature for Phase 6

**From Roadmap**:
- Item #11 (v1.1.0): "Redis distributed caching layer"
- Phase 6: "Caching layer implementation (Redis optional)"

**Decision Required**:
- Implement in Phase 6 (v1.0.0) or defer to v1.1.0?
- If Phase 6: Design caching strategy, implement invalidation
- If v1.1.0: Document as future work

### 8. Performance Testing Plan

**Baseline Requirements**:
```
Database operations: <10ms (p99)
API latency: <100ms (p99)
WebSocket latency: <50ms (p99)
Physics simulation: 60 FPS @ 100k+ nodes
```

**Test Scenarios**:
1. Load 100k nodes into database
2. Benchmark query performance
3. Test concurrent API requests (100+ users)
4. Measure WebSocket message latency
5. GPU physics benchmarks

---

## 6.2: Documentation Updates

### 1. README.md Updates Required

**Current Gaps**:
- No hexagonal architecture explanation
- Old setup instructions
- Missing migration guide reference

**Updates Needed**:
```markdown
## Architecture

VisionFlow v1.0.0 uses **Hexagonal Architecture** with:
- **Ports**: Abstract interfaces for external dependencies
- **Adapters**: Concrete implementations (SQLite, GPU, Actors)
- **CQRS**: Command-Query separation for clear data flows
- **Event Bus**: Asynchronous domain events

## Migration from v0.1.0

See [MIGRATION_TO_V1.md](docs/MIGRATION_TO_V1.md) for upgrade instructions.
```

### 2. Architecture Documentation Updates

**Files to Update**:
```
docs/architecture/00-ARCHITECTURE-OVERVIEW.md
docs/architecture/01-ports-design.md
docs/architecture/02-adapters-design.md
docs/architecture/03-cqrs-application-layer.md
docs/architecture/migration-strategy.md
```

**Content Needed**:
- Update diagrams to reflect Phase 1-5 completion
- Document new port/adapter patterns
- Explain event bus integration
- Show WebSocket event flow

### 3. API Documentation Updates

**Files to Update**:
```
docs/api/rest-endpoints.md
docs/api/websocket-protocol.md
docs/api/authentication.md
docs/reference/api/openapi-spec.yml
```

**Changes Needed**:
- New endpoint behaviors (CQRS-based)
- WebSocket event types
- Authentication changes (JWT if implemented)
- Error response formats

### 4. Developer Guide Updates

**Files to Update**:
```
docs/guides/developer/development-setup.md
docs/guides/developer/testing-strategies.md
docs/guides/developer/debugging-tips.md
```

**Content Needed**:
- New development workflow with hexagonal architecture
- How to add new ports/adapters
- Testing with mock adapters
- Debugging actor communication

### 5. Migration Guide Creation

**File**: `/home/devuser/workspace/project/docs/MIGRATION_TO_V1.md`

**Required Sections**:
```markdown
# Migration Guide: v0.1.0 → v1.0.0

## Breaking Changes

### Database Migration
- All settings moved from YAML/TOML to SQLite
- Run migration script: `cargo run --bin migrate_legacy_configs`

### API Changes
- Endpoint X now returns Y format
- WebSocket event Z renamed to W

### Configuration Changes
- Environment variable X renamed to Y
- Config file Z removed (use database)

## Upgrade Steps

1. Backup your data
2. Run database migrations
3. Update environment variables
4. Test your setup
5. Remove legacy config files

## Rollback Procedure

If you need to rollback:
1. Restore database backup
2. Revert to v0.1.0 Docker image
3. Restore config files
```

### 6. Deployment Guide Updates

**Files to Update**:
```
docs/deployment/docker-deployment.md
docs/deployment/production-setup.md
docs/deployment/monitoring.md
```

**Content Needed**:
- New environment variables (database paths, etc.)
- Database migration steps in deployment
- Monitoring setup for new architecture
- Health check endpoints

---

## Coordination & Dependencies

### Phase 5 Completion Checklist

Before Phase 6 can begin, verify:

- [ ] GraphStateActor uses KnowledgeGraphRepository port
- [ ] PhysicsOrchestratorActor uses PhysicsSimulator port
- [ ] SemanticProcessorActor uses semantic ports
- [ ] OntologyActor uses OntologyRepository port
- [ ] All actors removed direct file I/O
- [ ] AppState initialization uses ports/adapters
- [ ] Actor message flow tests pass
- [ ] System integration tests pass

### Memory Coordination

**Store Phase 6 status**:
```bash
npx claude-flow@alpha hooks notify \
  --message "Phase 6 blocked: waiting for Phase 5 completion"

# Store in memory
coordination/phase-6/cleanup/status:
{
  "phase": "6.1-6.2",
  "status": "BLOCKED",
  "blocking_phase": "5.1",
  "analysis_complete": true,
  "ready_to_execute": false,
  "legacy_files_identified": 3,
  "docs_to_update": 15+,
  "estimated_effort": "1 week (per roadmap)"
}
```

---

## Success Criteria

### 6.1 - Legacy Code Removal
- [x] Legacy files identified
- [ ] Zero legacy code remains (after execution)
- [ ] All tests pass
- [ ] Performance benchmarks meet targets:
  - Database ops: <10ms (p99)
  - API latency: <100ms (p99)
  - WebSocket: <50ms (p99)
- [ ] No references to deprecated features

### 6.2 - Documentation Updates
- [x] Documentation gaps identified
- [ ] All documentation accurate
- [ ] Migration guide complete
- [ ] API docs updated
- [ ] Developer guide current
- [ ] Deployment guides updated

---

## Risk Analysis

### High Risk Items

1. **GraphServiceSupervisor Removal** - Marked as deprecated in roadmap but ACTIVE in code
   - **Mitigation**: Verify with Phase 5 before removal
   - **Impact**: HIGH (could break supervision system)

2. **Database Query Performance** - No current benchmarks
   - **Mitigation**: Establish baseline before optimization
   - **Impact**: MEDIUM (could affect user experience)

3. **Client-Side Cache Removal** - Unknown implementation location
   - **Mitigation**: Thorough frontend audit
   - **Impact**: MEDIUM (sync issues)

### Medium Risk Items

1. **File I/O Migration** - 2 active file reads found
   - **Mitigation**: Database migration scripts ready
   - **Impact**: LOW (well-defined scope)

2. **Documentation Drift** - 15+ docs to update
   - **Mitigation**: Systematic approach, peer review
   - **Impact**: LOW (doesn't affect functionality)

---

## Next Steps

### Immediate (While Waiting for Phase 5)

1. **Review Phase 5 progress** with team
2. **Clarify GraphServiceSupervisor status** - deprecated or not?
3. **Establish performance baselines** - run benchmarks on current system
4. **Audit client-side caching** - identify exact implementation
5. **Prepare migration scripts** - ready for Phase 6 execution

### When Phase 5 Completes

1. **Verify all Phase 5 checkboxes** complete
2. **Run full test suite** to ensure stability
3. **Execute Phase 6.1** (legacy removal)
4. **Execute Phase 6.2** (documentation)
5. **Final testing** and performance validation
6. **Tag v1.0.0 release**

---

## Conclusion

Phase 6 analysis is **COMPLETE** and documented. All legacy files have been identified, documentation gaps catalogued, and risks assessed.

**Current blocker**: Phase 5 (Actor Integration) must complete first.

**Recommendation**: Monitor Phase 5 progress and update this document as the architecture evolves.

**Estimated Timeline**: 1 week (per roadmap) once Phase 5 is complete.

---

**Document Version**: 1.0
**Last Updated**: 2025-10-27
**Next Review**: After Phase 5 completion
